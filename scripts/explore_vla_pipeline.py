"""Single-process VLA pipeline exploration script.

Loads the GR00T model and creates the env in one process (no ZMQ server-client split).
Runs 1 episode with maximum logging of:
  - Language instructions sent to the model
  - Full 16-step action chunks (shape + per-dimension values)
  - Which sub-steps are executed vs. discarded
  - Per sub-step unmapped action dicts with labeled dimensions
  - 4-step denoising intermediates (noise -> final action)

Requires GPU. Intended for RoboCasa Panda environments.
"""

import argparse
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


# ---------------------------------------------------------------------------
# Monkey-patch to capture denoising intermediates
# ---------------------------------------------------------------------------

_denoising_log = []  # populated by the patched method


def _patched_get_action_with_features(self, backbone_features, state_features,
                                       embodiment_id, backbone_output):
    """Patched version that records intermediate denoising states."""
    from transformers.feature_extraction_utils import BatchFeature

    _denoising_log.clear()

    vl_embeds = backbone_features
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    actions = torch.randn(
        size=(batch_size, self.config.action_horizon, self.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )

    dt = 1.0 / self.num_inference_timesteps

    for t in range(self.num_inference_timesteps):
        t_cont = t / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)

        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -self.action_horizon:]

        action_norm = actions.float().norm().item()
        velocity_norm = pred_velocity.float().norm().item()
        _denoising_log.append({
            "step": t,
            "t_cont": t_cont,
            "t_disc": t_discretized,
            "action_norm_before": action_norm,
            "velocity_norm": velocity_norm,
            "action_snapshot": actions.float().cpu().numpy().copy(),
        })

        actions = actions + dt * pred_velocity

    # Record final
    _denoising_log.append({
        "step": "final",
        "action_norm": actions.float().norm().item(),
        "action_mean": actions.float().mean().item(),
        "action_std": actions.float().std().item(),
        "action_snapshot": actions.float().cpu().numpy().copy(),
    })

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


def print_denoising_log():
    """Print the captured denoising intermediates."""
    print("\n  === Denoising Trajectory ===")
    for entry in _denoising_log:
        if entry["step"] == "final":
            print(
                f"  [Final]  action_norm={entry['action_norm']:.4f} | "
                f"mean={entry['action_mean']:.4f} | std={entry['action_std']:.4f}"
            )
        else:
            print(
                f"  [Step {entry['step']}/{len(_denoising_log)-1}] "
                f"t_cont={entry['t_cont']:.3f} | t_disc={entry['t_disc']} | "
                f"action_norm={entry['action_norm_before']:.4f} | "
                f"velocity_norm={entry['velocity_norm']:.4f}"
            )


def save_denoising_plot(save_path="denoising_trajectory.png"):
    """Save a matplotlib plot of action norms across denoising steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = []
    action_norms = []
    velocity_norms = []

    for entry in _denoising_log:
        if entry["step"] == "final":
            continue
        steps.append(entry["step"])
        action_norms.append(entry["action_norm_before"])
        velocity_norms.append(entry["velocity_norm"])

    # Add the final action norm
    final = _denoising_log[-1]
    steps.append(len(steps))
    action_norms.append(final["action_norm"])
    velocity_norms.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(steps, action_norms, "o-", color="tab:blue")
    ax1.set_xlabel("Denoising Step")
    ax1.set_ylabel("Action Norm")
    ax1.set_title("Action Norm During Denoising")
    ax1.grid(True, alpha=0.3)

    ax2.bar(steps[:-1], velocity_norms[:-1], color="tab:orange", alpha=0.7)
    ax2.set_xlabel("Denoising Step")
    ax2.set_ylabel("Predicted Velocity Norm")
    ax2.set_title("Velocity Norm Per Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Denoising plot saved to: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Action labeling for Panda Omron
# ---------------------------------------------------------------------------

def label_unmapped_action(action_dict):
    """Print labeled action dimensions for a single sub-step action dict.

    action_dict keys come from MultiStepWrapper after unmap_action:
      action.end_effector_position  (3,)
      action.end_effector_rotation  (3,)
      action.gripper_close          scalar or (1,)
      action.base_motion            (4,)
      action.control_mode           scalar or (1,)
    """
    lines = []

    # EEF position (delta XYZ)
    if "action.end_effector_position" in action_dict:
        pos = action_dict["action.end_effector_position"]
        lines.append(f"    EEF delta pos : [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    # EEF rotation (delta axis-angle)
    if "action.end_effector_rotation" in action_dict:
        rot = action_dict["action.end_effector_rotation"]
        lines.append(f"    EEF delta rot : [{rot[0]:+.4f}, {rot[1]:+.4f}, {rot[2]:+.4f}]")

    # Gripper
    if "action.gripper_close" in action_dict:
        g = action_dict["action.gripper_close"]
        g_val = float(np.squeeze(g))
        label = "CLOSE" if g_val >= 0.5 else "OPEN"
        lines.append(f"    Gripper       : {g_val:.2f} ({label})")

    # Base motion (3D velocity + 1D torso)
    if "action.base_motion" in action_dict:
        bm = action_dict["action.base_motion"]
        lines.append(
            f"    Base velocity : [{bm[0]:+.4f}, {bm[1]:+.4f}, {bm[2]:+.4f}]"
        )
        if len(bm) > 3:
            lines.append(f"    Torso height  : {bm[3]:+.4f}")

    # Control mode
    if "action.control_mode" in action_dict:
        cm = float(np.squeeze(action_dict["action.control_mode"]))
        label = "NAVIGATION ON" if cm >= 0.5 else "NAVIGATION OFF"
        lines.append(f"    Control mode  : {cm:.2f} ({label})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-process VLA pipeline explorer with verbose logging"
    )
    parser.add_argument(
        "--model-path", type=str, default="nvidia/GR00T-N1.6-3B",
        help="Path to model checkpoint (HF hub or local)",
    )
    parser.add_argument(
        "--env-name", type=str,
        default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
        help="Gymnasium env name (e.g. robocasa_panda_omron/OpenDrawer_PandaOmron_Env)",
    )
    parser.add_argument(
        "--language-override", type=str, default=None,
        help="Override the language instruction (e.g. 'close the left drawer')",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=720,
        help="Maximum sub-steps per episode",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help="How many steps from the 16-step chunk to execute",
    )
    parser.add_argument(
        "--save-denoising-plot", action="store_true",
        help="Save a matplotlib figure of denoising trajectory",
    )
    args = parser.parse_args()

    # --- Resolve embodiment ---
    embodiment_tag = get_embodiment_tag_from_env_name(args.env_name)
    print(f"Env: {args.env_name}")
    print(f"Embodiment tag: {embodiment_tag}")
    print(f"Model: {args.model_path}")
    print(f"Action chunking: predict 16, execute {args.n_action_steps}")

    # --- Load model ---
    print("\nLoading model...")
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    sim_policy = Gr00tSimPolicyWrapper(policy)

    # --- Monkey-patch the action head to capture denoising intermediates ---
    import types
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
    policy.model.action_head.get_action_with_features = types.MethodType(
        _patched_get_action_with_features, policy.model.action_head
    )
    print("Monkey-patched action head for denoising logging")

    # --- Create environment ---
    print("\nCreating environment...")
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )
    env = create_eval_env(
        env_name=args.env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )

    # --- Run 1 episode ---
    print("\n" + "=" * 70)
    print("Starting episode")
    print("=" * 70)

    obs, info = env.reset()
    sim_policy.reset()

    # Add batch dimension (policy expects batched obs)
    def add_batch_dim(obs_dict):
        batched = {}
        for k, v in obs_dict.items():
            if isinstance(v, np.ndarray):
                batched[k] = np.expand_dims(v, axis=0)
            elif isinstance(v, str):
                batched[k] = (v,)
            else:
                batched[k] = v
        return batched

    done = False
    policy_step = 0
    total_sub_steps = 0

    while not done:
        batched_obs = add_batch_dim(obs)

        # --- Language override ---
        if args.language_override:
            for k in list(batched_obs.keys()):
                if k.startswith("annotation"):
                    batched_obs[k] = (args.language_override,)

        # Print language instruction
        lang_key = None
        for k in batched_obs:
            if k.startswith("annotation"):
                lang_key = k
                break
        lang_text = batched_obs[lang_key][0] if lang_key else "N/A"

        print(f"\n{'─' * 60}")
        print(f"Policy Step {policy_step}")
        print(f"  Language: \"{lang_text}\"")

        # --- Get action ---
        action, action_info = sim_policy.get_action(batched_obs)

        # --- Print denoising log ---
        print_denoising_log()

        if args.save_denoising_plot and policy_step == 0:
            save_denoising_plot("denoising_trajectory.png")

        # --- Print full action chunk ---
        print("\n  === Action Chunk (model output) ===")
        for ak, av in action.items():
            # av shape: (B, action_horizon, D)
            chunk = av[0]  # remove batch dim -> (action_horizon, D)
            print(f"  {ak}: shape={chunk.shape}")
            print(f"    Execute steps 0-{args.n_action_steps - 1}, "
                  f"discard steps {args.n_action_steps}-{chunk.shape[0] - 1}")

        # --- Print per-sub-step labeled actions ---
        print(f"\n  === Sub-step Actions (executed) ===")
        # Extract the first n_action_steps from the chunk
        for step_idx in range(args.n_action_steps):
            sub_action = {}
            for ak, av in action.items():
                sub_action[ak] = av[0, step_idx]  # (D,)
            print(f"  Sub-step {step_idx}/{args.n_action_steps}:")
            print(label_unmapped_action(sub_action))

        # --- Step environment ---
        # Remove batch dim for the env
        env_action = {k: v[0] for k, v in action.items()}
        obs, reward, done, truncated, step_info = env.step(env_action)
        done = done or truncated

        total_sub_steps += args.n_action_steps
        success = False
        if "success" in step_info:
            success_val = step_info["success"]
            if isinstance(success_val, (list, np.ndarray)):
                success = bool(np.any(success_val))
            else:
                success = bool(success_val)

        print(f"\n  reward={reward:.2f} | done={done} | success={success} | "
              f"total_sub_steps={total_sub_steps}")

        policy_step += 1

    print(f"\n{'=' * 70}")
    print(f"Episode finished after {policy_step} policy steps "
          f"({total_sub_steps} sub-steps)")
    print(f"Final success: {success}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
