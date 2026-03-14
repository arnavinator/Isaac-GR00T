"""Denoising Lab — interactive experimentation with GR00T N1.6 action denoising.

This module provides tools to:
- Load the model and encode backbone features (expensive, run once)
- Run the denoising loop with full user control (cheap, run many times)
- Decode raw actions to physical units
- Visualize and compare EEF trajectories from different denoising strategies

Runs in the main .venv (GPU). No robocasa/gymnasium imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BackboneFeatures:
    """Cached output from the Eagle VLM backbone + state encoder."""

    backbone_features: torch.Tensor  # (B, seq_len, 2048)
    state_features: torch.Tensor  # (B, state_horizon, hidden_dim)
    embodiment_id: torch.Tensor  # (B,)
    backbone_output: BatchFeature  # full backbone output (image_mask, attention_mask)
    states: dict[str, np.ndarray]  # raw state arrays for decode_action()


@dataclass
class DenoiseStepInfo:
    """Recorded information for a single Euler denoising step."""

    step: int
    t_cont: float
    t_discretized: int
    actions_before: np.ndarray  # (B, action_horizon, action_dim)
    velocity: np.ndarray  # (B, action_horizon, action_dim)
    actions_after: np.ndarray  # (B, action_horizon, action_dim)
    action_norm: float
    velocity_norm: float


@dataclass
class DenoiseResult:
    """Complete result from a denoising run."""

    action_pred: torch.Tensor  # (B, action_horizon, action_dim) final raw actions
    intermediates: list[DenoiseStepInfo] = field(default_factory=list)
    initial_noise: torch.Tensor | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Helper: recursive dtype conversion (mirrors gr00t_policy.py)
# ---------------------------------------------------------------------------


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class DenoisingLab:
    """Interactive experimentation with GR00T N1.6 denoising.

    Usage::

        lab = DenoisingLab("nvidia/GR00T-N1.6-3B", "ROBOCASA_PANDA_OMRON")
        obs = DenoisingLab.load_observation("/tmp/saved_observations/step_003.npz")
        features = lab.encode_features_from_sim_obs(obs)
        result = lab.denoise(features, seed=42)
        decoded = lab.decode_raw_actions(result.action_pred)
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str | EmbodimentTag,
        device: str | int = "cuda",
    ):
        import gr00t.model  # noqa: F401 — registers model classes

        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag(embodiment_tag)
            if embodiment_tag is None:
                embodiment_tag = EmbodimentTag[embodiment_tag]
        self.embodiment_tag = embodiment_tag

        # Load model
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model
        self.device = model.device
        self.dtype = model.dtype

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.eval()

        # Convenient references
        self.action_head = model.action_head
        self.backbone = model.backbone
        self.collate_fn = self.processor.collator

        # Modality configs for this embodiment
        self.modality_configs = self.processor.get_modality_configs()[
            self.embodiment_tag.value
        ]

        # Key config values
        self.num_inference_timesteps = self.action_head.config.num_inference_timesteps
        self.num_timestep_buckets = self.action_head.num_timestep_buckets
        self.action_horizon = self.action_head.action_horizon
        self.action_dim = self.action_head.action_dim

        # Language key
        language_keys = self.modality_configs["language"].modality_keys
        assert len(language_keys) == 1
        self.language_key = language_keys[0]

    # ------------------------------------------------------------------
    # Backbone encoding (expensive — run once per observation)
    # ------------------------------------------------------------------

    def encode_features(self, observation: dict[str, Any]) -> BackboneFeatures:
        """Encode observation through Eagle VLM backbone + state encoder.

        Args:
            observation: Nested dict with keys ``video``, ``state``, ``language``
                matching the format expected by ``Gr00tPolicy._get_action()``.

        Returns:
            BackboneFeatures with cached tensors for repeated ``denoise()`` calls.
        """
        # Unbatch → VLAStepData → process → collate  (mirrors Gr00tPolicy._get_action)
        batch_size = observation["video"][
            list(observation["video"].keys())[0]
        ].shape[0]

        processed_inputs = []
        states_list = []
        for i in range(batch_size):
            single_obs = {
                "video": {k: v[i] for k, v in observation["video"].items()},
                "state": {k: v[i] for k, v in observation["state"].items()},
                "language": {k: v[i] for k, v in observation["language"].items()},
            }
            vla = VLAStepData(
                images=single_obs["video"],
                states=single_obs["state"],
                actions={},
                text=single_obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            states_list.append(vla.states)
            messages = [
                {"type": MessageType.EPISODE_STEP.value, "content": vla}
            ]
            processed_inputs.append(self.processor(messages))

        collated = self.collate_fn(processed_inputs)
        collated = _rec_to_dtype(collated, dtype=torch.bfloat16)

        # Prepare inputs and run backbone
        # collate_fn returns BatchFeature({"inputs": batch_dict})
        # model.get_action(**collated) unpacks to get_action(inputs=batch_dict)
        with torch.inference_mode():
            backbone_inputs, action_inputs = self.model.prepare_input(
                **collated
            )
            backbone_output = self.backbone(backbone_inputs)

            # Encode features (state encoder + vlln)
            features = self.action_head._encode_features(
                backbone_output, action_inputs
            )

        # Batch states for decode_action later
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack(
                [s[k] for s in states_list], axis=0
            )

        return BackboneFeatures(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_inputs.embodiment_id,
            backbone_output=backbone_output,
            states=batched_states,
        )

    def encode_features_from_sim_obs(
        self, sim_obs: dict[str, Any]
    ) -> BackboneFeatures:
        """Encode from flat sim observation format (``video.cam``, ``state.joints``, …).

        This replicates the key transformation from
        ``Gr00tSimPolicyWrapper._get_action()`` then calls ``encode_features()``.
        """
        new_obs: dict[str, Any] = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.modality_configs[modality].modality_keys:
                if modality == "language":
                    parsed_key = key
                    arr = sim_obs[parsed_key]
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    parsed_key = f"{modality}.{key}"
                    new_obs[modality][key] = sim_obs[parsed_key]
        return self.encode_features(new_obs)

    # ------------------------------------------------------------------
    # Denoising (cheap — only the DiT runs)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def denoise(
        self,
        features: BackboneFeatures,
        *,
        num_steps: int = 4,
        dt: float | None = None,
        seed: int | None = None,
        initial_noise: torch.Tensor | None = None,
        guided_fn: Callable | None = None,
        step_callback: Callable | None = None,
    ) -> DenoiseResult:
        """Run the flow-matching denoising loop with full user control.

        Args:
            features: Cached backbone features from ``encode_features()``.
            num_steps: Number of Euler integration steps (default 4).
            dt: Euler step size. Defaults to ``1.0 / num_steps``.
            seed: Random seed for initial noise (for reproducibility).
            initial_noise: Provide custom starting noise tensor
                ``(B, action_horizon, action_dim)``.
            guided_fn: ``(actions, step_idx, velocity) -> modified_velocity``.
                Called before each Euler update to inject guidance.
            step_callback: ``(step_idx, actions, velocity) -> None``.
                Called after each step for inspection.

        Returns:
            DenoiseResult with final actions, all intermediates, and the initial noise.
        """
        if dt is None:
            dt = 1.0 / num_steps

        vl_embeds = features.backbone_features
        state_features = features.state_features
        embodiment_id = features.embodiment_id
        backbone_output = features.backbone_output
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        # Initial noise
        if initial_noise is not None:
            actions = initial_noise.to(device=device, dtype=vl_embeds.dtype)
        else:
            if seed is not None:
                gen = torch.Generator(device=device).manual_seed(seed)
            else:
                gen = None
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
                generator=gen,
            )

        saved_noise = actions.clone()
        intermediates: list[DenoiseStepInfo] = []

        for t in range(num_steps):
            actions_before = actions.clone()

            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            velocity, actions = self._denoise_step_inner(
                vl_embeds,
                state_features,
                embodiment_id,
                backbone_output,
                actions,
                t_discretized,
                dt,
                batch_size,
                device,
            )

            if guided_fn is not None:
                # Undo the default Euler update, apply guided velocity, redo
                actions = actions_before + dt * guided_fn(
                    actions_before, t, velocity
                )

            action_norm = actions.float().norm().item()
            velocity_norm = velocity.float().norm().item()

            intermediates.append(
                DenoiseStepInfo(
                    step=t,
                    t_cont=t_cont,
                    t_discretized=t_discretized,
                    actions_before=actions_before.float().cpu().numpy(),
                    velocity=velocity.float().cpu().numpy(),
                    actions_after=actions.float().cpu().numpy(),
                    action_norm=action_norm,
                    velocity_norm=velocity_norm,
                )
            )

            if step_callback is not None:
                step_callback(t, actions, velocity)

        return DenoiseResult(
            action_pred=actions,
            intermediates=intermediates,
            initial_noise=saved_noise,
            seed=seed,
        )

    @torch.no_grad()
    def denoise_single_step(
        self,
        features: BackboneFeatures,
        actions: torch.Tensor,
        step_idx: int,
        num_total_steps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single denoising step for manual cell-by-cell control.

        Args:
            features: Cached backbone features.
            actions: Current actions tensor ``(B, action_horizon, action_dim)``.
            step_idx: Current step index (0-based).
            num_total_steps: Total number of steps (for computing t_cont and dt).

        Returns:
            ``(velocity, updated_actions)`` tuple.
        """
        dt = 1.0 / num_total_steps
        t_cont = step_idx / float(num_total_steps)
        t_discretized = int(t_cont * self.num_timestep_buckets)

        velocity, updated = self._denoise_step_inner(
            features.backbone_features,
            features.state_features,
            features.embodiment_id,
            features.backbone_output,
            actions.to(device=self.device, dtype=self.dtype),
            t_discretized,
            dt,
            actions.shape[0],
            self.device,
        )
        return velocity, updated

    def _denoise_step_inner(
        self,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        actions: torch.Tensor,
        t_discretized: int,
        dt: float,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner body of one denoising step (shared by denoise / denoise_single_step)."""
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = self.action_head.action_encoder(
            actions, timesteps_tensor, embodiment_id
        )

        if self.action_head.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.action_head.config.use_alternate_vl_dit:
            model_output = self.action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = self.action_head.action_decoder(model_output, embodiment_id)
        velocity = pred[:, -self.action_horizon :]
        updated_actions = actions + dt * velocity
        return velocity, updated_actions

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_raw_actions(
        self,
        action_pred: torch.Tensor,
        states: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode raw ``(B, action_horizon, action_dim)`` to per-key physical units.

        Args:
            action_pred: Raw model output.
            states: Optional state dict for relative→absolute conversion.
                Defaults to the states cached during ``encode_features()``.

        Returns:
            Dict mapping action key names to arrays, e.g.
            ``{"end_effector_position": (B, 16, 3), ...}``.
        """
        raw = action_pred.float().cpu().numpy()
        return self.processor.decode_action(
            raw, self.embodiment_tag, states
        )

    def label_action_step(
        self, decoded_actions: dict[str, np.ndarray], step_idx: int
    ) -> str:
        """Return a human-readable label for a single sub-step."""
        parts = []
        for key, arr in decoded_actions.items():
            vals = arr[0, step_idx]  # batch 0
            fmt = " ".join(f"{v:+.4f}" for v in vals)
            parts.append(f"  {key}: [{fmt}]")
        return f"Step {step_idx}:\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Observation I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_observation(observation: dict[str, Any], path: str | Path) -> None:
        """Save an observation dict to a compressed ``.npz`` file.

        Handles video (uint8), state (float32), and language (string metadata).
        Compatible with both flat (sim) and nested formats.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, str] = {}

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif isinstance(value, (str, list, tuple)):
                # Language keys — store as string metadata
                if isinstance(value, (list, tuple)):
                    metadata[key] = str(value[0]) if len(value) == 1 else "|".join(str(v) for v in value)
                else:
                    metadata[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_key = f"{key}.{sub_key}"
                    if isinstance(sub_val, np.ndarray):
                        arrays[flat_key] = sub_val
                    elif isinstance(sub_val, (list, tuple)):
                        metadata[flat_key] = "|".join(str(v) for v in sub_val) if len(sub_val) > 1 else str(sub_val[0])
                    else:
                        metadata[flat_key] = str(sub_val)

        # Store language metadata as a special array so npz can hold it
        if metadata:
            import json

            arrays["__metadata__"] = np.array(
                json.dumps(metadata), dtype=object
            )

        np.savez_compressed(str(path), **arrays)

    @staticmethod
    def load_observation(path: str | Path) -> dict[str, Any]:
        """Load an observation dict from a ``.npz`` file saved by ``save_observation``.

        Returns a flat dict with keys like ``video.cam``, ``state.joints``, etc.
        """
        import json

        data = dict(np.load(str(path), allow_pickle=True))
        metadata: dict[str, str] = {}

        if "__metadata__" in data:
            metadata = json.loads(str(data.pop("__metadata__")))

        result: dict[str, Any] = {}
        for key, arr in data.items():
            result[key] = arr

        # Restore language/string metadata
        for key, val in metadata.items():
            if "|" in val:
                result[key] = tuple(val.split("|"))
            else:
                result[key] = (val,)

        return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


class TrajectoryVisualizer:
    """Accumulate and compare EEF trajectories from different denoising strategies."""

    def __init__(self) -> None:
        self.trajectories: list[dict[str, Any]] = []

    def add_trajectory(
        self,
        decoded_actions: dict[str, np.ndarray],
        label: str,
        color: str | None = None,
        start_pos: np.ndarray | None = None,
        eef_key: str = "end_effector_position",
    ) -> None:
        """Add a trajectory from decoded actions.

        Args:
            decoded_actions: Output of ``DenoisingLab.decode_raw_actions()``.
            label: Legend label for this trajectory.
            color: Optional matplotlib color.
            start_pos: Starting EEF position ``(3,)``. Defaults to origin.
            eef_key: Key for EEF position in decoded_actions.
        """
        if eef_key not in decoded_actions:
            available = list(decoded_actions.keys())
            raise KeyError(
                f"Key '{eef_key}' not in decoded_actions. Available: {available}"
            )

        deltas = decoded_actions[eef_key][0]  # (T, 3) — batch 0
        if start_pos is None:
            start_pos = np.zeros(3)

        positions = np.zeros((deltas.shape[0] + 1, 3))
        positions[0] = start_pos
        for t in range(deltas.shape[0]):
            positions[t + 1] = positions[t] + deltas[t]

        self.trajectories.append(
            {
                "positions": positions,
                "label": label,
                "color": color,
                "deltas": deltas,
            }
        )

    def add_from_denoise_result(
        self,
        result: DenoiseResult,
        lab: DenoisingLab,
        label: str,
        color: str | None = None,
        start_pos: np.ndarray | None = None,
        states: dict[str, np.ndarray] | None = None,
        eef_key: str = "end_effector_position",
    ) -> dict[str, np.ndarray]:
        """Decode a DenoiseResult and add the trajectory. Returns decoded actions."""
        decoded = lab.decode_raw_actions(result.action_pred, states)
        self.add_trajectory(decoded, label, color, start_pos, eef_key)
        return decoded

    def plot_eef_3d(
        self,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """3D plot of all accumulated EEF trajectories."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for traj in self.trajectories:
            pos = traj["positions"]
            kwargs: dict[str, Any] = {"label": traj["label"]}
            if traj["color"]:
                kwargs["color"] = traj["color"]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", markersize=3, **kwargs)
            ax.scatter(*pos[0], marker="^", s=60, zorder=5)
            ax.scatter(*pos[-1], marker="s", s=60, zorder=5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        if title:
            ax.set_title(title)

        # Equal axis scaling
        all_pos = np.concatenate([t["positions"] for t in self.trajectories])
        mid = all_pos.mean(axis=0)
        max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2
        if max_range < 1e-8:
            max_range = 0.01
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.tight_layout()
        return fig

    def plot_eef_components(
        self,
        title: str | None = None,
        figsize: tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """Plot X, Y, Z components over timesteps for all trajectories."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        labels = ["X", "Y", "Z"]

        for ax, dim, lbl in zip(axes, range(3), labels):
            for traj in self.trajectories:
                pos = traj["positions"]
                kwargs: dict[str, Any] = {"label": traj["label"]}
                if traj["color"]:
                    kwargs["color"] = traj["color"]
                ax.plot(pos[:, dim], "-o", markersize=3, **kwargs)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(lbl)
            ax.set_title(f"EEF {lbl}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig

    def plot_denoising_progression(
        self,
        result: DenoiseResult,
        lab: DenoisingLab,
        figsize: tuple[int, int] = (16, 4),
        eef_key: str = "end_effector_position",
    ) -> plt.Figure:
        """Show how the EEF trajectory evolves at each denoising step."""
        n_steps = len(result.intermediates)
        fig, axes = plt.subplots(1, n_steps + 1, figsize=figsize, subplot_kw={"projection": "3d"})

        titles = [f"After step {i}" for i in range(n_steps)] + ["Final"]
        tensors = [
            torch.tensor(info.actions_after, dtype=torch.float32)
            for info in result.intermediates
        ] + [result.action_pred.float().cpu()]

        all_positions = []
        for ax, tensor, ttl in zip(axes, tensors, titles):
            decoded = lab.decode_raw_actions(tensor)
            if eef_key in decoded:
                deltas = decoded[eef_key][0]
                pos = np.zeros((deltas.shape[0] + 1, 3))
                for t in range(deltas.shape[0]):
                    pos[t + 1] = pos[t] + deltas[t]
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-o", markersize=2)
                all_positions.append(pos)
            ax.set_title(ttl, fontsize=9)
            ax.tick_params(labelsize=6)

        # Uniform axis limits
        if all_positions:
            all_pos = np.concatenate(all_positions)
            mid = all_pos.mean(axis=0)
            max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2
            if max_range < 1e-8:
                max_range = 0.01
            for ax in axes:
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        fig.suptitle("Denoising Progression")
        plt.tight_layout()
        return fig


def compare_strategies(
    lab: DenoisingLab,
    features: BackboneFeatures,
    strategies: list[dict[str, Any]],
    labels: list[str] | None = None,
    eef_key: str = "end_effector_position",
) -> tuple[list[DenoiseResult], TrajectoryVisualizer]:
    """Run multiple denoising strategies and return results + populated visualizer.

    Args:
        lab: DenoisingLab instance.
        features: Cached backbone features.
        strategies: List of kwarg dicts for ``lab.denoise()``.
        labels: Optional labels; defaults to ``"Strategy 0"``, etc.
        eef_key: EEF position key in decoded actions.

    Returns:
        ``(results, visualizer)`` tuple.
    """
    if labels is None:
        labels = [f"Strategy {i}" for i in range(len(strategies))]

    results = []
    viz = TrajectoryVisualizer()

    for kwargs, label in zip(strategies, labels):
        result = lab.denoise(features, **kwargs)
        results.append(result)
        try:
            viz.add_from_denoise_result(result, lab, label, eef_key=eef_key)
        except KeyError:
            pass  # EEF key not available for this embodiment

    return results, viz
