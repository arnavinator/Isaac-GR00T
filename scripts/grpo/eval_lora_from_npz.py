"""Evaluate a LoRA-tuned GR00T policy by replaying from a saved sim state.

Runs in the ROBOCASA VENV. Talks to a running GRPO server (Terminal 1, started
separately with the LoRA loaded via scripts/grpo/grpo_server.py).

Usage:
    # Terminal 1 (main venv, GPU): start GRPO server with LoRA
    uv run python scripts/grpo/grpo_server.py \\
        --model-path nvidia/GR00T-N1.6-3B \\
        --embodiment-tag ROBOCASA_PANDA_OMRON \\
        --lora-checkpoint grpo_data/toy_lr3.0e-5_v3/checkpoints/iter_0006 \\
        --use-sim-policy-wrapper --port 5555

    # Terminal 2 (robocasa venv): run N parallel evaluations from the saved state
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/grpo/eval_lora_from_npz.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --obs-path /tmp/saved_observations/ep000_step010.npz \\
        --num-attempts 100 --num-envs 8 \\
        --output-dir /tmp/eval_iter_0006 \\
        --lora-checkpoint grpo_data/toy_lr3.0e-5_v3/checkpoints/iter_0006

The script:
- Connects to the GRPO server via ZMQ (server-side LoRA load is unchanged from
  the existing two-terminal pattern).
- Loads the saved sim state from --obs-path (interactive_rollout.py format:
  __sim_state__, __model_xml__, __ep_meta__, optional __step_info__).
- Runs --num-attempts rollouts, all starting bit-identically from the saved
  state via apply_scene_bundle. Within-attempt diversity comes from the
  policy's denoising noise (unseeded torch.randn inside the DiT), NOT from
  env randomness.
- Uses AsyncVectorEnv with --num-envs subprocess workers (parallel MuJoCo);
  --num-attempts must be divisible by --num-envs, with the work split into
  num_attempts // num_envs sequential turns.
- No video / image / per-step observation saving — only per-attempt
  {success, num_steps, env_seed, termination} aggregated into results.json.

The --lora-checkpoint argument is metadata only (recorded for lineage); the
server in Terminal 1 is responsible for actually loading the weights. To
suppress import-time robosuite/robocasa noise, set GRPO_CLEAN_OUTPUT=1 in
the environment (parity with collect_episodes.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zmq

# scripts/grpo/ on sys.path so we can import collect_episodes (which in turn
# imports its sibling dense_reward via plain `import dense_reward`). When this
# script is launched directly (`python scripts/grpo/eval_lora_from_npz.py ...`)
# Python already puts the script's dir on sys.path[0]; this insert is defensive
# for the imported-as-module case.
sys.path.insert(0, str(Path(__file__).parent))

from collect_episodes import EpisodeCollector, _load_init_bundle
from gr00t.policy.server_client import PolicyClient


# ---------------------------------------------------------------------------
# EvalCollector
# ---------------------------------------------------------------------------


class EvalCollector(EpisodeCollector):
    """EpisodeCollector subclass that drops per-step recording unneeded for eval.

    Three overrides (all small, none touch the vector-env / scene-bundle path):
      - ``_extract_video_single`` → ``{}`` skips per-chunk frame dicts. This is
        the largest memory cost otherwise: 100 attempts × ~60 chunks × 3 cams ×
        256² × 3 ≈ 12 GB of pixel arrays held resident through the whole run.
      - ``_extract_state_single`` → ``{}`` skips per-chunk state dicts (small
        but unused for binary-success eval).
      - ``_get_actions_from_server`` returns ``(action_dict, None, None, None)``
        instead of forwarding the server's noise / raw-action / action-mask
        captures. Those are FM-log-prob inputs needed by GRPO TRAINING; for
        eval they're dead weight (~460 MB per 100-attempt run).

    What's still recorded per chunk: ``ep["actions"]`` (~3 KB/chunk for Panda,
    ~18 MB for 100 attempts × 60 chunks). Cheap to keep in case the user wants
    to inspect them post-hoc by extending this script.
    """

    def _extract_video_single(self, obs: dict) -> dict:
        return {}

    def _extract_state_single(self, obs: dict) -> dict:
        return {}

    def _get_actions_from_server(self, observations) -> tuple:
        result = self.policy_client.get_action(observations)
        action_dict = result[0] if isinstance(result, tuple) else result
        return action_dict, None, None, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a LoRA-tuned policy by running N parallel rollouts from a "
            "saved sim state. The LoRA is loaded server-side; start "
            "grpo_server.py with --lora-checkpoint in Terminal 1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="Gymnasium env ID (e.g. robocasa_panda_omron/OpenDrawer_PandaOmron_Env).",
    )
    parser.add_argument(
        "--obs-path", type=str, required=True,
        help="Path to saved sim-state .npz (from interactive_rollout.py).",
    )
    parser.add_argument(
        "--num-attempts", type=int, default=100,
        help="Total rollouts to run from the saved state.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help=(
            "Parallel AsyncVectorEnv subprocess workers. Must be in "
            "[1, num-attempts] and divide num-attempts evenly. Each worker is "
            "one MuJoCo process (~5 GiB RAM)."
        ),
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=480,
        help=(
            "Per-rollout truncation horizon (sub-steps). The wrapper bills "
            "consumed_substeps from the saved __step_info__ against this "
            "budget, so a step-10 npz has fewer remaining sub-steps than a "
            "step-0 npz."
        ),
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=8,
        help=(
            "Sub-steps to execute per action chunk. Note: the consumed-substeps "
            "accounting uses the saved npz's n_action_steps (universal time "
            "units), not this value, so this can differ from training-time "
            "without affecting budget bookkeeping."
        ),
    )
    parser.add_argument(
        "--server-host", type=str, default="127.0.0.1",
        help="GRPO server host. Start grpo_server.py --lora-checkpoint <path> in Terminal 1.",
    )
    parser.add_argument(
        "--server-port", type=int, default=5555,
        help="GRPO server port.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Base group seed for env init. Within-attempt diversity comes from "
            "the server's unseeded denoising noise, NOT from this seed — "
            "re-running with the same --seed will not reproduce per-attempt "
            "outcomes."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for results.json (created if missing).",
    )
    parser.add_argument(
        "--lora-checkpoint", type=str, default=None,
        help=(
            "Path to the LoRA checkpoint loaded into the GRPO server "
            "(metadata only — recorded in results.json for lineage; not "
            "loaded by this script)."
        ),
    )
    return parser.parse_args()


def _validate_args(args) -> None:
    if args.num_attempts < 1:
        raise ValueError(f"--num-attempts must be >= 1, got {args.num_attempts}")
    if not (1 <= args.num_envs <= args.num_attempts):
        raise ValueError(
            f"--num-envs ({args.num_envs}) must satisfy "
            f"1 <= n <= --num-attempts ({args.num_attempts})"
        )
    if args.num_attempts % args.num_envs != 0:
        divisors = _list_divisors(args.num_attempts)
        raise ValueError(
            f"--num-envs ({args.num_envs}) must divide --num-attempts "
            f"({args.num_attempts}) evenly. Divisors of "
            f"{args.num_attempts}: {divisors}"
        )

    obs_path = Path(args.obs_path)
    if not obs_path.is_file():
        raise FileNotFoundError(f"--obs-path not found: {obs_path}")


def _read_step_info(obs_path: Path) -> tuple[int | None, int | None]:
    """Pull (branch_step, saved_n_action_steps) from __step_info__ for display.

    Returns (None, None) if the npz lacks __step_info__ or it's malformed. The
    collector's own _load_init_bundle path re-parses these to compute
    consumed_substeps; values returned here are used purely for the lineage
    banner / results.json metadata.
    """
    try:
        raw = dict(np.load(str(obs_path), allow_pickle=True))
    except Exception:
        return None, None
    if "__step_info__" not in raw:
        return None, None
    try:
        step_info = json.loads(str(raw["__step_info__"]))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(step_info, dict):
        return None, None
    return step_info.get("step"), step_info.get("n_action_steps")


def _summarize(episodes: list[dict]) -> dict:
    """Reduce per-episode dicts (from EpisodeCollector.collect) to results.

    Note: ``ep["env_seed"]`` is the per-group seed (collect_episodes.py:1593) and
    is identical across every attempt here — we only use one group, so all
    attempts inherit ``args.seed``. It would be misleading per-attempt; the
    value lives in ``lineage["seed"]`` instead.
    """
    attempts = [
        {
            "attempt_idx": i,
            "success": bool(ep["success"]),
            "num_steps": int(ep["num_steps"]),
            "termination": "success" if ep["success"] else "truncated",
        }
        for i, ep in enumerate(episodes)
    ]

    successes = sum(a["success"] for a in attempts)
    successful_steps = [a["num_steps"] for a in attempts if a["success"]]
    failed_steps = [a["num_steps"] for a in attempts if not a["success"]]
    total = len(attempts)

    summary = {
        "total": total,
        "successes": successes,
        "success_rate": (successes / total) if total > 0 else 0.0,
        "mean_num_steps_all": (
            float(np.mean([a["num_steps"] for a in attempts])) if total > 0 else None
        ),
        "mean_num_steps_successful": (
            float(np.mean(successful_steps)) if successful_steps else None
        ),
        "mean_num_steps_failed": (
            float(np.mean(failed_steps)) if failed_steps else None
        ),
    }
    return {"summary": summary, "attempts": attempts}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    _validate_args(args)

    obs_path = Path(args.obs_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load the init bundle for the lineage banner. The collector lazy-loads
    # again on the first .collect() call (cached per-process via _get_init_bundle).
    # _load_init_bundle is the canonical parser for the interactive_rollout.py
    # npz contract; reusing it keeps budget / warning behavior consistent with
    # the actual collection path.
    bundle = _load_init_bundle(str(obs_path))
    consumed_substeps = bundle["consumed_substeps"]
    branch_step, saved_n_action_steps = _read_step_info(obs_path)
    remaining_budget = args.max_episode_steps - consumed_substeps

    print("=" * 60)
    print("LoRA evaluation harness")
    print(f"  Env: {args.env_name}")
    print(f"  Obs path: {obs_path}")
    print(
        f"  Num attempts: {args.num_attempts} "
        f"(over {args.num_envs} parallel envs, "
        f"{args.num_attempts // args.num_envs} turn(s))"
    )
    print(f"  Server: {args.server_host}:{args.server_port}")
    if args.lora_checkpoint:
        print(f"  LoRA checkpoint (server-side): {args.lora_checkpoint}")
    if branch_step is not None:
        print(
            f"  Branch step: {branch_step} "
            f"(saved n_action_steps={saved_n_action_steps}, "
            f"consumed_substeps={consumed_substeps}, "
            f"remaining_budget={remaining_budget})"
        )
    else:
        print(
            "  Branch step: unknown — npz lacks __step_info__; "
            f"consumed_substeps defaulted to 0 "
            f"(full budget = {args.max_episode_steps})"
        )
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Pre-spawn ping. EvalCollector.__init__ spawns AsyncVectorEnv subprocess
    # workers BEFORE any server contact (collect_episodes.py:806), so without
    # a pre-check a server-down user pays the 10-20s robocasa import + worker
    # spawn cost before the first get_action call hangs.
    #
    # We can't use PolicyClient.ping() out of the box: PolicyClient stores
    # timeout_ms (server_client.py:164) but never applies it to the socket,
    # so its recv blocks forever on a downed server. Set RCVTIMEO/SNDTIMEO
    # explicitly on the temporary ping socket to make ping() actually
    # fail-fast (5 s budget, plenty for a healthy server that already booted
    # via grpo_server.py — its first response is a constant-time dict).
    print(f"Pinging server at {args.server_host}:{args.server_port}...")
    ping_client = PolicyClient(
        host=args.server_host, port=args.server_port, strict=False,
    )
    ping_client.socket.setsockopt(zmq.RCVTIMEO, 5000)
    ping_client.socket.setsockopt(zmq.SNDTIMEO, 5000)
    ping_client.socket.setsockopt(zmq.LINGER, 0)
    if not ping_client.ping():
        raise ConnectionError(
            f"GRPO server at {args.server_host}:{args.server_port} not "
            f"responding within 5 s. Start it in Terminal 1, e.g.\n"
            f"  uv run python scripts/grpo/grpo_server.py "
            f"--lora-checkpoint <path> "
            f"--embodiment-tag ROBOCASA_PANDA_OMRON --port {args.server_port}"
        )
    del ping_client  # close the temporary socket so the collector's PolicyClient gets a fresh one
    print("Server reachable.")

    # NB: we deliberately do NOT seed numpy on the parent process. Within-attempt
    # diversity comes from the server's unseeded torch.randn during denoising;
    # AsyncVectorEnv workers re-seed via envs.reset(seed=group_seed) inside
    # _align_envs_to_group_scene (collect_episodes.py:1290). Seeding parent np
    # here would imply reproducibility that this script cannot deliver.

    collector = EvalCollector(
        env_name=args.env_name,
        group_size=args.num_attempts,
        max_episode_steps=args.max_episode_steps,
        n_action_steps=args.n_action_steps,
        server_host=args.server_host,
        server_port=args.server_port,
        debug_fast_forward=False,
        output_dir=str(output_dir),
        num_async_vector_env=args.num_envs,
    )

    t0 = time.monotonic()
    try:
        episodes = collector.collect(
            num_groups=1,
            base_seed=args.seed,
            success_weight=1.0,         # binary; skips dense progress RPC
            fast_forward_steps=0,
            fast_forward_pct=0.0,
            min_alive_groups=0,         # disables dynamic group collection
            max_groups=1,
            init_state_npz_path=str(obs_path),
        )
    finally:
        collector.close()
    duration = time.monotonic() - t0

    summary_block = _summarize(episodes)

    result = {
        "lineage": {
            "obs_path": str(obs_path),
            "lora_checkpoint": args.lora_checkpoint,
            "env_name": args.env_name,
            "num_attempts": args.num_attempts,
            "num_envs": args.num_envs,
            "max_episode_steps": args.max_episode_steps,
            "n_action_steps": args.n_action_steps,
            "branch_step": branch_step,
            "saved_n_action_steps": saved_n_action_steps,
            "consumed_substeps": consumed_substeps,
            "remaining_substeps_budget": remaining_budget,
            "server_host": args.server_host,
            "server_port": args.server_port,
            "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(duration, 2),
        },
        **summary_block,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print()
    print("=" * 60)
    print(
        f"Successes: {s['successes']}/{s['total']} "
        f"({s['success_rate']*100:.1f}%)"
    )
    if s["mean_num_steps_all"] is not None:
        print(f"Mean num_steps (all): {s['mean_num_steps_all']:.1f}")
    if s["mean_num_steps_successful"] is not None:
        print(f"Mean num_steps (success): {s['mean_num_steps_successful']:.1f}")
    if s["mean_num_steps_failed"] is not None:
        print(f"Mean num_steps (fail/trunc): {s['mean_num_steps_failed']:.1f}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
