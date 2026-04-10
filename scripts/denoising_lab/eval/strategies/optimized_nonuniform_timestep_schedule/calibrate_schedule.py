"""Calibrate the optimal non-uniform timestep schedule for GR00T N1.6.

All-in-one script that:

1. Starts a temporary baseline GR00T server (background thread)
2. Spawns a sim-venv subprocess to collect diverse observations from rollouts
3. Encodes the collected observations through the model backbone
4. Random-searches for the 4-step tau schedule that best approximates a
   high-fidelity 64-step Euler reference
5. Outputs the optimal schedule as JSON and a ready-to-use --schedule flag

Runs in the **model venv** (needs GPU).  The sim rollout subprocess uses the
robocasa sim venv automatically.

Usage::

    uv run python scripts/denoising_lab/eval/strategies/optimized_nonuniform_timestep_schedule/calibrate_schedule.py \
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
        --n-episodes 5 --seed 42 \
        --output-dir /tmp/schedule_calibration

    # Or skip collection if you already have observations:
    uv run python .../calibrate_schedule.py \
        --obs-dir /tmp/my_saved_obs \
        --output-dir /tmp/schedule_calibration
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate non-uniform timestep schedule for GR00T N1.6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("observation collection (same flags as the eval benchmark)")
    g.add_argument("--env-names", nargs="+", default=None,
                   help="Gymnasium env IDs for observation collection.  "
                        "Not needed if --obs-dir is provided.")
    g.add_argument("--n-episodes", type=int, default=5,
                   help="Episodes per env to collect observations from")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n-action-steps", type=int, default=8)
    g.add_argument("--max-episode-steps", type=int, default=720)
    g.add_argument("--obs-per-episode", type=int, default=4,
                   help="Observations to save per episode (evenly spaced)")

    g = p.add_argument_group("model")
    g.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    g.add_argument("--embodiment-tag", default="robocasa_panda_omron")
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("grid search")
    g.add_argument("--n-candidates", type=int, default=1000,
                   help="Random schedules to evaluate")
    g.add_argument("--reference-steps", type=int, default=64,
                   help="Euler steps for the high-fidelity reference")

    g = p.add_argument_group("paths")
    g.add_argument("--output-dir", type=str, required=True)
    g.add_argument("--obs-dir", type=str, default=None,
                   help="Pre-collected observations.  If set and non-empty, "
                        "skip collection entirely.")
    g.add_argument("--sim-python",
                   default="gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python",
                   help="Path to the sim-venv Python interpreter")
    g.add_argument("--server-port", type=int, default=5556,
                   help="Port for the temporary baseline server "
                        "(pick something other than 5555 to avoid conflicts)")

    args = p.parse_args()
    args.output_dir = Path(args.output_dir)
    return args


# ---------------------------------------------------------------------------
# Phase 1 — collect observations
# ---------------------------------------------------------------------------

def _collect_observations(args: argparse.Namespace, obs_dir: Path) -> int:
    """Start a temporary server and spawn the sim-venv collector."""
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    print(f"Loading model from {args.model_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        model_path=args.model_path,
        device=args.device,
    )
    policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(policy=policy, host="0.0.0.0", port=args.server_port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    print(f"Temporary baseline server started on port {args.server_port}")
    time.sleep(2)  # give ZMQ time to bind

    collector = str(
        Path(__file__).resolve().parent / "_collect_observations.py"
    )

    for env_name in args.env_names:
        cmd = [
            args.sim_python, collector,
            "--env-name", env_name,
            "--n-episodes", str(args.n_episodes),
            "--seed", str(args.seed),
            "--obs-per-episode", str(args.obs_per_episode),
            "--n-action-steps", str(args.n_action_steps),
            "--max-episode-steps", str(args.max_episode_steps),
            "--host", "127.0.0.1",
            "--port", str(args.server_port),
            "--output-dir", str(obs_dir),
        ]
        print(f"\nCollecting observations from {env_name} ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: collector exited with code {result.returncode}")

    n_obs = len(list(obs_dir.glob("*.npz")))
    print(f"\nCollection complete — {n_obs} observations in {obs_dir}")

    # Free the server's model to reclaim GPU memory before Phase 2
    del server, policy
    gc.collect()
    torch.cuda.empty_cache()

    return n_obs


# ---------------------------------------------------------------------------
# Phase 2 — grid search
# ---------------------------------------------------------------------------

def _run_grid_search(args: argparse.Namespace, obs_dir: Path) -> None:
    import numpy as np
    import torch

    # Strategy imports (local — avoids polluting sys.path at module level)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from strategy import DEFAULT_SCHEDULE, denoise_with_lab  # noqa: E402

    # Late import so Phase 1 can release GPU first
    from scripts.denoising_lab.denoising_lab import DenoisingLab

    print(f"Loading model for grid search ...")
    lab = DenoisingLab(args.model_path, args.embodiment_tag, args.device)

    # Load and encode observations
    npz_files = sorted(obs_dir.glob("*.npz"))
    if not npz_files:
        print("ERROR: no .npz observations found — nothing to calibrate.")
        return

    print(f"Encoding {len(npz_files)} observations through backbone ...")
    features_list = []
    for path in npz_files:
        obs = DenoisingLab.load_observation(path)
        features_list.append(lab.encode_features_from_sim_obs(obs))

    # Compute high-fidelity references (once)
    print(f"Computing {args.reference_steps}-step Euler references ...")
    references = []
    for feat in features_list:
        ref = lab.denoise(feat, num_steps=args.reference_steps, seed=args.seed)
        references.append(ref.action_pred)

    # Baseline: uniform schedule error
    uniform_schedule = [0.0, 0.25, 0.5, 0.75]
    uniform_errors = []
    for feat, ref in zip(features_list, references):
        cand = denoise_with_lab(lab, feat, seed=args.seed, schedule=uniform_schedule)
        uniform_errors.append((cand - ref).float().norm().item())
    uniform_error = float(np.mean(uniform_errors))

    # Random search
    print(f"Searching {args.n_candidates} candidate schedules "
          f"over {len(features_list)} observations ...")
    rng = np.random.RandomState(args.seed)
    best_schedule = list(DEFAULT_SCHEDULE)
    best_error = float("inf")

    t0 = time.monotonic()
    log_interval = max(1, args.n_candidates // 20)  # ~5% progress ticks
    for i in range(args.n_candidates):
        tau_1 = float(rng.uniform(0.02, 0.30))
        tau_2 = float(rng.uniform(tau_1 + 0.05, 0.60))
        tau_3 = float(rng.uniform(tau_2 + 0.05, 0.95))
        schedule = [0.0, tau_1, tau_2, tau_3]

        errors = []
        for feat, ref in zip(features_list, references):
            cand = denoise_with_lab(lab, feat, seed=args.seed, schedule=schedule)
            errors.append((cand - ref).float().norm().item())

        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_schedule = schedule
            print(f"  [{i + 1:>{len(str(args.n_candidates))}}/{args.n_candidates}] "
                  f"new best  error={mean_error:.6f}  "
                  f"schedule={[round(v, 4) for v in schedule]}")
        elif (i + 1) % log_interval == 0:
            elapsed_so_far = time.monotonic() - t0
            eta = elapsed_so_far / (i + 1) * (args.n_candidates - i - 1)
            print(f"  [{i + 1:>{len(str(args.n_candidates))}}/{args.n_candidates}] "
                  f"best so far={best_error:.6f}  "
                  f"ETA {eta:.0f}s")

    elapsed = time.monotonic() - t0
    improvement = (uniform_error - best_error) / uniform_error * 100

    # Write result
    result = {
        "best_schedule": [round(v, 6) for v in best_schedule],
        "best_error": round(best_error, 6),
        "uniform_schedule": uniform_schedule,
        "uniform_error": round(uniform_error, 6),
        "improvement_pct": round(improvement, 2),
        "n_candidates": args.n_candidates,
        "reference_steps": args.reference_steps,
        "n_observations": len(features_list),
        "seed": args.seed,
        "search_time_s": round(elapsed, 1),
    }

    result_path = args.output_dir / "calibration_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    sched_str = " ".join(str(round(v, 4)) for v in best_schedule)
    print(f"\n{'=' * 60}")
    print("CALIBRATION RESULT")
    print(f"{'=' * 60}")
    print(f"  Uniform schedule:  {uniform_schedule}")
    print(f"  Uniform error:     {uniform_error:.6f}")
    print(f"  Optimal schedule:  {[round(v, 4) for v in best_schedule]}")
    print(f"  Optimal error:     {best_error:.6f}")
    print(f"  Improvement:       {improvement:.1f}%")
    print(f"  Search time:       {elapsed:.0f}s "
          f"({args.n_candidates} candidates × {len(features_list)} obs)")
    print(f"\n  Saved to: {result_path}")
    print(f"\n  To use this schedule, start the server with:")
    print(f"    bash .../run_server.sh --schedule {sched_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether we need to collect observations
    if args.obs_dir:
        obs_dir = Path(args.obs_dir)
        existing = list(obs_dir.glob("*.npz"))
        if existing:
            print(f"Using {len(existing)} pre-collected observations "
                  f"from {obs_dir}")
        else:
            print(f"ERROR: --obs-dir {obs_dir} has no .npz files.")
            sys.exit(1)
    else:
        if not args.env_names:
            print("ERROR: provide --env-names (for collection) or "
                  "--obs-dir (to skip collection).")
            sys.exit(1)

        obs_dir = args.output_dir / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Phase 1: Collecting observations from sim rollouts")
        print("=" * 60)
        n = _collect_observations(args, obs_dir)
        if n == 0:
            print("ERROR: no observations collected.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Phase 2: Grid search for optimal schedule")
    print("=" * 60)
    _run_grid_search(args, obs_dir)


if __name__ == "__main__":
    main()
