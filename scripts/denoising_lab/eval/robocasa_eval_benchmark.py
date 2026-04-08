"""Reproducible RoboCasa evaluation benchmark.

Runs in the **sim venv** (robocasa). Strategy-agnostic -- connects to whatever
GR00T server is already running over ZMQ.  Guarantees identical episodes given
the same seed by using a single env with explicit ``env.reset(seed=...)``
instead of AsyncVectorEnv (which auto-resets with random seeds).

Usage::

    # Terminal 1 (model venv) -- start the ZMQ server
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON \
        --use-sim-policy-wrapper --verbose

    # Terminal 2 (sim venv) -- run reproducible benchmark
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
        --n-episodes 10 --seed 42 \
        --output-dir /tmp/benchmark_results \
        --strategy-name baseline_euler
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)
from gr00t.policy.server_client import PolicyClient

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------

def generate_seeds(base_seed: int, n_episodes: int) -> list[int]:
    """Return deterministic seed list ``[base_seed, base_seed+1, ...]``."""
    return [base_seed + i for i in range(n_episodes)]


# ---------------------------------------------------------------------------
# Observation batching / unbatching (single-env <-> PolicyClient)
# ---------------------------------------------------------------------------

def batch_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """Add a leading batch dimension to a single-env observation.

    Pattern taken from ``interactive_rollout.py`` lines 261-270.
    """
    batched: dict[str, Any] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            batched[key] = val[np.newaxis]
        elif isinstance(val, str):
            batched[key] = (val,)
        elif isinstance(val, (tuple, list)):
            batched[key] = val
        else:
            batched[key] = val
    return batched


def unbatch_action(action: dict[str, Any]) -> dict[str, Any]:
    """Remove the leading batch dimension from a server action response."""
    unbatched: dict[str, Any] = {}
    for key, val in action.items():
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            unbatched[key] = val[0]
        else:
            unbatched[key] = val
    return unbatched


# ---------------------------------------------------------------------------
# Success extraction (mirrors rollout_policy.py lines 302-316)
# ---------------------------------------------------------------------------

def extract_success(info: dict[str, Any]) -> bool:
    """Return True if ``info`` indicates task success."""
    if "success" not in info:
        return False
    s = info["success"]
    if isinstance(s, (list, np.ndarray)):
        return bool(np.any(s))
    if isinstance(s, (bool, int)):
        return bool(s)
    return False


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_single_env(
    env_name: str,
    n_action_steps: int,
    max_episode_steps: int,
    video_dir: str | None,
):
    """Create a single wrapped evaluation env (no vectorisation)."""
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )
    return create_eval_env(
        env_name=env_name,
        env_idx=0,
        total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_single_episode(
    env,
    client: PolicyClient,
    seed: int,
    episode_idx: int,
    env_name: str,
) -> dict[str, Any]:
    """Run one seeded episode, return a result record."""
    t0 = time.monotonic()

    obs, _info = env.reset(seed=seed)
    batched = batch_obs(obs)

    done = False
    success = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action, _ = client.get_action(batched)
        obs, reward, terminated, truncated, info = env.step(unbatch_action(action))
        total_reward += float(reward)
        step_count += 1
        done = terminated or truncated
        success = success or extract_success(info)
        if not done:
            batched = batch_obs(obs)

    duration = time.monotonic() - t0
    return {
        "episode_idx": episode_idx,
        "seed": seed,
        "env_name": env_name,
        "success": success,
        "reward": round(total_reward, 4),
        "length": step_count,
        "n_action_chunks": step_count,
        "duration_s": round(duration, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Per-env benchmark loop
# ---------------------------------------------------------------------------

def _sanitize_env_dir(env_name: str) -> str:
    """Turn an env name into a filesystem-safe directory name."""
    return env_name.replace("/", "__")


def run_benchmark_for_env(
    env_name: str,
    seeds: list[int],
    client: PolicyClient,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run all episodes for a single env, writing JSONL progressively."""
    env_dir = args.output_dir / _sanitize_env_dir(env_name)
    env_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = env_dir / "episodes.jsonl"

    video_dir = str(env_dir / "videos") if args.video else None

    env = create_single_env(
        env_name=env_name,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        video_dir=video_dir,
    )

    records: list[dict[str, Any]] = []
    successes = 0

    n_episodes = len(seeds)
    short_name = env_name.split("/")[-1] if "/" in env_name else env_name

    iter_seeds = enumerate(seeds)
    if tqdm is not None:
        iter_seeds = tqdm(
            list(iter_seeds),
            desc=f"{short_name} [0/{n_episodes}]",
            leave=True,
        )

    for i, seed in iter_seeds:
        record = run_single_episode(env, client, seed, i, env_name)
        records.append(record)

        # Append immediately (crash-recoverable)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if record["success"]:
            successes += 1

        status = "SUCCESS" if record["success"] else "FAIL"
        print(
            f"  ep {i} (seed={seed}): {status} in "
            f"{record['length']} steps ({record['duration_s']}s)"
        )

        if tqdm is not None and hasattr(iter_seeds, "set_description"):
            iter_seeds.set_description(
                f"{short_name} [{i+1}/{n_episodes}] | "
                f"success={successes}/{i+1} ({100*successes/(i+1):.0f}%)"
            )

    rate = 100 * successes / n_episodes if n_episodes else 0
    mean_len = np.mean([r["length"] for r in records]) if records else 0
    total_time = sum(r["duration_s"] for r in records)
    print(
        f"{short_name}: {successes}/{n_episodes} ({rate:.1f}%) | "
        f"mean_len={mean_len:.1f} | {total_time:.1f}s total"
    )

    env.close()
    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(
    all_records: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    args: argparse.Namespace,
    wall_time: float,
) -> None:
    """Write aggregate ``summary.json``."""
    results: dict[str, Any] = {}
    total_episodes = 0

    for env_name, records in all_records.items():
        n = len(records)
        total_episodes += n
        successes_list = [r["success"] for r in records]
        lengths = [r["length"] for r in records]
        results[env_name] = {
            "n_episodes": n,
            "success_rate": round(sum(successes_list) / n, 4) if n else 0,
            "mean_reward": round(np.mean([r["reward"] for r in records]), 4) if n else 0,
            "mean_length": round(float(np.mean(lengths)), 1) if n else 0,
            "std_length": round(float(np.std(lengths)), 1) if n else 0,
            "mean_duration_s": round(float(np.mean([r["duration_s"] for r in records])), 2) if n else 0,
            "successes": successes_list,
            "seeds": [r["seed"] for r in records],
        }

    all_successes = [r["success"] for recs in all_records.values() for r in recs]
    overall_rate = round(sum(all_successes) / len(all_successes), 4) if all_successes else 0

    summary = {
        "strategy_name": args.strategy_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {
            "env_names": args.env_names,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "n_action_steps": args.n_action_steps,
            "host": args.host,
            "port": args.port,
        },
        "results": results,
        "overall": {
            "total_episodes": total_episodes,
            "total_envs": len(all_records),
            "overall_success_rate": overall_rate,
            "total_wall_time_s": round(wall_time, 2),
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible RoboCasa evaluation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env-names",
        nargs="+",
        required=True,
        help="One or more gymnasium env IDs",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Episodes per env",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Action sub-steps per chunk",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=720,
        help="Truncation limit (outer steps)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Results directory",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="unnamed",
        help="Label for this strategy (written to summary.json)",
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to server
    print(f"Connecting to server at {args.host}:{args.port}...")
    client = PolicyClient(host=args.host, port=args.port, strict=False)
    if not client.ping():
        print("ERROR: Server not responding. Make sure the server is running.")
        return
    print("Server connected.\n")

    seeds = generate_seeds(args.seed, args.n_episodes)
    all_records: dict[str, list[dict[str, Any]]] = {}
    wall_start = time.monotonic()

    for env_name in args.env_names:
        print(f"\n{'='*60}")
        print(f"Env: {env_name}")
        print(f"{'='*60}")
        records = run_benchmark_for_env(env_name, seeds, client, args)
        all_records[env_name] = records

    wall_time = time.monotonic() - wall_start
    write_summary(all_records, args.output_dir, args, wall_time)

    # Final console summary
    total_eps = sum(len(r) for r in all_records.values())
    total_succ = sum(r["success"] for recs in all_records.values() for r in recs)
    rate = 100 * total_succ / total_eps if total_eps else 0
    print(
        f"\nOverall: {total_succ}/{total_eps} ({rate:.1f}%) "
        f"across {len(all_records)} env(s)"
    )


if __name__ == "__main__":
    main()
