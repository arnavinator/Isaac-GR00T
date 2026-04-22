"""Profile inference latency of noise-space mode selection vs. K.

Loads the GR00T model ONCE, collects a small set of observations from a sim
env using random actions, then benchmarks the action-head latency for
K={3, 5, 8, 12} (or custom values).  No full episode rollout -- just
repeated get_action calls on cached observations to isolate DiT throughput.

Architecture:
    - Model loads once, runs on GPU
    - Observations collected from a short sim rollout using random actions
    - For each K value, re-patches the action head and times N inference calls
    - Also benchmarks baseline (no patch, K=1 equivalent) for comparison

Usage (single terminal, model venv):
    uv run python scripts/denoising_lab/eval/strategies/noise_space_mode_selection/profile_k_runtime.py \
        --K-values 3 5 8 12 \
        --n-warmup 5 --n-iters 50 \
        --output-dir /tmp/noise_sel_profile
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import NoiseSelectionConfig, patch_action_head


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def benchmark_get_action(policy, observations: list[dict], n_warmup: int,
                         n_iters: int) -> list[float]:
    """Call get_action repeatedly and return per-call latencies in ms."""
    n_obs = len(observations)

    # Warmup (not timed)
    for i in range(n_warmup):
        obs = observations[i % n_obs]
        policy.get_action(obs)

    torch.cuda.synchronize()

    latencies = []
    for i in range(n_iters):
        obs = observations[i % n_obs]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        policy.get_action(obs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

    return latencies


def compute_stats(latencies: list[float]) -> dict:
    """Compute summary statistics for a list of latencies (ms)."""
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "std_ms": round(float(arr.std()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "n_iters": len(latencies),
    }


# ---------------------------------------------------------------------------
# Observation collection
# ---------------------------------------------------------------------------

def collect_observations(env_name: str, n_obs: int, seed: int,
                         n_action_steps: int, max_episode_steps: int) -> list[dict]:
    """Collect observations from a short sim rollout (single env, no server).

    Uses the evaluation env infrastructure directly.  Returns a list of
    observation dicts ready for ``policy.get_action()``.
    """
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        create_eval_env,
    )

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=None, max_episode_steps=max_episode_steps),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        ),
    )
    env = create_eval_env(
        env_name=env_name, env_idx=0, total_n_envs=1,
        wrapper_configs=wrapper_configs,
    )

    observations = []
    obs, _ = env.reset(seed=seed)

    # Add batch dim (policy expects batched observations)
    def batch_obs(obs):
        batched = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                batched[key] = val[np.newaxis]
            elif isinstance(val, str):
                batched[key] = (val,)
            else:
                batched[key] = val
        return batched

    observations.append(batch_obs(obs))

    # Step through to collect diverse observations
    for _ in range(n_obs - 1):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(batch_obs(obs))
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + len(observations))

    env.close()
    return observations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile noise-space mode selection latency vs. K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--K-values", nargs="+", type=int, default=[3, 5, 8, 12],
        help="K values to benchmark",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=5,
        help="Warmup iterations (not timed)",
    )
    parser.add_argument(
        "--n-iters", type=int, default=50,
        help="Timed iterations per K value",
    )
    parser.add_argument(
        "--n-obs", type=int, default=10,
        help="Number of observations to collect for benchmarking",
    )
    parser.add_argument(
        "--env-name", type=str,
        default="robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
        help="Env for observation collection",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=400,
        help="Max episode steps for observation collection env",
    )
    parser.add_argument(
        "--model-path", type=str, default="nvidia/GR00T-N1.6-3B",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--embodiment-tag", type=str, default="ROBOCASA_PANDA_OMRON",
        help="Embodiment tag",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for model",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/noise_sel_profile",
        help="Directory for output JSON",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect observations ---
    print(f"Collecting {args.n_obs} observations from {args.env_name}...")
    observations = collect_observations(
        env_name=args.env_name,
        n_obs=args.n_obs,
        seed=42,
        n_action_steps=8,
        max_episode_steps=args.max_episode_steps,
    )
    print(f"  Collected {len(observations)} observations")

    # --- Load model ---
    print(f"\nLoading model from {args.model_path}...")
    t0 = time.monotonic()
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy = Gr00tSimPolicyWrapper(policy)
    load_time = time.monotonic() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # Save the original get_action_with_features for restoration
    inner_policy = policy.policy
    original_fn = inner_policy.model.action_head.get_action_with_features

    results = {}

    # --- Benchmark baseline (unpatched) ---
    print(f"\nBenchmarking baseline (4-step Euler, no patch)...")
    inner_policy.model.action_head.get_action_with_features = original_fn
    latencies = benchmark_get_action(policy, observations, args.n_warmup, args.n_iters)
    stats = compute_stats(latencies)
    results["baseline"] = {"K": 1, "NFEs": 4, **stats}
    print(f"  mean={stats['mean_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  "
          f"p95={stats['p95_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    # --- Benchmark each K value ---
    for K in args.K_values:
        print(f"\nBenchmarking K={K} ({K}+3={K+3} NFEs)...")

        # Restore original, then re-patch with new K
        inner_policy.model.action_head.get_action_with_features = original_fn
        cfg = NoiseSelectionConfig(K=K)
        _reset_fn = patch_action_head(inner_policy.model.action_head, cfg=cfg)

        latencies = benchmark_get_action(policy, observations, args.n_warmup, args.n_iters)
        stats = compute_stats(latencies)
        results[f"K={K}"] = {"K": K, "NFEs": K + 3, **stats}
        print(f"  mean={stats['mean_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  "
              f"p95={stats['p95_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    # --- Output ---
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": {
            "K_values": args.K_values,
            "n_warmup": args.n_warmup,
            "n_iters": args.n_iters,
            "n_obs": args.n_obs,
            "model_path": args.model_path,
            "device": args.device,
        },
        "results": results,
    }

    output_path = output_dir / "profile_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_path}")

    # --- Summary table ---
    print(f"\n{'='*72}")
    print(f"{'Config':<12} {'K':>4} {'NFEs':>5} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"{'='*72}")
    for name, stats in results.items():
        print(f"{name:<12} {stats['K']:>4} {stats['NFEs']:>5} "
              f"{stats['mean_ms']:>7.1f} {stats['p50_ms']:>7.1f} "
              f"{stats['p95_ms']:>7.1f} {stats['p99_ms']:>7.1f}")
    print(f"{'='*72}")

    # Relative slowdowns
    baseline_mean = results["baseline"]["mean_ms"]
    print(f"\nRelative to baseline ({baseline_mean:.1f}ms):")
    for name, stats in results.items():
        if name == "baseline":
            continue
        ratio = stats["mean_ms"] / baseline_mean
        print(f"  {name}: {ratio:.2f}x")


if __name__ == "__main__":
    main()
