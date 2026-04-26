"""Grid search for density-aware denoising hyperparameters.

Loads the GR00T model ONCE, starts a ZMQ policy server in a background thread,
then iterates over a grid of (alpha, h, mode) values.  For each config,
re-patches the action head and launches the eval client subprocess
(in the robocasa venv) to run episodes.  Results are parsed from episodes.jsonl
and ranked by combined success rate.

The density-aware strategy caches prev_actions across chunks for anchor
consistency.  The reset_fn returned by patch_action_head is hooked into
policy.reset() so the cache is cleared between episodes.

Usage:
    uv run python scripts/denoising_lab/eval/strategies/density_aware_denoising/calibrate_lambdas.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 400 480 \\
        --n-episodes 15 --seed 42 \\
        --alpha 0.05 0.08 0.10 0.15 \\
        --h 1e-3 5e-3 1e-2 \\
        --mode guided \\
        --output-dir ./calibration_results/density_aware
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strategy import DensityAwareConfig, patch_action_head


# ---------------------------------------------------------------------------
# Config naming
# ---------------------------------------------------------------------------

def config_dirname(cfg: DensityAwareConfig) -> str:
    return f"m{cfg.mode}_a{cfg.alpha}_h{cfg.h}"


def config_dict(cfg: DensityAwareConfig) -> dict[str, Any]:
    return {
        "mode": cfg.mode,
        "alpha": cfg.alpha,
        "h": cfg.h,
        "D0": cfg.D0,
        "N": cfg.N,
        "score_dims": cfg.score_dims,
        "score_horizon": cfg.score_horizon,
    }


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_episodes_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_success_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["success"]) / len(records)


# ---------------------------------------------------------------------------
# Eval client subprocess
# ---------------------------------------------------------------------------

ROBOCASA_PYTHON = "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
BENCHMARK_SCRIPT = "scripts/denoising_lab/eval/robocasa_eval_benchmark.py"


def run_eval_client(
    env_names: list[str],
    max_episode_steps: list[int],
    seeds: list[int],
    output_dir: Path,
    strategy_name: str,
    host: str,
    port: int,
    n_envs: int,
) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []

    for env_name, max_steps in zip(env_names, max_episode_steps):
        env_dir_name = env_name.replace("/", "__")
        jsonl_path = output_dir / env_dir_name / "episodes.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

        base_seed = min(seeds)
        n_episodes = max(seeds) - base_seed + 1

        cmd = [
            ROBOCASA_PYTHON,
            BENCHMARK_SCRIPT,
            "--env-names", env_name,
            "--n-episodes", str(n_episodes),
            "--seed", str(base_seed),
            "--max-episode-steps", str(max_steps),
            "--n-envs", str(n_envs),
            "--host", host,
            "--port", str(port),
            "--output-dir", str(output_dir),
            "--strategy-name", strategy_name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  WARNING: Client exited with code {result.returncode}")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"    {line}")

        env_dir_name = env_name.replace("/", "__")
        jsonl_path = output_dir / env_dir_name / "episodes.jsonl"
        records = parse_episodes_jsonl(jsonl_path)

        if seeds:
            seed_set = set(seeds)
            records = [r for r in records if r["seed"] in seed_set]

        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_grid(args: argparse.Namespace) -> list[DensityAwareConfig]:
    configs = []
    for mode, alpha, h in itertools.product(args.mode, args.alpha, args.h):
        cfg = DensityAwareConfig(
            mode=mode,
            alpha=alpha,
            h=h,
            N=args.N,
            score_dims=args.score_dims,
            score_horizon=args.score_horizon,
        )
        configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for density-aware denoising hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--env-names", nargs="+", required=True)
    parser.add_argument("--max-episode-steps", nargs="+", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds-only", nargs="*", type=int, default=None,
        help="Only report results for these specific seeds",
    )

    parser.add_argument(
        "--mode", nargs="+", type=str,
        default=["guided"],
        help="Operating modes to sweep (guided, rank)",
    )
    parser.add_argument(
        "--alpha", nargs="+", type=float,
        default=[0.05, 0.08, 0.10, 0.15],
        help="Guidance strength values (guided mode)",
    )
    parser.add_argument(
        "--h", nargs="+", type=float,
        default=[1e-3, 5e-3, 1e-2],
        help="Finite-difference perturbation scale values",
    )
    parser.add_argument("--N", type=int, default=4, help="Candidates for rank mode")
    parser.add_argument("--score-dims", type=int, default=12)
    parser.add_argument("--score-horizon", type=int, default=None)

    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", type=str, default="ROBOCASA_PANDA_OMRON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    if len(args.max_episode_steps) != len(args.env_names):
        parser.error("--max-episode-steps must have same length as --env-names")

    args.output_dir = Path(args.output_dir)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds_only is not None and len(args.seeds_only) > 0:
        eval_seeds = sorted(args.seeds_only)
    else:
        eval_seeds = list(range(args.seed, args.seed + args.n_episodes))

    grid = build_grid(args)
    n_configs = len(grid)
    print(f"Density-Aware Calibration: {n_configs} configs")
    print(f"  Episodes scored: {len(eval_seeds) * len(args.env_names) * n_configs}")
    print(f"  Seeds to evaluate: {eval_seeds}")
    print(f"  Output: {args.output_dir}")
    print()

    print("Loading model...")
    t0 = time.monotonic()
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )

    _active_reset_fn = [None]

    _original_reset = policy.reset

    def _patched_reset(options=None):
        if _active_reset_fn[0] is not None:
            _active_reset_fn[0]()
        return _original_reset(options)

    policy.reset = _patched_reset

    policy = Gr00tSimPolicyWrapper(policy)
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    server = PolicyServer(policy=policy, host="0.0.0.0", port=args.port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1.0)
    print(f"Server running on port {args.port}\n")

    results: list[dict[str, Any]] = []
    best_rate = -1.0
    best_cfg_name = ""

    for i, cfg in enumerate(grid):
        cfg_name = config_dirname(cfg)
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{n_configs}] {cfg_name}")

        inner_policy = policy.policy
        reset_fn = patch_action_head(inner_policy.model.action_head, cfg=cfg)
        _active_reset_fn[0] = reset_fn

        records = run_eval_client(
            env_names=args.env_names,
            max_episode_steps=args.max_episode_steps,
            seeds=eval_seeds,
            output_dir=cfg_dir,
            strategy_name=f"density_aware_{cfg_name}",
            host="127.0.0.1",
            port=args.port,
            n_envs=args.n_envs,
        )

        rate = compute_success_rate(records)
        n_success = sum(1 for r in records if r["success"])
        n_total = len(records)

        env_rates = {}
        for env_name in args.env_names:
            env_records = [r for r in records if r["env_name"] == env_name]
            env_rates[env_name] = compute_success_rate(env_records)

        seed_successes = {}
        for r in records:
            key = f"{r['env_name']}__seed{r['seed']}"
            seed_successes[key] = r["success"]

        entry = {
            "config": config_dict(cfg),
            "config_name": cfg_name,
            "success_rate": round(rate, 4),
            "n_success": n_success,
            "n_total": n_total,
            "env_rates": {k: round(v, 4) for k, v in env_rates.items()},
            "seed_successes": seed_successes,
        }
        results.append(entry)

        if rate > best_rate:
            best_rate = rate
            best_cfg_name = cfg_name

        status = "NEW BEST" if cfg_name == best_cfg_name and i > 0 else ""
        print(f"  -> {n_success}/{n_total} ({100*rate:.1f}%) {status}")
        for env_name, er in env_rates.items():
            short = env_name.split("/")[-1] if "/" in env_name else env_name
            print(f"     {short}: {100*er:.1f}%")
        print()

        _write_results(results, args.output_dir, args, eval_seeds)

    _write_results(results, args.output_dir, args, eval_seeds)
    _print_ranking(results)


def _write_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    eval_seeds: list[int],
) -> None:
    ranked = sorted(results, key=lambda r: r["success_rate"], reverse=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_configs_evaluated": len(results),
        "eval_seeds": eval_seeds,
        "env_names": args.env_names,
        "grid": {
            "mode": args.mode,
            "alpha": args.alpha,
            "h": args.h,
        },
        "ranked_results": ranked,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)


def _print_ranking(results: list[dict[str, Any]]) -> None:
    ranked = sorted(results, key=lambda r: r["success_rate"], reverse=True)

    print("\n" + "=" * 70)
    print("FINAL RANKING (top 10)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Config':<35} {'Rate':<10} {'Success'}")
    print("-" * 70)

    for i, entry in enumerate(ranked[:10]):
        print(
            f"{i+1:<5} {entry['config_name']:<35} "
            f"{100*entry['success_rate']:.1f}%     "
            f"{entry['n_success']}/{entry['n_total']}"
        )

    print("-" * 70)
    best = ranked[0]
    print(f"\nBest config: {best['config_name']}")
    print(f"  mode:         {best['config']['mode']}")
    print(f"  alpha:        {best['config']['alpha']}")
    print(f"  h:            {best['config']['h']}")
    print(f"  success_rate: {100*best['success_rate']:.1f}%")
    print()


if __name__ == "__main__":
    main()
