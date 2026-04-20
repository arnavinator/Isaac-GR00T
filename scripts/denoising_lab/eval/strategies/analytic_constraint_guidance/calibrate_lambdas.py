"""Grid search for analytic constraint guidance lambda hyperparameters.

Loads the GR00T model ONCE, starts a ZMQ policy server in a background thread,
then iterates over a grid of (lambda_smooth, lambda_discrete, lambda_mode, eta)
values.  For each config, re-patches the action head and launches the eval
client subprocess (in the robocasa venv) to run episodes.  Results are parsed
from the per-config episodes.jsonl and ranked by combined success rate.

Architecture:
    - Model loads once (~30s), runs as ZMQ server in a daemon thread
    - Re-patching between configs is safe: no ZMQ requests in flight after
      the client subprocess exits
    - Client subprocess uses the robocasa venv (separate from model venv)

Usage:
    uv run python scripts/denoising_lab/eval/strategies/analytic_constraint_guidance/calibrate_lambdas.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 400 480 \\
        --n-episodes 15 --seed 42 \\
        --seeds-only 45 52 54 55 56 \\
        --lambda-smooth 0.002 0.005 0.01 \\
        --lambda-discrete 0.005 0.01 0.02 \\
        --lambda-mode 0.001 0.003 0.01 \\
        --eta 0.1 \\
        --output-dir ./calibration_results

See also: calibrate_lambdas.sh for a ready-to-run wrapper.
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
from strategy import ConstraintConfig, patch_action_head


# ---------------------------------------------------------------------------
# Config naming
# ---------------------------------------------------------------------------

def config_dirname(cfg: ConstraintConfig) -> str:
    """Produce a short, filesystem-safe directory name for a config."""
    return (
        f"s{cfg.lambda_smooth}_d{cfg.lambda_discrete}"
        f"_m{cfg.lambda_mode}_e{cfg.eta}"
    )


def config_dict(cfg: ConstraintConfig) -> dict[str, float]:
    """Serialize the tunable params to a JSON-friendly dict."""
    return {
        "lambda_smooth": cfg.lambda_smooth,
        "lambda_discrete": cfg.lambda_discrete,
        "lambda_mode": cfg.lambda_mode,
        "eta": cfg.eta,
    }


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_episodes_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a newline-delimited JSON file of episode records."""
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
    """Combined success rate across all episodes."""
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
    """Launch the eval benchmark client subprocess for each env.

    Returns combined episode records across all envs.
    """
    all_records: list[dict[str, Any]] = []

    for env_name, max_steps in zip(env_names, max_episode_steps):
        # Clear stale episodes.jsonl to prevent duplicates on re-run
        # (the benchmark runner opens in append mode)
        env_dir_name = env_name.replace("/", "__")
        jsonl_path = output_dir / env_dir_name / "episodes.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

        # Build seed-specific args: seed=min(seeds), n_episodes covers all
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  WARNING: Client exited with code {result.returncode}")
            if result.stderr:
                # Print last 10 lines of stderr for debugging
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    print(f"    {line}")

        # Parse episodes.jsonl for this env
        env_dir_name = env_name.replace("/", "__")
        jsonl_path = output_dir / env_dir_name / "episodes.jsonl"
        records = parse_episodes_jsonl(jsonl_path)

        # Filter to only the requested seeds.
        # NOTE: The same seed filter applies to all envs. If hard seeds differ
        # per env (e.g., 45,52 for OpenDrawer vs 54,55,56 for CoffeeServeMug),
        # pass the union. Non-hard seeds that succeed easily just raise the
        # baseline floor uniformly across configs — configs that flip hard seeds
        # will still rank higher.
        if seeds:
            seed_set = set(seeds)
            records = [r for r in records if r["seed"] in seed_set]

        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_grid(args: argparse.Namespace) -> list[ConstraintConfig]:
    """Build the full grid of ConstraintConfig objects from CLI args."""
    configs = []
    for ls, ld, lm, eta in itertools.product(
        args.lambda_smooth, args.lambda_discrete, args.lambda_mode, args.eta
    ):
        configs.append(ConstraintConfig(
            lambda_smooth=ls,
            lambda_discrete=ld,
            lambda_mode=lm,
            eta=eta,
        ))
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for constraint guidance lambda hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environments
    parser.add_argument(
        "--env-names", nargs="+", required=True,
        help="Environment IDs to evaluate on",
    )
    parser.add_argument(
        "--max-episode-steps", nargs="+", type=int, required=True,
        help="Max episode steps per env (must match --env-names order)",
    )

    # Seeds / episodes
    parser.add_argument(
        "--n-episodes", type=int, default=15,
        help="Total episodes (seeds = base_seed .. base_seed+n_episodes-1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed",
    )
    parser.add_argument(
        "--seeds-only", nargs="*", type=int, default=None,
        help="If provided, only report results for these specific seeds "
             "(still runs all episodes between base_seed and max(seeds-only), "
             "but filters results). Useful for screening hard seeds only.",
    )

    # Lambda grid
    parser.add_argument(
        "--lambda-smooth", nargs="+", type=float,
        default=[0.002, 0.005, 0.01],
        help="Grid values for lambda_smooth",
    )
    parser.add_argument(
        "--lambda-discrete", nargs="+", type=float,
        default=[0.005, 0.01, 0.02],
        help="Grid values for lambda_discrete",
    )
    parser.add_argument(
        "--lambda-mode", nargs="+", type=float,
        default=[0.001, 0.003, 0.01],
        help="Grid values for lambda_mode",
    )
    parser.add_argument(
        "--eta", nargs="+", type=float,
        default=[0.1],
        help="Grid values for eta (overall guidance strength)",
    )

    # Model / server
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
        "--port", type=int, default=5555,
        help="ZMQ server port",
    )

    # Eval client
    parser.add_argument(
        "--n-envs", type=int, default=2,
        help="Parallel envs for batched inference",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Root directory for calibration results",
    )

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

    # Determine which seeds to evaluate
    if args.seeds_only is not None and len(args.seeds_only) > 0:
        eval_seeds = sorted(args.seeds_only)
    else:
        eval_seeds = list(range(args.seed, args.seed + args.n_episodes))

    grid = build_grid(args)
    n_configs = len(grid)
    n_evaluated = n_configs * len(eval_seeds) * len(args.env_names)
    # Actual episodes run is larger: we run the full range [min, max] of seeds
    # to cover gaps, then filter to eval_seeds
    n_run_per_env = max(eval_seeds) - min(eval_seeds) + 1
    n_actual = n_configs * n_run_per_env * len(args.env_names)
    print(f"Lambda Calibration: {n_configs} configs")
    print(f"  Episodes scored: {n_evaluated} "
          f"({len(eval_seeds)} seeds x {len(args.env_names)} envs x {n_configs} configs)")
    print(f"  Episodes run:    {n_actual} "
          f"({n_run_per_env} seeds x {len(args.env_names)} envs x {n_configs} configs)")
    print(f"Seeds to evaluate: {eval_seeds}")
    print(f"Output: {args.output_dir}")
    print()

    # --- Load model (ONCE) ---
    print("Loading model...")
    t0 = time.monotonic()
    embodiment_tag = EmbodimentTag[args.embodiment_tag]
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    policy = Gr00tSimPolicyWrapper(policy)
    load_time = time.monotonic() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # --- Start server in daemon thread ---
    server = PolicyServer(policy=policy, host="0.0.0.0", port=args.port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    # Give server a moment to bind
    time.sleep(1.0)
    print(f"Server running on port {args.port}")
    print()

    # --- Grid search ---
    results: list[dict[str, Any]] = []
    best_rate = -1.0
    best_cfg_name = ""

    for i, cfg in enumerate(grid):
        cfg_name = config_dirname(cfg)
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{n_configs}] {cfg_name}")

        # Patch action head with new config
        inner_policy = policy.policy  # unwrap Gr00tSimPolicyWrapper
        patch_action_head(inner_policy.model.action_head, cfg=cfg)

        # Run eval client
        records = run_eval_client(
            env_names=args.env_names,
            max_episode_steps=args.max_episode_steps,
            seeds=eval_seeds,
            output_dir=cfg_dir,
            strategy_name=f"constraint_guidance_{cfg_name}",
            host="127.0.0.1",
            port=args.port,
            n_envs=args.n_envs,
        )

        # Compute metrics
        rate = compute_success_rate(records)
        n_success = sum(1 for r in records if r["success"])
        n_total = len(records)

        # Per-env breakdown
        env_rates = {}
        for env_name in args.env_names:
            env_records = [r for r in records if r["env_name"] == env_name]
            env_rates[env_name] = compute_success_rate(env_records)

        # Per-seed success tracking
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

        # Track best
        if rate > best_rate:
            best_rate = rate
            best_cfg_name = cfg_name

        status = "NEW BEST" if cfg_name == best_cfg_name and i > 0 else ""
        print(f"  -> {n_success}/{n_total} ({100*rate:.1f}%) {status}")
        for env_name, er in env_rates.items():
            short = env_name.split("/")[-1] if "/" in env_name else env_name
            print(f"     {short}: {100*er:.1f}%")
        print()

        # Write incremental results (crash-recoverable)
        _write_results(results, args.output_dir, args, eval_seeds)

    # --- Final output ---
    _write_results(results, args.output_dir, args, eval_seeds)
    _print_ranking(results)


def _write_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    eval_seeds: list[int],
) -> None:
    """Write the ranked results JSON."""
    ranked = sorted(results, key=lambda r: r["success_rate"], reverse=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_configs_evaluated": len(results),
        "eval_seeds": eval_seeds,
        "env_names": args.env_names,
        "grid": {
            "lambda_smooth": args.lambda_smooth,
            "lambda_discrete": args.lambda_discrete,
            "lambda_mode": args.lambda_mode,
            "eta": args.eta,
        },
        "ranked_results": ranked,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)


def _print_ranking(results: list[dict[str, Any]]) -> None:
    """Print final top-10 ranking to console."""
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
    print(f"  lambda_smooth:   {best['config']['lambda_smooth']}")
    print(f"  lambda_discrete: {best['config']['lambda_discrete']}")
    print(f"  lambda_mode:     {best['config']['lambda_mode']}")
    print(f"  eta:             {best['config']['eta']}")
    print(f"  success_rate:    {100*best['success_rate']:.1f}%")
    print()


if __name__ == "__main__":
    main()
