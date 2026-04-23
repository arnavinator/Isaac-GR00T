"""Grid search for DDTO quality-loss hyperparameters.

Loads the GR00T model ONCE, starts a ZMQ policy server in a background thread,
then iterates over a grid of (lambda_smooth, lambda_anchor, lambda_mode, eta)
values.  For each config, re-patches the action head and launches the eval
client subprocess (in the robocasa venv) to run episodes.  Results are parsed
from the per-config episodes.jsonl and ranked by combined success rate.

Architecture:
    - Model loads once (~30s), runs as ZMQ server in a daemon thread
    - Re-patching between configs is safe: no ZMQ requests in flight after
      the client subprocess exits
    - Client subprocess uses the robocasa venv (separate from model venv)

Usage:
    uv run python scripts/denoising_lab/eval/strategies/differentiable_denoising_trajectory_optimization/calibrate_lambdas.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 400 480 \\
        --n-episodes 15 --seed 42 \\
        --lambda-smooth 0.5 1.0 \\
        --lambda-anchor 0.25 0.5 1.0 \\
        --lambda-mode 0.0 0.05 0.1 \\
        --eta 0.05 0.1 0.2 \\
        --output-dir ./calibration_results/ddto
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
from strategy import DDTOConfig, patch_action_head


# ---------------------------------------------------------------------------
# Config naming
# ---------------------------------------------------------------------------

def config_dirname(cfg: DDTOConfig) -> str:
    """Produce a short, filesystem-safe directory name for a config."""
    return (
        f"sm{cfg.lambda_smooth}_an{cfg.lambda_anchor}"
        f"_md{cfg.lambda_mode}_eta{cfg.eta}"
    )


def config_dict(cfg: DDTOConfig) -> dict[str, float]:
    """Serialise the tunable params to a JSON-friendly dict."""
    return {
        "lambda_smooth": cfg.lambda_smooth,
        "lambda_anchor": cfg.lambda_anchor,
        "lambda_mode": cfg.lambda_mode,
        "anchor_decay": cfg.anchor_decay,
        "eta": cfg.eta,
        "n_exec_steps": cfg.n_exec_steps,
        "num_steps": cfg.num_steps,
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
    """Launch the eval benchmark client subprocess for each env."""
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
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    print(f"    {line}")

        records = parse_episodes_jsonl(jsonl_path)

        if seeds:
            seed_set = set(seeds)
            records = [r for r in records if r["seed"] in seed_set]

        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_grid(args: argparse.Namespace) -> list[DDTOConfig]:
    """Build the full grid of DDTOConfig objects from CLI args."""
    configs = []
    for ls, la, lm, eta in itertools.product(
        args.lambda_smooth, args.lambda_anchor, args.lambda_mode, args.eta,
    ):
        configs.append(DDTOConfig(
            lambda_smooth=ls,
            lambda_anchor=la,
            lambda_mode=lm,
            anchor_decay=args.anchor_decay,
            n_exec_steps=args.n_exec_steps,
            eta=eta,
            num_steps=args.num_steps,
        ))
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for DDTO quality-loss hyperparameters",
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
        help="If provided, only report results for these specific seeds",
    )

    # Lambda grid
    parser.add_argument(
        "--lambda-smooth", nargs="+", type=float,
        default=[0.5, 1.0],
        help="Grid values for lambda_smooth",
    )
    parser.add_argument(
        "--lambda-anchor", nargs="+", type=float,
        default=[0.25, 0.5, 1.0],
        help="Grid values for lambda_anchor",
    )
    parser.add_argument(
        "--lambda-mode", nargs="+", type=float,
        default=[0.0, 0.05, 0.1],
        help="Grid values for lambda_mode (0.0 = Variant B, no HVP)",
    )
    parser.add_argument(
        "--eta", nargs="+", type=float,
        default=[0.05, 0.1, 0.2],
        help="Grid values for gradient step size eta",
    )

    # Fixed DDTO parameters
    parser.add_argument(
        "--anchor-decay", type=float, default=0.5,
        help="Per-step decay for distance-weighted anchor scoring",
    )
    parser.add_argument(
        "--n-exec-steps", type=int, default=8,
        help="Number of action steps executed per chunk",
    )
    parser.add_argument(
        "--num-steps", type=int, default=4,
        help="Number of denoising steps",
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
    n_run_per_env = max(eval_seeds) - min(eval_seeds) + 1
    n_actual = n_configs * n_run_per_env * len(args.env_names)
    print(f"DDTO Lambda Calibration: {n_configs} configs")
    print(f"  Grid: lambda_smooth={args.lambda_smooth}")
    print(f"         lambda_anchor={args.lambda_anchor}")
    print(f"         lambda_mode={args.lambda_mode}")
    print(f"         eta={args.eta}")
    print(f"  Fixed: anchor_decay={args.anchor_decay}, "
          f"n_exec_steps={args.n_exec_steps}, num_steps={args.num_steps}")
    print(f"  Episodes scored: {len(eval_seeds) * len(args.env_names) * n_configs}")
    print(f"  Episodes run:    {n_actual}")
    print(f"  Seeds: {eval_seeds}")
    print(f"  Output: {args.output_dir}")
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
    time.sleep(1.0)
    print(f"Server running on port {args.port}")
    print()

    # --- Grid search ---
    # Hook reset into inner policy so cached prev_actions is cleared between
    # episodes.  Each re-patch below creates a new closure, so we re-hook.
    inner_policy = policy.policy
    _original_reset = inner_policy.reset
    _current_reset_fn = [lambda: None]  # placeholder

    def _patched_reset(options=None):
        _current_reset_fn[0]()
        return _original_reset(options)

    inner_policy.reset = _patched_reset

    results: list[dict[str, Any]] = []
    best_rate = -1.0
    best_cfg_name = ""

    for i, cfg in enumerate(grid):
        cfg_name = config_dirname(cfg)
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{n_configs}] {cfg_name}")

        # Patch action head with new config (returns fresh reset callable)
        reset_fn = patch_action_head(inner_policy.model.action_head, cfg=cfg)
        _current_reset_fn[0] = reset_fn

        # Run eval client
        records = run_eval_client(
            env_names=args.env_names,
            max_episode_steps=args.max_episode_steps,
            seeds=eval_seeds,
            output_dir=cfg_dir,
            strategy_name=f"ddto_{cfg_name}",
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

        # Per-seed tracking
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

        status = " ** NEW BEST **" if cfg_name == best_cfg_name and i > 0 else ""
        print(f"  -> {n_success}/{n_total} ({100*rate:.1f}%){status}")
        for env_name, er in env_rates.items():
            short = env_name.split("/")[-1] if "/" in env_name else env_name
            print(f"     {short}: {100*er:.1f}%")
        print()

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
            "lambda_anchor": args.lambda_anchor,
            "lambda_mode": args.lambda_mode,
            "eta": args.eta,
            "anchor_decay": args.anchor_decay,
            "n_exec_steps": args.n_exec_steps,
            "num_steps": args.num_steps,
        },
        "ranked_results": ranked,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)


def _print_ranking(results: list[dict[str, Any]]) -> None:
    """Print final top-10 ranking to console."""
    ranked = sorted(results, key=lambda r: r["success_rate"], reverse=True)

    print("\n" + "=" * 75)
    print("FINAL RANKING (top 10)")
    print("=" * 75)
    print(f"{'Rank':<5} {'Config':<45} {'Rate':<10} {'Success'}")
    print("-" * 75)

    for i, entry in enumerate(ranked[:10]):
        print(
            f"{i+1:<5} {entry['config_name']:<45} "
            f"{100*entry['success_rate']:.1f}%     "
            f"{entry['n_success']}/{entry['n_total']}"
        )

    print("-" * 75)
    best = ranked[0]
    print(f"\nBest config: {best['config_name']}")
    print(f"  lambda_smooth: {best['config']['lambda_smooth']}")
    print(f"  lambda_anchor: {best['config']['lambda_anchor']}")
    print(f"  lambda_mode:   {best['config']['lambda_mode']}")
    print(f"  eta:           {best['config']['eta']}")
    print(f"  anchor_decay:  {best['config']['anchor_decay']}")
    print(f"  success_rate:  {100*best['success_rate']:.1f}%")
    print()


if __name__ == "__main__":
    main()
