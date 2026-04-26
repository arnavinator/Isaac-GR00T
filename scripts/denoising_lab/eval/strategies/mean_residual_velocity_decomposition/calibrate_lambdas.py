"""Grid search for mean-residual velocity decomposition hyperparameters.

Loads the GR00T model ONCE, starts a ZMQ policy server in a background thread,
then iterates over a grid of (rho, onset, energy_preserve) values.  For each
config, re-patches the action head and launches the eval client subprocess
(in the robocasa venv) to run episodes.  Results are parsed from episodes.jsonl
and ranked by combined success rate.

Usage:
    uv run python scripts/denoising_lab/eval/strategies/mean_residual_velocity_decomposition/calibrate_lambdas.py \\
        --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
                    robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --max-episode-steps 400 480 \\
        --n-episodes 15 --seed 42 \\
        --rho 1.0 1.05 1.10 1.15 1.25 1.50 \\
        --onset 1 2 3 \\
        --energy-preserve True False \\
        --output-dir ./calibration_results/mean_residual
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
from strategy import patch_action_head


# ---------------------------------------------------------------------------
# Config naming
# ---------------------------------------------------------------------------

def config_dirname(rho: float, onset: int, energy_preserve: bool) -> str:
    return f"r{rho}_o{onset}_e{1 if energy_preserve else 0}"


def config_dict(rho: float, onset: int, energy_preserve: bool) -> dict[str, Any]:
    return {
        "rho": rho,
        "onset": onset,
        "energy_preserve": energy_preserve,
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

def build_grid(args: argparse.Namespace) -> list[tuple[float, int, bool]]:
    configs = []
    for rho, onset, ep in itertools.product(
        args.rho, args.onset, args.energy_preserve
    ):
        configs.append((rho, onset, ep))
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for mean-residual velocity decomposition hyperparameters",
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
        "--rho", nargs="+", type=float,
        default=[1.0, 1.05, 1.10, 1.15, 1.25, 1.50],
    )
    parser.add_argument(
        "--onset", nargs="+", type=int,
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--energy-preserve", nargs="+", type=lambda x: x.lower() in ("true", "1", "yes"),
        default=[True, False],
        help="Grid values for energy_preserve (True/False)",
    )

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
    n_run_per_env = max(eval_seeds) - min(eval_seeds) + 1
    print(f"Mean-Residual Calibration: {n_configs} configs")
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

    for i, (rho, onset, energy_preserve) in enumerate(grid):
        cfg_name = config_dirname(rho, onset, energy_preserve)
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{n_configs}] {cfg_name}")

        inner_policy = policy.policy
        patch_action_head(
            inner_policy.model.action_head,
            rho=rho, onset=onset, energy_preserve=energy_preserve,
        )

        records = run_eval_client(
            env_names=args.env_names,
            max_episode_steps=args.max_episode_steps,
            seeds=eval_seeds,
            output_dir=cfg_dir,
            strategy_name=f"mean_residual_{cfg_name}",
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
            "config": config_dict(rho, onset, energy_preserve),
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
            "rho": args.rho,
            "onset": args.onset,
            "energy_preserve": args.energy_preserve,
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
    print(f"  rho:             {best['config']['rho']}")
    print(f"  onset:           {best['config']['onset']}")
    print(f"  energy_preserve: {best['config']['energy_preserve']}")
    print(f"  success_rate:    {100*best['success_rate']:.1f}%")
    print()


if __name__ == "__main__":
    main()
