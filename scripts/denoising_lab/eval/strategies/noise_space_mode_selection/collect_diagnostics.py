"""Collect per-chunk scoring diagnostics for noise-space mode selection.

Runs a grid of configs (like ``calibrate_lambdas.py``) but additionally
captures the full scoring internals at every action chunk: per-candidate
score breakdowns, all K noise/velocity/proxy tensors, and the winner's
denoising trajectory.  Saves one pickle per config for notebook analysis.

Architecture:
    - Model loads once (~30s), runs as ZMQ server in a daemon thread
    - Re-patching between configs is safe: no ZMQ requests in flight after
      the client subprocess exits
    - Client subprocess uses the robocasa venv (separate from model venv)
    - Diagnostics are collected server-side (no ZMQ overhead)

Usage:
    uv run python scripts/denoising_lab/eval/strategies/noise_space_mode_selection/collect_diagnostics.py \\
        --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \\
        --max-episode-steps 400 \\
        --n-episodes 5 --seed 42 \\
        --lambda-smooth 0.7 1.0 \\
        --lambda-mag 0.01 \\
        --lambda-anchor 1.0 2.0 \\
        --noise-type gaussian uniform \\
        --truncate-horizon 16 --truncate-dim 29 \\
        --output-dir ./diagnostics_results
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
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
from strategy import NoiseSelectionConfig, patch_action_head


# ---------------------------------------------------------------------------
# Config naming (reused from calibrate_lambdas.py)
# ---------------------------------------------------------------------------

def config_dirname(cfg: NoiseSelectionConfig) -> str:
    return (
        f"sm{cfg.lambda_smooth}_mg{cfg.lambda_mag}"
        f"_an{cfg.lambda_anchor}_K{cfg.K}_{cfg.noise_type[:4]}"
    )


def config_dict(cfg: NoiseSelectionConfig) -> dict:
    return {
        "K": cfg.K,
        "lambda_smooth": cfg.lambda_smooth,
        "lambda_mag": cfg.lambda_mag,
        "lambda_anchor": cfg.lambda_anchor,
        "anchor_decay": cfg.anchor_decay,
        "noise_type": cfg.noise_type,
        "num_steps": cfg.num_steps,
        "n_exec_steps": cfg.n_exec_steps,
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


# ---------------------------------------------------------------------------
# Eval client subprocess
# ---------------------------------------------------------------------------

ROBOCASA_PYTHON = "gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python"
BENCHMARK_SCRIPT = "scripts/denoising_lab/eval/robocasa_eval_benchmark.py"


def run_eval_client(
    env_name: str,
    max_episode_steps: int,
    n_episodes: int,
    seed: int,
    output_dir: Path,
    strategy_name: str,
    host: str,
    port: int,
) -> list[dict[str, Any]]:
    """Launch eval client subprocess (single env, single-threaded, with reset)."""
    env_dir_name = env_name.replace("/", "__")
    jsonl_path = output_dir / env_dir_name / "episodes.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    cmd = [
        ROBOCASA_PYTHON,
        BENCHMARK_SCRIPT,
        "--env-names", env_name,
        "--n-episodes", str(n_episodes),
        "--seed", str(seed),
        "--max-episode-steps", str(max_episode_steps),
        "--n-envs", "1",
        "--reset-between-episodes",
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

    return parse_episodes_jsonl(jsonl_path)


# ---------------------------------------------------------------------------
# Merge diagnostics with episode results
# ---------------------------------------------------------------------------

def _truncate_tensor(t, trunc_h, trunc_d):
    """Slice a tensor's last two dims to (trunc_h, trunc_d)."""
    if t is None:
        return None
    if t.ndim == 2:
        return t[:trunc_h, :trunc_d]
    elif t.ndim == 3:
        return t[:, :trunc_h, :trunc_d]
    return t


def merge_and_save(
    diagnostics_log: list[dict],
    episode_results: list[dict],
    cfg: NoiseSelectionConfig,
    metadata: dict,
    output_path: Path,
    trunc_h: int | None,
    trunc_d: int | None,
) -> None:
    """Split flat diagnostics by episode, optionally truncate, save pickle."""
    # Split diagnostics into episodes using n_action_chunks from jsonl
    episodes = []
    offset = 0
    for ep_result in episode_results:
        n_chunks = ep_result.get("n_action_chunks", ep_result.get("length", 0))
        ep_chunks = diagnostics_log[offset:offset + n_chunks]
        offset += n_chunks

        if len(ep_chunks) != n_chunks:
            print(f"  WARNING: Episode {ep_result['episode_idx']}: "
                  f"expected {n_chunks} chunks, got {len(ep_chunks)}")

        # Optionally truncate tensors
        if trunc_h is not None and trunc_d is not None:
            for chunk in ep_chunks:
                for key in ["noise_candidates", "action_proxies_1star",
                            "velocities"]:
                    if key in chunk and chunk[key] is not None:
                        chunk[key] = _truncate_tensor(chunk[key], trunc_h, trunc_d)
                for key in ["prev_actions", "final_actions"]:
                    if key in chunk and chunk[key] is not None:
                        chunk[key] = _truncate_tensor(chunk[key], trunc_h, trunc_d)
                for list_key in ["denoising_actions", "denoising_velocities"]:
                    if list_key in chunk:
                        chunk[list_key] = [
                            _truncate_tensor(t, trunc_h, trunc_d)
                            for t in chunk[list_key]
                        ]

        entry = dict(ep_result)
        entry["chunks"] = ep_chunks
        episodes.append(entry)

    if offset != len(diagnostics_log):
        print(f"  WARNING: {len(diagnostics_log)} total diagnostic chunks "
              f"but episode n_action_chunks sums to {offset}")

    output = {
        "config": config_dict(cfg),
        "metadata": metadata,
        "episodes": episodes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(output, f)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_grid(args: argparse.Namespace) -> list[NoiseSelectionConfig]:
    configs = []
    for ls, lm, la, nt in itertools.product(
        args.lambda_smooth, args.lambda_mag, args.lambda_anchor,
        args.noise_type,
    ):
        configs.append(NoiseSelectionConfig(
            K=args.K,
            lambda_smooth=ls,
            lambda_mag=lm,
            lambda_anchor=la,
            anchor_decay=args.anchor_decay,
            noise_type=nt,
        ))
    return configs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-chunk scoring diagnostics for noise selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--env-name", type=str, required=True,
        help="Single environment ID",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, required=True,
        help="Max episode steps",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Episodes to run",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed",
    )

    # Lambda grid
    parser.add_argument(
        "--lambda-smooth", nargs="+", type=float, default=[1.0],
    )
    parser.add_argument(
        "--lambda-mag", nargs="+", type=float, default=[0.01],
    )
    parser.add_argument(
        "--lambda-anchor", nargs="+", type=float, default=[0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--noise-type", nargs="+", type=str, default=["gaussian"],
        choices=["gaussian", "uniform"],
    )

    # Fixed params
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--anchor-decay", type=float, default=0.5)

    # Truncation
    parser.add_argument(
        "--truncate-horizon", type=int, default=None,
        help="Truncate action horizon dim (e.g., 16 for PandaOmron)",
    )
    parser.add_argument(
        "--truncate-dim", type=int, default=None,
        help="Truncate action dim (e.g., 29 for PandaOmron)",
    )

    # Model / server
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--embodiment-tag", type=str, default="ROBOCASA_PANDA_OMRON")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=5555)

    # Output
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args)
    n_configs = len(grid)
    trunc_h = args.truncate_horizon
    trunc_d = args.truncate_dim
    trunc_label = f"({trunc_h}, {trunc_d})" if trunc_h else "none (full padded)"

    print(f"Diagnostics Collection: {n_configs} configs")
    print(f"  Env: {args.env_name}")
    print(f"  Episodes: {args.n_episodes}, seed: {args.seed}")
    print(f"  K={args.K}, anchor_decay={args.anchor_decay}")
    print(f"  Truncation: {trunc_label}")
    print(f"  Output: {args.output_dir}")
    print()

    # --- Load model ---
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

    # --- Start server ---
    server = PolicyServer(policy=policy, host="0.0.0.0", port=args.port)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1.0)
    print(f"Server running on port {args.port}\n")

    # --- Reset hook (re-pointed per config) ---
    inner_policy = policy.policy
    _original_reset = inner_policy.reset
    _current_reset_fn = [lambda: None]

    def _patched_reset(options=None):
        _current_reset_fn[0]()
        return _original_reset(options)

    inner_policy.reset = _patched_reset

    # --- Iterate configs ---
    for i, cfg in enumerate(grid):
        cfg_name = config_dirname(cfg)
        cfg_dir = args.output_dir / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1}/{n_configs}] {cfg_name}")

        diagnostics_log: list[dict] = []

        reset_fn = patch_action_head(
            inner_policy.model.action_head, cfg=cfg,
            diagnostics_log=diagnostics_log,
        )
        _current_reset_fn[0] = reset_fn

        # Run eval
        records = run_eval_client(
            env_name=args.env_name,
            max_episode_steps=args.max_episode_steps,
            n_episodes=args.n_episodes,
            seed=args.seed,
            output_dir=cfg_dir,
            strategy_name=f"diag_{cfg_name}",
            host="127.0.0.1",
            port=args.port,
        )

        # Success rate
        n_success = sum(1 for r in records if r["success"])
        rate = n_success / len(records) if records else 0
        print(f"  Success: {n_success}/{len(records)} ({100*rate:.1f}%)")
        print(f"  Chunks collected: {len(diagnostics_log)}")

        # Summary stats
        if diagnostics_log:
            gaps = [d["score_gap"] for d in diagnostics_log]
            unique_best = len(set(d["best_k"] for d in diagnostics_log))
            print(f"  Score gap: mean={sum(gaps)/len(gaps):.4f}, "
                  f"min={min(gaps):.4f}, max={max(gaps):.4f}")
            print(f"  Unique best_k values: {unique_best}/{cfg.K}")

        # Merge and save
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "env_name": args.env_name,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "model_path": args.model_path,
            "truncate_horizon": trunc_h,
            "truncate_dim": trunc_d,
            "n_total_chunks": len(diagnostics_log),
        }

        pkl_path = cfg_dir / "diagnostics.pkl"
        merge_and_save(
            diagnostics_log, records, cfg, metadata, pkl_path,
            trunc_h, trunc_d,
        )
        print(f"  Saved: {pkl_path}\n")

    print(f"Done. {n_configs} diagnostic pickles in {args.output_dir}")


if __name__ == "__main__":
    main()
