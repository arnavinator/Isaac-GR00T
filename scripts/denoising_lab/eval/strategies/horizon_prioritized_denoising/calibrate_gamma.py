"""Calibrate the optimal gamma for horizon-prioritized denoising.

Sweeps gamma values to find the one whose 4-step horizon-prioritized output
best matches a high-fidelity 64-step Euler reference.

Runs in the **model venv** (needs GPU).  Reuses pre-collected observations
(e.g. from the timestep-schedule calibrator).

Usage::

    uv run python scripts/denoising_lab/eval/strategies/horizon_prioritized_denoising/calibrate_gamma.py \
        --obs-dir /tmp/schedule_calibration/observations \
        --output-dir /tmp/gamma_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate gamma for horizon-prioritized denoising",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    p.add_argument("--embodiment-tag", default="robocasa_panda_omron")
    p.add_argument("--device", default="cuda")

    p.add_argument("--gamma-min", type=float, default=0.0)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-step", type=float, default=0.1)
    p.add_argument("--sigma-w", type=float, default=3.0)
    p.add_argument("--effective-horizon", type=int, default=16)
    p.add_argument("--reference-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--obs-dir", type=str, required=True,
                   help="Directory with pre-collected .npz observations")
    p.add_argument("--output-dir", type=str, required=True)

    args = p.parse_args()
    args.output_dir = Path(args.output_dir)
    args.obs_dir = Path(args.obs_dir)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fix sys.path for imports
    strategy_dir = Path(__file__).resolve().parent
    repo_root = strategy_dir.parents[4]
    sys.path.insert(0, str(strategy_dir))
    sys.path.insert(0, str(repo_root))

    import numpy as np
    from scripts.denoising_lab.denoising_lab import DenoisingLab
    from strategy import denoise_with_lab

    # Load observations
    npz_files = sorted(args.obs_dir.glob("*.npz"))
    if not npz_files:
        print(f"ERROR: no .npz files in {args.obs_dir}")
        sys.exit(1)

    print(f"Loading model from {args.model_path} ...")
    lab = DenoisingLab(args.model_path, args.embodiment_tag, args.device)

    print(f"Encoding {len(npz_files)} observations through backbone ...")
    features_list = []
    for i, path in enumerate(npz_files):
        obs = DenoisingLab.load_observation(path)
        features_list.append(lab.encode_features_from_sim_obs(obs))
        if (i + 1) % 8 == 0 or i + 1 == len(npz_files):
            print(f"  encoded {i + 1}/{len(npz_files)}")

    # Compute high-fidelity references
    print(f"Computing {args.reference_steps}-step Euler references ...")
    references = []
    for i, feat in enumerate(features_list):
        ref = lab.denoise(feat, num_steps=args.reference_steps, seed=args.seed)
        references.append(ref.action_pred)
        if (i + 1) % 8 == 0 or i + 1 == len(features_list):
            print(f"  computed {i + 1}/{len(features_list)}")

    # Build gamma candidates
    gamma_candidates = []
    g = args.gamma_min
    while g <= args.gamma_max + 1e-9:
        gamma_candidates.append(round(g, 2))
        g += args.gamma_step

    print(f"Sweeping {len(gamma_candidates)} gamma values "
          f"over {len(features_list)} observations ...")

    t0 = time.monotonic()
    results = []
    for gamma in gamma_candidates:
        errors = []
        for feat, ref in zip(features_list, references):
            candidate = denoise_with_lab(
                lab, feat, seed=args.seed,
                gamma=gamma, sigma_w=args.sigma_w,
                effective_horizon=args.effective_horizon,
            )
            errors.append((candidate - ref).float().norm().item())
        mean_err = float(np.mean(errors))
        results.append((gamma, mean_err))
        print(f"  gamma={gamma:.2f}  error={mean_err:.6f}")

    results.sort(key=lambda x: x[1])
    best_gamma, best_error = results[0]
    baseline_error = next(err for g, err in results if g == 0.0)
    improvement = (baseline_error - best_error) / baseline_error * 100

    elapsed = time.monotonic() - t0

    result = {
        "best_gamma": best_gamma,
        "best_error": round(best_error, 6),
        "baseline_gamma": 0.0,
        "baseline_error": round(baseline_error, 6),
        "improvement_pct": round(improvement, 2),
        "sigma_w": args.sigma_w,
        "effective_horizon": args.effective_horizon,
        "reference_steps": args.reference_steps,
        "n_observations": len(features_list),
        "seed": args.seed,
        "all_results": [(g, round(e, 6)) for g, e in results],
        "search_time_s": round(elapsed, 1),
    }

    result_path = args.output_dir / "gamma_calibration_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    banner = "=" * 60
    print()
    print(banner)
    print("GAMMA CALIBRATION RESULT")
    print("=" * 60)
    print(f"  Baseline (gamma=0.0): error={baseline_error:.6f}")
    print(f"  Best gamma={best_gamma:.2f}:    error={best_error:.6f}")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Search time: {elapsed:.0f}s")
    print(f"\n  All results (sorted by error):")
    for g, e in results:
        marker = " <-- best" if g == best_gamma else ""
        print(f"    gamma={g:.2f}  error={e:.6f}{marker}")
    print(f"\n  Saved to: {result_path}")
    print(f"\n  To use this gamma, start the server with:")
    print(f"    bash .../run_server.sh --gamma {best_gamma}")


if __name__ == "__main__":
    main()
