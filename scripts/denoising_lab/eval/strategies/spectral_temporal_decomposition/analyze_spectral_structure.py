"""Analyze the spectral structure of GR00T velocity fields during denoising.

Profiles the DCT energy spectrum of the velocity field at each of the 4
denoising steps, using saved observations from schedule calibration (or any
DenoisingLab-compatible .npz files).

The key question: does each denoising step resolve a different frequency band?
If step 0 concentrates energy at low DCT coefficients and step 3 at high
coefficients, the spectral strategy's Gaussian affinity profiles are
well-motivated.  If not, the sigma values need recalibrating.

Approach:
  - Load observations, encode through backbone (expensive, once)
  - Run 4-step denoising with a guided_fn probe that captures each step's
    velocity, computes its DCT, and records spectral energy -- then returns
    the velocity unchanged (zero interference with the denoising)
  - Average across observations and seeds

Runs in the **model venv** (needs GPU).

Usage::

    uv run python scripts/denoising_lab/eval/strategies/spectral_temporal_decomposition/analyze_spectral_structure.py \\
        --obs-dir /tmp/schedule_calibration/observations \\
        --output-dir /tmp/spectral_analysis

    # Fewer observations for a quick check
    uv run python .../analyze_spectral_structure.py \\
        --obs-dir /tmp/schedule_calibration/observations \\
        --max-obs 5 --n-seeds 3

    # Save plot
    uv run python .../analyze_spectral_structure.py \\
        --obs-dir /tmp/schedule_calibration/observations \\
        --output-dir /tmp/spectral_analysis --plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# DCT (same orthonormal matrix approach as strategy.py)
# ---------------------------------------------------------------------------

def build_dct_matrix(N: int, device=None) -> torch.Tensor:
    """Build the N x N orthonormal DCT-II matrix."""
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = torch.arange(N, device=device, dtype=torch.float32)
    M = torch.cos(math.pi * k[:, None] * (2.0 * n[None, :] + 1.0) / (2.0 * N))
    M[0] *= 1.0 / math.sqrt(N)
    M[1:] *= math.sqrt(2.0 / N)
    return M


# ---------------------------------------------------------------------------
# Spectral probe
# ---------------------------------------------------------------------------

class SpectralProbe:
    """guided_fn that captures DCT energy at each step without modifying velocity."""

    def __init__(self, dct_matrix: torch.Tensor):
        self.M = dct_matrix
        self.step_spectra: list[torch.Tensor] = []

    def reset(self):
        self.step_spectra = []

    def __call__(
        self, actions_before: torch.Tensor, step_idx: int, velocity: torch.Tensor
    ) -> torch.Tensor:
        v = velocity.float()  # (B, H, D)

        # Forward DCT along horizon (dim=1): M @ v
        V = self.M @ v  # (H, H) @ (B, H, D) -> (B, H, D)

        # Energy per frequency bin, averaged over batch and action dims
        energy = (V ** 2).mean(dim=(0, 2))  # (H,)
        self.step_spectra.append(energy.cpu())

        # Passthrough -- do not modify velocity
        return velocity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze spectral structure of GR00T velocity fields",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-dir", required=True,
                   help="Directory with .npz observations")
    p.add_argument("--output-dir", default=None,
                   help="Where to save results (JSON + optional plot)")
    p.add_argument("--model-path", default="nvidia/GR00T-N1.6-3B")
    p.add_argument("--embodiment-tag", default="robocasa_panda_omron")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-obs", type=int, default=None,
                   help="Limit number of observations (for quick runs)")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Random seeds per observation")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--plot", action="store_true",
                   help="Generate matplotlib plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    obs_dir = Path(args.obs_dir)

    npz_files = sorted(obs_dir.glob("*.npz"))
    if not npz_files:
        print("ERROR: no .npz files in " + str(obs_dir))
        sys.exit(1)

    if args.max_obs is not None:
        npz_files = npz_files[:args.max_obs]

    print("Observations: %d files from %s" % (len(npz_files), obs_dir))
    print("Seeds per observation: %d" % args.n_seeds)
    print("Total denoising runs: %d" % (len(npz_files) * args.n_seeds))

    # Add project root to path for DenoisingLab import
    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(project_root))
    from scripts.denoising_lab.denoising_lab import DenoisingLab

    # Load model
    print("\nLoading model from %s ..." % args.model_path)
    lab = DenoisingLab(args.model_path, args.embodiment_tag, args.device)

    H = lab.action_horizon  # 50
    D = lab.action_dim       # 128
    print("Action space: horizon=%d, dim=%d" % (H, D))

    # Build DCT matrix
    dct_matrix = build_dct_matrix(H, device=lab.device)
    probe = SpectralProbe(dct_matrix)

    # Load and encode observations
    print("\nEncoding %d observations through backbone ..." % len(npz_files))
    features_list = []
    t0 = time.monotonic()
    for i, path in enumerate(npz_files):
        obs = DenoisingLab.load_observation(path)
        features_list.append(lab.encode_features_from_sim_obs(obs))
        if (i + 1) % 5 == 0 or i == len(npz_files) - 1:
            print("  encoded %d/%d" % (i + 1, len(npz_files)))
    encode_time = time.monotonic() - t0
    print("  Encoding took %.1fs" % encode_time)

    # Collect spectral energy at each step
    num_steps = 4
    all_spectra: dict[int, list[torch.Tensor]] = {s: [] for s in range(num_steps)}

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    n_runs = len(features_list) * len(seeds)

    print("\nRunning %d denoising passes to capture velocity spectra ..." % n_runs)
    t0 = time.monotonic()
    run_idx = 0
    for feat in features_list:
        for seed in seeds:
            probe.reset()
            lab.denoise(feat, num_steps=num_steps, guided_fn=probe, seed=seed)

            for step_idx, spectrum in enumerate(probe.step_spectra):
                all_spectra[step_idx].append(spectrum)

            run_idx += 1
            if run_idx % 10 == 0 or run_idx == n_runs:
                elapsed = time.monotonic() - t0
                eta = elapsed / run_idx * (n_runs - run_idx)
                print("  %d/%d runs  (%.1fs elapsed, ~%.0fs remaining)"
                      % (run_idx, n_runs, elapsed, eta))

    denoise_time = time.monotonic() - t0
    print("  Denoising took %.1fs (%.2fs per run)" % (denoise_time, denoise_time / n_runs))

    # Compute statistics
    avg_spectra: dict[int, np.ndarray] = {}
    std_spectra: dict[int, np.ndarray] = {}
    for step_idx in range(num_steps):
        stacked = torch.stack(all_spectra[step_idx])  # (n_runs, H)
        avg_spectra[step_idx] = stacked.mean(dim=0).numpy()
        std_spectra[step_idx] = stacked.std(dim=0).numpy()

    # Derived metrics
    sep = "=" * 70
    print("\n" + sep)
    print("SPECTRAL STRUCTURE ANALYSIS")
    print(sep)
    print("  Observations: %d, Seeds: %d, Runs: %d"
          % (len(npz_files), args.n_seeds, n_runs))
    print("  Action horizon: %d, Action dim: %d" % (H, D))
    print()

    tau_labels = ["0.00 (noise->structure)", "0.25 (mid-low)",
                  "0.50 (mid-high)", "0.75 (fine detail)"]

    for step_idx in range(num_steps):
        spectrum = avg_spectra[step_idx]
        total_energy = spectrum.sum()
        normalized = spectrum / (total_energy + 1e-12)

        # Cumulative energy -- what fraction in first K bins?
        cumulative = np.cumsum(normalized)
        energy_50pct = int(np.searchsorted(cumulative, 0.5) + 1)
        energy_90pct = int(np.searchsorted(cumulative, 0.9) + 1)

        peak_freq = int(np.argmax(spectrum))
        centroid = float(np.sum(np.arange(H) * normalized))

        # Low (k=0..H//4), mid (H//4..3H//4), high (3H//4..H) energy fractions
        q1, q3 = H // 4, 3 * H // 4
        low_frac = float(normalized[:q1].sum())
        mid_frac = float(normalized[q1:q3].sum())
        high_frac = float(normalized[q3:].sum())

        print("  Step %d (tau=%s):" % (step_idx, tau_labels[step_idx]))
        print("    Total energy:        %.4f" % total_energy)
        print("    Peak frequency bin:  k=%d" % peak_freq)
        print("    Spectral centroid:   k=%.1f" % centroid)
        print("    50%% energy in first: %d bins" % energy_50pct)
        print("    90%% energy in first: %d bins" % energy_90pct)
        print("    Energy distribution: low=%.1f%%  mid=%.1f%%  high=%.1f%%"
              % (low_frac * 100, mid_frac * 100, high_frac * 100))
        print()

    # Interpretation
    centroids = [float(np.sum(np.arange(H) * avg_spectra[s] / (avg_spectra[s].sum() + 1e-12)))
                 for s in range(num_steps)]
    monotonic = all(centroids[i] <= centroids[i + 1] for i in range(num_steps - 1))

    centroid_strs = ["%.1f" % c for c in centroids]
    print("  Spectral centroids: [%s]" % ", ".join(centroid_strs))
    if monotonic:
        print("  --> Centroids increase monotonically across steps.")
        print("      The low-pass -> high-pass assumption is SUPPORTED.")
        print("      The current Gaussian affinity profiles are well-motivated.")
    else:
        print("  --> Centroids do NOT increase monotonically.")
        print("      The assumed frequency progression needs recalibration.")
        print("      Consider adjusting sigma values or Gaussian centers in")
        print("      build_frequency_weights() based on the actual peak/centroid")
        print("      locations reported above.")

    # Save results
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        result = {
            "config": {
                "obs_dir": str(obs_dir),
                "n_observations": len(npz_files),
                "n_seeds": args.n_seeds,
                "base_seed": args.base_seed,
                "n_runs": n_runs,
                "model_path": args.model_path,
                "embodiment_tag": args.embodiment_tag,
                "action_horizon": H,
                "action_dim": D,
            },
            "per_step": {},
        }

        for step_idx in range(num_steps):
            spectrum = avg_spectra[step_idx]
            total = float(spectrum.sum())
            normalized = spectrum / (total + 1e-12)
            cumulative = np.cumsum(normalized)

            result["per_step"][str(step_idx)] = {
                "tau": step_idx / num_steps,
                "total_energy": total,
                "peak_freq_bin": int(np.argmax(spectrum)),
                "spectral_centroid": float(np.sum(np.arange(H) * normalized)),
                "energy_50pct_bins": int(np.searchsorted(cumulative, 0.5) + 1),
                "energy_90pct_bins": int(np.searchsorted(cumulative, 0.9) + 1),
                "avg_spectrum": avg_spectra[step_idx].tolist(),
                "std_spectrum": std_spectra[step_idx].tolist(),
            }

        result["interpretation"] = {
            "centroids": centroids,
            "monotonic_progression": monotonic,
        }

        result_path = out / "spectral_analysis.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print("\n  Results saved to %s" % result_path)

    # Optional plot
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Velocity Field Spectral Structure per Denoising Step",
                     fontsize=14)
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

        for step_idx, ax in enumerate(axes.flat):
            spectrum = avg_spectra[step_idx]
            std = std_spectra[step_idx]
            total = spectrum.sum() + 1e-12
            normalized = spectrum / total
            std_norm = std / total

            freq_bins = np.arange(H)
            ax.bar(freq_bins, normalized, color=colors[step_idx], alpha=0.7,
                   label="Normalized energy")
            ax.fill_between(freq_bins,
                            np.maximum(normalized - std_norm, 0),
                            normalized + std_norm,
                            color=colors[step_idx], alpha=0.2)

            centroid = float(np.sum(freq_bins * normalized))
            ax.axvline(centroid, color="black", linestyle="--", linewidth=1.5,
                       label="Centroid k=%.1f" % centroid)

            ax.set_title("Step %d (tau=%.2f)" % (step_idx, step_idx / num_steps))
            ax.set_xlabel("DCT frequency bin k")
            ax.set_ylabel("Normalized energy")
            ax.legend(fontsize=8)
            ax.set_xlim(-0.5, H - 0.5)

        plt.tight_layout()

        if args.output_dir:
            plot_path = Path(args.output_dir) / "spectral_structure.png"
        else:
            plot_path = Path("/tmp/spectral_structure.png")
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        print("  Plot saved to %s" % plot_path)
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
