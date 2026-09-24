"""Step 1 of the velocity-anchor experiment: calibrate `vel_anchor_coef`.

Runs ONE update-only trial of train_grpo.py per coefficient on a cached
iteration (no collection), measures how much each coefficient SHRINKS and TURNS
the first LoRA update relative to coefficient 0, and suggests sweep values by
log-interpolating shrink(c) at the target shrinks.

    .venv/bin/python scripts/grpo/calibrate_vel_anchor.py \\
        --coefs 0 0.1 1 10 100 --targets 0.05 0.15 0.35 \\
        --out-dir grpo_data/vel_anchor_calib [--dry-run] \\
        -- <train_grpo.py args for A's config, passed through verbatim>

Each trial appends, AFTER the passthrough (tyro is last-wins, so these override
any copy of the same flag in it):
    --vel-anchor-coef <c> --stop-after-iterations 1
    --resume-from-collected-data --checkpoint-dir <out-dir>/coef_<c>
`--resume-from` / `--vel-anchor-path` in the passthrough are kept (the
checkpoint-anchor variant). Trials run sequentially; a trial's stdout/stderr
goes to <out-dir>/coef_<c>.log. Trial dirs must not exist yet (use a fresh
--out-dir); --analyze-only re-reads finished ones. See scripts/grpo/README.md
"Velocity anchor".
"""

import argparse
import glob
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_grpo.py"
# Same canonical pattern as train_grpo.ITER_DIR_RE (not imported: that module
# pulls in the model stack).
ITER_RE = re.compile(r"iter_([0-9]+)")
# Flags this script sets itself; a copy in the passthrough is overridden.
MANAGED_FLAGS = (
    "--vel-anchor-coef",
    "--stop-after-iterations",
    "--resume-from-collected-data",
    "--no-resume-from-collected-data",
    "--checkpoint-dir",
)
GUARD_TAGS = ("episode/success_rate", "episode/num_chunks", "episode/num_train_chunks")
METRIC_TAGS = ("vel_anchor/train_last_epoch_mean", "vel_anchor/grad_ratio")
DEFAULT_LORA_RANK, DEFAULT_LORA_ALPHA = 16, 32


# ─── CLI and command construction ────────────────────────────────────────────

def coef_tag(c: float) -> str:
    """Trial directory name for coefficient c (0 -> coef_0, 0.1 -> coef_0.1)."""
    return f"coef_{float(c):g}"


def split_argv(argv: list) -> tuple:
    """(own args, passthrough) split at the first bare `--`."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def parse_args(argv: list) -> argparse.Namespace:
    own, passthrough = split_argv(list(argv))
    ap = argparse.ArgumentParser(
        description="Calibrate vel_anchor_coef from one-iteration trials.",
        epilog="Everything after `--` is passed to train_grpo.py verbatim.",
    )
    ap.add_argument("--coefs", type=float, nargs="+", required=True,
                    help="Coefficients to trial; must include 0 (the baseline).")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.05, 0.15, 0.35],
                    help="Target first-update shrinks, each in (0, 1).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the trial commands and exit.")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip the trials; analyse existing trial outputs.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter for train_grpo.py (default: this one).")
    args = ap.parse_args(own)
    coefs = [float(c) for c in args.coefs]
    if any((not math.isfinite(c)) or c < 0.0 for c in coefs):
        ap.error(f"--coefs must be finite and >= 0, got {coefs}")
    if len(set(coefs)) != len(coefs):
        ap.error(f"--coefs has duplicates: {coefs}")
    if 0.0 not in coefs:
        ap.error("--coefs must include 0: shrink and cos are measured against it")
    tags = [coef_tag(c) for c in coefs]
    if len(set(tags)) != len(tags):
        ap.error(f"--coefs map to colliding trial dirs {tags}; use fewer digits")
    if any(not (0.0 < t < 1.0) for t in args.targets):
        ap.error(f"--targets must lie in (0, 1), got {args.targets}")
    args.coefs = sorted(coefs)
    args.targets = sorted(float(t) for t in args.targets)
    args.passthrough = passthrough
    return args


def _flag_name(tok: str) -> str:
    """`--resume_from=x` -> `--resume-from`: tyro accepts both spellings."""
    return tok.split("=", 1)[0].replace("_", "-")


def passthrough_value(passthrough: list, flag: str):
    """Last value of `flag X` / `flag=X` in the passthrough (tyro is last-wins).

    Matches either spelling (`--resume-from` / `--resume_from`), as tyro does.
    """
    val = None
    for i, tok in enumerate(passthrough):
        if not tok.startswith("--") or _flag_name(tok) != flag:
            continue
        if "=" in tok:
            val = tok.split("=", 1)[1]
        elif i + 1 < len(passthrough):
            val = passthrough[i + 1]
    return val


def trial_command(python: str, passthrough: list, coef: float, out_dir) -> list:
    """train_grpo.py argv for one trial: passthrough verbatim, then our flags."""
    return [
        python, str(TRAIN_SCRIPT), *passthrough,
        "--vel-anchor-coef", repr(float(coef)),
        "--stop-after-iterations", "1",
        "--resume-from-collected-data",
        "--checkpoint-dir", str(Path(out_dir) / coef_tag(coef)),
    ]


def start_iteration(passthrough: list) -> int:
    """The iteration a trial trains, as train_grpo._parse_resume_iteration."""
    resume = passthrough_value(passthrough, "--resume-from")
    if resume in (None, "None"):
        return 1
    m = ITER_RE.fullmatch(Path(resume.rstrip("/")).name)
    if not m:
        raise SystemExit(
            f"--resume-from {resume!r} is not an iter_NNNN/ dir; "
            f"--resume-from-collected-data (always set here) requires one."
        )
    return int(m.group(1)) + 1


def start_weights(passthrough: list):
    """lora_weights.pt the trial starts from, or None (fresh: delta W = 0)."""
    resume = passthrough_value(passthrough, "--resume-from")
    if resume in (None, "None"):
        return None
    return Path(resume) / "lora_weights.pt"


# ─── Effective LoRA update, via the r x r trace trick ────────────────────────

def load_lora_factors(pt_path) -> dict:
    """{module: (B [out, r], A [r, in])} in float64, from a lora_weights.pt."""
    import torch
    sd = torch.load(pt_path, map_location="cpu")
    mods: dict = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            mods.setdefault(k.split(".lora_A.")[0], [None, None])[1] = v.double()
        elif ".lora_B." in k:
            mods.setdefault(k.split(".lora_B.")[0], [None, None])[0] = v.double()
    bad = [m for m, (B, A) in mods.items() if B is None or A is None]
    if not mods or bad:
        raise SystemExit(f"{pt_path}: not a LoRA checkpoint (unpaired: {bad[:3]})")
    return {m: (B, A) for m, (B, A) in mods.items()}


def update_factors(trial: dict, start) -> dict:
    """Factors of U = delta W(trial) - delta W(start), exactly, at rank 2r.

    B_t A_t - B_s A_s == [B_t, B_s] @ [A_t; -A_s], so no dense product and no
    cancellation between two large inner products.
    """
    import torch
    if start is None:
        return trial
    if set(start) != set(trial):
        raise SystemExit("start and trial checkpoints hold different LoRA modules")
    return {
        m: (torch.cat([trial[m][0], start[m][0]], dim=1),
            torch.cat([trial[m][1], -start[m][1]], dim=0))
        for m in trial
    }


def lora_inner(fa: dict, fb: dict, scale: float) -> float:
    """<scale B_a A_a, scale B_b A_b>_F summed over modules; only r x r ops."""
    import torch
    total = 0.0
    for m, (Ba, Aa) in fa.items():
        Bb, Ab = fb[m]
        total += float(torch.trace((Ba.T @ Bb) @ (Ab @ Aa.T)))
    return scale * scale * total


# ─── TensorBoard ─────────────────────────────────────────────────────────────

def read_tb(tb_dir) -> tuple:
    """({tag: {step: value}} for the tags we use, config text or '')."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    scalars: dict = {}
    config_text = ""
    for f in sorted(glob.glob(str(Path(tb_dir) / "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0, "tensors": 0})
        ea.Reload()
        tags = ea.Tags()
        for tag in set(GUARD_TAGS + METRIC_TAGS) & set(tags.get("scalars", [])):
            for e in ea.Scalars(tag):
                scalars.setdefault(tag, {})[int(e.step)] = float(e.value)
        if "config/text_summary" in tags.get("tensors", []):
            for e in ea.Tensors("config/text_summary"):
                if e.tensor_proto.string_val:
                    config_text = e.tensor_proto.string_val[0].decode("utf-8", "replace")
    return scalars, config_text


def lora_scale(config_text: str, passthrough: list) -> float:
    """alpha / r from the trial's TB config dump, else passthrough, else defaults."""
    def _from_text(name):
        m = re.search(r"\|\s*" + name + r"\s*\|\s*([0-9.eE+-]+)\s*\|", config_text or "")
        return float(m.group(1)) if m else None
    r = _from_text("lora_rank") or passthrough_value(passthrough, "--lora-rank")
    a = _from_text("lora_alpha") or passthrough_value(passthrough, "--lora-alpha")
    r = float(r) if r is not None else DEFAULT_LORA_RANK
    a = float(a) if a is not None else DEFAULT_LORA_ALPHA
    return a / r


def check_same_episodes(guard: dict) -> None:
    """Hard-fail unless every trial trained on the same cached episodes.

    guard: {coef: {tag: value at the start step}}. A missing SR / chunk count
    or any disagreement means the trials are not comparable.
    """
    coefs = sorted(guard)
    for tag in ("episode/success_rate", "episode/num_chunks"):
        vals = {c: guard[c].get(tag) for c in coefs}
        if any(v is None for v in vals.values()):
            raise SystemExit(f"cached-episode guard: {tag} missing for "
                             f"{[c for c, v in vals.items() if v is None]}")
        if len(set(vals.values())) != 1:
            raise SystemExit(f"cached-episode guard FAILED: trials did not load "
                             f"identical episodes ({tag} = {vals})")
    opt = {c: guard[c].get("episode/num_train_chunks") for c in coefs}
    if all(v is not None for v in opt.values()) and len(set(opt.values())) != 1:
        raise SystemExit(f"cached-episode guard FAILED: episode/num_train_chunks "
                         f"differs across trials ({opt})")


# ─── Suggestions ─────────────────────────────────────────────────────────────

def suggest_coefs(coefs: list, shrinks: list, targets: list) -> tuple:
    """Log-interpolate shrink(c) at each target over the measured c > 0.

    Returns (suggestions, warnings). A suggestion is {"target", "coef"} with
    coef a float, or {"target", "bracket": (lo, hi)} when the target is outside
    the measured range (hi = inf above it). Never extrapolates.
    """
    warns = []
    pts = sorted((c, s) for c, s in zip(coefs, shrinks) if c > 0.0)
    seq = [0.0] + [s for _, s in pts]
    if any(b < a for a, b in zip(seq, seq[1:])):
        warns.append(f"shrink is not monotone in the coefficient "
                     f"({[(c, round(s, 4)) for c, s in pts]}); using the first "
                     f"crossing of each target")
    out = []
    for t in targets:
        hit = next((i for i, (_, s) in enumerate(pts) if s >= t), None)
        if hit is None:
            out.append({"target": t,
                        "bracket": (pts[-1][0] if pts else 0.0, math.inf)})
        elif hit == 0:
            out.append({"target": t, "bracket": (0.0, pts[0][0])})
        else:
            (c0, s0), (c1, s1) = pts[hit - 1], pts[hit]
            frac = (t - s0) / (s1 - s0) if s1 != s0 else 0.0
            out.append({"target": t, "coef": math.exp(
                math.log(c0) + frac * (math.log(c1) - math.log(c0)))})
    return out, warns


def _labels(n: int) -> list:
    return ["c_lo", "c_mid", "c_hi"] if n == 3 else [f"c_{i + 1}" for i in range(n)]


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze(coefs: list, targets: list, out_dir, passthrough: list) -> dict:
    """Measure every trial, run the guards, and build the summary dict."""
    out_dir = Path(out_dir)
    step = start_iteration(passthrough)
    s_pt = start_weights(passthrough)
    s_fac = load_lora_factors(s_pt) if s_pt is not None else None
    trials, guard, updates = [], {}, {}
    scale = None
    for c in coefs:
        tdir = out_dir / coef_tag(c)
        pt = tdir / f"iter_{step:04d}" / "lora_weights.pt"
        if not pt.is_file():
            raise SystemExit(
                f"coef={c:g}: no {pt} (did the trial fail, or skip its update "
                f"and save nothing?). See {out_dir / (coef_tag(c) + '.log')}."
            )
        scalars, cfg_text = read_tb(tdir / "tb_logs")
        if scale is None or c == 0.0:
            scale = lora_scale(cfg_text, passthrough)
        guard[c] = {tag: scalars.get(tag, {}).get(step) for tag in GUARD_TAGS}
        updates[c] = update_factors(load_lora_factors(pt), s_fac)
        trials.append({
            "coef": c, "dir": str(tdir), "step": step,
            "last_epoch_D": scalars.get(METRIC_TAGS[0], {}).get(step),
            "grad_ratio": scalars.get(METRIC_TAGS[1], {}).get(step),
            **{f"start_{k.split('/')[-1]}": v for k, v in guard[c].items()},
        })
    check_same_episodes(guard)
    ref = updates[0.0]
    ref_sq = lora_inner(ref, ref, scale)
    if not ref_sq > 0.0:
        raise SystemExit("coef=0 trial produced a zero update; nothing to compare against")
    for tr in trials:
        u = updates[tr["coef"]]
        sq = lora_inner(u, u, scale)
        tr["dW_norm"] = math.sqrt(max(sq, 0.0))
        tr["shrink"] = 1.0 - tr["dW_norm"] / math.sqrt(ref_sq)
        tr["cos_vs_0"] = (lora_inner(u, ref, scale) / math.sqrt(sq * ref_sq)
                          if sq > 0.0 else float("nan"))
    sugg, warns = suggest_coefs([t["coef"] for t in trials],
                                [t["shrink"] for t in trials], targets)
    for lab, s in zip(_labels(len(sugg)), sugg):
        s["label"] = lab
    return {"step": step, "lora_scale": scale, "trials": trials,
            "suggestions": sugg, "warnings": warns,
            "start_weights": None if s_pt is None else str(s_pt)}


def format_report(summary: dict) -> str:
    def f(x, spec):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)
    lines = [f"Velocity-anchor calibration (trained step {summary['step']}, "
             f"alpha/r = {summary['lora_scale']:g})",
             f"{'coef':>10} {'||dW||':>10} {'shrink':>8} {'cos_vs_0':>9} "
             f"{'last_ep_D':>11} {'grad_ratio':>11}"]
    for t in summary["trials"]:
        lines.append(f"{t['coef']:>10g} {f(t['dW_norm'], '.5g'):>10} "
                     f"{f(t['shrink'], '.4f'):>8} {f(t['cos_vs_0'], '.4f'):>9} "
                     f"{f(t['last_epoch_D'], '.4g'):>11} {f(t['grad_ratio'], '.4g'):>11}")
    lines.append("Suggested sweep:")
    for s in summary["suggestions"]:
        if "coef" in s:
            lines.append(f"  {s['label']}: shrink {s['target']:.2f} -> "
                         f"vel_anchor_coef ~= {s['coef']:.4g}")
        else:
            lo, hi = s["bracket"]
            lines.append(f"  {s['label']}: shrink {s['target']:.2f} is outside the "
                         f"measured range -> between {lo:g} and {hi:g}; add a trial "
                         f"there rather than extrapolating")
    for w in summary["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def _jsonable(obj):
    """Strict-JSON copy: +-inf -> "inf"/"-inf", NaN -> None, tuples -> lists."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = Path(args.out_dir)
    cmds = [(c, trial_command(args.python, args.passthrough, c, out_dir))
            for c in args.coefs]
    if args.dry_run:
        for _, cmd in cmds:
            print(shlex.join(cmd))
        return 0
    overridden = sorted({_flag_name(t) for t in args.passthrough
                         if t.startswith("--")} & set(MANAGED_FLAGS))
    if overridden:
        print(f"[calib] passthrough flags overridden per trial: {overridden}")
    if not args.analyze_only:
        # Fail on passthrough problems BEFORE ~15-20 min per trial is spent.
        start_iteration(args.passthrough)
        s_pt = start_weights(args.passthrough)
        if s_pt is not None and not s_pt.is_file():
            raise SystemExit(f"[calib] --resume-from has no {s_pt}")
        # A leftover trial dir would be reused silently: with save_interval > 1
        # the trial saves nothing in-loop and the final save skips an existing
        # iter dir, so its OLD weights would be read against NEW TB scalars.
        stale = [str(out_dir / coef_tag(c)) for c in args.coefs
                 if (out_dir / coef_tag(c)).exists()
                 and any((out_dir / coef_tag(c)).iterdir())]
        if stale:
            raise SystemExit(
                f"[calib] trial dir(s) already exist: {stale}. Use a fresh "
                f"--out-dir, delete them, or pass --analyze-only to re-read them."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        for c, cmd in cmds:
            log = out_dir / f"{coef_tag(c)}.log"
            print(f"[calib] coef={c:g}: {shlex.join(cmd)}\n        log: {log}",
                  flush=True)
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                raise SystemExit(f"[calib] trial coef={c:g} exited {rc}; see {log}")
    summary = analyze(args.coefs, args.targets, out_dir, args.passthrough)
    summary["args"] = {"coefs": args.coefs, "targets": args.targets,
                       "passthrough": args.passthrough}
    print(format_report(summary))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calib_summary.json", "w") as fh:
        json.dump(_jsonable(summary), fh, indent=2, allow_nan=False)
    print(f"[calib] wrote {out_dir / 'calib_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
