"""Tests for scripts/grpo/calibrate_vel_anchor.py (PLAN_vel_anchor.md 3.3).

24. the r x r trace-trick norm/cosine of the effective LoRA update against dense
    products, from scratch (B0 = 0) and relative to a start checkpoint;
25. log-interpolation of shrink(c): exact on a log-linear synthetic curve, a
    warning when non-monotone, brackets (never extrapolation) out of range;
26. command construction: --dry-run prints one command per coefficient, the
    passthrough verbatim, then the four added flags — and tyro parses that argv
    into a GRPOConfig with OUR values winning (last-wins);
27. guards on synthetic TB event files: trials that did not load identical
    cached episodes hard-fail; a matching pair analyses end to end.

Run with the project venv (CPU):
    .venv/bin/python scripts/grpo/test_calibrate_vel_anchor.py
"""

import contextlib
import io
import math
import shlex
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_vel_anchor as cal  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


SHAPES = {"blk.0.attn1.to_q": (6, 3, 5), "blk.0.ff.net.2": (4, 3, 7),
          "proj_out_2": (5, 3, 6)}      # module -> (out, r, in)


def _factors(seed, zero_b=False):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m, (o, r, i) in SHAPES.items():
        B = torch.zeros(o, r, dtype=torch.float64) if zero_b else \
            torch.randn(o, r, generator=g, dtype=torch.float64)
        A = torch.randn(r, i, generator=g, dtype=torch.float64)
        out[m] = (B, A)
    return out


def _dense(f, scale):
    return {m: scale * (B @ A) for m, (B, A) in f.items()}


def _state_dict(f):
    sd = {}
    for m, (B, A) in f.items():
        sd[f"{m}.lora_A.default.weight"] = A.float()
        sd[f"{m}.lora_B.default.weight"] = B.float()
    return sd


# ─── 24. Effective delta-W norm and cosine ───────────────────────────────────

def test_24_trace_trick():
    print("\n[24] Trace-trick norm and cosine == dense computation")
    scale = 2.0
    start = _factors(1)
    t1, t2 = _factors(2), _factors(3)
    for label, s in (("from scratch (B0 = 0)", None), ("relative to a start", start)):
        u1, u2 = cal.update_factors(t1, s), cal.update_factors(t2, s)
        d1 = {m: v - (_dense(s, scale)[m] if s else 0) for m, v in _dense(t1, scale).items()}
        d2 = {m: v - (_dense(s, scale)[m] if s else 0) for m, v in _dense(t2, scale).items()}
        n_dense = math.sqrt(sum(float((v * v).sum()) for v in d1.values()))
        dot_dense = sum(float((d1[m] * d2[m]).sum()) for m in d1)
        n2_dense = math.sqrt(sum(float((v * v).sum()) for v in d2.values()))
        n_tt = math.sqrt(cal.lora_inner(u1, u1, scale))
        cos_tt = cal.lora_inner(u1, u2, scale) / math.sqrt(
            cal.lora_inner(u1, u1, scale) * cal.lora_inner(u2, u2, scale))
        check(f"{label}: ||U|| matches dense", math.isclose(n_tt, n_dense, rel_tol=1e-10),
              f"{n_tt} vs {n_dense}")
        check(f"{label}: cos(U1, U2) matches dense",
              math.isclose(cos_tt, dot_dense / (n_dense * n2_dense), rel_tol=1e-10))
        if s is not None:
            rank = next(iter(u1.values()))[0].shape[1]
            check(f"{label}: update factorised at rank 2r (no dense product)",
                  rank == 2 * 3)
    fresh = _factors(4, zero_b=True)
    check("fresh LoRA (B = 0) has a zero effective delta W",
          cal.lora_inner(fresh, fresh, scale) == 0.0)
    with tempfile.TemporaryDirectory() as tmp:
        pt = Path(tmp) / "lora_weights.pt"
        torch.save(_state_dict(t1), pt)
        back = cal.load_lora_factors(pt)
        check("load_lora_factors round-trips PEFT keys",
              set(back) == set(t1) and all(
                  torch.allclose(back[m][0], t1[m][0].float().double())
                  and torch.allclose(back[m][1], t1[m][1].float().double())
                  for m in t1))


# ─── 25. Interpolation ───────────────────────────────────────────────────────

def test_25_interpolation():
    print("\n[25] Suggested coefficients from shrink(c)")
    coefs = [0.0, 0.1, 1.0, 10.0, 100.0]

    def shrink(c):          # exactly linear in log(c) over the measured range
        return 0.0 if c == 0 else 0.2 + 0.05 * math.log(c)

    sh = [shrink(c) for c in coefs]
    targets = [0.12, 0.2, 0.3]
    sugg, warns = cal.suggest_coefs(coefs, sh, targets)
    want = [math.exp((t - 0.2) / 0.05) for t in targets]
    check("hits the targets exactly on a log-linear shrink(c)",
          all("coef" in s and math.isclose(s["coef"], w, rel_tol=1e-9)
              for s, w in zip(sugg, want)),
          f"{[s.get('coef') for s in sugg]} vs {want}")
    check("no warning when monotone", not warns, str(warns))
    lo, hi = sh[1], sh[-1]
    sugg2, _ = cal.suggest_coefs(coefs, sh, [lo / 2, hi + 0.1])
    check("target below the smallest c>0 shrink -> bracket (0, c_min)",
          sugg2[0].get("bracket") == (0.0, 0.1), str(sugg2[0]))
    check("target above the largest shrink -> bracket (c_max, inf)",
          sugg2[1].get("bracket") == (100.0, math.inf), str(sugg2[1]))
    check("brackets carry no extrapolated coefficient",
          all("coef" not in s for s in sugg2))
    nm = [0.0, 0.1, 0.05, 0.3, 0.4]          # dips at c = 1
    s3, w3 = cal.suggest_coefs(coefs, nm, [0.2])
    check("non-monotone shrink warns", any("not monotone" in w for w in w3), str(w3))
    check("... and interpolates at the first crossing (between c=1 and c=10)",
          "coef" in s3[0] and 1.0 < s3[0]["coef"] < 10.0, str(s3))


# ─── 26. Command construction ────────────────────────────────────────────────

PASSTHROUGH = ["--learning-rate", "5e-5", "--jitter-pos", "0.125",
               "--update-epochs", "3", "--num-iterations", "48",
               "--checkpoint-dir", "grpo_data/old_run", "--seed", "67"]


def test_26_command_construction():
    print("\n[26] --dry-run commands: passthrough verbatim + the four flags")
    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cal.main(["--coefs", "0", "0.1", "100", "--out-dir", tmp,
                           "--dry-run", "--", *PASSTHROUGH])
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        check("exit 0 and one command per coefficient", rc == 0 and len(lines) == 3,
              buf.getvalue())
        for coef, line in zip((0.0, 0.1, 100.0), lines):
            argv = shlex.split(line)
            i = argv.index(str(cal.TRAIN_SCRIPT))
            check(f"coef {coef:g}: passthrough follows train_grpo.py verbatim",
                  argv[i + 1:i + 1 + len(PASSTHROUGH)] == PASSTHROUGH)
            tail = argv[i + 1 + len(PASSTHROUGH):]
            check(f"coef {coef:g}: exactly the four added flags, after it",
                  tail == ["--vel-anchor-coef", repr(coef), "--stop-after-iterations",
                           "1", "--resume-from-collected-data", "--checkpoint-dir",
                           str(Path(tmp) / cal.coef_tag(coef))], str(tail))
            try:
                import tyro
                from grpo_config import GRPOConfig
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    cfg = tyro.cli(GRPOConfig, args=argv[i + 1:])
                ok = (cfg.vel_anchor_coef == coef and cfg.stop_after_iterations == 1
                      and cfg.resume_from_collected_data is True
                      and cfg.checkpoint_dir == str(Path(tmp) / cal.coef_tag(coef))
                      and cfg.learning_rate == 5e-5 and cfg.jitter_pos == 0.125)
                check(f"coef {coef:g}: tyro parses it and OUR flags win", ok)
            except ImportError:
                print("  (tyro unavailable: skipped the parse check)")
    check("coef_tag names", [cal.coef_tag(c) for c in (0, 0.1, 100, 1e-5)]
          == ["coef_0", "coef_0.1", "coef_100", "coef_1e-05"])
    check("start iteration: fresh -> 1, resume iter_0003 -> 4",
          cal.start_iteration(PASSTHROUGH) == 1
          and cal.start_iteration(PASSTHROUGH + ["--resume-from", "x/iter_0003/"]) == 4
          and cal.start_iteration(PASSTHROUGH + ["--resume-from=x/iter_0010"]) == 11)
    for bad in (["--coefs", "0.1", "1"], ["--coefs", "0", "-1"],
                ["--coefs", "0", "1", "--targets", "1.5"]):
        raised = False
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cal.parse_args(bad + ["--out-dir", "x"])
        except SystemExit:
            raised = True
        check(f"rejected: {bad}", raised)


# ─── 27. Guards on synthetic TB event files ──────────────────────────────────

def _write_trial(out_dir, coef, factors, *, step=1, sr=0.625, chunks=2093,
                 last_d=0.004, ratio=0.3):
    from torch.utils.tensorboard import SummaryWriter
    tdir = Path(out_dir) / cal.coef_tag(coef)
    ck = tdir / f"iter_{step:04d}"
    ck.mkdir(parents=True)
    torch.save(_state_dict(factors), ck / "lora_weights.pt")
    w = SummaryWriter(str(tdir / "tb_logs"))
    w.add_scalar("episode/success_rate", sr, step)
    w.add_scalar("episode/num_chunks", chunks, step)
    if coef > 0:
        w.add_scalar("vel_anchor/train_last_epoch_mean", last_d, step)
        w.add_scalar("vel_anchor/grad_ratio", ratio, step)
    w.add_text("config", "| lora_rank | 3 |\n| lora_alpha | 6 |", 0)
    w.close()


def test_27_guards():
    print("\n[27] Cached-episode guard and end-to-end analysis on synthetic trials")
    base = _factors(10)
    small = {m: (0.5 * B, A) for m, (B, A) in base.items()}     # half the update
    with tempfile.TemporaryDirectory() as tmp:
        _write_trial(tmp, 0.0, base)
        _write_trial(tmp, 1.0, small)
        with contextlib.redirect_stdout(io.StringIO()):
            s = cal.analyze([0.0, 1.0], [0.2, 0.5, 0.7], tmp, [])
        tr = {t["coef"]: t for t in s["trials"]}
        check("matching trials pass the guard and analyse", len(tr) == 2)
        check("alpha/r read from the TB config dump", s["lora_scale"] == 2.0)
        check("shrink == 1 - ||U(c)|| / ||U(0)|| (half update -> 0.5)",
              math.isclose(tr[1.0]["shrink"], 0.5, rel_tol=1e-9), str(tr[1.0]["shrink"]))
        check("cos vs coef 0 == 1 for a rescaled update",
              math.isclose(tr[1.0]["cos_vs_0"], 1.0, rel_tol=1e-9))
        check("TB metrics read at the trained step",
              math.isclose(tr[1.0]["last_epoch_D"], 0.004, rel_tol=1e-6)
              and math.isclose(tr[1.0]["grad_ratio"], 0.3, rel_tol=1e-6))
        check("targets above the only measured shrink are bracketed",
              s["suggestions"][2].get("bracket") == (1.0, math.inf))
        report = cal.format_report(s)
        check("report prints the table and the labelled suggestions",
              "shrink" in report and "c_lo" in report and "c_hi" in report)
    for label, kw in (("success rate", dict(sr=0.5)), ("chunk count", dict(chunks=2000))):
        with tempfile.TemporaryDirectory() as tmp:
            _write_trial(tmp, 0.0, base)
            _write_trial(tmp, 1.0, small, **kw)
            msg = ""
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.analyze([0.0, 1.0], [0.2], tmp, [])
            except SystemExit as exc:
                msg = str(exc)
            check(f"mismatched start-step {label} hard-fails", "guard FAILED" in msg, msg)
    with tempfile.TemporaryDirectory() as tmp:
        _write_trial(tmp, 0.0, base)
        msg = ""
        try:
            cal.analyze([0.0, 1.0], [0.2], tmp, [])
        except SystemExit as exc:
            msg = str(exc)
        check("a trial with no checkpoint fails with a pointer to its log",
              "coef_1.log" in msg, msg)


# ─── Audit regressions ───────────────────────────────────────────────────────

def _write_trial_at(out_dir, coef, factors, *, step, rank=4, alpha=12, **kw):
    from torch.utils.tensorboard import SummaryWriter
    tdir = Path(out_dir) / cal.coef_tag(coef)
    (tdir / f"iter_{step:04d}").mkdir(parents=True)
    torch.save(_state_dict(factors), tdir / f"iter_{step:04d}" / "lora_weights.pt")
    w = SummaryWriter(str(tdir / "tb_logs"))
    w.add_scalar("episode/success_rate", kw.get("sr", 0.625), step)
    w.add_scalar("episode/num_chunks", kw.get("chunks", 2093), step)
    w.add_text("config", f"| lora_rank | {rank} |\n| lora_alpha | {alpha} |", 0)
    w.close()


def test_c1_resume_scale_and_rotation():
    print("\n[C1] analyze(): resume start weights, alpha/r from TB, rotated update")
    start, t0, t1 = _factors(20), _factors(21), _factors(22)
    scale = 12 / 4
    with tempfile.TemporaryDirectory() as tmp:
        sdir = Path(tmp) / "run" / "iter_0003"
        sdir.mkdir(parents=True)
        torch.save(_state_dict(start), sdir / "lora_weights.pt")
        out = Path(tmp) / "calib"
        _write_trial_at(out, 0.0, t0, step=4)
        _write_trial_at(out, 1.0, t1, step=4)
        s = cal.analyze([0.0, 1.0], [0.2], out, ["--resume_from", str(sdir)])
    check("resume via the underscore spelling: trained step 4", s["step"] == 4)
    check("alpha/r read from the TB config (12 / 4 = 3, not the 2.0 default)",
          s["lora_scale"] == 3.0, str(s["lora_scale"]))

    def dense_update(f):   # float32 like the saved checkpoint
        f32 = {m: (B.float().double(), A.float().double()) for m, (B, A) in f.items()}
        s32 = {m: (B.float().double(), A.float().double()) for m, (B, A) in start.items()}
        return {m: scale * (f32[m][0] @ f32[m][1] - s32[m][0] @ s32[m][1]) for m in f}

    u0, u1 = dense_update(t0), dense_update(t1)
    n0 = math.sqrt(sum(float((v * v).sum()) for v in u0.values()))
    n1 = math.sqrt(sum(float((v * v).sum()) for v in u1.values()))
    cos = sum(float((u0[m] * u1[m]).sum()) for m in u0) / (n0 * n1)
    tr = {t["coef"]: t for t in s["trials"]}
    check("dW_norm relative to the START checkpoint, at alpha/r = 3",
          math.isclose(tr[0.0]["dW_norm"], n0, rel_tol=1e-9), f"{tr[0.0]['dW_norm']} vs {n0}")
    check("cos_vs_0 of a genuinely rotated update matches dense",
          math.isclose(tr[1.0]["cos_vs_0"], cos, rel_tol=1e-9) and abs(cos) < 0.99,
          f"{tr[1.0]['cos_vs_0']} vs {cos}")


FAKE_TRAINER = r'''#!{python}
"""Stand-in for train_grpo.py: writes a trial's checkpoint + TB and exits."""
import os, sys
from pathlib import Path
args = sys.argv[2:]                     # argv[1] is the train_grpo.py path
def last(flag):
    v = None
    for i, t in enumerate(args):
        if t == flag and i + 1 < len(args):
            v = args[i + 1]
    return v
ck, coef = Path(last("--checkpoint-dir")), float(last("--vel-anchor-coef"))
with open(os.environ["FAKE_MARKER"], "a") as fh:
    fh.write(f"{coef}\n")
if os.environ.get("FAKE_FAIL_COEF") and float(os.environ["FAKE_FAIL_COEF"]) == coef:
    sys.exit(3)
import torch
from torch.utils.tensorboard import SummaryWriter
g = torch.Generator().manual_seed(5)
sd = {}
for m, (o, r, i) in {"blk.q": (6, 3, 5), "blk.v": (4, 3, 7)}.items():
    B, A = torch.randn(o, r, generator=g), torch.randn(r, i, generator=g)
    sd[f"{m}.lora_A.default.weight"] = A
    sd[f"{m}.lora_B.default.weight"] = B / (1.0 + coef)      # shrink = c / (1 + c)
(ck / "iter_0001").mkdir(parents=True, exist_ok=True)
torch.save(sd, ck / "iter_0001" / "lora_weights.pt")
w = SummaryWriter(str(ck / "tb_logs"))
w.add_scalar("episode/success_rate", 0.625, 1)
w.add_scalar("episode/num_chunks", 2093, 1)
w.add_text("config", "| lora_rank | 3 |\n| lora_alpha | 6 |", 0)
w.close()
'''


def test_c2_main_run_path():
    print("\n[C2] main(): real run path with a fake trainer executable")
    import json
    import os
    import stat
    with tempfile.TemporaryDirectory() as tmp:
        fake = Path(tmp) / "fake_trainer.py"
        fake.write_text(FAKE_TRAINER.replace("{python}", sys.executable))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        marker = Path(tmp) / "ran.txt"
        env_before = dict(os.environ)
        os.environ["FAKE_MARKER"] = str(marker)
        try:
            out = Path(tmp) / "calib"
            argv = ["--coefs", "0", "1", "3", "--targets", "0.6", "--out-dir", str(out),
                    "--python", str(fake), "--", "--learning-rate", "5e-5"]
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cal.main(argv)
            summ = json.loads((out / "calib_summary.json").read_text())
            shr = [t["shrink"] for t in summ["trials"]]
            check("fresh out-dir: every trial ran, strict JSON written",
                  rc == 0 and marker.read_text().split() == ["0.0", "1.0", "3.0"])
            check("shrink(c) = c / (1 + c) measured from the trial checkpoints",
                  all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
                      for a, b in zip(shr, [0.0, 0.5, 0.75])), str(shr))
            check("suggestion log-interpolated between c=1 and c=3",
                  math.isclose(summ["suggestions"][0]["coef"], 3 ** 0.4, rel_tol=1e-6))
            n_ran = len(marker.read_text().split())
            msg = ""
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.main(argv)
            except SystemExit as exc:
                msg = str(exc)
            check("re-run into the same out-dir is refused before any trial",
                  "already exist" in msg and len(marker.read_text().split()) == n_ran,
                  msg)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cal.main(argv[:-3] + ["--analyze-only", "--", "--learning-rate", "5e-5"])
            check("--analyze-only re-reads without running a trial",
                  rc == 0 and len(marker.read_text().split()) == n_ran)
            os.environ["FAKE_FAIL_COEF"] = "1.0"
            out2 = Path(tmp) / "calib2"
            msg = ""
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.main(["--coefs", "0", "1", "--out-dir", str(out2), "--python",
                              str(fake), "--"])
            except SystemExit as exc:
                msg = str(exc)
            check("a failing trial stops the run and points at its log",
                  "exited 3" in msg and "coef_1.log" in msg, msg)
            del os.environ["FAKE_FAIL_COEF"]
            n_ran = len(marker.read_text().split())
            msg = ""
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    cal.main(["--coefs", "0", "1", "--out-dir", str(Path(tmp) / "c3"),
                              "--python", str(fake), "--", "--resume-from",
                              str(Path(tmp) / "nowhere" / "iter_0003")])
            except SystemExit as exc:
                msg = str(exc)
            check("missing --resume-from weights fail BEFORE any trial runs",
                  "lora_weights.pt" in msg and len(marker.read_text().split()) == n_ran,
                  msg)
        finally:
            os.environ.clear()
            os.environ.update(env_before)


def test_c3_guards_and_cli_edges():
    print("\n[C3] num_train_chunks guard, underscore flags, tag collisions")
    base = {"episode/success_rate": 0.6, "episode/num_chunks": 10}
    msg = ""
    try:
        cal.check_same_episodes({0.0: {**base, "episode/num_train_chunks": 7},
                                 1.0: {**base, "episode/num_train_chunks": 8}})
    except SystemExit as exc:
        msg = str(exc)
    check("differing num_train_chunks hard-fails", "num_train_chunks" in msg, msg)
    err = ""
    try:
        cal.check_same_episodes({0.0: {**base, "episode/num_train_chunks": None},
                                 1.0: {**base, "episode/num_train_chunks": 8}})
    except SystemExit as exc:
        err = str(exc)
    check("num_train_chunks absent in a trial is not a failure (optional tag)", err == "")
    check("--resume_from spelling recognised (start 8)",
          cal.start_iteration(["--resume_from", "x/iter_0007"]) == 8
          and cal.start_iteration(["--resume_from=x/iter_0007/"]) == 8)
    raised = False
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            cal.parse_args(["--coefs", "0", "0.1234567", "0.1234568", "--out-dir", "x"])
    except SystemExit:
        raised = True
    check("coefficients colliding in coef_<c> dir names are rejected", raised)


TESTS = [test_24_trace_trick, test_25_interpolation, test_26_command_construction,
         test_27_guards, test_c1_resume_scale_and_rotation, test_c2_main_run_path,
         test_c3_guards_and_cli_edges]


if __name__ == "__main__":
    for fn in TESTS:
        fn()
    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for name in _failures:
            print(f"  - {name}")
        sys.exit(1)
    print("\033[32mAll calibrate_vel_anchor tests passed.\033[0m")
