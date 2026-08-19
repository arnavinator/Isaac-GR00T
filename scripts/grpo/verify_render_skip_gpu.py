"""Real-stack verification for skip_intermediate_render — RUN THIS ON YOUR GPU VM
(in the robocasa venv). No model server needed.

The CPU unit tests (test_skip_intermediate_render.py) drive robosuite's
`Observable` state machine but stub MuJoCo and rendering. This script closes that
gap: it proves, against REAL robosuite/MuJoCo/EGL rendering, the one claim the
whole optimization rests on —

    the frame kept with skip_intermediate_render=True is BYTE-IDENTICAL to the
    frame the unskipped path would have produced

— and reports the actual render count and wall-clock saving.

Method: one env instance, two passes. The flag is toggled at runtime between
passes and the SAME fixed action chunks are replayed from the SAME reset seed, so
the scene, the initial state, and the physics are identical by construction and
any frame difference is attributable to the render gating alone. A reproducibility
pre-check runs first, so a non-deterministic env is reported rather than silently
compared.

HOW TO RUN (on the GPU VM, robocasa venv, no server):
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/grpo/verify_render_skip_gpu.py

Useful flags: --env-name, --n-chunks, --n-action-steps, --seed.

Exit code 0 = all checks passed; 1 = at least one failed.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from collect_episodes import _make_collector_env  # noqa: E402  (robocasa venv import)

# ── tiny PASS/FAIL harness (matches the other verify/test files) ─────────────
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"

_failures = []


def check(ok, label, detail=""):
    print(f"  {PASS if ok else FAIL}  {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def info(label):
    print(f"  {INFO}  {label}")


# ── helpers ──────────────────────────────────────────────────────────────────


def video_keys(obs):
    return sorted(k for k in obs if k.startswith("video."))


def frames_of(obs):
    return {k: np.asarray(obs[k]) for k in video_keys(obs)}


def fixed_actions(wrapper, n_chunks, seed=0):
    """One reusable list of action chunks, identical across both passes.

    Continuous entries are damped so the arm doesn't flail; integer entries
    (gripper_close / control_mode / base_mode) are left exactly as sampled so
    they stay inside their discrete spaces.
    """
    wrapper.action_space.seed(seed)
    chunks = []
    for _ in range(n_chunks):
        act = wrapper.action_space.sample()
        chunks.append(
            {
                k: (v * 0.25).astype(v.dtype)
                if np.issubdtype(np.asarray(v).dtype, np.floating)
                else v
                for k, v in act.items()
            }
        )
    return chunks


def instrument_renders(base_env, counter):
    """Patch the CURRENT sim's render() to count calls; returns a restore fn.

    Must be re-applied after every reset: robocasa envs are built with
    hard_reset=True (kitchen.py / tabletop.py), so MujocoEnv.reset destroys and
    rebuilds self.sim, and a patch installed on the previous sim object counts
    nothing afterwards.
    """
    sim = base_env.env.sim
    original = sim.render

    def counting_render(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    sim.render = counting_render
    return lambda: setattr(sim, "render", original)


def rollout(wrapper, base_env, counter, actions, skip, seed):
    """Replay `actions` with the flag forced to `skip`; return per-chunk frames."""
    wrapper.skip_intermediate_render = skip
    obs, _ = wrapper.reset(seed=seed)
    # Instrument AFTER the reset — see instrument_renders.
    restore = instrument_renders(base_env, counter)
    counter["n"] = 0
    t0 = time.perf_counter()
    per_chunk = []
    for act in actions:
        obs, _reward, terminated, truncated, _info = wrapper.step(act)
        per_chunk.append(frames_of(obs))
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - t0
    restore()
    return {
        "frames": per_chunk,
        "renders": counter["n"],
        "seconds": elapsed,
        "n_chunks": len(per_chunk),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--env-name",
        default="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
        help="Env id to verify (default matches GRPOConfig.env_names[0]).",
    )
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--max-episode-steps", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=== skip_intermediate_render real-stack verification ===\n")
    print(f"env={args.env_name}  n_action_steps={args.n_action_steps}  "
          f"chunks={args.n_chunks}  seed={args.seed}\n")

    # Construct with the flag ON so the render gate resolves and the wrapper's
    # construction guards run; the flag is toggled per pass below.
    wrapper = _make_collector_env(
        env_name=args.env_name,
        env_idx=0,
        total_n_envs=1,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        skip_intermediate_render=True,
    )
    gate = wrapper._render_gate
    check(gate is not None, "render gate resolved", type(gate).__name__)
    print(f"  {INFO}  gate class: {type(gate).__name__} "
          f"from {type(gate).__module__}")

    counter = {"n": 0}
    # Physical cameras, NOT observation keys: the base robocasa copy emits a
    # res512_* companion for every res256_* key, so counting video keys would
    # double the expected render count.
    n_cams = len(gate.camera_names)
    print(f"  {INFO}  physical cameras: {n_cams} ({list(gate.camera_names)})")
    try:
        actions = fixed_actions(wrapper, args.n_chunks, seed=0)

        # --- 0. Is the env reproducible enough for a byte-exact comparison? ---
        a = rollout(wrapper, gate, counter, actions[:1], skip=False, seed=args.seed)
        b = rollout(wrapper, gate, counter, actions[:1], skip=False, seed=args.seed)
        reproducible = all(
            np.array_equal(fa[k], fb[k])
            for fa, fb in zip(a["frames"], b["frames"])
            for k in fa
        )
        check(
            reproducible,
            "env replays identically from the same seed",
            "required for the byte-exact comparison below",
        )

        # --- 1. baseline vs skipped ------------------------------------------
        base = rollout(wrapper, gate, counter, actions, skip=False, seed=args.seed)
        skipped = rollout(wrapper, gate, counter, actions, skip=True, seed=args.seed)

        check(
            base["n_chunks"] == skipped["n_chunks"],
            "same number of chunks executed",
            f"{base['n_chunks']} vs {skipped['n_chunks']}",
        )

        info(
            f"renders: baseline={base['renders']} "
            f"skipped={skipped['renders']} "
            f"({n_cams} cameras x {base['n_chunks']} chunks)"
        )
        check(
            skipped["renders"] < base["renders"],
            "skipping actually reduced sim.render() calls",
            f"{base['renders']} → {skipped['renders']}",
        )
        expected = n_cams * skipped["n_chunks"]
        check(
            skipped["renders"] == expected,
            "exactly one render per camera per chunk",
            f"expected {expected}, got {skipped['renders']}",
        )

        # --- 2. THE claim: identical frames ---------------------------------
        if reproducible:
            mismatches = []
            for i, (fb, fs) in enumerate(zip(base["frames"], skipped["frames"])):
                if sorted(fb) != sorted(fs):
                    mismatches.append(f"chunk {i}: key set differs")
                    continue
                for k in fb:
                    if fb[k].dtype != fs[k].dtype:
                        mismatches.append(
                            f"chunk {i} {k}: dtype {fs[k].dtype} != {fb[k].dtype}"
                        )
                    elif not np.array_equal(fb[k], fs[k]):
                        diff = np.abs(
                            fb[k].astype(np.int16) - fs[k].astype(np.int16)
                        )
                        mismatches.append(
                            f"chunk {i} {k}: {int((diff > 0).sum())} px differ, "
                            f"max |Δ|={int(diff.max())}"
                        )
            check(
                not mismatches,
                "kept frames are byte-identical to the unskipped path",
                "; ".join(mismatches[:4]) if mismatches else
                f"{base['n_chunks']} chunks x {n_cams} cameras "
                f"({len(base['frames'][0])} video keys)",
            )
        else:
            info("skipping the byte-exact frame comparison (env not reproducible)")

        # --- 3. no blank frames ---------------------------------------------
        blanks = [
            f"chunk {i} {k}"
            for i, f in enumerate(skipped["frames"])
            for k in f
            if f[k].dtype != np.uint8 or f[k].max() == 0
        ]
        check(
            not blanks,
            "every kept frame is non-blank uint8",
            "; ".join(blanks[:4]) if blanks else f"{base['n_chunks']} chunks",
        )

        # --- 4. speed -------------------------------------------------------
        speedup = base["seconds"] / skipped["seconds"] if skipped["seconds"] else 0
        info(
            f"wall clock for {base['n_chunks']} chunks: "
            f"baseline={base['seconds']:.2f}s skipped={skipped['seconds']:.2f}s "
            f"({speedup:.2f}x)"
        )
        if speedup < 1.2:
            info(
                "speedup is small — rendering may not dominate this env's "
                "substep cost; check MUJOCO_GL/EGL and camera resolution"
            )
    finally:
        wrapper.close()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
