"""End-to-end / real-stack verification for the multi-turn collection fix
(num_async_vector_env) — RUN THIS ON YOUR GPU VM (in the robocasa venv).

This complements the CPU unit tests (test_turn_collection.py / test_balanced_fixes.py),
which use fakes. The bug that mattered most (gymnasium NEXT_STEP autoreset eating
the branch-point restart across turns) was a *framework-semantics* issue that
fakes initially masked, so it can only be fully trusted against the REAL
robocasa/MuJoCo + gymnasium vector envs. This script does that.

────────────────────────────────────────────────────────────────────────────
PHASE A — autoreset / branch-point integrity  (NO model server, NO GPU needed)
────────────────────────────────────────────────────────────────────────────
Decisively validates the autoreset_mode=DISABLED fix against REAL robocasa envs
using only zero actions (no policy). The logic:

  * Align all physical envs to a group's branch point S (seed-reset scene).
  * Drive turn 1 to truncation with zero actions (so the envs become "done" —
    exactly the state that, under gymnasium's default NEXT_STEP autoreset, would
    leave a pending autoreset flag).
  * Restart to S via _restart_at_branch_point (reset + apply_scene_bundle).
  * Step ONCE with a zero action and compare the resulting sim state to the
    state reached by the SAME zero action from S on turn 1.

  MuJoCo is deterministic, so "same start S + same zero action" MUST give the
  same next state. Under the autoreset bug the restart's first step would
  silently env.reset() to a FRESH RANDOM scene instead — a glaring mismatch
  (often even a different sim-state length). This is the decisive check and it
  needs no GPU and no model server.

────────────────────────────────────────────────────────────────────────────
PHASE B — full collection through the real policy  (NEEDS a running grpo_server)
────────────────────────────────────────────────────────────────────────────
Runs the real EpisodeCollector.collect() with num_async_vector_env < group_size
and fast-forward forced ON, through the GPU model server, then inspects the
collected episodes: episode counts, shared group_id/env_seed, bit-identical
branch points within a group, fresh-noise diversity across turns, and
chunk0→chunk1 continuity (a teleport would betray an autoreset). FF makes the
branch point a mid-task state, so any autoreset (robot snapping back to home) is
stark.

────────────────────────────────────────────────────────────────────────────
HOW TO RUN (on the GPU VM)
────────────────────────────────────────────────────────────────────────────
Phase A only (robocasa venv; no server, runs on CPU):
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/grpo/verify_multiturn_gpu.py --phase a

Phase B (Terminal 1 — start the model server, main venv, GPU):
    uv run python scripts/grpo/grpo_server.py \
        --model-path nvidia/GR00T-N1.6-3B \
        --embodiment-tag ROBOCASA_PANDA_OMRON --use-sim-policy-wrapper
  (Terminal 2 — robocasa venv):
    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
        scripts/grpo/verify_multiturn_gpu.py --phase b \
        --server-host 127.0.0.1 --server-port 5555

Both phases:  ... --phase both --server-host 127.0.0.1 --server-port 5555
For a visual cross-check of branch-point alignment, run collect_episodes.py with
--debug-fast-forward (it renders per-group camera montages).

Exit code 0 = all checks passed; 1 = at least one failed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from collect_episodes import EpisodeCollector  # noqa: E402  (robocasa venv import)

# ── tiny PASS/FAIL harness (matches the other test files) ────────────────────
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  ── {detail}" if detail else ""))
        _failures.append(name)


def info(msg: str):
    print(f"  {INFO}  {msg}")


# ── helpers ──────────────────────────────────────────────────────────────────
def _zero_action(sample):
    """Build an all-zeros action matching the (possibly Dict) action space sample."""
    if isinstance(sample, dict):
        return {k: np.zeros_like(v) for k, v in sample.items()}
    return np.zeros_like(sample)


def _sim_states(collector) -> list[np.ndarray]:
    """Per-env flat MuJoCo sim state (qpos/qvel/act/time) via the wrapper RPC."""
    return [np.asarray(s) for s in collector.envs.call("get_sim_state_flat")]


def _state_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs elementwise diff; +inf if shapes differ (e.g. scene/model changed)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _autoreset_mode(venv):
    m = getattr(venv, "autoreset_mode", None)
    if m is None:
        m = getattr(venv, "metadata", {}).get("autoreset_mode")
    return m


def _drive_to_done(collector, zero, max_outer: int) -> int:
    """Step every env with zero actions until all are done. Returns the number of
    outer steps taken. NOTE: with zero actions no env succeeds, so all envs
    truncate together at max_episode_steps — this does NOT exercise the
    staggered 'step a done env while a sibling is active' path (that is covered
    by the CPU unit test test_turn_collection.test_multistep_no_op_when_done, and
    by Phase B's real-policy rollouts of differing lengths)."""
    n = collector.num_envs
    done = np.zeros(n, dtype=bool)
    steps = 0
    while not done.all() and steps < max_outer + 4:
        _, _, terms, truncs, _ = collector.envs.step(zero)
        done = done | np.asarray(terms, bool) | np.asarray(truncs, bool)
        steps += 1
    return steps


# ── PHASE A ──────────────────────────────────────────────────────────────────
def phase_a(env_name: str, n_action_steps: int):
    print("\n" + "=" * 78)
    print("PHASE A — autoreset / branch-point integrity (no model server)")
    print("=" * 78)

    group_size, num_envs = 4, 2          # → turns_per_group = 2 (one restart)
    max_episode_steps = 2 * n_action_steps   # truncate after 2 outer steps

    collector = EpisodeCollector(
        env_name=env_name,
        group_size=group_size,
        max_episode_steps=max_episode_steps,
        n_action_steps=n_action_steps,
        server_host="127.0.0.1", server_port=5555,   # PolicyClient connects lazily; unused here
        num_async_vector_env=num_envs,
    )
    try:
        # 1. Construction. Autoreset handling is gymnasium-version dependent
        # (see EpisodeCollector.__init__): gymnasium>=1.0 → DISABLED is requested
        # (the load-bearing fix for NEXT_STEP autoreset); older gymnasium (e.g.
        # the robocasa venv) has no AutoresetMode and uses same-step autoreset,
        # which doesn't leak — so no autoreset_mode is set. Either is fine here;
        # the decisive cross-turn check in step 6 validates the actual behavior.
        import gymnasium as _gym
        mode = _autoreset_mode(collector.envs)
        if getattr(_gym.vector, "AutoresetMode", None) is not None:
            check("vector env requests autoreset_mode=DISABLED (gymnasium>=1.0)",
                  str(getattr(mode, "name", mode)) == "DISABLED", f"got {mode!r}")
        else:
            info(f"gymnasium<1.0 (no AutoresetMode): same-step autoreset (mode={mode!r}) "
                 f"— no cross-turn leak, no autoreset_mode needed; step 6 still validates it")
        check("num_envs == 2 and turns_per_group == 2",
              collector.num_envs == 2 and collector.turns_per_group == 2,
              f"num_envs={collector.num_envs}, turns={collector.turns_per_group}")

        zero = _zero_action(collector.envs.action_space.sample())
        seed = 12345

        # 2. Align all physical envs to the group branch point S.
        _, branch_bundle = collector._align_envs_to_group_scene(seed)
        S0 = _sim_states(collector)
        cross_env = max(_state_diff(S0[0], S0[i]) for i in range(1, num_envs))
        check("all physical envs bit-identical at the branch point (align)",
              cross_env < 1e-8, f"max cross-env sim-state diff = {cross_env:.2e}")

        # 3. Baseline: one clean zero-action step from S (env 0). This is the
        #    FIRST step on a fresh env — never an autoreset — so it defines the
        #    deterministic 'S + zero action' result.
        collector.envs.step(zero)
        S1 = _sim_states(collector)
        baseline = _state_diff(S0[0], S1[0])
        info(f"baseline clean one-step drift (env0) = {baseline:.4e}")

        # 4. Finish turn 1 (drive to truncation) so the envs become 'done'.
        steps = _drive_to_done(collector, zero, max_outer=max_episode_steps // n_action_steps)
        info(f"turn 1 driven to done in {steps} outer step(s) with zero actions")

        # 5. Restart to the SAME branch point and re-verify it took.
        collector._restart_at_branch_point(branch_bundle, seed)
        S0r = _sim_states(collector)
        restore_diff = _state_diff(S0[0], S0r[0])
        check("restart restored the bit-identical branch point (env0)",
              restore_diff < 1e-8, f"sim-state diff vs original branch = {restore_diff:.2e}")
        cross_env_r = max(_state_diff(S0r[0], S0r[i]) for i in range(1, num_envs))
        check("all envs bit-identical after restart",
              cross_env_r < 1e-8, f"max cross-env diff = {cross_env_r:.2e}")

        # 6. THE DECISIVE CHECK. Same start S + same zero action ⇒ same next state.
        #    A leftover autoreset would instead reset to a fresh random scene here.
        collector.envs.step(zero)
        S1r = _sim_states(collector)
        same_shape = (S1r[0].shape == S1[0].shape)
        post_diff = _state_diff(S1[0], S1r[0])
        info(f"post-restart first-step vs baseline first-step diff (env0) = {post_diff:.4e} "
             f"(shapes {'match' if same_shape else 'DIFFER'})")
        check("turn-2 first step ran from the branch point, NOT a fresh autoreset",
              same_shape and post_diff < 0.5,
              f"diff={post_diff:.3e}, shape_match={same_shape} — a large diff or shape "
              f"change means the restart was eaten by autoreset (branch point lost)")

        print("\n  (If the decisive check passed: turns 2..k genuinely resume from the "
              "shared\n   branch point on the real stack — the autoreset fix holds. For a "
              "visual\n   cross-check, run collect_episodes.py with --debug-fast-forward.)")
    finally:
        collector.close()


# ── PHASE B ──────────────────────────────────────────────────────────────────
def _chunk0_state(ep) -> dict:
    return ep["states"][0] if ep.get("states") else {}


def _state_vec(state: dict) -> np.ndarray:
    return np.concatenate([np.asarray(v, np.float64).ravel()
                           for _, v in sorted(state.items())]) if state else np.zeros(0)


def phase_b(env_name: str, server_host: str, server_port: int,
            n_action_steps: int, max_episode_steps: int):
    print("\n" + "=" * 78)
    print("PHASE B — full multi-turn collection through the real policy (needs server)")
    print("=" * 78)

    group_size, num_envs, num_groups = 4, 2, 2   # 2 turns/group, 2 groups → 8 eps
    collector = EpisodeCollector(
        env_name=env_name,
        group_size=group_size,
        max_episode_steps=max_episode_steps,
        n_action_steps=n_action_steps,
        server_host=server_host, server_port=server_port,
        num_async_vector_env=num_envs,
    )
    try:
        # Fast-forward FORCED on (pct=1.0) so the branch point is a mid-task state —
        # makes any autoreset (robot snapping back to home) starkly visible.
        episodes = collector.collect(
            num_groups=num_groups,
            base_seed=777,
            success_weight=1.0,
            fast_forward_steps=12,
            fast_forward_pct=1.0,
            min_successful_groups=0,
        )
    finally:
        collector.close()

    check("collected group_size*num_groups episodes",
          len(episodes) == group_size * num_groups,
          f"got {len(episodes)} (expected {group_size * num_groups})")

    by_group: dict = {}
    for ep in episodes:
        by_group.setdefault(ep["group_id"], []).append(ep)
    check("episodes split into num_groups groups",
          len(by_group) == num_groups, f"group_ids={sorted(by_group)}")
    check("each group has group_size episodes",
          all(len(v) == group_size for v in by_group.values()),
          f"{ {g: len(v) for g, v in by_group.items()} }")

    branch_fingerprints = {}
    for gid, eps in sorted(by_group.items()):
        # Within a group, ALL rollouts (across turns) must start from the
        # bit-identical branch point: their chunk-0 states must match.
        ref = _state_vec(_chunk0_state(eps[0]))
        max_within = max(_state_diff(ref, _state_vec(_chunk0_state(e))) for e in eps)
        check(f"group {gid}: all {len(eps)} rollouts share the bit-identical branch point (chunk 0)",
              max_within < 1e-6, f"max chunk-0 state diff within group = {max_within:.3e}")
        branch_fingerprints[gid] = ref

        # Fresh per-query denoising noise ⇒ rollouts must be DISTINCT.
        noises = [e["initial_noises"][0] for e in eps
                  if e.get("initial_noises") and e["initial_noises"][0] is not None]
        if len(noises) == len(eps):
            uniq = len({n.tobytes() for n in noises})
            check(f"group {gid}: all rollouts got distinct initial noise (fresh per turn)",
                  uniq == len(eps), f"{uniq}/{len(eps)} distinct")
        else:
            info(f"group {gid}: initial noise not captured on all rollouts (skipping distinctness)")

        # Continuity diagnostic: chunk0→chunk1 should be a small physical step,
        # NOT a teleport to a fresh scene (the autoreset-bug signature, stark under FF).
        deltas = []
        for e in eps:
            if len(e.get("states", [])) >= 2:
                deltas.append(_state_diff(_state_vec(e["states"][0]), _state_vec(e["states"][1])))
        if deltas:
            mx, md = max(deltas), float(np.median(deltas))
            info(f"group {gid}: chunk0→chunk1 state deltas: median={md:.3e} max={mx:.3e}")
            check(f"group {gid}: no rollout teleports between chunk 0 and 1 (continuity)",
                  mx < max(md * 8.0, 0.5) + 1e-9,
                  f"max delta {mx:.3e} >> median {md:.3e} suggests a turn restarted from a "
                  f"fresh scene (autoreset)")

    # Different groups use different seeds ⇒ different branch points.
    if len(branch_fingerprints) == 2:
        g = sorted(branch_fingerprints)
        across = _state_diff(branch_fingerprints[g[0]], branch_fingerprints[g[1]])
        check("the two groups have DISTINCT branch points (different seeds → different scenes)",
              across > 1e-4, f"cross-group chunk-0 diff = {across:.3e}")

    succ = sum(int(e["success"]) for e in episodes)
    info(f"success rate: {succ}/{len(episodes)} (informational — not a correctness check)")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["a", "b", "both"], default="a")
    p.add_argument("--env-name", default="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env")
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=5555)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=120,
                   help="Phase B per-episode horizon (kept modest so the run is quick).")
    args = p.parse_args()

    if args.phase in ("a", "both"):
        phase_a(args.env_name, args.n_action_steps)
    if args.phase in ("b", "both"):
        phase_b(args.env_name, args.server_host, args.server_port,
                args.n_action_steps, args.max_episode_steps)

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) FAILED:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\033[32mAll verification checks passed.\033[0m")


if __name__ == "__main__":
    main()
