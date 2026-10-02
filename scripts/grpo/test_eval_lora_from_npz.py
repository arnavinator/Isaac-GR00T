"""CPU-only tests for eval_lora_from_npz.py's two start modes.

  1. CLI: --group-seeds parses to a list and is mutually exclusive with
     --obs-path (exactly one is required); validation rejects repeated seeds and
     only checks --obs-path in saved-state mode.
  2. Seed mode through the REAL EvalCollector/EpisodeCollector.collect over
     test_turn_collection.py's fakes: one group per seed, each reset with its
     seed, no init-state bundle, nothing recorded per chunk, and one scene
     fingerprint per seed (logged on a `scene:` line and kept for results.json).
  3. `_collect` kwargs for both modes, and `_summarize`'s per-scene tallies.
  4. `main()` end to end in both modes, checking results.json.

No GPU, no robocasa, no MuJoCo.

Run with:
    .venv/bin/python scripts/grpo/test_eval_lora_from_npz.py
"""
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import collect_episodes as ce                                 # noqa: E402
import eval_lora_from_npz as ev                               # noqa: E402
import test_turn_collection as tc                             # noqa: E402  (fakes)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: list = []

SEEDS = [100_067, 101_067, 102_067]


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


def _argv(*extra, out="/tmp/eval_test_out"):
    return ["--env-name", "robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
            "--output-dir", out, *extra]


def _fake_vector_envs():
    """Patch the vector-env constructors the way tc._make_collector does."""
    return (
        mock.patch.object(
            ce.gym.vector, "AsyncVectorEnv",
            lambda env_fns, **kw: tc.FakeVectorEnv(len(env_fns), kw.get("autoreset_mode")),
        ),
        mock.patch.object(
            ce.gym.vector, "SyncVectorEnv",
            lambda env_fns, **kw: tc.FakeVectorEnv(len(env_fns), kw.get("autoreset_mode")),
        ),
        mock.patch.object(ce, "PolicyClient", tc.FakePolicyClient),
    )


def _make_eval_collector(group_size: int, num_envs: int) -> ev.EvalCollector:
    a, b, c = _fake_vector_envs()
    with a, b, c:
        return ev.EvalCollector(
            env_name="robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env",
            group_size=group_size,
            max_episode_steps=100,
            n_action_steps=8,
            server_host="127.0.0.1",
            server_port=5555,
            num_async_vector_env=num_envs,
            log_scene_fingerprint=True,
        )


def _write_saved_state(path: Path) -> None:
    np.savez_compressed(
        str(path),
        __sim_state__=np.zeros(60, dtype=np.float64),
        __model_xml__="<mujoco>saved kitchen</mujoco>",
        __ep_meta__=json.dumps({"layout_id": 2, "style_id": 5}),
        __step_info__=json.dumps({"step": 10, "n_action_steps": 8}),
    )


# ---------------------------------------------------------------------------
# 1. CLI + validation
# ---------------------------------------------------------------------------

def test_cli():
    print("\n[cli] --group-seeds / --obs-path")
    a = ev.parse_args(_argv("--group-seeds", "100067, 101067,102067"))
    check("--group-seeds parses to a list of ints", a.group_seeds == SEEDS,
          f"{a.group_seeds}")
    check("--obs-path defaults to None in seed mode", a.obs_path is None)

    b = ev.parse_args(_argv("--obs-path", "/tmp/x.npz"))
    check("--obs-path alone still parses (saved-state mode unchanged)",
          b.obs_path == "/tmp/x.npz" and b.group_seeds is None)

    for label, argv in (
        ("both flags", _argv("--group-seeds", "1", "--obs-path", "/tmp/x.npz")),
        ("neither flag", _argv()),
        ("a non-integer seed", _argv("--group-seeds", "100067,abc")),
    ):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            try:
                ev.parse_args(argv)
                ok = False
            except SystemExit as e:
                ok = e.code == 2
        check(f"{label} → argparse error", ok)


def test_validation():
    print("\n[validate] repeated seeds, obs-path check only in saved-state mode")
    try:
        ev._validate_args(ev.parse_args(_argv("--group-seeds", "100067,101067,100067",
                                              "--num-attempts", "4", "--num-envs", "2")))
        check("a repeated seed is rejected", False, "no error")
    except ValueError as e:
        check("a repeated seed is rejected, naming it", "100067" in str(e), str(e))

    try:
        ev._validate_args(ev.parse_args(_argv("--group-seeds", "100067",
                                              "--num-attempts", "4", "--num-envs", "2")))
        check("seed mode needs no --obs-path file", True)
    except Exception as e:  # noqa: BLE001
        check("seed mode needs no --obs-path file", False, repr(e))

    try:
        ev._validate_args(ev.parse_args(_argv("--group-seeds", "100067",
                                              "--num-attempts", "40", "--num-envs", "3")))
        check("--num-envs must still divide --num-attempts in seed mode", False)
    except ValueError as e:
        check("--num-envs must still divide --num-attempts in seed mode",
              "divide" in str(e), str(e))

    try:
        ev._validate_args(ev.parse_args(_argv("--obs-path", "/nonexistent/x.npz",
                                              "--num-attempts", "4", "--num-envs", "2")))
        check("saved-state mode still checks the file exists", False)
    except FileNotFoundError:
        check("saved-state mode still checks the file exists", True)


# ---------------------------------------------------------------------------
# 2. Seed mode through the real collector
# ---------------------------------------------------------------------------

def test_seed_mode_collects_one_group_per_seed():
    print("\n[collect] seed mode drives the real EpisodeCollector.collect")
    args = ev.parse_args(_argv("--group-seeds", ",".join(map(str, SEEDS)),
                               "--num-attempts", "4", "--num-envs", "2"))
    c = _make_eval_collector(group_size=4, num_envs=2)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            eps = ev._collect(c, args, None)
        init_path = c._active_init_bundle_path
    finally:
        c.close()

    by_group: dict = {}
    for e in eps:
        by_group.setdefault(e["group_id"], set()).add(e["env_seed"])
    check("one group per seed, in order",
          by_group == {g: {s} for g, s in enumerate(SEEDS)}, f"{by_group}")
    check("num_attempts rollouts per scene",
          len(eps) == 4 * len(SEEDS), f"{len(eps)}")
    check("no init-state bundle in seed mode", init_path is None, f"{init_path!r}")
    check("no frames / states / noise recorded (empty per-chunk entries)",
          all(all(f == {} for f in e["video_frames"])
              and all(s == {} for s in e["states"])
              and all(n is None for n in e["initial_noises"])
              and all(r is None for r in e["raw_actions"]) for e in eps))

    fps = c.scene_fingerprints
    check("one fingerprint per seed", sorted(fps) == sorted(SEEDS), f"{fps}")
    check("fingerprints are real strings (not n/a / None)",
          all(isinstance(v, str) and "xml=" in v and "state=" in v
              for v in fps.values()), f"{fps}")
    scene_lines = [l for l in buf.getvalue().splitlines()
                   if l.strip().startswith("scene:")]
    check("a `scene:` log line per group, matching the recorded fingerprint",
          [l.split("scene:", 1)[1].strip() for l in scene_lines]
          == [fps[s] for s in SEEDS], f"{scene_lines}")

    res = ev._summarize(eps, fps, seed_mode=True)
    per = res["per_scene"]
    check("per_scene lists each seed once, in group order",
          [p["group_seed"] for p in per] == SEEDS, f"{per}")
    check("per_scene carries that scene's fingerprint",
          all(p["scene_fingerprint"] == fps[p["group_seed"]] for p in per))
    check("per_scene totals = --num-attempts",
          all(p["total"] == 4 for p in per), f"{[p['total'] for p in per]}")
    check("pooled summary = sum of scenes",
          res["summary"]["total"] == sum(p["total"] for p in per)
          and res["summary"]["successes"] == sum(p["successes"] for p in per),
          f"{res['summary']}")
    check("attempts carry their group index",
          sorted({a["group_idx"] for a in res["attempts"]}) == [0, 1, 2])


# ---------------------------------------------------------------------------
# 3. _collect kwargs and _summarize in saved-state mode
# ---------------------------------------------------------------------------

class _RecordingCollector:
    def __init__(self):
        self.kwargs = None

    def collect(self, **kwargs):
        self.kwargs = kwargs
        return []


def test_collect_kwargs():
    print("\n[collect] kwargs per mode")
    seed_args = ev.parse_args(_argv("--group-seeds", ",".join(map(str, SEEDS))))
    rc = _RecordingCollector()
    ev._collect(rc, seed_args, None)
    k = rc.kwargs
    check("seed mode: num_groups == max_groups == len(seeds), not dynamic",
          k["num_groups"] == k["max_groups"] == len(SEEDS)
          and k["min_alive_groups"] == 0, f"{k}")
    check("seed mode: group_seeds passed through, no init-state path",
          k["group_seeds"] == SEEDS and k.get("init_state_npz_path") is None, f"{k}")
    check("seed mode: fast-forward off",
          k["fast_forward_steps"] == 0 and k["fast_forward_pct"] == 0.0)

    obs_args = ev.parse_args(_argv("--obs-path", "/tmp/x.npz"))
    rc = _RecordingCollector()
    ev._collect(rc, obs_args, Path("/tmp/x.npz"))
    k = rc.kwargs
    check("saved-state mode: one group from the npz, unchanged",
          k["num_groups"] == k["max_groups"] == 1
          and k["init_state_npz_path"] == "/tmp/x.npz"
          and k.get("group_seeds") is None, f"{k}")


def test_summarize_saved_state():
    print("\n[summarize] saved-state mode hides the reset seed")
    eps = [{"group_id": 0, "env_seed": 42, "success": s, "num_steps": n}
           for s, n in ((True, 100), (False, 480), (True, 200))]
    res = ev._summarize(eps, {42: "layout=2 style=5 xml=ab state=cd"}, seed_mode=False)
    p = res["per_scene"]
    check("one scene, group_seed null (the bundle sets the scene)",
          len(p) == 1 and p[0]["group_seed"] is None, f"{p}")
    check("fingerprint found via the group's reset seed",
          p[0]["scene_fingerprint"] == "layout=2 style=5 xml=ab state=cd", f"{p}")
    check("tallies", p[0]["successes"] == 2 and p[0]["total"] == 3
          and abs(p[0]["mean_num_steps_successful"] - 150.0) < 1e-9, f"{p}")


# ---------------------------------------------------------------------------
# 4. main() end to end
# ---------------------------------------------------------------------------

class _PingClient:
    def __init__(self, host=None, port=None, strict=False):
        self.socket = mock.MagicMock()

    def ping(self):
        return True


def _run_main(argv):
    a, b, c = _fake_vector_envs()
    buf = io.StringIO()
    with a, b, c, mock.patch.object(ev, "PolicyClient", _PingClient), \
            mock.patch.object(sys, "argv", ["eval_lora_from_npz.py", *argv]), \
            redirect_stdout(buf):
        ev.main()
    return buf.getvalue()


def test_main_seed_mode():
    print("\n[main] seed mode writes per-scene results")
    with tempfile.TemporaryDirectory() as d:
        out = _run_main(_argv("--group-seeds", ",".join(map(str, SEEDS)),
                              "--num-attempts", "4", "--num-envs", "2", out=d))
        res = json.loads((Path(d) / "results.json").read_text())
    lin = res["lineage"]
    check("lineage records the seeds and no obs_path",
          lin["group_seeds"] == SEEDS and lin["obs_path"] is None, f"{lin}")
    check("lineage: full budget from step 0",
          lin["consumed_substeps"] == 0
          and lin["remaining_substeps_budget"] == lin["max_episode_steps"], f"{lin}")
    check("per_scene in results.json, one per seed, with fingerprints",
          [p["group_seed"] for p in res["per_scene"]] == SEEDS
          and all(p["scene_fingerprint"] for p in res["per_scene"]))
    check("pooled total = attempts × scenes", res["summary"]["total"] == 12)
    check("banner lists the scene seeds", "Scene seeds: 100067, 101067, 102067" in out)
    check("final printout has a line per seed",
          all(f"seed={s}:" in out for s in SEEDS), out[-600:])


def test_main_saved_state_mode():
    print("\n[main] saved-state mode still works")
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "state.npz"
        _write_saved_state(npz)
        out = _run_main(_argv("--obs-path", str(npz),
                              "--num-attempts", "4", "--num-envs", "2", out=d))
        res = json.loads((Path(d) / "results.json").read_text())
    lin = res["lineage"]
    check("lineage keeps obs_path, no seeds",
          lin["obs_path"] == str(npz.resolve()) and lin["group_seeds"] is None, f"{lin}")
    check("branch-step accounting from __step_info__ unchanged",
          lin["branch_step"] == 10 and lin["consumed_substeps"] == 80, f"{lin}")
    check("one scene, fingerprinted from the saved bundle",
          len(res["per_scene"]) == 1
          and res["per_scene"][0]["group_seed"] is None
          and "layout=2" in (res["per_scene"][0]["scene_fingerprint"] or ""),
          f"{res['per_scene']}")
    check("summary total = --num-attempts", res["summary"]["total"] == 4)
    check("final printout labels the saved state", "saved state:" in out, out[-400:])


if __name__ == "__main__":
    test_cli()
    test_validation()
    test_seed_mode_collects_one_group_per_seed()
    test_collect_kwargs()
    test_summarize_saved_state()
    test_main_seed_mode()
    test_main_saved_state_mode()
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("All checks passed.")
