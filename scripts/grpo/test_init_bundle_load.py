"""Test that _load_init_bundle in collect_episodes.py correctly converts an
interactive_rollout.py-style npz into the dict shape apply_scene_bundle expects.

NPZ contract (set by scripts/denoising_lab/eval/interactive_rollout.py and
consumed by scripts/denoising_lab/eval/branching_rollout.py:182-210):
    __sim_state__  : np.ndarray (MjSimState flat)
    __model_xml__  : str        (scene XML)
    __ep_meta__    : str        (JSON-serialized robosuite ep_meta dict)

apply_scene_bundle (GroupAlignmentWrapper in collect_episodes.py) expects:
    sim_state      : np.ndarray
    model_xml      : str
    ep_meta        : dict       (NOT a JSON string)

NOTE: collect_episodes.py top-level imports robocasa via rollout_policy. On a
robocasa-equipped host (the user's VM) this test imports the REAL function and
catches implementation regressions. On a host without robocasa it falls back to
a local re-implementation and still validates the npz key contract.
"""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from collect_episodes import _load_init_bundle as _load_init_bundle_real
    _USING_REAL = True
except Exception:
    _USING_REAL = False

    def _load_init_bundle_real(npz_path: str) -> dict:  # type: ignore[no-redef]
        # Local mirror of collect_episodes.py:_load_init_bundle. Update both
        # together if the npz contract changes (see branching_rollout.py:182-210).
        if not npz_path:
            raise ValueError("init_state_npz_path is empty; expected a path string")
        data = dict(np.load(npz_path, allow_pickle=True))
        if "__sim_state__" not in data:
            raise ValueError(f"{npz_path}: missing __sim_state__.")
        if "__model_xml__" not in data or "__ep_meta__" not in data:
            raise ValueError(f"{npz_path}: missing __model_xml__ and/or __ep_meta__.")
        try:
            ep_meta = json.loads(str(data["__ep_meta__"]))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{npz_path}: __ep_meta__ is not valid JSON ({e})."
            ) from e
        if not isinstance(ep_meta, dict):
            raise ValueError(
                f"{npz_path}: __ep_meta__ JSON-decoded to {type(ep_meta).__name__}, "
                "expected dict."
            )
        consumed_substeps = 0
        branch_step = None
        saved_n_action_steps = None
        if "__step_info__" in data:
            try:
                step_info = json.loads(str(data["__step_info__"]))
                if isinstance(step_info, dict):
                    branch_step = step_info.get("step")
                    saved_n_action_steps = step_info.get("n_action_steps")
            except json.JSONDecodeError:
                pass
        if branch_step is None:
            import re
            m = re.search(r"step(\d+)", Path(npz_path).stem)
            if m:
                branch_step = int(m.group(1))
        if branch_step is not None and saved_n_action_steps is not None:
            consumed_substeps = int(branch_step) * int(saved_n_action_steps)
            if consumed_substeps < 0:
                raise ValueError(
                    f"{npz_path}: consumed_substeps={consumed_substeps} (negative)."
                )
        elif branch_step is not None:
            import warnings as _w
            _w.warn(
                f"{npz_path}: missing n_action_steps; cannot compute "
                f"consumed_substeps. Defaulting to 0.",
                stacklevel=3,
            )
        else:
            import warnings as _w
            _w.warn(
                f"{npz_path}: no __step_info__ and filename has no step; "
                f"cannot compute consumed_substeps. Defaulting to 0.",
                stacklevel=3,
            )
        return {
            "ep_meta": ep_meta,
            "model_xml": str(data["__model_xml__"]),
            "sim_state": np.asarray(data["__sim_state__"]),
            "consumed_substeps": consumed_substeps,
        }


def _write_minimal_npz(path: Path, *, include_sim_state: bool = True,
                       include_model_xml: bool = True,
                       include_ep_meta: bool = True,
                       step_info: dict | None = None) -> None:
    """Write an npz with as many of the three required keys as requested.

    The keys/values mirror what interactive_rollout.py would save, but the
    contents are dummies so no robocasa or MuJoCo runtime is required.

    If `step_info` is a dict (e.g., {"step": 10, "n_action_steps": 8}), the
    __step_info__ key is included so consumed_substeps gets billed against
    the wrapper's max_episode_steps budget.
    """
    save_dict: dict = {}
    if include_sim_state:
        save_dict["__sim_state__"] = np.zeros(60, dtype=np.float64)
    if include_model_xml:
        save_dict["__model_xml__"] = np.array("<mujoco/>", dtype=object)
    if include_ep_meta:
        save_dict["__ep_meta__"] = np.array(
            json.dumps({"layout_id": 7, "style_id": 2}),
            dtype=object,
        )
    if step_info is not None:
        save_dict["__step_info__"] = np.array(json.dumps(step_info), dtype=object)
    np.savez_compressed(str(path), **save_dict)


def test_happy_path_keys_and_types():
    """A well-formed npz produces a dict with the four expected keys and types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "ep000_step010.npz"
        _write_minimal_npz(npz_path, step_info={"step": 10, "n_action_steps": 8})

        bundle = _load_init_bundle_real(str(npz_path))

        assert set(bundle.keys()) == {
            "ep_meta", "model_xml", "sim_state", "consumed_substeps",
        }, f"Unexpected keys: {sorted(bundle.keys())}"
        # ep_meta must be a NATIVE dict (json.loads), not a JSON string. The
        # robosuite set_ep_meta path indexes into it like a dict downstream.
        assert isinstance(bundle["ep_meta"], dict), (
            f"ep_meta should be dict, got {type(bundle['ep_meta'])}"
        )
        assert bundle["ep_meta"] == {"layout_id": 7, "style_id": 2}
        # model_xml must be a plain str — apply_scene_bundle passes it directly
        # to robosuite_env.edit_model_xml which expects str.
        assert isinstance(bundle["model_xml"], str), (
            f"model_xml should be str, got {type(bundle['model_xml'])}"
        )
        assert bundle["model_xml"] == "<mujoco/>"
        # sim_state must be a numpy array — apply_scene_bundle passes it to
        # robosuite_env.sim.set_state_from_flattened which expects np.ndarray.
        assert isinstance(bundle["sim_state"], np.ndarray), (
            f"sim_state should be np.ndarray, got {type(bundle['sim_state'])}"
        )
        assert bundle["sim_state"].shape == (60,)

        print(
            f"  [PASS] happy-path returns "
            f"{set(bundle.keys())} with correct types "
            f"(using {'real' if _USING_REAL else 'inline'} impl)"
        )


def test_missing_sim_state_raises():
    """A npz without __sim_state__ raises ValueError (named in the message)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "bad.npz"
        _write_minimal_npz(npz_path, include_sim_state=False)

        try:
            _load_init_bundle_real(str(npz_path))
        except ValueError as e:
            assert "__sim_state__" in str(e), (
                f"Error message should mention the missing key, got: {e}"
            )
            print("  [PASS] missing __sim_state__ raises ValueError")
            return
        raise AssertionError("expected ValueError for missing __sim_state__")


def test_missing_model_xml_or_ep_meta_raises():
    """A npz with sim_state but missing model_xml or ep_meta raises ValueError.

    These two are also required — without model_xml the scene can't be rebuilt,
    without ep_meta the layout/style aren't pinned. Restoring just sim_state on
    top of a freshly-reset scene would put the saved robot pose into a randomly-
    laid-out kitchen.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Missing model_xml
        npz1 = Path(tmpdir) / "no_xml.npz"
        _write_minimal_npz(npz1, include_model_xml=False)
        try:
            _load_init_bundle_real(str(npz1))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for missing __model_xml__")

        # Missing ep_meta
        npz2 = Path(tmpdir) / "no_ep_meta.npz"
        _write_minimal_npz(npz2, include_ep_meta=False)
        try:
            _load_init_bundle_real(str(npz2))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for missing __ep_meta__")

        print("  [PASS] missing __model_xml__ or __ep_meta__ raises ValueError")


def test_apply_scene_bundle_contract():
    """The dict returned must use the keys apply_scene_bundle reads, not the
    npz key spelling. Catches drift if either side renames its keys.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "ep000_step010.npz"
        _write_minimal_npz(npz_path, step_info={"step": 10, "n_action_steps": 8})

        bundle = _load_init_bundle_real(str(npz_path))

        # apply_scene_bundle (collect_episodes.py:412-522) reads bundle["ep_meta"],
        # bundle["model_xml"], bundle["sim_state"], bundle["consumed_substeps"].
        # The bundle MUST NOT carry the npz-side double-underscore spellings.
        for npz_spelling in (
            "__ep_meta__", "__model_xml__", "__sim_state__", "__step_info__",
        ):
            assert npz_spelling not in bundle, (
                f"Bundle leaked npz-side key {npz_spelling}; apply_scene_bundle "
                f"will silently ignore it and fall back to wrong defaults."
            )
        for bundle_spelling in (
            "ep_meta", "model_xml", "sim_state", "consumed_substeps",
        ):
            assert bundle_spelling in bundle, (
                f"Bundle missing apply_scene_bundle-expected key {bundle_spelling}."
            )
        # The contract is not just "key present" — it must carry the right
        # VALUE. step=10, n_action_steps=8 → consumed_substeps = 80.
        assert bundle["consumed_substeps"] == 80, (
            f"contract violation: expected consumed_substeps=80 (10*8), "
            f"got {bundle['consumed_substeps']}"
        )
        print("  [PASS] bundle uses apply_scene_bundle key spellings + correct values")


def test_consumed_substeps_extraction():
    """consumed_substeps must be billed correctly so the post-restore rollout
    truncates at max_episode_steps - consumed_substeps remaining sub-steps,
    matching branching_rollout.py:488-505. Covers four scenarios:

      1. __step_info__ has both `step` and `n_action_steps` → consumed = step * n_action_steps
      2. __step_info__ has only `step` → fall back to filename parse for `step`,
         but without n_action_steps we can't compute → default 0 (with warning)
      3. No __step_info__ but filename matches `ep*_step*.npz` → still need
         n_action_steps; default 0 (with warning)
      4. No __step_info__ and no parseable filename → default 0 (with warning)
    """
    import warnings as _warnings
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Canonical case: full step_info → consumed = 10 * 8 = 80
        npz1 = Path(tmpdir) / "ep000_step010.npz"
        _write_minimal_npz(npz1, step_info={"step": 10, "n_action_steps": 8})
        bundle1 = _load_init_bundle_real(str(npz1))
        assert bundle1["consumed_substeps"] == 80, (
            f"expected 80 (10 outer × 8 substeps), got {bundle1['consumed_substeps']}"
        )

        # 2. step_info with only `step` — n_action_steps missing → default 0 + warn
        npz2 = Path(tmpdir) / "ep000_step005.npz"
        _write_minimal_npz(npz2, step_info={"step": 5})
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            bundle2 = _load_init_bundle_real(str(npz2))
            assert bundle2["consumed_substeps"] == 0
            assert any("n_action_steps" in str(x.message) for x in w), (
                f"expected warning about missing n_action_steps, got: {[str(x.message) for x in w]}"
            )

        # 3. No step_info, filename has step → can extract step but not n_action_steps
        npz3 = Path(tmpdir) / "ep007_step012.npz"
        _write_minimal_npz(npz3, step_info=None)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            bundle3 = _load_init_bundle_real(str(npz3))
            # branch_step=12 from filename, but no n_action_steps → default 0
            assert bundle3["consumed_substeps"] == 0
            assert any("n_action_steps" in str(x.message) for x in w)

        # 4. Neither — default 0, warn that budget accounting is missing
        npz4 = Path(tmpdir) / "random_name.npz"
        _write_minimal_npz(npz4, step_info=None)
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            bundle4 = _load_init_bundle_real(str(npz4))
            assert bundle4["consumed_substeps"] == 0
            assert any("step" in str(x.message).lower() for x in w)

        # 5. consumed_substeps respects different n_action_steps
        npz5 = Path(tmpdir) / "ep000_step020.npz"
        _write_minimal_npz(npz5, step_info={"step": 20, "n_action_steps": 4})
        bundle5 = _load_init_bundle_real(str(npz5))
        assert bundle5["consumed_substeps"] == 80, (
            f"expected 80 (20 outer × 4 substeps), got {bundle5['consumed_substeps']}"
        )

        # 6. branch_step=0 — the legitimate "just-reset, no time elapsed" case.
        # Should produce consumed_substeps=0 (full fresh budget) without warning.
        npz6 = Path(tmpdir) / "ep000_step000.npz"
        _write_minimal_npz(npz6, step_info={"step": 0, "n_action_steps": 8})
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            bundle6 = _load_init_bundle_real(str(npz6))
            assert bundle6["consumed_substeps"] == 0
            assert not any(
                "consumed_substeps" in str(x.message) for x in w
            ), f"step=0 should not warn, got: {[str(x.message) for x in w]}"

        # 7. Negative step from a hand-edited __step_info__ should be REJECTED,
        # not silently produce an empty pre-fill via Python's `[x] * -n == []`.
        npz7 = Path(tmpdir) / "ep000_bad.npz"
        _write_minimal_npz(npz7, step_info={"step": -5, "n_action_steps": 8})
        try:
            _load_init_bundle_real(str(npz7))
        except ValueError as e:
            assert "negative" in str(e).lower() or "consumed" in str(e).lower(), (
                f"error should mention negative/consumed, got: {e}"
            )
        else:
            raise AssertionError(
                "expected ValueError for negative consumed_substeps"
            )

        print("  [PASS] consumed_substeps extracted/defaulted correctly across 7 scenarios")


def test_ep_meta_must_be_dict():
    """If __ep_meta__ JSON-decodes to a non-dict (e.g., a list), load fails fast.

    Without this check, the non-dict propagates to robosuite_env.set_ep_meta(),
    which would either crash deep inside robosuite or — worse — silently store
    the wrong type for later mutation/read.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "bad_ep_meta.npz"
        # Valid JSON but not a dict — a list this time.
        np.savez_compressed(
            str(npz_path),
            __sim_state__=np.zeros(60, dtype=np.float64),
            __model_xml__=np.array("<mujoco/>", dtype=object),
            __ep_meta__=np.array(json.dumps([1, 2, 3]), dtype=object),
        )
        try:
            _load_init_bundle_real(str(npz_path))
        except ValueError as e:
            assert "dict" in str(e).lower(), (
                f"Error message should mention expected type, got: {e}"
            )
            print("  [PASS] non-dict __ep_meta__ rejected with helpful error")
            return
        raise AssertionError(
            "expected ValueError for non-dict __ep_meta__"
        )


def test_malformed_ep_meta_json_rejected():
    """If __ep_meta__ is not valid JSON, load fails with the path quoted.

    Catches contracts where a saver writes raw Python repr() instead of JSON.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "bad_json.npz"
        np.savez_compressed(
            str(npz_path),
            __sim_state__=np.zeros(60, dtype=np.float64),
            __model_xml__=np.array("<mujoco/>", dtype=object),
            # Not valid JSON (single quotes, trailing comma).
            __ep_meta__=np.array("{'layout_id': 7,}", dtype=object),
        )
        try:
            _load_init_bundle_real(str(npz_path))
        except ValueError as e:
            msg = str(e)
            assert "JSON" in msg or "json" in msg, (
                f"Error message should mention JSON, got: {e}"
            )
            assert str(npz_path) in msg, (
                f"Error message should quote the failing path, got: {e}"
            )
            print("  [PASS] malformed JSON in __ep_meta__ rejected with path quoted")
            return
        raise AssertionError("expected ValueError for malformed JSON in __ep_meta__")


def test_defensive_copy_consecutive_calls():
    """Consecutive _get_init_bundle calls must return FRESH ep_meta and
    sim_state objects, not the cached references.

    Regression test for the deepcopy(ep_meta) + np.copy(sim_state) fix.
    Robosuite's set_ep_meta stores the dict by reference and later
    get_ep_meta MUTATES it in place; without fresh copies, the cache would
    accumulate state across iterations and break determinism. Likewise,
    MuJoCo's set_state_from_flattened has undocumented copy semantics, so
    np.copy is cheap insurance.

    This test mirrors the cache + copy structure of
    collect_episodes.py:_get_init_bundle. If the real function ever drops
    one of the copies, this test would still pass against the inline
    mirror — but the inline mirror exists precisely to encode the invariant,
    so a contributor changing one MUST change the other.
    """
    # Minimal mirror of EpisodeCollector's cache + copy logic.
    class _MiniCollector:
        def __init__(self):
            self._init_bundle = None
            self._init_bundle_path = None

        def get_bundle(self, npz_path: str) -> dict:
            if self._init_bundle is None or self._init_bundle_path != npz_path:
                self._init_bundle = _load_init_bundle_real(npz_path)
                self._init_bundle_path = npz_path
            return {
                "ep_meta": copy.deepcopy(self._init_bundle["ep_meta"]),
                "model_xml": self._init_bundle["model_xml"],
                "sim_state": np.copy(self._init_bundle["sim_state"]),
                "consumed_substeps": self._init_bundle["consumed_substeps"],
            }

    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "ep000_step010.npz"
        # Nested dict so deepcopy actually has work to do — a shallow copy
        # would still share the inner dict.
        np.savez_compressed(
            str(npz_path),
            __sim_state__=np.arange(60, dtype=np.float64),
            __model_xml__=np.array("<mujoco/>", dtype=object),
            __ep_meta__=np.array(
                json.dumps({
                    "layout_id": 7,
                    "object_cfgs": [{"name": "mug", "pos": [0.1, 0.2, 0.3]}],
                }),
                dtype=object,
            ),
            __step_info__=np.array(
                json.dumps({"step": 10, "n_action_steps": 8}), dtype=object,
            ),
        )
        col = _MiniCollector()
        b1 = col.get_bundle(str(npz_path))
        b2 = col.get_bundle(str(npz_path))

        # The bundle dicts themselves must be different objects (each call
        # builds a new outer dict via the {...} return).
        assert b1 is not b2, "outer bundle dict must be a fresh object"
        # ep_meta must be a fresh dict, with fresh nested containers too.
        assert b1["ep_meta"] is not b2["ep_meta"], (
            "ep_meta must be a fresh dict each call (regression: shallow copy)"
        )
        assert b1["ep_meta"]["object_cfgs"] is not b2["ep_meta"]["object_cfgs"], (
            "nested ep_meta containers must be fresh (regression: shallow copy "
            "instead of deepcopy)"
        )
        # sim_state must be a fresh numpy array, not the cached one.
        assert b1["sim_state"] is not b2["sim_state"], (
            "sim_state must be a fresh np.ndarray each call (regression: "
            "missing np.copy)"
        )
        # But VALUES must match — the copy must be faithful.
        assert b1["ep_meta"] == b2["ep_meta"]
        assert np.array_equal(b1["sim_state"], b2["sim_state"])
        # consumed_substeps is an int (immutable) so it's value-shared safely.
        # Asserting equality protects against a future refactor that wraps it
        # in a mutable container without updating the copy logic.
        assert b1["consumed_substeps"] == b2["consumed_substeps"] == 80, (
            f"consumed_substeps must round-trip; got b1={b1['consumed_substeps']} "
            f"b2={b2['consumed_substeps']}"
        )

        # Simulate robosuite mutating the dict it received from b1 — this
        # is what get_ep_meta does after set_ep_meta stores the reference.
        b1["ep_meta"]["lang"] = "INJECTED_BY_ROBOSUITE"
        b1["sim_state"][:] = 0.0
        # b2 — obtained BEFORE the mutation — must be untouched. And a
        # fresh third call must ALSO be untouched (cache must be clean).
        assert "lang" not in b2["ep_meta"], "b2 was mutated through b1"
        assert (b2["sim_state"] != 0).any(), "b2's sim_state was mutated through b1"
        b3 = col.get_bundle(str(npz_path))
        assert "lang" not in b3["ep_meta"], (
            "cache was corrupted by mutation through a prior returned bundle"
        )
        assert (b3["sim_state"] != 0).any(), (
            "cached sim_state was corrupted by mutation through a prior bundle"
        )
        assert b3["consumed_substeps"] == 80, (
            "consumed_substeps must survive mutations to other bundle fields"
        )
        print("  [PASS] consecutive calls return fresh copies; cache is mutation-safe")


if __name__ == "__main__":
    print("=== _load_init_bundle Tests ===\n")
    test_happy_path_keys_and_types()
    test_missing_sim_state_raises()
    test_missing_model_xml_or_ep_meta_raises()
    test_apply_scene_bundle_contract()
    test_consumed_substeps_extraction()
    test_ep_meta_must_be_dict()
    test_malformed_ep_meta_json_rejected()
    test_defensive_copy_consecutive_calls()
    print(
        f"\nAll _load_init_bundle tests PASSED "
        f"({'real impl' if _USING_REAL else 'inline fallback'})."
    )
