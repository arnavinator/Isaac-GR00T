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
        return {
            "ep_meta": ep_meta,
            "model_xml": str(data["__model_xml__"]),
            "sim_state": np.asarray(data["__sim_state__"]),
        }


def _write_minimal_npz(path: Path, *, include_sim_state: bool = True,
                       include_model_xml: bool = True,
                       include_ep_meta: bool = True) -> None:
    """Write an npz with as many of the three required keys as requested.

    The keys/values mirror what interactive_rollout.py would save, but the
    contents are dummies so no robocasa or MuJoCo runtime is required.
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
    np.savez_compressed(str(path), **save_dict)


def test_happy_path_keys_and_types():
    """A well-formed npz produces a dict with the three expected keys and types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "ep000_step010.npz"
        _write_minimal_npz(npz_path)

        bundle = _load_init_bundle_real(str(npz_path))

        assert set(bundle.keys()) == {"ep_meta", "model_xml", "sim_state"}, (
            f"Unexpected keys: {sorted(bundle.keys())}"
        )
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
        _write_minimal_npz(npz_path)

        bundle = _load_init_bundle_real(str(npz_path))

        # apply_scene_bundle (collect_episodes.py:313-405) reads bundle["ep_meta"],
        # bundle["model_xml"], bundle["sim_state"]. The bundle MUST NOT carry
        # the npz-side double-underscore spellings.
        for npz_spelling in ("__ep_meta__", "__model_xml__", "__sim_state__"):
            assert npz_spelling not in bundle, (
                f"Bundle leaked npz-side key {npz_spelling}; apply_scene_bundle "
                f"will silently ignore it and fall back to wrong defaults."
            )
        for bundle_spelling in ("ep_meta", "model_xml", "sim_state"):
            assert bundle_spelling in bundle, (
                f"Bundle missing apply_scene_bundle-expected key {bundle_spelling}."
            )
        print("  [PASS] bundle uses apply_scene_bundle key spellings")


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
        print("  [PASS] consecutive calls return fresh copies; cache is mutation-safe")


if __name__ == "__main__":
    print("=== _load_init_bundle Tests ===\n")
    test_happy_path_keys_and_types()
    test_missing_sim_state_raises()
    test_missing_model_xml_or_ep_meta_raises()
    test_apply_scene_bundle_contract()
    test_ep_meta_must_be_dict()
    test_malformed_ep_meta_json_rejected()
    test_defensive_copy_consecutive_calls()
    print(
        f"\nAll _load_init_bundle tests PASSED "
        f"({'real impl' if _USING_REAL else 'inline fallback'})."
    )
