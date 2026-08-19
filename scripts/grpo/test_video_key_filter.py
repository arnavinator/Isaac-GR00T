"""Tests for the unused-video-key filter in collect_episodes.py.

RoboCasa's GrootRoboCasaEnv emits full-resolution passthrough copies alongside
the keys the model consumes — `video.res512_image_*` next to every
`video.res256_image_*` (base copy) and `video.ego_view_res1280x800_freq20`
(GR1). Nothing in scripts/grpo, gr00t/eval or gr00t/data reads them, and they are
~80% of the per-chunk video bytes, so the collector drops them before an
observation reaches the policy server or an episode .npz.

Runs without robosuite, MuJoCo, or a GPU:

    .venv/bin/python scripts/grpo/test_video_key_filter.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collect_episodes import (  # noqa: E402
    DEFAULT_DROPPED_VIDEO_KEYS,
    EpisodeCollector,
    _drop_unused_video_keys,
)

# What GrootRoboCasaEnv (base robocasa copy) emits for PandaOmron.
PANDA_OBS_KEYS = [
    "video.res256_image_side_0",
    "video.res512_image_side_0",
    "video.res256_image_side_1",
    "video.res512_image_side_1",
    "video.res256_image_wrist_0",
    "video.res512_image_wrist_0",
    "state.gripper_qpos",
    "state.base_position",
    "annotation.human.action.task_description",
]
MODEL_VIDEO_KEYS = [
    "video.res256_image_side_0",
    "video.res256_image_side_1",
    "video.res256_image_wrist_0",
]


def _panda_obs():
    obs = {}
    for k in PANDA_OBS_KEYS:
        if k.startswith("video.res256"):
            obs[k] = np.full((256, 256, 3), 7, dtype=np.uint8)
        elif k.startswith("video.res512"):
            obs[k] = np.full((512, 512, 3), 9, dtype=np.uint8)
        elif k.startswith("state."):
            obs[k] = np.zeros(3, dtype=np.float32)
        else:
            obs[k] = "do the thing"
    return obs


def _fake_collector(dropped=DEFAULT_DROPPED_VIDEO_KEYS):
    """An EpisodeCollector with only the fields the filter paths touch —
    __init__ would build MuJoCo envs."""
    c = object.__new__(EpisodeCollector)
    c.dropped_video_keys = tuple(dropped)
    c.env_name = "robocasa_panda_omron/Fake_Env"
    return c


def test_drops_exactly_the_passthrough_copies():
    kept = _drop_unused_video_keys(_panda_obs(), DEFAULT_DROPPED_VIDEO_KEYS)
    video = sorted(k for k in kept if k.startswith("video."))
    assert video == sorted(MODEL_VIDEO_KEYS), video
    # Non-video keys must be untouched.
    assert "state.gripper_qpos" in kept and "state.base_position" in kept
    assert "annotation.human.action.task_description" in kept
    print("  [PASS] drops res512_* only, leaves state/annotation keys alone")


def test_drops_gr1_full_res_passthrough():
    obs = {
        "video.ego_view_pad_res256_freq20": np.zeros((256, 256, 3), np.uint8),
        "video.ego_view_bg_crop_pad_res256_freq20": np.zeros((256, 256, 3), np.uint8),
        "video.ego_view_res1280x800_freq20": np.zeros((800, 1280, 3), np.uint8),
        "state.left_arm": np.zeros(7, np.float32),
    }
    kept = _drop_unused_video_keys(obs, DEFAULT_DROPPED_VIDEO_KEYS)
    assert "video.ego_view_res1280x800_freq20" not in kept
    # The cotrain crop IS a configured modality for some GR1 checkpoints — it
    # must survive.
    assert "video.ego_view_bg_crop_pad_res256_freq20" in kept
    assert "video.ego_view_pad_res256_freq20" in kept
    print("  [PASS] drops GR1 1280x800 passthrough, keeps the cotrain crop")


def test_empty_drop_list_is_a_no_op_and_returns_the_same_dict():
    obs = _panda_obs()
    assert _drop_unused_video_keys(obs, ()) is obs
    # Also identity when nothing matches, so the common path allocates nothing.
    lean = {k: v for k, v in obs.items() if "res512" not in k}
    assert _drop_unused_video_keys(lean, DEFAULT_DROPPED_VIDEO_KEYS) is lean
    print("  [PASS] no-op paths return the original dict (no copy)")


def test_server_payload_excludes_the_dropped_keys():
    """_batch_per_env_obs is the only path by which obs reach the policy server."""
    c = _fake_collector()
    batched = c._batch_per_env_obs([_panda_obs(), _panda_obs()])
    video = sorted(k for k in batched if k.startswith("video."))
    assert video == sorted(MODEL_VIDEO_KEYS), video
    for k in MODEL_VIDEO_KEYS:
        assert batched[k].shape == (2, 256, 256, 3), (k, batched[k].shape)
    # Text obs must still batch as a tuple-of-strings for the server.
    assert batched["annotation.human.action.task_description"] == (
        "do the thing",
        "do the thing",
    )
    print("  [PASS] server payload carries only the model's video keys")


def test_saved_frames_exclude_the_dropped_keys():
    """_extract_video_single is the only path by which frames reach an .npz."""
    c = _fake_collector()
    frames = c._extract_video_single(_panda_obs())
    # Stored keys have the 'video.' prefix stripped.
    assert sorted(frames) == sorted(k.removeprefix("video.") for k in MODEL_VIDEO_KEYS)
    assert all(v.shape == (256, 256, 3) for v in frames.values())
    print("  [PASS] npz frames carry only the model's video keys")


def test_payload_reduction_is_what_we_claim():
    obs = _panda_obs()
    c = _fake_collector()
    before = sum(v.nbytes for v in obs.values() if isinstance(v, np.ndarray))
    after = sum(v.nbytes for v in c._extract_video_single(obs).values())
    full = sum(
        v.nbytes
        for k, v in obs.items()
        if isinstance(v, np.ndarray) and k.startswith("video.")
    )
    saved = 1 - after / full
    assert saved > 0.75, f"expected >75% of video bytes dropped, got {saved:.0%}"
    print(
        f"  [PASS] video payload {full / 1e6:.2f} MB → {after / 1e6:.2f} MB "
        f"({saved:.0%} smaller; whole obs was {before / 1e6:.2f} MB)"
    )


def test_keeping_everything_restores_the_old_behavior():
    c = _fake_collector(dropped=())
    frames = c._extract_video_single(_panda_obs())
    assert len(frames) == 6, sorted(frames)
    batched = c._batch_per_env_obs([_panda_obs()])
    assert len([k for k in batched if k.startswith("video.")]) == 6
    print("  [PASS] empty drop list reproduces the pre-filter behavior")


def test_trainer_passes_the_config_list_to_the_collector():
    """The config value must reach the subprocess verbatim — a drift between the
    config default and the collector default would silently change what the
    policy sees and what gets recorded."""
    import subprocess as sp
    import tempfile

    from grpo_config import GRPOConfig
    from train_grpo import GRPOTrainer

    from collect_episodes import parse_args

    for configured in (["res512", "ego_view_res1280x800"], ["res512"], []):
        trainer = object.__new__(GRPOTrainer)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.config = GRPOConfig(
                use_wandb=False, episode_dir=tmp, dropped_video_keys=list(configured)
            )
            trainer.iteration = 1
            captured = {}

            def fake_popen(cmd, **kwargs):
                captured["cmd"] = cmd
                raise RuntimeError("stop — argv only")

            real = sp.Popen
            sp.Popen = fake_popen
            try:
                trainer._collect_via_subprocess(
                    "robocasa_panda_omron/Fake_Env", Path(tmp), 480, 0
                )
            except RuntimeError:
                pass
            finally:
                sp.Popen = real

            cmd = captured["cmd"]
            i = cmd.index("--dropped-video-keys")
            passed = []
            for tok in cmd[i + 1:]:
                if tok.startswith("--"):
                    break
                passed.append(tok)
            assert passed == configured, f"{configured} -> argv {passed}"

            # And the collector must parse that argv back to the same list.
            argv = sys.argv
            try:
                sys.argv = [
                    "collect_episodes.py", "--env-name", "e", "--output-dir", tmp
                ] + cmd[i:i + 1 + len(configured)]
                assert parse_args().dropped_video_keys == configured
            finally:
                sys.argv = argv
    print("  [PASS] config -> argv -> collector round-trips (incl. empty list)")


TESTS = [
    test_drops_exactly_the_passthrough_copies,
    test_drops_gr1_full_res_passthrough,
    test_empty_drop_list_is_a_no_op_and_returns_the_same_dict,
    test_server_payload_excludes_the_dropped_keys,
    test_saved_frames_exclude_the_dropped_keys,
    test_payload_reduction_is_what_we_claim,
    test_keeping_everything_restores_the_old_behavior,
    test_trainer_passes_the_config_list_to_the_collector,
]


if __name__ == "__main__":
    print("=== unused-video-key filter tests ===\n")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\nAll {len(TESTS)} tests PASSED.")
