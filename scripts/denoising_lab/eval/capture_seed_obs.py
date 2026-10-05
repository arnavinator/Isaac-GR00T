"""Capture the first observation of a GRPO scene seed — sim-side, no policy server.

Runs in the **sim venv** (robocasa_uv/.venv). Rebuilds the scene the way the GRPO
collector does for a group seed (collect_episodes.py `_align_envs_to_group_scene`):
clear_ep_meta -> reset(seed) -> get_scene_bundle -> apply_scene_bundle, on the
collector's own env factory. The post-apply observation is the one every rollout of
that group starts from (with fast-forward off).

Writes it in the interactive_rollout.py npz layout, so
`DenoisingLab.load_observation` reads it and `--init-state-npz-path` accepts it.
The scene fingerprint is printed and stored in `__step_info__`; compare it with the
training log's `scene:` line for the same seed.

Usage::

    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \\
        scripts/denoising_lab/eval/capture_seed_obs.py \\
        --env-name robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \\
        --seed 103067 --out /tmp/denoising_lab_seed_obs/CoffeeServeMug_seed103067_step000.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "grpo"))

import collect_episodes as ce  # noqa: E402  (collector env factory + scene helpers)
from interactive_rollout import _NumpyEncoder  # noqa: E402  (same ep_meta JSON as the canonical saver)


def capture(env, seed: int) -> tuple[dict, dict]:
    """Collector's group alignment on one env. Returns (obs, pristine bundle)."""
    env.clear_ep_meta()
    env.reset(seed=seed)
    bundle = copy.deepcopy(env.get_scene_bundle())  # apply mutates ep_meta in place
    obs = env.apply_scene_bundle(copy.deepcopy(bundle))
    return obs, bundle


def save_npz(path: Path, obs: dict, bundle: dict, seed: int, n_action_steps: int) -> str:
    """Write obs (batched like the collector's server request) + bundle. Returns the fingerprint."""
    obs = ce._drop_unused_video_keys(obs, ce.DEFAULT_DROPPED_VIDEO_KEYS)
    save, meta = {}, {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            save[key] = val[np.newaxis]  # (1, T, ...)
        elif isinstance(val, (tuple, list)):
            meta[key] = "|".join(str(v) for v in val) if len(val) > 1 else str(val[0])
        else:
            meta[key] = str(val)
    fingerprint = ce.scene_fingerprint(bundle)
    if meta:
        save["__metadata__"] = np.array(json.dumps(meta), dtype=object)
    save["__sim_state__"] = np.asarray(bundle["sim_state"])
    save["__ep_meta__"] = np.array(json.dumps(bundle["ep_meta"], cls=_NumpyEncoder), dtype=object)
    save["__model_xml__"] = np.array(bundle["model_xml"], dtype=object)
    save["__step_info__"] = np.array(json.dumps({
        "episode": 0, "step": 0, "n_action_steps": n_action_steps, "seed": seed,
        "scene_fingerprint": fingerprint,
    }), dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")  # atomic: the notebook caches by path
    np.savez_compressed(str(tmp), **save)
    tmp.replace(path)
    return fingerprint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", required=True)
    p.add_argument("--seed", type=int, required=True, help="GRPO scene seed, e.g. 103067")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-action-steps", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=480)
    args = p.parse_args()

    env = ce._make_collector_env(args.env_name, 0, 1, args.n_action_steps, args.max_episode_steps)
    try:
        obs, bundle = capture(env, args.seed)
        fingerprint = save_npz(args.out, obs, bundle, args.seed, args.n_action_steps)
    finally:
        env.close()
    print(f"seed {args.seed} scene: {fingerprint}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
