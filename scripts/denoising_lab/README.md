# Denoising Lab

Interactive experimentation with GR00T N1.6's 4-step flow-matching action denoising.

## What this does

GR00T N1.6 generates robot actions in two stages:

1. **Eagle VLM backbone** — encodes camera images + language instruction into vision-language embeddings. This is expensive (~1s on GPU).
2. **DiT action head** — runs a 4-step flow-matching denoising loop that refines Gaussian noise into an action chunk (e.g. 16 future EEF deltas). This is cheap (~50ms per run).

The Denoising Lab splits these two stages apart so you can run the backbone **once**, then experiment with the denoising loop **many times** — varying the number of steps, the random seed, the step size, or injecting custom guidance functions — and compare the resulting action trajectories side by side.

## Architecture

```
scripts/denoising_lab/
  __init__.py
  denoising_lab.py                       # Core library (model loading, open denoising loop, decoding, visualization)
  interactive_denoising_gr1.ipynb        # GR1 notebook — uses bundled demo dataset, no simulator needed
  interactive_denoising_panda.ipynb      # PandaOmron notebook — uses .npz observations from simulator
  interactive_rollout.py                 # Sim-side script (robocasa venv, CPU)
```

The toolkit enforces a strict two-venv separation, matching the existing server/client architecture:

| Component | Venv | GPU | What it does |
|-----------|------|-----|-------------|
| `denoising_lab.py` + notebooks | Main `.venv` | Yes | Model loading, backbone encoding, denoising experiments, visualization |
| `interactive_rollout.py` | Sim venv (`robocasa_uv/.venv`) | No | Runs the simulator, captures observations, executes actions |

The bridge between them is `.npz` files on disk — the rollout script saves observations, and the PandaOmron notebook loads them.

## Quick start

### GR1 notebook — bundled demo dataset (no simulator needed)

The repo ships `demo_data/gr1.PickNPlace`, a small LeRobot-format dataset of GR1 humanoid pick-and-place episodes (5 episodes, ~2k frames, 1 ego camera). No simulator, server, or external data download is required.

```bash
# From repo root
uv run jupyter lab scripts/denoising_lab/interactive_denoising_gr1.ipynb
```

Run cells 1–13 in order. Cell 3 loads a single observation from the demo dataset using the same `LeRobotEpisodeLoader` / `extract_step_data` pattern as `getting_started/GR00T_inference.ipynb`.

> **Note on GR1 trajectory plots:** GR1 uses joint-angle actions (not EEF deltas), so there is no `end_effector_position` key. The notebook plots the first 3 dimensions of the first action key (e.g. `right_arm`) as a proxy 3D trajectory. This is useful for comparing denoising strategies — the shape and spread of the curves show how different seeds/step counts affect the output — but the axes represent joint angles, not Cartesian position. For true EEF trajectory plots, use the PandaOmron notebook.

### PandaOmron notebook — live sim observations

This requires capturing observations from the simulator first, then loading them in the notebook.

**Step 1 — Start the model server** (main `.venv`, GPU):

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag ROBOCASA_PANDA_OMRON \
  --use-sim-policy-wrapper
```

**Step 2 — Run the interactive rollout** (sim venv):

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/denoising_lab/interactive_rollout.py \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --host 127.0.0.1 --port 5555 \
  --n-action-steps 8 --max-episode-steps 720 \
  --save-dir /tmp/saved_observations
```

In the interactive rollout, use the menu to step through the episode and save observations:

```
Step 0 | Reward so far: 0.00
Menu: [s]tep  [d]etails  [o]save-obs  [r]e-query  [q]uit
> o
Observation saved to: /tmp/saved_observations/ep000_step000.npz
> s
```

**Step 3 — Open the PandaOmron notebook** (main `.venv`, GPU):

```bash
code scripts/denoising_lab/interactive_denoising_panda.ipynb
```
Set the python kernel to be .venv/bin/python

Set `OBS_PATH` in cell 1 to point to your saved `.npz` file, then run all cells. The 3D plots show true EEF (end-effector) Cartesian trajectories. Cell 3b shows the camera views so you can see the scene.

### Replay — visualize action chunks in the simulator

After experimenting with denoising in the notebook, you can replay action chunks in the actual simulator to see the robot move:

**Step 1** — In the PandaOmron notebook, run cell 14 to export an action chunk:
```python
DenoisingLab.save_action_chunk(decoded, "/tmp/action_chunks/my_strategy.npz")
```

**Step 2** — Run the replay in the sim venv:
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/denoising_lab/interactive_rollout.py \
  --replay \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --obs-path /tmp/saved_observations/ep000_step001.npz \
  --action-path /tmp/action_chunks/my_strategy.npz \
  --video-out /tmp/replay.mp4
```

The replay restores the sim to the exact state when the observation was captured, applies the action chunk, and saves a video of the result. You can compare different denoising strategies by replaying each and examining the videos.

### Video recording during rollouts

Add `--video-dir` to record full episode videos during interactive rollouts:

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/denoising_lab/interactive_rollout.py \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --host 127.0.0.1 --port 5555 \
  --n-action-steps 8 --max-episode-steps 720 \
  --save-dir /tmp/saved_observations \
  --video-dir /tmp/rollout_videos
```

## API reference

### `DenoisingLab`

```python
lab = DenoisingLab(model_path, embodiment_tag, device="cuda")
```

**`encode_features(observation) -> BackboneFeatures`**

Runs the Eagle VLM backbone + state encoder. Takes a nested observation dict (same format as `Gr00tPolicy._get_action()`). This is the expensive call — run it once per observation, then call `denoise()` many times with the cached features.

**`encode_features_from_sim_obs(sim_obs) -> BackboneFeatures`**

Same as above, but takes the flat sim observation format (keys like `video.cam`, `state.joints`) that `interactive_rollout.py` saves. Handles the flat-to-nested conversion internally.

**`denoise(features, *, num_steps=4, dt=None, seed=None, initial_noise=None, guided_fn=None, step_callback=None) -> DenoiseResult`**

The core experimentation method. Runs the flow-matching denoising loop with full user control:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 4 | Number of Euler integration steps |
| `dt` | `1/num_steps` | Euler step size |
| `seed` | None | Random seed for reproducible initial noise |
| `initial_noise` | None | Custom starting noise tensor `(B, 50, 128)` |
| `guided_fn` | None | `(actions, step_idx, velocity) -> modified_velocity` — inject guidance before each Euler update |
| `step_callback` | None | `(step_idx, actions, velocity) -> None` — inspect at each step |

Returns a `DenoiseResult` containing the final raw actions, all intermediate step info, and the initial noise used.

**`denoise_single_step(features, actions, step_idx, num_total_steps=4) -> (velocity, updated_actions)`**

Runs a single denoising step. Use this for cell-by-cell manual control in the notebook.

**`decode_raw_actions(action_pred, states=None) -> dict[str, np.ndarray]`**

Converts raw model output `(B, 50, 128)` to per-key physical units (e.g. `{"end_effector_position": (B, 16, 3), ...}`). Applies denormalization and per-embodiment slicing.

**`label_action_step(decoded_actions, step_idx) -> str`**

Returns a human-readable string showing all action dimensions for one sub-step.

**`save_observation(observation, path)`** / **`load_observation(path)`** (static methods)

Save/load observation dicts as compressed `.npz` files. The bridge between the sim venv and model venv.

**`save_action_chunk(decoded_actions, path)`** / **`load_action_chunk(path)`** (static methods)

Save/load decoded action chunks for replay in the simulator.

**`plot_camera_views(observation, figsize=None) -> Figure`** (static method)

Display all camera images from an observation side-by-side. Works with both flat (sim) and nested observation formats. The PandaOmron notebook uses this to show what the robot sees at the saved observation, making it easier to identify interesting states to experiment with.

### `TrajectoryVisualizer`

Accumulates EEF trajectories from different denoising runs for comparison plotting.

```python
viz = TrajectoryVisualizer()
viz.add_trajectory(decoded_actions, "label", eef_key="end_effector_position")
viz.add_from_denoise_result(result, lab, "label")

viz.plot_eef_3d()                               # 3D trajectory plot
viz.plot_eef_3d(show_orientation=True)          # + mini RGB coordinate frames (R=X, G=Y, B=Z)
viz.plot_eef_3d(show_gripper=True)              # + green→red markers for gripper state
viz.plot_eef_3d(show_orientation=True,          # both at once
                show_gripper=True,
                frame_stride=2)                 # orientation frames every 2nd timestep
viz.plot_eef_components()                       # X/Y/Z over timesteps
viz.plot_denoising_progression(result, lab)      # trajectory at each denoising step
```

`add_trajectory()` automatically stores rotation deltas and gripper values from the decoded actions dict (when present) for use by the orientation/gripper visualization. The defaults (`rotation_key="end_effector_rotation"`, `gripper_key="gripper_close"`) match PandaOmron. For embodiments without these keys (e.g. GR1 joint-angle actions), the data is simply `None` and the orientation/gripper toggles are silently ignored.

### `compare_strategies()`

Convenience function to run multiple denoising configurations and get back results + a populated visualizer:

```python
strategies = [
    {"num_steps": 2, "seed": 42},
    {"num_steps": 4, "seed": 42},
    {"num_steps": 8, "seed": 42},
]
results, viz = compare_strategies(lab, features, strategies, ["2-step", "4-step", "8-step"])
viz.plot_eef_3d()
```

### `InteractiveRollout`

Sim-side class for step-by-step environment control. Runs in the sim venv only.

```python
runner = InteractiveRollout(
    env_name="robocasa_panda_omron/OpenDrawer_PandaOmron_Env",
    host="127.0.0.1", port=5555,
    n_action_steps=8, max_episode_steps=720,
    save_dir="/tmp/saved_observations",
    video_dir="/tmp/rollout_videos",  # optional — records episode video
)
runner.run_episode()
```

The interactive menu at each step:

| Key | Action |
|-----|--------|
| `s` | Execute the action chunk in the environment |
| `d` | Print per-sub-step action dimensions and values |
| `o` | Save current observation + sim state to `.npz`, plus a camera snapshot `.png` |
| `r` | Discard action, re-query the server for a new one |
| `q` | End the episode |

When saving (`[o]`), two files are written:
- `ep000_step003.npz` — full observation with state, camera images, language, and MuJoCo sim state
- `ep000_step003.png` — camera montage image for quick visual reference

The MuJoCo sim state embedded in the `.npz` enables **replay** of action chunks from the notebook.

### `ReplayRollout`

Replays an action chunk exported from the notebook in the simulator, recording a video. This lets you try different denoising strategies in the notebook and see their effect in the actual environment.

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
  scripts/denoising_lab/interactive_rollout.py \
  --replay \
  --env-name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
  --obs-path /tmp/saved_observations/ep000_step001.npz \
  --action-path /tmp/action_chunks/default_seed42.npz \
  --video-out /tmp/replay_default.mp4
```

The replay:
1. Creates the environment and resets it
2. Restores the MuJoCo sim state from the saved observation `.npz`
3. Steps through each sub-step of the action chunk
4. Records all camera frames and saves an `.mp4` video

## Notebook cells

Both notebooks share the same cell structure. The PandaOmron notebook has two extra cells (3b and 14):

| Cell | Purpose |
|------|---------|
| 1 | Imports + configuration (model path, embodiment tag, device) |
| 2 | Load model into `DenoisingLab` |
| 3 | Load observation (GR1: from demo dataset; Panda: from `.npz` file) |
| 3b | *(Panda only)* Visualize camera views from the saved observation |
| 4 | Encode backbone features (expensive — run once) |
| 5 | Default 4-step denoising with step-by-step metrics |
| 6 | Decode to physical units + human-readable inspection |
| 7 | 3D trajectory plot (GR1: joint-angle proxy; Panda: true EEF Cartesian) |
| 7b | *(Panda only)* 3D trajectory with orientation frames (RGB=XYZ) and gripper state |
| 8 | Compare 5 random seeds on the same observation |
| 9 | Compare 2/4/8/16 denoising steps |
| 10 | Manual step-by-step denoising via `denoise_single_step()` |
| 11 | Guided denoising (velocity scaling example) |
| 12 | Raw denoising loop — fully editable playground mode |
| 13 | Denoising progression visualization (noise → final trajectory) |
| 14 | *(Panda only)* Export action chunk for replay in the simulator |

## How the denoising loop works

GR00T N1.6 uses **flow matching** (not DDPM/DDIM). The denoising process is a simple Euler ODE integration:

```
actions = randn(B, 50, 128)           # pure Gaussian noise
dt = 1.0 / num_steps                   # step size (default: 0.25)

for t in range(num_steps):              # default: 4 steps
    t_cont = t / num_steps              # 0.0, 0.25, 0.5, 0.75
    t_disc = int(t_cont * 1000)         # 0, 250, 500, 750

    velocity = DiT(actions, t_disc, vl_embeddings, state_features)
    actions = actions + dt * velocity   # Euler step
```

The DiT (Diffusion Transformer) is a 32-layer, ~1.5B parameter transformer that takes the noised actions, the timestep, and the vision-language embeddings, and predicts a velocity field. The Euler integration follows this velocity field from noise to a clean action chunk.

The raw output is `(B, 50, 128)` — padded to accommodate multiple embodiments. After denoising, `decode_action()` slices to the relevant dimensions for the target embodiment (e.g. Panda → `(B, 16, 29)`), splits into per-key arrays, and denormalizes to physical units.

## Demo data

The repo bundles two demo datasets in `demo_data/`:

| Dataset | Embodiment | Episodes | Frames | Action type |
|---------|-----------|----------|--------|------------|
| `gr1.PickNPlace` | GR1 humanoid | 5 | ~2k | Joint angles (relative) |
| `cube_to_bowl_5` | so101 follower | 5 | ~4k | Joint positions |

The GR1 notebook (`interactive_denoising_gr1.ipynb`) defaults to `gr1.PickNPlace` because GR1 is a pretrained embodiment (id=20) with normalization statistics in the checkpoint. The `cube_to_bowl_5` dataset uses an `so101_follower` robot which is not a pretrained embodiment.

There is no bundled `ROBOCASA_PANDA_OMRON` demo dataset. The PandaOmron notebook (`interactive_denoising_panda.ipynb`) requires `.npz` observations captured from the simulator via `interactive_rollout.py`.

## Supported embodiments

The toolkit works with any embodiment tag supported by GR00T N1.6:

| Tag | Robot | Cameras | Action Horizon |
|-----|-------|---------|---------------|
| `ROBOCASA_PANDA_OMRON` | Panda arm + Omron mobile base | 3 (2 side + 1 wrist) | 16 steps |
| `GR1` | Fourier GR1 humanoid | 1 ego camera | 16 steps |
| `BEHAVIOR_R1_PRO` | Galaxea R1 Pro (Isaac Sim) | varies | 32 steps |
| `LIBERO_PANDA` | Panda (LIBERO benchmark) | varies | 16 steps |

Choose the notebook matching your embodiment, or use the core `DenoisingLab` API directly with any supported tag and an appropriate observation.
