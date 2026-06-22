# GRPO Finetuning for GR00T N1.6

Group Relative Policy Optimization (GRPO) with LoRA adapters for online RL
finetuning of the GR00T N1.6 DiT action head on RoboCasa manipulation tasks.

The pretrained GR00T model is trained with offline imitation learning. This
package layers an episodic RL loop on top: collect group rollouts in
simulation, compute group-relative advantages on the resulting rewards, and
update only LoRA adapters on the DiT via a clipped surrogate objective with a
Flow-Matching (FM) log-probability surrogate.

---

## Contents

| File | Purpose |
|------|---------|
| `train_grpo.py` | Main orchestrator (`GRPOTrainer`): model+LoRA setup, iter loop, ref log-probs, GRPO update, checkpointing. |
| `grpo_config.py` | `GRPOConfig` dataclass — every tunable knob lives here. |
| `grpo_server.py` | Extends `PolicyServer` to capture per-call denoising noise + raw `(B, 50, 128)` action. Required for FM log-prob. |
| `collect_episodes.py` | Runs in the robocasa venv. `EpisodeCollector` does group rollouts via `AsyncVectorEnv`, including fast-forward branching and scene-bundle alignment. |
| `collector_server.py` | Long-running ZMQ collector daemon. Skips per-iter robocasa import + worker-spawn cost (~10-20s/iter). |
| `episode_buffer.py` | `EpisodeBuffer`, `GRPOEpisode`, `ActionChunk`. Loads `.npz` episodes, computes group-relative advantages. |
| `fm_log_prob.py` | FM-loss-as-log-prob surrogate (`compute_fm_log_prob`), jittered timestep sampler (`_sample_jittered_timesteps`). |
| `lora_dit.py` | `apply_lora_to_dit`, `save_lora_checkpoint`, `load_lora_checkpoint`, default target-module list. |
| `dense_reward.py` | Per-task continuous progress extraction (drawers, doors, PnP, stove, microwave). Used when `success_weight < 1.0`. |
| `eval_lora_from_npz.py` | Eval harness: runs N parallel rollouts of a LoRA policy from a saved `interactive_rollout.py` `.npz`, aggregates per-attempt success/num_steps into `results.json`. Subclasses `EpisodeCollector` in init-state mode. |
| `test_*.py` | Sanity checks for sim-wrapper / `.npz` key roundtrip. |

---

## Architecture

Three processes share the work:

```
┌─────────────────────────┐   ZMQ obs/action    ┌──────────────────────────┐
│ Trainer (main .venv)    │ ◄──────────────────►│ Collector (robocasa venv)│
│  GPU model + LoRA       │  port 5555          │  AsyncVectorEnv workers  │
│  In-process PolicyServer│                     │  Writes .npz per iter    │
└────────────┬────────────┘                     └──────────────────────────┘
             │ optional ZMQ RPC (port 5556)
             ▼
┌──────────────────────────┐
│ collector_server.py      │  long-running collector for the trainer to call
│ (robocasa venv)          │  (skips per-iter startup)
└──────────────────────────┘
```

The trainer spawns the policy server **in a background thread** of its own
process, so the LoRA weights it updates are immediately visible to the next
collection round — no checkpoint shuffling. A re-entrant lock
(`self._model_lock`) serializes forward and backward passes between the
server thread (inference for the collector) and the main thread (ref
log-probs / GRPO update).

The collector runs in a separate venv because robocasa depends on MuJoCo
and gym wrappers that don't coexist cleanly with the main training stack.

---

## Quick Start

### 1. Subprocess mode (simpler, ~10-20s/iter startup tax)

Just run the trainer; it spawns one `collect_episodes.py` subprocess per
iteration:

```bash
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200 \
    --group-size 4 --num-groups 5 \
    --checkpoint-dir grpo_data/grpo_checkpoints
```

### 2. Long-running collector mode (recommended for multi-task runs)

**Terminal 1** — start the collector daemon in the robocasa venv. It will
boot one `EpisodeCollector` per `--env-names` entry, paying the import +
worker-spawn cost once.

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/grpo/collector_server.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
                robocasa_panda_omron/OpenDoubleDoor_PandaOmron_Env \
    --max-episode-steps 480 480 \
    --group-size 4 --n-action-steps 8 \
    --policy-server-host 127.0.0.1 --policy-server-port 5555 \
    --listen-port 5556
```

**Terminal 2** — start the trainer, pointing it at the collector daemon:

```bash
uv run python scripts/grpo/train_grpo.py \
    --collector-server-host 127.0.0.1 --collector-server-port 5556
```

The trainer pings the collector at startup and **fails fast** if the
daemon's bake-time args (`--group-size`, `--n-action-steps`, per-env
`--max-episode-steps`, env-name set) don't match its own config. Restart
the daemon with matching flags if it does.

### 3. Standalone server (debug / eval only)

`scripts/grpo/grpo_server.py` is the standalone variant of the in-process
policy server. Use it to serve a trained LoRA checkpoint without spinning up
the trainer:

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --lora-rank 16 --lora-alpha 32 \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --port 5555
```

Do NOT use `gr00t/eval/run_gr00t_server.py` for GRPO collection — it
does not install the noise/raw-action capture hooks that `_prepare_batch`
requires.

---

## Loading a Trained LoRA Checkpoint for Inference

Each `iter_NNNN/` checkpoint dir contains:

```
iter_NNNN/
  lora_weights.pt   # filtered LoRA-only state dict (~80 MB at rank=16)
  optimizer.pt      # only needed for resuming training; ignored for inference
```

There are two supported inference paths: a **server-client benchmark** (drop
into the existing denoising-lab eval pipeline) and an **in-process notebook**
(direct `DenoisingLab` API for trajectory experimentation).

### Reproducible benchmark via `robocasa_eval_benchmark.py`

`scripts/denoising_lab/eval/robocasa_eval_benchmark.py` is strategy-agnostic —
it just connects to whatever ZMQ server is running on `--port`. So the only
thing that changes for a LoRA strategy is the **server**: instead of
`gr00t/eval/run_gr00t_server.py` (baseline), use `grpo_server.py`, which
already supports loading a LoRA checkpoint via `--lora-checkpoint`.

**Terminal 1 — model venv, GRPO server with LoRA:**

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --use-sim-policy-wrapper \
    --port 5555 \
    --verbose
```

**Terminal 2 — sim venv, identical to baseline_euler eval:**

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --n-episodes 15 --seed 42 --n-envs 2 --port 5555 \
    --max-episode-steps 480 \
    --output-dir ~/benchmark_results/grpo_iter_0100 \
    --strategy-name grpo_iter_0100
```

Use whatever env(s) the LoRA was trained on (see `GRPOConfig.env_names`) —
benchmarking on tasks the policy never saw will mostly measure the base
model. Override `--lora-rank` / `--lora-alpha` / `--lora-target-modules` on
the server command **only** if you trained with non-default values; mismatch
hard-fails inside `load_lora_checkpoint` (`lora_dit.py:165-185`) rather than
silently degrading.

`grpo_server.py` does not track gradients during inference. The `Gr00tPolicy`
forward pass runs inside `torch.inference_mode()`
(`gr00t/policy/gr00t_policy.py:347`), so the `requires_grad=True` flag that
PEFT sets on the LoRA params is a no-op — no autograd graph is built and the
extra cost beyond the baseline server is just the LoRA matmuls themselves.

### Interactive notebook via `DenoisingLab`

For the trajectory-fan / seed-sweep experiments in
`scripts/denoising_lab/notebooks/`, inject the LoRA into the existing
`DenoisingLab` after it loads the base model. See
`scripts/denoising_lab/notebooks/interactive_denoising_panda_lora_v1.ipynb`
for a working copy of `interactive_denoising_panda_v2.ipynb` with the
injection cell pre-wired. The full pattern:

```python
# After: lab = DenoisingLab(MODEL_PATH, EMBODIMENT_TAG, device=DEVICE)
import sys, os
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "grpo"))
from lora_dit import apply_lora_to_dit, load_lora_checkpoint

apply_lora_to_dit(lab.model, rank=16, alpha=32, dropout=0.0)
load_lora_checkpoint(lab.model, "grpo_data/grpo_checkpoints/iter_0100")
# Pin freshly-injected LoRA Linears to the DiT's device/dtype:
lab.model.action_head.model.to(device=lab.device, dtype=lab.dtype)
```

Caveats:

- **`.to(device=lab.device, dtype=lab.dtype)` is required.** PEFT's
  `inject_adapter_in_model` creates the new Linear submodules at default
  device/dtype; without the cast, the first `lab.denoise(...)` call hits a
  cross-device or cross-dtype error.
- **The `lab.action_head` reference set in `DenoisingLab.__init__` is
  unchanged** — LoRA injection mutates the same `model.action_head.model`
  object in place, so subsequent `lab.encode_features_from_sim_obs(...)` /
  `lab.denoise(...)` calls automatically route through the trained adapters.
- **LoRA only touches the DiT, not the Eagle backbone.** A `BackboneFeatures`
  cached from a base-model run remains valid input to a LoRA `denoise`, and
  vice versa — useful for A/B comparing the same observation through both
  policies.
- **For A/B comparisons**, build a second `DenoisingLab` instance for the
  base model rather than trying to "uninject" LoRA — `merge_lora_weights`
  (`lora_dit.py:205`) is irreversible and there is no `unmerge` helper.

### Parallel evaluation from a saved sim state via `eval_lora_from_npz.py`

`scripts/grpo/eval_lora_from_npz.py` is the eval-side counterpart to the
"Init from saved sim state" training mode (covered later in this README): it
loads the same `interactive_rollout.py` `.npz` (`__sim_state__`,
`__model_xml__`, `__ep_meta__`, optional `__step_info__`) and runs
`--num-attempts` parallel rollouts, all starting bit-identically from that
state. Use it to measure how often a LoRA succeeds from a specific
intermediate state and at what speed — complementary to
`robocasa_eval_benchmark.py`, which measures end-to-end performance from
fresh randomized scenes.

Within-attempt diversity comes from the server's unseeded `torch.randn`
during denoising, NOT from env randomness. AsyncVectorEnv subprocess
workers parallelize the MuJoCo cost: with `--num-envs W < --num-attempts N`,
the script collects N rollouts over `N // W` sequential turns of W rollouts
each (mirroring `num_async_vector_env` in training).

**Terminal 1 — model venv, GRPO server with the LoRA loaded:**

```bash
uv run python scripts/grpo/grpo_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100 \
    --use-sim-policy-wrapper --port 5555
```

**Terminal 2 — sim venv:**

```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/grpo/eval_lora_from_npz.py \
    --env-name robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --obs-path /tmp/saved_observations/ep000_step010.npz \
    --num-attempts 100 --num-envs 10 \
    --max-episode-steps 480 --n-action-steps 8 \
    --output-dir /tmp/eval_iter_0100 \
    --lora-checkpoint grpo_data/grpo_checkpoints/iter_0100
```

The script writes `results.json` to `--output-dir`:

```json
{
  "lineage": {
    "obs_path": "...", "lora_checkpoint": "...",
    "branch_step": 10, "saved_n_action_steps": 8,
    "consumed_substeps": 80, "remaining_substeps_budget": 400,
    "seed": 42, "timestamp": "...", "duration_s": 432.5,
    "...": "..."
  },
  "summary": {
    "total": 100, "successes": 47, "success_rate": 0.47,
    "mean_num_steps_all": 234.5,
    "mean_num_steps_successful": 156.2,
    "mean_num_steps_failed": 314.6
  },
  "attempts": [{"attempt_idx": 0, "success": true, "num_steps": 142,
                "termination": "success"}, "..."]
}
```

Constraints and caveats:

- **`--num-attempts` must be divisible by `--num-envs`** (the script
  reuses `EpisodeCollector`'s `group_size % num_async_vector_env == 0`
  invariant). The error message lists divisors of the chosen
  `--num-attempts` so you can adjust either knob.
- **`--lora-checkpoint` is metadata only.** The script records the path
  in `results.json` but does NOT load weights itself — the server in
  Terminal 1 is responsible. Mismatch (server running base model or a
  different LoRA than the path you record) cannot be detected
  client-side; verify the server's startup log shows the expected
  checkpoint path before running.
- **Pre-spawn ping fails fast on server-down.** Before paying the
  ~10-20 s robocasa import + AsyncVectorEnv worker spawn cost, the
  script pings the GRPO server with explicit ZMQ `RCVTIMEO`/`SNDTIMEO`
  (5 s budget). If Terminal 1 isn't running, you get a
  `ConnectionError` with a corrected start command, not a 20 s wait
  followed by a hang inside the first `get_action`.
- **No video / image / per-step observation saving.** The
  `EvalCollector` subclass overrides `_extract_video_single` /
  `_extract_state_single` / `_get_actions_from_server` to drop those
  recordings (~12 GB + ~460 MB savings on a 100-attempt Panda run). If
  you want per-step inspection, use `branching_rollout.py` for
  single-trajectory analysis instead.
- **`consumed_substeps` accounting is correct across an
  `n_action_steps` change.** The .npz's saved `n_action_steps` (in
  `__step_info__`) drives `consumed_substeps`, not the eval-time
  `--n-action-steps`, so a chunk-size change between save and replay
  doesn't break budget bookkeeping.

---

## Episode Collection

### Groups, seeds, and within-group variance

- A **group** is `group_size` rollouts started from an identical initial
  state (same RoboCasa seed → same kitchen layout, same object poses).
- Within-group diversity comes **only from policy denoising noise**
  (`torch.randn` inside the DiT). The env contributes zero variance once
  the scene is aligned.
- Different groups use seeds `base_seed + g * 1000` (`GROUP_SEED_STRIDE` in
  `collect_episodes.py`), wide-spaced so consecutive groups land on
  visually-distinct kitchens.
- The trainer's per-iter seed stride is 100,000
  (`config.seed + iteration * 100_000`), so two consecutive iters' group
  ranges never collide. This caps `max_groups` at 100.

### Multi-env / multi-task support

Pass multiple env names in `config.env_names`. The trainer **round-robins**
tasks: iteration 1 → task 0, iteration 2 → task 1, etc. Each iteration
collects ALL `num_groups` for a SINGLE task; tasks are never mixed within
a group (group-relative normalization only makes sense among rollouts of
the same task with the same initial scene).

Per-task tuning:

- `max_episode_steps: int | list[int]` — single value applied to every env,
  or a list parallel to `env_names`.
- `fast_forward_steps: int | list[int]` — same convention.

With 8 tasks × 200 iters, each task gets 25 updates. The
`collector_server` validates that its boot-time env list matches the
trainer's `env_names` and raises on mismatch.

### AsyncVectorEnv + scene-bundle alignment

`num_async_vector_env > 1` uses `gym.vector.AsyncVectorEnv` (subprocess
workers, parallel MuJoCo); `== 1` uses `SyncVectorEnv` (no IPC). RoboCasa
picks layout/textures at env construction via a per-instance RNG, so
identically-seeded parallel workers still render **different** scenes.
`GroupAlignmentWrapper` (`collect_episodes.py`) exposes composite RPCs
(`get_scene_bundle`, `apply_scene_bundle`) that the parent invokes via
`env.call()` to copy env-0's scene XML + flat MuJoCo state to all other
workers. After alignment, every env in the group is bit-identical
(verifiable via `--debug-fast-forward`).

### Decoupling group size from worker count (`num_async_vector_env`)

`group_size` is the **logical** number of rollouts per group;
`num_async_vector_env` is the **physical** number of parallel sim workers.
By default (`None`) they're equal — one worker per rollout, unchanged from
before this knob existed. Set `num_async_vector_env < group_size` to cap
peak worker RAM (each MuJoCo worker is ~5 GiB) on RAM-limited hosts: a group
is then collected over `k = group_size // num_async_vector_env` sequential
**turns** of `num_async_vector_env` rollouts each.

- **Constraint:** `1 <= num_async_vector_env <= group_size` and
  `group_size % num_async_vector_env == 0` (validated in
  `GRPOConfig.__post_init__`; non-divisor or `> group_size` raises). Going
  *above* `group_size` (packing multiple groups into one batch) is out of
  scope.
- Turn 1 establishes and **captures** the branch-point bundle; turns 2..k
  re-apply it (`apply_scene_bundle`) so every turn restarts from the
  bit-identical state. All `group_size` rollouts share one `group_id`, so
  group-relative advantage normalization is unaffected — it sees one group
  of `group_size`, not `k` smaller groups.
- Diversity across turns is genuine: the server's denoising noise
  (`torch.randn`) is unseeded, so each turn's fresh query yields distinct
  rollouts even from the identical initial state.
- **Cost:** ~`k`× collection wall time per group (turns are sequential). The
  trainer scales its subprocess / RPC collector timeouts by `k`
  automatically.
- Bake-time, like `group_size`: pass `--num-async-vector-env` to
  `collector_server.py` so it matches the trainer; the ping check fails fast
  with a restart hint on mismatch.

### Fast-Forward Branching

Tasks like "open the right drawer" spend most of an episode on the
**approach phase** (navigate, position the gripper). That phase carries
relatively little reward signal compared to the **manipulation phase**
(grasp + pull). Fast-forward focuses GRPO signal on the latter:

```
   t=0 ──────► t=FF (rollout one env, save MuJoCo state)
                    │
                    │ apply_scene_bundle to all G envs
                    ▼
   t=FF ──── independent rollouts ──── t=done
```

1. After scene-bundle alignment, **one env** runs solo for
   `fast_forward_steps` outer steps (each outer step = `n_action_steps`
   sub-steps + one model query).
2. The env's MuJoCo state is captured and pushed to all G envs via the
   same scene-bundle RPC.
3. All G envs continue independently from that state; within-group
   variance comes from the post-branch denoising noise.

Knobs:

- `fast_forward_steps` (int | list[int]): outer steps to fast-forward.
  Default 12; `0` disables. With `n_action_steps=8`, 12 outer steps = 96
  sub-steps (~9.6 sim seconds at 10 Hz).
- `fast_forward_pct` (float, 0-1): probability that a **single iteration**
  uses FF for ALL its groups. Default 0.8. The Bernoulli draw is once per
  `collect()` call, not per group — mixing FF and non-FF groups within an
  iteration would distort cross-group reward comparisons (FF groups have
  shorter `num_steps` and thus larger time-scaled rewards). Long-run FF
  fraction across iterations still approaches `fast_forward_pct` because
  each call gets a different `base_seed`.

Edge cases handled:

- If any env terminates during the FF prefix (e.g., accidental success),
  the collector falls back to a normal seed-aligned group for that group.
- With `num_async_vector_env < group_size`, the post-FF branch point is
  captured once (turn 1, via `get_scene_bundle`) and re-applied for turns
  2..k — the lockstep FF prefix is **not** re-run on later turns (it would
  diverge, since the model-query denoising noise is unseeded).
- FF prefix steps are **not** counted in `episode.num_steps`, so
  time-scaled rewards compare post-branch effort fairly within a group.
- `--debug-fast-forward` saves a per-group montage of camera views to
  `<output_dir>/debug_ff/group<G>_seed<S>_ff<F>.png` so you can eyeball
  that every env in a group really is bit-identical at the branch point.

### Init from saved sim state

A second, more explicit branching mode: instead of having env 0 run the
*current model* forward N steps to produce a branch state (Fast-Forward),
load a **pre-saved** scene + sim state from a `.npz` and start every env in
every group from there. Intended for overfitting / curriculum experiments —
e.g., training GRPO on a single known-hard intermediate state (step 10 of a
specific failing trajectory) to study how the policy refines its behavior at
that state without burning compute on the upstream approach.

```bash
uv run python scripts/grpo/train_grpo.py \
    --init-state-npz-path /path/to/ep000_step010.npz \
    --success-weight 0.3 \
    --fast-forward-pct 0.0 \
    --min-alive-groups 0 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 50
```

The npz must be produced by `scripts/denoising_lab/eval/interactive_rollout.py`
(or any saver that follows the same contract: `__sim_state__`,
`__model_xml__`, `__ep_meta__` keys; see `branching_rollout.py:182-210`).

Mechanics:

1. `_load_init_bundle` (`collect_episodes.py`) parses the npz once per
   collector process and caches the resulting `{ep_meta, model_xml,
   sim_state, consumed_substeps}` dict, keyed by path. The
   `consumed_substeps` field is what makes the post-restore rollout
   truncate at the **remaining** budget rather than a fresh full one —
   see "Budget accounting" below.
2. `_align_envs_to_group_scene` short-circuits the usual "env 0's bundle →
   all envs" handshake and broadcasts the loaded bundle to every env via
   the same `apply_scene_bundle` RPC (`collect_episodes.py:412-522`).
3. Within-group and across-group divergence comes entirely from per-env
   denoising noise; the env starts bit-identical everywhere.

Interactions with other knobs:

- **Forces Fast-Forward off internally.** `fast_forward_steps` /
  `fast_forward_pct` are ignored when `init_state_npz_path` is set
  (logged at iter start). Set `fast_forward_pct=0.0` on the CLI too if
  you want the intent visible in logs without trusting the override.
- **`min_alive_groups` is a gradient-stability knob.** With every
  group starting from the same saved state, each group is an independent
  sample of the per-group outcome distribution (different denoising
  noise → different mixes of success/failure). Requiring ≥N alive
  (mixed) groups via `min_alive_groups=N` reduces gradient noise and
  the risk of policy collapse from low-alive-group updates — at the
  cost of more wall time when the success rate is at an extreme (the
  dynamic loop extends toward `max_groups`).
- **`success_weight` choice.** Default `1.0` (pure binary reward) is
  fully supported and a common choice. Setting `success_weight < 1.0`
  blends in `max_progress`, which provides advantage variance even
  before any rollout succeeds — useful if early-iter all-failure dead
  groups would stall training. Pick whichever fits the analysis; the
  trainer does not warn for either choice.

#### Budget accounting (`consumed_substeps`)

The saved npz represents an env state captured **partway through** an
original trajectory — `ep000_step010.npz` is 10 outer chunks into episode
0. Naively restoring the sim state without telling the wrapper about
that elapsed time would grant the post-restore rollout a **fresh full**
`max_episode_steps` budget — i.e., a step-10 restore would get the same
horizon as a step-0 restore, contradicting "this is what happens after
10 steps have already elapsed in the original trajectory." The
`consumed_substeps` field fixes this by billing the elapsed sub-steps
against the wrapper's truncation check, so the rollout has only the
**remaining** budget. Mirrors `branching_rollout.py:488-505` exactly.

**Formula.** `consumed_substeps = branch_step × n_action_steps`. Worked
example for the user's typical setup:

| Knob | Value |
|------|-------|
| `__step_info__["step"]` (from npz) | 10 |
| `__step_info__["n_action_steps"]` (from npz) | 8 |
| `consumed_substeps` (derived) | 80 |
| `--max-episode-steps` (CLI / config) | 480 |
| **Remaining post-restore budget** | **400 sub-steps = 50 outer chunks** |

**Mechanism.** `apply_scene_bundle` pre-fills `self.reward` and
`self.done` with `consumed_substeps` placeholders so
`MultiStepWrapper`'s truncation check
(`len(self.reward) >= max_episode_steps` at `multistep_wrapper.py:271-275`)
already accounts for the elapsed time. The first post-restore sub-step
truncates at `max_episode_steps - consumed_substeps` more sub-steps,
not at `max_episode_steps`.

**NPZ contract.** `__step_info__` is a JSON object with the keys `step`
(outer chunk index when the npz was saved) and `n_action_steps`
(sub-steps per outer chunk used by the original rollout). Both are
written by `scripts/denoising_lab/eval/interactive_rollout.py` and read
by `branching_rollout.py:182-210`. Note that the SAVED `n_action_steps`
is used — not the current run's `--n-action-steps` — so consumed sub-
steps reflect actual wall-clock time elapsed in the original trajectory
regardless of any chunk-size changes between save and replay.

**Fallbacks and warnings.** Three fallback paths, in order:

1. `__step_info__` present with both `step` and `n_action_steps` →
   compute `consumed_substeps` precisely. No warning.
2. `__step_info__` present but `n_action_steps` missing → warn, default
   to `consumed_substeps=0` (fresh full budget). The user should re-save
   with `interactive_rollout.py` to get correct accounting.
3. No `__step_info__` AND filename doesn't match `ep*_step*.npz` → warn,
   default to `consumed_substeps=0`. Same remediation.

**Sanity checks.** Two more guards fire at runtime in
`apply_scene_bundle`:

- If `consumed_substeps >= max_episode_steps`, a warning suggests either
  picking an earlier branch point or raising `--max-episode-steps`; the
  rollout would otherwise truncate after a single sub-step with
  near-zero training signal.
- If `reward_agg_method` is not `"max"` or `"sum"` (defaults to `"max"`
  in `MultiStepWrapper`), a warning fires because the pre-filled zeros
  would dilute a `"mean"` aggregation.

A negative `consumed_substeps` (from a hand-edited `__step_info__` with
a negative `step` or `n_action_steps`) raises `ValueError` at load time
rather than silently no-op'ing via Python's `[0.0] * -n == []`.

**What `consumed_substeps` does NOT change:** the recorded episode
`num_steps` still counts from 0 post-restore (matching the FF
convention), so time-scaled advantages compare post-restore effort
fairly within a group. Only the wrapper's truncation horizon is
affected.

### Dynamic group collection

Many RoboCasa tasks have a wide success-rate distribution early in
training: some groups produce 0/G successes and contribute no gradient
signal (per-group reward std falls below the dead-group threshold). To
avoid wasting an iteration on a buffer with zero live signal:

```
config.num_groups = 5              # MINIMUM groups per iter (was fixed)
config.min_alive_groups = 4        # keep adding groups until ≥4 are alive (mixed)
config.max_groups = 10             # hard cap on dynamic collection
```

After the first `num_groups` groups, the collector keeps adding **one
group at a time** until either:

1. `alive_groups >= min_alive_groups` (a group is "alive" if it is
   mixed: `0 < group_successes < group_size`, equivalently per-group
   reward std > 0 under `success_weight=1.0` with time-scaling
   disabled — the only regime this exact predicate is valid for; for
   `success_weight<1.0` the criterion would have to inspect shaped-
   reward variance, which the collector does not currently compute),
   or
2. `group_idx >= max_groups` (hard cap, logs a WARNING).

The "alive" predicate matches the trainer's gradient-signal filter
exactly: `compute_advantages` zeros the advantage of any group with
std < 1e-4, and the GRPO update drops zero-advantage chunks before
backward (`abs(c.advantage) < 1e-12`). All-success groups
(`group_successes == group_size`) and all-fail groups
(`group_successes == 0`) both have std = 0 and contribute zero
gradient — neither is "alive". An earlier version of this loop used
"≥1 success" as a proxy, which silently counted all-success groups
as satisfying the gate; that has been replaced by the exact mixed
criterion. In the early/low-success regime (no group fully solved
yet) the two criteria are equivalent.

To disable dynamic collection entirely, set `min_alive_groups = 0` —
the collector then always stops at exactly `num_groups`.

Constraints (enforced in `GRPOConfig.__post_init__`):

- `max_groups >= num_groups`
- `max_groups <= 100` (seed-stride collision boundary)
- `min_alive_groups <= max_groups`

Subprocess/RPC timeouts auto-scale at 7 min/group:
`timeout = 420 * effective_max_groups` seconds.

---

## GRPO Algorithm

### Per-iteration phases

```
for iteration in range(start, num_iterations + 1):
    # Phase 0: pre-flight memory cleanup
    _release_memory_to_os()                                # gc + cuda + malloc_trim

    # Phase 1: collect this iter's task
    env_name = env_names[(iter-1) % len(env_names)]
    _collect_episodes(env_name)                            # via subprocess OR RPC

    # Phase 2: compute advantages
    buffer.compute_advantages(success_weight, max_steps)   # per-group z-score

    # Phase 2b: pre-compute reference log-probs (current model == ref before update)
    _compute_ref_log_probs()                               # caches backbone features

    # Phase 3: GRPO update
    _grpo_update()                                         # update_epochs × minibatches

    # Phase 4: log + checkpoint
    if iteration % save_interval == 0: _save_checkpoint(...)
```

### Reward shaping → advantage

```
shaped = success_weight * success + (1 - success_weight) * max_progress
scaled = shaped / num_steps * max_episode_steps          # faster = better
A_episode = (scaled - group_mean) / (group_std + 1e-8)   # PER GROUP
A_chunk = A_episode / num_chunks_in_episode
```

- `success_weight = 1.0` (default) → pure binary reward, time-scaled. The
  collector skips `compute_dense_progress` entirely in this mode.
- `success_weight < 1.0` → blend in continuous progress from
  `dense_reward.py` (per-task: drawer/door joint position, PnP distance to
  target, stove knob state, etc.). Variance even in all-fail groups.
- Time-scaling (`/ num_steps * max_episode_steps`) means faster solutions
  get larger reward, creating advantage variance even within all-success
  groups.
- `A_chunk = A_episode / num_chunks` preserves the within-group
  zero-sum invariant at the chunk level, so every trajectory contributes
  equal **total** gradient weight regardless of length.

A group with reward std < 1e-4 is **dead**: its chunks get advantage
exactly 0 and are filtered out before any forward pass (see "Minibatch
construction").

### FM log-prob surrogate

Flow-matching has no closed-form log-probability. Following DPPO
(Ren et al. 2024), `compute_fm_log_prob` uses negative FM loss as a
surrogate:

```
x_τ = (1 - τ) ε + τ a            # interpolate noise → action
v_target = a - ε                  # true velocity
v_pred = action_head(x_τ, τ, cond)
log π(a | obs) ≈ −E_τ[MSE(v_pred, v_target)]
```

Critical invariants for the importance ratio:

1. **Same ε** for ref pass and current pass. The collector captures the
   actual noise tensor used at inference time via `grpo_server.py`'s
   `torch.randn` hook (thread-local; see "Noise capture" below); training
   reuses it.
2. **Same τ samples** for ref pass and current pass. After sampling
   jittered timesteps for the ref pass, they are stored on each chunk
   (`chunk.tau_samples`) and replayed during `_grpo_update`.

The MSE is computed in **fp32** even though the model runs in bf16: bf16
mantissa is too coarse to resolve the small (current − ref) differences
GRPO depends on, which otherwise inflate `mean_log_ratio_abs` and clip
fraction.

### Noise capture (`grpo_server.py`)

The denoising loop creates ε via `torch.randn` inside
`Gr00tN1d6ActionHead.get_action_with_features`. To recover it without
breaking other code paths:

- `torch.randn` is patched **once at module import** with a thread-local
  router. Other threads see pass-through.
- `GRPOPolicyWrapper.get_action` sets a thread-local capture context only
  during the denoising call, captures the **first 3-D randn** as ε, and
  clears the context on exit (`try/finally`).
- The raw `(B, 50, 128)` action prediction (before
  `decode_action()` slices to the embodiment's actual dims) is captured
  by monkey-patching `get_action_with_features` for the duration of the
  call.

Both are returned to the collector in the `info` dict and persisted into
each chunk's `.npz`. A `compute_action_mask` derived from the loaded
embodiment's modality config is also returned so FM-MSE ignores padded
dims.

### tau_centers

`compute_fm_log_prob` averages MSE over `K = len(tau_centers)` evaluation
points along the same ε → action path. **One DiT forward pass per
center.**

```python
tau_centers = [0.0, 0.25, 0.35, 0.5, 0.6, 0.75]   # default (late-biased)
```

Each iteration, every center gets a small Gaussian jitter
(`std=0.02`, `_sample_jittered_timesteps` in `fm_log_prob.py`) and is
clamped to `[0, noise_s]` where `noise_s = 0.999`. The jittered samples
are then **shared** between the ref pass and the current pass for every
chunk so the importance ratio reflects only model difference, not
sampling noise.

Why late-biased: at inference time, the model takes only **4 Euler
steps** (`t = 0, 0.25, 0.5, 0.75`). Velocity errors at late τ (closer to
the clean action) have fewer remaining steps to correct, so weighting the
surrogate toward late τ aligns the training signal with what matters at
inference.

This is **independent of inference** — the inference loop always uses
exactly 4 Euler steps regardless of `tau_centers`. `tau_centers` only
affects training log-prob evaluation. Adding more centers improves the
log-prob estimate but linearly increases per-minibatch compute.

### Minibatch construction (stratified, dead-group filter)

`_grpo_update_inner` does NOT use `EpisodeBuffer.iter_minibatches` (a flat
shuffle). It uses `_iter_stratified_minibatches` instead:

1. **Dead-group filter**: drop every chunk with `|advantage| < 1e-12`
   (advantage was set to literal 0 by `compute_advantages` for groups with
   std < 1e-4). Filtering here keeps every minibatch uniformly live-only
   and avoids a `(0 - mean) / std` term polluting the per-minibatch
   advantage renorm.

2. **Bin live chunks by `group_id`** and shuffle within each bin.

3. **Each minibatch**:
   - GUARANTEED: take up to `mb_size // n_live_groups` chunks from EACH
     live group (best-effort if a group's queue is short).
   - FILLER: fill the remaining `mb_size % n_live_groups` slots from a
     globally-shuffled pool, skipping chunks already used in this batch.

With `mb_size=8` and 5 live groups: 1 guaranteed per group + 3 filler
chunks. Every chunk is yielded exactly once per epoch (across epochs the
permutation reshuffles).

Why stratify: chunks within an episode share an identical
`A_episode / num_chunks` advantage. A flat-shuffled minibatch dominated
by 1-2 episodes has near-zero advantage variance, and the per-minibatch
z-score renorm in `_grpo_update_inner` then squashes that batch's
gradient signal toward zero. Stratification guarantees every minibatch
spans all live groups.

Why uniform-over-CHUNKS for the filler (vs uniform-over-GROUPS): it
self-balances. Fuller groups contribute filler proportionally more often,
so all groups drain in lockstep and the "≥1 per group" guarantee holds
for essentially the whole epoch.

### Balanced Training

Two **independent** mechanisms that address the common failure mode in
early-stage GRPO where most rollouts fail: negative-advantage chunks
vastly outnumber positives, individual mini-batches carry a weak or
one-sided gradient signal, and a small number of sparse successes are
over- or under-weighted relative to the training budget they warrant.

Each is controlled by its **own** flag — `balanced_minibatch_training`
(mechanism 1) and `dynamic_epoch_training` (mechanism 2) — both default
`True`. They are fully decoupled, so any of the four on/off combinations is
valid. With both off, training is bit-identical to the unmodified
stratified-minibatch, fixed-epoch (`update_epochs`) path.

#### Mechanism 1: balanced mini-batch sampling (`balanced_minibatch_training`)

**What it does.** Each mini-batch enforces `balanced_minibatch_positive_adv_ratio`
(X) in **both directions**. The sign class that is underrepresented relative to
X is the "minority" and is oversampled with replacement; the overrepresented
class is the "majority" and is drawn without replacement, controlling when the
epoch ends.

**When it activates.** Always when both sign classes are present:
- `natural_pos_frac < X`: too few positives → cycle positives, drain negatives
- `natural_pos_frac ≥ X`: too few negatives → cycle negatives, drain positives

Falls back to `_iter_stratified_minibatches` only when one sign class is
entirely absent (all episodes fail or all succeed within live groups).

**Why bidirectional matters.** At high success rates (e.g. 70% positive), the
few negative-advantage chunks (failures) receive a very large magnitude from
per-minibatch z-score renorm, producing an outsized "avoid failure" gradient
that can collapse the policy in the next iteration. Cycling negatives caps this
by ensuring each batch has the targeted proportion regardless of the natural
distribution.

**Sampling strategy.** The minority pool reshuffles when exhausted to give
best-effort equal exposure across minority chunks. The majority pool advances
monotonically and may not be fully consumed before the epoch-length anchor is
reached — some majority chunks go unseen each epoch, which is the documented
cost of the rebalancing.

**Epoch length.** Anchored to `ceil(n_live_chunks / mb_size)`, matching the
vanilla stratified path so `update_epochs` remains directly comparable between
balanced and vanilla runs. When the majority pool drains early, the epoch stops
rather than yielding minority-only tail batches that would defeat the balance
guarantee.

**Relationship to Jitter-GRPO.** With `jitter_lambda > 0`, `entries` is
doubled (`fixed + jitter` copies of each chunk). Both copies of a positive
chunk are independent entries in the positive pool. The balanced sampler
draws from them in shuffled order; the Jacobian regularizer accumulates at
epoch granularity (not within a single mini-batch), so the pairing
requirement is satisfied regardless of whether fixed and jitter copies land
in the same batch. The combination of both features is sound.

#### Mechanism 2: dynamic epoch count (`dynamic_epoch_training`)

**What it does.** Scales `update_epochs` using a **tent function** of the
positive-advantage fraction, implemented via exact integer arithmetic:

```
m = min(successful_eps, total_eps − successful_eps)
actual_num_epochs = max(1, (4·m·update_epochs + total_eps) // (2·total_eps))
```

This is the integer form of `floor(2·min(sf, 1−sf)·update_epochs + 0.5)`.
The formula peaks at `success_frac = 0.5` (→ full `update_epochs`) and
decays symmetrically toward both extremes:

- **Near 0% success:** all-failure, purely negative advantages, sparse useful
  signal → 1 epoch
- **Near 50% success:** balanced +/− advantages, most informative → full
  `update_epochs`
- **Near 100% success:** all-success, highly asymmetric advantages (the few
  failures get very large negative advantage from group-relative normalisation,
  dominating gradient direction) → reduced epochs

The integer formula avoids ULP cancellation that can corrupt `float`-based
implementations at specific episode counts when `update_epochs ≥ 6`.

**What counts as `successful_eps / total_eps`.**

- `total_eps` is the number of episodes in **live groups** only — groups
  whose per-group reward std is ≥ 1e-4 and thus produce non-zero gradient
  signal. Dead all-success or all-fail groups are excluded from both
  numerator and denominator to prevent their inflating `success_frac` and
  keeping `actual_num_epochs` near `update_epochs` when real signal is
  sparse.
- `successful_eps` counts live-group episodes with **positive advantage**
  (`self.buffer.advantages[i] > 0`), not `ep.success`. This is intentional:
  with shaped rewards (`success_weight < 1.0`) a failing episode with high
  `max_progress` can contribute positive advantage, and `ep.success` would
  undercount it. Using advantage sign keeps the epoch formula consistent
  with mechanism 1, which oversamples chunks with `c.advantage > 0`.

**Examples.** 5 groups × 4 rollouts, `update_epochs = 4`:
- `success_frac = 0.25` (2/8 positive): `m=2`, `(32+8)//16 = 2` epochs
- `success_frac = 0.50` (4/8): `m=4`, `(64+8)//16 = 4` epochs (peak)
- `success_frac = 0.70` (14/20): `m=6`, `(96+20)//40 = 2` epochs — fewer
  than the old monotonic formula's 3, preventing overshoot at high success

#### CLI usage

```bash
# Both mechanisms are ON by default. Use the tyro switch flags to toggle them
# (--flag enables, --no-flag disables); booleans take no value.
uv run python scripts/grpo/train_grpo.py \
    --no-dynamic-epoch-training \
    --balanced-minibatch-positive-adv-ratio 0.7 \
    --update-epochs 5 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env
```

The two flags are independent — e.g. the run above keeps the balanced
sampler (`balanced_minibatch_training` stays on) but runs exactly
`update_epochs` epochs every iteration (`--no-dynamic-epoch-training`). To do
the reverse, pass `--no-balanced-minibatch-training` and leave the dynamic
epochs on.

The startup banner prints one line per enabled mechanism:
`Balanced mini-batch sampling: ON (positive_adv_ratio=…)` and/or
`Dynamic epoch count: ON (tent epochs=max(1, floor(2·min(sf,1-sf)·N+0.5)))`.
When `dynamic_epoch_training` is on, a per-iteration line `Dynamic epochs:
X/Y positive-advantage live-group episodes (tent scale=Z) → A/N epochs` is
printed. TensorBoard logs `balanced/actual_epochs` and
`balanced/success_fraction` (gated on `dynamic_epoch_training` and at least
one optimizer step in that iteration).

#### Files touched

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `balanced_minibatch_training: bool = True`, `dynamic_epoch_training: bool = True`, and `balanced_minibatch_positive_adv_ratio: float = 0.5` with `__post_init__` validation (ratio strictly in `(0, 1)` when `balanced_minibatch_training=True`). |
| `train_grpo.py` | `_grpo_update_inner` computes `actual_num_epochs` via the integer tent formula when `dynamic_epoch_training` is on (else `update_epochs`), and dispatches to `_iter_balanced_minibatches` when `balanced_minibatch_training` is on (else `_iter_stratified_minibatches`). `_iter_balanced_minibatches` applies the target ratio bidirectionally — cycles the minority sign class with replacement, drains the majority without replacement. `_log_metrics` emits `balanced/actual_epochs` and `balanced/success_fraction` (gated on `dynamic_epoch_training` and `n_updates > 0`). |
| `test_balanced_fixes.py` | Unit tests for both mechanisms plus their independence (all four on/off combinations): per-batch ratio in both directions, epoch-length anchor, minority cycling, fallback paths, tent formula correctness including integer ULP cases. |

---

### Clipped surrogate + KL

```
ratio = (current_log_prob - ref_log_prob).exp()
advantages = (A - A.mean()) / (A.std() + 1e-8)            # renorm per-batch
surr1 = A * ratio
surr2 = A * clamp(ratio, 1 - clip_eps_low, 1 + clip_eps_high)
clip_loss = -min(surr1, surr2).mean()

# Schulman k3 KL estimator (non-negative pointwise, symmetric gradient):
inv = ref_log_prob - current_log_prob
kl_loss_last_iter = kl_coef_last_iter * (inv.exp() - inv - 1).mean()

# Optional KL anchor to the base frozen DiT (LoRA disabled). Skipped when
# kl_coef_base_model = 0; otherwise base_log_prob is pre-computed once per
# iter inside the same no_grad pass that produces ref_log_prob, with
# `with disabled_adapters(model.action_head.model)`.
inv_base = base_log_prob - current_log_prob
kl_loss_base_model = kl_coef_base_model * (inv_base.exp() - inv_base - 1).mean()

loss = clip_loss + kl_loss_last_iter + kl_loss_base_model
```

NaN/Inf guard: a minibatch with non-finite loss (typically bf16 ratio
overflow when `|log_ratio|` is large) is **skipped**, the
`n_skipped_nonfinite` counter increments, and training continues.
`clip_grad_norm_` only bounds finite gradients — it does not rescue NaNs.

If ZERO minibatches commit a gradient step in an iteration (every batch
non-finite, or every group dead), the iteration is treated as **skipped**
and the resume checkpoint is saved under the last successfully-updated
iter's name (see "Checkpointing").

### Reference log-prob caching

`_compute_ref_log_probs` runs once per iteration, BEFORE the GRPO update,
in a `no_grad` block. It serves two purposes:

1. Captures `ref_log_prob` + `tau_samples` per chunk for reuse in the
   update.
2. **Caches per-chunk Eagle backbone + state encoder features** onto each
   `ActionChunk`. Both are frozen (no LoRA), so their output is identical
   across all `update_epochs × minibatches` in this iteration.

In `_grpo_update`, `_prepare_batch` checks if every chunk in the batch
has cached features and takes the fast path (`_rebuild_encoded_from_cache`)
— restacking cached slices instead of re-running the backbone. This is
the largest single training-time speedup in the loop.

The cache is invalidated each iteration by `buffer.clear()` (called by
`_release_memory_to_os` at iter start).

---

## Jitter-GRPO (Jacobian regularizer)

An optional, feature-flagged extension layered on top of the standard GRPO
loop. Default `jitter_lambda = 0.0` is bit-identical to vanilla GRPO; setting
`--jitter-lambda 0.05` activates the full mechanism.

### Motivation

Standard GRPO trains the DiT velocity field along the rolled-out denoising
trajectory: each update tightens `v_θ(x_t, t | obs)` toward `(a − ε)` at the
single point `x_t = (1−t)·ε + t·a`. Trajectories from noise samples *near* `ε`
rely entirely on architectural smoothness of the velocity field to land near
`a`. When that smoothness is poor, a successful action chunk's basin can be
narrow — the model is fragile to tiny perturbations of the inference noise.

However, the promise of Flow Matching is to be *noise-resillient* and have the
denoising velocity field push noise into good action basins, whereas today, the 
velocity field is quite sensitive to perturbations in noise, leading to fragility 
when picking between high-advantage and low-advantage actions.
In order to encourage the the velocity field to be more robust, we would like
to encourage neighboring noise to `ε` to also lead to `a`.

Jitter-GRPO adds a Frobenius-norm Jacobian penalty
`(1−t)²·λ²·‖∇_x v_θ‖_F²` *in expectation* to the existing loss, encouraging
the velocity field to be locally smooth along each rolled-out path. The
implementation is a one-line trick: feed the DiT a variance-preserving
jittered noise input `ε' = √(1−λ²)·ε + λ·ξ` (ξ ~ N(0, I)) but keep the
velocity target at the **original** `a − ε`. Taking expectation over ξ gives
the standard FM loss + the Jacobian penalty, with no double-backward and no
architecture changes. The cached `chunk.ref_log_prob` (computed at the
original ε) is reused for both branches — the cached-vs-recomputed-ref bias
is `O(λ²)` and θ-independent, so the gradient direction is unaffected.

### Knob: `GRPOConfig.jitter_lambda`

- Default `0.0` → bit-identical to vanilla GRPO (no jittered passes, no
  per-branch metrics, no extra CUDA syncs).
- Suggested value `0.05` (variance-preservation multiplier `(1−t)²·λ²` ≤
  2.5e-3, comfortably below the bf16 mantissa noise floor).
- Range-checked in `GRPOConfig.__post_init__`: must satisfy `0.0 ≤ λ < 1.0`
  (variance preservation requires `λ < 1`).
- The trainer prints a one-line `[Jitter-GRPO] lambda=...` banner at startup
  when active, including the doubled-step warning.

### Paired scheduling (entries doubling)

Each live chunk produces TWO entries per epoch when jitter is active:

```python
entries = (
    [(c, "fixed") for c in live_chunks]      # always
    + [(c, "jitter") for c in live_chunks]   # only when jitter_lambda > 0
)
```

Both entries reference the **same** `ActionChunk` object (so they share
`tau_samples`, `ref_log_prob`, `initial_noise`, and the cached backbone
features). The only difference is the DiT input noise during the forward
pass: "fixed" rows use the original `ε`, "jitter" rows use
`ε' = √(1−λ²)·ε + λ·ξ`.

Doubling the entries list doubles the number of optimizer steps per epoch.
**Halve `update_epochs` MANUALLY** when running with jitter (e.g., 4 → 2)
to match the per-iter optimizer-step budget of vanilla GRPO. The trainer
does not auto-halve — the relationship is left explicit so the user can
audit it from the CLI.

### `compute_fm_log_prob`: per-τ jittered input noise

`fm_log_prob.compute_fm_log_prob` gains an optional `noise_for_input` kwarg:

```python
def compute_fm_log_prob(..., noise, noise_for_input=None):
    eps = noise                       # original ε; drives velocity_target
    velocity_target = actions - eps   # ALWAYS at the ORIGINAL ε

    if noise_for_input is not None:   # required shape: [K, B, H, D]
        eps_input_all = noise_for_input
    else:
        eps_input_all = None          # back-compat fallback

    for k in range(n_samples):        # K-loop over tau_centers
        eps_input = eps if eps_input_all is None else eps_input_all[k]
        noisy_trajectory = (1 - t)*eps_input + t*actions
        # ... DiT forward, MSE per row, accumulate
```

Two design choices:

1. **`velocity_target` stays at the ORIGINAL ε.** It's `actions - noise`,
   NOT `actions - noise_for_input`. The asymmetry between input and target
   is what produces the Jacobian regularizer in expectation. Swapping the
   target to ε' would gain an `O(λ²)` model-independent floor that doesn't
   shrink as the model improves.

2. **Per-τ independent ξ_k.** The trainer already probes the FM log-prob
   at `K = len(tau_centers)` different τ values per chunk per minibatch
   (see the `tau_centers` subsection above — defaults to a length-6
   late-biased schedule). Jitter-GRPO draws ONE fresh ξ_k for each of
   those K τ-evaluations, so a paired chunk's jittered forward pass uses
   K different ε'_k = √(1−λ²)·ε + λ·ξ_k along its K τ samples. The
   caller therefore passes a 4-D `[K, B, H, D]` tensor where each
   `noise_for_input[k]` carries the ξ-jitter for one τ-evaluation. This
   gives K independent samples of the Jacobian expectation per minibatch,
   matching the variance-reduction structure of `tau_centers`. Only the
   4-D shape is supported (validated with a shape check); 3-D broadcast
   would diverge from the per-τ-fresh-ξ design.

Backward compat: when `noise_for_input=None` (the default), the function
falls back to `eps_input = eps` and the K-loop is bit-identical to the
pre-Jitter-GRPO code.

### `_iter_stratified_minibatches`: now yields entries

Refactored to operate on `list[(ActionChunk, str)]` instead of
`list[ActionChunk]`. Group binning still uses `chunk.group_id` (read off
the tuple's first element); both copies of a paired chunk share `group_id`
so they land in the same group's queue but typically end up in different
minibatches across the epoch. Yielded type: `list[(ActionChunk, str)]`.

Same deterministic shuffle behavior — at `jitter_lambda=0`,
`entries = [(c, "fixed") for c in live_chunks]` has identical length and
ordering to the old `live_chunks`, and the same RNG seed produces the
same minibatch composition.

### `_prepare_batch`: carries mode through

Takes `batch: list[(ActionChunk, str)]`. The order-preserving filter
`valid_pairs = [(c, m) for (c, m) in batch if c.raw_action is not None]`
keeps modes aligned 1:1 with `valid_batch`. Returns the same `batch_data`
dict with one new key:

```python
batch_data["modes"]: list[str]   # length B, parallel to valid_batch
```

### `_compute_ref_log_probs`: always tags as "fixed"

The reference log-prob pass uses the original ε for both branches (per the
cached-ref invariant), so its single call site simply wraps the chunk list
as `[(c, "fixed") for c in batch]` before passing into `_prepare_batch`.
No `noise_for_input` is constructed; the ref pass is bit-identical
regardless of `jitter_lambda`.

### ξ sampling and `noise_for_input` construction

Inside `_grpo_update_inner`, after `_prepare_batch` returns and the
`ready_*` slicing is done:

```python
ready_modes = [batch_data["modes"][i] for i in ready_indices]
lam = self.config.jitter_lambda

if lam > 0.0 and any(m == "jitter" for m in ready_modes):
    K = len(self.config.tau_centers)
    B_r, H, D = ready_noise.shape

    # Unseeded; uses global torch RNG, matching _sample_jittered_timesteps.
    xi = torch.randn(K, B_r, H, D,
                     device=self.device, dtype=ready_noise.dtype)

    jitter_mask = torch.tensor(
        [m == "jitter" for m in ready_modes],
        device=self.device, dtype=torch.bool,
    )

    # expand returns a stride-0 view; clone() materializes a writable
    # [K, B_r, H, D] tensor so __setitem__ writes per-K rows independently.
    noise_for_input = (
        ready_noise.unsqueeze(0).expand(K, -1, -1, -1).clone()
    )
    sqrt_one_minus = (1.0 - lam * lam) ** 0.5
    noise_for_input[:, jitter_mask] = (
        sqrt_one_minus * ready_noise[jitter_mask].unsqueeze(0)
        + lam * xi[:, jitter_mask]
    )
else:
    noise_for_input = None
```

Three notable details:

- **ξ is unseeded.** Uses the global torch RNG, matching how
  `_sample_jittered_timesteps` jitters the τ centers. On-policy collection
  noise also isn't seeded per-call, so making ξ a special case would be
  inconsistent with the rest of the training-time stochasticity. Resume
  across iters proceeds without errors but ξ values are not bit-reproducible
  across the resume boundary at `jitter_lambda > 0`.
- **`expand+clone` is required.** `unsqueeze(0).expand(K, -1, -1, -1)`
  returns a stride-0 view across the K dim; `__setitem__` on the view would
  alias all K rows. The explicit `.clone()` materializes a writable per-K
  tensor before the assignment.
- **Fixed rows pass through unchanged.** Only
  `noise_for_input[:, jitter_mask]` is overwritten. Rows where
  `mode == "fixed"` retain `ready_noise` from the broadcast clone — ε for
  both target and input, identical to vanilla GRPO behavior.

The constructed `noise_for_input` then flows into:

```python
current_log_probs = compute_fm_log_prob(
    ..., noise=ready_noise, noise_for_input=noise_for_input,
    n_samples=len(self.config.tau_centers),
)
```

When the gate is False (λ=0 or no jitter rows in this mb),
`noise_for_input=None` and the K-loop takes the original-ε path.

VRAM cost: `xi + noise_for_input ≈ 2 × 614 KB` per minibatch at
`K=6, B=8, H=50, D=128` in bf16. Negligible vs the DiT activations.

### Per-branch metrics (`*_fixed` / `*_jitter` TB scalars)

The KL is refactored to expose `kl_per_row_last_iter` (and optionally
`kl_per_row_base_model`) as named intermediates so they can be indexed by
branch. The final `kl_loss_last_iter = kl_coef_last_iter *
kl_per_row_last_iter.mean()` is numerically identical to the previous
inlined form.

Inside the no-grad accumulator block, **gated on `lam > 0.0`**, we split
the per-row tensors by mode and accumulate row-level sums:

```python
if lam > 0.0:
    fixed_mask = torch.tensor([m == "fixed" for m in ready_modes], ...)
    jit_mask = ~fixed_mask

    n_f = int(fixed_mask.sum().item())
    n_j = int(jit_mask.sum().item())
    if n_f > 0:
        ratio_sum_fixed                  += ratio[fixed_mask].sum().item()
        log_ratio_abs_sum_fixed          += log_ratio_abs[fixed_mask].sum().item()
        clipfrac_sum_fixed               += int(over_clip[fixed_mask].sum().item())
        kl_per_row_sum_last_iter_fixed   += kl_per_row_last_iter[fixed_mask].sum().item()
        if compute_base:
            kl_per_row_sum_base_model_fixed += kl_per_row_base_model[fixed_mask].sum().item()
        n_rows_fixed                     += n_f
    # ... analogous for jitter
```

End-of-iter, per-branch metrics are added to `update_stats` only when at
least one row of that branch fired:

```python
if n_rows_fixed > 0:
    result["clipfrac_fixed"]                 = clipfrac_sum_fixed / n_rows_fixed
    result["mean_ratio_fixed"]               = ratio_sum_fixed / n_rows_fixed
    result["mean_log_ratio_abs_fixed"]       = log_ratio_abs_sum_fixed / n_rows_fixed
    result["kl_loss_last_iter_fixed"]        = kl_coef_last_iter * (kl_per_row_sum_last_iter_fixed / n_rows_fixed)
    if compute_base:
        result["kl_loss_base_model_fixed"]   = kl_coef_base_model * (kl_per_row_sum_base_model_fixed / n_rows_fixed)
# ... analogous for jitter
```

The gating on `lam > 0.0` matters: at `jitter_lambda=0`, the per-mb
accumulator block is skipped entirely, the per-branch counters stay at
their zero defaults, the result-dict gating `if n_rows_fixed > 0:` is
False, and no `_fixed`/`_jitter` keys are emitted. Vanilla GRPO runs see
exactly the same TB curves they always did.

**Aggregation note.** Legacy aggregated metrics (`clipfrac`, `mean_ratio`,
`mean_log_ratio_abs`, `kl_loss_last_iter`, `kl_loss_base_model`) are
means-of-per-mb-means — each minibatch contributes one entry regardless of
size. The new `*_fixed` / `*_jitter` metrics are **row-weighted**
(sum / n_rows). The two will differ slightly when minibatch sizes vary
across the iter (e.g., the last mb is smaller than `mb_size`).

### TB / wandb writing (`_log_metrics`)

A small loop in `_log_metrics` iterates `for branch in ("fixed", "jitter")`
× `for metric in ("clipfrac", "mean_ratio", "mean_log_ratio_abs",
"kl_loss_last_iter", "kl_loss_base_model")` and writes
`train/<metric>_<branch>` only if the key is present in `update_stats`:

```python
for branch in ("fixed", "jitter"):
    for metric in ("clipfrac", "mean_ratio",
                   "mean_log_ratio_abs",
                   "kl_loss_last_iter", "kl_loss_base_model"):
        key = f"{metric}_{branch}"
        if key in update_stats:
            self.writer.add_scalar(f"train/{key}", update_stats[key], iteration)
```

The wandb path already iterates `update_stats.items()` so it picks up the
new keys automatically.

### What this surfaces (the diagnostic signal)

The empirical Jacobian-norm signal is the **gap** between the fixed and
jitter branches' `mean_log_ratio_abs` (and analogous `clipfrac`):

- A jitter row's `current_log_prob = -MSE(v_θ(x_t', t), v_target)` evaluates
  the velocity field at a perturbed input; the expected gap to the
  fixed-row evaluation is `(1−t)²·λ²·‖∇_x v_θ‖_F²`.
- If the gap **shrinks** across iters, the regularizer is doing its job —
  the velocity field is becoming smoother along the rolled-out trajectory.
- If the gap is **flat**, λ may be too small to provide signal.
- If `clipfrac_jitter ≫ clipfrac_fixed`, the jitter branch's ratio variance
  has grown beyond the clip threshold — λ is likely too aggressive for the
  current model state, or the model is genuinely sensitive in a way the
  regularizer is fighting.

### Bit-identical guarantee at `jitter_lambda = 0`

| Path | Behavior at λ=0 |
|------|-----------------|
| `entries` construction | `[(c, "fixed") for c in live_chunks]` — same length and order as old `live_chunks`. |
| `_iter_stratified_minibatches` | Same RNG seed, same shuffle, same minibatch composition; yields the same chunks just wrapped in 1-tuples of `(c, "fixed")`. |
| `_prepare_batch` | Same `valid_batch` ordering; new `modes` list emitted but unused downstream. |
| `compute_fm_log_prob` | `noise_for_input=None` → `eps_input = eps` → K-loop math unchanged. |
| ξ-sampling block | Gated on `lam > 0.0`; not entered. |
| Per-branch metric block | Gated on `lam > 0.0`; not entered. No extra CUDA syncs from `.item()`. |
| Legacy aggregated metrics | Identical formulation; per-mb-mean accumulators preserved. |
| TB scalars | No `_fixed`/`_jitter` keys emitted; legacy TB curves byte-identical. |

Resume across iters at `jitter_lambda=0` is bit-reproducible end-to-end.
At `jitter_lambda > 0`, ξ samples are not bit-reproducible across the
resume boundary (intentional — ξ uses global torch RNG, matching τ-jitter
and on-policy collection noise).

### Toy / production CLI

Toy-mode (fixed-seed diagnostic, fast turnaround):

```bash
uv run python scripts/grpo/toy_train_grpo.py \
    --jitter-lambda 0.05 --update-epochs 2
```

Production (single-task or multi-task):

```bash
uv run python scripts/grpo/train_grpo.py \
    --jitter-lambda 0.05 --update-epochs 2 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200
```

To compare against vanilla GRPO at the **same per-iter step budget**, run
a baseline at `--jitter-lambda 0 --update-epochs 4`. Same total
optimizer-step count; the difference is solely the Jacobian regularizer
pressure on positive-advantage chunks (basin sharpening) and
negative-advantage chunks (neighborhood-carve).

The toy script's startup banner prints `Jitter lambda: <value>` so you can
confirm the flag flowed through. `grpo_data/` collisions between jitter and
non-jitter runs at the same LR are the user's responsibility to manage
(rename or override `--checkpoint-dir` / `--episode-dir` to keep TB curves
separate).

### Files touched by this feature

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `jitter_lambda: float = 0.0` field + range check in `__post_init__`. |
| `fm_log_prob.py` | `compute_fm_log_prob` accepts optional `noise_for_input: Tensor[K,B,H,D] \| None`. K-loop uses `eps_input_all[k]` per τ when provided; `velocity_target = actions - eps` unchanged. |
| `train_grpo.py` | `_iter_stratified_minibatches` and `_prepare_batch` operate on `(chunk, mode)` entries. `_compute_ref_log_probs` wraps as `("fixed", chunk)` tuples. `_grpo_update_inner` builds doubled entries, samples ξ via global RNG, constructs `noise_for_input` via expand+clone, threads it into `compute_fm_log_prob`, and adds gated per-branch metric accumulators. `_log_metrics` writes the per-branch TB scalars. Startup banner prints jitter lambda when active. |
| `toy_train_grpo.py` | Prints `Jitter lambda` in the startup banner; inherits the field from `GRPOConfig` automatically. |

### Scope

Implemented: paired scheduling on top of the existing single-chunk-per-row
training loop, per-τ independent ξ_k jitter on the DiT input (one fresh ξ
per τ in `tau_centers`), cached-ref reuse for both branches, gated
per-branch TB metrics.

Not implemented: adaptive λ schedules, an offline noise-sensitivity
validation eval (the per-branch `mean_log_ratio_abs` gap is the live
signal), and the alternative formulation that recomputes the reference at
the jittered input (the cached-ref bias is `O(λ²)` and `θ`-independent —
recomputation costs an extra DiT pass per minibatch for no observable
training-direction change).

---

## Checkpointing & Resuming

### What gets saved

Every `save_interval` iterations into `<checkpoint_dir>/iter_NNNN/`:

```
iter_0050/
  lora_weights.pt   # only the LoRA A/B tensors (~80 MB at rank=16)
  optimizer.pt      # {"optimizer_state": ..., "param_names": [...]}
```

LoRA weights are extracted by filtering for `"lora_" in name` (works with
the low-level `inject_adapter_in_model` API, where
`get_peft_model_state_dict` is unreliable).

`optimizer.pt` bundles the AdamW state with the **ordered list of
trainable param names**. AdamW serializes its state by integer position;
a peft/torch version bump that reshuffles same-shape LoRA tensors would
silently mis-attach Adam moments without the name check.

### Resume

```bash
uv run python scripts/grpo/train_grpo.py \
    --resume-from grpo_data/grpo_checkpoints/iter_0050
```

On resume:

1. LoRA arch is rebuilt from the **current** `lora_rank` / `lora_alpha` /
   `lora_target_modules`.
2. `load_lora_checkpoint` does a strict two-sided key match and per-key
   shape check — **hard-fails** on:
   - keys in save but not in model (target_modules shrank);
   - keys in model but not in save (target_modules grew);
   - shape mismatch (rank changed).
3. `optimizer.pt` is loaded with `_validate_optimizer_param_names`
   (positional order) + `_validate_optimizer_state` (param count +
   exp_avg shape) — both raise actionable errors on mismatch.
4. `start_iteration` is parsed from the basename via
   `re.fullmatch(r"iter_([0-9]+)", dir_name)` (ASCII digits only — Unicode
   digits like `iter_０` are rejected). Non-canonical names (`best/`,
   `latest/`, `iter_50.bak`) silently fall back to `start_iteration=1`,
   preserving backward compat for unstructured checkpoint names.

### Resume + reuse cached collection (`resume_from_collected_data`)

When a prior run crashed AFTER finishing collection but BEFORE the model
update completed, the on-disk `episode_dir/iter_NNNN/` already contains
fully-collected episodes that are still on-policy for the resumed
checkpoint (they were produced by the policy whose weights live in
`resume_from`). Set `--resume-from-collected-data` to skip the FIRST
resumed iter's ~7 min × num_groups simulation and load those `.npz`
files directly:

```bash
uv run python scripts/grpo/train_grpo.py \
    --resume-from grpo_data/grpo_checkpoints/iter_0050 \
    --resume-from-collected-data
```

Validation runs in `setup()` BEFORE the model loads, so misconfigured
caches fail fast. The validator checks (in order):

1. `resume_from` follows the canonical `iter_NNNN/` pattern (required —
   without it the validator can't infer which iter dir to load).
2. `episode_dir/iter_{start_iteration:04d}/` exists and is readable
   (PermissionError surfaces with a `chmod` hint, not a misleading "Cache
   is empty").
3. Per-file scalars are present and well-typed: `env_name` matches the
   round-robin task for `start_iteration`, `group_id` is a non-negative
   int (no silent default to 0), `success` is bool/int/float (no string
   coercion), `num_chunks` is a positive int.
4. The first `.npz` exposes `raw_action_*` / `action_mask_*` /
   `initial_noise_*` keys for every chunk (FM log-prob surrogate
   prerequisite).
5. Group counts: `num_groups <= n_observed <= max_groups`, with the
   `min_alive_groups` criterion satisfied OR `n_observed ==
   max_groups` exactly (the collector's exit conditions). Alive is
   defined as `0 < group_successes < group_size` (mixed) — same
   predicate the live collector uses.
6. Per-group sizes: undercount warns (mirrors `_collect_episodes`'s
   partial-collection policy), overcount raises (manual cache merge or
   collector bug — within-group `env_seed` invariant broken).

Only the FIRST resumed iter consumes the cache; subsequent iters collect
normally. The decision is rederived as `iteration ==
self._start_iteration AND config.resume_from_collected_data`, no mutable
flag carried across phases.

**When NOT to use:** do NOT enable when you've changed any
collection-affecting config since the cache was written. The validator
catches `env_name` and group-count mismatches but does NOT detect changes
to `n_action_steps`, `fast_forward_steps` / `fast_forward_pct`,
`init_state_npz_path`, `max_episode_steps`, or `success_weight` — the
cached iter would silently train on episodes from the old config while
subsequent iters collect under the new one. If in doubt, leave this
disabled and pay the collection cost.

**TB cosmetic:** the cached iter's `time/collect_seconds` is logged as
NaN (filtered out by `_log_metrics`) so the curve shows a clean gap at
the resumed iter rather than a near-zero plunge that distorts autoscale.
Other phase-time scalars (`advantage`, `update`) and overall
`time/iteration_seconds` log normally.

### Skip semantics (preserves iteration budget)

An iteration is "skipped" if no `optimizer.step()` actually fired. Two
paths produce this:

1. Outer skip: global `std_reward < 1e-8` → entire buffer pruned, no
   update.
2. Inner skip: per-iter `n_updates == 0` (every minibatch non-finite, OR
   every group dead with global std still > 1e-8).

In both cases the checkpoint (if scheduled this iter) is written under
the **last successfully-updated** iter's name — not the current loop
iter. Resuming from that dir then sets `start_iteration = last + 1`,
which is exactly the skipped iter, so it gets a fresh attempt rather than
being burned from `num_iterations`. If that dir already exists (the
previous iter was itself a save-interval boundary), the write is skipped
— the on-disk state is bit-identical.

### Episode dir retention

`episode_dirs_to_keep` controls how many `iter_NNNN/` `.npz` directories
under `episode_dir` are kept around for post-mortem inspection. Default
is 3 (current + 2 prior). Pruning runs **before** the current iter's
directory is created, so the on-disk count never temporarily exceeds the
cap.

At ~0.5 GB/iter for the default config, 200 iters unpruned would burn
~100 GB; the default keeps disk under ~1.5 GB.

---

## Configuration Reference

`GRPOConfig` (in `grpo_config.py`) is the single source of truth. CLI
overrides go through `tyro`:

```bash
uv run python scripts/grpo/train_grpo.py \
    --lora-rank 32 --kl-coef-last-iter 0.005 --kl-coef-base-model 0.005 \
    --num-iterations 500 \
    --tau-centers 0.0 0.3 0.5 0.7 0.9
```

### Key knobs

**Model & LoRA**
- `model_path` (default `nvidia/GR00T-N1.6-3B`)
- `embodiment_tag` (default `ROBOCASA_PANDA_OMRON`)
- `lora_rank` / `lora_alpha` / `lora_dropout` (default 16 / 32 / 0.0)
- `lora_target_modules` — defaults to `DEFAULT_LORA_TARGET_MODULES` from
  `lora_dit.py`: 8 module patterns inside each of the 32 DiT blocks
  (`attn1.to_{q,k,v}`, `attn1.to_out.0`, `ff.net.{0.proj,2}`,
  `proj_out_{1,2}`). ~20M trainable params at rank=16.

**Episode collection**
- `group_size` (G) — logical rollouts per group. Default 4.
- `num_async_vector_env` — physical parallel-env workers per group. Default
  `None` → `group_size` (one worker per rollout, unchanged). Set lower (must
  divide `group_size` and be `<=` it) to collect each group over
  `group_size // num_async_vector_env` sequential turns and cap peak worker
  RAM. See "Decoupling group size from worker count". Bake-time: must match
  `collector_server.py --num-async-vector-env`.
- `num_groups` — minimum groups per iter. Default 5.
- `min_alive_groups` / `max_groups` — see "Dynamic group
  collection". Default 2 / 5.
- `max_episode_steps: int | list[int]` — per-env truncation horizon.
  Default 480.
- `n_action_steps` — sub-steps to execute from each 16-step chunk.
  Default 8.
- `fast_forward_steps: int | list[int]`, `fast_forward_pct` — see
  "Fast-Forward Branching". Default 12 / 0.8.
- `init_state_npz_path` — see "Init from saved sim state". Default None
  (disabled). When set, overrides the seed-based scene init for every
  group; intended for overfitting / curriculum experiments.
- `env_names: list[str]` — round-robin task selection.
- `episode_dir`, `episode_dirs_to_keep`.

**ZMQ wiring**
- `server_host` / `server_port` — in-process policy server (default
  `127.0.0.1:5555`).
- `collector_server_host` / `collector_server_port` — long-running
  collector daemon. Empty host disables it (subprocess fallback).

**Reward shaping**
- `success_weight` (0-1) — binary weight in shaped reward. Default 1.0
  (pure binary, no dense progress collected). Set < 1.0 to blend in
  `max_progress`, which provides continuous advantage variance from
  denoising noise; useful when binary-only would produce all-failure
  groups early on.

**GRPO algorithm**
- `clip_eps_low` / `clip_eps_high` (both default 0.2) — asymmetric clip
  bounds; ratio clamped to `[1 - clip_eps_low, 1 + clip_eps_high]`. Each must
  be in `(0, 1)` (no ordering constraint between them).
- `update_epochs` (default 5)
- `mini_batch_size` (default 8 chunks)
- `kl_coef_last_iter` (default 0.1) — KL anchor to this iter's start-of-update
  policy snapshot. Bounds per-iter drift.
- `kl_coef_base_model` (default 0.0) — KL anchor to the pretrained DiT
  (LoRA disabled). Bounds cumulative drift from the base policy. 0.0 disables
  the term entirely (no extra forward pass per iter, no per-mb KL formula).
- `tau_centers` (default `[0.0, 0.25, 0.35, 0.5, 0.6, 0.75]`)
- `balanced_minibatch_training` (default `True`) — balanced mini-batch
  sampling; see "Balanced Training" mechanism 1.
- `dynamic_epoch_training` (default `True`) — tent-function epoch scaling;
  see "Balanced Training" mechanism 2. Independent of the flag above.
- `balanced_minibatch_positive_adv_ratio` (default `0.5`) — target fraction
  of positive-advantage chunks per mini-batch. Must be strictly in `(0, 1)`.
  Only active when `balanced_minibatch_training=True`. Raise above 0.5 (e.g.
  0.7) to bias more gradient steps toward success examples.

**Optimizer**
- `learning_rate` (default 1e-5; ~10× lower than supervised FT because RL
  gradients are noisier)
- `weight_decay` (1e-5)
- `max_grad_norm` (0.5)

LoRA params are upcast to **fp32 for AdamW** while the frozen base stays
bf16. Without this, Adam moments underflow at small lr × bf16 ULP and
the policy barely moves. PEFT's `LoraLayer.forward` handles the dtype
mismatch internally.

**Training loop**
- `num_iterations` (default 200)
- `resume_from` (default None)
- `resume_from_collected_data` (default False) — see "Resume + reuse
  cached collection". Only valid when `resume_from` is also set; rejected
  at config-construction time otherwise. Skips the first resumed iter's
  collection by loading on-disk episodes from
  `episode_dir/iter_{start_iteration:04d}/`.
- `checkpoint_dir`, `save_interval` (default every 2 iters)
- `seed` (default 67)

**Logging**
- TensorBoard writer always on; logs at `<checkpoint_dir>/tb_logs/`.
- `use_wandb` + `wandb_project` + `wandb_run_name` for optional W&B.

Logged scalars include `episode/{success_rate,mean_reward,std_reward}`,
`train/{loss,clip_loss,kl_loss_last_iter,kl_loss_base_model,clipfrac,mean_ratio,mean_log_ratio_abs,n_skipped_nonfinite}`,
`train/learning_rate`, `time/iteration_seconds`, and (when
`dynamic_epoch_training=True` and at least one gradient step fired)
`balanced/{actual_epochs,success_fraction}`. `mean_log_ratio_abs`
is the primary diagnostic for DPPO-style surrogates: large values mean
the FM-MSE log-prob is noisy enough that most updates clip.

---

## Operational Notes

- **GPU**: a single 24-GB+ NVIDIA GPU (training keeps frozen base in
  bf16, only LoRA params in fp32). Tested on A10G with `mini_batch_size=8`.
- **CPU/RAM**: the collector spawns `num_async_vector_env` MuJoCo workers
  per env (default `group_size`); in long-running mode,
  `len(env_names) × num_async_vector_env` total. 64+ GB RAM is comfortable
  for 8 tasks × 5 workers. Lower `num_async_vector_env` (collecting each
  group over multiple turns) to fit larger groups on a RAM-limited host.
- **Robocasa venv**: located at
  `gr00t/eval/sim/robocasa/robocasa_uv/.venv/`. The subprocess collector
  path hard-codes this path; if you've put robocasa elsewhere, switch to
  long-running mode and just point the trainer at it.
- **Memory creep**: there are small leaks in robosuite/MuJoCo's model
  reload path. The collector workers `gc.collect()` + `malloc_trim(0)`
  after every `apply_scene_bundle`; the trainer does the same at the
  start of each iter. Even so, restart the long-running collector every
  ~50-100 iterations as a hygiene measure — the trainer's
  consecutive-failure budget will simply retry on the next iter once the
  collector is back up.
- **Consecutive-failure abort**: the trainer aborts after 3 consecutive
  collector failures (timeout, non-zero exit, zero episodes loaded). The
  log line right before the abort lists common causes (wrong venv path,
  stuck port, missing MuJoCo backend, OOM, collector not running).
- **Fatal vs transient errors**: `FatalCollectorError` (raised on
  `ValueError`-class server errors, primarily env-name mismatches) is
  re-raised immediately rather than burning the retry budget. The error
  message includes the exact restart command needed.
