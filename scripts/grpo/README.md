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
| `episode_buffer.py` | `EpisodeBuffer`, `GRPOEpisode`, `ActionChunk`. Loads `.npz` episodes, computes group-relative advantages. |
| `fm_log_prob.py` | FM-loss-as-log-prob surrogate (`compute_fm_log_prob`), jittered timestep sampler (`_sample_jittered_timesteps`), production sampler schedule (`inference_schedule`) and the last-step-differentiable chunk rollout (`_smooth_chunk_rollout`) the roughness constraint measures. |
| `lora_dit.py` | `apply_lora_to_dit`, `save_lora_checkpoint`, `load_lora_checkpoint`, default target-module list. |
| `smoothness.py` | Trajectory-roughness ("jerk") constraint primitives: `second_difference`, `roughness_moments`, `pooled_hf`, `roughness_hf`, the continuous-action-dim selector and `build_key_dim_span`. Model-free and fully unit-testable. The 4-step chunk rollout lives in `fm_log_prob._smooth_chunk_rollout` (it needs the DiT); the hinge lives in `train_grpo._grpo_update_inner`. |
| `eval_lora_from_npz.py` | Eval harness: runs N parallel rollouts of a LoRA policy from a saved `interactive_rollout.py` `.npz`, aggregates per-attempt success/num_steps into `results.json`. Subclasses `EpisodeCollector` in init-state mode. |
| `test_*.py` | Sanity checks for sim-wrapper / `.npz` key roundtrip. `test_grad_accum.py` drives the real `_grpo_update_inner` on CPU to pin the gradient-accumulation semantics and the PAWS mass accounting / cold start. `test_jitter_metrics.py` does the same for the `jitter/*` / `ref_mse/*` / sign-split / effective-clipfrac instrumentation. `test_anchor_groups.py` does the same for anchor groups (classification, row budget, renorm isolation, sampler/PAWS/epoch exclusions). |
| `verify_multiturn_gpu.py` | Real-stack check for multi-turn collection / branch-point integrity. Run on the GPU VM in the robocasa venv. |
| `test_video_key_filter.py` | Covers the unused-video-key filter (`dropped_video_keys`). |
| `test_smoothness.py` | CPU suite for the trajectory-roughness constraint: HF calibration, the `a_hat = a + (1−τ)r` identity, hinge semantics, dim/horizon selection, the derived sampler schedule, the last-step-differentiable rollout (exact value + gradient localized to the final step), the `compute_fm_log_prob` return contract, both instruments' jitter-invariance, the executed-chunk metrics, `smooth_ref.json` guard rejection **including instrument mismatch**, and `smooth_coef=0` bit-identity (stats, weights and RNG stream) through the real `_grpo_update_inner`. |
| `verify_render_skip_gpu.py` | Real-stack check for `skip_intermediate_render`: proves the kept frame is byte-identical to the unskipped path against real MuJoCo/EGL rendering, and reports the render count + speedup. Robocasa venv, no model server. |
| `test_scene_seed_pool.py` | CPU suite for the frozen scene seed pool: base resolution, the stateless cursor + pass alignment, within-iteration seed distinctness (including a non-divisible K), all four config validations plus the pass-alignment warning, `GROUP_SEED_STRIDE` agreement between the two files, byte-identity of the disabled collector argv, the real `EpisodeCollector.collect` consuming `--group-seeds` (and refusing to wrap), and `per_scene_success` → `episode/scene_sr/*` emission through the real `_log_metrics`. |
| `test_clip_floor.py` | CPU suite for the per-row MSE-referenced lower clip (`clip_low_mse_coef`), the PAWS `k` floor (`paws_k_floor_at_target`) and the three added diagnostics: off-switch determinism + additivity (the bit-identity-vs-baseline check is an out-of-tree differential, recipe in that test's docstring), the `rho_floor` arithmetic incl. the binding `clip_eps_low` ceiling, agreement of **all six** lower-bound consumers on rows straddling their own floors, positive/anchor-row inertness against the four-case table, both `k` floors and both untouched `k` branches, monotonicity in the coefficient, hand-computed `drift/*` values, `jitter/pos_clip_budget_used`, and the `lora/cos_step_*` cosines incl. the sign flip and the two `L_early` sources. |

---

## Architecture

Two processes share the work:

```
┌─────────────────────────┐   ZMQ obs/action    ┌──────────────────────────┐
│ Trainer (main .venv)    │ ◄──────────────────►│ Collector (robocasa venv)│
│  GPU model + LoRA       │  port 5555          │  AsyncVectorEnv workers  │
│  In-process PolicyServer│                     │  Writes .npz per iter    │
└─────────────────────────┘                     └──────────────────────────┘
```

Each iteration the trainer spawns one `collect_episodes.py` subprocess in the
robocasa venv, which connects back to the in-process policy server over ZMQ,
runs the group rollouts, and writes the episodes as `.npz` files.

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

### 1. Run training

Run the trainer; it spawns one `collect_episodes.py` subprocess per iteration
(in the robocasa venv) to collect the group rollouts:

```bash
uv run python scripts/grpo/train_grpo.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200 \
    --group-size 4 --num-groups 5 \
    --checkpoint-dir grpo_data/grpo_checkpoints
```

### 2. Standalone server (debug / eval only)

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

### Frozen scene seed pool (`scene_seed_pool_size`)

The per-iter stride above means **every iteration trains on, and reports,
brand-new scenes**. Measured between-scene success-rate sd is **0.285**, so at
`num_groups=4` roughly **84% of the per-iteration `episode/mean_reward` variance
is scene resampling**, not policy change. The training curve is then
uninterpretable: a swing between consecutive iterations says nothing about the
update that happened in between.

`scene_seed_pool_size = K > 0` freezes a pool of K scene seeds and cycles it
deterministically across iterations, so the same scenes recur.

```bash
uv run python scripts/grpo/train_grpo.py \
    --scene-seed-pool-size 12 \
    --num-groups 4 --max-groups 4 --min-alive-groups 0 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env
# optional: pin the pool explicitly instead of deriving it from --seed
#   --scene-seed-pool-base 300000
```

`--min-alive-groups 0` is the only *mandatory* companion flag (the default is 2).
`max_groups` does not have to be lowered to `num_groups` — it only has to satisfy
`K >= max_groups`, so the stock `max_groups=5` is fine at `K=12`. It stays inert
anyway: with `min_alive_groups=0` the collector always stops at exactly
`num_groups`, which is why the trainer sends exactly `num_groups` seeds.

**Knobs.**

| knob | meaning |
|---|---|
| `scene_seed_pool_size` (K) | Pool size. **0 = DISABLED**, and disabled is bit-identical to a pre-feature run: no `--group-seeds` argument is appended to the collector argv, and no per-scene TB series are emitted. |
| `scene_seed_pool_base` | First seed; the pool is `base + j * GROUP_SEED_STRIDE` for `j in [0, K)`. `None` → resolved **in `__post_init__`** to `seed + 100_000`, which is exactly the seed block iteration 1 would have drawn under the old formula. At the default `seed=67` and `K=12` the pool is `100067, 101067, ..., 111067`. Resolution happens in place (not at the use site) so the concrete value lands in the TensorBoard `config` text dump — otherwise the run's own artifacts would not record which scenes it trained on. |

**Cursor.** Per iteration, `num_groups` consecutive pool slots, wrapping:

```
pool     = [base + j * GROUP_SEED_STRIDE for j in range(K)]
seeds[g] = pool[((iteration - 1) * num_groups + g) % K]
```

`self.iteration` is **1-based** (`train()` loops
`range(self._start_iteration, num_iterations + 1)`, and the legacy `--seed`
formula yields `100067` at iteration 1 with `seed=67`), so the `- 1` makes
iteration 1 start at pool index 0. At `K=12, num_groups=4`:

| iteration | seeds | `episode/pool_pass` |
|---|---|---|
| 1 | 100067, 101067, 102067, 103067 | 0 |
| 2 | 104067, 105067, 106067, 107067 | 0 |
| 3 | 108067, 109067, 110067, 111067 | 0 |
| 4 | 100067, 101067, 102067, 103067 | 1 |

**Why stateless.** The cursor is a pure function of `iteration`, not a persisted
counter. That makes `--resume-from` correct *by construction*: resuming at
iteration 37 recomputes exactly the seeds iteration 37 would have used in an
uninterrupted run, with no new checkpoint state to write, load, validate or
forget. A persisted counter would have to survive `_save_checkpoint` / load
**and** the `resume_from_collected_data` path that skips a collection entirely,
and would silently drift on any resume from a checkpoint written before the
counter existed.

The `* num_groups` term is the *advance*, not the number of slots sent — see
"Dynamic collection" below for why they differ and what that costs.

**Validations** (all hard `ValueError` at config construction, because every one
of these failure modes is otherwise *silent* — the run completes, the curves look
plausible, and the property the pool exists to provide is simply gone):

| check | reason |
|---|---|
| `K >= 1` (or exactly `0` to disable) | `0` is the only disabling value. A negative K would sail past the internal `> 0` gate and disable the feature silently; a fractional `0 < K < 1` would build an empty pool and raise `ZeroDivisionError` inside the cursor, minutes into the run. |
| `K >= max(num_groups, max_groups)` | Within one iteration two groups must never land on the same seed. Two groups on one scene correlates their group-relative advantages and double-counts that scene in the iteration mean — GRPO's group-relative baseline assumes independent scenes per group, and nothing downstream would flag the violation. The bound uses `max_groups` because a dynamic extension (`min_alive_groups > 0 and max_groups > num_groups`) can consume up to that many consecutive slots in one call, and the trainer sends that many. Consequence: `K == num_groups` additionally requires `max_groups == num_groups`. `EpisodeCollector.collect`'s own `_max_reachable_groups` mirrors this; the two must agree or one rejects a run the other accepts. |
| `init_state_npz_path is None` | An init bundle overrides the scene entirely — the reset's own seeded scene is immediately overwritten by `apply_scene_bundle` — so a scene pool would be **silently inert**: seeds passed, logged and plotted with no effect on a single pixel. Silent inertness is the exact failure mode worth erroring on. |

**Dynamic collection (`min_alive_groups > 0`) is supported.** The cursor advances
by `num_groups` per iteration, but the trainer hands the collector
`max(num_groups, max_groups)` consecutive slots so an extension cannot run off the
end of the list mid-iteration. Advancing by the *realized* group count would keep
exposure exactly balanced but requires the trainer to learn that count after the
fact and carry it across resumes — the persisted state the stateless cursor exists
to avoid. The trade: on an extended iteration the extra group borrows the seed the
**next** iteration opens with, so that scene gets a bonus visit and appears in two
consecutive iterations. It is never duplicated *within* one iteration (that is what
`K >= max(num_groups, max_groups)` guarantees). Read `episode/n_groups` to see which
iterations extended, and prefer the per-scene mean of `episode/scene_sr/<seed>` over
the raw pass mean if you want a pass statistic immune to the bonus visit's extra
weight. Note `K == num_groups` — the every-iteration-comparable setting — therefore
also needs `max_groups == num_groups`.

**Pass-alignment warning** (a `warnings.warn`, *not* an error): when
`K % num_groups != 0` the pool still cycles deterministically and still never
repeats a seed within an iteration, but a full pass no longer aligns with a whole
number of iterations, so **you cannot read a pass mean off a fixed iteration
stride** and `episode/pool_pass` increments mid-block.

**New TB scalars** (emitted only when the pool is enabled):

| scalar | meaning |
|---|---|
| `episode/scene_sr/<seed>` | Per-scene success rate for every seed present in the iteration. One curve per scene over the run — the only view that separates "the policy improved" from "this iteration drew easier scenes". Routed through `_log_metrics`' `_emit` helper, so a non-finite value is dropped with a warning rather than poisoning chart autoscale. |
| `episode/pool_pass` | `((iteration - 1) * num_groups) // K` — 0-based pass index, for binning iterations into passes. |

Gated on the pool because with it **off** every iteration has fresh seeds, so
unconditional emission would create `num_groups` brand-new single-point series
per iteration (200+ over a 50-iteration run at `num_groups=4`), bloating the
event file and the TB sidebar for no benefit.

**How to read the result — the important caveat.** With `K > num_groups` the
**per-ITERATION success rate is exactly as noisy as before**: each iteration
still samples only `num_groups` of the K scenes, so between-scene variance is
undiminished within one point. What the pool buys is that the *sequence* of
iterations covers a fixed scene set, so the readable unit is the **pass mean over
`K / num_groups` consecutive iterations** — and `episode/scene_sr/<seed>` gives
per-scene curves that are directly comparable across iterations. If you want
every single iteration to be comparable on its own, set `K == num_groups`, at the
cost of training on only `num_groups` distinct scenes for the whole run.

#### Per-group scene uniqueness (`clear_ep_meta`)

Until this was fixed, **every group after the first in a collector process
inherited group 1's kitchen.** Observed directly in the fingerprint log: four
seeds in one iteration all reporting `layout=7 style=10` with four different
`xml=` hashes — i.e. one kitchen, four object arrangements. That contradicts
`collect_episodes.py`'s own claim that "each group gets a unique seed → unique
initial kitchen/object configuration".

The chain, all four links verified in source:

1. `apply_scene_bundle` calls `set_ep_meta(bundle["ep_meta"])` — this robosuite has
   no `set_attrs_from_ep_meta`, so the `hasattr` chain falls through to the `elif`
   — and `set_ep_meta` is a plain `self._ep_meta = meta` that **persists for the
   life of the process** (`robosuite/environments/base.py:404-410`).
2. `KitchenEnv._load_model` takes `layout_id` / `style_id` straight from
   `self._ep_meta` when present, and only otherwise draws
   `self.rng.choice(self.layout_and_style_ids)`
   (`robocasa/environments/kitchen/kitchen.py:355-360`).
3. `reset()` *does* re-run `_load_model` every time — robosuite defaults are
   `hard_reset=True`, `deterministic_reset=False` (`base.py:106,139,290`) — so the
   layout is re-decided each reset; it just kept re-deciding in favour of the pin.
4. `reset_from_xml_string` sets `deterministic_reset=True` and then restores it to
   `False` (`base.py:664-673`), so it is *not* an additional persistent pin.

`GroupAlignmentWrapper.clear_ep_meta()` drops the pin via robosuite's own
`unset_ep_meta()`, and `_align_envs_to_group_scene` calls it on every env
**immediately before** the reset (clearing afterwards would be a no-op, since the
reset is where `_load_model` consults it). Each group's kitchen then comes from
`default_rng(group_seed)` — unique per seed and still reproducible across
iterations, which is what the pool depends on. Roughly 11 distinct kitchens from 12
seeds, versus 3 before.

Three placement details, each pinned by a test in `test_scene_seed_pool.py`:

| where | behaviour | why |
|---|---|---|
| `_align_envs_to_group_scene` | clears, once per group, before the reset | the group's own seed must choose its kitchen |
| `_restart_at_branch_point` | **never** clears | turns 2..k must reproduce turn 1's scene; the pin turn 1 left is what stops this reset drawing a different kitchen in the moment before the bundle restores it |
| init-state mode | skips the clear | the loaded bundle overrides the scene wholesale; clearing would only buy a wasted model rebuild per group |

Unconditional, not behind a flag — the previous behaviour contradicted the
documented contract, so there is no configuration in which inheriting group 1's
kitchen is the desired outcome. Note the cost: four *different* kitchens per
iteration instead of one means four MjModel builds per collector process rather
than one repeated layout, so expect higher peak worker RSS and more wall-clock
variance (heavy kitchens are real — one observed iteration collected at 1 ep/min
against 2–3 elsewhere).

#### Verifying the pool's premise (`--log-scene-fingerprint`)

The pool assumes a `group_seed` reproduces the same **scene** across iterations.
That is not obviously true in RoboCasa, and if it is false the pool still trains
fine but delivers none of the measurement gain — while printing byte-identical
logs either way. So the trainer turns this diagnostic on for **every** pooled run
(no config knob to disable it) and `collect_episodes.scene_fingerprint` appends
scene identity to each group's log line:

```
Group 1/4 (seed=100067 layout=3 style=7 xml=a3f9c1d2 state=7b2e40aa) (from seed): 10/12 success | ...
```

Why it might be false: RoboCasa chooses layout/style/textures at env
**construction** time from a per-instance RNG and stores them in the model XML,
not in `MjSimState` (this module's header; `_align_envs_to_group_scene` — the
scene-bundle broadcast exists *because* same-seed envs render different scenes),
whereas `RoboCasaEnv.reset(seed=)` reseeds `random`/`np.random`/`env.rng` and
re-places objects. Every iteration is a fresh collector subprocess with fresh
workers, so a group's kitchen may come from whatever env 0 happened to construct
rather than from its pool seed.

| part | covers | reading |
|---|---|---|
| `layout=` / `style=` | RoboCasa's own `layout_id` / `style_id`, straight out of `ep_meta` | The kitchen choice itself — eyeballable with no hash comparison. Omitted until robosuite's reset has populated `ep_meta`. |
| `xml=` | sha1 of the full model XML | Stronger: also covers textures, fixture placement, camera poses. |
| `state=` | sha1 of the flattened `MjSimState` | Strongest: additionally pins the reset's **object placements**, the part `reset(seed=)` does control. |

**How to read it.** Compare the same pool slot across two passes (at `K=12,
num_groups=4`: iteration 1 group 1 vs iteration 4 group 1):

- `xml=` matches → the kitchen is pinned; the pool works and `episode/scene_sr/<seed>`
  is a genuine per-scene learning curve.
- `xml=` matches, `state=` differs → same kitchen, different object placements.
- `xml=` differs → **the pool is not freezing the scene.** Gradients are still
  valid (the scene-bundle broadcast keeps every rollout in a group on one scene,
  so group-relative advantages hold), but the cross-iteration comparison is
  confounded and the pass mean is not a fixed-scene average.

Also worth checking on the first iteration: whether the four groups share one
`xml=`. If they do, layout is per-process and only placements vary per seed.

Cost is one sha1 over the branch-point bundle `_align_envs_to_group_scene`
**already captured** for the turn driver — no extra `get_scene_bundle` RPC, no sim
stepping, ~1 ms against a ~5 min group. It fingerprints the pristine deep copy
rather than `bundles[0]`, because `apply_scene_bundle` mutates `ep_meta` in place
and hashing after that would make one scene read differently depending on when it
was measured. `scene_fingerprint` never raises (malformed fields degrade to `?`, a
missing bundle to `n/a`), and the published value is cleared at the top of every
alignment so a path that captures no bundle — `group_size == 1` — cannot leave the
previous group's fingerprint to be misattributed. In `init_state_npz_path` mode it
is a positive control: it should read identically on every group of every
iteration, so variation there indicts the diagnostic or the bundle cache, not the
pool.

**Interaction with `--resume-from` / `resume_from_collected_data`.** Resume needs
nothing extra (that is the point of the stateless cursor), as long as `K`,
`scene_seed_pool_base` and `num_groups` are unchanged — change any of them and
the resumed run trains on different scenes than the prefix did, which
`_validate_collected_data_cache` does not check (it verifies env name, group
counts and FM keys). `resume_from_collected_data` is consistent by construction:
the cached iteration skips collection, and the per-scene curves are keyed on the
`env_seed` stored in each `.npz`, so they carry the seeds the episodes were
actually collected under. Episodes written before `env_seed` existed default to
`0` (`episode_buffer.load_episodes`) and would all pool into
`episode/scene_sr/0`.

**Interaction with `toy_train_grpo.py`.** Do **not** combine the pool with the
toy trainer. `ToyGRPOTrainer._collect_episodes` achieves fixed seeds by a
different mechanism — it temporarily mutates `config.seed`, `config.num_groups`
and `self.iteration` around each one-group subprocess call — which the pool
cursor would read as its inputs and override with pool seeds. `ToyGRPOConfig`
inherits the default `scene_seed_pool_size = 0`, so this only arises if you pass
the flag explicitly; it is not validated against.

Covered by `test_scene_seed_pool.py`.

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

With 8 tasks × 200 iters, each task gets 25 updates.

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
  trainer scales its collector subprocess timeout by `k` automatically.

### Skipping intermediate-substep renders (`skip_intermediate_render`)

Default **on** for collection (`GRPOConfig.skip_intermediate_render`), off for
eval. Each outer step executes `n_action_steps` substeps, and robosuite renders
every camera on every one of them — but the collector reads observations with
`video_delta_indices=[0]`, so `MultiStepWrapper._get_obs` returns
`self.obs[-1]` alone. For PandaOmron that means **21 of every 24 renders**
(3 cameras × 8 substeps) were computed, flipped, resized to 256², and thrown
away.

With the flag on, `MultiStepWrapper.step` disables the robosuite camera
observables for the chunk, then **primes** them back to their natural sampling
phase immediately before the last substep (`RoboCasaEnv.prime_camera_obs`) so
robosuite takes that one sample itself, inside `env.step()`.

- **Observationally exact, not an approximation.** robosuite samples a camera
  observable once per control step, on the LAST of its
  `control_timestep / model_timestep` physics substeps (the phase is established
  by the `force_update` at the end of `MujocoEnv.reset`). Priming reproduces that
  phase exactly, so the kept frame is taken at the same physics substep as
  baseline.
- **Why priming rather than a forced render after the substep** (the earlier
  design): `MujocoEnv.step` runs `_post_action` *between* the physics loop and
  the observation read, and `Kitchen._post_action` calls `update_state()`, which
  writes fixture visuals into `sim.model` — coffee-liquid, burner-flame and
  sink-water `site_rgba`, cabinet interior sites (`accessories.py:89-94`,
  `stove.py`, `sink.py`, `cabinets.py`). A frame rendered once `step()` has
  returned therefore shows fixture state **one control step ahead** of baseline:
  a silent, systematic distribution shift versus eval and the pretraining data.
  `recompute_observation()` survives only as the early-termination fallback,
  where the collector discards the observation anyway.
- **Why not just re-enable on the last substep** (the obvious approach, which is
  wrong): `Observable.set_enabled()` calls `Observable.reset()`, which zeroes the
  sampling timer but does **not** clear `_sampled`. Every env reset ends with a
  `force_update` that leaves `_sampled=True`, so a re-enabled observable cannot
  sample for a full control step — the first chunk of every episode renders
  nothing and returns `reset()`'s all-zero **float64** buffer, and later chunks
  sample on the FIRST physics substep, i.e. one control step (48 ms) stale.
- **`enabled`, not `active`:** `Observable.update` gates *computation* on
  `_enabled`; `_active` only gates whether the value is returned, so toggling
  `active` would drop the key without skipping the `sim.render`.
- **Skipped substeps carry blank placeholder frames, and the key set NEVER
  changes.** `RoboCasaEnv.get_basic_observation` backfills a blank frame for any
  `*_image` key robosuite omitted, before any processing, so every derived key
  (`res512_*`, `ego_view_*`) is produced normally. This is mandatory, not
  cosmetic: the vector-env observation concatenate indexes EVERY space key on
  EVERY step (`gymnasium/vector/utils/space_utils.py` on 1.x,
  `numpy_utils.py` on 0.29.1), so a missing key raises `KeyError` inside the
  worker. `gym.make`'s `PassiveEnvChecker` asserts the same equality but only on
  the FIRST step (it latches on `checked_step`), so it is a backstop, not the
  enforcer. Dropping the video keys on skipped substeps killed every
  AsyncVectorEnv worker in production.
  The placeholders are never read — `_get_obs` returns `self.obs[-1]`, which is
  always the forced end-of-chunk render.
- **Attribute probing must not use `hasattr` on the chain.** gymnasium 0.29.1
  (pinned by the collector venv, `setup_RoboCasa.sh`) forwards unknown
  attributes down the wrapper chain and warns on every lookup; 1.x dropped
  forwarding. `MultiStepWrapper._declares` inspects `type()`/`__dict__` instead,
  which is silent and correct on both.
- **Guards** (all raise at construction): `video_horizon` must be 1, since a
  longer horizon reads earlier substeps; an env in the chain must implement the
  two methods — resolved by walking `.env` because `gym.make` wraps the base env
  and gymnasium ≥ 1.0 dropped `Wrapper.__getattr__` forwarding; and no wrapper in
  the chain may be marked `consumes_every_substep_obs` (`VideoRecordingWrapper`
  is, since it reads every substep's `video.*` keys).
- Both robocasa copies carry the two methods: `external_dependencies/robocasa`
  (installed into `gr00t/eval/sim/robocasa/robocasa_uv/.venv`, the venv the
  collector subprocess uses) and `external_dependencies/robocasa-gr1-tabletop-tasks`.
- Escape hatch: `skip_intermediate_render=False` in the config, or
  `--no-skip-intermediate-render` on the collector CLI.
- Tests: `test_skip_intermediate_render.py` — render counting, frame provenance
  and dtype against a `skip=False` baseline driven through the real `Observable`
  state machine, early-exit re-render, restore-on-exit, `SyncVectorEnv`
  round-trip. Runs without MuJoCo.

### Dropping unused video keys (`dropped_video_keys`)

RoboCasa's `GrootRoboCasaEnv` emits full-resolution passthrough copies next to
the keys the model consumes: the base copy adds `video.res512_image_*` beside
every `video.res256_image_*`, and the GR1 path adds
`video.ego_view_res1280x800_freq20`. **Nothing reads them** — the processor
indexes `images[view]` only for the embodiment's configured modality keys
(`processing_gr00t_n1d6.py:403-412`), which for PandaOmron are the three
`res256_*` frames — but they are ~80% of the per-chunk video bytes:

| per chunk (PandaOmron) | bytes |
|---|---|
| 3 × `res256` (256²×3) | 0.59 MB |
| 3 × `res512` (512²×3) | 2.36 MB |

The collector drops them (substring match) at the only two points where
observations leave it: `_batch_per_env_obs` (everything sent to the policy
server) and `_extract_video_single` (everything written to an `.npz`). The env's
declared observation space is untouched, so the vector-env key-set contract is
unaffected.

That shortens four things at once: the npz write, the trainer-side read-back
(~30 s/iter on a 48-episode buffer), the ZMQ round trip on every outer step
(3 × 512²×3 × num_envs per query), and the trainer's resident heap — the last
being what pushes MuJoCo workers into swap and, per
`_release_memory_to_os`, makes collection "2-3x slower than its non-swapping
baseline".

- Behavior-preserving: extra keys were always ignored by the processor. If a
  future checkpoint does consume one, the policy server raises on the missing
  modality key — loud, not silent.
- Old cached episodes still load: `EpisodeBuffer.load_episodes` discovers camera
  names from the npz keys (`episode_buffer.py:247-253`), so a mix of 6-key and
  3-key episodes is fine.
- Set `dropped_video_keys=[]` (or pass `--dropped-video-keys` with no values) to
  restore the previous behavior.

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
- **Sparse binary reward only.** From a hard saved state, binary-only reward
  can produce dead groups early (every rollout fails identically → std=0 →
  zero advantage → no learning). Pick a branch point the policy already
  solves at least intermittently, or the iteration yields no gradient signal.
  Note `include_anchor_groups` does not help here: it rescues the all-SUCCESS
  end of the distribution, not the all-fail end.

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
   reward std > 0 under the sparse binary reward with time-scaling
   disabled), or
2. `group_idx >= max_groups` (hard cap, logs a WARNING).

The "alive" predicate matches the trainer's **improvement**-signal filter
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

**`include_anchor_groups` does not change this gate.** All-success groups do
train under that flag (as anchors — see "Anchor groups"), but they carry no
within-group contrast, so counting them as alive would stop dynamic collection
early at high success, exactly when mixed groups are scarcest and extending
toward `max_groups` matters most. "Alive" stays mixed-only in both
`collect_episodes.py` and `_validate_collected_data_cache`.

To disable dynamic collection entirely, set `min_alive_groups = 0` —
the collector then always stops at exactly `num_groups`.

Constraints (enforced in `GRPOConfig.__post_init__`):

- `max_groups >= num_groups`
- `max_groups <= 100` (seed-stride collision boundary)
- `min_alive_groups <= max_groups`

Subprocess timeouts auto-scale at 7 min/group:
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
    _collect_episodes(env_name)                            # via collect_episodes.py subprocess

    # Phase 2: compute advantages
    buffer.compute_advantages(max_steps, anchor_advantage, ...)  # per-group z-score

    # Phase 2b: pre-compute reference log-probs (current model == ref before update)
    _compute_ref_log_probs()                               # caches backbone features

    # Phase 3: GRPO update
    _grpo_update()                                         # update_epochs × minibatches

    # Phase 4: log + checkpoint
    if iteration % save_interval == 0: _save_checkpoint(...)
```

Each phase is timed and logged to TensorBoard:

| scalar | phase |
|---|---|
| `time/iteration_seconds` | `iter_start` → end of Phase 3 (excludes logging + checkpointing, which are sampled after it) |
| `time/collect_seconds` | Phase 1 total |
| `time/collect_rollout_seconds` | collector subprocess (imports + worker spawn + rollouts + npz writes) |
| `time/collect_load_seconds` | trainer-side npz read-back into the buffer |
| `time/advantage_seconds` | Phase 2 |
| `time/ref_logprob_seconds` | Phase 2b |
| `time/update_seconds` | Phase 3 |

`collect_rollout + collect_load` is slightly LESS than `collect`: Phase 1 also
covers `buffer.clear()`, `_prune_old_episode_dirs()` (an `rmtree` of aged iter
dirs) and the stale-`.npz` unlink before the subprocess starts. Likewise
`collect + advantage + ref_logprob + update` is slightly less than
`iteration_seconds`, whose remainder is Phase 0 (`_release_memory_to_os()`: two
`gc.collect()` passes + `malloc_trim`) plus the per-iter task/LR setup. Treat the
residual as "untimed glue", not as a missing phase.

A NaN sub-phase is skipped rather than logged as 0, so cached-episode iters
(`resume_from_collected_data`) show a clean gap on `collect` / `collect_rollout`
instead of dragging the autoscale toward zero — on those iters `collect_load` is
the only Phase 1 curve with data. Covered by `test_phase_timing_logs.py`.

### Reward → advantage

```
reward = float(success)                                  # sparse binary (1.0 on success)
scaled = reward / num_steps * max_episode_steps          # faster = better (currently DISABLED)
A_episode = (reward - group_mean) / (group_std + 1e-8)   # PER GROUP
A_chunk = A_episode / num_chunks_in_episode
```

- The reward is **sparse binary**: `1.0` on task success, `0.0` otherwise.
  There is no reward shaping — the codebase does not compute dense progress.
- Time-scaling (`/ num_steps * max_episode_steps`) would make faster solutions
  get larger reward, creating advantage variance even within all-success
  groups. It is currently **DISABLED** in `compute_advantages`
  (see the block comment there for the ablation rationale); the reward fed to
  the group-relative normalization is the raw binary value.
- `A_chunk = A_episode / num_chunks` preserves the within-group
  zero-sum invariant at the chunk level, so every trajectory contributes
  equal **total** gradient weight regardless of length.

A group with reward std < 1e-4 is **degenerate**: the group-mean baseline gives
every episode an advantage of exactly 0. Under the binary reward this happens
for all-success groups (every rollout succeeded) and all-fail groups (every
rollout failed) — only **mixed** groups produce an improvement gradient. This is
not a threshold artifact: the per-group std is either exactly 0 (all G outcomes
identical) or at least `1/sqrt(G)`, which is 3500× the threshold at G=8, so the
`std_r < 1e-4` test is an exact "were all outcomes the same?" check.

By default degenerate groups are **dead**: their chunks are filtered out before
any forward pass (see "Minibatch construction"). `include_anchor_groups`
reclassifies the all-success half as **anchor** groups instead — see the next
section.

### Anchor groups

An all-success group being zero-advantage is correct policy-gradient behavior:
the group mean *is* the Monte-Carlo value estimate, and no rollout beat it, so
there is nothing to improve. But dropping those groups has two costs:

1. **The trust region never covers the solved states.** `kl_coef_last_iter` and
   `kl_coef_base_model` are evaluated only over live chunks, so the constraint
   binds where the policy is uncertain and is blind to where it succeeds.
   Because group seeds are fresh every iteration
   (`seed + iter*100_000 + group_idx*1000`), the anchor states are *different
   scenes* from the live ones, so admitting them genuinely widens the
   constraint's support.
2. **At high success most of the buffer disappears**, and what survives is
   dominated by rare failures: in a 7/8 group the single failure carries
   −2.47 against successes at +0.35. `balanced_minibatch_training` and the
   tent-shaped epoch decay both exist to damp that asymmetry by reweighting
   scarce data; anchor rows instead restore positive mass that is *real*.

Three-way classification in `compute_advantages` (k = successes, G = group size):

| | condition | advantage | role |
|---|---|---|---|
| **signal** | `0 < k < G` | `(r − mean) / std_r` — formula untouched | improvement |
| **anchor** | `k == G` | `anchor_advantage` (constant) | retention |
| **dead** | `k == 0`, or `G == 1` | 0, filtered | — |

**All-fail groups stay dead, deliberately.** Pushing down on every rollout from
a state gives no target to move toward, and it is the avoidance gradient the v2
ablation identified as the collapse mechanism. Run the pseudo-count baseline
below on a `k == 0` group and it hands every episode a *negative* advantage, so
the asymmetry falls out of the math too — it lives in one `if`.

#### Choosing `anchor_advantage`

`k == G` is not proof that `p == 1`: at G=8 a state with true p=0.85 returns 8/8
about 27% of the time (0.85⁸), so the MLE baseline 1.0 over-estimates and each
success really did earn positive advantage. Replace the group mean with the
Beta-Bernoulli posterior mean under κ pseudo-counts at prior success rate p̄,
and divide by a fixed scale (the group's own std is 0):

```
b_g      = (Σ r_i + κ·p̄) / (G + κ)
A_anchor = (1 − b_g) / σ_fixed  =  κ(1 − p̄) / ((G + κ)·σ_fixed)
```

κ is "how many imaginary rollouts my prior is worth"; p̄ is "what success rate
they had". With κ=2, p̄=0.5 (Laplace's rule of succession) and σ_fixed=0.5 (the
max Bernoulli std, ≈ the std of a balanced G/2 group) this is `2/(G+2)`:

| group_size | `anchor_advantage` (κ=2) | balanced-group success, for scale | weakest signal row |
|---|---|---|---|
| 8 | **0.200** | ±0.935 | ±0.354 |
| 12 | **0.143** | ±0.957 | ±0.289 |
| 16 | 0.111 | ±0.968 | ±0.250 |

Those comparisons are at the **episode** level (`A_episode`), which is where the
value is set. What a row contributes also passes through `÷ num_chunks` and the
iteration-wide scale, so the realized row-level ratio varies with group
composition and episode length — larger against a lopsided group whose
advantages are small, smaller against a balanced one. That is the intended
behavior for a fixed absolute magnitude.

Today's dead-group behavior is the κ=0 case. The correction shrinks as G grows —
more real evidence, less prior — which is what makes it a finite-sample
correction rather than a bonus. Since κ, p̄ and σ_fixed are only identifiable as
this one combination, the value is configured directly; recompute it if you
change `group_size` (κ=3 at G=12 reproduces the G=8, κ=2 magnitude if you want
to hold gradient scale fixed across a group-size change).

It is deliberately **not** tied to the running success rate: the estimator wants
the anchor to fade as success climbs, while the negative-mass asymmetry wants it
strongest exactly then. A fixed value keeps the effect readable and leaves the
asymmetry to the balanced-sampler mechanisms.

`anchor_advantage = 0` with `include_anchor_groups = True` is the KL-only
setting — the rows join the batch and the trust region, but their clip term is
identically 0, so they carry no reward signal of their own. Not a literal no-op,
though: the rows occupy minibatch slots, which changes each batch's renorm sample
and raises the per-iteration step count.

#### Bound

With A > 0 the surrogate is `min(A·ρ, A·(1+clip_eps_high))`, so an anchor row's
gradient dies once ρ exceeds 1.2 — the FM surrogate can improve by at most
log(1.2) ≈ 0.18 nats on that path per iteration, regardless of the constant.
`train/mean_ratio_anchor` saturating near `1 + clip_eps_high` means the clip is
bounding the retention move, which is the designed cap.

#### What anchor rows are excluded from

Anchor rows are a third class, so every mechanism *defined by advantage sign*
skips them. Getting any of these wrong silently defeats the feature:

| mechanism | why anchors are excluded |
|---|---|
| per-minibatch z-score | An anchor-only minibatch has no variance except `anchor_advantage / num_chunks` — i.e. **episode length**. A z-score there amplifies length to ±1 and reproduces the time-scaling gradient that collapsed v2. In a mixed batch, all-positive anchor rows also lift the mean and can flip weak real positives negative. |
| `buffer_adv_mean` / `buffer_adv_std` | Computed over signal rows only, so the mean stays ≈0 and `per_iteration_advantage_norm` keeps its sign-preservation property. |
| balanced sampler pos/neg pools | At high success anchors would *dominate* the positive pool and crowd out the genuine mixed-group successes the sampler exists to preserve, while inflating `natural_pos_frac`. |
| dynamic-epoch `success_frac` | Every anchor episode succeeded, so counting them drives the tent toward 1 epoch exactly when anchors were added. |
| PAWS `N`/`D` alive mass | Inflating D drives k → 1 and silently disables the mechanism. |
| pos/neg clipfrac buckets, `n_pos_flipped_by_renorm`, `mean_ratio_fixed/jitter` | All keyed on group-relative sign. Anchors get their own `*_anchor` curves instead. |
| `ref_mse/*`, `chunk_gap/*` | Split by advantage sign; anchor rows are simply dropped from these diagnostics (no `*_anchor` counterpart). |
| jitter (`jitter_pos`/`jitter_neg`) | Anchor entries are always tagged `"fixed"` — λ is selected by advantage sign. `_jitter_gap_diagnostics` takes the jitter set as an explicit mask rather than `~fixed_row_mask`, since excluding anchors from one mask would otherwise sweep them into the other. |
| balanced-sampler viability | The anchor reservation shrinks the sampler's batch size, which can round the minority slot count to 0 and make `_iter_balanced_minibatches` fall back to stratified. At the default `balanced_minibatch_positive_adv_ratio=0.5` this needs `signal_mb_size == 1`, i.e. `anchor_slots == mini_batch_size − 1`. The pool ratio that implies **scales with `mini_batch_size`** — measured first fallback at 1.05:1 (mb=4), 3.05:1 (mb=8), 5.05:1 (mb=12), 7.05:1 (mb=16) — so it is roughly `(mini_batch_size − 1):1`, not a constant. It IS reachable at the default `anchor_max_row_frac=1.0`: that budget caps *chunks*, but the one-whole-episode floor below can admit an anchor episode several times larger than the cap, so a small mixed group plus one 65-chunk anchor episode reaches 6.5:1. Do not confuse this with the ~7:1 *coverage* limit below; they are different thresholds. The fallback logs a WARNING either way. |

The headline `clipfrac` / `mean_ratio` / `mean_log_ratio_abs` curves are the
deliberate exception: they cover **all** trained rows, anchors included, because
they describe the batch the optimizer saw. Use `clipfrac_effective_{pos,neg}`
for the signal-only view and `train/mean_ratio_anchor` for the anchor split.

Anchor rows instead get their **scale** from the buffer-wide signal std
(`anchor_scale`), so an anchor row's weight does not depend on which rows happen
to share its minibatch. It does still depend on the iteration's signal spread
(`buffer_adv_std` is per-iteration): the anchor is a fixed *absolute* magnitude,
so its weight relative to the signal rows grows as the signal advantages shrink
— which is what happens at high success, and is the direction you want. Anchor
rows also never get the mean subtracted, only the scale divided.
`per_iteration_advantage_norm=True` is the intended pairing: under per-minibatch
norm the signal rows rescale per batch while anchors don't, so the ratio wobbles
with batch composition (the startup banner warns about this).

#### Row budget and cost

Each anchor row costs the same `len(tau_centers)` DiT forwards as a signal row
in the ref pass (×2 with the base-model KL) and in every update epoch. At high
success they can be a large fraction of the buffer, so `anchor_max_row_frac`
caps anchor chunks at that multiple of the signal chunk count — one knob for
both compute and the anchor's share of the gradient. Anchor episodes are kept in
index order — first-fit, so an episode that doesn't fit is skipped and a later
shorter one may still be admitted — until the budget is met, and the rest revert
to dead, logged rather than silently dropped. The budget has an implicit floor of
one whole episode: the first anchor episode is always admitted so a small value
shrinks the anchor share instead of deleting the feature, which at ~30–65
chunks/episode can overshoot a small budget several-fold. Because anchor advantages are constant rather than
zero-sum within a group, dropping individual anchor episodes distorts nothing —
unlike a signal group, where it would break `Σ A_ep = 0`.

The budget is **waived when there are no signal chunks at all**: there is no
denominator to measure it against. That is not only the all-success case — an
all-fail *plus* all-success mix also has zero signal chunks while carrying a
non-zero `std_reward`, so the outer skip doesn't fire either. The waiver is
logged, because it means `anchor_max_row_frac` is bounding nothing that
iteration.

Two more bounds worth knowing:

- Above roughly `anchor_max_row_frac ≈ 7` the per-batch cap (`anchor_slots ≤
  mini_batch_size − 1`) stops the epoch from covering the pool: measured coverage
  is 1.00× up to a 5:1 anchor:signal ratio, then 0.70× at 10:1 and 0.35× at 20:1.
  Far outside the default of 1.0, but the excess rows are simply never trained.
- The budget admits episodes in index order, so a small value systematically
  favours the lowest `group_id`s — the earliest-collected groups. Deterministic
  across runs; group seeds rotate per iteration, so it doesn't compound.
- **`anchor_max_row_frac` is therefore not a hard cap.** The one-episode floor
  admits the first anchor episode whatever its size, so the realized ratio can
  exceed the configured budget several-fold (measured 3.25× with a 10-chunk
  episode against an 8-chunk signal pool at `frac=0.5`). Reason about it as a
  target, not a bound — and note the interaction with first-fit: the overshoot is
  worst when a long episode sorts first, and the shorter episodes that would have
  fit are then dropped.

The iteration skip is keyed on `n_signal_chunks` / `n_anchor_chunks` rather than
`std_reward` — see "Skip semantics". An iteration with no signal chunks trains on
its anchor rows when `anchor_advantage > 0` **or** `kl_coef_base_model > 0`; with
both at 0 it has no gradient at all (clip term identically 0, `KL(ref ‖ current)`
zero at `θ == θ_ref`) and stays skipped rather than firing steps that would apply
only weight decay and carried momentum while consuming an iteration.

#### Additive, not diluting

`clip_loss` and both KL terms divide by `signal_mb_size` — the **intended**
signal-row count, held constant across the epoch — when anchor rows are present,
rather than by the total row count. Two consequences:

- A signal row's weight is `1/signal_mb_size`, exactly what it would be in an
  anchor-free minibatch of that size, so turning anchors on doesn't rescale the
  rows that drive improvement, and the anchor KL genuinely *adds* a constraint
  rather than reallocating the existing KL budget across more rows.
- Using the **realized** signal count instead would spike any batch the sampler
  under-fills: a trailing batch with 1 signal + 3 anchor rows would weight every
  row at 1.0 instead of `1/signal_mb_size`, making a 4-row batch the largest step
  of the epoch and — at `max_grad_norm=0.5` — the only clipped one. A constant
  divisor makes a row's weight independent of batch composition; an under-filled
  batch simply contributes proportionally less.

A minibatch left with fewer than 2 signal rows also can't support a
per-minibatch z-score, so its signal rows fall back to the buffer-wide one
rather than entering the surrogate at raw `A_ep / num_chunks` scale.

The divisor is gated on **anchors being enabled this iteration**, not on "this
minibatch happens to hold an anchor row". Because the quota is fractional, the
credit accumulator leaves some batches anchor-free; gating per batch would send
those through `.mean()` and put the composition-dependent weight straight back —
a 1-signal-row trailing batch would weight its row at 1.0 instead of
`1/signal_mb_size`, and whether it did would be decided by the credit counter.

Two caveats on "additive". The loss *weights* are additive as described, but each
anchor row still carries full `1/signal_mb_size` KL weight, so a batch's pre-clip
gradient norm rises by roughly `1 + n_anchor/signal_mb_size`. At the default
`max_grad_norm=0.5` that can put a batch into active clipping, and the rescale
then applies to the signal gradient too. And anchor rows occupy slots, so the
per-iteration optimizer step count rises (see above).

With no anchor rows in the iteration the expression is `row_loss.mean()`,
bit-identical to the pre-anchor path.

#### Metrics

`episode/n_anchor_groups`, `episode/n_anchor_episodes`,
`episode/n_anchor_episodes_dropped` (only with the flag on — `buffer.stats()`
reports the counters unconditionally, so the wandb bulk-dump strips them too);
`train/n_anchor_rows_trained`, `train/mean_ratio_anchor`, `train/kl_loss_anchor`
(only when anchor rows actually trained, and dropped rather than written if
non-finite). `episode/n_live_groups` still counts **signal** groups only, and
`mean_advantage` / `std_advantage` / `pct_positive_advantage` are still computed
over signal episodes only.

Caveats on cross-run comparability, since not every pre-existing curve is
untouched:

- `clipfrac`, `mean_ratio` and `mean_log_ratio_abs` cover anchor rows too (the
  deliberate exception noted above).
- `train/loss`, `train/clip_loss`, `train/kl_loss_last_iter` **and
  `train/kl_loss_base_model`** switch from `.mean()` to
  `.sum() / signal_mb_size` whenever anchors are in play, so their magnitudes are
  not directly comparable to an anchors-off run.
- `episode/n_dead_groups` falls (an anchor group is no longer dead — inherent to
  the feature), and `episode/pct_positive_advantage` / `episode/std_advantage`
  shift because their denominator is the non-anchor episodes: anchor episodes
  leave the sample entirely rather than contributing zeros as they did when they
  were dead. Only `episode/n_live_groups` and `episode/mean_advantage` are
  numerically preserved.
- `train/kl_loss_anchor` covers the `kl_coef_last_iter` term only; the anchor
  rows' base-model KL contribution is not surfaced separately.
- `train/ratio_max`, `train/ratio_min` and `train/grad_norm_*` also shift, since
  anchor rows enter the ratio extremes and the rescaled loss.
- Budget-DROPPED anchor episodes revert to `is_anchor=False` with advantage 0, so
  they still contribute zeros to `pct_positive_advantage` / `std_advantage`. Only
  admitted anchor episodes leave that sample.
- `train/kl_loss_anchor` is a per-anchor-row mean, whereas the `kl_loss_last_iter`
  term inside the loss divides by `signal_mb_size`. Similar names, different
  normalizations — don't read them side by side as one quantity.

#### Suggested ablation

`init_state_npz_path` single-scene mode is the right harness — it is where
v2/v3 ran, where all-success groups are pure "G/G noise draws succeeded", and
where the buffer actually goes empty at high success. Ladder:
`anchor_advantage` ∈ {0 (KL-only), 0.10, 0.143, 0.25} at `group_size=12`,
against the v3 baseline that held at 0.83. Watch whether success *holds above*
the prior plateau rather than merely reaching it; watch
`train/mean_ratio_anchor` for clip saturation; and watch
`episode/group_success_{min,median,max}` spread for the one genuinely uncertain
risk — reinforcing the model's own (ε → a) mappings is reflow-style
self-distillation, and since all exploration here comes from denoising noise
through a shared DiT, over-sharpening on solved states could shrink within-group
variance everywhere.

#### What not to do instead

Do not create within-group variance among the successes to make the group
non-degenerate. That includes the capped speed multiplier suggested in
`compute_advantages`' block comment: capping the reward at
`min(1.5, max_steps/num_steps)` bounds the reward *spread*, but the advantage
divides by the group's own std, which rescales whatever spread survives back to
±1. An all-success group with rewards in [1.0, 1.5] and std 0.15 yields
advantages of ±1.7 — the same magnitude as real succeed-vs-fail signal, so "be
20% faster" gets weighted like "succeed instead of failing". That is the v2
mechanism exactly, cap or no cap. A speed term would need a fixed-scale
denominator, not the per-group std.

#### CLI usage

```bash
# Layer 1 — retention constraint only. Anchor rows join the batch and the KL
# terms; their clip term is identically 0, so they cannot move the policy.
uv run python scripts/grpo/train_grpo.py --include-anchor-groups

# Layer 2 — add the positive pull. 0.143 is the kappa=2 value at group_size=12.
uv run python scripts/grpo/train_grpo.py \
    --include-anchor-groups \
    --anchor-advantage 0.143 \
    --per-iteration-advantage-norm \
    --group-size 12 \
    --anchor-max-row-frac 0.5
```

`--anchor-advantage` without `--include-anchor-groups` is a hard config error
rather than a silent no-op. The startup banner prints
`Anchor groups: ON (advantage=…, row budget=…× signal rows)`, plus a NOTE when
a positive advantage is combined with per-minibatch renorm.

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

1. **Dead-group filter**: drop every chunk with `|advantage| < 1e-12` (advantage
   was set to literal 0 by `compute_advantages` for groups with std < 1e-4),
   keeping anchor chunks when `include_anchor_groups` is on. Filtering here
   keeps every minibatch uniformly live-only and avoids a `(0 - mean) / std`
   term polluting the per-minibatch advantage renorm.

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

**Anchor rows** (see "Anchor groups") are appended by `_with_anchor_rows`
*around* whichever sampler ran, not inside it — so both sampler paths stay
signal-only and are a transparent pass-through when no anchors exist. The inner
sampler is driven at `mini_batch_size - anchor_slots` (both samplers take an
optional `mb_size` override), so total rows per minibatch — and hence peak VRAM —
stay at `mini_batch_size`. At least one signal slot is always reserved: at
`anchor_max_row_frac` large enough for the anchor pool to dwarf the signal pool,
the proportional quota would otherwise reach `mini_batch_size`, leaving the inner
sampler a batch size of 0 — which its `mb_size or config.mini_batch_size` default
silently turns back into `mini_batch_size`, overfilling every minibatch. At
`mini_batch_size = 1` there is no room for both, so the anchor rows are skipped
with a WARNING rather than exceeding the budget.

The quota may be FRACTIONAL: `_with_anchor_rows` carries a credit accumulator
and emits `floor(credit)` rows per batch, so one epoch consumes the anchor pool
about once. Flooring it at one row per batch instead would ride a small pool
along in every minibatch — 1 anchor chunk against 100 signal chunks would train
~15× per epoch while every signal row trains once. Within a batch the pool is
drawn without replacement even across a reshuffle, so a chunk is never served
twice into the same minibatch.

Two details make the realized share match the pool share rather than merely
approximating it:

- The reserved slot count is chosen by solving for **delivery capacity**: the
  smallest `slots` whose `slots × n_batches` covers the pool, where `n_batches`
  comes from `_min_expected_batches` (which models both samplers, including the
  balanced one's early termination and both of its fallbacks). `ceil(target)`
  alone is not enough — it implicitly assumes the *stratified* batch count, and
  on the balanced sampler the smaller realized count makes the target exceed the
  cap, pinning every batch. Measured 0.83–0.91× delivery in the band where the
  ceil lands on 1. The reservation is only an estimate; `_with_anchor_rows`
  measures the real count and WARNS if capacity still fell short.
- The target is `pool / n_batches` against the batch count the sampler
  **actually produced** — `_with_anchor_rows` materializes the epoch's batches
  before distributing. Estimating it does not work: `ceil(len(entries) /
  signal_mb_size)` is the *stratified* count, while `_iter_balanced_minibatches`
  (the default) stops early once its majority pool drains, and its fallbacks
  change the count again. Under that estimate a 1-chunk pool trained **zero**
  rows on the balanced path. Measured exposure is now 1.00× on both samplers for
  every pool up to the coverage limit below (~7:1 anchor:signal); above that the
  per-batch cap binds and delivery falls off as documented, with a WARNING naming
  the shortfall.

Under `jitter_paired=True` the anchor share is computed against an entry pool
that jitter has doubled, so the realized anchor:signal **mass** ratio is about
half what the same `anchor_max_row_frac` gives with jitter off. Preserving both
that ratio and 1× exposure per epoch is not possible — pairing doubles signal
mass without doubling anchor mass — so exposure is preserved and the ratio moves.
Raise `anchor_max_row_frac` (or `anchor_advantage`) if you want the same anchor
pressure under paired jitter.

When there are no signal rows at all, the anchor entries go through the
stratified sampler directly at full `mini_batch_size`.

**Anchor rows raise the per-iteration optimizer step count.** They occupy
minibatch slots, so the signal rows spread over more batches — up to ~2× the
steps at the same LR when the anchor pool matches the signal pool. Same caveat as
`jitter_paired`'s 2× warning; the startup banner states it. Lower
`update_epochs` or `anchor_max_row_frac` to match an anchors-off baseline's step
budget.

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
entirely absent (all episodes fail or all succeed within live groups). Anchor
rows are not in either pool — the caller holds them out and appends them as a
separate quota (see "Anchor groups").

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

**Relationship to Jitter-GRPO.** With paired jitter active (`jitter_pos` or
`jitter_neg` > 0 and `jitter_paired=True`), `entries` is doubled (`fixed +
jitter` copies of each chunk). Both copies of a positive chunk are independent
entries in the positive pool. The balanced sampler draws from them in shuffled
order; the Jacobian regularizer accumulates at epoch granularity (not within a
single mini-batch), so the pairing requirement is satisfied regardless of
whether fixed and jitter copies land in the same batch. In jitter-only mode
(`jitter_paired=False`) each chunk contributes a single `jitter` entry, so the
pool is the same size as vanilla. The combination of these features is sound.

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
  sparse. Anchor groups are excluded for the same reason — every one of their
  episodes succeeded, so counting them would drive `success_frac` toward 1 and
  collapse the tent to 1 epoch exactly when anchors were added.
- `successful_eps` counts live-group episodes with **positive advantage**
  (`self.buffer.advantages[i] > 0`), not `ep.success`. Under the sparse
  binary reward these coincide for live (mixed) groups — a group's successes
  get positive advantage, its failures negative — so this equals counting
  `ep.success`, while keeping the epoch formula consistent with mechanism 1,
  which oversamples chunks with `c.advantage > 0`.

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

#### Files touched (anchor groups)

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `include_anchor_groups: bool = False`, `anchor_advantage: float = 0.0`, `anchor_max_row_frac: float = 1.0` with `__post_init__` validation (non-negative advantage; advantage > 0 requires the gate; positive row budget). |
| `episode_buffer.py` | `compute_advantages` takes the three knobs and classifies signal / anchor / dead; `_resolve_anchor_groups` applies the row budget at episode granularity. `GRPOEpisode.is_anchor` / `ActionChunk.is_anchor` carry the flag; `_build_chunks` propagates it. `stats()` adds `n_anchor_{groups,episodes,episodes_dropped}`, keeps `n_live_groups` and the advantage summaries signal-only. `__main__` self-test covers the classification. |
| `train_grpo.py` | `train()` passes the knobs through and makes the `std_reward < 1e-8` skip anchor-aware. `_compute_ref_log_probs` admits anchor chunks (gated on config) and passes signal-only chunks to `_summarize_ref_mse` / `_per_chunk_gap_survey`. `_grpo_update_inner` splits live chunks into signal/anchor, computes `anchor_scale`, excludes anchors from the renorm statistics / sign masks / PAWS mass / sign-keyed metrics, divides the loss by the signal row count, and appends the anchor quota via `_with_anchor_rows`. Both samplers take an optional `mb_size`. `_log_metrics` emits `episode/n_anchor_*` and `train/{n_anchor_rows_trained,mean_ratio_anchor,kl_loss_anchor}`. |
| `collect_episodes.py` | Comment only — the `min_alive_groups` "alive" predicate stays mixed-only. |
| `test_anchor_groups.py` | Buffer classification, row budget, config validation, and the real `_grpo_update_inner` on CPU: bit-identity with no anchor rows, renorm isolation, the anchor-only iteration, additive KL, and the PAWS / balanced-sampler / dynamic-epoch exclusions. |

---

### PAWS: dynamic positive-advantage weighting

`positive_advantage_weight_scaling` scales the per-row clip loss on group-good
rows by a live factor `k`, chosen so that reinforcement mass is
`positive_advantage_weight_target_ratio` times erosion mass:

```
N = alive erosion       = sum |row_loss| over negative-advantage rows still
                          passing gradient (dead iff ratio < rho_floor_i, the
                          row's own lower bound — flat at 1 - clip_eps_low
                          unless clip_low_mse_coef > 0)
D = alive reinforcement = sum |row_loss| over amplified positive rows still
                          passing gradient (dead iff ratio > 1 + clip_eps_high)
k = clamp(target_ratio * N / D, 1.0, positive_advantage_weight_max)
                          # lower clamp becomes target_ratio with
                          # paws_k_floor_at_target=True
```

Mass is measured on the **unweighted** row loss, so the estimate never feeds
back on `k`. Anchor rows are in neither term. Both terms pool per **trained**
micro-batch across the whole iteration.

**Read `pos_adv_realized_ratio`, not `pos_adv_weight_k`** — but read it as a
coarse "which side is this iteration pushing on", not as a precise estimator of
`target_ratio`. It is `Σ kᵢ·Dᵢ / Σ Nᵢ`, pooled per micro-batch at the `k` that
micro-batch was actually weighted by:

- `≈ 1.0` → the two sides are balanced, i.e. the mechanism is off in effect.
  When `target_ratio > 1` this is the reading to worry about: it is what the
  removed cross-iteration EMA produced on every resume, survivable at 52 %
  success and fatal (0.67 → ~0.04 in one iteration) at 67 %. **At the config
  default `target_ratio = 1.0` it is instead the on-target reading** — the two
  cases are only distinguishable by knowing `target_ratio`.
- `>> target_ratio` → erosion is largely clip-dead (`N << D`). The raw ratio
  *diverges* as `N → 0` (measured 12–15 in that regime), so the emitted value is
  clamped to `positive_advantage_weight_max` to keep the curve readable; the
  unclamped terms are always available as `pos_adv_pos_mass` /
  `pos_adv_alive_neg_mass`. Cross-check `clipfrac_effective_neg` and `k_min`.

**Deviation from `target_ratio` does not by itself mean a clamp is binding.**
Each `kᵢ` is a *prefix* estimate (the pool excluding its own micro-batch), so
when the running prefix ratio differs from the whole-iteration ratio the pooled
result drifts off target with no clamp involved — measured +9…+20 % on skewed
group shapes with `k` comfortably inside `[1, max]`. On the 238-micro-batch
reference iterations, where the prefix is stable, it read 1.7500–1.7501 against
a target of 1.75. Use `k_min` / `k_max` to tell a clamp from prefix drift.

`k` itself is a poor headline because it moves for a benign reason: under
per-minibatch renorm the z-score forces `Σ_{post>0}|A| ≡ Σ_{post≤0}|A|`. `N` and
`D` are keyed on the **pre**-renorm sign, so that gives `N/D ≡ 1` only absent
renorm sign flips (a flipped row falls out of *both* masses; one pos→neg flip in
an 8-row minibatch measures `N/D ≈ 0.88`) — watch
`n_pos_flipped_by_renorm`, which read 0 on 15 of 16 iterations of the reference
run. There `N/D` sat at `exp(jitter/gap_pos)`, measured 1.0464 ± 0.0057 over
every non-clipping iteration and matching `exp(gap_pos)` to within 0.5 %, so
`k ≈ target_ratio·1.046`. When drift starts clip-killing negatives, `N` falls and
`k` falls with it — the mechanism correctly tracking a real drop in erosion, not
the mechanism weakening. In that same run `k` slid 1.83 → 1.49 over the last
three iterations while the realized ratio never left 1.750.

`pos_adv_weight_k_{min,max}` bracket the **measured** `k`s (the unmeasured prior
is excluded — it is a config-derived constant, and folding it in would pin
`k_min` to it and hide the real spread). They are absent on an iteration that
never measured. Together with `k_last` they separate a clamp from prefix drift,
and they surface a mid-iteration excursion — e.g. a run of one-sided minibatches
pinning `k` at the cap — that `k_last` alone would miss.

**No cross-iteration state.** `k` is derived from the current iteration's pool
alone. Until the pool holds any amplified-positive mass there is nothing to
measure, so `k` falls back to the analytic prior `k = target_ratio` — the
fallback **tracks the target** instead of being pinned to `1.0` independently of
it. (At the config default `target_ratio = 1.0` the prior *is* 1.0 — correct,
since that config asks for equal masses.) There is no count-based warm-up beyond
that: the prior is unmeasured and is *not* floored by the measurement, so holding
it longer over-amplifies in the clip-dead-erosion regime, where the measured `k`
floors at 1.0 while the prior still says `target_ratio`. Under
`per_iteration_advantage_norm` the minibatch zero-mean identity does not hold, so
there is no prior to stand on and the fallback is `1.0`; that combination
measures much worse overall anyway (see "Gradient accumulation"). See
"Checkpointing & Resuming → Resume" for the resume bug this design replaced.

#### Flooring `k` at `target_ratio` (`paws_k_floor_at_target`)

`paws_k_floor_at_target = False` (default) keeps the historical lower clamp of
`1.0` on the **measured** branch:

```
k = clamp(target_ratio * N / D,  1.0 (default) | target_ratio (flag on),
          positive_advantage_weight_max)
```

**Why.** A tighter lower clip (`clip_low_mse_coef > 0`) deliberately kills more
negative rows, which shrinks `N`, which through `target_ratio · N/D` *lowers*
`k`. So tightening the erosion brake would also weaken reinforcement — the
opposite of the intent. Flooring at `target_ratio` removes only the "amplify
*less* than target" case; it never amplifies more than the measurement asks for,
because the `min(…, max)` cap is still applied afterwards.

Measured: `N/D` sits at **1.04–1.06** on healthy iterations, so
`target_ratio · N/D > target_ratio` and the floor is **inert** there; `N/D` falls
to **0.66** during collapse, which is exactly where it binds.

Two knife edges worth knowing:

- At exactly `N/D == 1` the `D_iter + 1e-8` denominator puts the measured `k` a
  part in 1e8 *below* `target_ratio`, so the floor binds by that much. Harmless,
  but it means "inert" is a statement about `N/D > 1`, not `N/D ≥ 1`.
- Validation rejects `paws_k_floor_at_target=True` with
  `positive_advantage_weight_target_ratio < 1.0`: that would pin `k` *under* the
  no-op point and force de-amplification, inverting the mechanism.

The other two `k` branches are **untouched**, and each for its own reason:

- The unmeasured prior (`D_iter == 0`) is already
  `min(max(target_ratio, 1.0), max)` — it already tracks `target_ratio`, so with
  `target_ratio ≥ 1.0` enforced there is nothing to change.
- The `per_iteration_advantage_norm` fallback stays `k = 1.0` because that path
  has **no prior to stand on** — the buffer-wide z-score breaks the minibatch
  zero-mean identity that makes `N/D ≈ 1`. Flooring an unmeasured, unjustified
  value at `target_ratio` would amplify on the strength of nothing.

---


**Two guards on the flag.** `positive_advantage_weight_target_ratio` must be
`≥ 1.0` (below the no-op point the floor would force de-amplification on every
healthy iteration) **and** `≤ positive_advantage_weight_max`. The second is not
cosmetic: `k = min(max(measured, target_ratio), max)`, so a target above the cap
collapses the expression to the constant `max` for *every* measurement — the
measurement-driven controller silently becomes a fixed amplifier, while the banner
advertises an inverted interval (`clamped to [5, 2]`) as if it were a range.

**`train/pos_adv_k_floor_binds_frac`** and **`train/pos_adv_k_cap_binds_frac`** give the
fraction of MEASURED micro-batches whose `k` the floor / the cap moved. Read them before
`k_min`/`k_max`: a `floor_frac` near 1 means `k_min` **is** the floor and carries no
information about `N/D` — exactly the ambiguity they resolve.

They replaced an earlier `pos_adv_weight_k_raw_min`, which had three defects. `clamp` is
monotone, so `k_min == clamp(k_raw_min)` and a **cap**-bound minimum could never satisfy
`k_raw_min < k_min` — the cap was structurally unreportable. It fired on default unfloored
configs too, since the historical floor is still `1.0`. And `k_raw` is a *prefix* estimate,
so a minimum over ~300 micro-batches selects the shortest, noisiest prefix and reads
exactly `0.0` whenever the first trained micro-batch had no alive negative rows — a
fabricated zero of exactly the kind this codebase avoids elsewhere.

### Clipped surrogate + KL

```
ratio = (current_log_prob - ref_log_prob).exp()
advantages = (A - A.mean()) / (A.std() + 1e-8)            # renorm per-batch
# rho_floor is a [B] tensor, == 1 - clip_eps_low on every row at the default
# clip_low_mse_coef = 0.0 (see "Per-row, MSE-referenced lower clip").
surr1 = A * ratio
surr2 = A * clamp(ratio, min=rho_floor, max=1 + clip_eps_high)
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

When anchor rows are present in the minibatch, all three `.mean()`s become
`.sum() / signal_mb_size` (a constant, not the realized row count) — see
"Anchor groups → Additive, not diluting". With no anchor rows the expression is
exactly the `.mean()` above.

NaN/Inf guard: a minibatch with non-finite loss (typically bf16 ratio
overflow when `|log_ratio|` is large) is **skipped**, the
`n_skipped_nonfinite` counter increments, and training continues.
`clip_grad_norm_` only bounds finite gradients — it does not rescue NaNs.
The guard fires BEFORE `backward()`, so a skipped minibatch never puts
anything into the gradient buffer (see "Gradient accumulation" for what that
means when several minibatches share one optimizer step).

Second, independent guard on the **gradient** side: if `clip_grad_norm_` reports
a non-finite norm — either because `backward()` produced inf/NaN even though the
forward loss was finite, or because the fp32 sum-of-squares of large-but-finite
gradients overflowed — the optimizer step is **dropped**, the gradient buffer is
zeroed, and `n_nonfinite_grad_steps` increments (with a console WARNING).
Clipping cannot save that buffer: `total_norm = inf` gives a clip coefficient of
0, so the buffer becomes either all-NaN (`inf * 0`) or exactly `0.0` (finite
gradients scaled by 0) — nothing to rescue either way. Stepping on the NaN case
would write NaN into every LoRA param, poison AdamW's moments for the rest of the
run, and — because the iteration would still report `n_updates > 0` — persist a
NaN checkpoint that a later `--resume-from` would load, all while `grad_norm_*`
still looked normal (the offending norm is excluded from that average). Dropping
the step instead leaves the weights at their last good value and training
continues with the next window. Expected reading is a flat
`train/n_nonfinite_grad_steps == 0`; anything above zero is worth investigating
even though the run survives it.

If ZERO minibatches commit a gradient step in an iteration (every batch
non-finite, every window dropped, or every group dead), the iteration is
treated as **skipped** and the resume checkpoint is saved under the last
successfully-updated iter's name (see "Checkpointing").

### Per-row, MSE-referenced lower clip (`clip_low_mse_coef`)

`clip_low_mse_coef = 0.0` (default) is OFF and bit-identical to a flat
`1 − clip_eps_low` floor on every row. `test_clip_floor.py` asserts the
flags-off path's DETERMINISM and the additivity of the new keys; the
bit-identity-vs-baseline check is an **out-of-tree differential** (recipe in that
test's docstring) — it cannot live in-tree, because once this change is committed
`HEAD` contains it.

**Why a flat `clip_eps_low` is the wrong shape.** The importance ratio is
`ρ = exp(MSE_ref − MSE_θ)`, and `MSE_θ ≥ 0`, so the whole reachable range is
`ρ ≤ exp(MSE_ref)` — measured 1.002–1.03 against `1 + clip_eps_high = 1.2`.
Consequences, both measured:

- **Positive-advantage rows are never clipped.**
  `train/clipfrac_effective_pos` is identically 0 in 69 of 69 logged
  iterations. The negative branch is the only one with room to run, so the
  lower clip is the *only* live brake in the objective.
- **A flat epsilon grants wildly non-uniform MSE headroom inside ONE
  iteration.** `clip_eps_low` is in log-ratio (nat) units while the quantity
  that diverges is `MSE_θ`. At `clip_eps_low = 0.08` the allowed MSE inflation
  spanned **261× at `ref_mse/p10` down to 2.1× at `ref_mse/max`** in a single
  iteration.

**The rule.** When `clip_low_mse_coef > 0`, each row gets

```
MSE_ref_i   = max(-ref_log_prob_i, 0)                        # clamped, see below
budget_i    = min(clip_low_mse_coef * MSE_ref_i,
                  |ln(1 - clip_eps_low)|)                    # nats
rho_floor_i = exp(-budget_i)
```

so every row is allowed the same **relative** inflation, `1 + coef`, instead of
the same absolute nat count.

Three properties, all load-bearing:

1. **`clip_eps_low` stays an absolute CEILING on the budget** (the `min`). The
   mechanism can therefore only ever be *tighter* than today, never looser. That
   is enforced, not merely implied: `exp(-(-log(1-eps)))` does **not** round-trip
   to `1-eps` in fp32 (123 of 999 `eps` values land one ULP *below* it), so
   `rho_floor` is `maximum(exp(-budget), 1-clip_eps_low)`, which also makes a
   ceiling-pinned row's floor bitwise equal to the flags-off path's.
   That is deliberate: `MSE_ref` **grows** as the field degrades (measured
   0.0023 → 0.0297 over one run), so an uncapped `c · MSE_ref` budget would
   *widen* the clip exactly when it needs to tighten.
2. `MSE_ref` is clamped at `≥ 0` before use. It is `−MSE` upstream so a
   positive `ref_log_prob` should be impossible; the clamp exists so an fp edge
   case cannot put the floor **above** 1.0, which would clip every row that
   failed to move.
3. `coef == 0.0` is numerically identical to today, RNG stream and every
   pre-existing TB scalar included.

**One tensor, six consumers.** `1 − clip_eps_low` was read at six places, and
if any of them disagreed with the loss then PAWS's alive-erosion mass `N` (hence
`k`) and every clip metric would describe a different clip than the optimizer
applied. So `_grpo_update_inner` materialises **exactly one** `[B]` tensor
`rho_floor` immediately before the surrogate and hands that same object to all
five; none of them consults `config.clip_eps_low` for its bound:

| site | what it feeds | what breaks if it desynchronises |
|---|---|---|
| `surr2 = A · clamp(ρ, min=rho_floor, max=1+hi)` | the loss itself | — |
| `alive_neg_mask = … & (ρ_det >= rho_floor)` | PAWS `N` → `k` | `N` counts rows the loss already clip-killed, inflating `k` |
| `clipfrac = (ρ < rho_floor) \| (ρ > 1+hi)` | `train/clipfrac` | — |
| `clip_killed_gradient(ρ, surr1, surr2, rho_floor, hi)` | `train/clipfrac_effective_{pos,neg}` | — |
| `over_clip = (ρ < rho_floor) \| (ρ > 1+hi)` | `train/clipfrac_{fixed,jitter}_{pos,neg}` | — |

`clip_killed_gradient` accepts `float | Tensor` for its low argument: a float is
an *epsilon* (bound `1 − eps`, the legacy form every existing call site uses), a
Tensor is the **bound itself**. Its four-case table is per row and holds for
either form.

`torch.clamp` with **both** bounds as tensors. Value-identical to the
scalar-bound form (bitwise over 2e5 random rows × fp32/bf16/fp64, exact boundary
values included) **and** gradient-identical to it, including at an exact tie.

An earlier revision used `torch.maximum(torch.minimum(…))`, which is
value-identical but **not** gradient-identical: at `ρ == bound`, `clamp` routes the
full gradient to the selected branch while `maximum`/`minimum` split it 0.5/0.5,
measuring `d(loss)/d(ρ) = −0.75` against clamp's `−1.00` for either advantage
sign. The enclosing `torch.min` does *not* compensate — a differential test
against the pre-change tree diverged in `grad_norm_max` and in the final weights
on a batch containing boundary ratios, i.e. it broke the flags-off invariant.
Ties are rare but reachable: `ρ` is fp32 (`fm_log_prob.py` accumulates in fp32),
and a 4–8 fp32-ULP window of `log_ratio` maps exactly onto each bound (~6e-7 per
row). Clamp also removes an order dependence — `max(min(x,hi),lo)` disagrees with
clamp's `min(max(x,lo),hi)` whenever `lo > hi`, unreachable today given the `eps`
validation but latent. Both bounds must be tensors: `torch.clamp` rejects a mixed
`(Tensor min, Number max)` call.

**Anchor rows** get the same formula — their `ref_log_prob` is a real
measurement and they do pass through the surrogate, so they need an entry. It is
inert for them: an anchor carries a constant *positive* advantage, and by the
four-case table a positive row's `min()` always picks the unclamped branch below
the lower bound. A tighter floor can move only the sign-agnostic
`train/clipfrac`, which is a metric. Anchors are already excluded from the PAWS
masses, so it cannot move `k` through them either.

**`jitter/neg_clip_budget_used` keeps its flat `|ln(1 − clip_eps_low)|`
denominator** so the curve stays comparable across runs; the coefficient does
not change its meaning. Same for the new `pos_clip_budget_used`.

**Pair it with `paws_k_floor_at_target`.** Killing more negative rows shrinks
`N`, which under the current controller *lowers* `k` — so (A) alone tightens the
erosion brake and weakens reinforcement at the same time. See "PAWS → Flooring
`k` at `target_ratio`".

```bash
uv run python scripts/grpo/train_grpo.py \
  --clip-low-mse-coef 8.0 --clip-eps-low 0.2 \
  --positive-advantage-weight-scaling --paws-k-floor-at-target \
  --positive-advantage-weight-target-ratio 1.75
```

The startup banner prints the resolved formula, the uniform inflation factor,
the `MSE_ref` at which the ceiling starts binding, and the budget at two
representative `MSE_ref` values (`train_grpo.MSE_REF_BANNER_PROBES`, which
bracket the measured range) — so a coefficient large enough to pin every row to
the ceiling, i.e. to silently revert to the flat clip, is visible in the first
screen of output.

### Per-row erosion-drift distribution (`drift/*`)

Emitted every iteration, unconditionally — no flag, no cost worth measuring.
**Pooled over every trained micro-batch** of the iteration, over
**pre-renorm-negative advantage, non-anchor** rows.

**Why.** Every `train/*` number is a mean over 266–336 micro-batches, so the
per-**row** spread of erosion drift was completely unmeasured — and a lower-clip
threshold calibrated from iteration means will clip far more rows than intended
the moment `p90` is several times `p50`. These are the numbers
`clip_low_mse_coef` should be picked from.

**Why pooled, and not the first micro-batch** (which an earlier revision used):
`_compute_ref_log_probs` runs *before* `_grpo_update`, so on the first trained
micro-batch `n_updates == 0` and the weights **are** the reference weights —
`log_ratio` is identically 0 for a fixed row and exactly `−gap` for a jittered
one. The percentiles then read ~0 no matter how far the policy drifted, which is
the opposite of what they exist to measure. (The sibling `jitter/*` block gates on
`n_updates == 0` *deliberately*, for the opposite reason: it wants a drift-free
measurement.) Pooling also means a first micro-batch holding no negative row no
longer loses the family for the whole iteration — ~17 % of iterations at an
80 %-positive mix.

**Signed and one-sided.** The clip fires on `ratio < rho_floor`, i.e.
`log_ratio < −budget`, so the quantity is `−log_ratio` (**positive = eroded
downward**) tested against `+budget`. A row that drifted *up* shows as a negative
value and is *not* counted over budget: for a negative-advantage row an upward
move hits the **upper** bound, which by `clip_killed_gradient`'s four-case table
leaves the row **alive**. An earlier revision compared `\|log_ratio\|` against the
budget and over-reported 4× on a batch containing one up-drifted row.

| Scalar | Meaning |
|---|---|
| `drift/neg_down_{p10,p50,p90,max}` | percentiles of per-row `−log_ratio` over that population, pooled across the iteration. Positive = eroded toward the floor; negative = drifted up. `torch.quantile`'s linear interpolation. |
| `drift/neg_rows` | the surviving (finite) row count behind the pooled numbers — an **iteration** total, roughly `n_micro_batches × negative rows per batch`. |
| `drift/budget_mean` | pooled mean of the per-row nat budget, so the percentiles can be read against the constraint without recomputing it. Not a threshold — the budget is per row. |
| `drift/neg_frac_over_budget` | fraction of pooled rows past **their own** budget — the flat `\|ln(1−clip_eps_low)\|` when `clip_low_mse_coef == 0`, and `min(coef·MSE_ref_i, that ceiling)` when it is on (read back off `rho_floor`, one expression for both). Strict `>`: a row exactly at its floor is not clipped. The direct "how much erosion is this clip killing" readout. |
| `drift/neg_frac_born_dead`, `drift/neg_born_rows` | the same fraction restricted to the **pre-step** micro-batches — those whose FORWARD ran before any `optimizer.step()` — captured at the forward, so it is all `gradient_accumulation_steps` of the first window (inferring it after the step drops the window-closing micro-batch, which cost 1 of every k) — where `θ == θ_ref` and "over budget" therefore means "**born** clip-dead". This is the tripwire the `clip_low_mse_coef` × `jitter_neg` config warning tells you to watch; it reads ~1.0 when the budget is below `gap_neg`. **Read the sample size with it**: the denominator is `neg_born_rows`, at most `gradient_accumulation_steps × mini_batch_size` rows (3–5 at the defaults, so the curve is quantized to `1/n`), and both keys are **absent** on any iteration whose first window held no negative signal row (~17 % at an 80 %-positive mix). |

Non-finite rows are **dropped**, not counted — same policy as the
`ratio_maxes` / `ratio_mins` accumulators, because `ratio = log_ratio.exp()` can
overflow while the clipped loss stays finite. A poisoned micro-batch therefore
shows up as a small `neg_rows` rather than a NaN curve.

Ungated on `n_updates`, like `ref_mse/*` and `jitter/*`: the numbers come off a
micro-batch that *trained*, so they survive an iteration whose gradient windows
were all dropped — and a blown-up per-row drift is a likely *cause* of landing
there. A micro-batch with no pre-renorm-negative signal rows emits **nothing**
(a curve gap, not a fake 0).

### Weight-step direction cosines (`lora/cos_step_*`)

Emitted every iteration alongside the pre-existing `lora/weight_delta_norm`.
`_compute_lora_step_cosines()` is called from the same two `_log_metrics` sites
as `_compute_lora_delta_norm()`.

| Scalar | Meaning |
|---|---|
| `lora/cos_step_early` | `cos(step_now, L_early)`. **This is the one to read.** Measured across 6 runs its minimum is −0.058 over 41 updates on the runs that stayed healthy, and it reaches **−0.49** and **−0.62** on the two that collapsed directionally — i.e. it turns negative before the success curve does. Emitted only when a reference is available. |
| `lora/cos_step_prev` | `cos(step_now, step_prev)`. Step-to-step consistency; useful for telling a genuine direction change from per-iteration sampling noise. |
| `lora/cos_step_cumulative` | `cos(step_now, W_prev − W_init)`. Emitted **because it is free, not because it is informative**: it is self-referential — once a run turns, `W_prev − W_init` turns with it — and measured POORLY, holding +0.37…+0.53 straight through a collapse. |
| `lora/step_norm` | `‖step_now‖`, so a cosine can be read against the size of the move that produced it. |
| `lora/cos_ref_source` | TB **text** summary (`global_step 0`), written once per distinct source: `"paths"`, `"frozen_after_N_logged_iters_of_run"`, or `"none"`. `M = N + 1`, because the first logged iteration only seeds the history and returns before the counter advances. `_of_run` is load-bearing: `setup()` resets the history, so on a **resumed** run the reference is `W_(resume+M) − W_resume` — the run is scored against its own *post-resume* direction. Pass `--cos-ref-lora-paths` to pin a reference from the pre-resume lineage. |

**The reference `L_early`**, two forms:

```
cos_ref_lora_paths: tuple[str, str] | None = None   # (path_a, path_b); L_early = W(b) - W(a)
cos_ref_iterations: int = 2                          # else freeze L_early = W_now - W_init
                                                     # after N logged iterations
```

`cos_ref_lora_paths` points at two existing `iter_NNNN/` LoRA checkpoint dirs (or
`lora_weights.pt` files) and is loaded ONCE at setup by a plain `torch.load` of
the filtered state dict — deliberately *not* `load_lora_checkpoint`, which loads
*into* the live model. Checkpoint keys are DiT-relative while
`named_parameters()` is model-relative, so the prefix is derived by **module
identity** (walking `named_modules()` until the object *is*
`model.action_head.model`) rather than hardcoded. A key-set or per-key shape
mismatch against the live trainable set, or a degenerate `W(b) == W(a)`, is a
hard failure with an explanatory message: a silently-partial reference would make
`cos_step_early` a cosine against a different subspace than the step lives in,
which reads as a plausible number and is meaningless.

**Why the frozen reference matters.** The in-run freeze scores a run against its
*own* early direction, so it cannot detect a run that had already turned by
iteration N. Pass `--cos-ref-lora-paths` from a known-good run's checkpoints for
an external reference. `--cos-ref-iterations` must be `≥ 1`; a zero reference
vector has no defined cosine.

**Memory and syncs.** Two device-resident snapshots of `_lora_init_params`' size
(~80 MB each at rank 16): `_lora_prev_params` (W at the previous logged
iteration) and `_lora_prev_step`. Both are `copy_`-ed **in place** rather than
rebound, so no history accumulates and no rebuild ever holds an old and a new
copy at once. `L_early` lives on **CPU** — it is a frozen constant read once per
~13 minutes of wall clock, so a per-tensor H2D copy costs nothing and keeps
device-resident extra memory at exactly the budgeted two snapshots. Every dot
product and squared norm accumulates into ONE device vector with a single sync at
the end, matching `_compute_lora_delta_norm`'s pattern — no `.item()` per tensor.

**Edges.** The first logged iteration and any zero-step iteration (the early-skip
log path, or an iteration whose every gradient window was dropped) emit nothing
AND leave the history untouched, so the next real step is still compared against
the last real *step* rather than against zero. Nothing is lost at iteration 1:
the only quantity that would be defined there is `step_norm`, and
`lora/weight_delta_norm` already *is* `‖W_1 − W_init‖`.

### Knobs and files: per-row clip floor, `k` floor, new diagnostics

| Knob | Default | Meaning |
|---|---|---|
| `clip_low_mse_coef` (`--clip-low-mse-coef`) | `0.0` = OFF | Per-row lower clip: `budget_i = min(coef·MSE_ref_i, \|ln(1−clip_eps_low)\|)`, `rho_floor_i = exp(−budget_i)`. Uniform *relative* MSE budget `1 + coef`. `clip_eps_low` remains an absolute ceiling, so it can only ever be tighter than the flat clip. |
| `paws_k_floor_at_target` (`--paws-k-floor-at-target`) | `False` | Floor the MEASURED PAWS `k` at `positive_advantage_weight_target_ratio` instead of `1.0`. Requires `1.0 ≤ target_ratio < positive_advantage_weight_max`. Only consulted with `positive_advantage_weight_scaling=True`. |
| `cos_ref_lora_paths` (`--cos-ref-lora-paths`) | `None` | `(path_a, path_b)` — two existing LoRA checkpoint dirs (or `lora_weights.pt` files). `L_early = W(b) − W(a)`, loaded once at setup. |
| `cos_ref_iterations` (`--cos-ref-iterations`) | `2` | When `cos_ref_lora_paths` is unset, freeze `L_early = W_now − W_init` after this many logged iterations. Must be `≥ 1`. |

New TB families: `drift/*` (9 scalars), `lora/cos_step_{prev,cumulative,early}`,
`lora/step_norm`, `lora/cos_ref_source` (text), `jitter/pos_clip_budget_used`.
All are pure additions and emit unconditionally where their inputs exist.

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `clip_low_mse_coef`, `paws_k_floor_at_target`, `cos_ref_lora_paths`, `cos_ref_iterations` + four `__post_init__` validations (`coef ≥ 0` and finite; the flag/`target_ratio ≥ 1.0` interaction; `cos_ref_iterations ≥ 1`; `cos_ref_lora_paths` a 2-tuple of existing paths, normalised to `tuple[str, str]`). |
| `train_grpo.py` | `clip_killed_gradient` takes `float \| Tensor` for its low bound. `_grpo_update_inner` builds the single `rho_floor` tensor and threads it to all six consumers, applies the `k` floor, and pools per-row erosion travel over every trained micro-batch plus a pre-step-scoped born-dead fraction (returned as `_drift_diag`, carried onto the `n_updates == 0` path too). `_jitter_gap_diagnostics` hoists `lo_budget` and adds `pos_clip_budget_used`. New `_dit_param_prefix`, `_load_lora_state`, `_load_cos_ref_direction`, `_compute_lora_step_cosines`; `setup()` resolves `L_early`; `_log_metrics` takes `lora_cosines=` and emits `drift/*`, `lora/cos_*` and the provenance text (wandb mirrored). `train()` gains the two banner blocks; `MSE_REF_BANNER_PROBES` is the banner's `MSE_ref` probe pair. Class-level OFF defaults for the six `_lora_cos_*` attributes so `__new__`-built test trainers degrade instead of raising. |
| `toy_train_grpo.py` | `_log_metrics` override forwards `lora_cosines`; banner prints `Clip low MSE c`. |
| `test_clip_floor.py` | New CPU suite (see the Contents table). |
| `test_jitter_metrics.py` | The `clip_killed_gradient` call-site spy now asserts the low argument is a per-ROW `rho_floor` tensor equal to `1 − clip_eps_low` (was: the float epsilon), with `clip_eps_high` still a scalar. |

### Gradient accumulation

`gradient_accumulation_steps = k` (default 1) accumulates the gradients of `k`
consecutive mini-batches into a single optimizer step:

```
per micro-batch that survives the non-finite guard:
    if the window is empty:  optimizer.zero_grad()
    (loss / k).backward()                  # 1/k → the buffer holds the MEAN
    if the window now holds k:             # close the window
        clip_grad_norm_(...); optimizer.step()
at the end of EVERY epoch:
    if the window is non-empty: clip_grad_norm_(...); optimizer.step()  # flush
in either case: if the accumulated gradient is non-finite, drop the step,
    zero the buffer, and count it (n_nonfinite_grad_steps) instead
```

**Why.** `mini_batch_size` cannot be raised: peak VRAM at `mini_batch_size=8`
is ~21.5 GB of ~25.3 GB on an A10G (~1.48 GB per row → ~8-9 rows is the
ceiling). The per-row cost is dominated by the K-loop in
`compute_fm_log_prob`, which accumulates the log-prob across all
`len(tau_centers)` DiT forward passes and calls `backward()` once — so autograd
retains the activations of all K passes simultaneously. Accumulation is
therefore the only route to a larger effective batch. Peak VRAM is unchanged
(each micro-batch's graph is still freed by its own backward; the retained fp32
grad buffers are ~80 MB at rank 16) and total forward/backward work is
unchanged. What changes: the update direction averages `k` micro-batch
gradients, and the optimizer-step count drops by ~`k`. **LR is per-iteration**
(the monotone anneal ramp in `train()` — there is no warmup; iteration 1 already
runs at the full configured LR), not per-step, so `k` does not rescale the step
size — hold LR fixed.

**Deliberately NOT one wide batch.** The advantage z-score still runs
independently on each micro-batch of `mini_batch_size` rows
(`per_iteration_advantage_norm` stays `False`), so what gets averaged is `k`
independently normalized gradients. That is the intent, not an approximation:
the group-relative binary-reward advantage is strongly asymmetric (at 12.5%
success, +2.475 for a success vs −0.354 for a failure, ~7:1), and
per-minibatch z-scoring restores symmetry. Switching to per-iteration norm to
make accumulation "exact" passes that asymmetry straight through, strips the
failure-avoidance signal, and silently pins `pos_adv_weight_k` to its 1.0 floor
(disabling PAWS) — it measured much worse on matched iterations
(success 0.125 / 0.25 / 0.083 / 0.29 vs 0.125 / 0.625 / 0.625 / 0.54).

**Edges.**
- A partial window at an epoch boundary is **flushed, never discarded**. The
  scale is a uniform `1/k`, so a flushed window of `m < k` micro-batches steps
  with `(m/k)×` the average gradient — at most one such step per epoch
  (`train/n_partial_windows`). The flush is required, not cosmetic:
  `_iter_balanced_minibatches` anchors epoch length to `ceil(n / mb_size)` but
  returns early when the majority pool drains, so the micro-batch count per
  epoch is not a multiple of `k`.
- A minibatch dropped by the non-finite guard contributes nothing AND does not
  advance the window, so every full window carries exactly `k` **trained**
  micro-batches. PAWS mass (`N_iter` / `D_iter`) likewise commits per trained
  micro-batch, so the "pooled mass == trained rows" invariant is unchanged —
  except on the rare dropped-window path below, where up to `k` micro-batches'
  mass is pooled without a weight update (accepted; `k` is a ratio, so the
  effect is second-order).
- If the accumulated gradient is non-finite the step is dropped and the window
  discarded (`n_nonfinite_grad_steps`) — see the gradient-side guard under
  "Clipped surrogate + KL". This protects a `k=1` run identically; the only
  k-specific note is that a dropped `k > 1` window forfeits up to `k`
  micro-batches of work instead of one.
- `train/n_updates` counts real `optimizer.step()` calls that actually reached
  the weights (so it drops by ~`k`, excludes dropped windows, and `did_update` /
  checkpoint naming still mean "the model moved");
  `train/n_micro_batches` counts trained mini-batches (unchanged by `k`). Every
  per-minibatch mean — `loss`, `clip_loss`, `kl_loss_*`, `clipfrac`,
  `mean_ratio`, `mean_log_ratio_abs` — divides by `n_micro_batches`, so these
  curves are **not k-inflated** and stay on the same scale across `k`. They are
  NOT bit-identical across `k`: within a window all `k` micro-batches see the
  same un-stepped weights, so the log-probs (and hence loss / ratio / clipfrac)
  shift by a few percent versus a `k=1` baseline — that is expected, not a
  regression. The `_fixed` / `_jitter` branch metrics are row-weighted
  (`sum / n_rows_*`) rather than per-minibatch means, and are likewise
  unaffected by `k`. `train/grad_norm_*` is the deliberate exception: it
  measures the ACCUMULATED gradient, so expect it to read lower at `k > 1` —
  that is noise cancelling between micro-batches, not weaker signal.
- `k = 1` is bit-identical to the pre-accumulation code path, and emits no
  `grad_accum_steps` / `n_partial_windows` curves and no banner line.

`test_grad_accum.py` covers all of the above by driving the real
`_grpo_update` / `_grpo_update_inner` on CPU. It substitutes the GPU-bound and
setup-bound pieces: `_prepare_batch` (tiny CPU tensors instead of a backbone
re-encode), `compute_fm_log_prob` (a 2-parameter analytic stand-in for the
K-loop DiT forward), the model, the episode buffer, and the optimizer (plain SGD
so a reference trajectory is exactly reproducible — production uses AdamW), and
it builds the trainer via `__new__` to skip `setup()`. The accumulation window,
guards, flush, step cadence and every metric divisor are the production ones.

```bash
# k=2: ~150 optimizer steps/iter instead of ~300, same LR, same peak VRAM
uv run python scripts/grpo/train_grpo.py \
  --gradient-accumulation-steps 2 --learning-rate 1.5e-5 \
  --group-size 12 --num-groups 4 --num-iterations 40
```

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
loop. Defaults `jitter_pos = 0.0` and `jitter_neg = 0.0` are bit-identical to
vanilla GRPO; setting either (e.g. `--jitter-pos 0.05 --jitter-neg 0.05`)
activates the full mechanism.

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

### Knob: `GRPOConfig.jitter_pos` / `GRPOConfig.jitter_neg`

The Jacobian-penalty strength is split by advantage sign: `jitter_pos` applies
to positive-advantage chunks (the "good" chunks GRPO reinforces — basin
sharpening), `jitter_neg` to negative-advantage chunks (the "bad" chunks GRPO
suppresses — neighborhood carve). The sign is the chunk's PRE-renormalization
group-relative advantage, matching the `*_pos` / `*_neg` metric split.

- Both default `0.0` → bit-identical to vanilla GRPO (no jittered passes, no
  per-branch metrics, no extra CUDA syncs). Jitter is "active" when EITHER
  side is `> 0`.
- Suggested value `0.05` for each (variance-preservation multiplier
  `(1−t)²·λ²` ≤ 2.5e-3, comfortably below the bf16 mantissa noise floor).
- Each side is range-checked in `GRPOConfig.__post_init__`: must satisfy
  `0.0 ≤ λ < 1.0` (variance preservation requires `λ < 1`).
- Setting only ONE side to `0.0` still emits that sign's jitter copy, but with
  λ=0 the copy is identical to its fixed row (a redundant forward pass, no
  Jacobian penalty on that sign). Set BOTH to `0.0` to fully disable.
- Scheduling is controlled by `jitter_paired` (see next subsection): paired
  (default, 2× steps) vs jitter-only (1× steps, directly comparable to
  vanilla).
- The trainer prints a one-line `Jitter-GRPO: pos=… neg=… paired=…` banner at
  startup when active, including the step-budget note for the chosen mode.

### Scheduling: paired vs jitter-only (`jitter_paired`)

`jitter_paired` (default `True`, `--no-jitter-paired` to disable) decides how
many entries each chunk contributes per epoch when jitter is active. It is
N/A when jitter is off.

**Paired (`jitter_paired=True`, default).** Each live chunk produces TWO
entries per epoch:

```python
entries = (
    [(c, "fixed") for c in live_chunks]      # DiT input = original ε
    + [(c, "jitter") for c in live_chunks]   # DiT input = ε'
)
```

Both entries reference the **same** `ActionChunk` object (so they share
`tau_samples`, `ref_log_prob`, `initial_noise`, and the cached backbone
features). The only difference is the DiT input noise during the forward
pass: "fixed" rows use the original `ε`, "jitter" rows use
`ε' = √(1−λ²)·ε + λ·ξ`, where `λ` is `jitter_pos` for positive-advantage
chunks and `jitter_neg` for negative.

Doubling the entries list doubles the number of optimizer steps per epoch.
**Halve `update_epochs` MANUALLY** when running paired jitter (e.g., 4 → 2)
to match the per-iter optimizer-step budget of vanilla GRPO. The trainer
does not auto-halve — the relationship is left explicit so the user can
audit it from the CLI. This mode keeps the fixed-vs-jitter per-branch
diagnostic (the `mean_log_ratio_abs` gap that estimates the Jacobian norm).

**Jitter-only (`jitter_paired=False`).** Each live chunk produces ONLY its
jitter entry:

```python
entries = [(c, "jitter") for c in live_chunks]   # DiT input = ε'
```

The per-iter optimizer-step count then matches a vanilla GRPO run at the
**same** `update_epochs` — no manual halving, directly comparable curves.
The trade-off: with no "fixed" rows, the `_fixed` per-branch metrics and the
fixed-vs-jitter gap diagnostic are unavailable (only `_jitter` metrics are
emitted), and the loss trains purely on the jittered input noise. Use this
when you want an apples-to-apples step-budget comparison against a no-jitter
baseline rather than the paired diagnostic.

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

Same deterministic shuffle behavior — with jitter off (both sides 0),
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
regardless of the jitter settings.

### ξ sampling and `noise_for_input` construction

Inside `_grpo_update_inner`, after `_prepare_batch` returns and the
`ready_*` slicing is done:

```python
ready_modes = [batch_data["modes"][i] for i in ready_indices]
lam_pos = self.config.jitter_pos
lam_neg = self.config.jitter_neg

if (lam_pos > 0.0 or lam_neg > 0.0) and any(m == "jitter" for m in ready_modes):
    K = len(self.config.tau_centers)
    B_r, H, D = ready_noise.shape

    # Unseeded; uses global torch RNG, matching _sample_jittered_timesteps.
    xi = torch.randn(K, B_r, H, D,
                     device=self.device, dtype=ready_noise.dtype)

    jitter_mask = torch.tensor(
        [m == "jitter" for m in ready_modes],
        device=self.device, dtype=torch.bool,
    )

    # Per-row λ by PRE-renorm advantage sign: jitter_pos for adv > 0,
    # jitter_neg otherwise. float32 keeps the scalar full-precision through
    # the sqrt/multiply (a 0.0 side collapses that row to ε).
    lam_row = torch.where(
        ready_advantages > 0,
        ready_advantages.new_full((B_r,), lam_pos, dtype=torch.float32),
        ready_advantages.new_full((B_r,), lam_neg, dtype=torch.float32),
    )
    lam_j = lam_row[jitter_mask]
    sqrt_one_minus_j = (1.0 - lam_j * lam_j).sqrt()

    # expand returns a stride-0 view; clone() materializes a writable
    # [K, B_r, H, D] tensor so __setitem__ writes per-K rows independently.
    noise_for_input = (
        ready_noise.unsqueeze(0).expand(K, -1, -1, -1).clone()
    )
    # Explicit .to(dtype): masked index-put will NOT auto-cast f32 -> bf16.
    noise_for_input[:, jitter_mask] = (
        sqrt_one_minus_j[None, :, None, None] * ready_noise[jitter_mask].unsqueeze(0)
        + lam_j[None, :, None, None] * xi[:, jitter_mask]
    ).to(ready_noise.dtype)
else:
    noise_for_input = None
```

Three notable details:

- **ξ is unseeded.** Uses the global torch RNG, matching how
  `_sample_jittered_timesteps` jitters the τ centers. On-policy collection
  noise also isn't seeded per-call, so making ξ a special case would be
  inconsistent with the rest of the training-time stochasticity. Resume
  across iters proceeds without errors but ξ values are not bit-reproducible
  across the resume boundary when jitter is active.
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

When the gate is False (both λ=0, or no jitter rows in this mb),
`noise_for_input=None` and the K-loop takes the original-ε path.

VRAM cost: `xi + noise_for_input ≈ 2 × 614 KB` per minibatch at
`K=6, B=8, H=50, D=128` in bf16. Negligible vs the DiT activations.

### Per-branch metrics (`*_fixed` / `*_jitter` TB scalars)

The KL is refactored to expose `kl_per_row_last_iter` (and optionally
`kl_per_row_base_model`) as named intermediates so they can be indexed by
branch. The final `kl_loss_last_iter = kl_coef_last_iter *
kl_per_row_last_iter.mean()` is numerically identical to the previous
inlined form.

Inside the no-grad accumulator block, **gated on `lam_pos > 0.0 or lam_neg >
0.0`**, we split the per-row tensors by mode and accumulate row-level sums:

```python
if lam_pos > 0.0 or lam_neg > 0.0:
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

The gating on `lam_pos > 0.0 or lam_neg > 0.0` matters: with jitter off (both
sides 0), the per-mb accumulator block is skipped entirely, the per-branch
counters stay at their zero defaults, the result-dict gating
`if n_rows_fixed > 0:` is False, and no `_fixed`/`_jitter` keys are emitted.
Vanilla GRPO runs see exactly the same TB curves they always did.

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

### Measuring the gap directly (`jitter/*`, `ref_mse/*`)

Differencing two TB curves only works with `jitter_paired=True`, so the gap
was unobservable in `jitter_paired=False` runs — which is most of them. It also
inherits the ~5e-4 bf16/batching noise floor between the ref pass (batch
`2·mini_batch_size`, fresh `_prepare_batch`) and the update pass (batch
`mini_batch_size`, rebuilt from cache).

`GRPOTrainer._jitter_gap_diagnostics` measures it directly instead: **two
matched `no_grad` forwards** on the first minibatch of the iteration — clean ε
and jittered ε′, same cached features, same τ, differing only in
`noise_for_input`. Gated on `n_updates == 0` so it is always taken at
θ ≡ θ_ref (no optimizer step has fired, so the gap carries zero policy-drift
contamination); if the first minibatch happens to hold no jitter rows the
measurement is **skipped** rather than deferred to a post-step one. Wrapped in
`try/except` — a diagnostic failure costs the metric, not the iteration.

Cost: ~12 extra DiT passes against the ~780+ a normal iteration runs (~1.5%),
no activations retained, and **no RNG consumed** (with `timesteps` and `noise`
both supplied, `compute_fm_log_prob` takes no sampling branch), so the global
torch RNG stream is unchanged versus runs recorded before this existed.

| Scalar | Meaning |
|---|---|
| `jitter/gap_pos`, `gap_neg` | `E[MSE_θ(ε′) − MSE_θ(ε)]` on jitter rows, split by **pre**-renorm advantage sign (matching how `lam_row` picks `jitter_pos` vs `jitter_neg`) |
| `jitter/jacobian_fro_sq` | the gap with the `E_k[(1−τ)²]·λ²` prefactor divided out, so it is comparable **across** `jitter_pos` settings. Two documented biases: it is `‖J‖²_F / D_valid` (the MSE is a per-valid-dim mean), and it reads high by `1 + ((√(1−λ²)−1)/λ)²` (+1.6% at λ=0.25, +7% at λ=0.50) because the true per-element perturbation variance is `(√(1−λ²)−1)² + λ²`. A comparable proxy, not an exact invariant. |
| `jitter/gap_at_tau{k}`, `tau{k}_value` | per-τ profile over positive rows, `k` indexing `tau_centers`. Shows **where** along the denoising path the field is noise-sensitive; should fall as τ rises. `tau{k}_value` is the mean of the actual jittered τ, read back from the bf16 `timesteps` tensor, so it carries ~0.4% relative quantization versus the fp32 `tau_samples` on the chunks. |
| `jitter/headroom_multiplier` | `(ref_mse/pos_mean + gap_pos) / ref_mse/pos_mean`. The second reading of the gap: without jitter `log ρ ≤ MSE_ref` caps reinforcement at ~0.01–0.05; with jitter the row starts at `ρ = e^-gap` with `MSE_ref + gap` of usable room, all of it usable because `clip_eps_low` cannot clip a positive row. ~1.0 means `jitter_pos` is doing nothing. Note the two terms differ in scope — `pos_mean` is an iteration-wide chunk mean, `gap_pos` a one-minibatch row mean. |
| `jitter/neg_clip_budget_used` | `gap_neg / |log(1−clip_eps_low)|`. The hard ceiling on `jitter_neg`: → 1.0 means every negative row is born outside the clip and contributes no gradient. ~0.028 at `jitter_neg=0.05`; the wall is ≈0.30. This is the only place `clip_eps_low` interacts with jitter in a way that can KILL a row — it cannot clip a positive one. Denominator is the FLAT `clip_eps_low` budget even when `clip_low_mse_coef > 0`, so the curve stays comparable across runs. |
| `jitter/pos_clip_budget_used` | `gap_pos / |log(1−clip_eps_low)|` — the same share on the positive side, on the **same flat denominator**. NOT a ceiling on `jitter_pos`: a positive row cannot die on the lower bound. It exists because only the harmless negative side used to be reported (1.9–3.1 %) while `gap_pos` silently ate **53–76 %** of the same budget, which is what makes the sign-agnostic `train/clipfrac` read near 1.0 while killing nothing. Read it as the explanation for an alarming-looking `clipfrac`; the real positive-side ceiling is `headroom_multiplier`. |
| `jitter/gap_fixed_rows_selfcheck` | gap on paired-mode "fixed" rows. **Must read ~0** (≤1e-4); anything else means the fixed rows are not being fed the original ε. Absent in `jitter_paired=False`. |
| `jitter/n_rows_pos`, `n_rows_neg` | row counts backing the two gaps — a **one-minibatch** count, not an iteration total. |
| `ref_mse/{mean,p10,p50,p90,max}` | distribution of `MSE_ref = −ref_log_prob` over live chunks. Free (the values are already on the chunks). |
| `ref_mse/pos_mean`, `neg_mean` | split by advantage sign. `pos_mean` is the reinforcement headroom on successful chunks; it decaying toward 0 while success rate plateaus **is** positive-branch saturation, since the FM loss is least-squares so the gradient is `∝ residual`. |
| `ref_mse/ratio_ceiling_{mean,max}` | `exp(MSE_ref)` — the analytic ceiling on the importance ratio, since `log ρ = MSE_ref − MSE_θ` and `MSE_θ ≥ 0`. Compare against `1 + clip_eps_high`: the per-iteration console line prints REACHABLE / UNREACHABLE. |
| `ref_mse/log_base_ratio_{mean,p10,min}` | `ref_log_prob − base_log_prob` = cumulative drift of the adapted field from the pretrained one, in MSE units and unscaled by any coefficient (unlike `kl_loss_base_model`). Positive = fits the sampled action better than base. Emitted only when `kl_coef_base_model > 0`. |

`ref_mse/*` and `jitter/*` are emitted **outside** the `n_updates > 0` gate:
both are measured before any optimizer step, so they stay valid on an iteration
whose update was discarded — and a blown-up gap is a likely *cause* of landing
there (large `|log_ratio|` → bf16 `exp` overflow → non-finite loss).

### Advantage-sign-split ratio metrics and the *effective* clipfrac

`mean_ratio_{fixed,jitter}` pool both advantage signs, and because the gap
scales as `λ²` the two signs sit at very different ratios (at
`jitter_pos=0.25` / `jitter_neg=0.05` the biases are ≈−0.058 vs ≈−0.002), so
the pooled curve is dominated by the positive rows and neither branch is
legible. `train/{mean_ratio,mean_log_ratio_abs}_{fixed,jitter}_{pos,neg}` split
them:

- `mean_ratio_jitter_pos` starts each iteration at `e^-gap_pos` and its
  movement **up** within the iteration is headroom being consumed — the direct
  "is the positive branch learning?" readout.
- `mean_ratio_jitter_neg` starts at ≈1.0 and moves down; that is erosion.

`train/clipfrac_effective_{pos,neg}` counts rows whose **clip-term** gradient
the clamp actually zeroed, which is *not* what `clipfrac` measures. `clipfrac`
is the sign-agnostic test `(ratio < 1−lo) | (ratio > 1+hi)`, and for a
positive-advantage row that is a false positive: with `A>0` and `ρ < 1−lo`,
`min(A·ρ, A·(1−lo)) = A·ρ` — the unclamped branch wins and the gradient is
fully alive. Predicate: `clip_killed_gradient()` (module-level in
`train_grpo.py`, so tests exercise the real expression), which is
`clamp_moved & (surr2 <= surr1)`; positives can only die on the **upper** bound
and negatives only on the **lower** one.

That distinction is cosmetic at today's `jitter_pos` (observed `ratio_max` ≈
1.05) but becomes load-bearing above `jitter_pos ≈ 0.30`, where `gap_pos`
exceeds `|log(1−clip_eps_low)|` and **every** positive row reports as
"clipped" while training normally. Two caveats:

- Buckets by the **post**-renorm advantage sign (unlike the sibling
  `clipfrac_{branch}_{sign}` metrics, which use pre-renorm), because which
  bound a row can die on is decided by the sign the loss saw. Expect
  `_pos` ≡ 0 at any sane `jitter_pos`.
- Do **not** read `_neg` as a drop-in for `clipfrac_*_neg`: under
  per-minibatch renorm the two have different denominators, and a group-good
  row carrying `λ = jitter_pos` that renorm flipped negative is a genuine
  lower-bound death booked here — so a large `jitter_pos` inflates `_neg`.
  Cross-reference `n_pos_flipped_by_renorm`.

Values routed through `_log_metrics._emit` are filtered for non-finite (and
non-numeric) entries before reaching TB/wandb, because a bf16
`ratio = log_ratio.exp()` overflow reaches `ratio_sum_*` while the clipped loss
stays finite, and one `nan`/`inf` poisons wandb's chart autoscale for the rest
of the run. Note the **pre-existing** `train/loss`, `train/clipfrac`,
`train/mean_ratio` and `train/mean_log_ratio_abs` are deliberately left
unfiltered.

`test_jitter_metrics.py` covers all of the above on CPU: the gap / Jacobian /
headroom arithmetic against a closed-form stand-in whose FM residual vanishes
at `ε_in = ε` (so the Taylor expansion the estimator inverts applies), the
`clip_killed_gradient` truth table cross-checked against autograd on the real
loss, the `clipfrac_effective_*` aggregation **values** (forced dead-patterns
pin each bucket's denominator), and the θ ≡ θ_ref property functionally via
AdamW's lazily-populated `optimizer.state`.

### Bit-identical guarantee with jitter off (both sides `0`)

| Path | Behavior when jitter off |
|------|-----------------|
| `entries` construction | `[(c, "fixed") for c in live_chunks]` — same length and order as old `live_chunks`. |
| `_iter_stratified_minibatches` | Same RNG seed, same shuffle, same minibatch composition; yields the same chunks just wrapped in 1-tuples of `(c, "fixed")`. |
| `_prepare_batch` | Same `valid_batch` ordering; new `modes` list emitted but unused downstream. |
| `compute_fm_log_prob` | `noise_for_input=None` → `eps_input = eps` → K-loop math unchanged. |
| ξ-sampling block | Gated on `lam_pos > 0.0 or lam_neg > 0.0`; not entered. |
| Per-branch metric block | Gated on `lam_pos > 0.0 or lam_neg > 0.0`; not entered. No extra CUDA syncs from `.item()`. |
| Legacy aggregated metrics | Identical formulation; per-mb-mean accumulators preserved. |
| TB scalars | No `_fixed`/`_jitter` keys emitted; legacy TB curves byte-identical. |

Resume across iters with jitter off (both sides 0) is bit-reproducible
end-to-end. With jitter active, ξ samples are not bit-reproducible across the
resume boundary (intentional — ξ uses global torch RNG, matching τ-jitter
and on-policy collection noise).

### Toy / production CLI

Toy-mode (fixed-seed diagnostic, fast turnaround):

```bash
uv run python scripts/grpo/toy_train_grpo.py \
    --jitter-pos 0.05 --jitter-neg 0.05 --update-epochs 2
```

Production, **paired** (2× steps — keeps the fixed-vs-jitter diagnostic;
halve `update_epochs` to match vanilla's per-iter step budget):

```bash
uv run python scripts/grpo/train_grpo.py \
    --jitter-pos 0.05 --jitter-neg 0.05 --update-epochs 2 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200
```

Production, **jitter-only** (`--no-jitter-paired`; 1× steps — directly
comparable to a vanilla run at the same `update_epochs`, no manual halving):

```bash
uv run python scripts/grpo/train_grpo.py \
    --jitter-pos 0.05 --jitter-neg 0.05 --no-jitter-paired --update-epochs 4 \
    --env-names robocasa_panda_omron/CoffeeServeMug_PandaOmron_Env \
    --num-iterations 200
```

To compare against vanilla GRPO at the **same per-iter step budget**, either
run paired jitter at half the epochs, or run jitter-only at the same epochs;
the vanilla baseline is `--jitter-pos 0 --jitter-neg 0 --update-epochs 4`. The
difference is solely the Jacobian regularizer pressure on positive-advantage
chunks (basin sharpening, strength `jitter_pos`) and negative-advantage chunks
(neighborhood carve, strength `jitter_neg`). The strength knobs tune
independently — e.g. jitter only positives with `--jitter-pos 0.05
--jitter-neg 0`.

The toy script's startup banner prints `Jitter pos/neg: <pos> / <neg>` so you
can confirm the flags flowed through. `grpo_data/` collisions between jitter
and non-jitter runs at the same LR are the user's responsibility to manage
(rename or override `--checkpoint-dir` / `--episode-dir` to keep TB curves
separate).

### Files touched by this feature

| File | Change |
|------|--------|
| `grpo_config.py` | Adds `jitter_pos: float = 0.0`, `jitter_neg: float = 0.0`, and `jitter_paired: bool = True` fields + per-side range check in `__post_init__`. |
| `fm_log_prob.py` | `compute_fm_log_prob` accepts optional `noise_for_input: Tensor[K,B,H,D] \| None`. K-loop uses `eps_input_all[k]` per τ when provided; `velocity_target = actions - eps` unchanged. |
| `train_grpo.py` | `_iter_stratified_minibatches` and `_prepare_batch` operate on `(chunk, mode)` entries. `_compute_ref_log_probs` wraps as `("fixed", chunk)` tuples. `_grpo_update_inner` builds entries per `jitter_paired` (fixed+jitter when True, jitter-only when False), samples ξ via global RNG, constructs `noise_for_input` with a per-row λ (`jitter_pos`/`jitter_neg` by advantage sign) via expand+clone, threads it into `compute_fm_log_prob`, and adds gated per-branch metric accumulators. `_log_metrics` writes the per-branch TB scalars. Startup banner prints jitter pos/neg and scheduling mode when active. |
| `toy_train_grpo.py` | Prints `Jitter pos/neg` in the startup banner; overrides both strength fields (default `0.05`), inherited from `GRPOConfig` (paired by default). |

### Scope

Implemented: paired **and** jitter-only scheduling (`jitter_paired`) on top of
the existing single-chunk-per-row training loop, per-τ independent ξ_k jitter
on the DiT input (one fresh ξ per τ in `tau_centers`), cached-ref reuse for
both branches, gated per-branch TB metrics.

Not implemented: adaptive λ schedules, an offline noise-sensitivity
validation eval (the per-branch `mean_log_ratio_abs` gap is the live
signal), and the alternative formulation that recomputes the reference at
the jittered input (the cached-ref bias is `O(λ²)` and `θ`-independent —
recomputation costs an extra DiT pass per minibatch for no observable
training-direction change).

---

## Trajectory-Roughness Constraint (the "jerk constraint")

Optional, feature-flagged. `smooth_coef = 0.0` (default) is bit-identical to a run
without it: no extra tensors, no extra DiT forwards, no extra RNG consumption, no
calibration, no `smooth/*` curves, no banner line. That off-switch invariant is
asserted bit-for-bit — including RNG-stream identity — in `test_smoothness.py`.

Orthogonal to Jitter-GRPO, and both are needed. Jitter bounds the **magnitude** of
the velocity field's noise response (`E_ξ‖Jξ‖² = ‖J‖²_F`, isotropic); this bounds
its **spectrum** along the horizon axis `h`. Measured independence:
`jitter/jacobian_fro_sq` fell 32% over the same iterations in which the residual's
high-frequency fraction rose 1.7–2.9× and relative seed dispersion rose 4.5×.

Full derivation in `jerk-constraint.md`. Summary of what it constrains:

```
HF(u)    = mean((D²u)²) / (6 · mean(u²).detach())            D² along h
L_smooth = smooth_coef · relu( HF_pooled − hf_ref )
```

`u` is the trajectory named by `smooth_instrument`. The operator is the same
`(1,−2,1)` second difference either way; only *which trajectory it differences*
changes.

### Which instrument, and why the endpoint was abandoned

| `smooth_instrument` | `u` | forwards | gradient |
|---|---|---|---|
| **`"chunk"`** (default) | the `num_inference_timesteps`-step **generated chunk** — what the robot executes | N (only 1 with a graph) | last step only, biased |
| `"endpoint"` | the 1-step **implied endpoint** `â(0) = ε + v_θ(ε,0)` at τ=0 | 1 | exact |

The endpoint was the original instrument. A sweep over 16 checkpoints of a real
training run showed it **does not control the quantity the constraint exists to
bound** (physical EEF path jerk):

- Over iterations 10–16 of the unconstrained run the endpoint HF **fell 9%**
  (0.331 → 0.300) while EEF path jerk **rose 11%** (0.516 → 0.572). Spearman ρ
  between them over that window: **+0.00**.
- A run *with* the constraint at `smooth_coef=0.15` pinned endpoint HF at 3–6×
  base for six iterations, and its executed chunks degraded anyway: chunk HF
  2.2× → 8.6× base, path jerk 1.45× → 2.86× base.

The **4-step chunk's** HF correlates with path jerk at **ρ = +0.98** overall and
**+0.96** over the late iterations. So the instrument moved; the operator, the
hinge, the pooling and the calibration machinery did not.

`"endpoint"` is retained and reproduces the previous **compute path** bit-for-bit. Note it does not by itself reproduce an old run end-to-end: the same diff moved `smooth_hf_ref_scale`'s default 4.0 → 15.0, so an otherwise-identical CLI now calibrates a 3.75× looser threshold. Pass `--smooth-hf-ref-scale 4.0` alongside it (the trainer warns if you do not). So
runs calibrated against it stay reproducible. It is not recommended for new runs,
and the startup banner says so.

### Last-step-differentiable rollout

The chunk instrument rolls out the **production sampler**: step count, `dt` and
the continuous `t` values all come from `action_head.num_inference_timesteps`, read
the same way `Gr00tN1d6ActionHead.get_action_with_features` reads them
(`gr00t_n1d6.py:317-321`) — nothing is hardcoded, so a checkpoint that overrides the
config gets *its* schedule. At the shipped `num_inference_timesteps = 4` that is
`t = [0, 0.25, 0.5, 0.75]`, `dt = 0.25`. `fm_log_prob.inference_schedule` is the one
place this is derived; the banner prints the resolved values so a mismatch is
visible rather than inferred.

The rollout starts from the **clean collected ε** (never the jittered `ε'`) and
runs the first N−1 velocity evaluations under `torch.no_grad()` with an explicit
`detach()` on the carried state. Only the final `x = x + dt·v` carries a graph.

The `t` tensor is built in **float64**, not the batch's bf16. Note the reason is *not*
that bf16 cannot represent the timesteps — `0.75` is exactly representable in bf16. What
rounds is the **product**: bucketizing computes `t · 1000`, and the nearest bf16 to `750.0`
is `752.0`, so a bf16 `t` conditions the DiT on bucket **752** against production's
**750** — a different AdaLayerNorm conditioning from the sampler's, on the very step that
carries the gradient, and entirely silent. Not unique to N=4 nor always in the same
direction (N=8: `0.375 → 376`, `0.625 → 624`; N=3: `1/3 → 334` vs 333). fp32 would also
be exact; fp64 matches the width of the Python floats the schedule is built from. Pinned
per-bucket at N=4 and N=8 in `test_smoothness.py`.

- **The forward VALUE is exact** — it is the true 4-step chunk the sampler would
  produce from this ε, so calibration and measurement remain the same functional
  and a frozen threshold stays meaningful.
- **The gradient is biased.** It misses how θ shapes the earlier steps and hence
  the sampler path the last step is evaluated on — roughly a quarter of the true
  gradient's magnitude at N=4. This is an accepted, documented tradeoff:
  differentiating all four steps costs 4 graph-forwards, estimated **29.1 GB
  against 25.3 GB available** at `mini_batch_size=8`, i.e. an OOM or a halved
  batch. With one graph-forward, VRAM is unchanged and `mini_batch_size=8` is
  preserved. Compensate with a larger `smooth_coef` (suggested **0.15–0.5** for
  the chunk instrument).

Covered by `test_smoothness.py`: the rollout's value is bit-identical to a plain
no-grad rollout; against a fake head whose every step uses its own parameter, the
earlier steps' gradients are zero/None and only the last step's is non-zero, and
it equals what a fully-differentiated rollout puts on that same step.

### Why a hinge and not a penalty

Below the threshold both the value and the gradient are exactly zero, so the term
exerts no force — it neither rewards extra smoothness nor pulls toward it. A plain
`coef · HF` has a nonzero gradient everywhere and would drive toward the
conditional-mean map, which `consensus_ns4` measures at **0.365** against
`baseline_euler`'s **0.600**. Meanwhile base itself sits at `HF(a) = 0.0014` and
scores 0.600, so smoothness and competence do coexist — the hinge is what
distinguishes "don't get rougher than the pretrained field" from "be as smooth as
possible".

### Dedicated CLEAN forwards, never the K-loop's

Neither instrument reuses the K-loop's velocity. Under Jitter-GRPO the K-loop's DiT
input is `x'_τ` built from `ε' = √(1−λ²)ε + λξ`, so its velocity carries the model's
response to that perturbation, which lands in `â` as `(1−τ)²·J·(ε′−ε)` — white and
θ-independent. At the production `λ=0.25` with the measured `jacobian_fro_sq ≈ 2.4`
it dominates: HF at τ=0 goes **0.000347 → 0.790** (2275×), and since calibration
would be contaminated identically `hf_ref` freezes above HF's theoretical maximum for
H=16 (2.984), so the hinge could **never fire**.

Every smooth forward therefore starts from the original ε.
`test_smoothness.py` asserts both instruments' moments are **bit-identical** with
and without jitter.

Cost, only when the constraint is on:

| instrument | extra DiT forwards | retained activations |
|---|---|---|
| `"endpoint"` | 1 (~1/K of the K-loop, ~17% at K=6) | that 1 forward |
| `"chunk"` | `num_inference_timesteps` (4) | **1** forward |

### Pooled, not a mean of per-row ratios

`HF`'s denominator is a row's own energy, so a near-idle chunk (`M → 0`, routine
during a grasp) reports hundreds of times a moving row's value. An unweighted mean
is dominated by such rows, and `hf_ref` is frozen and persisted, so one unlucky draw
would neuter the feature for the whole run lineage.
`Σ R / (6 · Σ M)` is energy-weighted by construction: measured on 63 normal rows plus
one idle one it shifts **0.31%** where the mean of ratios shifts 29%. It is exactly
associative over batch splits, so the threshold transfers across batch sizes.

Consequence: the term is one scalar per minibatch, so its magnitude is independent of
row count and it needs **no anchor loss divisor** — unlike `clip_loss`/KL, which are
per-row means. Anchor rows still contribute their `R` and `M`, which is intended.

Below `SMOOTH_MIN_ROWS_PER_MB = 4` rows the term is **skipped**: relu convexity means
a near-single-row minibatch reinstates the idle-row blow-up pooling exists to remove
(8 singleton minibatches deliver 4.6× the penalty of one 8-row minibatch). The skips
are counted in `smooth/undersized_mbs`, and `mini_batch_size < 4` with
`smooth_coef > 0` is a hard config error rather than a silent no-op.

### `hf_ref`: frozen scalar, calibrated from the base policy

| `smooth_hf_ref` | behaviour |
|---|---|
| `float` | flat threshold, **in the units of the selected instrument**. |
| `list[float]` | first entry used, with a warning (single-scalar design). |
| `None` (default) | **auto-calibrate** at the first iteration of a fresh run, then `× smooth_hf_ref_scale`. |

Auto-calibration works because PEFT zero-inits `lora_B`, so before the first optimizer
step `θ ≡ θ_base` and the collected chunks **are** base-policy samples — confirmed by
`ref_mse/log_base_ratio_mean` reading exactly **0** at iteration 1 of a fresh run
versus 0.0572 at a resumed run's first iteration. Whole minibatches are pooled up to
`smooth_calib_min_rows`, capped at 3 iterations, and the term contributes nothing to
the loss while calibrating.

**`smooth_hf_ref_scale` default is 15.0, and it is instrument-specific.** The scale
is a multiple of the *measured base value*, and the two instruments have completely
different useful ranges even though their base values are similar (chunk 0.00141,
endpoint 0.00157). Measured chunk HF on the control run:

| iter | 1 | 2 | 3 | 4 | 6 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| chunk HF | 0.0023 | 0.0072 | 0.0152 | 0.0244 | 0.0408 | 0.0959 | 0.1131 |

i.e. it reaches ~80× base (0.00141). Corresponding EEF path jerk: base 0.0689,
iter4 0.2720, iter12 0.5358. A bound near **15–17× base (~0.024)** targets the
iteration-4 roughness level — about **2× less path jerk** than the control's
peak-success iteration — which is a real reduction that is still reachable.
**For `smooth_instrument="endpoint"`, 4.0 remains the right value** (base endpoint
HF 0.0012–0.0051 → threshold ~0.005–0.02).

Persisted to `smooth_ref.json` in every checkpoint. A resumed run with
`smooth_coef > 0` and neither an explicit `--smooth-hf-ref` nor a cached file
**hard-fails** rather than calibrating off a non-base policy.

**Guard key** (hard-fail on mismatch): `{tau_centers, jitter_std, jitter_pos,
jitter_neg, jitter_paired, dims (the constrained set C), horizon, instrument, embodiment_tag, model_path}`, plus
`{sampler_steps, sampler_dt}` **only under `"chunk"`** — the endpoint is one forward at
τ=0 and provably cannot depend on a schedule it never walks, so gating it on those would
be a false rejection. `instrument` matters more than the rest: a threshold calibrated on
the endpoint is *meaningless* for the chunk, and because the two base values are so close
(0.00157 vs 0.00141) a stale sidecar would load with **no numeric red flag at all** while
thresholding at 15× the wrong quantity.

**Back-compat.** A sidecar written before `smooth_instrument` existed has no `instrument`
key. Its value is not a guess — the endpoint was the only instrument then — so it is
backfilled as `"endpoint"` with a printed NOTE, and such a checkpoint resumes normally
under `--smooth-instrument endpoint`. Under `"chunk"` the backfilled value still
mismatches and still refuses, which is correct: that threshold measured another quantity.
`env_names` is recorded outside the guard and only **warns**, since extending a run to
new tasks is legitimate while `hf_ref` is state-dependent (~1.7× measured). Multi-task
runs calibrate on `env_names[0]` alone (per-iteration round-robin); the banner says so.

### Constrained dims and horizon

Built from the checkpoint's action `modality_keys` the same way `decode_action`
slices, so nothing is hardcoded. Note this is the **action** layout, which differs
from the state layout: `end_effector_rotation` is 3-dim axis-angle, while the
state's `end_effector_rotation_relative` is a 4-dim quaternion.

| dims | key | constrained? |
|---|---|---|
| 0:3 | `end_effector_position` | yes |
| 3:6 | `end_effector_rotation` | yes |
| 6 | `gripper_close` | **never** — 0/1 thresholded at 0.5, a grasp IS a step function |
| 7:11 | `base_motion` | off by default (`control_mode` gates it; inert in arm mode) |
| 11 | `control_mode` | **never** — same reason as `gripper_close` |

The **full 16-step horizon** is measured, not `n_action_steps=8`: the FM loss masks
with the full valid rectangle so `M` stays comparable to `ref_mse`; 14
second-differences instead of 6 cuts the per-entry standard deviation of `R` by 1.5×
(variance 2.2×), which is what keeps the hinge from switching on and off at random;
and `n_action_steps` is a deployment knob while the 16-step horizon is a property of
the checkpoint. Slicing happens **before** differencing — a `D²` straddling the pad
boundary of the `(50, 128)` output is meaningless.

### `smooth/*` metrics

All three HF readings are pooled the same way (`ΣR / 6ΣM`), so they sit on one axis.

| scalar | meaning |
|---|---|
| `smooth/hf_mean` | pooled HF of **whichever instrument is constrained** — the number `hf_ref` is compared against. Comparable across runs *as "the constrained quantity"*, not as a fixed physical quantity. |
| `smooth/instrument` | provenance (a TB text summary at step 0, a wandb string). Read this before comparing two runs' `hf_mean`. |
| `smooth/chunk_hf_mean` | pooled HF of the differentiable N-step chunk. Numerically identical to `hf_mean` under `"chunk"`; **absent** under `"endpoint"`, where no chunk is rolled out. |
**Running a control that still logs the executed metrics.** `smooth_coef = 0` is a
*hard* off-switch: no extra forwards, no extra RNG, and no `smooth/*` key at all — which
means a `coef=0` control emits no baseline for `executed_hf_mean` / `executed_jerk_ratio`
either, even though those need no forward pass. That is deliberate (the invariant is worth
more than the convenience), so for an A/B where the control must still report the
deliverable, pass a negligible coefficient instead of zero:

```
--smooth-coef 1e-8      # measurement on, pressure off (~1e-8 x dHF/dtheta against a
                        # grad norm of ~0.02, i.e. nothing), same compute as the
                        # constrained arm so the two are paired on cost as well
```

| `smooth/hf_max` | the largest single-minibatch pooled HF of the iteration. A rising max against a flat `hf_mean` means individual chunks are roughening while the energy-weighted pool hides it. |
| `smooth/calib_prestep_hf` | pooled HF over only the rows measured at exactly θ_base (before the first optimizer step of iteration 1). Compare against `hf_ref / smooth_hf_ref_scale` to see the in-iteration drift baked into the frozen threshold. At production sizes the window is `gradient_accumulation_steps × mini_batch_size` rows, so its own sampling noise is comparable to the drift it reports — read it as a coarse check, not a correction. |
| `smooth/endpoint_hf_mean` | pooled HF of the τ=0 implied endpoint. Monitoring only, and **free** — the chunk rollout's first step *is* `v_θ(ε, 0)`. |
| `smooth/executed_hf_mean` | pooled HF of `ready_actions` — the chunks the sampler actually emitted during collection. No forward at all, non-differentiable, measured on the real rollout distribution rather than reconstructed. Covers the **full valid horizon**, so it is directly comparable to `hf_mean`. Emitted whenever the feature is active *and* the pooled denominator is non-zero (an all-zero action buffer yields no reading rather than a divide-by-zero). |
| `smooth/executed_jerk_ratio` | `Σ\|D²a\| / Σ\|a\|` (L1) over the **`end_effector_position` dims** of `ready_actions`, restricted to the **executed prefix** `h < n_action_steps` — a normalized-space analogue of the physical path-jerk metric, needing no denormalization or decoding. `MultiStepWrapper` plays only steps `0..n_action_steps-1` into the sim and discards the rest (8 of 16 at the default), so a full-horizon number would half-measure something the robot never performed. The `(n-2)`-term numerator over an `n`-term denominator is the reference statistic's own shape, matching the notebook's `Σ\|diff(pos,n=3)\| / Σ\|deltas\|`. The position span comes from `modality_keys` exactly as the constrained set does; if the embodiment lacks the key the metric is **omitted** rather than redefined over dims that mix metres with radians. Neither executed metric sees the **chunk-boundary seam**, where physical jerk is typically worst — treat both as a lower bound. |
| `smooth/hf_ref` | the threshold actually in force, every iteration. |
| `smooth/loss`, `excess_mean`, `active_frac`, `hinge_mbs` | **hinge**-describing. Deliberately **ABSENT** — not 0.0 — on the calibration iteration and on any iteration where the hinge never evaluated, since a reported 0.0 is indistinguishable from "the field is already smooth", which is the reading the docs tell an operator to expect at the fixed point. |
| `smooth/rows`, `measured_mbs`, `nonfinite_mbs`, `undersized_mbs`, `nonfinite_loss_mbs` | coverage and skip accounting. |
| `smooth/calib_*` | calibration progress, including `calib_prestep_rows` (the strictly-`θ_base` subtotal, so residual drift is auditable). |

Every one of these is **ungated on `n_updates`**: they are measurements of the field
taken before the non-finite loss guard, so they stay valid on an iteration whose
update was discarded — and an iteration the smooth term itself killed is exactly the
one where they are most needed.

### Integration decisions

- **No loss divisor needed.** The term is one scalar per minibatch, so its magnitude
  is already independent of row count and anchor composition.
- **Anchor rows are included.** Roughness is not advantage-keyed, and anchors are
  the retention set we most want smooth.
- **Jitter-independent by construction.** Every smooth forward starts from the
  original ε, so `jitter_pos`/`jitter_neg`/`jitter_paired` do not change what the
  term measures.
- **The denominator is detached.** `∂HF/∂M < 0`, so a live denominator lets the
  model satisfy the term by adding DC (constant-along-`h`) energy: `D²` annihilates
  a constant, so `R` is untouched while `M` rises and `HF` falls. Detached, the
  directional derivative along "add DC" is **exactly zero**, because the `(1,−2,1)`
  stencil sums to zero. Covered by `test_smoothness.py`.
- **A non-finite HF never reaches the loss.** The smooth forwards run on
  large-magnitude inputs (pure noise at the first step) and are the likeliest of the
  K+N to overflow bf16 while every other term stays healthy. The term screens its own
  value and counts rejections in `smooth/nonfinite_loss_mbs`, so a bad reading costs
  one measurement instead of an iteration mis-attributed to bf16 ratio overflow.
- `model_path` (default `nvidia/GR00T-N1.6-3B`)
- `embodiment_tag` (default `ROBOCASA_PANDA_OMRON`)
- `lora_rank` / `lora_alpha` / `lora_dropout` (default 16 / 32 / 0.0)
- `lora_target_modules` — defaults to `DEFAULT_LORA_TARGET_MODULES` from
  `lora_dit.py`: 8 module patterns inside each of the 32 DiT blocks
  (`attn1.to_{q,k,v}`, `attn1.to_out.0`, `ff.net.{0.proj,2}`,
  `proj_out_{1,2}`). ~20M trainable params at rank=16.

**Episode collection**
- `group_size` (G) — logical rollouts per group. Default 8.
- `num_async_vector_env` — physical parallel-env workers per group. Default 4;
  `None` → `group_size` (one worker per rollout, unchanged). Set lower (must
  divide `group_size` and be `<=` it) to collect each group over
  `group_size // num_async_vector_env` sequential turns and cap peak worker
  RAM. See "Decoupling group size from worker count".
- `num_groups` — minimum groups per iter. Default 3.
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
- `scene_seed_pool_size` / `scene_seed_pool_base` — see "Frozen scene seed
  pool". Default 0 / None (**disabled**, bit-identical to a pre-feature run).
  `K > 0` freezes K scene seeds and cycles them across iterations so the same
  scenes recur; requires `K >= max(num_groups, max_groups)` and no
  `init_state_npz_path`. `base=None` resolves in place to
  `seed + 100_000`.
- `env_names: list[str]` — round-robin task selection.
- `episode_dir`, `episode_dirs_to_keep`.

**ZMQ wiring**
- `server_host` / `server_port` — in-process policy server (default
  `127.0.0.1:5555`).

**GRPO algorithm**
- `clip_eps_low` / `clip_eps_high` (both default 0.2) — asymmetric clip
  bounds; ratio clamped to `[1 - clip_eps_low, 1 + clip_eps_high]`. Each must
  be in `(0, 1)` (no ordering constraint between them).
- `clip_low_mse_coef` (default `0.0` = **OFF**, bit-identical to a flat
  `1 - clip_eps_low` floor). Makes the LOWER bound per-row and MSE-referenced:
  `budget_i = min(coef · MSE_ref_i, |ln(1 - clip_eps_low)|)` nats,
  `rho_floor_i = exp(-budget_i)`, so every row gets the same *relative* MSE
  budget `1 + coef` instead of the same absolute nat count (which measured 261×
  vs 2.1× allowed inflation across one iteration's rows). `clip_eps_low` stays an
  absolute ceiling, so this can only ever be tighter, never looser (enforced by
  snapping `rho_floor` up to the flat floor — see above). Must be
  finite and `>= 0`. Pair with `paws_k_floor_at_target`. See "Per-row,
  MSE-referenced lower clip".
- `update_epochs` (default 2)
- `mini_batch_size` (default 8 chunks)
- `gradient_accumulation_steps` (default 1) — mini-batches accumulated per
  optimizer step; see "Gradient accumulation". 1 is bit-identical to no
  accumulation. `k > 1` gives an effective batch of `k × mini_batch_size` rows
  at constant peak VRAM and ~`1/k` the optimizer steps; LR is per-iteration so
  it does NOT need rescaling. Must be `>= 1`.
- `kl_coef_last_iter` (default 0.2) — KL anchor to this iter's start-of-update
  policy snapshot. Bounds per-iter drift.
- `kl_coef_base_model` (default 0.2) — KL anchor to the pretrained DiT
  (LoRA disabled). Bounds cumulative drift from the base policy. 0.0 disables
  the term entirely (no extra forward pass per iter, no per-mb KL formula).
- `tau_centers` (default `[0.0, 0.25, 0.35, 0.5, 0.6, 0.75]`)
- `balanced_minibatch_training` (default `True`) — balanced mini-batch
  sampling; see "Balanced Training" mechanism 1.
- `dynamic_epoch_training` (default `False`) — tent-function epoch scaling;
  see "Balanced Training" mechanism 2. Independent of the flag above.
- `balanced_minibatch_positive_adv_ratio` (default `0.5`) — target fraction
  of positive-advantage chunks per mini-batch. Must be strictly in `(0, 1)`.
  Only active when `balanced_minibatch_training=True`. Raise above 0.5 (e.g.
  0.7) to bias more gradient steps toward success examples.
- `include_anchor_groups` (default `False`) — admit all-success groups into the
  ref pass and the update instead of dropping them as dead. See "Anchor
  groups". All-fail and singleton groups stay dead either way.
- `anchor_advantage` (default `0.0`) — constant advantage per anchor episode, in
  the same units as mixed-group z-scores. `0.0` = KL-only (retention constraint,
  no pull). Requires `include_anchor_groups=True`. Recommended starting value is
  the κ=2 pseudo-count figure `2/(G+2)`: **0.143 at `group_size=12`**, 0.200 at
  `group_size=8`. Pairs with `per_iteration_advantage_norm=True`.
- `anchor_max_row_frac` (default `1.0`) — cap on anchor chunks as a multiple of
  the signal chunk count; the compute knob and the strength knob at once. Waived
  when there are no signal chunks (an all-success iteration).
- `paws_k_floor_at_target` (default `False`) — floor the MEASURED PAWS `k` at
  `positive_advantage_weight_target_ratio` instead of at `1.0`. Removes only the
  "amplify less than target" case, so it cannot over-amplify. Exists because a
  tighter lower clip shrinks `N` and would otherwise *lower* `k`, weakening
  reinforcement exactly as the erosion brake tightens. Inert while `N/D > 1`
  (measured 1.04–1.06 when healthy); binds during collapse (measured 0.66).
  Requires `target_ratio >= 1.0`; only consulted when
  `positive_advantage_weight_scaling=True`. See "PAWS → Flooring `k` at
  `target_ratio`".

**Trajectory-roughness constraint** — see "Trajectory-Roughness Constraint" above.
- `smooth_coef` (default `0.0` = **OFF**, bit-identical to a run without the
  feature). Suggested starting range for the chunk instrument **0.15–0.5**: the
  last-step-differentiable rollout retains roughly a quarter of the true 4-step
  gradient's magnitude, so the coefficient may need raising relative to the 0.15
  that was calibrated on the endpoint. Bracket ±3×.
- `smooth_instrument` (default `"chunk"`) — `"chunk"` constrains the N-step
  generated chunk (ρ = +0.98 with EEF path jerk); `"endpoint"` constrains the
  1-step implied endpoint and reproduces the pre-change behaviour bit-for-bit
  (ρ = +0.00 late, which is why it is no longer the default). Anything else is a
  hard config error, validated even when `smooth_coef == 0`.
- `smooth_hf_ref` (default `None` = auto-calibrate from iteration 1 of a fresh
  run). A float is a flat threshold **in the units of the selected instrument**.
- `smooth_hf_ref_scale` (default **15.0**, was 4.0) — multiplier on the
  auto-calibrated base value. Instrument-specific: 15–17× base (~0.024) targets
  the iteration-4 chunk roughness level, ~2× less path jerk than the control
  run's peak-success iteration. **Use 4.0 with `smooth_instrument="endpoint"`.**
  Ignored when `smooth_hf_ref` is set explicitly, and ignored on a resume (the
  cached value already has its scale baked in — a warning says so).
- `smooth_include_base_motion` (default `False`) — admit `base_motion` into the
  constrained dim set. Off because `control_mode` gates it, so it is inert in arm
  mode. Discrete keys are excluded unconditionally.
- `smooth_calib_min_rows` (default `512`) — rows the auto-calibration must pool
  before freezing `hf_ref`, capped at 3 iterations.

**Optimizer**
- `learning_rate` (default 3e-5; ~3× lower than supervised FT because RL
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
- `cos_ref_lora_paths` (default `None`) — `(path_a, path_b)`, two existing
  `iter_NNNN/` LoRA checkpoint dirs (or `lora_weights.pt` files). Sets
  `L_early = W(b) - W(a)`, the reference direction `lora/cos_step_early` is
  measured against, loaded ONCE at setup. Validated at config construction: must
  be a 2-tuple of existing paths. See "Weight-step direction cosines".
- `cos_ref_iterations` (default `2`) — when `cos_ref_lora_paths` is unset, freeze
  `L_early = W_now - W_init` after this many LOGGED iterations. Must be `>= 1`.
  The own-run reference cannot detect a run that had already turned by then;
  that is what the explicit-paths form is for.

Logged scalars include `episode/{success_rate,mean_reward,std_reward}`,
`train/{loss,clip_loss,kl_loss_last_iter,kl_loss_base_model,clipfrac,mean_ratio,mean_log_ratio_abs,n_skipped_nonfinite}`,
`train/{n_updates,n_micro_batches}` (optimizer steps vs trained mini-batches —
these differ under gradient accumulation),
`train/n_nonfinite_grad_steps` (steps dropped to protect the weights from a
non-finite gradient; should stay flat at 0),
`train/learning_rate`, `time/iteration_seconds`, and (when
`dynamic_epoch_training=True` and at least one gradient step fired)
`balanced/{actual_epochs,success_fraction}`. With `scene_seed_pool_size > 0`,
also `episode/scene_sr/<seed>` (one curve per pooled scene) and
`episode/pool_pass`. `mean_log_ratio_abs`
is the primary diagnostic for DPPO-style surrogates: large values mean
the FM-MSE log-prob is noisy enough that most updates clip.

Always-on diagnostic families added alongside the per-row clip floor (pure
additions — emitted whatever the feature flags say):
`drift/{neg_down_p10,p50,p90,max, neg_rows, budget_mean, neg_frac_over_budget,
neg_frac_born_dead, neg_born_rows}` (the
per-ROW erosion-drift distribution pooled over every trained micro-batch — the only
view of the row spread that every `train/*` mean averages away, and what
`clip_low_mse_coef` should be calibrated from),
`lora/{cos_step_prev,cos_step_cumulative,cos_step_early,step_norm}` plus the
`lora/cos_ref_source` text summary (**read `cos_step_early`**: it reached −0.49 /
−0.62 on the two runs that collapsed directionally, while `cos_step_cumulative`
held +0.37…+0.53 straight through), and
`jitter/pos_clip_budget_used` (the positive side of the clip-budget share, on the
same flat denominator as its negative sibling; measured 53–76 % against the
negative side's 1.9–3.1 %, which is why `train/clipfrac` reads near 1.0 while
killing nothing). See "Per-row erosion-drift distribution" and "Weight-step
direction cosines".

---

## Operational Notes

- **GPU**: a single 24-GB+ NVIDIA GPU (training keeps frozen base in
  bf16, only LoRA params in fp32). Tested on A10G with `mini_batch_size=8`.
- **CPU/RAM**: each iteration's collector subprocess spawns
  `num_async_vector_env` MuJoCo workers (default `group_size`) for the one
  task being collected. 64+ GB RAM is comfortable for 5 workers. Lower
  `num_async_vector_env` (collecting each group over multiple turns) to fit
  larger groups on a RAM-limited host.
- **Robocasa venv**: located at
  `gr00t/eval/sim/robocasa/robocasa_uv/.venv/`. The subprocess collector
  path hard-codes this path (see `_collect_via_subprocess` in
  `train_grpo.py`); if you've put robocasa elsewhere, edit that path.
- **Memory creep**: there are small leaks in robosuite/MuJoCo's model
  reload path. The collector workers `gc.collect()` + `malloc_trim(0)`
  after every `apply_scene_bundle`; the trainer does the same at the
  start of each iter. Because a fresh collector subprocess is spawned and
  torn down every iteration, cross-iteration creep in the collector is
  bounded by construction.
- **Consecutive-failure abort**: the trainer aborts after 3 consecutive
  collector failures (timeout, non-zero exit, zero episodes loaded). The
  log line right before the abort lists common causes (wrong venv path,
  stuck port, missing MuJoCo backend, OOM).
