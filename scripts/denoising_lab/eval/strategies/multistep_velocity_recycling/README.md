## Strategy 3: Multistep Velocity Recycling

**Category:** Drop-in replacement | **NFEs:** 4 (same as baseline) | **Retraining:** None

### Overview

The key insight: in a multi-step ODE solver, each step produces a velocity prediction $v_i$ that is discarded after use. But that velocity contains information about the local curvature of the flow. By *caching* the velocity from the previous step and combining it with the current velocity, we can construct a 2nd-order update (Adams-Bashforth) with **zero additional NFEs**.

This is inspired by the **DPM-Solver++ multistep** variant, which achieves 2nd-order accuracy with only 1 NFE per step by reusing cached model outputs.

### Mathematical Formulation

**Adams-Bashforth 2-step (AB2):** Given velocity predictions at two consecutive $\tau$ values, the 2nd-order update is:

$$a_t^{\tau_{i+1}} = a_t^{\tau_i} + \Delta\tau \cdot \left(\frac{3}{2} v_i - \frac{1}{2} v_{i-1}\right)$$

where $v_i = v(a_t^{\tau_i}, \tau_i, o_t, l_t)$ and $v_{i-1}$ is the cached velocity from the previous step.

**For the 4-step denoising loop:**

| Step | Method | Formula | NFEs Used |
|------|--------|---------|-----------|
| 0 | Euler (no cache yet) | $a_t^{0.25} = a_t^0 + 0.25 \cdot v_0$ | 1 |
| 1 | AB2 | $a_t^{0.50} = a_t^{0.25} + 0.25 \cdot (\frac{3}{2} v_1 - \frac{1}{2} v_0)$ | 1 |
| 2 | AB2 | $a_t^{0.75} = a_t^{0.50} + 0.25 \cdot (\frac{3}{2} v_2 - \frac{1}{2} v_1)$ | 1 |
| 3 | AB2 | $a_t^{1.0} \;\;= a_t^{0.75} + 0.25 \cdot (\frac{3}{2} v_3 - \frac{1}{2} v_2)$ | 1 |

Total: 4 NFEs (identical to baseline), but steps 1–3 are 2nd-order accurate instead of 1st-order.

### Pseudocode

```python
def denoise_ab2(a_noise, vl_embeds, state_embeds, embodiment_id):
    """4-step Adams-Bashforth-2: 2nd-order accuracy with zero extra NFEs."""
    a = a_noise
    dt = 0.25
    prev_velocity = None

    for step in range(4):
        tau = step / 4
        tau_bucket = int(tau * 1000)
        velocity = DiT(a, tau_bucket, vl_embeds, state_embeds, embodiment_id)

        if prev_velocity is None:
            # Step 0: standard Euler (no history to use)
            a = a + dt * velocity
        else:
            # Steps 1-3: Adams-Bashforth 2-step (2nd-order, uses cached velocity)
            a = a + dt * (1.5 * velocity - 0.5 * prev_velocity)

        prev_velocity = velocity

    return a
```

### How It Replaces Action Chunking

Completely transparent. Output shape, normalization, decoding, and chunk execution are all unchanged. The only difference is how successive velocity predictions are combined in the update rule.

### Analysis

| Aspect | Assessment |
|--------|------------|
| **Expected quality** | Moderate. AB2 is provably 2nd-order for smooth velocity fields, giving $O(\Delta\tau^2) = O(0.0625)$ global error vs Euler's $O(0.25)$ — a ~4× reduction. The practical improvement depends on velocity field smoothness. |
| **Risk** | Low-moderate. AB2 has a smaller stability region than Euler, which could cause divergence if the velocity field has sharp discontinuities. However, flow matching with rectified flow produces relatively smooth fields. |
| **Latency** | Identical — 4 NFEs × ~16ms = ~64ms. |
| **Implementation** | Trivial — cache one tensor, change one line of the update rule. |

### Prior Work

- **Lu et al., "DPM-Solver++: Fast Solver for Guided Diffusion Sampling"** — arXiv:2211.01095. The multistep variant achieves 2nd-order accuracy with 1 NFE per step. Original DPM-Solver (arXiv:2206.00927) introduced the exponential integrator approach.
- **Adams-Bashforth methods** — classical numerical analysis. AB2 is the simplest linear multistep method. See Hairer, Norsett, Wanner, "Solving Ordinary Differential Equations I" (1993).

### How to run

From the **repo root**:

```bash
# Terminal 1 (model venv) — start the AB2 server
bash scripts/denoising_lab/eval/strategies/multistep_velocity_recycling/run_server.sh

# Terminal 2 (sim venv) — run the reproducible benchmark
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
    scripts/denoising_lab/eval/robocasa_eval_benchmark.py \
    --env-names robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n-episodes 10 --seed 42 \
    --output-dir /tmp/benchmark_results/multistep_velocity_recycling \
    --strategy-name multistep_velocity_recycling
```

To override server options (e.g., port):

```bash
bash scripts/denoising_lab/eval/strategies/multistep_velocity_recycling/run_server.sh --port 5556
```

---
