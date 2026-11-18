# PPO throughput, parity and optimization dynamics

The completed V3 experiment retained 3 epochs / 256 minibatch and full D4. Optimization changes were tested separately from copies of saved states. Timings here are synchronized wall times, not CUDA compute efficiency or device utilization. On the RTX 4070 Laptop GPU (8 GB, WSL/JAX CUDA), the overnight benchmark used a 1900 MHz cap and thermal pause/resume at 80/70°C. Earlier study clock regimes changed; these timings are not hardware constants.

## Isolated measurements and chunk-rate estimates

| Arm | Epochs / minibatch | Update wall s | Estimated transitions/s including warm preparation |
|---|---|---:|---:|
| CNN | 3 / 256 | 10.590 | 330.7 |
| CNN | 3 / 512 | 9.186 | 372.9 |
| CNN | 1 / 256 | 3.480 | 776.3 |
| CNN | 1 / 512 | 3.455 | 779.9 |
| Hybrid | 3 / 256 | 9.180 | 375.0 |
| Hybrid | 3 / 512 | 8.727 | 391.3 |
| Hybrid | 1 / 256 | 3.953 | 719.4 |
| Hybrid | 1 / 512 | 3.146 | 838.1 |

Two fixed-batch timing replicates per setting with shared measured warm prep correcting an early cold-JIT interval. Sizes 1024/2048 failed the memory precheck and have no steady measurement. Independent profiling placed the optimizer at ~85.5% of the reconstructed warm CNN chunk (rollout 0.652 s, actor features 0.914 s, critic inference 0.142 s, GAE 0.089 s), and ~82.8% of the profiled hybrid chunk (rollout 0.595 s, features 0.925 s, critic 0.148 s, GAE 0.074 s). The initial dedicated CNN profile failed; its numbers come from the sweep. GPU wait is not pure host idle, and separately compiled components are not additive. The original 10M report's **last-session** rates were 336.1/s CNN and 362.8/s hybrid; logged chunks over the whole study included crash-replay duplicates and thermal/launch regimes differed. Do not extrapolate either number to a new run with evaluation, cooling, compilation and serialization.

## Exact parity and short learning tests

- Fast `lax.scan` reproduced the checked weights and Adam states but used FP32 running KL instead of the original NumPy float64 prefix mean, which can change early-stop and RNG consumption near threshold. Strict scan with the original NumPy mean callback passed representative state and forced-stop checks: measured optimizer speedups 1.232× CNN / 1.303× hybrid. `RECOMMENDATION.md` gives derived chunk estimates 1.112× / 1.242×, excluding evaluation/compile/serialization; its CNN estimate differs from the broader range in the executive summary. Neither scan replaced the reference. Bulk transfer preserved state/RNG on checked inputs; timing samples were too few for a strong speed claim. Critic features use absolute actor scores: normalized log probabilities lose the offset and are not an exact substitute.
- Three CNN-only forks from exact 6.5M states trained 131,072 **new** transitions each. Fresh paired 256-game evaluation (5100000–5100255): `e1_b512` scored 47,794.4 (−137.9 vs origin; actual 767.0 transitions/s), `e3_b256` 47,664.1 (−268.3; 330.2/s), `e3_b512` 46,515.1 (−1,417.3; 372.1/s). One-epoch/512 minus original +130 [−3,449, +3,656]. All within-origin gain intervals include zero. Epoch and minibatch change together in 1/512 vs 3/256; learning efficiency is unresolved. Rate here is measured on short training, not proof of sustained gains.
- Optional frozen-rollout horizon work did **not run** after an expected audit file was missing. The original study audit later passed CPU-only. λ=.98 assigns residual weight `.98^k`; longer rollouts reduce the boundary effect without eliminating internal critic bias. No 25M/50M V3 PPO extension or rollout-horizon learning comparison occurred.

## Historical telemetry vs diagnostic replay

The later audit removed 13 duplicate CNN chunks and found 2,460 committed chunks/arm, 117,240 optimizer steps per actor (and critic), and **zero** KL early stops. Recorded PPO ratio clipping averaged 1.683% CNN vs 1.350% hybrid; categorical forward KL .000786 vs .000615; hybrid led initially, CNN after 2.5M. These are chunk averages, not the KL running prefix used for stopping. The .5 global gradient clip is a separate mechanism. Averaged branch norms imply at least one clipping event in 2,459/2,460 CNN chunks and all hybrid chunks, not the historical fraction of clipped minibatches. KL is measured before the step; stopping checks the per-epoch running mean after the update.

Four diagnostic replays used one new fixed 4,096-transition batch each (two checkpoints per arm), not historical epoch logs or resumed learning. Actor clipping occurred in 190/192 minibatches across the replays. Epochs 2–3 accounted for ~62–68% of measured in-sample surrogate improvement, in both arms. This does not show gameplay utility or architecture-specific epoch needs. The hybrid late score decline is unresolved in paired development games. High critic EV is a bootstrapped-target fit, not calibration to remaining score. A future causal test could hold batch 256 fixed and compare 1 vs 3 epochs for both architectures with replicated forks and paired fresh endpoint gameplay.

**Local evidence:** `jax_experiments/ntuple_hybrid_v3/ppo_benchmarks/{PROFILE.md,PPO_SWEEP.md,PARITY.md,RECOMMENDATION.md,OPTIMIZATION_DYNAMICS_AUDIT.md,EXECUTIVE_SUMMARY.md,ROLLOUT_HORIZONS.md}`, `results/sweep_analysis.json`, `results/pilot_evaluation.json`, and `optimization_dynamics_audit/historical_summary.json`. Rates and intervals are historical measurements/estimates, not rerun for publication.
