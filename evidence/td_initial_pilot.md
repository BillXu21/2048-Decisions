# Afterstate TD: formulation and initial pilot

Let `x_t` be the post-slide/pre-spawn board and `r_t` the merge score for producing it. The CNN chooses the legal action maximizing `r_t / 12800 + V(x_t)`. Target for `V(x_t)` is the greedy max of `r_(t+1)/12800 + V_target(x_(t+1))` after the actual spawn, or zero if the spawned board is terminal. Do **not** add `r_t` to that target: rewards 4, 8, 16 imply afterstate remaining scores 24, 16, 0. γ=1; resets do not participate in the previous path. TD(8) accumulates seven actual subsequent rewards then greedily bootstraps at the eighth decision, with terminal masks. Its replay and intermediate actions are uncorrected off-policy. SARSA backup was implemented/tested as an analytic option, not compared as a learning run. A lagged max network is not Double Q-learning.

| Learner | Implementation |
|---|---|
| CNN TD | 494,657 params; 18 one-hot planes, conv64 plus three two-conv residual blocks, flatten 1024→256 GELU→scalar, differentiable D4 mean, small random head, not the V3 PPO actor or critic. Adam 3e−4, global clip 1, Huber δ=1, replay 8192, 32×8 collection and two 128-example updates per 256 generated moves, target copy every 2048. Sample UTD≈1. |
| Scalar lookup-table TD | Four scalar `18^4` tables (419,904 params), outer/inner rows and corner/central 2×2, eight shared D4 occurrences each. Sum of 32 active terms. Direct online one-step update (α=.1 divided by squared active multiplicities to handle repeated addresses); no CNN, vector embeddings, replay, Adam, or target lag. Zero scratch tables; greedy with random environment spawns. |

The four arms used one seed each. Warm starts fit complete return labels from 16 frozen PPO games (12 fit/4 heldout, 37,163 moves; four passes over the 25,230 fit rows). This is teacher-policy regression, not imitation of action preferences; PPO state coverage and initialization cost were additional. Selection by mean over 128 shared dev games (6100000–6100127); no independent final confirmation. Values below are reported development means; loop time includes cold compile/guards and excludes gameplay/serialization/MC fit.

| Arm | 0 | 65,536 | 131,072 | 262,144 | Actual loop transitions/s |
|---|---:|---:|---:|---:|---:|
| `cnn_scratch` | 1,458.2 | 3,209.6 | 5,388.0 | — | 1,587.0 |
| `cnn_warm` | 298.1 | 3,438.4 | 5,677.0 | — | 1,610.3 |
| `tuple_scratch` | 2,972.8 | 4,198.9 | 4,797.8 | 6,419.5 | 7,710.0 |
| `tuple_warm` | 1,003.1 | 3,066.5 | 3,983.1 | — | 5,980.1 |

At 131,072 the paired warm-minus-scratch CNN effect was +289 [−431, +1,000]; tuple −815 [−1,271, −366]. The tuple's 6,419.5 used twice the new transitions but only 34.0 loop seconds versus CNN scratch 82.59 s. This supports a cost advantage in that window, not learning-rule superiority. Frozen PPO scored 48,391.3 on these seeds. Selected policies reached no 2048 on this screen. A synthetic tuple direct-update microbenchmark (38.7k/s) is not full simulator throughput. The supplemental environment/policy-only profile overlapped a separate workload and is not a clean decomposition. Compared with contemporaneous PPO fixed-update projections at 281.5 (3/256) and 531.9 (1/512) transitions/s, CNN TD was faster per move, but neither architecture nor learning outcomes were matched.

## Calibration and operations

On eight complete own-policy diagnostic games per selected policy, CNN warm bias +38,016 (RMSE 38,906), versus scratch −2,108; tuple warm bias +14,380 versus scratch −1,360. Better fixed-PPO-return error did not predict better self-policy calibration. Long games produce more diagnostic rows; no late-stage calibration claim follows. Greedy selected-action optimism is observable; causal max-backup bias and distribution mismatch were not isolated. The warm protocol was not retained.

Analytic target, legality/spawn, D4, repeat-index, target detach/copy and serialized resume checks were reported. A CUDA collection error stopped an early attempt; the unreturned move count is unknown. A separately authorized smaller-chunk smoke passed before completion (no change to the training budget). Hardware/game workload monitoring was strengthened later; do not retroactively attribute the original timings to that change.

**Local evidence:** `jax_experiments/afterstate_td/{REPORT.md,DECISIONS.md,config.json,EXPERIMENTS.jsonl,td_core.py}`, `runs/*/selfplay_diagnostics.json`, `results/{preflight.json,profile.json}`. Background references: [literature](../references/literature.md). Next evidence: [scratch follow-ups](td_followups.md).
