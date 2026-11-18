# Project state

| Best confirmed within project evidence | Checkpoint | Result |
|---|---|---|
| Overall learned policy | `jax_experiments/ntuple_hybrid_v3/ppo_pilot_10m/runs/cnn/SELECTED.pkl` | **50,407.04** mean on **1,024** fresh games, greedy with full eight-way D4. 51.95% reached 4096; 9 games reached 8192. |
| Afterstate TD | `jax_experiments/afterstate_td/bc_initialized/runs/td1/new_12000000.pkl` | **24,926.47** mean on **128** fresh games, 55.47% reached 2048, 6.25% reached 4096. One seed; scratch continuation, 12M new / 13,238,528 cumulative TD transitions. |

See [hashes](references/artifact_index.md), [CNN confirmation](evidence/cnn_ppo_10m.md), and [TD confirmation](evidence/td_long_horizon.md). The TD packet re-evaluated frozen PPO on its own 128 seeds (49,336.41 mean); that is the relevant direct control. The overall designation is for mean score, not rare-tile probability, single-orientation strength, or a claim of general superiority across hardware and protocols.

## What is complete and what to retain

- Search teacher and early learned critics; scratch board-to-action PPO; imitation and afterstate ablations; historical MLP PPO up to 50M with **45M** selected; three generations of neural tuple experiments; V3 BC and 10M PPO per arm; and isolated PPO optimization studies. The V3 CNN actor with full D4 is the strongest deployed configuration. Retain the reference 3-epoch/256-minibatch protocol for comparisons, not as a claim of optimality.
- TD: completed pilot, three-seed 65k screen and 262k continuation, partial matched-time follow-up (eight of nine runs), and the long scratch fallback with one guarded stop and a selected-policy confirmation. Neither MC teacher-return warm start nor the tested BC calibration/repair was accepted. Slow-refresh one-step TD is the strongest observed TD candidate, not stable by default. TD(8) is a comparison arm, not a robust winner.
- The demonstrated failure modes are orientation dependence, cross-compile argmax drift, worsening gameplay despite improved imitation loss, harmful reference anchoring and neural residual branches, fast-refresh TD overestimation, and later excessive bias even with slow refresh. See the [decision index](DECISIONS.md).

## Reproducibility and open questions

Headline arithmetic and selected checkpoint bytes were independently checked from stored files, not independently retrained or replayed. Historical reports include parity and resume checks; most architecture studies used one training seed. Missing source Git identity, absent raw V1 final arrays, and unexecuted work are recorded in the [reproduction guide](references/reproduction.md).

Possible next experiments, **not implemented here or committed plans**: compare stable one-step and longer backups over several seeds with matched starting weights and compute; evaluate Double Q-style selection/evaluation separation and verify calibration against complete own-policy returns; test joint imitation plus value/terminal calibration and compare to scratch; measure PPO gameplay benefit of epoch reuse at fixed minibatch size; audit exact compiled optimizer parity over many chunks; compare architectures under matched data and compute. Longer or full-episode PPO rollouts remain untested as a learning intervention.
