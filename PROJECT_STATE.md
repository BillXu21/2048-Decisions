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

## Current research handoff (2026-11-20)

Read the [attention and exact-pair experiment progress report](evidence/attention_pair_2026-11-20.md) before planning new model work. This is a text-first handoff of local-agent reports, not a separate replay or independent experiment audit.

- Packet 1A/1B audited a bounded expert sample, surveyed 120 spatial cell pairs, and selected a small exact-pair lookup. Packets 2A–2B.2 implemented attention-only (A, 30,849 parameters), parameter-matched nonlookup (B, 34,181), and exact-pair memory (C, 34,182). One seed completed 249,856 PPO transitions/arm from one-epoch BC origins, with mean PPO development scores **12,274.9 / 10,529.0 / 11,086.0** for A/B/C on 128 paired games. No lookup superiority established.
- Packets 3A–3C simplified the active Windows/WSL launch and paired recovery path while preserving historical source and checkpoints. Production CUDA parity was locally reported as passing; Packet 4A later confirmed the corrected launcher's successful live exit.
- Packet 4A trained **12 BC epochs per arm**, selecting **A/e12 = 13,473**, **B/e12 = 14,676**, **C/e8 = 14,732** mean on **128 BC development games**. These selected origins have **not** undergone PPO. Performance is substantially below the earlier 4.8M-parameter CNN BC reference, but the protocols and datasets are not matched enough to isolate model capacity.
- **Next discussion, not an authorized run:** inspect the actual unique expert examples/games used by BC and consider whether model capacity (potentially ~250k–1M parameters), architectural allocation, or training-data coverage is the best lever. The proposed ~2M-transition PPO learning curve was not launched. Keep existing A/B/C artifacts and decisions; no scaling architecture or GPU budget has been approved.

The [older spatial pattern plan](.agents/notes/draft/architecture/2026-11-18-pattern-memory-and-attention.md) records the initial rationale but is not the current task list or an implemented decision. New unverified drafts follow the global decisions skill's `.agents/drafts/` convention.
