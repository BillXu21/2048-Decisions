# Scratch TD: target refresh, eight-step targets and matched loop time

Matched within each of three scratch replicas (initial seeds 20261101–20261103): initial CNN weights, environment and replay RNG streams (trajectories diverge when actions do). Every stage restores full state and keeps the [pilot](td_initial_pilot.md) CNN/D4/reward/optimizer recipe. The arms vary only practical future backup and hard-copy interval: `one_step_slow` (1/2048), `one_step_fast` (1/256), `eight_step_slow` (8/2048). Intermediate actions and replay are uncorrected off-policy. None of these packets evaluated a fresh PPO control.

## `followup_01` and `followup_02`: early fast refresh reversed

At 65,536 moves per run, all nine completed (589,824 generated; 510 Adam updates and 65,280 admitted replay rows per run, last 256 pending). Follow-up 02 added 196,608 each, 1,769,472 new overall / 2,359,296 cumulative. Fixed milestones on 128 paired development games per replica; no best-checkpoint or held-out confirmation.

| Condition | 65,536 | 131,072 | 196,608 | 262,144 | 262,144 seed means (r0/r1/r2) |
|---|---:|---:|---:|---:|---|
| 1 / 2048 | 2,929.3 | 6,054.3 | 6,733.0 | 7,951.0 | 7,986.7 / 7,125.4 / 8,740.9 |
| 1 / 256 | 4,883.8 | 6,102.3 | 6,841.7 | 6,861.7 | 5,448.8 / 6,769.9 / 8,366.4 |
| 8 / 2048 | 5,037.3 | 6,351.9 | 7,822.8 | 9,860.6 | 10,897.2 / 10,986.9 / 7,697.7 |

At 65k, the fast one-step effect vs slow +1,954.5 [1,435.6, 2,510.0]; eight-step +2,108.0 [1,376.0, 2,946.2]; their contrast +153.5 [−457.8, +808.2]. Fast refresh was then the simpler candidate to test longer, not a settled choice. At 262k, slow beat fast in all three seeds; fast−slow −1,089.3 [−2,478.8, +144.7]. Eight−slow +1,909.6 [−847.0, +4,064.7]; eight−fast +2,998.9 [−475.0, +5,526.2]. Eight-step had the highest mean but one seed regressed in the final segment. Only eight-step reached 2048 at this endpoint (0.78% / 6.25% / 0.78%).

State-weighted self-policy bias (eight games per model, averaged across models) at 65k→131k→262k: slow one-step −1,974→−3,160→−2,708; fast +305→+3,697→+28,018; eight-step −1,766→−405→+354. The fast curve flattened as estimates inflated. Loop rates at 65k were ~1,344–1,355/s across arms; the continuation measured ~1,401–1,461/s (not full research wall time).

## `followup_03`: 600 additional loop seconds, not matched transitions

Started from the nine exact 262,144-transition states; wall clocks include synchronous collection, target/update, preparation and guards, not evaluations or serialization. Same 128 dev streams (7100000–7100127). Packet **failed overall**: eight runs completed; fast r2 last saved at 154.23 s / 483,328 cumulative and hit the normalized absolute bound 100 (1.28M score units), with no retry. Exact failure batch values/time are unknown; the last saved checkpoint was finite.

| Condition | 0s | 120s | 240s | 360s | 480s | 600s |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 2048 | 7,951.0 | 9,323.0 | 10,866.9 | 11,384.9 | 13,395.7 | 14,744.0 |
| 1 / 256 | 6,861.7 | 4,747.6 | 8,020.6 (2) | 6,352.9 (2) | 6,351.5 (2) | 8,765.9 (2) |
| 8 / 2048 | 9,860.6 | 10,404.0 | 11,441.8 | 12,178.7 | 10,992.3 | 12,121.8 |

The late fast-refresh means use the two survivors only, not three seeds. Fast r0 and r1 biases nevertheless fell from +28,488/+38,677 to +521/+11,851, so overvaluation is not necessarily monotone. Slow-refresh one-step gained +6,793 [3,755, 9,626] and improved in every replica. Eight-step +2,261 [−1,027, +6,057], with one declining replica. Eight−slow at endpoint −2,622 [−5,795, +1,365]; gain difference −4,532 [−9,190, +2,281]. Completed arms accrued ~879k–976k additional transitions; this cannot be turned into an equal-transition comparison. Retain slow one-step and eight-step for study, not fast refresh unchanged. The later [long study](td_long_horizon.md) restored the corresponding replica-0 states, not the three-replica averages.

**Evidence:** `jax_experiments/afterstate_td/followup_01`, `followup_02`, `followup_03`: each `REPORT.md`, `DECISIONS.md`, `config.json`, `EXPERIMENTS.jsonl` and `results/`; follow-up 02 full `results/learning_curves.csv`. Intervals reported from 10,000 crossed bootstrap draws over three training replicas and paired game seeds, exploratory (shared games, repeated looks, no multiplicity correction).
