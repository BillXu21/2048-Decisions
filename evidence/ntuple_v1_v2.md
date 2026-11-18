# Neural N-tuple V1 and V2: retrofits to the historical MLP

These are learned vector embeddings feeding neural policy logits, not traditional scalar lookup-table TD. Both start at the preserved 45,023,232-transition MLP PPO checkpoint. V1 used an additive zero-output-init branch over four four-cell tuple families (18 exponent categories, width 8, D4-shared occurrences, 3,361,409 parameters). A 3,304,961-parameter MLP residual tested capacity without tuples. The base actor stayed frozen for branch arms, while branch and critic trained. Plain continuation was the control. V2 concatenated 512-d pretrained hidden features and 32 aggregated tuple features as input to a nonlinear zero-output-init fusion head; it tested joint actor training and frozen screens, both with a capacity control. Both kept matched PPO settings within comparison (γ=1, λ=.98, clip .15, three epochs of 4096-row minibatches, full D4, no reference KL).

## V1: 256-game screens and 1,024-game fresh confirmation

Official branch milestone was cumulative 50,003,968 (+4,980,736 from the 45M origin). Separate game seed blocks: 853000–853255 development; 1700000–1701023 final. Each row compared to the original actor in-process. Fresh V1 files store aggregates, so **screen CIs do not apply to final differences**.

| Arm | Screen paired TTA change [95% CI] | Fresh TTA mean | Fresh change |
|---|---|---:|---:|
| Frozen 45M start | — | 47,100.1 | — |
| Plain `control_5m` | +93.7 [−3,992.0, +4,225.7] | 47,497.8 | +397.8 |
| Frozen-base tuple `hybrid_5m` | −3,047.9 [−7,116.2, +1,082.6] | 42,219.9 | −4,880.2 |
| Frozen-base `capacity_5m` | −5,547.0 [−9,251.3, −1,775.9] | 41,381.8 | −5,718.3 |

Counter semantics caused an unplanned ~50M **additional** transition branch-only run; it is not the official 5M arm. Paired development change at 5.05/10.03/15.01/20.05/25.03/30.02/35.06/40.04/45.02/50.00M was −5,327 / −16,285 / −19,917 / −18,027 / −15,961 / −18,562 / −26,350 / −26,785 / −24,307 / −23,254 (rounded). Decline was not monotone. Earlier optimism came from sampled training scores, not deployed paired gameplay. Critic EV fell to −.59 in the overrun; critic-feature drift is a hypothesis, not isolated causality. V1 joint extensions H2/C2 were skipped.

## V2: nonlinear fusion did not rescue the residual

Development seeds 870000–870255. Frozen fusion/capacity changes −2,727.8 / −1,758.6 were unresolved; joint changes −5,251.4 / −10,368.9 were negative. The conditional 15M extension gate was not met. Fresh 1,024 games (2600000–2601023) re-evaluated V1 comparators, not the two V2 frozen screens:

| Candidate | Single mean | D4 mean | D4 difference vs start [paired 95% CI] | 4096 / 8192 |
|---|---:|---:|---|---|
| Start45M | 7,533.4 | 45,244.8 | — | 42.6% / 1.6% |
| V1 plain control | 7,327.9 | 47,257.1 | +2,012.3 [+30.5, +3,970.7] | 45.8% / 3.3% |
| V1 frozen tuple | 7,457.8 | 42,680.7 | −2,564.1 [−4,431.9, −705.0] | 38.1% / 1.7% |
| V2 joint fusion | 7,730.4 | 41,321.6 | −3,923.1 [−5,813.7, −2,038.4] | 35.3% / 1.2% |
| V2 joint capacity | 8,363.9 | 36,051.0 | −9,193.7 [−10,990.5, −7,429.0] | 25.5% / 0.3% |

The capacity control's positive single-orientation result did not survive D4. Plain continuation's small positive V2 comparison qualifies V1's “plateau” language; no general plateau or reliable extra-budget win is established. Per-stage rematerialization resolved capacity-arm memory pressure. Matmul precision/highest and deterministic GPU flags reduced cross-context variation; confirmed in-process comparisons are preferable to cross-compile score subtraction. V2 fusion joint was slower than plain in measured update/4096 (0.277 vs 0.048 s); neither neural fusion variety was retained. A direct V2-vs-V1 paired interval is not inferred from separate baseline contrasts.

**Local evidence:** `jax_experiments/ntuple_hybrid/{REPORT.md,DECISIONS.md,EXPERIMENTS.jsonl,results/final_confirmation.json}` and `jax_experiments/ntuple_fusion_v2/{REPORT.md,DECISIONS.md,EXPERIMENTS.jsonl}`, especially §§3–4. Reported results, one training seed per treatment; targeted controls are not proof against other tuple representations.
