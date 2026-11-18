# Long-horizon CNN TD: BC rejected, scratch fallback completed

The directory `jax_experiments/afterstate_td/bc_initialized/` names the question, not the outcome. Actual initialization: **`scratch_fallback`**. The recovered worker completed the bounded study, but TD(1) was guard-stopped; only TD(8) reached its cap. New generated transitions **26,999,808**, excluding prior TD, BC examples and diagnostics. One training seed per arm.

## Why action BC did not initialize TD

The 494,657-parameter value CNN was trained as a policy (`r/12800 + V` logits, legal mask, full differentiable D4) using the 61,042-state soft teacher pool only, episode split 112/16, T=32. Hard-action corpus lacks Q vectors and was not substituted. Batch 64, Adam 3e−4, clip 1, ten actual epochs on 80-epoch cosine schedule. Initial workflow finished all ten BC epochs but failed in selection (missing contrast helper); no calibration or TD in that attempt. CPU recovery rebuilt selection without altering weights/retraining. The 128-game BC dev means for epochs 1/2/3/5/8/10 were 15,357 / 19,324 / 16,963 / 18,575 / **22,135** / 20,375. Epoch 8 selected despite lower validation CE at epoch 10 (.70661 vs .71188); epoch 8 accuracy 80.64%, pairwise 90.77%.

Softmax preferences cannot identify a common logit offset; neither low CE nor action agreement ensures calibration or correct terminal values. The single own-policy calibration attempt (12 complete games, 8 fit/4 heldout, four passes) improved held-out RMSE 204,786→7,152 and bias +156,165→+98, but failed terminal mean, terminal q95, ranking (33% agreement) and gameplay gates. No rescue, rescale or switch to TD on this head. Earlier planned budgets in the failed report and lower README retention notes are not executed budgets; the completed report and arrays govern.

Separate `bc_repair_01` optimized only the scalar output head on frozen features, trading off return fit and ranking preservation. Strengths 1/10/100 yielded fit RMSE 8,227/15,659/32,635 and fit agreement 57.4%/73.6%/84.4%; all reached the 300-iteration limit. The selected strength 100 reduced held-out RMSE 203,192→31,455 and bias +153,759→+7,129; agreement 83.56%. Terminal values remained far from zero (about +25k to −37k); required gates failed and gameplay was not run. No candidate was accepted/injected. This does not test a joint representation-and-value pretraining objective.

## Full development history of the scratch continuation

Restored full `followup_03` replica-0 states, 1,238,528 prior moves (TD1) and 1,191,424 (TD8); weights, optimizer, target, env, replay, pending and RNG differ by arm. Same CNN/scale/Huber/clip/Adam, 32×8 collection, 2×128 sample updates, target sync 2048 cumulative. Alternating bounded segments, common **32** dev games 8500000–8500031. Cells show mean with own-policy diagnostic bias in parentheses; bias from four complete games per recorded checkpoint and state-weighted; missing diagnostics marked `—`. Cumulative transitions at any row = new + that arm's prior moves.

| New moves | TD(1) mean (bias) | TD(8) mean (bias) |
|---:|---:|---:|
| 0 | 14,420.8 (−600) | 11,538.8 (+3,546) |
| 499,968 | 15,755.5 (—) | 11,211.4 (—) |
| 999,936 | 17,295.9 (+9,988) | 13,434.1 (+5,969) |
| 1,999,872 | 14,850.5 (+18,299) | 15,818.8 (+10,354) |
| 4,000,000 | 27,768.0 (+45,933) | **18,024.2** (+5,233) |
| 5,999,872 | 23,623.6 (+67,845) | 9,828.6 (+8,227) |
| 8,000,000 | 28,322.1 (+65,994) | 12,472.9 (+5,989) |
| 9,999,872 | 18,209.6 (+88,857) | 13,751.1 (+9,667) |
| 12,000,000 | **28,813.8** (+105,729) | 11,102.2 (+4,772) |
| 14,999,808 | No observation: stopped | 14,458.6 (+12,164) |

Bold = highest dev mean (including origin, earlier ties). TD(8) rose early then reversed; it did not return to its 4M peak. TD(1) swung and recovered at 12M while crossing the confirmed own-return bias guard. Two successive excessive-bias flags (bias above max of 20,000 or four times mean realized remaining score) followed by 16-game verification: bias +98,896, prediction mean 120,217. This stop was not a nonfinite/CUDA error, nor caused by low score. The separate absolute bound is 100 normalized units. TD(8) ended at 16,191,232 cumulative and TD(1) at 13,238,528. TD(1) ~1,370.8 loop moves/s, TD(8) ~1,365.0, excluding evaluation/serialization and periodic target-drift diagnostics; full TD segments ~21,024 s combined. No 15M TD(1) curve is inferred. Target-copy equality and pre-copy drift are separate diagnostics from independent own-return calibration; replay residuals are not complete-return errors.

## Fresh held-out gameplay: 128 paired games (8800000–8800127)

| Policy | Mean | Median | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| Selected TD(1), 12M new | 24,926.47 | 26,780 | 55.47% | 6.25% | 0 |
| Selected TD(8), 4M new | 18,163.50 | 16,006 | 32.81% | 0% | 0 |
| Frozen CNN PPO | 49,336.41 | 44,566 | 88.28% | 46.88% | 2 games |
| TD(1) origin | 14,337.34 | 14,640 | 14.06% | 0% | 0 |
| TD(8) origin | 10,698.09 | 11,692 | 3.12% | 0% | 0 |

Reported paired 95% game-only intervals: TD1−own origin +10,589.1 [8,027.4, 13,077.5]; TD8−own origin +7,465.4 [5,727.3, 9,220.4]; TD1−PPO −24,409.9 [−29,029.0, −19,933.2]; TD8−PPO −31,172.9 [−35,073.6, −27,326.1]; TD8−TD1 −6,763.0 [−9,447.9, −4,020.2]. Selected TD(1) is the strongest observed TD checkpoint but not a safe default for more training. TD(8) avoided this guard stop and scored lower; different inherited histories and one seed do not establish general stability or causal superiority.

**Audit and sources:** `bc_initialized/REPORT.md`, `README.md` recovery section, `config.json`, `initialization_decision.json`, `EXPERIMENTS.jsonl`, `results/{aggregate.json,curves.csv,paired_comparisons.json,final_audit.json,fallback_sources.json}`, `historical/*_final.json`, `runs/*/origin.json`, and `bc_repair_01/REPORT.md` / `decision_result.md`. Postprocessing originally globbed trajectory pickles alongside checkpoints; the preserved failure and CPU-only recovery are in `finalization_recovery/REPORT.md`. Recovered audit passed, not a gameplay rerun. [Checkpoint identities](../references/artifact_index.md). No Double Q or joint BC-plus-value learning established by this packet.
