# V3 CNN versus hybrid PPO: exactly 10M per arm

One training seed, same optimizer settings and separate critics initialized identically, starting from the respective [selected BC checkpoints](ntuple_v3.md) (not shared actor weights). Actor sizes 4,826,049 CNN / 4,746,569 hybrid. The pre-action critic uses conv64 and two residual blocks; its flattened board plus eight detached score/legal features feed 512→256→1, with D4-averaged output (818,625 parameters, counted from `core.py`). Frozen own-policy truncated-50 rewards warmed the critic; warm collection and smoke moves are not counted in PPO transitions. Full eight-way D4 scoring throughout. 32×128 rollouts, 256 minibatch, 3 epochs, shorter boundary chunks; γ=1, λ=.98, reward /128, critic scale 100, Adam actor 1e−5 and critic 3e−4, grad norm clip .5, PPO ratio clip .15 and KL stop .02; no entropy or reference penalty. Interleaved 500k stages retained exact actor/critic/optimizer/environment and permutation RNG state.

| PPO transitions | CNN D4 | Hybrid D4 | CNN single | Hybrid single |
|---:|---:|---:|---:|---:|
| 0 | 24,212.5 | 20,975.3 | 8,867.7 | 8,408.8 |
| 1,000,000 | 34,076.8 | 25,650.1 | 4,835.3 | 11,410.3 |
| 2,500,000 | 40,790.8 | 30,174.8 | 4,495.8 | 10,203.9 |
| 5,000,000 | 42,906.7 | 33,737.3 | 3,665.8 | 10,884.5 |
| 7,500,000 | 44,291.3 | **36,575.3** | 4,951.2 | 12,484.5 |
| 10,000,000 | **48,854.9** | 35,737.5 | 5,925.0 | 9,160.1 |

256 dev games (4700000–4700255) at every milestone; bold selects by D4 mean. Hybrid's late drop was not resolved (10M−7.5M −837.8 [−3,393.4, +1,714.4]). This is a complete experiment, including the unselected hybrid 10M endpoint. Fresh paired confirmation (4800000–4801023, 1,024 games) uses selected weights only:

| Policy | D4 mean | D4 median | Reach 1024 / 2048 / 4096 | 8192 count | Single mean |
|---|---:|---:|---|---:|---:|
| CNN BC | 24,789.10 | 25,796 | 92.38% / 55.76% / 3.22% | 0 | 8,854.32 |
| Hybrid BC | 20,183.46 | 16,604 | 86.23% / 39.45% / 1.17% | 0 | 8,733.75 |
| **CNN PPO 10M** | **50,407.04** | **51,450** | 99.51% / 92.68% / 51.95% | **9** (0.88%) | 6,130.64 |
| Hybrid PPO 7.5M | 36,089.86 | 35,552 | 98.54% / 81.45% / 20.21% | 0 | 12,397.52 |
| Historical MLP 45,023,232 | 44,599.13 | 36,580 | 98.34% / 85.94% / 40.82% | **19** (1.86%) | 7,626.79 |

Reported paired 95% bootstrap intervals (10,000 game resamples): CNN PPO−CNN BC +25,617.9 [24,234.8, 27,004.5]; hybrid PPO−hybrid BC +15,906.4 [14,856.4, 16,976.8]; hybrid PPO−CNN PPO −14,317.2 [−15,770.9, −12,822.1]; CNN PPO−historical MLP +5,807.9 [4,020.4, 7,627.8]. Single-orientation hybrid PPO beat single CNN by +6,266.9 [5,759.4, 6,780.4] but lost under D4. CNN PPO leads on mean and 4096 reach, not on observed 8192 count versus MLP. These rare counts do not establish a reliable tail ranking. Full D4 is integral to the result, not an optional deployment speedup. One training seed and unequal starting policies preclude a training-seed or asymptotic conclusion.

**Local evidence:** `jax_experiments/ntuple_hybrid_v3/ppo_pilot_10m/REPORT.md`, `EXPERIMENTS.jsonl`, `core.py` (`CONFIG`, seed 20261024), checkpoint-embedded config, `results/{confirmation.json,selection.json,final_audit.json}` and `results/dev_*.json`. Means/medians/reaches/paired point differences were recomputed from saved arrays. [Selected paths and SHA-256](../references/artifact_index.md). For exact optimizer timing and the historical deduplication, see [optimization](ppo_benchmarks.md).
