# Experiment index

Ordering follows research dependencies, not filesystem timestamps. All listed directories were found. `ntuple_hybrid/` is V1; V2 is `ntuple_fusion_v2/`, not a subfolder of V1.

| Phase / workspace packet | Historic outcome | Read |
|---|---|---|
| `2048_snake_baseline/` and `jax_experiments/` initial critic work | Search teacher retained; learned search-leaf win did not confirm; JAX port exposed action ranking and CPU budget limits | [Initial agent](evidence/initial_agent.md) |
| `jax_experiments/ppo_gpu/` | Scratch masked MLP PPO finished 52.43M transitions; far below teacher | [MLP BC and PPO](evidence/mlp_bc_ppo.md) |
| `jax_experiments/bc_imitation/`, `overnight_afterstate/` | Afterstate architecture and then capacity helped; harder labels and bounded self-learning did not resolve gains | [MLP BC and PPO](evidence/mlp_bc_ppo.md) |
| `jax_experiments/afterstate_ppo/`, `afterstate_ppo_continuation/` | Plain R3 improved; 45M selected after 50M training, not a 50M confirmation; 100M not run | [MLP BC and PPO](evidence/mlp_bc_ppo.md) |
| `jax_experiments/ntuple_hybrid/`, `ntuple_fusion_v2/` | Frozen V1 and joint V2 experiments did not improve full-D4 play; unplanned V1 overrun preserved | [V1 and V2](evidence/ntuple_v1_v2.md) |
| `jax_experiments/ntuple_hybrid_v3/` | Two 30-epoch scratch BC arms; earlier epochs selected; CNN won | [V3 BC](evidence/ntuple_v3.md) |
| `jax_experiments/ntuple_hybrid_v3/ppo_pilot_10m/` | Both completed exact cap; selected CNN 10M, hybrid 7.5M; fresh 1,024-game confirmation | [CNN PPO](evidence/cnn_ppo_10m.md) |
| `jax_experiments/ntuple_hybrid_v3/ppo_benchmarks/` | Fixed-batch timing, short CNN forks, separate optimization audit; optional frozen rollout timings did not run | [PPO optimization](evidence/ppo_benchmarks.md) |
| `jax_experiments/afterstate_td/` | One-step pilot, scratch and teacher-return warm starts for CNN and scalar lookup tables | [TD pilot](evidence/td_initial_pilot.md) |
| `jax_experiments/afterstate_td/followup_01/`, `followup_02/`, `followup_03/` | 3×3 scratch continuations; early refresh advantage reversed; one divergence stop in matched-time packet | [TD follow-ups](evidence/td_followups.md) |
| `jax_experiments/afterstate_td/bc_initialized/`, `bc_repair_01/` | BC calibration and separate head-only repair rejected; scratch fallback 26,999,808 new transitions, one arm stopped, final held-out complete | [Long-horizon TD](evidence/td_long_horizon.md) |

## Evidence labels

- **Recomputed:** arithmetic from existing per-game records or measured checkpoint bytes, with no new inference. See [reproduction](references/reproduction.md) for exact scope.
- **Reported:** historical measurements with a traceable local report, log, config, or summary. A source audit is not a new independent run.
- **Exploratory:** screens, checkpoint selection, partial records, diagnostic replays, short slopes, and single- or few-seed comparisons. Intervals over game seeds are not intervals over training seeds.
- **Hypothesis / proposed:** plausible explanations or possible new tests, not results. “Unverified” means local evidence is missing, inconsistent, or not sufficient to establish the claim.
