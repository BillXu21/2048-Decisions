# Artifact index

Paths below are relative to the authorized implementation workspace, not downloads or links into this repository. All listed SHA-256 values were recomputed from the checkpoint file bytes during consolidation. Hashes identify exact saved payloads, not necessarily weights-only files; some include optimizer, replay or RNG state. Do not deserialize untrusted pickle files.

| Role | Checkpoint path | SHA-256 |
|---|---|---|
| **Best PPO: CNN 10M** | `jax_experiments/ntuple_hybrid_v3/ppo_pilot_10m/runs/cnn/SELECTED.pkl` | `fe6a4daf70b062682b5c94003a34b11a009645d094495c75c7f7e1a44a76b031` |
| V3 hybrid PPO selected 7.5M | `jax_experiments/ntuple_hybrid_v3/ppo_pilot_10m/runs/hybrid/SELECTED.pkl` | `0c694e65ff4f52739965bc0b2d1e538a0d2d606e27755a09b56fcbd85ff55610` |
| V3 CNN BC e8 | `jax_experiments/ntuple_hybrid_v3/runs/cnn/SELECTED.pkl` | `8bea40d5b89311f3528f1e208223bebf12ede7bab81f3ab7e8e5433f23014a06` |
| V3 hybrid BC e3 | `jax_experiments/ntuple_hybrid_v3/runs/hybrid/SELECTED.pkl` | `8464d6373c0f8974c5a35849d1a0da2f9b593fe2a02adbc51787ac9a31ee8961` |
| Historical R3 MLP PPO 45,023,232 (preserved copy) | `jax_experiments/ntuple_hybrid/preserved/r3_45M.pkl` | `bbfb35dfe45741c3a4cce24ebac0a4edd5f040b69a0c62a9b7ca90a87541e276` |
| Historical R3 MLP PPO 25,034,752 (preserved copy) | `jax_experiments/afterstate_ppo_continuation/preserved/r3_25M.pkl` | `f51872b38b3c2bdf7651839e0ac0728050e259c799ea0dd1a51d3bd02ae4fe29` |
| V1 plain continuation control 50,003,968 | `jax_experiments/ntuple_hybrid/runs/control_5m/checkpoints/step_0050003968.pkl` | `4b47463726837148f75c7f7514b161d1b2bd0e1b911e2f720b179e596fadfdbe` |
| MLP PPO BC source A, 512×512 e80 | `jax_experiments/overnight_afterstate/runs/A/SELECTED.pkl` | `55f80cc1f5bdacceb300a5b6a71d5a19d8f4f50e81b80767671aacf9c86fdab5` |
| Earlier soft T32 MLP BC, 256×256 e80 | `jax_experiments/bc_imitation/runs_arch/afterstate_soft_t32/checkpoints/checkpoint_epoch_080.pkl` | `510ec41803d92a1aaaecfd0dc3d424bb01a0b439c1fbc6ba934ce9116b960a28` |
| Scratch board-action PPO | `jax_experiments/ppo_gpu/runs/main/checkpoint_00200.pkl` | `3c3dfb7eb889c6c2eea01c6031dbef5c568f5ca1793112e8bd8033ab89135f1a` |
| **Best TD: TD(1), 12M new** | `jax_experiments/afterstate_td/bc_initialized/runs/td1/new_12000000.pkl` | `d21aa57815bbc22bc3ea30f6b322935295a41e0a696cc53974b28f590b9076cf` |
| TD(8) selected 4M new | `jax_experiments/afterstate_td/bc_initialized/runs/td8/new_04000000.pkl` | `b8a711456828ed07c634f266281c1e718c26a3d1190a5282700879b8539e115e` |
| TD(8) final cap (not selected) | `jax_experiments/afterstate_td/bc_initialized/runs/td8/new_14999808.pkl` | `4953f1124214e1592f4fcbaa63e9876c8f17a5e107171316ef7ea3e75dc36266` |
| TD(1) fallback origin (preserved copy) | `jax_experiments/afterstate_td/bc_initialized/runs/td1/origin_source.pkl` | `552fafbab13e3b1e914bf22f39b2e65d09d34db1ab3238728b3df15a6c8863e8` |
| TD(8) fallback origin (preserved copy) | `jax_experiments/afterstate_td/bc_initialized/runs/td8/origin_source.pkl` | `d6b5c075eaf3d68b3a9968d509bfd892ae756f81748f806aa17266661cb76ca2` |
| TD-study BC e8 (not accepted for TD) | `jax_experiments/afterstate_td/bc_initialized/bc/SELECTED.pkl` | `9d69a5cff1c241d3b5064d6e889055361cab2eb29b2af9d01356cf54c5224277` |

## Where the rest lives

- **Exact settings and source:** each packet's `config.json` where present, otherwise named source and embedded checkpoint settings. `source_snapshots/` and source-inventory/`READY.json` hashes identify executed versions where retained. No source Git revision was available from the implementation workspace. The publication Git commit identifies this history, not the code run.
- **Learning and gameplay:** `runs/` holds milestones and `.pkl` resume state; `results/`, run-specific JSON/CSV, and TD `historical/` retain development and final rows. `EXPERIMENTS.jsonl`, `logs/`, `telemetry.jsonl`, and status files distinguish failed, partial and completed jobs. See each [packet's evidence](../EXPERIMENT_INDEX.md) for precise paths.
- **Private/large evidence:** teacher demonstrations and derived arrays remain under original research data directories, e.g. `jax_experiments/bc_imitation/data/` and `jax_experiments/afterstate_td/data/`; TD calibration data and diagnostic trajectories stay with their packets. Prior data inventories record hashes. No location implies permission to redistribute. No original content was deleted or relocated; individual histories, plots, code snapshots and whole logs were not mirrored.
