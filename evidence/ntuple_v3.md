# V3: scratch CNN and CNN-plus-neural-tuple cloning

V3 made a clean break from retrofitting a frozen MLP. It trained two scratch actors with 18-plane input, conv64 and three two-convolution residual spatial blocks; flatten to 256, append branch features and move metadata, then 512 GELU, residual 512, 256 GELU and scalar. The **CNN control includes a conventional dense branch** (18→1792→1792→128), not just convolutions, and has 4,826,049 actor parameters. The hybrid substitutes four `18^4 × 8` tables, 32 ordered type/occurrence tokens and a position/channel/token MLP mixer to 128 features; 4,746,569 actor parameters. These are not additive scalar N-tuples. The comparison tests entire architectures, not isolated learned table or token-pooling effects.

Same seed 20261026, episode splits, example order, Adam 3e−4, clip 1, batch 256 and fully differentiable eight-way scorer average; 30 actual epochs per arm on an original 80-epoch cosine schedule. Hard labels from the 320,667 base corpus (200/24 episode split); 61,042 weak-pool rows (112/16 episodes) use T=32 within-state soft targets. 335,778 train / 45,931 validation rows, no score-to-go regression. One seed per arm. Deeper historical MLP comparison differs in data/schedule.

| Arm | Development checkpoint, mean (256 games) | Later checkpoints, development mean | Fresh D4 mean | Fresh single mean | D4 reach 2048 / 4096 |
|---|---|---|---:|---:|---|
| Historical 512×512 BC | epoch 80, 18,809.5 | — | 19,014.7 | 15,613.2 | 33.59% / 1.37% |
| CNN | **epoch 8, 26,207.4** | e13 22,012; e20 21,396; e27 21,434; e30 20,895 | 24,493.7 | 8,761.1 | 54.88% / 3.03% |
| Hybrid | **epoch 3, 21,068.4** | e20 15,282; e30 13,692 | 20,144.2 | 8,608.8 | 37.79% / 1.17% |

Development seeds 3500000–3500255; 1,024 fresh paired games 3600000–3601023, none reaching 8192. Reported paired game-bootstrap intervals: hybrid minus CNN −4,349.5 [−5,194.1, −3,524.2]; CNN minus historical BC +5,479.1 [4,632.2, 6,321.1]. The hybrid tables changed on visited rows (~16–20% coverage), but this is numerical evidence of learning, not gameplay utility. More BC epochs degraded both actors, especially the hybrid; keep gameplay-selected checkpoints. No pooled-token control or multi-seed architecture result. No PPO transitions in this BC packet; both selected actors were later tested in the [separate 10M study](cnn_ppo_10m.md).

**Local evidence:** `jax_experiments/ntuple_hybrid_v3/REPORT_BC.md`, `results/{development.json,final.json,final_audit.json,benchmark.json}`, `DECISIONS.md`, `runs/{cnn,hybrid}/SELECTED.pkl`. Final arithmetic and selected weights were checked against artifacts for this publication; historical intervals were taken from the saved results. [Artifact hashes](../references/artifact_index.md).
