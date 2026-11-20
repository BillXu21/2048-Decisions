# Attention and exact-pair memory: November 2026 progress

**Status:** Text-first handoff of locally agent-reported research, through 2026-11-20. Published for continuity, **not independently replayed or retrained by this repository update**. The referenced local reports, checkpoints, raw data, and executable source remain in the separate implementation workspace. One training seed and development-game evaluations do not establish a general architecture ranking.

## Research aim and comparison

Study whether a small 16-cell attention afterstate policy benefits from exact spatial-pattern memory. Three actors were held under a common full-eight-way-D4 scorer:

| Arm | Description | Parameters |
|---|---|---:|
| A | Attention-only | 30,849 |
| B | Attention with parameter-matched nonlookup branch | 34,181 |
| C | Attention with exact relative-exponent pair lookup and context gate | 34,182 |

The prior project strength reference remains the much larger V3 CNN (4,826,049 actor parameters), **50,407.04 mean on 1,024 fresh games after 10M PPO transitions**, starting from a separately selected BC origin of 24,789.10 on those games. That protocol differs from the new experiments; it is not a controlled parameter-scaling comparison.

## Packet 1A–1B: expert-data and pattern investigation

- The bounded study used 128 discovery and 32 confirmation games, with **199,944** decisions after excluding single-legal-action states. It surveyed all **120** unordered cell pairs across **21** D4 position orbits. Local audit, reproduction, and focused tests were reported as passing. Do not confuse these 160 survey games with the actual BC training corpus.
- The agent reported held-out controlled associations for equal occupied tiles in adjacent interior cells (**+1.44 percentage points**), adjacent corner/edge tiles with the edge twice the corner (**−0.60 pp**), and a staggered nonadjacent 1:2 pair across opposite edges (**+0.38 pp**); intervals and support are in the local 1B report. Among 49 locked patterns, 24 retained direction and survived the reported multiple-testing correction after simple controls.
- Adding selected nonadjacent pairs reportedly improved held-out log loss by **0.00383 nats/decision** beyond adjacent pairs; adjacent pairs were stronger at equal feature count. Relative tile-exponent normalization reportedly improved log loss by **0.00854 nats/decision** against absolute keys at matched feature count/support. Longer patterns had diminishing incremental signal and greater sparsity.
- The bounded first lookup design selected **11** discovery-selected pair orbits, **50** placements per afterstate view, and **232** exact keys. These are observational imitation-prediction results, not evidence of improved learned gameplay, nor proof of teacher independence.

Local sources: `jax_experiments/expert_dataset_audit_1a/` and `jax_experiments/spatial_pattern_survey_1b/{PATTERN_SURVEY.md,MEMORY_DESIGN_RECOMMENDATION.md,EVIDENCE_SUMMARY.md}`.

## Packets 2A–2B.2: feasibility and short PPO pilot

- Packet 2A implemented A/B/C, legal afterstate/D4 parity, exact lookup gradients, and deterministic reload. Packet 2B established exact same-executable BC/PPO resume. The clock-capped Windows/WSL training route subsequently completed under a **1,800 MHz maximum GPU clock**, with watchdog monitoring and restored clock defaults.
- A first short run stopped on a Windows/WSL heartbeat problem after 16 PPO chunks. The repaired continuation reached **61 chunks × 4,096 = 249,856 PPO transitions per arm**, **749,568 total**, from separate one-epoch BC origins. Resumption matched bit-for-bit across two fresh CUDA processes.

**One-seed results on the same 128 PPO development games:**

| Arm | Frozen one-epoch BC mean | 249,856-transition PPO mean | Paired change |
|---|---:|---:|---:|
| A | 11,242.3 | 12,274.9 | +1,032.6 |
| B | 10,752.9 | 10,529.0 | −224.0 |
| C | 11,310.9 | 11,086.0 | −224.9 |

A led that particular PPO endpoint; C was ~557 points above B, with no increase in the C-minus-B margin relative to their origins. The result does **not** establish memory benefit or lack of long-run benefit. Local report: `jax_experiments/attention_pair_overnight_2b2/runs/overnight-20261119-145742/{RESULT.md,EVIDENCE_SUMMARY.md}`.

## Packets 3A–3C: active infrastructure cleanup

- Audit found BC/PPO numerical kernels were already shared; operational duplication was in launcher, paired checkpoint publication, recovery, and evaluation scheduling.
- A new supported active path was implemented at `jax_experiments/attention_pair_active/`, leaving historical executables and artifacts intact. It provides a Windows front door, status/plan mode, common paired PPO loop, verified atomic three-arm commit boundaries, and new-run recovery from complete PPO boundaries.
- CPU recovery and safety tests and production-shape CUDA parity passed as locally reported. A/B/C PPO continuation matched uninterrupted execution and two fresh CUDA restores bit-for-bit; BC future updates matched; all 128 frozen B/chunk-16 games replayed exactly. Disposable watchdog and 1,800 MHz clock checks passed. A launcher success-exit reporting bug was corrected and later observed returning zero on a live run in Packet 4A.
- Scope limitations: new runner does not support recovery from arbitrary historical checkpoints or interrupted BC/warmup. This is engineering acceptance, not a new scientific-performance result.

Local sources: `jax_experiments/attention_pair_active/` and its Packet 3C acceptance run records.

## Packet 4A: longer BC learning curves (one training seed)

Using the validated new active path, A/B/C each completed **12 BC epochs** in **22m47s** including evaluations and cleanup, with full-D4 gameplay selection on the **same 128 BC development games** (distinct from the PPO development evaluation block). Selected frozen origins:

| Arm | Selected epoch | BC development mean | Games reaching ≥2048 /128 |
|---|---:|---:|---:|
| A | 12 | **13,473** | 11 |
| B | 12 | **14,676** | 17 |
| C | 8 | **14,732** | 24 |

At matched epoch 12, scores were A **13,473**, B **14,676**, C **13,410**. C peaked at epoch 8 then lost **1,322** mean points by epoch 12 even as its validation cross-entropy improved from **0.65783** to **0.65208**. A/e12 and B/e12 were best at the last scheduled evaluation; a plateau was **not** established. The selected C-minus-B mean margin is only **+56 points** across 128 games. One training seed and checkpoint selection on the development games preclude a reliable architecture ranking. The 12-epoch selected origins have **not** undergone PPO.

Full local source: `jax_experiments/attention_pair_active/runs/overnight-20261120-161121/{CURVES.md,RESULT.md}`. The live launcher returned exit zero, historical source/checkpoints remained unchanged, and the capped GPU peaked at **70°C**.

## Open research questions and boundaries

1. **BC data utilization:** Audit the actual unique expert decisions/games consumed by the 12-epoch BC pipeline. The public dataset publisher reports ~54 million retained moves across 9,000 filtered games, but this does not establish that local BC traversed all of them. The local survey sample is not the BC corpus. Treat the hypothesis of a small fixed training subset as unverified until manifests/data loading are checked.
2. **Capacity versus optimization/data coverage:** The A/B/C actors are only ~31–34k parameters versus ~4.8M for the historical CNN. Evaluate whether wider/deeper convolution/attention backbones (potentially hundreds of thousands to ~1M parameters), better use of expert data, or different optimization offers a better return; do not presuppose a million parameters is optimal.
3. **Memory benefit:** Exact-pair lookup beat B on some short BC checkpoints but not consistently at matched epoch. A longer PPO curve from A/e12, B/e12, C/e8 was **proposed but not run**; using separately selected epochs is valid for best-origin development, **not** an equal-BC-exposure ablation.
4. **Evaluation standards:** Keep full D4, episode-disjoint BC data, matched development games, untouched final confirmation seeds, checkpoint selection by gameplay, exact resume, and the 1,800 MHz/thermal safeguards. Separate game-seed uncertainty from training-seed uncertainty.
5. **Decision status:** No larger model architecture, next GPU budget, multi-seed expansion, or memory promotion has been accepted. The previous pattern-memory plan under `.agents/notes/draft/` is an older exploratory handoff, **not** a newly approved or implemented decision. Current skill convention places new unverified drafts in `.agents/drafts/` pending local verification.

**Next discussion:** review capacity, BC data coverage, and alternative architecture/scaling experiments before deciding whether to spend compute on the proposed ~2M PPO transitions per arm. Do not launch training from this publication.
