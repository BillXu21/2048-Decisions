# Draft decision — Can spatial pattern memory improve 2048 play after imitation?

**Status — proposed, 2026-11-18.** The current task covers research planning and draft decisions. Implementation, data processing, gameplay, and training require separate authorization. The next chat should stay in the research-planning role and prepare a local agent packet when requested.

## Objective and evidence

Start a fresh agent with a bounded amount of behavior cloning (BC) from a strong expert, then improve it through its own gameplay with reinforcement learning (RL). The research question is whether explicit pattern memory helps a small attention policy learn under a single RTX 4070 Laptop GPU budget. The historical full-D4 CNN is the strength benchmark. In the same 1,024 fresh games, its BC and PPO checkpoints averaged 24,789 and 50,407. Comparisons against each new agent’s own frozen BC policy will show what RL adds. [PPO evidence](../../../../evidence/cnn_ppo_10m.md)

V1 and V2 tested residual and fusion branches on an existing MLP. V3 trained fresh policies but replaced a dense branch with four-cell tuple embeddings and a mixer. Its selected hybrid averaged 36,090 after PPO. Those comparisons motivate holding the new backbone fixed when adding memory. Rare-table coverage, orientation effects, and the advantage of earlier BC checkpoints deserve attention. [V1/V2](../../../../evidence/ntuple_v1_v2.md) · [V3](../../../../evidence/ntuple_v3.md)

## Proposed comparison

Use a small convolutional stem and one attention block over 16 position-aware cell tokens to score each legal afterstate. Compare that backbone with and without a context-gated pattern lookup. Give the two arms the same training episodes, scorer conventions, and full eight-way rotation/reflection averaging (D4). The control tests whether the added module earns its cost. A win alone would leave the effects of extra parameters, scale sharing, and gating unresolved.

Encode occupied tiles by exponent relative to the smallest occupied tile in the pattern. Keep empties distinct. Thus [2, 4, 8] and [4, 8, 16] share a lookup key. Supply the absolute scale, placement, and board context to the model so it can distinguish their strategic meanings. Tile spawns and merge rewards make absolute scale relevant. Use a separate empty-only convention if needed.

Two memory designs remain open. DeepSeek’s Engram hashes predefined pattern lengths into learned tables and gates the retrieved vectors using context. A 2048 version could use exact indexing for small shapes. A more exploratory option mines a bounded dictionary of variable-length spatial patterns from training games, with paths allowed to turn. The latter could freeze its keys through the first PPO comparison while continuing to train their vectors and gate. Discovering new keys online is a later question. The language-model results motivate the mechanism but give no estimate of its benefit in 2048. [Engram paper](https://arxiv.org/abs/2601.07372)

## First packet to prepare — expert data and candidate patterns

1. Review the expert data card, available files, filtering, and episode splits. The publisher reports 54 million retained moves and a filtered mean score of 143,847. The viewer and card give different game counts. Verify counts, action mapping, legal afterstates, reward convention, and sample episode replay locally once authorized. Keep episodes disjoint across discovery and validation. [Dataset](https://huggingface.co/datasets/yethdev/2048-gameplay-dataset)
2. Survey all 120 unordered cell pairs on a bounded sample. There are 24 edge-adjacent and 96 other pairs. Keep displacement under D4 and distinguish corners, edges, or interior where placement matters. Begin longer candidate patterns with connected cells. A snake is one possible shape.
3. For each candidate, measure prevalence across distinct games and how often it appears in the expert-chosen afterstate versus the other legal afterstates of the **same board**, before random spawn. Weight each decision equally across its alternatives. Report positive and negative associations, support by game stage, and replication on held-out episodes. A frequent pattern present after every move supplies little evidence for action selection.
4. Use those results and estimated lookup cost to recommend the first memory design. Record what data would change that recommendation. Set explicit sample, dictionary, time, and storage limits in the packet before execution.

## Later BC and RL evaluation

Select BC checkpoints by development gameplay. Freeze a copy of each selected policy and carry both arms into a bounded afterstate PPO comparison, even if BC scores are close. Measure fresh paired game score, tile reach, gain over each arm’s own BC origin, and wall time including evaluation and thermal pauses. Report score at equal compute and shared transition milestones where available. Preserve full-D4 evaluation, common runtime and seed blocks, separate development and final games, old checkpoints, and existing thermal safeguards. Game-seed uncertainty and training-seed uncertainty are separate.

Before proposing a run, specify the training seeds, GPU-hour and transition caps, evaluation schedule, stopping conditions, and checkpoint selection rule. Model width, pattern lengths, dictionary size, and run budgets remain open. A from-random RL control, late-game restarts, and TD stabilization are possible follow-ups. The first decision is which patterns merit testing on the available hardware.
