# Decision Note: Learn candidate-afterstate policies and evaluate full D4

Status: implemented

## Problem

The snake heuristic was useful as a teacher, but learned search leaves did not confirm an advantage under matched time. Board-to-four-action imitation also trailed the scratch PPO baseline despite strong demonstrations. The initial data-shift explanation preceded the architecture and label audits.

## Decision

Use the search teacher for complete game demonstrations, not in the learned deployment path. Score legal candidate afterstates with shared weights; train on hard actions where no teacher score vectors exist and within-state soft preferences where available. Select by development gameplay, confirm on untouched paired games, and make full D4 scoring part of the deployed policy. In V3 it is also part of the differentiated scorer. Keep the single view only as a diagnostic, not an equivalent runtime shortcut.

## Consequences

Retained: afterstate comparison, episode-level splits, D4 and gameplay-based checkpointing. Preference values are logits, not calibrated returns. Paired policies should share seed blocks, a runtime/compile regime, transforms with mapped actions, tie-break and legality. Cross-compile roundoff at close logits can change whole games. Historical owners: BC imitation, BC architecture ablation, overnight capacity/supervision, and scratch D4 BC implementation notes in the implementation workspace.

## Alternatives considered

- Keep search and learn the leaf: ranking improved but the equal-time confirmation did not win.
- Train the original four-action MLP longer, or change only the teacher: controlled architecture/width changes had clearer gains; the stronger-label factorial and bounded on-policy relabelling did not settle a benefit.
- Use the latest checkpoint or lowest held-out CE: later epochs did not reliably improve complete gameplay.

## Results

- [Search and MLP evidence](../../../../evidence/mlp_bc_ppo.md): afterstate scoring and capacity improved play; early TTA discrepancies motivated in-process pairing.
- [V3 BC](../../../../evidence/ntuple_v3.md): CNN e8 and hybrid e3 outranked their later checkpoints. [PPO confirmation](../../../../evidence/cnn_ppo_10m.md) made the inference dependence even more pronounced.
