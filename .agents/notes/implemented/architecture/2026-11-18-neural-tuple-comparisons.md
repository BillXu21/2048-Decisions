# Decision Note: Do not promote the tested neural tuple hybrids

Status: implemented

## Problem

Tuple-indexed features might cheaply encode useful positional patterns. V1 asked whether a cheap frozen-actor residual helped; it could not decide whether learned tuples work in different architectures.

## Decision

Use non-tuple controls and full-D4 paired gameplay to judge complete models. Do not promote the tested V1 frozen additive branch, V2 nonlinear fusion branches, or V3 scratch hybrid. Retain scalar N-tuple TD as a distinct research comparison; it does not share the neural embedding policy design.

## Consequences

Rejected for deployment: the tested hybrid configurations, not the class of tuple representations. V2 moved to joint updates after initial frozen residual evidence (critic drift was suspected, not isolated). V3 replaced the residual retrofit with a scratch comparison controlling approximate capacity. No pooled-token or embedding freeze ablation identifies individual causal components. Historic owners: V1, V2, V3 BC and V3 PPO architecture notes in their local packets.

## Alternatives considered

- Continue plain MLP PPO: a small protocol-sensitive benefit remained possible; used as a control, not evidence for fusion.
- Add dense residual capacity without tuples: similarly harmful or worse in V1/V2.
- Promote V3 on numeric table movement or single-orientation improvement: neither established a deployed D4 gain.

## Results

- [V1/V2](../../../../evidence/ntuple_v1_v2.md): preserve the counter overrun, the stopped/skipped extensions and the positive plain control result.
- [V3 BC](../../../../evidence/ntuple_v3.md) and [PPO](../../../../evidence/cnn_ppo_10m.md): hybrid lost in deployed D4 both before and after PPO, even though it gained in PPO.
