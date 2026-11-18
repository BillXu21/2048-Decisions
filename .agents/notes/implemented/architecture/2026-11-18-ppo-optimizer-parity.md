# Decision Note: Separate PPO engineering parity from algorithm changes

Status: implemented

## Problem

Full-D4 CNN optimization dominated the measured V3 chunk. Host statistics transfers and a Python optimizer loop offered apparent speedups, but changing stop rules, order or minibatch size may change the learned policy.

## Decision

Benchmark from immutable copies. Preserve exact NumPy order, float64 epoch-prefix KL stop, state/Adam/RNG, full D4 and learning configuration in the reference run. Require evidence for both speed and semantic parity for engineering replacements, and held-out learning evidence for changes to update count or minibatches.

## Consequences

Reference unchanged. Fast FP32 scan rejected as exact equivalent; strict scan and bulk transfer experimental, approved only for the checked cases, not broad production parity. No architecture-specific epoch count is established. Historical owner: `ppo_benchmarks` isolated PPO throughput audit note.

## Alternatives considered

- Switch to one epoch or 512 minibatch mid-study: faster, but changes learning and breaks the matched study protocol.
- Reuse log probabilities as critic features: loses the offset required by raw score features.
- Treat low KL, high EV or extra in-sample surrogate gain as learning evidence: not a substitute for gameplay.

## Results

- [Benchmark evidence](../../../../evidence/ppo_benchmarks.md): strict scan improved optimizer timing; near-threshold reasons to reject fast scan persisted. The paired short learning pilots and epoch diagnostics did not justify a settings change. Horizon measurements remained incomplete.
