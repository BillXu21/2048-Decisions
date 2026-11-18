# Decision Note: Keep the confirmed full-D4 CNN as PPO reference

Status: implemented

## Problem

The original MLP BC policy was strong relative to its predecessors but needed self-generated improvement. Then V3 produced stronger BC starts with different architectures. An apples-to-apples control was needed; wider models made old minibatches impractical.

## Decision

Historical afterstate PPO used γ=1, GAE λ=.98, the deployed D4 policy and an independent pre-action critic with detached actor features and truncated-50 warm returns. Keep no reference KL for the reference protocol, and keep the confirmed V3 CNN 10M actor as the project benchmark. The V3 run maintained 32×128, 3 epochs/256 minibatch and exact 10M per arm; keep those as a comparison control, not a proven optimal recipe. Selection uses D4 development mean, never held-out performance, and need not use the latest weights.

## Consequences

Retained benchmark: CNN at 10M, not hybrid at 7.5M or MLP 45M. Preserve both selected checkpoints and resumable state. Existing training budgets are closed. Historical successor to the MLP continuation as current strength reference; it does not invalidate the MLP's distinct results or prove asymptotic superiority. Original owners: `afterstate_ppo`, `afterstate_ppo_continuation`, `ppo_pilot_10m` verified notes.

## Alternatives considered

- Anchor to the frozen BC reference: coefficient-1 KL capped learning in the tested MLP regime; not a verdict on all regularizers.
- Warm a critic on full returns: unsuitable fit for short-horizon GAE in the early study.
- Evaluate 50M or latest only: 45M was the MLP development winner, and V3 hybrid selected 7.5M.
- Only run V3 CNN: both arms were needed to test whether hybrid closed its BC gap under self-generated learning.

## Results

- [MLP history](../../../../evidence/mlp_bc_ppo.md) corrects the early mislabelled 50M confirmation. The conditional 100M continuation did not run.
- [V3 PPO evidence](../../../../evidence/cnn_ppo_10m.md): CNN led the 1,024-game paired mean, but historical MLP had more rare 8192 games. Single orientation is not interchangeable.
