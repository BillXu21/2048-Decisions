# Decision Note: Gate TD initialization and continuation on gameplay and calibration

Status: implemented

## Problem

Afterstate TD avoids PPO's actor-critic optimization burden, but afterstate returns, teacher rankings and bootstrapped loss are different quantities. Early gains did not show whether the resulting values were calibrated or stable.

## Decision

Use the next-reward convention, legal targets, terminal masks and saved full-state restart packets. Treat selected PPO as the strength reference; keep slow one-step and eight-step as experimental TD alternatives. Inspect complete own-policy returns and gameplay across milestones; refuse the tested BC head fits when gates fail, and stop arms that trip verified degradation/divergence conditions. Record scratch fallback explicitly and never quietly continue a stopped arm.

## Consequences

The early fast-refresh recommendation is superseded by longer three-seed evidence and a divergence stop. Later TD(1) with slow refresh provided the best confirmed TD checkpoint but also stopped for excessive positive bias. TD(8) completed its cap but did not recover its early score peak. Stable long-run TD remains unresolved; target lag is not Double Q, and joint BC/value initialization is proposed. Historical rationale owner: bounded-afterstate-TD note and append-only results in the local TD packet.

## Alternatives considered

- Train from teacher MC fits: worse initial play and severe own-policy optimism did not support adoption.
- Start from raw BC, or from head-only calibration: neither ranking nor absolute/terminal calibration gates were jointly met.
- Copy more rapidly: initially helped short runs, but later inflated predictions and caused a guarded stop.
- Declare eight-step best at 262k: crossed intervals included zero and later learning reversed that mean ranking.

## Results

- [TD pilot](../../../../evidence/td_initial_pilot.md): conventional scalar tuples were faster at the tested budget; CNN TD improved but did not approach PPO.
- [Scratch follow-ups](../../../../evidence/td_followups.md): fast refresh flattened; one replica diverged in the matched-time packet.
- [Long-horizon evidence](../../../../evidence/td_long_horizon.md): both calibration attempts rejected; best TD(1) mean 24,926.47 on 128 held-out games, with bias stop. No TD replacement for PPO established.
