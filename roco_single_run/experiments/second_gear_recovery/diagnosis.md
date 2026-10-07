# ACT seed-23 second-gear failure diagnosis

This diagnosis uses the existing full rollout only.  It does not alter the
checkpoint, runner, simulator, action mapping, or score function.  Reproduce
the numbers with:

```bash
python roco_single_run/experiments/second_gear_recovery/analyze_trace.py \
  roco_single_run/runs/learned_full_act_seed23_20260909T082411Z/trace.npz \
  --json roco_single_run/experiments/second_gear_recovery/seed23_diagnosis.json
```

The trace contains one post-action object pose for each of 590 policy steps,
the denormalized policy action in `[L arm 6, L gripper, R arm 6, R gripper]`
order, and the measured qpos before and after each action.

## What was actually picked

The first successful lift is `sun_planetary_gear_2` (the fifth object in the
sorted pose vector).  Its center rises from `z=0.9200 m` to a peak of about
`z=1.1007 m` at step 61, then settles near the carrier at
`(0.5608, 0.0261, 0.9132) m`.  The official score changes `0 -> 1` at step
79.  This is the only gear with a large positive vertical excursion.

The later target with the largest non-first motion is
`sun_planetary_gear_1`.  Its anomalous event is concentrated around steps
320–335: it moves from roughly `(0.701, 0.101, 0.9027) m`, reaches a peak
near `(0.682, 0.077, 0.9175) m`, rotates by about `0.96 rad`, and returns to
`z=0.9023 m` by step 336.  At the end it is still off the original XY point,
but it was never carried to a carrier pin.  Gears 3 and 4 show only the common
initial settling and small later drift; neither has a comparable lift.

This identifies the second event from object state, rather than naming it from
a video impression: it is the gear-1 contact/pick attempt, not a successful
second installation.

## What the actuators did during that event

Around steps 320–335 the left arm is the active arm.  Its gripper target
contracts from about `0.0343 m` to `0.0167–0.0183 m` while the measured left
gripper follows it.  The target gear moves at the same time, but its center
does not follow a lifted trajectory: it rises only about `15.6 mm`, rotates,
then falls back to the table.  This is a collision or partial pinch, not a
stable grasp.

The right arm does not initiate a competing second pick.  After the first
score transition, its gripper target stays around `0.0385 m` and measured
qpos around `0.0370 m` through step 300; that is near the open reset value
`0.0400 m`.  In the full trace its policy-action standard deviation is only
about `0.00057 m`, and the right-arm joint standard deviations are
`0.0024–0.0124 rad` by channel.  The right wrist video panel remains visually
unchanged.  Thus “the second gear cannot be picked” is not caused by a right
gripper closing and slipping: the right-side pick sequence is never emitted.

## Causal attribution

The strongest supported cause is checkpoint behavior.  `roco_model_act_2`
produces one left-arm gear placement, then enters a left-arm approach/contact
pattern for the next object while leaving the right arm close to reset.  The
second object is contacted but not lifted, and the policy has no recovery or
phase transition after that failed contact.  The episode therefore continues
with score 1 until the 590-step limit.

The trace does not support these explanations:

* A dropped second gear after a completed right-arm grasp: the right gripper
  never closes and no right-arm transport motion occurs.
* A stale camera or dead simulator: all camera hashes are unique, all actions
  and object states are finite, and the first object is genuinely moved and
  scored.
* An action-order failure introduced by the local runner: the first left-arm
  pick reaches the expected object, and the environment reorder is exercised
  consistently.  A separate action-interface probe already validates the
  channel mapping.

The remaining ambiguity is whether gear 1 is the policy's intended second
target or a side effect of an already-diverging policy.  Object-state evidence
still establishes the mechanical failure window: the only post-first-score
gear contact with a lift-like transient is gear 1 at steps 320–335.

## Falsifiable next experiment

Run the same checkpoint and seed with a trace extension that records end
effector poses and raw ACT chunks, then compare the model output against the
unchanged rollout around steps 300–340.  The diagnosis is supported if the
same conditions recur: left-gripper target closes below `0.02 m`, gear 1 rises
less than `0.02 m` and returns to table height within 20 steps, while right
gripper target remains above `0.038 m`.  If instead a fresh run produces a
right-gripper close command or a carried gear, the failure is seed/layout
sensitive and should not be attributed to a deterministic phase bug.

The minimal model-level remedy is a capable checkpoint or new training data
covering the full two-arm sequence.  Injecting an oracle action, mirroring the
right arm, teleporting the gear, or changing the scorer would no longer be a
faithful ACT rollout.
