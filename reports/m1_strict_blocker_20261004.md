# M1 strict blocker — 2026-10-04

## What is established

- The mainline is a **single Franka arm with a two-finger gripper**. Dual-arm
  runs were diagnostics/ablations, not the default M1 policy.
- The M0 REPAIR placement artifact is reproducible and passes its six
  placement checks, including the same-worker blocked → help → continue
  episode. It is explicitly `feedback_mode=privileged_debug` and does not
  prove 6-DoF seating, bolt alignment, or pick-and-carry.
- A full-gravity dual-arm synchronized lift is physically real: the dynamic
  Hub follows the wrists by about `0.138 m`, all four gripper links report
  contact, and no pose write or fixed grasp constraint is used.

## Repeated strict failures

The current transport controller cannot preserve a stable assembly pose once
the dynamic Hub leaves the source support. The completed ablations include
both arms through transport, release-and-withdraw of the right arm, release
in place, direct transport, orthogonal transport, high friction, and the
historical single-arm grasp correction. Every full pick-and-place route
remained below the physical `4/4` gate (all scored `2/4`). Typical failures
were multi-kN gripper/casing impulses, `35–221 mm` radial/axial errors,
`22–68°` orientation errors, or the Hub returning to its source support.

The most informative positive partial result is
`validation_logs/m1_dual_true_gravity_sync_lift_only_20261004/metrics.json`:
it proves load-bearing lift only. The most informative full negative is
`validation_logs/m1_dual_true_gravity_sync_fast_place_20261004/metrics.json`.
The complete dual run that was kept through strict telemetry exceeded the
40-minute limit and is recorded as an execution timeout, not a success.

## Why this is a blocker

This is no longer a missing flag or a missing model checkpoint. The next
required change is a new transport controller/physical handoff design that
keeps the one-inner/one-outer contact wrench bounded while moving the dynamic
Hub through the casing approach. It must be validated from contact and pose
logs before any ACT/REPAIR policy claim. Increasing friction or changing the
release timing alone did not solve it.

No USD geometry was changed by these runs. The strict M1 goal therefore
remains **not complete**; existing M0 and lift artifacts must not be promoted
to M1 success.
