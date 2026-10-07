# R34 protocol: ring passive-seating counter-bias

## Question

Can the one-sided ring release displacement be kept inside the official
center-to-ring relation without changing physics or the score definition?
R31's insert snapshot was about 0.45 mm from the center gear; release moved
the ring relative to the center by approximately (+3.9, -3.8) mm, just beyond
the 5 mm XY criterion.

## Intervention

During ring align/insert, add a fixed (-2, +2) mm XY bias to the existing
measured carrier feedback target.  The bias is applied only to the robot's IK
target; it does not teleport an object or alter friction/collision parameters.

## Prediction

The release-side passive displacement should be partially cancelled, so both
carrier-ring and center-ring XY relations remain below 5 mm.  Ring transport
and grasp should remain unchanged.  If the ring-only control phase regresses,
reject this hypothesis before attempting a full episode.

## Measurement

Compare insert/release/settle relations and official score transitions, then
run a full episode only if the ring phase remains physically valid.  Report
official and socket-aware scores separately.
