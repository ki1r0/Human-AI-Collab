# R30 protocol: bounded ring XY correction

## Question

Why did the R29 full rule rollout stop at 2/6 after the ring phase even
though all three planet relations had scored?  R29 measured the carrier yaw
at about 0.20 rad and the ring at roughly `(carrier + 12 mm, carrier + 42
mm)` in XY during ring alignment.  The existing bounded correction (gain
0.8, limit 40 mm) was therefore a plausible saturation/under-correction
failure.

## Intervention

Change only the ring Cartesian feedback constants to gain 1.6 and limit 80 mm
in `controller.py`.  The controller still derives the target from measured
poses and never writes object poses, contact state, or the evaluator score.
All other schedule, friction, release, and socket-scoring code remains
unchanged.

## Prediction

The ring XY error at `ring.align` should decrease relative to R29 and the
official score should remain at least 3 through ring release.  A positive
result requires a complete run with final official score 6 and a one-second
post-settle minimum of 6; otherwise this hypothesis is rejected or refined
using the recorded event snapshots.

## Measurement

Record `initial/best/final` official and socket-aware scores, score transitions,
ring/carrier relation metrics at each event, and post-settle minimum scores.
Keep the official scorer unchanged; the socket-aware value is reported only as
the separately labelled physical-seating diagnostic.
