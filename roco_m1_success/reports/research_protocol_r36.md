# R36/R37 protocol: stronger ring feedback after arm reset

R30 tested gain 1.6 before the official inter-phase arm resets and did not
move the ring out of its unreachable IK basin.  R31 restored the resets and
reached a valid ring insertion (score 5), but release remained sensitive.
This follow-up keeps the R31 reset schedule and changes only the measured ring
XY feedback from gain 0.8/limit 40 mm to gain 1.6/limit 80 mm.  A ring-only
sanity run is required first; a full run follows only if it remains stable.
The prediction is that the reset puts the arm in a reachable basin and the
larger correction reduces full-stack release displacement.  Official scoring,
object poses, collision, friction, and schedule durations are unchanged.
