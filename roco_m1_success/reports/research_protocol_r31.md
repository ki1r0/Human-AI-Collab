# R31 protocol: restore official inter-phase arm resets

## Question

Does the compact M1 schedule fail at the ring because it omitted the official
policy's arm reset intervals?  The pinned official policy resets the left arm
after planet2 and the right arm after the center gear.  The local controller
previously kept both arms at their retreat configurations.

## Intervention

During the already existing one-second settle intervals, command only the
corresponding arm's recorded initial joint positions after `planet2` and
`center`.  Leave all object poses, collision parameters, release heights,
feedback values, and scoring code unchanged.  This does not add time or a
state write to the simulator objects.

## Prediction

The ring pickup/transport should enter the same IK basin as the official
sequence.  Relative to R29/R30, the ring XY error at `ring.align` should
decrease and the score should not lose the three planet points during ring
insertion.  A successful result still requires official best, final, and
one-second post-settle minimum scores all equal to 6.

## Measurement

Compare the event snapshots for arm phase, carrier yaw, ring/carrier XY and
orientation errors, score transitions, and post-settle scores.  The official
score remains primary; the separately labelled socket-aware diagnostic is
reported alongside it.
