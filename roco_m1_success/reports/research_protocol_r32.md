# R32 protocol: official reset during reducer pickup

## Question

Why did R31's reducer fall during the full-stack pickup even though the
reducer-only run retained the grasp?  The official policy combines reducer
pickup actions with a reset of the other (ring) arm when the two arms differ;
the compact controller omitted this transition.

## Intervention

During the existing reducer pickup interval, concatenate the right-arm
pickup/grasp action with the left arm's initial joint target.  Do not change
the reducer target, release trajectory, ring state, physics, or evaluator.

## Prediction

The reducer should remain attached through lift/transport and reach the
center-target corridor.  The ring score behavior should match R31 because the
ring release code is unchanged.  A positive reducer result is a valid contact
and final relation, not merely a transient score.

## Measurement

Record reducer pose and gripper values at grasp/lift/transport/align/release,
score transitions, and post-settle official/socket-aware scores.  Primary
success still requires stable official 6/6; any ring-only loss is tracked as a
separate hypothesis.
