# Official RoCo ACT deployment audit — 2026-09-21

This audit was opened after the temporal-horizon experiments.  It deliberately
does not treat episode length as a model correction.  The reference is the
official `gearboxAssembly` checkout at `094a1f76d18c207caec198315f23b1a60dbca94f`.

## What the official files actually do

The documented launcher is `scripts/VLA_agent.py`.  It composes qpos in
`[left arm(6), left gripper, right arm(6), right gripper]` order, stacks RGB in
`[head, left hand, right hand]` order, calls `ACTPolicyWrapper.predict`, then
reorders the returned action to `[left arm, right arm, left gripper, right
gripper]` before one `env.step` (lines 83–113).

The official ACT training/evaluation path is different in two important ways:

* `act/utils.py:68–74` converts RGB to float and divides by 255, and
  standardizes qpos with `qpos_mean/qpos_std`.
* `act/imitate_episodes.py:141–148,177–178,240–269` repeats those transforms at
  inference, denormalizes actions with the paired action statistics, and uses
  the upstream ACT temporal aggregation (`k=0.01`, oldest prediction first).

The RoCo `ACTPolicyWrapper` only loads action statistics and applies action
denormalization (`policy_wrapper.py:165–178,210–223`).  It does not standardize
qpos.  `VLA_agent.py:90–96` casts the raw camera tensors to float but does not
divide by 255.  The live agent environment returns the camera tensors directly
from Isaac camera output (`galaxea_lab_agent_env.py:188–217`); the measured
contract is `uint8`, `(1,240,320,3)`.

Therefore the public RoCo launcher and the ACT training contract are not
numerically self-consistent.  This is not an inference: a static test on the
saved live observation produced:

* raw official VLA input: `TypeError: Input tensor should be a float tensor`;
* RGB `/255` but raw qpos: finite output, but qpos remains unstandardized;
* RGB `/255` plus qpos standardization: finite output matching the current
  training-faithful adapter.

The local `RocoActPolicy` follows the latter path: raw uint8 RGB is converted
to float `/255` exactly once, qpos is standardized with the checkpoint-paired
statistics, action chunks are aggregated with the RoCo wrapper's `0.1`
newest-first rule, and actions are denormalized and reordered once.  This is a
preprocessing correction supported by the official ACT data code, not a time
or score adjustment.

## Environment audit

The official HEAD selects `GALAXEA_R1_LITE_BUNDLE` in
`robots/robot_bundles.py`, while the same checkout's README describes the
original R1 as the default and the available ACT model card only says
“gearbox assembly” (it does not publish the embodiment).  The previous R1
rollouts therefore cannot be called an exact clean HEAD reproduction: they
used the R1 compatibility preparation because the public demonstration/data
evidence was R1-era.  To separate this from policy behavior, an isolated
worktree was prepared from the official commit with only the required public
PhysX/texture compatibility fixes, retaining the HEAD R1_Lite selection.

The isolated R1_Lite launch exposed repeated official USD/PhysX joint-limit
errors before policy stepping.  This is recorded separately; it is not folded
into the R1 ACT result and does not justify changing the episode horizon.

The run nevertheless completed its fixed 590 control steps with the same
checkpoint and training-faithful adapter.  It scored `0/6` throughout, had no
score transition, and its normalized qpos range reached `8.5846`; the initial
R1_Lite pose contains several joint values far from the checkpoint statistics.
The corresponding R1 seed-23 run starts in the expected R1 pose and stays
within approximately `[-2.78, 2.20]`.  This is direct evidence that switching
the bundle to R1 Lite would be an embodiment/statistics mismatch, not a fix for
the second-gear failure.  Artifact:
`roco_single_run/runs/official_r1lite_seed23_20260921T060917Z/`.

The official agent environment also contains a termination defect at
`galaxea_lab_agent_env.py:330–335`: it compares the tuple returned by
`evaluate_score()` with integer `6`, so native success termination is not the
reliable success signal.  The runner therefore reads the official score
explicitly after each step and reports native timeout/reset independently.

## Decision

No further time-horizon or temporal-decay tuning is authorized by this audit.
The supported learned path is the training-faithful preprocessing adapter with
the official action reorder and the fixed 590-step data/rollout horizon.  The
remaining full-task failure must be attributed to checkpoint/embodiment/data
compatibility only after the isolated R1_Lite run is recorded and compared;
it must not be “fixed” by extending completion time.
