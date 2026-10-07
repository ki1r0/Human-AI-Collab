# H1 result: newest-only temporal aggregation

Date: 2026-09-09

H1 used the same ACT checkpoint, paired statistics, seed 23, task, physics
settings, and 590-step budget as the baseline. It changed only the existing
temporal aggregation parameter from 0.1 to 1000.0. At this value the
float32 exponential weights effectively retain only the newest prediction.

Output directory:

roco_single_run/runs/secondgear_newest_seed23_20260909T092724Z/

The run completed all 590 live ACT inferences and actions and exited cleanly.

- Initial, best, and final score: 0 / 0 / 0
- Termination: max_steps
- Task success: false
- Checkpoint: yjsm1203/roco_model_act_2 at revision
  52344a203e0739638cb2c7b11ea632e7b2eb2608
- Checkpoint SHA-256:
  a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1
- Stats SHA-256:
  4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e
- 590 finite, nonzero policy action rows
- 590 unique input hashes per camera
- Video: H.264/yuv420p, 591 frames, 960x270, 20 fps
- Container FFmpeg decoded the video without errors

GPU2 was occupied by another user's training process, so this run used GPU0
after checking that only desktop graphics processes were using it. No other
user process was stopped or changed. The shared checkout preparation script
was skipped and no source file was changed by this run.

The object trace shows that the first pickup did not occur:

- Gear 1 maximum displacement: 0.01956 m
- Gear 2 maximum displacement: 0.01950 m
- Gear 2 had no positive Z lift
- Left gripper action stayed between 0.03747 m and 0.03848 m

The baseline with temporal decay 0.1 moved gear 2 by 0.20117 m, raised it by
0.18067 m, and reached score 1 at step 79. H1 therefore removed the baseline
first pickup and did not reach the second-object phase. The hypothesis that
newest-only aggregation would recover the second pickup is rejected for this
seed and rollout. No further H1 run was performed.
