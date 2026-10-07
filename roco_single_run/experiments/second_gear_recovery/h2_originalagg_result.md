# H2 result: original-style aggregation

Date: 2026-09-21

H2 tested the original ACT-style oldest-first temporal aggregation hypothesis.
It used the same checkpoint, paired statistics, seed, task, physics settings,
and 590-step budget as the baseline and H1. Only the aggregation parameter was
changed to `temporal_decay=-0.01`. Under the existing implementation this
makes the oldest retained prediction receive the larger weight, matching the
original ACT evaluation ordering mathematically.

## Execution

GPU2 was free and was used for the run. The shared checkout preparation script
was skipped; no source or scene files were changed by this run.

Output directory:
`roco_single_run/runs/secondgear_originalagg_seed23_20260921T040004Z/`

The exact invocation was:

```bash
docker run --rm --entrypoint /isaac-sim/python.sh --gpus device=2 --ipc=host --network=host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/home/sunsiliang/roco_runtime/torch_cache -e PYTHONPATH=/home/sunsiliang/Human-AI-Collab/roco_single_run:/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT:/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External -v /home/sunsiliang/Human-AI-Collab:/home/sunsiliang/Human-AI-Collab -v /home/sunsiliang/roco_runtime:/home/sunsiliang/roco_runtime nvcr.io/nvidia/isaac-lab:2.3.0 /home/sunsiliang/Human-AI-Collab/roco_single_run/scripts/run_roco_single_episode.py --headless --enable_cameras --checkpoint /home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/policy_best.ckpt --stats /home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/dataset_stats.pkl --expected-checkpoint-sha256 a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1 --expected-stats-sha256 4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e --candidate-id yjsm1203/roco_model_act_2 --candidate-revision 52344a203e0739638cb2c7b11ea632e7b2eb2608 --seed-sequence 23,17,42,2026 --integration-git-commit c079c66f740b37d8832c34c8822f900c81b7f430 --seed 23 --output-dir /home/sunsiliang/Human-AI-Collab/roco_single_run/runs/secondgear_originalagg_seed23_20260921T040004Z --max-steps 590 --temporal-decay=-0.01 2>&1 | tee /home/sunsiliang/Human-AI-Collab/roco_single_run/runs/secondgear_originalagg_seed23_20260921T040004Z/run.log
```

## Result

The run completed all 590 live ACT inferences and actions and exited cleanly.

- Checkpoint: `yjsm1203/roco_model_act_2@52344a203e0739638cb2c7b11ea632e7b2eb2608`
- Checkpoint SHA-256:
  `a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1`
- Stats SHA-256:
  `4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e`
- `temporal_decay`: `-0.01`
- Initial/best/final score: `0/0/0`
- Score transitions: none
- Termination: `max_steps`
- `task_success`: `false`
- `policy_timestep`: `590`
- All arrays finite; 590 nonzero action rows; 590 unique hashes per camera
- Video: H.264/yuv420p, 591 frames, 960x270, 20 fps
- Container FFmpeg decoded the video with exit code 0 and no errors

## Aggregation weight check

The trace records both ends of the active aggregation weights. After the first
step, H2 had oldest weight greater than newest weight on every sampled step:

- Step 1: oldest = 1.000000, newest = 1.000000
- Step 2: oldest = 0.502500, newest = 0.497500
- Final 100-action window: oldest = 0.01574093, newest = 0.00584896

This confirms that H2 exercised the intended oldest-first weighting direction.
The normalized weights were finite.

## Object-level comparison

The baseline
`learned_full_act_seed23_20260909T082411Z` with `temporal_decay=0.1`
moved gear 2 by up to 0.20117 m and raised its Z position by up to
+0.18067 m at step 61; its score changed from 0 to 1 at step 79.

H2 did not reproduce that pickup:

- Gear 1 maximum displacement: 0.01871 m
- Gear 2 maximum displacement: 0.02100 m
- Gear 2 maximum `Δz`: -0.01060 m (there was no positive Z lift)
- Gear 2 had no upward lift and no scored transition
- Left gripper policy action range: 0.02553 m to 0.04016 m

H2 therefore did not reach a second-object pickup. It also lost the first
pickup that occurred in the baseline. The hypothesis that oldest-first
aggregation would preserve or improve the second-object lift is unsupported by
this complete seed-23 rollout.

## Artifacts

- `result.json`: result and integrity checks
- `trace.npz`: qpos/actions, aggregation weights, camera hashes, scores, and
  object poses
- `episode.mp4`: complete reset-through-final video
- `run.log`: full console output
- Convenience symlinks: `video.mp4`, `action_trace.npz`, `metrics.json`,
  `run_manifest.json`, and `console.log`

No additional H2 run was performed.
