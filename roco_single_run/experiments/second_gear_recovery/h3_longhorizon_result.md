# H3 result: extended horizon with raw chunk recording

Date: 2026-09-21

H3 restored the baseline temporal decay of 0.1 and extended the live ACT
episode budget from 590 to 1180 policy steps. It also recorded every raw
normalized ACT action chunk for diagnosis. The checkpoint, statistics, seed,
task, environment, physics, and action path were unchanged.

Output directory:

roco_single_run/runs/secondgear_longhorizon_seed23_20260921T040730Z/

The exact invocation was:

docker run --rm --entrypoint /isaac-sim/python.sh --gpus device=2 --ipc=host --network=host -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/home/sunsiliang/roco_runtime/torch_cache -e PYTHONPATH=/home/sunsiliang/Human-AI-Collab/roco_single_run:/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT:/home/sunsiliang/roco_runtime/gearboxAssembly/source/Galaxea_Lab_External -v /home/sunsiliang/Human-AI-Collab:/home/sunsiliang/Human-AI-Collab -v /home/sunsiliang/roco_runtime:/home/sunsiliang/roco_runtime nvcr.io/nvidia/isaac-lab:2.3.0 /home/sunsiliang/Human-AI-Collab/roco_single_run/scripts/run_roco_single_episode.py --headless --enable_cameras --checkpoint /home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/policy_best.ckpt --stats /home/sunsiliang/roco_runtime/checkpoints/roco_model_act_2/dataset_stats.pkl --expected-checkpoint-sha256 a2d0aa42ec1d39609637a40ac09b420ebc16335a199807ae42e2edff2bfce2b1 --expected-stats-sha256 4627d5316f8d6a29915124ea198cf16f82d82b5ea98d55ed3d0f9b5bb7da0b4e --candidate-id yjsm1203/roco_model_act_2 --candidate-revision 52344a203e0739638cb2c7b11ea632e7b2eb2608 --seed-sequence 23,17,42,2026 --integration-git-commit c079c66f740b37d8832c34c8822f900c81b7f430 --seed 23 --output-dir /home/sunsiliang/Human-AI-Collab/roco_single_run/runs/secondgear_longhorizon_seed23_20260921T040730Z --max-steps 1180 --temporal-decay 0.1 --record-raw-chunks 2>&1 | tee /home/sunsiliang/Human-AI-Collab/roco_single_run/runs/secondgear_longhorizon_seed23_20260921T040730Z/run.log

## Result

The environment terminated natively at step 684 before the requested 1180-step
budget. The run completed 684 real ACT inferences and actions and exited
cleanly.

- Initial score: 0
- Best score: 1
- Final score: 0
- Score transitions: 0 to 1 at step 79, then 1 to 0 at step 684
- Termination: native_terminated_before_success
- Task success: false
- Policy timestep: 684
- Checkpoint and stats hashes: exact expected values
- All recorded arrays and raw chunks were finite
- Raw chunk array: shape (684, 100, 14)
- 684 nonzero policy-action rows
- 684 unique hashes per camera
- Video: H.264/yuv420p, 685 frames, 960x270, 20 fps
- Container FFmpeg decoded the video without errors

The terminal object pose at step 684 is a post-termination/reset-like state,
so object comparison uses the last pre-termination state as well as the raw
trajectory. Before termination, gear 2 reproduced the baseline first pickup:

- Gear 2 maximum displacement: 0.19958 m
- Gear 2 maximum positive Z displacement: 0.18077 m at step 63
- At step 683, gear 2 remained at approximately 0.0984 m displacement and
  -0.00685 m Z displacement after the pickup/drop sequence
- No second-object score transition occurred during the extra horizon
- Gear 1 did not achieve a stable second pickup; its pre-termination
  displacement was approximately 0.0500 m

The additional horizon therefore reproduced the first baseline pickup but did
not produce the second gear recovery. The native termination then dropped the
reported score from 1 to 0. This does not support early 590-step truncation as
the main blocker.

## Stop condition

H1 newest-only, H2 oldest-first, and H3 extended-horizon baseline all failed
to produce a complete ACT Task 1 rollout. No further parameter scans, retries,
training, rule-policy substitution, scorer changes, or geometry changes were
performed.
