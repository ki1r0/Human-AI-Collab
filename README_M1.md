# M1 Gearbox HRC

This package is an auditable single-step integration for the red
`Hub_Cover_Output_Top` into `Casing_Top/socket_hub_output`.  The identity is
`step_14_hub_cover_output_top` / `inst_025_combine_hub_cover_output_top`.

## Current status

The Python contract layer and Isaac Lab scene run.  The latest collision-on
reset-only probe is `PROVISIONAL_PASS` at
`validation_logs/m1_physics_trial_default_20260926T1810Z/`; it reproduces the
task config defaults. The detailed calibration record
`validation_logs/m1_physics_trial_sdf0_20260926T1630Z/` measured valid contact
and `0.0986 mm` penetration after a slow insertion with the Hub
SDF margin explicitly set to zero. This is not
yet a robot-held pick/insert/release calibration, so G0 remains provisional and
G1 is blocked.  No LLM/VLM credentials are configured in this workstation
session, and no human operator has completed HIL.  Therefore no run in this
checkout is a nominal, HIL, or task-success claim.

The latest fixed-skill trace is
`validation_logs/m1_skill_smoke_hold_gate_20260926T1700Z/`. The adapter now
returns `FAILED / HELD_UNVERIFIED` after the pick motion because the current
scene has no public calibrated grasp signal; it does not continue to
preinsert/insert. The earlier `m1_skill_smoke_framefix_20260925T1230Z/` is
preserved as historical controller-only evidence. R1 TCP/gripper contact
calibration is therefore an explicit G1 prerequisite. A collision-on grasp
probe is recorded in `validation_logs/m1_grasp_blocker_20260926.md`. The
supplied photo's indicated region is treated as an inner-bore/inner-annular-
wall grasp, not an outer-diameter grasp. The follow-up calibrated-offset runs
`validation_logs/m1_inner_wall_probe_calibrated_20260929T134657/`,
`validation_logs/m1_inner_wall_probe_preopen_calibrated_20260929T135303/`, and
`validation_logs/m1_inner_wall_probe_open040_calibrated_20260929T135626/`
removed the source R1's large speculative contact-offset confound, but did not
yet establish a load-bearing held lift. G1 is therefore blocked by an
unvalidated grasp path and missing public gripper-to-Hub verification signal;
this is not evidence that the canonical Hub or inner-wall strategy is
physically impossible. Two filtered link-level ContactSensors are now present
for diagnosis; the pre-open run
`validation_logs/m1_inner_wall_probe_sensors_20260929T140000/` measured `0 N`
on both fingers and `0 m` Hub displacement, so that waypoint did not contact
the Hub. The raw diagnostic fields are not yet exposed as the action-model's
held predicate. Increasing the actual opening to approximately `0.0442 m` and
`0.0461 m` instead ejected the Hub during approach (`0.335 m` and `0.673 m`
displacement; see `validation_logs/m1_inner_wall_probe_midopen_20260929T145000/`
and `validation_logs/m1_inner_wall_probe_maxopen_20260929T144000/`).

The post-guard no-help regression is
`validation_logs/m1_fault_no_help_prompt_hash_20260926T1930Z/`. It reaches
`SAFE_ABORT` after `HELD_UNVERIFIED` and verifier `UNKNOWN`, with no insertion
attempt. This is a safety/contract result using the dev-only rule planner, not
an autonomous-model result.

Container manifests record checkout provenance through an explicit Git
`safe.directory` query. The regression
`validation_logs/m1_manifest_git_smoke_20260926T1830Z/` confirms that both
repository HEADs and dirty states are present in `manifest.json`.
The Isaac runner also assigns the episode seed before scene construction;
`validation_logs/m1_seed_smoke_20260926T1900Z/` records `Environment seed:
1201` in the startup trace.
When a real provider is configured, decision and verification events also
record prompt/request hashes and response hashes; the evaluator-only fault
profile remains in `evaluator_private.jsonl`.

## Run the contract smoke

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_hrc_m1*.py'
python3 -m hrc_m1.run --backend mock \
  --task-config m1_hub_cover_output_top_seat.yaml \
  --mode nominal --seed 1201 --out runs/mock_contract
```

The output is deliberately marked `MOCK_ONLY`; it cannot establish physical
success.

## Run the collision-on interface probe

The probe is a calibration diagnostic, not an evaluated robot episode.  It
places the cover at reset, gives it one initial velocity, and then advances
PhysX without per-step pose writes:

```bash
docker run --rm --gpus 'device=0' --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_ENV_PRIVACY_CONSENT=YES \
  -e PYTHONUNBUFFERED=1 \
  -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e ROCO_ROOT=/workspace/gearboxAssembly \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v /home/sunsiliang/Human-AI-Collab:/workspace/Human-AI-Collab \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 -m hrc_m1.run \
  --backend roco --task-config /workspace/Human-AI-Collab/m1_hub_cover_output_top_seat.yaml \
  --mode nominal --seed 1201 \
  --out /workspace/Human-AI-Collab/validation_logs/m1_physics_trial_default_20260926T1810Z \
  --physics-trial \
  --headless --enable_cameras
```

The private evaluator record reports the contact distance and the public
`metrics.json` reports `PROVISIONAL_PASS`; it deliberately keeps
`task_success=false` until robot-held calibration is complete.  The MP4 in
that directory is a three-camera diagnostic video.

## Inspect the Hub inner-wall grasp

The following diagnostic keeps the Hub dynamic, overrides the source R1 USD's
`0.05 m` contact offset only inside this run, and records both filtered
gripper-link-to-Hub sensors. It never writes the Hub pose after reset:

```bash
docker run --rm --gpus 'device=0' --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e ROCO_ROOT=/workspace/gearboxAssembly \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v /home/sunsiliang/Human-AI-Collab:/workspace/Human-AI-Collab \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 -m hrc_m1.debug_inner_wall_grasp \
  --output-dir /workspace/Human-AI-Collab/validation_logs/inner_wall_probe \
  --z-offset 0.048 --opening 0.044 --gripper-contact-offset 0.0001 \
  --preopen --no-video --headless --enable_cameras
```

Treat `metrics.json` as a diagnostic: a nonzero Hub displacement is not a
success unless the Hub follows the commanded lift and both contact/object-
follow checks are calibrated. The current adapter deliberately keeps the
held predicate unverified.

## Run the Isaac reset/camera smoke

Isaac Sim is supplied by the pinned RoCo container.  The camera flag is
required because the R1 bundle contains RTX cameras:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_ENV_PRIVACY_CONSENT=YES \
  -e PYTHONUNBUFFERED=1 \
  -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e ROCO_ROOT=/workspace/gearboxAssembly \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v /home/sunsiliang/Human-AI-Collab:/workspace/Human-AI-Collab \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 -m hrc_m1.run \
  --backend roco --task-config /workspace/Human-AI-Collab/m1_hub_cover_output_top_seat.yaml \
  --mode nominal --seed 1201 --out /workspace/Human-AI-Collab/runs/sim_smoke \
  --smoke --smoke-steps 3 --headless --enable_cameras
```

`manifest.json`, `events.jsonl`, `metrics.json`, `episode.mp4`, `frames/` and
the `0600` evaluator-only stream are written under the run directory.  The
smoke result is `SIM_SMOKE_ONLY`, not a task result.

## Record the completed M1 camera rollout

The direct M1 control baseline has a separate RoCo-style live-camera launcher:

```bash
M1_GPU_DEVICE=1 \
M1_DIRECT_CAMERA_OUTPUT_DIR=/absolute/path/to/output \
M1_CAMERA_UPDATE_STRIDE=10 M1_RENDER_INTERVAL=5 \
M1_CAMERA_WIDTH=320 M1_CAMERA_HEIGHT=240 \
M1_STEP_SCALE=0.5 M1_RELEASE_OPEN_STEPS=1000 M1_GRAVITY_SETTLE_STEPS=100 \
./tools/run_m1_direct_camera_policy_success.sh
```

It writes the three live RGB views to `inner_wall_probe.mp4` using RoCo's
`imageio`/H.264 `yuv420p` path.  The verified reference artifact is
`validation_logs/m1_direct_camera_policy_success_20261001_final/`; its MP4 has
941 frames, `320x240` per camera, and `SEAT_RELEASE_CANDIDATE` metrics.

## Real rollout prerequisites

Before using the three plan commands, complete the pending robot-held
calibration in `hrc_m1/calibration_manifest.json`, provide a real
OpenAI-compatible planner and VLM endpoint through
`HRC_M1_PLANNER_ENDPOINT`/`HRC_M1_PLANNER_MODEL` and
`HRC_M1_VLM_ENDPOINT`/`HRC_M1_VLM_MODEL`, and attach a human operator for
`fault_hil`.  The runner refuses to substitute a fake model.  The current
checkout also has pre-existing dirty changes and the RoCo checkout is mounted
read-only; their provenance is recorded in `M1_PRECHECK.md`.

## VADER on the M0 task

The M0-aligned VADER adaptation uses the same preplace task as local REPAIR M0:
`Hub_Cover_Output_Top` → `Casing_Top/socket_hub_output`, with the existing
persistent Isaac worker and M0 placement evaluator. It does not add a separate
pick-and-carry task. Setup and run commands are in
[`hrc_repair/README.md`](hrc_repair/README.md). This adapts VADER's
plan-execute-detect loop to this one assembly step; it is not the original
HRFS/multi-robot system.
