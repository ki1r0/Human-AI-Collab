# Hub-cover HRC harness

Thin M0/M1 harness for `Hub_Cover_Output_Top -> Casing_Top/socket_hub_output`,
following `HRC_Harness_M0_M1_Build_Plan_ZH.md` (2026-10-07). It does not migrate
the scene, modify the REPAIR/VADER runners, or claim their physical acceptance.

Scope update (2026-10-08): the active pipeline stops at emitting a help request.
Both supplied configurations use `help_mode: request_only`; there is no helper
execution, handover, repair, verification or robot resume after a request.
The request is emitted to the local public trace and `help_request.json` outbox,
not delivered to a human, remote service or HRFS.

## Run Now

From the repository root, with the existing host Python/PyYAML:

```bash
python -m unittest discover -s tests -v
python -m hrc_harness.run --condition nominal --output runs/harness/contract_nominal
python -m hrc_harness.run --condition blocked --output runs/harness/contract_blocked
```

These are **contract-only**, not physics experiments or paper reproductions.
They exercise pick/prealign, bounded contact, public finish or a terminal help
request. The blocked smoke ends `help_requested`, without repairing the scene.
Run directories must be new.

In the existing Isaac Lab environment, with `Galaxea_Lab_External` on
`PYTHONPATH`, run the real scene/sensor smoke without manipulation or model calls:

```bash
python -m hrc_harness.isaac --headless --enable_cameras --smoke-steps 60 --output runs/harness/sensors
```

The already tested container command uses the existing image and runtime:

```bash
docker run --rm --gpus device=3 --shm-size=16g --network none \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v "$PWD:/workspace/Human-AI-Collab" \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  -w /workspace/Human-AI-Collab nvcr.io/nvidia/isaac-lab:2.3.0 \
  -m hrc_harness.isaac --headless --enable_cameras --smoke-steps 60 \
  --output runs/harness/sensors
```

Choose an actually idle GPU; GPU 3 was idle during the initial smoke.
The offline pilot's optional `dual_arm_tcp_offset_m` commands both arms through
ordinary IK at each tick. Its fixed nominal offset is not runtime part-pose
feedback. Four finger contacts are guarded, and private release scoring requires
both hands clear. Current dual-arm lift evidence does not authorize production
insertion or certify the holding classifier.
The source USD audit is available as `python -m hrc_harness.audit` in a Python
environment with `pxr`. Installed Isaac USD libraries can also be used without
starting Kit. Results: `reports/harness_asset_audit_20261007.json`.

For the online model loop, `harness_isaac.yaml` uses a dedicated local Qwen
endpoint at `127.0.0.1:18082`. The existing model launcher can serve the installed
weights with `--port 18082 --max-new-tokens 2048`; its former 128-token setting
is insufficient for the harness proposal schema. Use host networking to reach
the loopback endpoint from Isaac. Give concurrent Isaac containers separate IPC
namespaces (`--shm-size=16g` instead of `--ipc=host`). Replace `--smoke-steps 60`
in the container command with no smoke flag to run the complete online loop.
`model_trace.jsonl` retains the public context and raw model response, without
Authorization headers or inline image bytes. Complete JSON fences are decoded;
truncated or invalid proposals still get one repair and then stop.
The model receives recent full packets in `observation`/`outcome_window` and
historical observation IDs/ticks in the ledger, alongside all tool results.
Full raw observations remain in `public_trace.jsonl`; they are not repeated
unboundedly in each model request. Candidate-level evidence references are optional
and checked against the same public ledger as proposal-level references.

With the current uncertified manipulation profiles, a live model run exercises
camera/encoder/contact observations, proposal validation, selection, observation
tools and stop/help-request termination. It cannot validate physical pick/seat
execution or `seat_gt`; those require the physical pilot below.

## Boundaries

- `contracts/evidence`: typed public observations, acquisition timestamps,
  missing channels, image IDs rather than local paths, append-only raw events,
  fixed features, <=3 nonexclusive evidence-linked hypotheses.
- `runtime`: whitelisted profiles, per-tick cancellation/protection, duplicate
  and stale-command rejection, robot/helper ownership with epochs, contact/probe/
  inspection/request/wall budgets, and a safe-held terminal help-request outbox.
  Older intervention/resume code and safety tests remain as an inactive extension;
  neither supplied configuration invokes them.
- `isaac`: measured robot/camera/contact channels and nominal Cartesian profiles.
  Commands do not query actual part root poses or set part poses. Gravity is
  enabled from reset; the casing is an explicitly fixed fixture. The only runtime
  part-area mutation is the deferred helper's authorized blocker-clearing operation;
  the active request-only pipeline never calls it.
- `proposer`: shared multimodal HTTP model and candidate context across methods.
  Visual estimates have their own source/events and do not overwrite sensor data.
  A single image cannot establish stable dwell. One schema repair is allowed;
  API failure stops safely, without silently switching to a fake model.
- `decision/calibration`: public-bin empirical counts, support/unsupported status,
  depth-2 branch enumeration for autonomous actions, grouped
  train/validation/test splits, Brier/coverage reports. Probe costs cover only the
  probe; continuation costs are added once. Information gain is entropy reduction
  of offline initial categories, not entropy of success probability.
  Request-only help is not assigned a fictitious completion probability. Until
  its terminal decision utility is defined, proposed logs a shared-reasoner
  fallback when comparing with help. Existing helper-completion estimates are
  ignored in this mode, including in depth-2 continuations.
- `evaluation`: private terminal scoring after online finish/stop, insertion-axis
  tilt, released contact/stability dwell, symmetry-aware orientation registration.
  Missing axes/sensors or provisional tolerances produce UNKNOWN, never PASS.

Physical hard limits and profiles in `configs/harness_isaac.yaml` and
`configs/task_hub_cover.yaml` are intentionally **uncertified**. Do not copy mock
limits from `harness_contract.yaml` to the robot. Supply approved limits, measured
nominal waypoints and an accepted grasp estimate before registering
contact tools. Each waypoint specifies `tcp_xyz_m`, normalized `quat_wxyz`, and
positive integer `ticks`; optional `gripper_opening_m` commands the gripper.
Certified release waypoints specify `phase: release`. XY/angle profiles must include
a certified unloading segment and change only their declared factor; this requires
pilot validation, not merely setting a YAML flag.

`inspect` currently refreshes the existing three camera views. It does not move a
camera or add an unseen sensor. ACT is not registered: its existing full trajectory
path has not been verified as interruptible under this harness. Public RGB-D
depth/tilt estimation is not implemented; its channels remain null. The HTTP
reasoner can provide explicitly labeled visual inferences, whose accuracy still
requires independent validation.

## Online State Check

The existing public `evidence.monitor` now reports `CANDIDATE_COMPLETE`, `BLOCKED`,
`IN_PROGRESS`, or `UNKNOWN`, with a reason in the public trace and reasoner context.
Completion still requires fresh visual seating, release and temporal stability;
neither contact force nor reaching a robot target proves seating. Invalid/stale
observations cannot authorize finish. Terminal `seat_gt` remains independent.

`InsertionCheck` adds a small rule-based check to the actual Isaac waypoint loop:
only contact-tool waypoints marked `phase: insert` count toward insertion time.
A sustained contact-force window with insufficient forward TCP progress, while
still short of the final insertion target, returns `STALLED`. So does insertion
timeout or a final insertion waypoint ending short of that target. Execution
safe-holds and skips all remaining waypoints, including release; the shared
decision loop then chooses retry, inspection, help request or stop. No helper is
invoked in request-only mode. Unload/hold phases clear the contact window.

Set the five `online_check` parameters in `configs/task_hub_cover.yaml` from measured
task data: window duration, minimum progress, contact threshold, TCP target
tolerance, and insertion timeout. They intentionally remain null, and the
grasp estimate remains uncertified: this implementation does not
authorize contact execution. Missing axis, force or verified grasp produces
`UNKNOWN`, not zero force or success. Progress is signed TCP travel along the
declared unit insertion axis, oriented toward the nominal insertion target rather
than assuming world +Z. It is not true part depth or commanded motion. Reports
retain their measurement timestamps rather than becoming fresh on re-observation.
The scalar filtered cover/casing contact is not a wrist six-axis wrench; no
lateral-force/tilt diagnosis is claimed.

This reuses the harness's existing progress result, public channels and completion
monitor, and follows the visible-evidence/UNKNOWN distinction in
`hrc_m1/observer.py` and `hrc_repair/state_recognizer.py`. It does not import their
diagnostic GT scoring, Hayami's fPCA/SVM classifier, or Jev. Unit checks exercise
the actual Isaac waypoint function with synthetic encoders/contact data, including
early stop before release and the terminal request boundary. They are not physical
seating validation or an accuracy comparison on real assembly episodes.

## Physical Pilot

`reports/harness_axis_geometry_20261008.json` measures the transformed CAD meshes:
the cover's plane normal is local Y, mapped to world Z by its nominal assembly
quaternion; the casing's normal is local Z. These axes are configured, but the
nominal seated root offset and scoring tolerances still require physical acceptance.

Run a guarded offline pilot in the same Isaac container, replacing the smoke
module and arguments above with:

```bash
python -m hrc_harness.pilot --headless --no-video --stage jaws --output runs/harness/pilot_jaws
python -m hrc_harness.pilot --headless --no-video --stage grasp --output runs/harness/pilot_grasp
python -m hrc_harness.pilot --headless --enable_cameras --stage insert --output runs/harness/pilot_insert
python -m hrc_harness.pilot --headless --enable_cameras --stage place --output runs/harness/pilot_place
```

`configs/harness_pilot.yaml` records provisional simulator-only limits and fixed
CAD/reset-based commands. The cover remains dynamic under gravity; collisions,
the original contact materials and per-tick protection remain enabled. No grasp
constraint, runtime part-pose write or hidden-pose action correction is used.
Gripper targets are ramped per tick along with the waypoint; a large instantaneous
target jump caused the source mimic jaws to collapse in a contact-free diagnostic.
The source custom reset now initializes both mimic jaws consistently. Table and
support collision geometry follows the configured heights; the table front edge
is 0.15 m from the robot origin to avoid initial torso penetration.

This pilot is deliberately separate from authorization of production motion
profiles. It collects public encoders/contacts and private offline measurements,
then stops on missing pair contact, lost contact or protection. The insertion
pilot does not release from an unverified seat. Neither reaching its target nor
its post-run grasp measurements certify seating or checker accuracy.
The separate `place` stage can stop descent at `seat_contact_stop_N`, retain the
current command, verify contact presence, then release/retract and record gravity
settling. That scalar trigger does not certify seating. AU completed this full
direct-command flow with four fingers clear and sustained casing support;
formal `seat_gt` remains UNKNOWN and production/VLM motion profiles stay disabled.

## Calibration and Comparisons

To collect a branch, supply `--branch-plan path/to/plan.json` to either runner:

```json
{
  "snapshot_id": "development_initial_state_01",
  "split": "train",
  "prefix": ["pick", "seat"],
  "candidate_id": "xy",
  "continuation": ["seat", "finish"]
}
```

This is offline intervention, not an online method. Each run uses independent reset
and prefix replay, not USD-pose snapshot restore. Verify replay consistency before
interpreting branches as the same initial state. Terminal GT labels are extracted
only after the scripted rollout; UNKNOWN GT or rejected tools are not fitted.
`calibration_record.json` contains offline labels and is excluded from Git.
A request-only help branch is not fitted as help-assisted task completion.
Combine accepted records into a JSONL file, preserving one split per initial state:

```bash
python -m hrc_harness.calibration records.jsonl --output runs/harness/calibration_v1
```

Training uses train records only; validation reports coverage and Brier error.
Test records never update the table. Contract data cannot fit a physical estimator.
Sparse/ambiguous next-state bins abstain rather than inventing a response model.
No real calibration dataset or physical completion probability is supplied here.

Set `estimator_path` to the measured table before meaningful M1 comparison.
`--method` supports `repair_adapted`, `generic`, `random_safe`, `information`, and
`proposed`, plus M0 `auto`/`fixed_retry`. These names specify comparison policies;
they do not claim exact reproduction of the original REPAIR model. Random-safe
samples registered certified probes; information requires category calibration;
proposed explicitly logs an unsupported-estimate fallback when coverage is absent.
All methods share candidates, model input, executor, budgets and request boundary.

`seat_gt` scores actual autonomous seating, not the quality of asking. A terminal
request may leave `seat_gt=FAIL` while being the correct decision. Ask appropriateness
needs separate offline labels or a declared request utility; it cannot be inferred
from the final unseated part alone.

Artifacts: redacted config/Git-revision manifest, incrementally written public and
decision JSONL, private terminal GT (mode 0600), private per-tick physical samples,
summary, camera PNGs, and acquisition-cadence video. The summary's wall time already
includes model/help time: do not add those components again. Scalar contact exposure
is a burden proxy, not axial F/T or material damage. Local-model monetary cost is
currently zero; remote billing integration is not implemented. There are no new
hashes or frozen-contract mechanisms.

Request-boundary smoke (real Isaac, scripted actions, no model decision claim):
`python -m hrc_harness.isaac --config configs/harness_isaac.yaml --branch-plan runs/harness/request_boundary_plan_20261008.json --output runs/harness/request_boundary_new --headless --enable_cameras`.
The plan uses `inspect`, then `help`, with `record_calibration: false`. It must
terminate at the local request outbox, without invoking a helper or fitting a
completion label. This checks plumbing, not physical assembly or ask quality.
