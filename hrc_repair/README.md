# `hrc_repair` M0

This is the minimal project adapter for the supplied REPAIR M0 plan.  It
reuses `hrc_m1` and `tools/run_m1_isaac_strict_success.sh`; it does not add a
new policy or silently replace an unavailable model.

Use the physical config only through the Isaac/Kit launcher environment:

```bash
python3 -m hrc_repair.preflight --config configs/repair_m0.yaml --strict
python3 -m hrc_repair.run --config configs/repair_m0.yaml --method repair --scenario nominal --seed 100
```

For contract and epoch/handoff tests that do not claim physics evidence:

```bash
python3 -m hrc_repair.run --config configs/repair_m0_contract_smoke.yaml \
  --method repair --scenario blocked --seed 0
```

The `isaac_inprocess` backend keeps one Isaac worker alive across a guarded
place, helper handoff, and resumed placement. The older subprocess backend is
reset-per-skill and records that reduced handoff fidelity explicitly.

## REPAIR RGB task adaptation

The author repository is a prompt release, not a complete robot-control stack.
The task-adapted prompts stay under `hrc_repair/prompts/`, while the unmodified
author prompts remain under `third_party/em_repair/`. `--official-rgb` enables
a separate Qwen RGB state-recognition call at reset, after the pick/lift, after
placement, and after help. The planner receives the resulting text state, not local image paths. A
guarded failure pauses the same Isaac worker; after the helper report and new
RGB observation, Isaac resumes only if the planner explicitly chooses `place`.

Run the scattered, full-gravity task adaptation with the local Qwen endpoint:

```bash
export HRC_REPAIR_PLANNER_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions
export HRC_REPAIR_API_KEY=local
bash tools/run_repair_m0_official_rgb.sh
```

This reuses the existing strict waypoint controller as the registered physical
skill. The planner chooses `pick` and `place` separately; Isaac pauses in the
same dynamic episode after the lift and waits for the planner again. This is a
task adaptation, not a claim to reproduce REPAIR's original household robot
skills. The current helper is still the runbook's registered-blocker script,
not the paper's human teleoperation interface.

The first RGB run exposed a reachability/grasp failure: the requested Casing
socket tool pose was about 0.96 m from the robot base, while the measured arm
stayed much closer and the Hub lost contact during transport. The isolated
dual-arm variant moves both parts into the measured workspace and keeps both
arms synchronized; its resolved coordinates and controller flags are saved
with the run:

```bash
bash tools/run_repair_m0_official_rgb_dual.sh
```

## VADER on the same M0 task

The VADER adaptation starts from the scattered full-physics scene in
`configs/repair_m0_scattered_pure.yaml`: the dynamic `Hub_Cover_Output_Top`
and `Casing_Top` are separate on physical supports. It captures and checks the
initial state, asks the planner for a physical `pick`, checks the grasp from
RGB, then plans `place`. No part pose is written during the episode. `finish`
requires VQA success, public placement/release signals, and the independent M0
physics evaluator. This adapts the VADER loop; it is not the original HRFS or
multi-robot system.

The verifier uses the head RGB view until public placement and release are
confirmed, then the right-hand RGB view for final seating verification. The
expected outcome describes the output cover's raised center opening so it is
not mistaken for an unseated part. Earlier preplace runs and their limitations
are recorded in the [prior report](../reports/vader_m0_reproduction_20261007.md);
the new scattered full-physics run is recorded separately.

Start the local multimodal Qwen endpoint on an available GPU:

```bash
docker run --rm --gpus 'device=1' --network=host \
  --entrypoint /isaac-sim/python.sh \
  -v "$PWD:/workspace/Human-AI-Collab:ro" \
  -v "$PWD/pilot_12pair/outputs/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct:/models/qwen:ro" \
  human-ai-collab:latest \
  /workspace/Human-AI-Collab/tools/repair_qwen_server.py \
  --model /models/qwen/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  --host 127.0.0.1 --port 18081 --max-new-tokens 128
```

Run the scattered pick-to-place episode on a separate Isaac GPU:

```bash
M1_GPU_DEVICE=2 \
HRC_REPAIR_PLANNER_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions \
HRC_REPAIR_PLANNER_MODEL=Qwen/Qwen3-VL-2B-Instruct \
HRC_REPAIR_VLM_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions \
HRC_REPAIR_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct \
HRC_REPAIR_API_KEY=local \
python3 -m hrc_repair.run --config configs/repair_m0_scattered_pure.yaml \
  --method vader --scenario nominal --seed 100 --backend isaac_inprocess \
  --out validation_logs/vader_m0_scattered_full_physics
```

The VLM sees only RGB snapshots; their local file paths are not sent to the
LMP. A VADER `finish` is accepted only with VQA `SUCCESS` plus the existing
public placement and release signals; the independent M0 evaluator remains
authoritative.
