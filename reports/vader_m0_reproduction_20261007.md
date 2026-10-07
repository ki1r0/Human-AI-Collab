# VADER Adaptation on REPAIR M0 (2026-10-07)

## Task and implementation

This adapts VADER's plan-execute-detect loop to the local REPAIR M0 task in
`configs/repair_m0_pure.yaml`: a physically pre-held `Hub_Cover_Output_Top`
is placed into `Casing_Top/socket_hub_output` using the existing persistent
Isaac worker. It is not a pick-and-carry or full HRFS/multi-robot reproduction.

The LMP plans only registered M0 actions. After placement, VQA assesses the
saved RGB view; public contact/release/retract signals remain independently
required for `finish`. The head camera is used before release is confirmed;
the right-hand camera is used for final seating once public placement and
release signals are both affirmative. Malformed or stale LMP output receives
one corrective request; a second invalid response is rejected.

## Evidence

`validation_logs/vader_m0_blocked_20261007_r4/` contains a real blocked M0
episode with the trace `place -> help -> place`. Its independent
`REPAIR_M0_PLACEMENT_V1` evaluator passed all six checks. The runner stopped
with `BUDGET_EXHAUSTED` because the then-current head-only VQA falsely rejected
the successful final placement.

The saved R4 public observation and camera frames were replayed through the
updated assessment and live local Qwen LMP: the blocked state returned VQA
`FAILED`; the release-confirmed final state returned `SUCCESS`; the LMP then
returned `finish`. This validates the corrected perception/planner decision
on a real successful physics result, but is an artifact replay, not a new
live episode reaching `DONE`.

Later runs show the physical controller is variable: R6 ejected the cover
after help (post-retract drift `235.7 m`), while nominal R7 missed the M0 goal
(radial error `22.8 mm`, axial error `141.5 mm`). These are retained as
failures; neither the evaluator nor controller was relaxed to improve the
reported result.

## Reproduction command

With the local Qwen endpoint running at port `18081`:

```bash
M1_GPU_DEVICE=3 \
HRC_REPAIR_PLANNER_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions \
HRC_REPAIR_VLM_ENDPOINT=http://127.0.0.1:18081/v1/chat/completions \
HRC_REPAIR_API_KEY=local \
python3 -m hrc_repair.run --config configs/repair_m0_pure.yaml \
  --method vader --scenario blocked --seed 100 --backend isaac_inprocess \
  --out validation_logs/vader_m0_blocked
```

Verification: strict M0 preflight passed and all 28 unit tests passed with
`python3 -m unittest discover -s tests -p 'test_*.py' -v`.
