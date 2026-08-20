# Stage E result

Status: **IN PROGRESS**.

## Official DataReplay attempts

- Attempt 1: **ERROR** before environment creation. The upstream reference
  wrapper assigns a getter-only `device` property. Evidence is preserved under
  `artifacts/replay_probe/run1/`.
- Attempt 2: **FAIL** under the locked replay gate. It executed all 590 official
  actions at 20 Hz with exact source hash, exact reset qpos, exact wrapper/file
  ordering, no done/reset, finite state, and mean absolute target error 0.0100.
  Its demanded-motion direction agreement was 0.626 versus the preregistered
  0.80 threshold. Best/final task score was 0 because seed 1 does not reproduce
  the demonstration image/layout and the HDF5 omits its source reset metadata.

The independent preregistered equivalent-interface follow-up is pending. The
full replay is not retroactively reclassified.
