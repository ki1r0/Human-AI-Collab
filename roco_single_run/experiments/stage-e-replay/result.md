# Stage E result

Status: **PASS by preregistered equivalent action-interface test**.

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

## Equivalent-interface follow-up

The fresh seed-42 follow-up passed all locked criteria across 40 steps. Positive,
return, negative, and final-return phases each exercised 13–14 demanded
channels, achieved direction agreement 1.0, displacement cosine 0.9924–0.9940,
and final target mean absolute error 0.00186–0.00188. Final return error was
0.001878; mapping sentinel, timing, finiteness, and no-done checks all passed.

Evidence:

- `artifacts/replay_probe/equivalent_run1/result.json` SHA-256
  `d124664334322d15b4015adee34cf90c359fc3bc248e02c97196a30ce0b5b321`.
- `artifacts/replay_probe/equivalent_run1/trace.npz` SHA-256
  `e648121dcd24a05d75b6702535cf0b09d3f95e5f890399d5bd8a727ce6f93a25`.
- `artifacts/replay_probe/equivalent_run1/console.log` SHA-256
  `c980c3c94cbc70a8555732493b173e9ddd2289e41912712255a4d568744a910c`.

The full replay remains FAIL and is not retroactively reclassified; Stage E
passes through the independently preregistered equivalent route allowed by the
mission.
