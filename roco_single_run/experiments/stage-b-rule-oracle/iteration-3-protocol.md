# Stage B iteration 3 — minimum-legal contact margin

Classification: **EXPLORATORY**.

## Hypothesis

H8: reducing the collision compatibility patch to the smallest practical PhysX-legal margin allows the ring to seat without destroying the three carrier-pin placements.

## Rationale

The official source deliberately uses `contact_offset=0.0` and, since commit `8f57e9b`, `rest_offset=-0.0005` for gears so mating surfaces can overlap slightly during insertion. PhysX 5.1 rejects zero contact offset and requires contact offset to exceed rest offset. The initial compatibility patch was legal but added 0.1 mm contact offset to the gears and used 1.0/0.5 mm contact/rest offsets on the carrier. The latter creates a 0.5 mm contact margin beyond rest. A 1 micrometre epsilon is closer to source intent while satisfying the API constraint.

## Locked procedure

1. Change only compatibility-added contact offsets: ring/sun/reducer `0.0001 -> 0.000001` m and carrier `0.001 -> 0.000501` m. Preserve all rest offsets.
2. Rerun the Stage A health probe and require zero collision-offset errors, missing assets, NaNs, or exceptions.
3. If Stage A passes, rerun the identical official R1 oracle with seed 23, 0.20 m workspace, 20 Hz control, cameras, phase snapshots, and 700-step bound.
4. Preserve the official trajectory, score function, randomization, solver settings, and object assets.

## Prediction and decision rule

- H8 is supported if the seed-23 run exceeds best score 4 and the three pre-ring carrier-pin relations survive ring rotation.
- Stage B passes only at official score at least 6 without timeout or runtime error.
- If Stage A reports any invalid offset, reject the numeric patch without running Stage B.
- If Stage A passes but Stage B remains at most 4 or loses the pin relations during rotation, reject H8 and next isolate the ring rotation/descent trajectory rather than adjusting multiple physics properties.
