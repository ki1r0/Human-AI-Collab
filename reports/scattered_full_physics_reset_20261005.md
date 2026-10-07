# Scatter full-physics reset (2026-10-05)

This is a new reset variant; the accepted `configs/repair_m0_pure.yaml`
preplace baseline is intentionally unchanged.

## Scene contract

- Table collision size: `1.8 x 1.8 x 0.10 m`; top surface `z=0.934 m`,
  centered at `(1.20, 0.00, 0.884) m` so its X range starts just beyond the
  R1 base while remaining reachable by the arms.
- Casing initial XY: `(0.55, 0.00) m`.
- Hub Cover initial XY: `(0.55, 0.43) m`; the two CAD parts are separated by
  approximately 50 mm in Y at their measured bounds.
- The root Z for each part is calculated from the authored USD world AABB so
  its bottom starts 2 mm above the table, then gravity settles it onto the
  table.  No staging pads or kinematic casing support are spawned.
- Hub and Casing are both dynamic and gravity-enabled before the first physics
  integration step.  The table remains a kinematic support surface.
- Dynamic Casing uses an explicit SDF collision approximation. Isaac rejects a
  dynamic raw triangle mesh and would otherwise fall back to a convex hull,
  which would seal the socket opening.

## Reset probe

Command (camera smoke only; no placement claim):

```bash
M1_ISAAC_STRICT_OUTPUT_DIR=validation_logs/scatter_reset_camera_20261005_raised \
M1_GPU_DEVICE=1 M1_SCATTER_RESET=1 \
M1_HUB_RESET_X=0.55 M1_HUB_RESET_Y=0.43 \
M1_CASING_RESET_X=0.55 M1_CASING_RESET_Y=0.0 \
M1_TABLE_SIZE_X=1.8 M1_TABLE_SIZE_Y=1.8 M1_TABLE_TOP_Z=0.934 \
M1_TABLE_CENTER_X=1.20 M1_TABLE_CENTER_Y=0.0 \
M1_CAMERA_PROBE_STEPS=10 M1_RENDER_INTERVAL=5 \
tools/run_m1_isaac_strict_success.sh
```

Artifacts:

- `validation_logs/scatter_reset_camera_20261005_raised/scatter_reset.json`
  records the derived roots and predicted AABBs.
- `validation_logs/scatter_reset_camera_20261005_raised/scatter_settle.json`
  records the actual post-step root states and velocities.
- `validation_logs/scatter_reset_camera_20261005_raised/camera_probe.mp4` is a
  live RTX-camera probe, not a placement-success video.

Measured result after 10 physics/control steps:

- Hub root: `(0.550000, 0.430000, 0.948009) m`, linear speed
  `4.8e-9 m/s`.
- Casing root: `(0.549999, 0.000001, 0.989847) m`, linear speed
  `3.6e-4 m/s`.
- Both parts had settled from the 2 mm spawn margin to the table surface;
  neither remained suspended over the old socket location.

This probe does not yet establish a successful pick-and-place.  The next run
should execute the controlled full-physics grasp/transport path from this
scatter reset and evaluate grasp, socket alignment, release, and retract.

The raised-table lift probe reached the real Hub area and remained in full
physics, but scored only `1/6` (supported dynamic reset).  The filtered finger
contacts were zero and the Hub moved laterally on the table, so this run is
recorded as a grasp-calibration failure, not a placement success.  The reset
change therefore fixes the hovering/support problem without masking the next
real blocker: the R1 approach/grasp waypoint still needs to be calibrated for
the table-mounted Hub.
