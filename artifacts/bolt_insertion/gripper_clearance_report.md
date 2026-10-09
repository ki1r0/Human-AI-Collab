# Panda Upper-Cap Gripper Clearance

## Decision

Candidate #01 supports a geometry-feasible upper-cap side pinch with the bolt upright on world Z. Align the Panda finger-closing axis across the cap diameter. At the nominal top-down TCP height below, the finger tips clear the cover bearing plane; this is a geometric clearance result, not proof of grasp stability or force closure.

## Measured Geometry

- `runtime/bolt_harness/env_cfg.py` sets TCP body `tool_center` and hand-local offset `(0, 0, +0.1034) m`. Positive local Z is exact; the top-down pose points it toward the cover. The Panda USD has `/panda/panda_hand/tool_center` as an Xform, not a rigid-body link. The runtime articulation `body_names` observed by the environment run excludes `tool_center`, so `env.py` uses its hand-pose plus rotated local-offset fallback.
- Local collision assets are `.../Props/panda_hand.usd` (3,319 vertices), `.../Props/panda_leftfinger.usd` (313), and `.../Props/panda_rightfinger.usd` (313); each mesh carries collision schemas. The authored finger joints move oppositely along hand-local Y (positive/negative Y), with 0–40 mm travel per finger.
- The maximum distal finger collision height is `hand-local Z = 0.112299735 m`. Compared with the `0.1034 m` TCP offset, the tips extend `8.899735 mm` past the TCP along local +Z, or below it in the top-down pose. The palm maximum is `Z = 0.0659999 m`.
- The cap is `8.299995 mm` tall and measures `23.094 mm × 20.000 mm` across its two horizontal axes at the configured CAD scale. The target opening is therefore 20–23.094 mm, with the jaw axis aligned to the chosen cap diameter.

## Nominal Clearance

Set TCP to `cover_top + 9.899735 mm`: the distal finger tips then remain `1.000 mm` above the bearing plane. The cap overlaps the vertical finger-pad span by `7.299995 mm` while seated. At this same TCP height, the palm's lowest collision point is about `47.30 mm` above the cover (`9.899735 + 103.4 - 65.9999 mm`); the fingers, not the palm, are the limiting geometry. Rounding TCP height to 10 mm gives about 47.4 mm palm clearance.

Recommended step: pinch the sides of the upright cap, keep the distal finger ends 1 mm above the cover top, then lift/hold the bolt for the existing straight axial seating step. No bolt reorientation or threading is needed. The mesh check does not establish contact forces, frictional stability, reachability, or dynamic collision behavior.

## Audit Corrections And Evidence

The audit JSON was corrected to label cap widths in millimeters, avoid inferring runtime fallback from the mere existence of a `tool_center` Xform, compute palm clearance independently from the whole-hand limiting clearance, and describe the cover height as a relative reference. Runtime fallback is determined by `env.py`'s articulation `body_names` check, not by this offline USD inspection.

Sources: [environment config](../../runtime/bolt_harness/env_cfg.py), [environment TCP fallback](../../runtime/bolt_harness/env.py), [local Panda USD](../../assets/vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd), [palm collision USD](../../assets/vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/Props/panda_hand.usd), [left-finger collision USD](../../assets/vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/Props/panda_leftfinger.usd), and [right-finger collision USD](../../assets/vendor/remote_mirror/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/Props/panda_rightfinger.usd).

The independent installed-PXR audit completed in 0.36 s, as reported by the main environment run. No GPU/Kit physics or asset edits were used for this gripper clearance check; no full audit rerun was performed for this report update.
