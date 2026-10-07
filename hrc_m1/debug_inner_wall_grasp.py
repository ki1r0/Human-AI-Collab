"""Collision-on diagnostic for the Hub Cover's hollow inner-wall grasp.

This is a diagnostic, not an M1 success path.  It uses the normal reset-only
scene initialization and then drives the existing R1 pose/gripper controller;
it never writes the Hub pose after reset.  The report records the two gripper
link poses, per-body contact-force norms, and the Hub displacement before and
after a lift so that an inner-wall grasp is distinguishable from a mere TCP
reach.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1201)
parser.add_argument("--z-offset", type=float, default=0.037,
                    help="Hub-root-relative world-Z target for the finger pair.")
parser.add_argument("--hub-reset-y", type=float, default=None,
                    help=("Diagnostic reset Y for the free Hub. This changes "
                          "only the initial scene placement before physics; "
                          "the default is the canonical 0.40 m."))
parser.add_argument("--hub-reset-x", type=float, default=None,
                    help="Diagnostic reset X for the free Hub; used to place a dual-arm source outside the Casing footprint.")
parser.add_argument("--hub-reset-z", type=float, default=None,
                    help="Diagnostic reset Z for the free Hub; use a validated staging height for a preplace-start run.")
parser.add_argument("--scatter-reset", action="store_true",
                    help=("Start with the Hub Cover and Casing separated on the enlarged table. "
                          "Both CAD bodies are dynamic with gravity enabled from reset; "
                          "their Z positions are derived from the authored USD bounds."))
parser.add_argument("--table-size-x", type=float, default=1.8,
                    help="Table collision size in X for the scatter reset (metres).")
parser.add_argument("--table-size-y", type=float, default=1.8,
                    help="Table collision size in Y for the scatter reset (metres).")
parser.add_argument("--table-top-z", type=float, default=0.934,
                    help="Table top world Z for the scatter reset (metres).")
parser.add_argument("--table-center-x", type=float, default=1.20,
                    help="Table center X for the scatter reset (metres).")
parser.add_argument("--table-center-y", type=float, default=0.0,
                    help="Table center Y for the scatter reset (metres).")
parser.add_argument("--casing-reset-x", type=float, default=0.55,
                    help="Casing scatter-reset X position on the table (metres).")
parser.add_argument("--casing-reset-y", type=float, default=0.0,
                    help="Casing scatter-reset Y position on the table (metres).")
parser.add_argument("--scatter-spawn-margin-m", type=float, default=0.002,
                    help="Positive clearance above the table used for the reset-time USD-bound snap (metres).")
parser.add_argument("--scatter-physical-supports", action="store_true",
                    help=("Add explicit kinematic support pads under the dynamic Hub and Casing. "
                          "The tabletop remains below the robot finger path; object support is "
                          "still resolved by PhysX contact, not by a pose write."))
parser.add_argument("--scatter-support-top-z", type=float, default=0.934,
                    help="Top Z of the explicit scatter support fixtures (metres).")
parser.add_argument("--scatter-hub-support-top-z", type=float, default=None,
                    help="Top Z of the Hub scatter supports; defaults to --scatter-support-top-z.")
parser.add_argument("--hub-mass-kg", type=float, default=None,
                    help=("Override the Hub mass used by PhysX for material calibration. "
                          "The default preserves the 5.7 kg CAD-density estimate."))
parser.add_argument("--radial-offset", type=float, default=0.0,
                    help=("Hub-root-relative world-Y offset for the whole jaw pair. "
                          "A nonzero value moves the pair off the bore centre so "
                          "one jaw can be inside and the other outside the ring."))
parser.add_argument("--opening", type=float, default=0.045,
                    help="Final R1 gripper-axis target used for the inner-wall pinch.")
parser.add_argument("--approach-opening", type=float, default=0.045,
                    help="R1 gripper-axis target while approaching; normally wider than --opening.")
parser.add_argument("--safe-approach", action="store_true",
                    help=("Diagnostic route: first move above and on the +Y side "
                          "of the casing before descending to the Hub."))
parser.add_argument("--orientation", choices=("radial", "radial_outward", "radial_flip", "radial_y90", "radial_y90_flip", "axial"), default="radial",
                    help="Radial bore-wall pinch, jaw-flip candidates, or axial inner-rim thickness pinch.")
parser.add_argument("--gripper-contact-offset", type=float, default=None,
                    help="Diagnostic robot collision contact offset override; production remains unchanged.")
parser.add_argument("--preopen", action="store_true",
                    help="Keep the radial jaw at the final opening while entering the bore.")
parser.add_argument("--disable-fixture-collisions", action="store_true",
                    help="Calibration-only: disable table/casing collision to isolate gripper-Hub contact.")
parser.add_argument("--disable-table-collisions", action="store_true",
                    help="Calibration-only: disable only the support-table collision.")
parser.add_argument("--disable-casing-collisions", action="store_true",
                    help="Calibration-only: disable only the casing collision.")
parser.add_argument("--filter-robot-casing-collisions", action="store_true",
                    help="Calibration-only: filter Robot↔Casing pairs while retaining Hub↔Casing contact.")
parser.add_argument("--filter-robot-casing-arm", choices=("left", "right"), default=None,
                    help="Calibration-only: filter only the named arm↔Casing pairs.")
parser.add_argument("--filter-robot-table-collisions", action="store_true",
                    help="Calibration-only: filter Robot↔Table pairs while retaining fixture/Hub contacts.")
parser.add_argument("--disable-casing-during-approach", action="store_true",
                    help=("Calibration-only: disable Casing collision while the robot "
                          "approaches, then re-enable it before closure."))
parser.add_argument("--gripper-only-collisions", action="store_true",
                    help="Calibration-only: keep left gripper collision meshes and disable other robot links.")
parser.add_argument("--collision-finger", choices=("both", "link1", "link2"), default="both",
                    help="With --gripper-only-collisions, retain both fingers or only one for contact isolation.")
parser.add_argument("--wrist-roll-deg", type=float, default=0.0,
                    help="Diagnostic additive left_arm_joint6 roll after reaching the radial waypoint.")
parser.add_argument("--insert-after-lift", action="store_true",
                    help=("After the corrected grasp/lift, move the still-dynamic Hub to "
                          "the calibrated Casing preinsert and seat targets, open the "
                          "gripper, and retract. This is an opt-in full M1 insertion diagnostic."))
parser.add_argument("--stop-after-lift", action="store_true",
                    help="With --insert-after-lift, stop after the real segmented lift and write grasp evidence only.")
parser.add_argument("--release-before-seat", action="store_true",
                    help=("With --insert-after-lift, release at the calibrated 0.22 m "
                          "preinsert height, retract the arm, and let the dynamic Hub "
                          "seat under gravity instead of pushing it down while clamped."))
parser.add_argument("--insert-safe-waypoint", action="store_true",
                    help=("With --insert-after-lift, route above the Casing before "
                          "descending to the preinsert height."))
parser.add_argument("--transport-clearance-z", type=float, default=0.55,
                    help=("Extra world-Z clearance above the Casing used by the "
                          "orthogonal transport route. Keep this just above the "
                          "fixture; it is not a pose write to the Hub."))
parser.add_argument("--transport-side-clearance-y", type=float, default=0.0,
                    help=("Additional signed Y clearance for the source-side "
                          "waypoint before crossing over the Casing. A positive "
                          "value keeps a Hub whose source is at +Y farther from "
                          "the Casing rim; zero preserves the legacy route."))
parser.add_argument("--transport-side-offset-x", type=float, default=0.0,
                    help=("Signed X offset of the source-side waypoint from the "
                          "Casing center. A negative value routes around the "
                          "Casing's left outer edge before crossing above it; "
                          "zero preserves the legacy center-X route."))
parser.add_argument("--transport-x-first", action="store_true",
                    help=("Translate to the source-side X waypoint at the "
                          "current lift height before raising and crossing "
                          "the Casing. This is useful when the high-Z IK "
                          "configuration cannot reach the side waypoint."))
parser.add_argument("--transport-direct", action="store_true",
                    help=("With the strict full-place route, move directly "
                          "from the measured lift state to the above-socket "
                          "waypoint. This is a bounded reachability/corner "
                          "ablation; it keeps the Hub dynamic and does not "
                          "disable Casing collision."))
parser.add_argument("--preserve-lift-grasp-frame", action="store_true",
                    help=("With --insert-after-lift, preserve the measured link6-to-Hub "
                          "translation and current link6 orientation from the lift "
                          "when commanding insertion. This is a physics diagnostic "
                          "for the observed grasp frame, not a pose write."))
parser.add_argument("--reanchor-at-insertion-above", action="store_true",
                    help=("At the measured above-socket waypoint, refresh the "
                          "link6-to-Hub grasp frame before the collision-sensitive "
                          "preinsert descent. This is ordinary measured feedback "
                          "and never writes the dynamic Hub pose."))
parser.add_argument("--correct-orientation-at-insertion-above", action="store_true",
                    help=("With controlled placement, rotate the dynamically held "
                          "Hub toward its authored orientation at the measured "
                          "high-Z waypoint before preinsert. The correction uses "
                          "ordinary incremental IK and no Hub pose write."))
parser.add_argument("--stop-after-preinsert", action="store_true",
                    help=("With --insert-after-lift, stop after the preinsert "
                          "waypoint and write alignment metrics without attempting "
                          "seat or release."))
parser.add_argument("--socket-center-y", type=float, default=0.08683718,
                    help="Calibrated Casing socket-center Y used by insertion diagnostics.")
parser.add_argument("--seat-depth-m", type=float, default=0.062,
                    help=("Hub-root Z offset above the Casing root used as the final "
                          "seat target. The canonical MagicAssembly value is 0.062 m; "
                          "a calibrated alternative must be recorded as a separate run."))
parser.add_argument("--preinsert-height-m", type=float, default=0.22,
                    help="Hub-root height above the Casing center at release; default is the canonical 0.22 m trial height.")
parser.add_argument("--preinsert-offset-x-m", type=float, default=0.0,
                    help=("Measured Cartesian X offset for the preinsert/release waypoint. "
                          "The seat target remains calibrated; this is a normal approach "
                          "offset used to compensate observed gravity-seat slip."))
parser.add_argument("--preinsert-offset-y-m", type=float, default=0.0,
                    help=("Measured Cartesian Y offset for the preinsert/release waypoint. "
                          "The seat target remains calibrated; this is a normal approach "
                          "offset used to compensate observed gravity-seat slip."))
parser.add_argument("--hub-gravity", action="store_true",
                    help="Enable gravity for the dynamic Hub in this insertion diagnostic.")
parser.add_argument("--release-opening", type=float, default=0.05,
                    help="Gripper target used for the release phase; default is the R1 maximum-open command used by the prior diagnostic.")
parser.add_argument("--release-open-steps", type=int, default=100,
                    help="Physics steps to hold the release opening before any lateral clearance motion (default: 100).")
parser.add_argument("--freeze-release-open", action="store_true",
                    help=("During the controlled-place opening dwell, hold the "
                          "measured arm joint targets while the gripper actuator "
                          "opens. This is an ordinary joint action that prevents "
                          "pose IK from changing branches during release; it "
                          "does not freeze or move the dynamic Hub."))
parser.add_argument("--post-release-settle-steps", type=int, default=0,
                    help=("After the normal upward retract, advance this many "
                          "additional physics steps before recording the final "
                          "retract pose. The Hub remains dynamic and the arms "
                          "remain clear; zero preserves the historical trace."))
parser.add_argument("--release-at-preinsert", action="store_true",
                    help=("With the full-gravity controlled-place path, open the real gripper at "
                          "the measured preinsert height and let PhysX seat the free Hub under "
                          "gravity before the ordinary upward retract. This is a physical "
                          "release/seat diagnostic; it never writes the Hub pose."))
parser.add_argument("--gripper-static-friction", type=float, default=4.0,
                    help="Strict-path Coulomb static friction for the two authored gripper tips.")
parser.add_argument("--gripper-dynamic-friction", type=float, default=3.0,
                    help="Strict-path Coulomb dynamic friction for the two authored gripper tips.")
parser.add_argument("--gripper-effort-limit", type=float, default=None,
                    help="Optional run-scoped R1 gripper actuator effort limit; unset preserves the robot USD config.")
parser.add_argument("--gripper-stiffness", type=float, default=None,
                    help="Optional run-scoped R1 gripper actuator stiffness; unset preserves the robot USD config.")
parser.add_argument("--gripper-damping", type=float, default=None,
                    help="Optional run-scoped R1 gripper actuator damping; unset preserves the robot USD config.")
parser.add_argument("--gripper-velocity-limit", type=float, default=None,
                    help="Optional run-scoped R1 gripper actuator velocity limit; unset preserves the robot USD config.")
parser.add_argument("--arm-effort-limit", type=float, default=None,
                    help="Optional run-scoped effort limit for R1 arm/eef actuators; unset preserves the robot USD config.")
parser.add_argument("--arm-stiffness", type=float, default=None,
                    help="Optional run-scoped stiffness for R1 arm/eef actuators; unset preserves the robot USD config.")
parser.add_argument("--arm-damping", type=float, default=None,
                    help="Optional run-scoped damping for R1 arm/eef actuators; unset preserves the robot USD config.")
parser.add_argument("--arm-velocity-limit", type=float, default=None,
                    help="Optional run-scoped velocity limit for R1 arm/eef actuators; unset preserves the robot USD config.")
parser.add_argument("--m1-contact-static-friction", type=float, default=0.45,
                    help="Run-scoped static friction for Hub↔Casing CAD contact; default is the nominal 0.45.")
parser.add_argument("--m1-contact-dynamic-friction", type=float, default=0.35,
                    help="Run-scoped dynamic friction for Hub↔Casing CAD contact; default is the nominal 0.35.")
parser.add_argument("--gravity-settle-steps", type=int, default=100,
                    help="With --release-before-seat, physics steps after gravity is enabled before measuring the free Hub (default: 100).")
parser.add_argument("--release-wrist-roll-deg", type=float, default=0.0,
                    help=("With --release-before-seat, apply this bounded left-wrist joint roll after opening "
                          "and before the contact-clearance check; zero preserves the straight release."))
parser.add_argument("--release-clearance-y", type=float, default=0.12,
                    help=("With --release-before-seat, move the opened tool by this world-Y clearance "
                          "while Hub gravity remains disabled, then enable gravity and let PhysX seat the free part."))
parser.add_argument("--release-clearance-x", type=float, default=0.0,
                    help=("With --release-before-seat, move the opened tool tangentially by this world-X "
                          "clearance before gravity is enabled. This can clear annular fingers without "
                          "pushing them farther through the socket."))
parser.add_argument("--release-clearance-z", type=float, default=0.0,
                    help=("Additional world-Z clearance for release; a positive value withdraws the inner finger "
                          "axially before gravity is enabled."))
parser.add_argument("--release-clearance-step-m", type=float, default=0.02,
                    help="Maximum Cartesian clearance increment used for the release withdrawal path (default: 0.02 m).")
parser.add_argument("--step-scale", type=float, default=1.0,
                    help=("Scale the diagnostic controller hold durations. 1.0 is the "
                          "calibrated trace; smaller values are only for fast waypoint "
                          "probes and are not success evidence."))
parser.add_argument("--episode-length-s", type=float, default=None,
                    help=("Override the Isaac episode timeout for long diagnostics. "
                          "The default scene limit is 60 s; complete place traces "
                          "should pass an explicit larger value."))
parser.add_argument("--closed-loop-preinsert-corrections", type=int, default=0,
                    help=("After the first preinsert waypoint, apply this many bounded "
                          "position corrections from the measured dynamic Hub root. "
                          "This remains an ordinary IK action; 0 preserves the original "
                          "open-loop diagnostic."))
parser.add_argument("--closed-loop-seat-corrections", type=int, default=0,
                    help=("After the seat waypoint, apply this many bounded position "
                          "corrections from the measured Hub root before release. "
                          "The object remains dynamic and clamped."))
parser.add_argument("--grasp-constraint", action="store_true",
                    help=("Enable the optional disabled-at-reset physical FixedJoint "
                          "after two-finger contact. This is a separate direct-control "
                          "grasp-abstraction baseline, not nominal contact-only evidence."))
parser.add_argument("--grasp-correction-x-deg", type=float, default=0.0,
                    help="Optional post-close world-X tilt correction applied before lift.")
parser.add_argument("--grasp-correction-y-deg", type=float, default=0.0,
                    help="Optional post-close world-Y tilt correction applied before lift.")
parser.add_argument("--closed-loop-insertion", action="store_true",
                    help=("Move the held dynamic Hub in short Cartesian segments, "
                          "updating link6-to-Hub offset and Hub orientation error "
                          "from each simulated state. This is a direct feedback "
                          "controller; it never writes the Hub pose."))
parser.add_argument("--fixed-ik-target", action="store_true",
                    help=("Compute one joint target per Cartesian waypoint and hold it "
                          "through the waypoint. This avoids repeatedly re-solving a "
                          "lagging pose target during a loaded grasp."))
parser.add_argument("--ik-position-only", action="store_true",
                    help=("Use the existing differential IK controller in position-only "
                          "mode for diagnostic transport; orientation remains measured "
                          "in the evaluator and is not silently assumed."))
parser.add_argument("--transport-position-only", action="store_true",
                    help=("Use position-only differential IK only for the high-Z "
                          "transport, preinsert registration, and controlled seat. "
                          "Use --seat-full-pose-ik to restore full pose IK for the "
                          "final seat phase."))
parser.add_argument("--torso-runtime-override", action="store_true",
                    help=("Diagnostic only: reset the R1 torso to the robot bundle's "
                          "nonzero workspace posture and widen its authored zero-width "
                          "joint limits around that posture for this run."))
parser.add_argument("--seat-full-pose-ik", action="store_true",
                    help=("With --transport-position-only, use full pose IK for the "
                          "controlled seat after the preinsert approach."))
parser.add_argument("--rotate-held-orientation", action="store_true",
                    help=("With --controlled-place, rotate the TCP toward the canonical Hub "
                          "orientation during transport. The default strict path holds the "
                          "measured grasp orientation to avoid twisting the frictional pinch."))
parser.add_argument("--correct-orientation-before-seat", action="store_true",
                    help=("With the strict closed-loop place route, keep the dynamic Hub at "
                          "the measured preinsert root and rotate the held assembly toward "
                          "its authored orientation before the final axial seat."))
parser.add_argument("--seat-yaw-correction-deg", type=float, default=0.0,
                    help=("Apply a bounded ordinary wrist/held-assembly yaw correction "
                          "at the measured preinsert root before the final seat. "
                          "The Hub remains dynamic; zero disables the ablation."))
parser.add_argument("--seat-yaw-correction-after-seat-deg", type=float, default=0.0,
                    help=("After the dynamic Hub reaches the seat waypoint, apply a "
                          "bounded per-segment ordinary wrist orientation correction "
                          "before release. "
                          "The Hub remains dynamic and its pose is never written."))
parser.add_argument("--reanchor-preinsert-hold", action="store_true",
                    help=("Before the short preinsert hold, re-issue the measured "
                          "TCP pose and refresh the grasp frame. This is an ordinary "
                          "IK hold for dynamic-contact stability."))
parser.add_argument("--freeze-preinsert-hold", action="store_true",
                    help=("During the preinsert settling window, hold the measured "
                          "arm joint positions instead of re-solving a Cartesian pose. "
                          "This preserves the real contact/grasp state without writing "
                          "the Hub pose."))
parser.add_argument("--preinsert-hold-steps", type=int, default=40,
                    help="Physics steps for the preinsert closed-grasp hold; zero skips it.")
parser.add_argument("--post-seat-hold-steps", type=int, default=0,
                    help=("After the controlled seat reaches the socket, hold the "
                          "measured arm joints with the gripper closed for this many "
                          "physics steps so real contact can settle the dynamic Hub; "
                          "zero preserves the calibrated trace."))
parser.add_argument("--controlled-seat-segments", type=int, default=0,
                    help="Optional waypoint-count override for the controlled seat only; zero uses --insertion-segments.")
parser.add_argument("--controlled-seat-segment-steps", type=int, default=0,
                    help="Optional hold-step override per controlled-seat waypoint; zero uses --insertion-segment-steps.")
parser.add_argument("--release-only-if-seated", action="store_true",
                    help=("For controlled placement, open the grippers only if the measured Hub has Casing contact, "
                          "is stable, and meets the existing relaxed placement pose tolerances."))
parser.add_argument("--reanchor-before-seat", action="store_true",
                    help=("Refresh the measured TCP-to-Hub grasp frame immediately "
                          "before the controlled seat descent."))
parser.add_argument("--follow-link-orientation", action="store_true",
                    help=("During closed-loop transport, use the current measured "
                          "link6 orientation for each waypoint instead of forcing "
                          "the original grasp quaternion."))
parser.add_argument("--adaptive-grasp-frame", action="store_true",
                    help=("During a loaded closed-loop move, recompute the measured "
                          "link6-to-Hub translation at each segment. This is an "
                          "explicit feedback ablation against the fixed grasp-frame "
                          "controller; it never writes either rigid-body pose."))
parser.add_argument("--insertion-segments", type=int, default=8,
                    help="Number of segments used by --closed-loop-insertion (default: 8).")
parser.add_argument("--insertion-segment-steps", type=int, default=40,
                    help="Controller hold steps per closed-loop insertion segment (default: 40).")
parser.add_argument("--held-orientation-step-deg", type=float, default=6.0,
                    help=("Maximum orientation correction per closed-loop segment when "
                          "--rotate-held-orientation is active (default: 6 degrees)."))
parser.add_argument("--lift-segments", type=int, default=8,
                    help="With --controlled-place, number of short held-lift segments (default: 8).")
parser.add_argument("--lift-segment-steps", type=int, default=60,
                    help="With --controlled-place, physics steps per held-lift segment (default: 60).")
parser.add_argument("--full-gravity", action="store_true",
                    help=("Strict Isaac rollout: keep the dynamic Hub under gravity from reset "
                          "through release; do not toggle gravity at a mid-air release boundary."))
parser.add_argument("--physical-supports", action="store_true",
                    help=("Spawn the kinematic staging pads under the annular Hub and the Casing "
                          "support. Required for the strict full-gravity rollout."))
parser.add_argument("--retract-staging-supports-after-grasp", action="store_true",
                    help=("After the real gripper contact closes, withdraw the four "
                          "kinematic staging pads; this changes only setup fixtures "
                          "and is recorded before the lift."))
parser.add_argument("--staging-support-drop-m", type=float, default=0.05,
                    help=("Downward clearance for the kinematic Hub staging pads "
                          "after grasp. The default 50 mm clears the pad thickness "
                          "without applying the prior 300 mm impulse."))
parser.add_argument("--staging-support-retract-steps", type=int, default=1,
                    help=("Number of small physics actions used to withdraw the "
                          "staging pads. Values >1 avoid a single kinematic jump."))
parser.add_argument("--staging-support-withdraw-mode", choices=("down", "lateral", "disable_collision"), default="down",
                    help=("How to withdraw staging pads after grasp. 'down' is the "
                          "legacy vertical baseline; 'lateral' moves each pad radially "
                          "outward at fixed height; 'disable_collision' removes only "
                          "the four pad collision APIs as a no-impulse ablation."))
parser.add_argument("--controlled-place", action="store_true",
                    help=("Keep the two fingers closed while descending into the socket, verify "
                          "contact/alignment, then open and retract. This is the strict physical "
                          "place path and is mutually exclusive with --release-before-seat."))
parser.add_argument("--seat-vertical-only", action="store_true",
                    help=("During the strict controlled seat, keep the measured Hub X/Y fixed "
                          "and descend only along Z. This isolates the axial insertion contact "
                          "from simultaneous lateral motion; the Hub remains dynamic."))
parser.add_argument("--seat-position-only", action="store_true",
                    help=("During the strict controlled seat only, use the existing position-only "
                          "DLS controller while leaving lift/transport pose IK unchanged."))
parser.add_argument("--seat-jacobian-position", action="store_true",
                    help=("During the strict controlled seat only, derive a small joint "
                          "position correction from the measured R1 position Jacobian instead "
                          "of asking pose IK to choose a new branch."))
parser.add_argument("--release-from-preplace", action="store_true",
                    help=("Placement-only baseline: after real contact and staging-pad retraction, "
                          "do not transport the Hub; open the gripper at the calibrated preplace "
                          "pose, withdraw the tool, and let the dynamic Hub seat under gravity. "
                          "This is not a pick-and-carry success claim."))
parser.add_argument("--preplace-controlled-place", action="store_true",
                    help=("Placement-only baseline: after real contact and staging-pad retraction, "
                          "use a short ordinary IK descent while keeping the real gripper closed, "
                          "then open and retract at the measured socket pose. This omits pick-and-carry "
                          "transport but tests a controlled physical place from preplace."))
parser.add_argument("--preplace-release-clearance-y", type=float, default=0.0,
                    help=("Optional world-Y clearance after release in controlled/preplace "
                          "paths; zero keeps the release at the measured seat TCP."))
parser.add_argument("--preplace-release-clearance-z", type=float, default=0.0,
                    help="Optional world-Z clearance after preplace release before the normal upward retract.")
parser.add_argument("--preplace-correct-orientation", action="store_true",
                    help="Apply the measured grasp-frame wrist rotation toward the canonical Hub orientation during preplace descent.")
parser.add_argument("--preplace-seat-extra-depth-m", type=float, default=0.0,
                    help="Additional closed-gripper axial command below the canonical seat target; the evaluated Hub pose is still the simulated state.")
parser.add_argument("--controlled-seat-extra-depth-m", type=float, default=0.0,
                    help=("Additional axial IK command below the canonical seat target for the "
                          "strict controlled-place descent. The canonical seat target remains "
                          "the evaluation reference; the dynamic Hub pose is never written."))
parser.add_argument("--dual-gripper", action="store_true",
                    help=("Use both real R1 grippers in mirrored inner/outer-wall pinches. "
                          "This is an optional physical fallback when one gripper cannot "
                          "carry the dynamic Hub without slip."))
parser.add_argument("--synchronous-dual-lift", action="store_true",
                    help=("With --dual-gripper, compute both arm IK targets before each "
                          "lift step and apply them together. This isolates sequential "
                          "dual-arm update torque from the physical grasp result."))
parser.add_argument("--synchronous-dual-transport", action="store_true",
                    help=("With --dual-gripper, compute both arm IK targets from the "
                          "same measured state and apply them in one physics step for "
                          "each transport/seat waypoint."))
parser.add_argument("--release-right-after-lift", action="store_true",
                    help=("With --dual-gripper, open and retract the right gripper after "
                          "the verified lift, then transport with the left gripper only. "
                          "This isolates dual-arm lift support from right-arm fixture "
                          "collision during the insertion route."))
parser.add_argument("--release-right-at-insertion-above", action="store_true",
                    help=("With --dual-gripper and --controlled-place, keep both arms "
                          "closed through the horizontal transport, then open and "
                          "withdraw the right arm at the collision-free above-socket "
                          "waypoint before the final preinsert descent."))
parser.add_argument("--release-right-at-transport-segment", type=int, default=0,
                    help=("With --dual-gripper and --controlled-place, release and "
                          "withdraw the right gripper immediately after the given "
                          "above-socket transport segment, then continue the route "
                          "with the measured left grasp. Zero disables this handoff. "
                          "This is a physical handoff ablation, not a pose/constraint "
                          "shortcut."))
parser.add_argument("--release-right-in-place", action="store_true",
                    help=("With --release-right-after-lift, open the right gripper "
                          "but leave its arm at the measured lift pose. This isolates "
                          "right-arm withdrawal dynamics from the left load-bearing "
                          "pinch; the open right tool remains a real rigid body."))
parser.add_argument("--freeze-right-release-open", action="store_true",
                    help=("With --release-right-after-lift, hold both measured arm "
                          "joint targets while the right gripper opens. This isolates "
                          "opening-actuator torque from the right-arm withdrawal "
                          "motion; the Hub remains dynamic and the open right tool "
                          "remains a real rigid body."))
parser.add_argument("--hold-right-release-pose", action="store_true",
                    help=("With --release-right-after-lift, hold both measured TCP "
                          "poses through their ordinary IK joint targets while the "
                          "right gripper opens. This preserves Cartesian arm pose "
                          "without freezing the raw loaded joints."))
parser.add_argument("--right-release-clearance-y-m", type=float, default=0.0,
                    help=("Optional signed lateral clearance for the right tool "
                          "after a transport-segment handoff. A nonzero value "
                          "moves the open right TCP along Y away from the Hub "
                          "instead of using the default vertical 0.20 m exit. "
                          "This is an ordinary measured arm motion."))
parser.add_argument("--correct-orientation-after-lift", action="store_true",
                    help=("With the dual-arm hybrid route, correct the held Hub's "
                          "orientation while it is still at the lift height, before "
                          "transport toward the Casing. The left gripper remains closed."))
parser.add_argument("--track-tip-contact", action="store_true",
                    help=("Record filtered contact points for both gripper links and classify "
                          "inner-wall versus outer-wall contact around the Hub."))
parser.add_argument("--strict-acceptance", action="store_true",
                    help=("Emit SUCCESS only when two-sided tip topology, seat contact, 6DoF "
                          "pose, bolt-hole alignment, release, and stable retract all pass."))
parser.add_argument("--place-acceptance", action="store_true",
                    help=("Use the relaxed physical-place acceptance: dynamic support, "
                          "load-bearing grasp/lift, finite transport into the socket, "
                          "and stable release/retract. It does not require a particular "
                          "finger-tip topology."))
parser.add_argument("--no-video", action="store_true")
parser.add_argument("--camera-update-stride", type=int, default=10,
                    help=("When cameras/video are enabled, update RTX sensors "
                          "every N environment steps and reuse the last frame "
                          "between updates; physics/actions are unchanged."))
parser.add_argument("--video-width", type=int, default=320,
                    help="RGB camera width for the live recording (default: 320).")
parser.add_argument("--video-height", type=int, default=240,
                    help="RGB camera height for the live recording (default: 240).")
parser.add_argument("--camera-probe", type=int, default=0,
                    help=("Camera-only smoke mode. Reset the real scene, advance this many "
                          "zero-action control steps, and write a synchronized camera MP4 "
                          "without running the long grasp/place controller."))
parser.add_argument("--render-interval", type=int, default=None,
                    help=("Override Isaac's render interval for video diagnostics. "
                          "This changes rendering cadence only; physics still runs "
                          "at the configured simulation dt."))
parser.add_argument("--m0-serve", action="store_true",
                    help=("Keep a blocked preplace episode alive for the REPAIR M0 "
                          "helper handoff. The worker waits for a command file, "
                          "moves only the registered blocker, and then continues "
                          "the same dynamic episode."))
parser.add_argument("--m0-help-timeout-s", type=float, default=900.0,
                    help="Maximum wall time to wait for the M0 helper command.")
parser.add_argument("--m0-blocked-attempt-steps", type=int, default=40,
                    help="Physics steps used for the real blocked descent before M0 help.")
parser.add_argument("--m0-precontact-guard", action="store_true",
                    help=("Stop at the registered physical blocker before contact and enter "
                          "the safe-hold handoff; use this when collision contact would "
                          "damage the calibrated grasp."))
parser.add_argument("--m0-help-settle-steps", type=int, default=50,
                    help="Physics settle steps after helper moves the blocker.")
parser.add_argument("--m0-help-poll-step-interval", type=int, default=1,
                    help="Poll iterations between safe-hold physics steps while awaiting help.")
parser.add_argument("--m0-hold-opening", type=float, default=0.005,
                    help="Gripper opening used during M0 safe hold and helper settle.")
parser.add_argument("--m0-defer-staging-support-retraction", action="store_true",
                    help="Keep physical staging supports under the Hub during the M0 handoff, then retract them after help.")
parser.add_argument("--m0-regrasp-steps", type=int, default=60,
                    help="Physics steps used to re-align the gripper on the supported Hub before support withdrawal.")
parser.add_argument("--m0-freeze-release-joints", action="store_true",
                    help="Hold the measured robot arm joints while opening for M0 release.")
parser.add_argument("--m0-retract-segments", type=int, default=1,
                    help=("For persistent M0 release, split the upward arm withdrawal "
                          "into this many measured short waypoints."))
parser.add_argument("--m0-retract-segment-steps", type=int, default=20,
                    help="Physics steps per segmented M0 withdrawal waypoint.")
parser.add_argument("--m0-retract-step-z-m", type=float, default=0.02,
                    help="World-Z increment for each segmented M0 withdrawal waypoint.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from hrc_m1.roco_adapter import RocoTaskAdapter  # noqa: E402
from hrc_m1.roco_env import make_env_classes  # noqa: E402


def _vec(value: torch.Tensor) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).detach().cpu().item())


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions without touching simulator state."""
    w, x, y, z = a.unbind(-1)
    W, X, Y, Z = b.unbind(-1)
    return torch.stack((w * W - x * X - y * Y - z * Z,
                        w * X + x * W + y * Z - z * Y,
                        w * Y - x * Z + y * W + z * X,
                        w * Z + x * Y - y * X + z * W), dim=-1)


def _qinv(value: torch.Tensor) -> torch.Tensor:
    norm = torch.sum(value * value, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1) / norm


def _qrotate(q: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(value[..., :1]), value), dim=-1)
    return _qmul(_qmul(q, pure), _qinv(q))[..., 1:]


def _quat_angle_deg(current: torch.Tensor, target: torch.Tensor) -> float:
    current = current / torch.linalg.vector_norm(current).clamp_min(1.0e-8)
    target = target / torch.linalg.vector_norm(target).clamp_min(1.0e-8)
    dot = torch.sum(current * target).abs().clamp(-1.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.arccos(dot)).detach().cpu().item())


def _roco_style_six_point_score(
    *,
    records: list[dict[str, object]],
    args: argparse.Namespace,
    candidate_verdict: str,
    insertion_metrics: dict[str, object],
) -> dict[str, object]:
    """Return a transparent six-point M1 completion score.

    Each point is binary and tied to a directly logged physical observation;
    a partial score is useful for debugging, while only 6/6 can qualify for a
    strict SUCCESS verdict.  The score is intentionally independent of any
    learned policy and therefore also applies to the current scripted oracle.
    """
    def numeric(value: object, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    reset_record = next((record for record in records if record.get("label") == "gravity_reset_settle"), records[0])
    reset_state = reset_record.get("hub_root_state", [float("nan")] * 13)
    support_point = bool(
        args.full_gravity
        and (args.physical_supports or args.full_gravity)
        and len(reset_state) >= 10
        and numeric(reset_record.get("hub_speed_mps"), 1.0) <= 0.01
    )

    topology_point = bool(insertion_metrics.get("tip_one_inner_one_outer", False))
    if not topology_point:
        for record in records:
            geometry = record.get("tip_contact_geometry", {})
            if not isinstance(geometry, dict):
                continue
            link1 = geometry.get("left_gripper_link1_contact", {})
            link2 = geometry.get("left_gripper_link2_contact", {})
            if not isinstance(link1, dict) or not isinstance(link2, dict):
                continue
            topology_point = topology_point or bool(
                (link1.get("inner_wall_candidate") and link2.get("outer_wall_candidate"))
                or (link1.get("outer_wall_candidate") and link2.get("inner_wall_candidate"))
            )
    force_names = ["left_gripper_link1_contact", "left_gripper_link2_contact"]
    if getattr(args, "dual_gripper", False):
        force_names.extend(["right_gripper_link1_contact", "right_gripper_link2_contact"])
    force_point = all(
        max(
            (
                numeric(record.get("gripper_to_hub_contact_force_norm_N", {}).get(name), 0.0)
                for record in records
                if isinstance(record.get("gripper_to_hub_contact_force_norm_N"), dict)
            ),
            default=0.0,
        ) > 1.0e-3
        for name in force_names
    )
    # Score the measured grasp itself even when the legacy candidate label
    # rejects object-follow; that distinction is exactly what a six-point
    # partial score is meant to expose.
    grasp_point = bool(topology_point and force_point)

    lift_record = next((record for record in reversed(records) if record.get("label") == "lift"), None)
    close_record = next(
        (record for record in reversed(records) if str(record.get("label", "")).startswith("close_")),
        None,
    )
    lift_delta = 0.0
    link_delta = 0.0
    if lift_record is not None and close_record is not None:
        lift_state = torch.tensor(lift_record.get("hub_root_state", [0.0, 0.0, 0.0])[:3])
        close_state = torch.tensor(close_record.get("hub_root_state", [0.0, 0.0, 0.0])[:3])
        lift_delta = _norm(lift_state - close_state)
        lift_link = torch.tensor(lift_record.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3])
        close_link = torch.tensor(close_record.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3])
        link_delta = _norm(lift_link - close_link)
    lift_point = bool(lift_delta >= 0.02 and lift_delta >= 0.5 * max(link_delta, 1.0e-6))

    transport_labels = (
        "insertion_above",
        "preinsert",
        "preinsert_grasp_hold",
    )
    transport_records = [
        record for record in records
        if str(record.get("label", "")).startswith(("closed_loop_rise", "closed_loop_side_above", "closed_loop_above", "closed_loop_preinsert"))
        or record.get("label") in transport_labels
    ]
    finite_transport = True
    contact_transport = 0
    for record in transport_records:
        state = record.get("hub_root_state", [])
        finite_transport = finite_transport and len(state) >= 10 and all(
            math.isfinite(float(value)) for value in state[:10]
        )
        finite_transport = finite_transport and numeric(record.get("hub_speed_mps"), 999.0) <= 1.0
        contact = record.get("gripper_to_hub_contact_force_norm_N", {})
        # A hybrid dual-arm route deliberately releases the right gripper
        # after the supported lift.  After that event, zero right contact is
        # the expected evidence rather than a transport failure; score the
        # active left load-bearing pair for those samples.  Before the event,
        # a true dual-arm route still requires all four contacts.
        transport_names = ["left_gripper_link1_contact", "left_gripper_link2_contact"]
        if getattr(args, "dual_gripper", False) and not getattr(args, "release_right_after_lift", False):
            transport_names.extend(["right_gripper_link1_contact", "right_gripper_link2_contact"])
        if isinstance(contact, dict) and all(
            numeric(contact.get(name), 0.0) > 1.0e-3
            for name in transport_names
        ):
            contact_transport += 1
    transport_point = bool(
        transport_records
        and finite_transport
        and contact_transport >= max(1, math.ceil(0.7 * len(transport_records)))
    )

    seat_point = bool(
        numeric(insertion_metrics.get("insert_hub_casing_force_norm_N"), 0.0) > 1.0e-3
        and numeric(insertion_metrics.get("radial_error_m"), 999.0) <= 0.004
        and numeric(insertion_metrics.get("axial_error_m"), 999.0) <= 0.008
        and numeric(insertion_metrics.get("orientation_error_deg"), 999.0) <= 2.0
        and numeric(insertion_metrics.get("bolt_hole_alignment_max_error_m"), 999.0) <= 0.003
    )
    release_point = bool(
        bool(insertion_metrics.get("release_performed", True))
        and seat_point
        and numeric(insertion_metrics.get("release_drift_m"), 999.0) <= 0.005
        and numeric(insertion_metrics.get("retract_drift_m"), 999.0) <= 0.02
        and bool(insertion_metrics.get("strict_checks", {}).get("release_contact_free", False))
    )
    components = {
        "1_supported_dynamic_reset": support_point,
        "2_one_inner_one_outer_grasp": grasp_point,
        "3_lift_follows_gripper": lift_point,
        "4_collision_free_transport": transport_point,
        "5_seated_6dof_and_bolt_alignment": seat_point,
        "6_release_and_stable_retract": release_point,
    }
    return {
        "scheme": "M1_ROCO_STYLE_6_POINT_V1",
        "total": int(sum(bool(value) for value in components.values())),
        "max": 6,
        "components": {name: int(bool(value)) for name, value in components.items()},
        "evidence": {
            "lift_delta_m": lift_delta,
            "link6_delta_m": link_delta,
            "transport_samples": len(transport_records),
            "transport_contact_samples": contact_transport,
            "transport_active_grippers": (
                "left_only_after_right_release"
                if getattr(args, "dual_gripper", False) and getattr(args, "release_right_after_lift", False)
                else "left_and_right"
            ),
            "candidate_verdict": candidate_verdict,
        },
        "interpretation": "6/6 is required for strict SUCCESS; partial points identify the first failed physical stage.",
    }


def _physical_place_four_point_score(
    *,
    records: list[dict[str, object]],
    args: argparse.Namespace,
    insertion_metrics: dict[str, object],
) -> dict[str, object]:
    """Score the requested outcome without requiring a particular fingertip topology."""
    def numeric(value: object, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    reset = next((r for r in records if r.get("label") == "gravity_reset_settle"), records[0])
    supported = bool(
        args.full_gravity
        and (args.physical_supports or args.full_gravity)
        and numeric(reset.get("hub_speed_mps"), 999.0) <= 0.01
    )
    close = next((r for r in reversed(records) if str(r.get("label", "")).startswith("close_")), None)
    lift = next((r for r in reversed(records) if r.get("label") == "lift"), None)
    contact_names = ["left_gripper_link1_contact", "left_gripper_link2_contact"]
    if args.dual_gripper:
        contact_names.extend(["right_gripper_link1_contact", "right_gripper_link2_contact"])
    peak_contacts = {
        name: max(
            (numeric(r.get("gripper_to_hub_contact_force_norm_N", {}).get(name), 0.0)
             for r in records if isinstance(r.get("gripper_to_hub_contact_force_norm_N"), dict)),
            default=0.0,
        )
        for name in contact_names
    }
    active_contact_count = sum(value > 1.0e-3 for value in peak_contacts.values())
    lift_delta = 0.0
    link_delta = 0.0
    if close is not None and lift is not None:
        lift_delta = _norm(torch.tensor(lift.get("hub_root_state", [0.0, 0.0, 0.0])[:3])
                           - torch.tensor(close.get("hub_root_state", [0.0, 0.0, 0.0])[:3]))
        link_delta = _norm(torch.tensor(lift.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3])
                           - torch.tensor(close.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3]))
    grasp_and_lift = bool(
        active_contact_count >= 2
        and lift_delta >= 0.03
        and lift_delta >= 0.4 * max(link_delta, 1.0e-6)
    )
    if getattr(args, "release_from_preplace", False) or getattr(args, "preplace_controlled_place", False):
        # This is deliberately a separate score: it proves a real-gravity
        # placement from a measured, load-bearing preplace state, but it does
        # not claim the omitted pick-and-carry transport stage.
        insert_state = insertion_metrics.get("insert_hub_root_m", [])
        finite_insert = bool(
            isinstance(insert_state, list)
            and len(insert_state) >= 3
            and all(math.isfinite(float(value)) for value in insert_state[:3])
        )
        seated = bool(
            finite_insert
            and numeric(insertion_metrics.get("insert_hub_casing_force_norm_N"), 0.0) > 1.0e-3
            and numeric(insertion_metrics.get("radial_error_m"), 999.0) <= 0.012
            and numeric(insertion_metrics.get("axial_error_m"), 999.0) <= 0.015
            and numeric(insertion_metrics.get("orientation_error_deg"), 999.0) <= 5.0
            and numeric(insertion_metrics.get("bolt_hole_alignment_max_error_m"), 999.0) <= 0.010
        )
        release_contact_free = bool(
            insertion_metrics.get("strict_checks", {}).get("release_contact_free", False)
        )
        stable_retract = bool(
            seated
            and numeric(insertion_metrics.get("post_settle_retract_drift_m"), 999.0) <= 0.03
            and release_contact_free
        )
        components = {
            "1_dynamic_supported_preplace": supported,
            "2_real_contact_before_release": active_contact_count >= 2,
            "3_gravity_seated_6dof_and_bolt_alignment": seated,
            "4_release_and_stable_retract": stable_retract,
        }
        return {
            "scheme": (
                "M1_PHYSICAL_PREPLACE_CONTROLLED_4_POINT_V1"
                if getattr(args, "preplace_controlled_place", False)
                else "M1_PHYSICAL_PREPLACE_4_POINT_V1"
            ),
            "total": int(sum(bool(value) for value in components.values())),
            "max": 4,
            "components": {name: int(bool(value)) for name, value in components.items()},
            "evidence": {
                "peak_contact_force_N": peak_contacts,
                "active_contact_count": active_contact_count,
                "gravity_seat_motion_m": numeric(insertion_metrics.get("release_to_insert_motion_m"), 999.0),
                "post_settle_retract_drift_m": numeric(insertion_metrics.get("post_settle_retract_drift_m"), 999.0),
            },
            "interpretation": (
                "4/4 proves physical controlled placement from a preplace state; "
                "it is not full pick-and-carry evidence."
                if getattr(args, "preplace_controlled_place", False)
                else "4/4 proves gravity-supported placement from a preplace state; it is not full pick-and-carry evidence."
            ),
        }
    transport_records = [
        r for r in records
        if str(r.get("label", "")).startswith((
            "closed_loop_rise", "closed_loop_side_above", "closed_loop_above",
            "closed_loop_direct_above", "closed_loop_preinsert"
        )) or r.get("label") in {"insertion_above", "preinsert", "preinsert_grasp_hold"}
    ]
    finite = bool(transport_records)
    contact_samples = 0
    for record in transport_records:
        state = record.get("hub_root_state", [])
        finite = finite and len(state) >= 10 and all(math.isfinite(float(v)) for v in state[:10])
        finite = finite and numeric(record.get("hub_speed_mps"), 999.0) <= 2.0
        forces = record.get("gripper_to_hub_contact_force_norm_N", {})
        if isinstance(forces, dict) and sum(numeric(forces.get(name), 0.0) > 1.0e-3 for name in contact_names) >= 2:
            contact_samples += 1
    seat = bool(
        finite
        and contact_samples >= max(1, math.ceil(0.5 * len(transport_records)))
        and numeric(insertion_metrics.get("insert_hub_casing_force_norm_N"), 0.0) > 1.0e-3
        and numeric(insertion_metrics.get("radial_error_m"), 999.0) <= 0.012
        and numeric(insertion_metrics.get("axial_error_m"), 999.0) <= 0.015
        and numeric(insertion_metrics.get("orientation_error_deg"), 999.0) <= 5.0
        and numeric(insertion_metrics.get("bolt_hole_alignment_max_error_m"), 999.0) <= 0.010
    )
    release = bool(
        bool(insertion_metrics.get("release_performed", True))
        and seat
        and numeric(insertion_metrics.get("release_drift_m"), 999.0) <= 0.012
        and numeric(insertion_metrics.get("retract_drift_m"), 999.0) <= 0.03
        and bool(insertion_metrics.get("strict_checks", {}).get("release_contact_free", False))
    )
    components = {
        "1_dynamic_supported_reset": supported,
        "2_load_bearing_grasp_and_lift": grasp_and_lift,
        "3_finite_transport_and_socket_seat": seat,
        "4_release_and_stable_retract": release,
    }
    return {
        "scheme": "M1_PHYSICAL_PLACE_4_POINT_V1",
        "total": int(sum(bool(v) for v in components.values())),
        "max": 4,
        "components": {name: int(bool(v)) for name, v in components.items()},
        "evidence": {
            "peak_contact_force_N": peak_contacts,
            "active_contact_count": active_contact_count,
            "lift_delta_m": lift_delta,
            "link6_delta_m": link_delta,
            "transport_samples": len(transport_records),
            "transport_contact_samples": contact_samples,
        },
        "interpretation": "4/4 is required for physical-place SUCCESS; no fingertip topology label is required.",
    }


def _post_release_placement_score(
    *,
    records: list[dict[str, object]],
    args: argparse.Namespace,
    insertion_metrics: dict[str, object],
) -> dict[str, object]:
    """Score the measured *final released* placement separately from strict M1.

    This is intentionally not a replacement for either the strict six-point
    score or the pre-release physical-place score.  A dynamic annular part can
    settle a few millimetres after the ``insert`` snapshot; this score records
    that useful outcome with explicit, relaxed placement tolerances.  It must
    never be interpreted as bolt-ready assembly or as evidence of a learned
    policy.
    """
    def numeric(value: object, default: float = 0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else default

    reset = next((r for r in records if r.get("label") == "gravity_reset_settle"), records[0])
    reset_state = reset.get("hub_root_state", [])
    dynamic_reset = bool(
        args.full_gravity
        and len(reset_state) >= 10
        and numeric(reset.get("hub_speed_mps"), 999.0) <= 0.01
    )

    close = next((r for r in reversed(records) if str(r.get("label", "")).startswith("close_")), None)
    lift = next((r for r in reversed(records) if r.get("label") == "lift"), None)
    lift_delta = 0.0
    link_delta = 0.0
    if close is not None and lift is not None:
        lift_delta = _norm(
            torch.tensor(lift.get("hub_root_state", [0.0, 0.0, 0.0])[:3])
            - torch.tensor(close.get("hub_root_state", [0.0, 0.0, 0.0])[:3])
        )
        link_delta = _norm(
            torch.tensor(lift.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3])
            - torch.tensor(close.get("bodies", {}).get("left_arm_link6", [0.0, 0.0, 0.0])[:3])
        )
    grasp_lift = bool(lift_delta >= 0.03 and lift_delta >= 0.4 * max(link_delta, 1.0e-6))

    transport_records = [
        r for r in records
        if str(r.get("label", "")).startswith((
            "closed_loop_rise", "closed_loop_side_above", "closed_loop_above", "closed_loop_preinsert"
        )) or r.get("label") in {"insertion_above", "preinsert", "preinsert_grasp_hold"}
    ]
    finite_transport = bool(transport_records)
    transport_contact_samples = 0
    for record in transport_records:
        state = record.get("hub_root_state", [])
        finite_transport = finite_transport and len(state) >= 10 and all(
            math.isfinite(float(value)) for value in state[:10]
        )
        finite_transport = finite_transport and numeric(record.get("hub_speed_mps"), 999.0) <= 2.0
        forces = record.get("gripper_to_hub_contact_force_norm_N", {})
        if isinstance(forces, dict) and sum(
            numeric(forces.get(name), 0.0) > 1.0e-3
            for name in ("left_gripper_link1_contact", "left_gripper_link2_contact")
        ) >= 2:
            transport_contact_samples += 1
    grasp_and_transport = bool(
        grasp_lift
        and finite_transport
        and transport_contact_samples >= max(1, math.ceil(0.5 * len(transport_records)))
    )

    # These thresholds are deliberately wider than strict M1 (4 mm / 8 mm /
    # 2 deg / 3 mm).  They are a placement diagnostic only: the relaxed score
    # says the part ended on the socket after release, not that bolts can be
    # inserted without further alignment.
    final_pose = bool(
        numeric(insertion_metrics.get("final_radial_error_m"), 999.0) <= 0.012
        and numeric(insertion_metrics.get("final_axial_error_m"), 999.0) <= 0.015
        and numeric(insertion_metrics.get("final_orientation_error_deg"), 999.0) <= 5.0
        and numeric(insertion_metrics.get("final_bolt_hole_alignment_max_error_m"), 999.0) <= 0.010
        and numeric(insertion_metrics.get("final_hub_casing_force_norm_N"), 0.0) > 1.0e-3
    )
    retract = next((r for r in reversed(records) if r.get("label") == "retract"), None)
    retract_forces = retract.get("gripper_to_hub_contact_force_norm_N", {}) if retract else {}
    retract_contact_free = bool(
        isinstance(retract_forces, dict)
        and all(numeric(retract_forces.get(name), 0.0) <= 1.0e-3 for name in (
            "left_gripper_link1_contact", "left_gripper_link2_contact",
            "right_gripper_link1_contact", "right_gripper_link2_contact",
        ))
        and numeric(retract.get("hub_speed_mps") if retract else None, 999.0) <= 0.05
        and numeric(insertion_metrics.get("post_settle_retract_drift_m"), 999.0) <= 0.03
    )
    components = {
        "1_dynamic_supported_reset": dynamic_reset,
        "2_load_bearing_grasp_lift_and_transport": grasp_and_transport,
        "3_final_gravity_seated_placement": final_pose,
        "4_contact_free_stable_retract": retract_contact_free,
    }
    return {
        "scheme": "M1_POST_RELEASE_PLACEMENT_4_POINT_V1",
        "total": int(sum(bool(value) for value in components.values())),
        "max": 4,
        "components": {name: int(bool(value)) for name, value in components.items()},
        "thresholds": {
            "final_radial_error_m": 0.012,
            "final_axial_error_m": 0.015,
            "final_orientation_error_deg": 5.0,
            "final_bolt_hole_alignment_max_error_m": 0.010,
            "post_settle_retract_drift_m": 0.030,
        },
        "evidence": {
            "lift_delta_m": lift_delta,
            "link6_delta_m": link_delta,
            "transport_samples": len(transport_records),
            "transport_contact_samples": transport_contact_samples,
            "final_radial_error_m": numeric(insertion_metrics.get("final_radial_error_m"), 999.0),
            "final_axial_error_m": numeric(insertion_metrics.get("final_axial_error_m"), 999.0),
            "final_orientation_error_deg": numeric(insertion_metrics.get("final_orientation_error_deg"), 999.0),
            "final_bolt_hole_alignment_max_error_m": numeric(
                insertion_metrics.get("final_bolt_hole_alignment_max_error_m"), 999.0
            ),
            "post_settle_retract_drift_m": numeric(
                insertion_metrics.get("post_settle_retract_drift_m"), 999.0
            ),
        },
        "interpretation": (
            "4/4 means the dynamic part ended on the socket after release within "
            "the relaxed placement tolerances; it is not strict M1 success, "
            "not bolt-ready alignment, and not learned-policy evidence."
        ),
    }


def main() -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_cls, env_cls = make_env_classes()
    cfg = cfg_cls()
    cfg.torso_runtime_override = bool(args.torso_runtime_override)
    # Keep actuator calibration run-scoped.  The default leaves the imported
    # R1 USD configuration untouched; an explicit override is recorded in the
    # metrics and is useful for separating insufficient jaw effort from bad
    # contact geometry without editing the robot asset.
    gripper_actuators = [
        actuator
        for name, actuator in cfg.robot_cfg.actuators.items()
        if "gripper" in str(name).lower()
    ]
    for actuator in gripper_actuators:
        if args.gripper_effort_limit is not None:
            actuator.effort_limit_sim = float(args.gripper_effort_limit)
        if args.gripper_stiffness is not None:
            actuator.stiffness = float(args.gripper_stiffness)
        if args.gripper_damping is not None:
            actuator.damping = float(args.gripper_damping)
        if args.gripper_velocity_limit is not None:
            actuator.velocity_limit_sim = float(args.gripper_velocity_limit)
    arm_actuators = [
        actuator
        for name, actuator in cfg.robot_cfg.actuators.items()
        if "gripper" not in str(name).lower()
        and ("arm" in str(name).lower() or "eef" in str(name).lower())
    ]
    for actuator in arm_actuators:
        if args.arm_effort_limit is not None:
            actuator.effort_limit_sim = float(args.arm_effort_limit)
        if args.arm_stiffness is not None:
            actuator.stiffness = float(args.arm_stiffness)
        if args.arm_damping is not None:
            actuator.damping = float(args.arm_damping)
        if args.arm_velocity_limit is not None:
            actuator.velocity_limit_sim = float(args.arm_velocity_limit)
    if hasattr(cfg, "seed"):
        cfg.seed = int(args.seed)
    if args.episode_length_s is not None:
        if args.episode_length_s <= 0.0:
            raise ValueError("--episode-length-s must be positive")
        cfg.episode_length_s = float(args.episode_length_s)
    if any(value is not None for value in (args.hub_reset_y, args.hub_reset_x, args.hub_reset_z)):
        reset_pos = (
            float(args.hub_reset_x) if args.hub_reset_x is not None else 0.30,
            float(args.hub_reset_y) if args.hub_reset_y is not None else 0.45,
            float(args.hub_reset_z) if args.hub_reset_z is not None else 1.08,
        )
        # ``hub_cfg`` is constructed in the config class body, so changing the
        # convenience field alone would not alter its already-created initial
        # state.  Update the actual reset record before the scene is built.
        cfg.hub_reset_pos = reset_pos
        cfg.hub_cfg.init_state.pos = reset_pos
        # Keep the annular staging pads under the reset cover when a diagnostic
        # explicitly changes its Y position.
        if args.physical_supports or args.full_gravity:
            y = float(reset_pos[1])
            x = float(reset_pos[0])
            support_z = float(reset_pos[2]) - 0.019
            cfg.hub_support_n_cfg.init_state.pos = (x, y + 0.10, support_z)
            cfg.hub_support_s_cfg.init_state.pos = (x, y - 0.10, support_z)
            cfg.hub_support_e_cfg.init_state.pos = (x + 0.10, y, support_z)
            cfg.hub_support_w_cfg.init_state.pos = (x - 0.10, y, support_z)
    if args.scatter_reset:
        if not args.full_gravity:
            raise ValueError("--scatter-reset requires --full-gravity")
        if args.table_size_x <= 0.0 or args.table_size_y <= 0.0:
            raise ValueError("scatter table dimensions must be positive")
        if args.scatter_spawn_margin_m < 0.0:
            raise ValueError("--scatter-spawn-margin-m must be non-negative")
        if args.scatter_physical_supports and args.scatter_support_top_z <= args.table_top_z:
            raise ValueError("--scatter-support-top-z must be above --table-top-z")
        scatter_hub_support_top_z = (
            float(args.scatter_hub_support_top_z)
            if args.scatter_hub_support_top_z is not None
            else float(args.scatter_support_top_z)
        )
        if args.scatter_physical_supports and scatter_hub_support_top_z <= args.table_top_z:
            raise ValueError("--scatter-hub-support-top-z must be above --table-top-z")
        cfg.scatter_reset = True
        cfg.table_size_xy = (float(args.table_size_x), float(args.table_size_y))
        cfg.table_top_z = float(args.table_top_z)
        cfg.scatter_spawn_margin_m = float(args.scatter_spawn_margin_m)
        cfg.table_cfg.spawn.size = (float(args.table_size_x), float(args.table_size_y), 0.10)
        cfg.table_center_xy = (float(args.table_center_x), float(args.table_center_y))
        cfg.scatter_support_top_z = (
            float(args.scatter_support_top_z)
            if args.scatter_physical_supports
            else float(args.table_top_z)
        )
        cfg.scatter_hub_support_top_z = (
            scatter_hub_support_top_z if args.scatter_physical_supports else float(args.table_top_z)
        )
        if args.scatter_physical_supports:
            support_margin = float(args.scatter_spawn_margin_m)
            hub_support_height = scatter_hub_support_top_z + support_margin - float(args.table_top_z)
            casing_support_height = float(args.scatter_support_top_z) + support_margin - float(args.table_top_z)
            for name in ("n", "s"):
                support_cfg = getattr(cfg, f"hub_support_{name}_cfg")
                support_cfg.spawn.size = (0.060, 0.020, hub_support_height)
            for name in ("e", "w"):
                support_cfg = getattr(cfg, f"hub_support_{name}_cfg")
                support_cfg.spawn.size = (0.020, 0.060, hub_support_height)
            cfg.casing_support_cfg.spawn.size = (0.50, 0.60, casing_support_height)
        cfg.table_cfg.init_state.pos = (float(args.table_center_x), float(args.table_center_y), float(args.table_top_z) - 0.05)
        cfg.casing_reset_pos = (float(args.casing_reset_x), float(args.casing_reset_y), 1.0)
        cfg.casing_cfg.init_state.pos = cfg.casing_reset_pos
        # The scatter route deliberately starts from a dynamic Casing as well
        # as a dynamic Hub.  When requested, the low tabletop is supplemented
        # by explicit kinematic staging supports whose contact surfaces are
        # measured in ``prepare_scatter_reset``.
        cfg.casing_cfg.spawn.rigid_props.kinematic_enabled = False
        cfg.casing_cfg.spawn.rigid_props.disable_gravity = False
        scatter_hub_pos = (
            float(args.hub_reset_x) if args.hub_reset_x is not None else 0.55,
            float(args.hub_reset_y) if args.hub_reset_y is not None else 0.43,
            float(args.hub_reset_z) if args.hub_reset_z is not None else 1.08,
        )
        cfg.hub_reset_pos = scatter_hub_pos
        cfg.hub_cfg.init_state.pos = scatter_hub_pos
    if args.gripper_contact_offset is not None:
        # The pinned R1 USD defaults to a 50 mm contact offset.  A diagnostic
        # override lets us separate that robot-level speculative contact from
        # the Hub's actual inner-wall geometry without changing the task scene.
        cfg.robot_cfg.spawn.collision_props.contact_offset = float(args.gripper_contact_offset)
        cfg.robot_cfg.spawn.collision_props.rest_offset = 0.0
    if args.hub_mass_kg is not None:
        if args.hub_mass_kg <= 0.0:
            raise ValueError("--hub-mass-kg must be positive")
        # The USD config is instantiated in the config class body; update its
        # authored PhysX mass before scene construction, never the live root
        # state.  This keeps mass calibration explicit and reproducible.
        cfg.hub_cfg.spawn.mass_props.mass = float(args.hub_mass_kg)
    # Keep cameras in the production/video path.  Physics-only diagnostics
    # omit camera prims entirely so Isaac can run without --enable_cameras and
    # without RTX rendering overhead.
    cfg.update_cameras = not args.no_video
    cfg.spawn_cameras = not args.no_video
    cfg.camera_update_stride = max(1, int(args.camera_update_stride))
    if args.fixed_ik_target:
        cfg.recompute_pose_ik_each_step = False
    if args.ik_position_only:
        cfg.ik_position_only = True
    if not args.no_video:
        if args.video_width < 2 or args.video_height < 2:
            raise ValueError("camera resolution must be at least 2x2")
        for camera_name in (
            "head_camera_cfg", "overhead_camera_cfg",
            "left_hand_camera_cfg", "right_hand_camera_cfg",
        ):
            camera_cfg = getattr(cfg, camera_name)
            setattr(cfg, camera_name, camera_cfg.replace(width=int(args.video_width), height=int(args.video_height)))
    if args.render_interval is not None:
        cfg.sim.render_interval = max(1, int(args.render_interval))
    cfg.spawn_physical_supports = bool(
        (args.physical_supports or args.full_gravity)
        and (not args.scatter_reset or args.scatter_physical_supports)
    )
    cfg.enable_right_contact_sensors = bool(
        args.dual_gripper or (args.track_tip_contact or args.strict_acceptance)
    )
    if args.full_gravity:
        # This is set before env construction so PhysX authors the dynamic Hub
        # with gravity enabled from the first integration step.  The strict
        # path never calls set_hub_gravity(False) or writes a Hub pose.
        cfg.hub_cfg.spawn.rigid_props.disable_gravity = False
        # The user's requested one-inner/one-outer tip grasp is modeled with
        # a high-friction contact material, still solved by PhysX and still
        # observable through the filtered contact forces/points.  This avoids
        # the legacy steel-on-steel coefficient losing the 5.7 kg cover during
        # a lateral transfer without introducing a fixed joint.
        cfg.gripper_contact_static_friction = float(args.gripper_static_friction)
        cfg.gripper_contact_dynamic_friction = float(args.gripper_dynamic_friction)
        if args.scatter_reset:
            # Both task parts are dynamic from scene construction.  This is a
            # hard invariant for the scattered full-physics fixture.
            cfg.casing_cfg.spawn.rigid_props.kinematic_enabled = False
            cfg.casing_cfg.spawn.rigid_props.disable_gravity = False
    cfg.m1_contact_static_friction = float(args.m1_contact_static_friction)
    cfg.m1_contact_dynamic_friction = float(args.m1_contact_dynamic_friction)
    if args.track_tip_contact or args.strict_acceptance:
        cfg.left_gripper_link1_contact_cfg.track_contact_points = True
        cfg.left_gripper_link2_contact_cfg.track_contact_points = True
        # Keep the mirrored gripper on force-only reporting for relaxed place
        # runs. Strict dual-arm acceptance, however, needs the same point-level
        # inner/outer evidence on both arms; enable those streams only for the
        # explicit strict dual configuration rather than silently scoring a
        # missing right topology as a failed grasp.
        right_track_points = bool(args.dual_gripper and args.strict_acceptance)
        cfg.right_gripper_link1_contact_cfg.track_contact_points = right_track_points
        cfg.right_gripper_link2_contact_cfg.track_contact_points = right_track_points
        cfg.left_gripper_link1_contact_cfg.max_contact_data_count_per_prim = 512
        cfg.left_gripper_link2_contact_cfg.max_contact_data_count_per_prim = 512
        if right_track_points:
            cfg.right_gripper_link1_contact_cfg.max_contact_data_count_per_prim = 512
            cfg.right_gripper_link2_contact_cfg.max_contact_data_count_per_prim = 512
    if args.controlled_place and args.release_before_seat:
        raise ValueError("--controlled-place and --release-before-seat are mutually exclusive")
    if args.controlled_place and not args.full_gravity:
        raise ValueError("--controlled-place requires --full-gravity")
    if args.release_only_if_seated and (
        not args.controlled_place or not args.insert_after_lift or args.release_at_preinsert
    ):
        raise ValueError(
            "--release-only-if-seated requires the full controlled-place route and cannot be combined with --release-at-preinsert"
        )
    if (args.release_right_after_lift or args.release_right_at_insertion_above) and not args.dual_gripper:
        raise ValueError("right-arm release options require --dual-gripper")
    if args.release_right_at_insertion_above and args.release_right_after_lift:
        raise ValueError("choose one right-arm release boundary")
    if args.release_right_at_transport_segment < 0:
        raise ValueError("--release-right-at-transport-segment must be non-negative")
    if args.release_right_at_transport_segment and not args.dual_gripper:
        raise ValueError("transport-segment right release requires --dual-gripper")
    if args.release_right_at_transport_segment and not args.controlled_place:
        raise ValueError("transport-segment right release requires --controlled-place")
    if args.release_right_at_transport_segment and (
        args.release_right_after_lift or args.release_right_at_insertion_above
    ):
        raise ValueError("choose one right-arm release boundary")
    if args.release_right_in_place and not (
        args.release_right_after_lift or args.release_right_at_transport_segment
    ):
        raise ValueError(
            "--release-right-in-place requires a right-arm release boundary"
        )
    if args.freeze_right_release_open and not args.release_right_after_lift:
        raise ValueError(
            "--freeze-right-release-open requires --release-right-after-lift"
        )
    if args.hold_right_release_pose and not args.release_right_after_lift:
        raise ValueError(
            "--hold-right-release-pose requires --release-right-after-lift"
        )
    if args.freeze_right_release_open and args.hold_right_release_pose:
        raise ValueError(
            "choose one right release opening hold mode"
        )
    if args.correct_orientation_after_lift and not args.controlled_place:
        raise ValueError("--correct-orientation-after-lift requires --controlled-place")
    if args.correct_orientation_at_insertion_above and not args.controlled_place:
        raise ValueError("--correct-orientation-at-insertion-above requires --controlled-place")
    if args.release_from_preplace and args.controlled_place:
        raise ValueError("--release-from-preplace is mutually exclusive with --controlled-place")
    if args.release_from_preplace and not args.full_gravity:
        raise ValueError("--release-from-preplace requires --full-gravity")
    if args.release_from_preplace and not args.insert_after_lift:
        raise ValueError("--release-from-preplace requires --insert-after-lift for placement metrics")
    if args.preplace_controlled_place and (args.controlled_place or args.release_from_preplace):
        raise ValueError("--preplace-controlled-place is mutually exclusive with the other place modes")
    if args.preplace_controlled_place and not args.full_gravity:
        raise ValueError("--preplace-controlled-place requires --full-gravity")
    if args.preplace_controlled_place and not args.insert_after_lift:
        raise ValueError("--preplace-controlled-place requires --insert-after-lift for placement metrics")
    if args.full_gravity and args.release_before_seat:
        raise ValueError("--full-gravity cannot use the mid-air --release-before-seat diagnostic")
    cfg.spawn_grasp_constraint = bool(args.grasp_constraint)
    if args.disable_fixture_collisions:
        cfg.disable_fixture_collisions = True
    if args.disable_table_collisions:
        cfg.disable_table_collisions = True
    if args.disable_casing_collisions:
        cfg.disable_casing_collisions = True
    if args.filter_robot_casing_collisions or args.filter_robot_casing_arm:
        cfg.filter_robot_casing_collisions = True
        cfg.filter_robot_casing_arm = args.filter_robot_casing_arm or "both"
    if args.filter_robot_table_collisions:
        cfg.filter_robot_table_collisions = True
    env = env_cls(cfg)
    scatter_report = env.prepare_scatter_reset() if args.scatter_reset else {"enabled": False}
    if args.scatter_reset:
        (args.output_dir / "scatter_reset.json").write_text(
            json.dumps(scatter_report, indent=2) + "\n", encoding="utf-8"
        )
    # Cache the simulator device before the first long physics segment.  The
    # DirectRLEnv device accessor can become invalid after a timeout/close on
    # this Isaac build; actions must retain a stable torch.device reference.
    sim_device = env.device
    if args.camera_probe > 0:
        # Keep this probe deliberately below the M1 controller/evaluator: it
        # answers one narrow question (can this custom scene produce live RTX
        # frames?) without spending minutes in contact-point diagnostics.  A
        # zero action is a real robot hold, not a teleport or a pose write.
        import imageio.v2 as imageio
        import numpy as np

        env.reset(seed=int(args.seed))
        probe_video = args.output_dir / "camera_probe.mp4"
        probe_frame_dir = args.output_dir / "camera_probe_frames"
        probe_frame_dir.mkdir(parents=True, exist_ok=True)
        frames_written = 0
        probe_error = None
        writer = None
        try:
            writer = imageio.get_writer(
                str(probe_video), fps=20, codec="libx264", quality=7,
                pixelformat="yuv420p", macro_block_size=None,
                ffmpeg_log_level="error",
            )
            for probe_step in range(int(args.camera_probe)):
                env.step(torch.zeros((1, 14), device=sim_device))
                raw = getattr(env, "_last_obs", {})
                images = []
                overhead_frame = None
                for name in ("head_rgb", "overhead_rgb", "left_hand_rgb", "right_hand_rgb"):
                    value = raw.get(name)
                    if value is None:
                        continue
                    array = value[0].detach().cpu().numpy() if getattr(value, "ndim", 0) == 4 else value.detach().cpu().numpy()
                    if array.shape[-1] == 4:
                        array = array[..., :3]
                    frame = array.astype(np.uint8, copy=False)
                    images.append(frame)
                    if name == "overhead_rgb":
                        overhead_frame = frame
                if not images:
                    raise RuntimeError("camera probe received no RGB image")
                height = max(int(frame.shape[0]) for frame in images)
                width = max(int(frame.shape[1]) for frame in images)
                if len(images) == 4:
                    canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
                    for index, frame in enumerate(images):
                        row, column = divmod(index, 2)
                        y, x = row * height, column * width
                        canvas[y:y + frame.shape[0], x:x + frame.shape[1]] = frame
                else:
                    canvas = np.zeros((height, width * len(images), 3), dtype=np.uint8)
                    for index, frame in enumerate(images):
                        canvas[: frame.shape[0], index * width:index * width + frame.shape[1]] = frame
                writer.append_data(canvas)
                if probe_step in (0, int(args.camera_probe) - 1):
                    from PIL import Image
                    Image.fromarray(canvas).save(probe_frame_dir / f"{probe_step:04d}.png")
                    if overhead_frame is not None:
                        Image.fromarray(overhead_frame).save(
                            probe_frame_dir / f"{probe_step:04d}_overhead.png"
                        )
                frames_written += 1
            writer.close()
            writer = None
        except Exception as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
        actual_settle = {}
        if args.scatter_reset:
            states = env.root_states()
            for name, state in states.items():
                actual_settle[name] = {
                    "root_position_m": _vec(state[0, :3]),
                    "linear_speed_mps": _norm(state[0, 7:10]),
                    "angular_speed_rad_s": _norm(state[0, 10:13]),
                }
            (args.output_dir / "scatter_settle.json").write_text(
                json.dumps(actual_settle, indent=2) + "\n", encoding="utf-8"
            )
        probe_result = {
            "mode": "camera_probe",
            "seed": int(args.seed),
            "requested_steps": int(args.camera_probe),
            "frames_written": int(frames_written),
            "video": str(probe_video),
            "video_bytes": int(probe_video.stat().st_size) if probe_video.exists() else 0,
            "error": probe_error,
            "camera_update_stride": int(args.camera_update_stride),
            "render_interval": int(args.render_interval) if args.render_interval is not None else int(cfg.sim.render_interval),
            "video_frame_size": [int(args.video_width * 2), int(args.video_height * 2)],
            "scatter_settle": actual_settle,
        }
        (args.output_dir / "camera_probe.json").write_text(json.dumps(probe_result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(probe_result, indent=2), flush=True)
        env.close()
        app.close()
        return 0 if frames_written == int(args.camera_probe) else 2
    if args.gripper_only_collisions:
        # Isolate the object/jaw geometry while retaining real PhysX contact;
        # this is useful for separating a bad approach path or an oversized
        # forearm collider from a jaw-vs-Hub fit.  It is never an evaluated
        # task configuration.
        from pxr import Usd, UsdPhysics  # noqa: E402

        stage = env.sim.get_initial_stage()
        keep = {
            "both": ("/Robot/left_gripper_link1/collisions", "/Robot/left_gripper_link2/collisions"),
            "link1": ("/Robot/left_gripper_link1/collisions",),
            "link2": ("/Robot/left_gripper_link2/collisions",),
        }[args.collision_finger]
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(robot_root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not any(token in str(prim.GetPath()) for token in keep):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    def set_casing_collision(enabled: bool) -> None:
        from pxr import Usd, UsdPhysics  # noqa: E402

        stage = env.sim.get_initial_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Casing_Top")
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(bool(enabled))

    def set_hub_gravity(enabled: bool) -> None:
        # The Hub starts with gravity disabled so it remains available for the
        # grasp calibration.  For the release-before-seat experiment, switch
        # only the existing rigid body's PhysX flag at the release boundary;
        # this is a physics property change, not a pose write or kinematic
        # shortcut.
        from pxr import PhysxSchema  # noqa: E402

        stage = env.sim.get_initial_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Hub_Cover_Output_Top")
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).GetDisableGravityAttr().Set(not bool(enabled))
        env.sim.forward()

    if args.disable_casing_during_approach:
        set_casing_collision(False)
    video_path = args.output_dir / "inner_wall_probe.mp4"
    m0_vader_vqa = args.m0_serve and os.environ.get("M1_M0_VADER_VQA") == "1"
    m0_repair_rgb = args.m0_serve and os.environ.get("M1_M0_REPAIR_RGB") == "1"
    m0_capture_rgb = m0_vader_vqa or m0_repair_rgb
    rgb_frame_dir_name = "rgb_frames" if m0_repair_rgb else "vader_frames"
    rgb_frame_dir = args.output_dir / rgb_frame_dir_name if m0_capture_rgb else None
    adapter = RocoTaskAdapter(env, frame_dir=rgb_frame_dir, video_path=None if args.no_video else video_path)
    try:
        adapter.reset("inner_wall_probe", int(args.seed), "nominal")
        if float(args.step_scale) <= 0.0:
            raise ValueError("--step-scale must be positive")

        total_env_steps = 0

        def step(count: int, before_each_step=None) -> None:
            nonlocal total_env_steps
            actual_steps = max(1, int(round(float(count) * float(args.step_scale))))
            if before_each_step is None:
                adapter._step(actual_steps)
                total_env_steps += actual_steps
            else:
                for _ in range(actual_steps):
                    before_each_step()
                    adapter._step(1)
                    total_env_steps += 1

        # M0's persistent-worker protocol uses append-only event/command
        # files because the Isaac launcher runs inside a container while the
        # supervisor runs outside it. The files live in the mounted output
        # directory, so this does not add a second stage writer or network
        # control channel.
        m0_events_path = args.output_dir / "m0_events.jsonl"
        m0_command_path = args.output_dir / "m0_command.json"
        m0_event_index = 0
        m0_initial_rgb_observe = args.m0_serve and os.environ.get("M1_M0_INITIAL_RGB_OBSERVE") == "1"
        m0_replan_after_help = args.m0_serve and os.environ.get("M1_M0_REPLAN_AFTER_HELP") == "1"
        m0_stepwise_actions = args.m0_serve and os.environ.get("M1_M0_STEPWISE_ACTIONS") == "1"

        def m0_event(name: str, **payload: object) -> None:
            nonlocal m0_event_index
            m0_event_index += 1
            record = {
                "index": m0_event_index,
                "event": name,
                "wall_time": time.time(),
                **payload,
            }
            with m0_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

        def m0_capture_rgb_frames(label: str) -> list[str]:
            if not m0_capture_rgb:
                return []
            raw = getattr(env, "_last_obs", {})
            saved = []
            for camera in ("head_rgb", "left_hand_rgb", "right_hand_rgb"):
                prefix = "repair" if m0_repair_rgb else "vader"
                path = adapter._save_image(raw.get(camera), f"{prefix}_{label}_{camera}")
                if path:
                    saved.append(Path(path).name)
            return saved

        def m0_wait_for_planner_action(expected_action: str, stage: str) -> None:
            """Keep this Isaac episode live until the planner explicitly chooses the next skill."""
            if not args.m0_serve:
                return
            started = time.time()
            poll_iteration = 0
            m0_event("awaiting_planner_action", stage=stage, expected_action=expected_action, safe_hold=True)
            while True:
                if time.time() - started > float(args.m0_help_timeout_s):
                    m0_event("place_decision_timeout", stage=stage)
                    raise TimeoutError(f"M0 planner did not resume placement after {stage}")
                command = None
                try:
                    if m0_command_path.exists():
                        command = json.loads(m0_command_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    command = None
                if isinstance(command, dict) and command.get("action") == expected_action:
                    try:
                        m0_command_path.unlink()
                    except OSError:
                        pass
                    m0_event("planner_action_authorized", stage=stage, action=expected_action)
                    return
                if isinstance(command, dict) and command.get("action") == "stop":
                    try:
                        m0_command_path.unlink()
                    except OSError:
                        pass
                    m0_event("planner_stop_authorized", stage=stage)
                    raise RuntimeError(f"M0 planner stopped after {stage}")
                poll_iteration += 1
                if poll_iteration % max(1, int(args.m0_help_poll_step_interval)) == 0:
                    if stage in {"after_help", "after_pick"}:
                        env.set_gripper(float(args.m0_hold_opening))
                        if args.dual_gripper:
                            env.set_dual_gripper_targets(float(args.m0_hold_opening), float(args.m0_hold_opening))
                    step(1)
                time.sleep(0.2)

        def m0_wait_for_help() -> None:
            """Pause in a safe hold until the supervisor authorizes help."""
            if not args.m0_serve:
                return
            env.set_gripper(float(args.m0_hold_opening))
            if args.dual_gripper:
                right_body_id = env.right_arm_cfg.body_ids[0]
                right_hold = robot.data.body_state_w[0, right_body_id, :7].clone()
                env.set_pose_target(
                    "right",
                    right_hold[:3].unsqueeze(0),
                    right_hold[3:7].unsqueeze(0),
                )
                env.set_gripper(float(args.m0_hold_opening))
            if m0_capture_rgb:
                step(1)
            camera_frame_names = m0_capture_rgb_frames("blocked")
            m0_event("blocked_waiting_help", control_owner="robot", safe_hold=True, camera_frame_names=camera_frame_names)
            started = time.time()
            poll_iteration = 0
            while True:
                if time.time() - started > float(args.m0_help_timeout_s):
                    m0_event("help_timeout", timeout_s=float(args.m0_help_timeout_s))
                    raise TimeoutError("M0 helper command timed out")
                command = None
                try:
                    if m0_command_path.exists():
                        command = json.loads(m0_command_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    command = None
                if isinstance(command, dict) and command.get("action") == "help":
                    if not hasattr(env, "clear_target_blocker") or not env.clear_target_blocker():
                        m0_event("helper_failed", reason="registered_blocker_unavailable")
                        raise RuntimeError("M0 helper could not clear the registered blocker")
                    # Let the kinematic blocker move and contacts settle while
                    # the robot remains held; this is not a reset or teleport.
                    step(max(1, int(args.m0_help_settle_steps)))
                    camera_frame_names = m0_capture_rgb_frames("after_help")
                    m0_event("helper_applied", request_type="clear_target_area", target="hub", safe_hold=True, camera_frame_names=camera_frame_names)
                    try:
                        m0_command_path.unlink()
                    except OSError:
                        pass
                    if m0_replan_after_help:
                        m0_wait_for_planner_action("place", "after_help")
                    return
                # Hold the last valid joint target and continue physics at a
                # bounded cadence.  Sparse stepping avoids accumulating an
                # uncontrolled dynamic drift during a wall-clock handoff,
                # while still keeping the same Isaac stage live.
                poll_iteration += 1
                if poll_iteration % max(1, int(args.m0_help_poll_step_interval)) == 0:
                    env.set_gripper(float(args.m0_hold_opening))
                    if args.dual_gripper:
                        right_body_id = env.right_arm_cfg.body_ids[0]
                        right_hold = robot.data.body_state_w[0, right_body_id, :7].clone()
                        env.set_pose_target(
                            "right",
                            right_hold[:3].unsqueeze(0),
                            right_hold[3:7].unsqueeze(0),
                        )
                        env.set_gripper(float(args.m0_hold_opening))
                    step(1)
                time.sleep(0.2)

        robot = env.robot
        names = list(robot.body_names)
        ids = {name: names.index(name) for name in names if name in {
            "left_arm_link6", "left_gripper_link1", "left_gripper_link2",
            "right_arm_link6", "right_gripper_link1", "right_gripper_link2",
        }}

        def snapshot(label: str) -> dict[str, object]:
            root = env.root_states()["hub"][0]
            bodies = robot.data.body_state_w[0]
            contact_forces: dict[str, float] = {}
            try:
                values = robot.root_physx_view.get_net_contact_forces(dt=float(env.cfg.sim_dt))
                for name, idx in ids.items():
                    contact_forces[name] = _norm(values[0, idx])
            except Exception as exc:  # diagnostic data may be unavailable in a build
                contact_forces["error"] = type(exc).__name__
            gripper_contact: dict[str, float | str] = {}
            tip_contact_geometry: dict[str, object] = {}
            for name in (
                "left_gripper_link1_contact", "left_gripper_link2_contact",
                "right_gripper_link1_contact", "right_gripper_link2_contact",
            ):
                sensor = getattr(env, name, None)
                if sensor is None:
                    # The optional mirrored-arm path avoids extra filtered
                    # ContactSensor instances on Isaac builds that crash when
                    # both arms stream contact buffers.  Net PhysX body force
                    # is still real contact evidence (with less pair detail).
                    gripper_contact[name] = contact_forces.get(name, 0.0)
                    continue
                try:
                    # DirectRLEnv updates the scene sensors during the step,
                    # but force-recompute is needed for a deterministic
                    # diagnostic sample immediately after a commanded
                    # closing/lift waypoint.
                    sensor.update(
                        float(env.cfg.sim_dt) * int(env.cfg.decimation),
                        # PhysX contact buffers are already refreshed by the
                        # preceding environment step.  Recomputing a dense
                        # CAD/support contact stream here can terminate the
                        # Isaac process without a Python exception on this
                        # build, so use the current sensor snapshot.
                        force_recompute=False,
                    )
                    matrix = sensor.data.force_matrix_w
                    net = sensor.data.net_forces_w
                    matrix_norm = _norm(matrix) if matrix is not None and matrix.numel() else 0.0
                    net_norm = _norm(net) if net is not None and net.numel() else 0.0
                    gripper_contact[name] = matrix_norm
                    gripper_contact[f"{name}_net"] = net_norm
                    if args.track_tip_contact or args.strict_acceptance:
                        # ContactSensor exposes raw PhysX points only when
                        # track_contact_points=True.  Keep this diagnostic
                        # best-effort: a missing API is recorded as an error,
                        # never silently converted into a topology pass.
                        points: list[list[float]] = []
                        try:
                            data = sensor.contact_physx_view.get_contact_data(
                                dt=float(env.cfg.sim_dt) * int(env.cfg.decimation)
                            )
                            raw_points = data[1]
                            raw_counts = data[4]
                            count_tensor = raw_counts.reshape(-1)
                            count = int(count_tensor[0].item()) if count_tensor.numel() else 0
                            point_tensor = raw_points.reshape(-1, 3)
                            points = _vec(point_tensor[:max(0, count)].reshape(-1)).copy()
                            points = [
                                points[index:index + 3]
                                for index in range(0, len(points), 3)
                            ]
                        except Exception as point_exc:
                            tip_contact_geometry[name] = {"error": type(point_exc).__name__}
                            continue
                        center = root[:3].detach().cpu()
                        inner_points = []
                        outer_points = []
                        radii = []
                        for point in points:
                            p = torch.tensor(point)
                            radius = float(torch.linalg.vector_norm(p[:2] - center[:2]).item())
                            z_error = abs(float(p[2] - center[2]))
                            radii.append(radius)
                            if 0.050 <= radius <= 0.080 and z_error <= 0.035:
                                inner_points.append(point)
                            if 0.115 <= radius <= 0.145 and z_error <= 0.035:
                                outer_points.append(point)
                        tip_contact_geometry[name] = {
                            "point_count": len(points),
                            "points_w_m": points,
                            "radial_distances_m": radii,
                            "inner_wall_candidate": bool(inner_points),
                            "outer_wall_candidate": bool(outer_points),
                            "inner_wall_points_w_m": inner_points,
                            "outer_wall_points_w_m": outer_points,
                        }
                except Exception as exc:
                    gripper_contact[name] = type(exc).__name__
            hub_casing_force = 0.0
            try:
                env.hub_contact.update(
                    float(env.cfg.sim_dt) * int(env.cfg.decimation),
                    force_recompute=False,
                )
                hub_matrix = env.hub_contact.data.force_matrix_w
                hub_casing_force = _norm(hub_matrix) if hub_matrix is not None and hub_matrix.numel() else 0.0
            except Exception:
                # Keep the evaluator conservative when a build cannot expose
                # the filtered Hub↔Casing force buffer.
                hub_casing_force = 0.0
            return {
                "label": label,
                "controller_env_steps_total": total_env_steps,
                "episode_elapsed_s": float(env.episode_length_buf[0].item())
                * float(env.cfg.sim_dt)
                * int(env.cfg.decimation),
                "hub_root_state": _vec(root[:13]),
                "hub_speed_mps": _norm(root[7:10]),
                "gripper_joint": _vec(robot.data.joint_pos[0, env.left_gripper_cfg.joint_ids]),
                "arm_joint_states": {
                    side: {
                        "names": [robot.joint_names[int(i)] for i in arm_cfg.joint_ids],
                        "position_rad": _vec(robot.data.joint_pos[0, arm_cfg.joint_ids]),
                        "velocity_rad_s": _vec(robot.data.joint_vel[0, arm_cfg.joint_ids]),
                        "commanded_position_rad": (
                            _vec(env._joint_target[0 if side == "left" else 1][0])
                            if env._joint_target is not None
                            else None
                        ),
                        "command_error_rad": (
                            _vec(
                                env._joint_target[0 if side == "left" else 1][0]
                                - robot.data.joint_pos[0, arm_cfg.joint_ids]
                            )
                            if env._joint_target is not None
                            else None
                        ),
                        "soft_limits_rad": _vec(robot.data.soft_joint_pos_limits[0, arm_cfg.joint_ids]),
                    }
                    for side, arm_cfg in (
                        ("left", env.left_arm_cfg),
                        ("right", env.right_arm_cfg),
                    )
                },
                "bodies": {name: _vec(bodies[idx, :7]) for name, idx in ids.items()},
                "all_body_positions": {
                    name: _vec(bodies[idx, :3]) for idx, name in enumerate(names)
                },
                "body_contact_force_norm_N": contact_forces,
                "gripper_to_hub_contact_force_norm_N": gripper_contact,
                "tip_contact_geometry": tip_contact_geometry,
                "hub_casing_force_norm": hub_casing_force,
            }

        records: list[dict[str, object]] = [snapshot("reset")]
        if args.full_gravity:
            # Let the dynamic cover settle onto the annular staging pads before
            # any arm action.  This is deliberately a physics step, not a
            # corrective pose write; the resulting state is the grasp target.
            step(100)
            records.append(snapshot("gravity_reset_settle"))
        if m0_initial_rgb_observe:
            camera_frame_names = m0_capture_rgb_frames("initial")
            m0_event("initial_ready", control_owner="robot", safe_hold=True, camera_frame_names=camera_frame_names)
            m0_wait_for_planner_action("pick" if args.scatter_reset else "place", "initial")
        hub = env.root_states()["hub"][0, :3].clone()
        if args.orientation in ("radial", "radial_outward", "radial_flip", "radial_y90", "radial_y90_flip"):
            # ``radial`` is the original R1 pose.  Rotating the whole tool by
            # 180 degrees about the jaw-separation axis keeps the pair across
            # the bore but reverses the authored finger-tip direction; this is
            # the physically distinct inner-wall candidate.
            q_values = {
                "radial": (0.0, 1.0, 0.0, 0.0),
                "radial_outward": (0.0, 0.0, 1.0, 0.0),
                "radial_flip": (1.0, 0.0, 0.0, 0.0),
                # 90-degree rotations about the tool's local Y axis; these
                # turn the authored long finger axis away from the radial
                # direction while retaining a radial pair target.
                "radial_y90": (0.0, 0.7071067812, 0.7071067812, 0.0),
                "radial_y90_flip": (0.0, 0.7071067812, -0.7071067812, 0.0),
            }[args.orientation]
            try:
                q = torch.as_tensor(q_values, dtype=torch.float32, device=sim_device).reshape(1, 4)
            except BaseException:
                raise
            approach_opening = float(args.opening) if args.preopen else float(args.approach_opening)
            insertion_offset = float(args.z_offset)
            # Keep one independently interpretable closing action per run.
            # A prior convenience sweep silently changed the topology while
            # the object was already in contact, so it is not appropriate for
            # the corrected grasp validation.
            openings = (float(args.opening),)
        else:
            # Compose the pinned RoCo gripper pose with -90 degrees around
            # world/local Y.  The measured jaw-pair separation then lies on
            # world Z, the cover's 28 mm thin axis, so the fingers can pinch
            # the inner rim thickness instead of opposing across the bore.
            base = torch.tensor([[0.0, 0.0, 0.7071067812, -0.7071067812]], device=sim_device)
            y90 = torch.tensor([[0.7071067812, 0.0, -0.7071067812, 0.0]], device=sim_device)
            aw, ax, ay, az = base.unbind(-1)
            bw, bx, by, bz = y90.unbind(-1)
            q = torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                             aw * bx + ax * bw + ay * bz - az * by,
                             aw * by - ax * bz + ay * bw + az * bx,
                             aw * bz + ax * by - ay * bx + az * bw), dim=-1)
            approach_opening = 0.04
            insertion_offset = -0.003
            openings = (0.02, 0.01, 0.0)

        # The corrected one-inside/one-outside route uses the same jaw
        # separation axis as the original radial probe, but shifts the *whole
        # pair* radially off the bore centre.  At a suitable offset, one jaw
        # lies near the inner radius and the other near the outer radius.  A
        # zero offset intentionally reproduces the old (wrong) both-inside
        # topology for comparison.
        pair_offset = torch.tensor([[0.0, float(args.radial_offset), 0.0]], device=sim_device)
        right_pair_offset = torch.tensor([[0.0, -float(args.radial_offset), 0.0]], device=sim_device)
        env.set_gripper(approach_opening)
        if args.safe_approach:
            # Keep the arm above the casing while moving to the positive-Y
            # side of the fixture.  This is a normal controller waypoint, not
            # an object pose write or collision bypass.
            safe_waypoint = hub.unsqueeze(0) + torch.tensor(
                [[0.0, 0.25, 0.40]], device=sim_device
            )
            env.set_pose_target("left", safe_waypoint, q)
            step(140)
        env.set_pose_target("left", hub.unsqueeze(0) + pair_offset + torch.tensor([[0.0, 0.0, 0.25]], device=sim_device), q)
        step(80)
        records.append(snapshot("above_bore"))
        env.set_pose_target("left", hub.unsqueeze(0) + pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device), q)
        step(100)
        records.append(snapshot("inside_bore_before_open"))
        if args.dual_gripper:
            # Mirror the same real inner/outer pinch with the right R1 arm on
            # the opposite radial side.  The two arms are commanded through
            # ordinary IK targets; the dynamic Hub remains untouched.
            env.set_gripper(approach_opening)
            env.set_pose_target(
                "right",
                hub.unsqueeze(0) + right_pair_offset + torch.tensor([[0.0, 0.0, 0.25]], device=sim_device),
                q,
            )
            step(80)
            env.set_pose_target(
                "right",
                hub.unsqueeze(0) + right_pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device),
                q,
            )
            step(100)
            records.append(snapshot("right_inside_bore_before_open"))
        if args.disable_casing_during_approach:
            # Re-enable the fixture before any closing or lift action; the
            # diagnostic only isolates the approach-phase collision source.
            set_casing_collision(True)
        if abs(float(args.wrist_roll_deg)) > 1.0e-6:
            # Freeze the pose-IK target and apply a bounded wrist-joint offset
            # as an ordinary robot action.  This tests the physically
            # meaningful alternative in which the fingers roll around their
            # approach/separation axis rather than re-solving an unreachable
            # six-DoF TCP orientation.
            arm_target = env._pose_joint_target[1].clone()
            arm_target[:, -1] += torch.deg2rad(torch.tensor(float(args.wrist_roll_deg), device=sim_device))
            right_target = env.robot.data.joint_pos[:, env.right_arm_cfg.joint_ids].clone()
            env.clear_pose_target()
            env._joint_target = (arm_target, right_target)
            step(80)
            records.append(snapshot("wrist_roll"))
        if args.orientation in ("radial", "radial_outward", "radial_flip", "radial_y90", "radial_y90_flip"):
            openings = openings
        for opening in openings:
            env.set_pose_target(
                "left",
                hub.unsqueeze(0) + pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device),
                q,
            )
            env.set_gripper(opening)
            step(60)
            if args.dual_gripper:
                env.set_pose_target(
                    "right",
                    hub.unsqueeze(0) + right_pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device),
                    q,
                )
                env.set_gripper(opening)
                step(60)
            records.append(snapshot(f"close_{opening:.3f}"))
        if abs(float(args.grasp_correction_x_deg)) > 1.0e-6 or abs(float(args.grasp_correction_y_deg)) > 1.0e-6:
            # Apply a bounded orientation correction as an ordinary pose target
            # while both finger contacts are present.  This is a controller
            # action; it does not edit the Hub transform.
            ax = math.radians(float(args.grasp_correction_x_deg)) * 0.5
            ay = math.radians(float(args.grasp_correction_y_deg)) * 0.5
            qx = torch.tensor([math.cos(ax), math.sin(ax), 0.0, 0.0], device=sim_device)
            qy = torch.tensor([math.cos(ay), 0.0, math.sin(ay), 0.0], device=sim_device)
            qw, qxv, qyv, qzv = q.unbind(-1)
            cw, cx, cy, cz = qx
            q_after_x = torch.stack((cw * qw - cx * qxv - cy * qyv - cz * qzv,
                                     cw * qxv + cx * qw + cy * qzv - cz * qyv,
                                     cw * qyv - cx * qzv + cy * qw + cz * qxv,
                                     cw * qzv + cx * qyv - cy * qxv + cz * qw), dim=-1)
            qw, qxv, qyv, qzv = q_after_x.unbind(-1)
            cw, cx, cy, cz = qy
            q = torch.stack((cw * qw - cx * qxv - cy * qyv - cz * qzv,
                             cw * qxv + cx * qw + cy * qzv - cz * qyv,
                             cw * qyv - cx * qzv + cy * qw + cz * qxv,
                             cw * qzv + cx * qyv - cy * qxv + cz * qw), dim=-1)
            env.set_pose_target("left", hub.unsqueeze(0) + pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device), q)
            step(100)
            if args.dual_gripper:
                env.set_pose_target("right", hub.unsqueeze(0) + right_pair_offset + torch.tensor([[0.0, 0.0, insertion_offset]], device=sim_device), q)
                step(100)
            records.append(snapshot("grasp_tilt_correction"))
        defer_m0_supports = bool(
            args.m0_serve and args.m0_defer_staging_support_retraction
            and getattr(env, "target_blocker", None) is not None
        )
        if args.retract_staging_supports_after_grasp and not defer_m0_supports:
            retract_supports = getattr(env, "retract_hub_staging_supports", None)
            if not callable(retract_supports):
                raise RuntimeError("staging support retraction requested but unavailable")
            if float(args.staging_support_drop_m) <= 0.0:
                raise ValueError("--staging-support-drop-m must be positive")
            retract_steps = max(1, int(args.staging_support_retract_steps))
            for _ in range(retract_steps):
                retract_supports(
                    drop_m=float(args.staging_support_drop_m) / retract_steps,
                    mode=args.staging_support_withdraw_mode,
                )
                step(2)
            # A scatter Hub is supported only until the final pad withdraws;
            # letting it free-fall for the legacy 18-step settling window
            # biases the grasp test against a valid closed jaw.  The next
            # lift segment is the physical handoff, so use one integration
            # step for the scattered fixture and retain the legacy dwell for
            # the preplace fixture.
            step(1 if args.scatter_reset else 18)
            records.append(snapshot("staging_supports_retracted"))
        if args.grasp_constraint:
            begin_check = getattr(env, "begin_grasp_verification", None)
            if callable(begin_check):
                begin_check()
            status, reason = env.grasp_verification()
            if status != "HELD_CONFIRMED":
                raise RuntimeError(f"grasp constraint requested before contact: {status} {reason}")
            if not env.enable_grasp_constraint():
                raise RuntimeError("grasp constraint was requested but not authored")
            records.append(snapshot("grasp_constraint_enabled"))
        if args.controlled_place and not args.release_from_preplace and not args.preplace_controlled_place:
            # A single 150 mm Cartesian jump can outrun the contact solver and
            # leave a dynamically supported ring on its staging pads.  Break
            # the lift into ordinary IK waypoints while keeping the gripper
            # command closed and recording each real contact transition.
            lift_start = env.root_states()["hub"][0, :3].clone()
            # Preserve the actual grasp frame.  The link6 TCP is above the Hub
            # root by the ring thickness/tool geometry (about 37 mm in this
            # scene); commanding ``Hub + pair_offset`` would first drive the
            # fingers downward and make a valid pinch slip before the lift.
            lift_body_id = env.left_arm_cfg.body_ids[0]
            lift_link_state = robot.data.body_state_w[0, lift_body_id, :7].clone()
            lift_pair_offset = lift_link_state[:3] - lift_start
            lift_orientation = lift_link_state[3:7].unsqueeze(0)
            if args.dual_gripper:
                right_lift_body_id = env.right_arm_cfg.body_ids[0]
                right_lift_link_state = robot.data.body_state_w[0, right_lift_body_id, :7].clone()
                right_lift_pair_offset = right_lift_link_state[:3] - lift_start
                right_lift_orientation = right_lift_link_state[3:7].unsqueeze(0)
            def _set_synchronous_dual_target(
                left_position: torch.Tensor,
                left_orientation: torch.Tensor,
                right_position: torch.Tensor,
                right_orientation: torch.Tensor,
                opening: float,
            ) -> None:
                """Apply two ordinary IK joint targets in one env action."""
                left_joint_target = env._ik_target(
                    left_position, left_orientation, env.left_arm_cfg
                )
                right_joint_target = env._ik_target(
                    right_position, right_orientation, env.right_arm_cfg
                )
                # DirectRLEnv's normal joint-target branch holds both tuples
                # on every physics tick. This is still a controller action;
                # the Hub remains a dynamic rigid body under PhysX.
                env.clear_pose_target()
                env._joint_target = (left_joint_target, right_joint_target)
                env._active_arm = "left"
                if args.dual_gripper:
                    env.set_dual_gripper_targets(float(opening), float(opening))
                else:
                    env.set_gripper(float(opening))

            for lift_segment in range(1, max(1, int(args.lift_segments)) + 1):
                lift_fraction = float(lift_segment) / float(max(1, int(args.lift_segments)))
                lift_goal = lift_start.unsqueeze(0) + torch.tensor(
                    [[0.0, 0.0, 0.15 * lift_fraction]], device=sim_device
                )
                # A fixed wrist quaternion can inject an artificial torque
                # into the frictional annular pinch while the arm translates.
                # When the existing closed-loop orientation option is enabled,
                # refresh the target from the measured TCP at each lift
                # segment.  This is ordinary pose feedback; it does not write
                # the Hub pose or relax contact physics.
                segment_lift_orientation = lift_orientation
                if args.follow_link_orientation:
                    segment_lift_orientation = robot.data.body_state_w[
                        0, lift_body_id, 3:7
                    ].clone().unsqueeze(0)
                if args.dual_gripper and args.synchronous_dual_lift:
                    _set_synchronous_dual_target(
                        lift_goal + lift_pair_offset.unsqueeze(0),
                        segment_lift_orientation,
                        lift_goal + right_lift_pair_offset.unsqueeze(0),
                        right_lift_orientation,
                        float(args.opening),
                    )
                    step(max(1, int(args.lift_segment_steps)))
                else:
                    env.set_gripper(float(args.opening))
                    env.set_pose_target("left", lift_goal + lift_pair_offset.unsqueeze(0), segment_lift_orientation)
                    step(max(1, int(args.lift_segment_steps)))
                if args.dual_gripper and not args.synchronous_dual_lift:
                    env.set_dual_gripper_targets(float(args.opening), float(args.opening))
                    env.set_pose_target("right", lift_goal + right_lift_pair_offset.unsqueeze(0), right_lift_orientation)
                    step(max(1, int(args.lift_segment_steps)))
                records.append(snapshot(f"lift_segment_{lift_segment}"))
        elif not args.release_from_preplace and not args.preplace_controlled_place:
            env.set_pose_target("left", hub.unsqueeze(0) + pair_offset + torch.tensor([[0.0, 0.0, 0.15]], device=sim_device), q)
            step(120)
        # Keep a stable, well-known label for the evaluator while preserving
        # the segmented trace above.  A placement-only run intentionally has
        # no lift stage; its preplace hold is recorded separately so it cannot
        # be mistaken for a pick-and-carry result.
        records.append(snapshot("preplace_hold" if (args.release_from_preplace or args.preplace_controlled_place) else "lift"))
        if m0_stepwise_actions and args.scatter_reset:
            camera_frame_names = m0_capture_rgb_frames("after_pick")
            m0_event("pick_ready_for_place", safe_hold=True, camera_frame_names=camera_frame_names)
            m0_wait_for_planner_action("place", "after_pick")

        # The dual-arm lift is a useful physical check, but the mirrored
        # right gripper can later sweep into the Casing wall as the assembly
        # crosses the fixture.  In this explicit hybrid ablation, release and
        # retract the right tool while the left tool remains closed and
        # load-bearing.  The Hub stays dynamic throughout: only ordinary
        # gripper/Cartesian actions are sent, and no rigid-body pose is
        # written.  Keeping this event in the trace makes the topology change
        # auditable instead of silently treating the run as single-arm.
        right_released_after_lift = False
        if (
            args.dual_gripper
            and args.release_right_after_lift
            and not args.release_right_at_insertion_above
            and args.controlled_place
            and not args.release_from_preplace
            and not args.preplace_controlled_place
        ):
            right_release_body_id = env.right_arm_cfg.body_ids[0]
            right_release_link_state = robot.data.body_state_w[
                0, right_release_body_id, :7
            ].clone()
            if args.hold_right_release_pose:
                # Preserve the measured TCP pose with ordinary IK targets for
                # both arms while the right jaw opens.  Raw joint freezing
                # changes the loaded-arm compliance; recomputing each arm's
                # current pose target instead keeps the Cartesian support
                # condition unchanged without touching the Hub state.
                left_release_body_id = env.left_arm_cfg.body_ids[0]
                left_release_link_state = robot.data.body_state_w[
                    0, left_release_body_id, :7
                ].clone()
                left_release_joints = env._ik_target(
                    left_release_link_state[:3].unsqueeze(0),
                    left_release_link_state[3:7].unsqueeze(0),
                    env.left_arm_cfg,
                )
                right_release_joints = env._ik_target(
                    right_release_link_state[:3].unsqueeze(0),
                    right_release_link_state[3:7].unsqueeze(0),
                    env.right_arm_cfg,
                )
                env.clear_pose_target()
                env._joint_target = (left_release_joints, right_release_joints)
                env._active_arm = "left"
            elif args.freeze_right_release_open:
                # The right jaw opening itself can leave a large residual
                # normal load on the Hub.  Holding the measured arm joints
                # during this dwell prevents pose IK from changing the
                # loaded wrist branch while the gripper actuator opens.
                # This is an ordinary joint target: the Hub remains dynamic
                # and no object pose or velocity is written.
                left_release_joints = robot.data.joint_pos[
                    :, env.left_arm_cfg.joint_ids
                ].clone()
                right_release_joints = robot.data.joint_pos[
                    :, env.right_arm_cfg.joint_ids
                ].clone()
                env.clear_pose_target()
                env._joint_target = (left_release_joints, right_release_joints)
                env._active_arm = "left"
            else:
                env.set_pose_target(
                    "right",
                    right_release_link_state[:3].unsqueeze(0),
                    right_release_link_state[3:7].unsqueeze(0),
                )
            # Open only the right jaw while explicitly continuing to command
            # the load-bearing left jaw closed.  ``set_gripper`` follows the
            # active arm; using it here would leave the left actuator without
            # a fresh target during the entire right-arm withdrawal.
            env.set_dual_gripper_targets(float(args.opening), float(args.release_opening))
            step(max(1, int(args.release_open_steps)))
            records.append(snapshot("right_release_after_lift"))
            # Withdraw vertically first.  A lateral move while the fingers
            # are still opening can drag the dynamic annulus even when the
            # measured right-link force is already small; a pure upward
            # clearance leaves the left arm load-bearing and avoids sweeping
            # either right tip through the Hub.  The right tool is already on
            # the source side, so no fixture crossing is needed here.
            right_clearance_segments = 0 if args.release_right_in_place else 8
            for clearance_segment in range(1, right_clearance_segments + 1):
                clearance_fraction = float(clearance_segment) / float(right_clearance_segments)
                right_clearance_target = right_release_link_state[:3].unsqueeze(0) + torch.tensor(
                    [[0.0, 0.0, 0.20 * clearance_fraction]], device=sim_device
                )
                env.set_pose_target(
                    "right",
                    right_clearance_target,
                    right_release_link_state[3:7].unsqueeze(0),
                )
                env.set_dual_gripper_targets(float(args.opening), float(args.release_opening))
                step(max(1, int(180 / right_clearance_segments)))
                records.append(snapshot(f"right_retract_after_lift_{clearance_segment}"))
            records.append(snapshot(
                "right_release_in_place" if args.release_right_in_place
                else "right_retracted_after_lift"
            ))
            right_released_after_lift = True
        # This local route flag is consumed by the closed-loop transport
        # helper.  Subsequent release/retract bookkeeping still knows that
        # the episode was dual-arm, but no longer commands the withdrawn
        # right tool through the fixture.
        dual_transport = bool(args.dual_gripper and not right_released_after_lift)
        # If the right jaw is opened at a transport segment but deliberately
        # left in place, retract it only after the left arm reaches the
        # above-socket waypoint.  This avoids both the immediate withdrawal
        # impulse and leaving an open tool in the preinsert corridor.
        right_retract_pending = False

        insertion_target = None
        # Keep the manifest field defined even for --stop-after-lift runs,
        # which intentionally skip the insertion-target construction block.
        # Keep the high-Z transport waypoint and the final preinsert target
        # in the same registered socket frame.  The historical default is
        # still 0.08683718 m, but a run-scoped ``--socket-center-y`` override
        # must move both waypoints together; otherwise the controller crosses
        # the fixture at one Y and then asks the dynamic Hub to make an
        # unexplained lateral correction during the collision-sensitive
        # descent.
        route_socket_y = float(args.socket_center_y)
        if args.insert_after_lift and not args.stop_after_lift:
            # MagicAssembly's calibrated Hub_Cover_Output_Top → Casing_Top
            # interface target, expressed as the dynamic Hub root in this
            # scene.  These are reset-time calibration values, not a pose
            # write: the Hub moves only through the R1 controller and PhysX.
            casing_root = env.root_states()["casing"][0, :3].clone()
            socket_center_y = float(args.socket_center_y)
            preinsert_root = casing_root + torch.tensor(
                [[float(args.preinsert_offset_x_m),
                  socket_center_y + float(args.preinsert_offset_y_m),
                  float(args.preinsert_height_m)]], device=sim_device
            )
            if float(args.seat_depth_m) <= 0.0:
                raise ValueError("--seat-depth-m must be positive")
            seat_root = casing_root + torch.tensor(
                [[0.0, socket_center_y, float(args.seat_depth_m)]], device=sim_device
            )
            preinsert_target = preinsert_root[0].detach().cpu().tolist()
            insertion_target = seat_root[0].detach().cpu().tolist()
            # A radial clamp does not, in general, keep the Hub root at the
            # nominal pair-offset origin while the arm translates.  Measure
            # the actual grasp frame at the lift boundary and optionally use
            # that frame for insertion.  This changes only the subsequent
            # controller target; the rigid body remains fully dynamic.
            lift_hub_root = env.root_states()["hub"][0, :3].clone()
            lift_link6_state = env.robot.data.body_state_w[0, env.left_arm_cfg.body_ids[0], :7].clone()
            insertion_pair_offset = (
                lift_link6_state[:3] - lift_hub_root
                if args.preserve_lift_grasp_frame else pair_offset[0]
            )
            insertion_orientation = (
                lift_link6_state[3:7].unsqueeze(0)
                if args.preserve_lift_grasp_frame else q
            )
            if args.dual_gripper:
                right_body_id = env.right_arm_cfg.body_ids[0]
                right_link6_state = env.robot.data.body_state_w[0, right_body_id, :7].clone()
                right_insertion_pair_offset = (
                    right_link6_state[:3] - lift_hub_root
                    if args.preserve_lift_grasp_frame else right_pair_offset[0]
                )
                right_insertion_orientation = (
                    right_link6_state[3:7].unsqueeze(0)
                    if args.preserve_lift_grasp_frame else q
                )

            def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                w, x, y, z = a.unbind(-1)
                W, X, Y, Z = b.unbind(-1)
                return torch.stack((w * W - x * X - y * Y - z * Z,
                                    w * X + x * W + y * Z - z * Y,
                                    w * Y - x * Z + y * W + z * X,
                                    w * Z + x * Y - y * X + z * W), dim=-1)

            def _qinv(value: torch.Tensor) -> torch.Tensor:
                norm = torch.sum(value * value, dim=-1, keepdim=True).clamp_min(1.0e-8)
                return torch.cat((value[..., :1], -value[..., 1:]), dim=-1) / norm

            def _qrotate(qrot: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
                """Rotate a world-space grasp offset with a wrist correction."""
                single = vector.ndim == 1
                vectors = vector.unsqueeze(0) if single else vector
                rotations = qrot.unsqueeze(0) if qrot.ndim == 1 else qrot
                if rotations.shape[0] == 1 and vectors.shape[0] > 1:
                    rotations = rotations.expand(vectors.shape[0], -1)
                pure = torch.cat(
                    (torch.zeros((vectors.shape[0], 1), device=vectors.device), vectors),
                    dim=-1,
                )
                rotated = _qmul(
                    _qmul(rotations, pure),
                    _qinv(rotations),
                )[..., 1:]
                return rotated[0] if single else rotated

            orientation_goal_override: torch.Tensor | None = None

            def _closed_loop_move(target_root: torch.Tensor, label: str) -> None:
                """Move the dynamic held object using measured state feedback."""
                nonlocal insertion_pair_offset, insertion_orientation
                nonlocal right_insertion_pair_offset, right_insertion_orientation
                nonlocal dual_transport, right_released_after_lift, right_retract_pending
                body_id = env.left_arm_cfg.body_ids[0]
                position_only_transport = bool(
                    args.transport_position_only
                    and not (label == "controlled_seat" and args.seat_full_pose_ik)
                    and label.startswith((
                        "closed_loop_direct_above",
                        "closed_loop_side_low",
                        "closed_loop_rise_side",
                        "closed_loop_rise",
                        "closed_loop_side_above",
                        "closed_loop_above",
                        "closed_loop_preinsert_xy",
                        "closed_loop_preinsert",
                        "controlled_seat",
                    ))
                )
                if position_only_transport:
                    env.set_ik_position_only_override(True)
                start_root = env.root_states()["hub"][0, :3].clone()
                commanded_target_root = target_root.clone()
                if args.seat_vertical_only and label == "controlled_seat":
                    # The preinsert stage already performs the horizontal
                    # registration above the socket.  Do not combine a
                    # lateral correction with the final axial descent: a
                    # dynamic annular pinch can lose its normal load when
                    # the socket wall is contacted diagonally.  This keeps
                    # the measured current X/Y and changes only Z.
                    commanded_target_root[0, :2] = start_root[:2]
                goal_quat = (
                    orientation_goal_override.clone()
                    if orientation_goal_override is not None
                    else env.hub.data.default_root_state[0, 3:7].clone()
                )
                segments = max(1, int(args.insertion_segments))
                segment_steps = max(1, int(args.insertion_segment_steps))
                if label == "controlled_seat":
                    if int(args.controlled_seat_segments) > 0:
                        segments = int(args.controlled_seat_segments)
                    if int(args.controlled_seat_segment_steps) > 0:
                        segment_steps = int(args.controlled_seat_segment_steps)
                if label == "closed_loop_seat_yaw_correction_after" and args.rotate_held_orientation:
                    initial_error = _qmul(
                        goal_quat.unsqueeze(0),
                        _qinv(env.root_states()["hub"][0, 3:7].unsqueeze(0)),
                    )[0]
                    initial_angle = float(
                        (2.0 * torch.arccos(initial_error[0].clamp(-1.0, 1.0))).item()
                    )
                    max_step = max(1.0e-4, float(args.held_orientation_step_deg)) * math.pi / 180.0
                    segments = max(segments, int(math.ceil(initial_angle / max_step)))
                # When a loaded wrist is deliberately rotated, the measured
                # link6→Hub vector must rotate with that incremental wrist
                # motion.  Holding the old world-space vector fixed makes the
                # fingers sweep across the annulus and can eject the dynamic
                # Hub; the accumulated vector below preserves the physical
                # grasp frame instead.
                rotated_pair_target = insertion_pair_offset.clone()
                rotated_right_pair_target = (
                    right_insertion_pair_offset.clone() if args.dual_gripper else None
                )
                for segment in range(1, segments + 1):
                    hub_state = env.root_states()["hub"][0]
                    link_state = env.robot.data.body_state_w[0, body_id, :7]
                    fraction = float(segment) / float(segments)
                    desired_root = start_root + (commanded_target_root[0] - start_root) * fraction
                    pair = link_state[:3] - hub_state[:3]
                    # Once the object is lifted, preserve the measured
                    # grasp-frame offset.  Recomputing this from a slipping
                    # object at every segment feeds radial slip back into the
                    # next IK target and can amplify it into a release.
                    dynamic_controlled = bool(args.controlled_place or args.preplace_controlled_place)
                    pair_target = (
                        pair
                        if dynamic_controlled and args.adaptive_grasp_frame
                        else rotated_pair_target
                        if dynamic_controlled and args.preserve_lift_grasp_frame
                        else pair
                    )
                    orientation_delta = None
                    hub_quat = hub_state[3:7]
                    if dynamic_controlled and args.follow_link_orientation and not args.rotate_held_orientation:
                        link_goal_quat = link_state[3:7].unsqueeze(0)
                    elif dynamic_controlled and not args.rotate_held_orientation:
                        # Preserve the measured grasp orientation while the
                        # object is transported; rotating a frictional
                        # one-inner/one-outer pinch in mid-air was the source
                        # of the earlier drop at the casing corner.
                        link_goal_quat = insertion_orientation
                    else:
                        q_error = _qmul(goal_quat.unsqueeze(0), _qinv(hub_quat.unsqueeze(0)))[0]
                        angle = float((2.0 * torch.arccos(q_error[0].clamp(-1.0, 1.0))).item())
                        max_angle = torch.tensor(
                            max(1.0e-4, float(args.held_orientation_step_deg)) * math.pi / 180.0,
                            device=sim_device,
                        )
                        if angle > float(max_angle.item()):
                            axis = q_error[1:] / torch.linalg.vector_norm(q_error[1:]).clamp_min(1.0e-8)
                            half = max_angle * 0.5
                            q_step = torch.cat((torch.cos(half).reshape(1), axis * torch.sin(half)))
                        else:
                            q_step = q_error
                        orientation_delta = q_step
                        link_goal_quat = _qmul(q_step.unsqueeze(0), link_state[3:7].unsqueeze(0))
                    # Keep the measured world-space pair offset during a
                    # waypoint.  When the route reaches the stable above-
                    # socket plane, the caller refreshes this offset from the
                    # actual link/Hub state before descending; rotating it at
                    # every transport segment can itself pull the fingers off
                    # the annulus.
                    command_pair_target = pair_target
                    if orientation_delta is not None and dynamic_controlled and args.preserve_lift_grasp_frame:
                        command_pair_target = _qrotate(orientation_delta, command_pair_target)
                    right_pair = None
                    if dual_transport and args.synchronous_dual_transport:
                        right_link_state = env.robot.data.body_state_w[0, right_body_id, :7]
                        right_pair = right_link_state[:3] - env.root_states()["hub"][0, :3]
                    # In a true dual-arm route, command both wrists from the
                    # same measured state.  The old path stepped the left arm
                    # first and only then solved the right arm, so the Hub was
                    # exposed to one-sided motion for an entire controller
                    # hold.  That is a control-order artifact, not a physical
                    # assembly result.
                    right_pair_target = None
                    if dual_transport and args.synchronous_dual_transport:
                        right_pair_target = (
                            rotated_right_pair_target
                            if dynamic_controlled and args.rotate_held_orientation and args.preserve_lift_grasp_frame
                            else right_pair
                            if dynamic_controlled and args.adaptive_grasp_frame
                            else right_insertion_pair_offset
                            if dynamic_controlled and args.preserve_lift_grasp_frame
                            else right_pair
                        )
                    right_command_pair_target = right_pair_target
                    right_link_goal_quat = (
                        right_insertion_orientation if dual_transport else None
                    )
                    if (
                        right_pair_target is not None
                        and orientation_delta is not None
                        and dynamic_controlled
                        and args.preserve_lift_grasp_frame
                    ):
                        right_command_pair_target = _qrotate(
                            orientation_delta, right_pair_target
                        )
                        right_link_goal_quat = _qmul(
                            orientation_delta.unsqueeze(0),
                            right_link_state[3:7].unsqueeze(0),
                        )
                    per_step_target_update = None
                    if args.seat_jacobian_position and label == "controlled_seat":
                        # Final-seat-only measured Jacobian control for each
                        # loaded wrist. Refresh the bounded joint correction at
                        # each environment step so the next target uses the
                        # latest measured arm pose, not one stale segment-start
                        # measurement.
                        from isaaclab.utils.math import subtract_frame_transforms

                        arm_targets = [(env.left_arm_cfg, body_id, command_pair_target)]
                        if right_command_pair_target is not None:
                            arm_targets.append((
                                env.right_arm_cfg,
                                right_body_id,
                                right_command_pair_target,
                            ))

                        def update_jacobian_targets() -> None:
                            root_pose = robot.data.root_state_w[0, :7]
                            joint_targets = []
                            for arm_cfg, arm_body_id, pair_target in arm_targets:
                                current_link = robot.data.body_state_w[0, arm_body_id, :7]
                                current_pos_b, _ = subtract_frame_transforms(
                                    root_pose[:3].unsqueeze(0), root_pose[3:7].unsqueeze(0),
                                    current_link[:3].unsqueeze(0), current_link[3:7].unsqueeze(0),
                                )
                                target_link_pos = desired_root + pair_target
                                target_pos_b, _ = subtract_frame_transforms(
                                    root_pose[:3].unsqueeze(0), root_pose[3:7].unsqueeze(0),
                                    target_link_pos.unsqueeze(0), current_link[3:7].unsqueeze(0),
                                )
                                jacobian_id = arm_body_id - 1 if robot.is_fixed_base else arm_body_id
                                jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_id, :, arm_cfg.joint_ids]
                                j_pos = jacobian[:, :3, :]
                                delta_b = target_pos_b - current_pos_b
                                delta_q = torch.bmm(
                                    torch.linalg.pinv(j_pos), delta_b.unsqueeze(-1)
                                ).squeeze(-1)
                                delta_q = 0.8 * delta_q.clamp(-0.08, 0.08)
                                joint_targets.append(
                                    robot.data.joint_pos[:, arm_cfg.joint_ids].clone() + delta_q
                                )
                            left_joint_target = joint_targets[0]
                            right_joint_target = (
                                joint_targets[1]
                                if len(joint_targets) > 1
                                else robot.data.joint_pos[:, env.right_arm_cfg.joint_ids].clone()
                            )
                            env.clear_pose_target()
                            env._joint_target = (left_joint_target, right_joint_target)
                            env._active_arm = "left"

                        per_step_target_update = update_jacobian_targets
                    elif dual_transport and args.synchronous_dual_transport:
                        left_joint_target = env._ik_target(
                            (desired_root + command_pair_target).unsqueeze(0),
                            link_goal_quat,
                            env.left_arm_cfg,
                        )
                        right_joint_target = env._ik_target(
                            (desired_root + right_command_pair_target).unsqueeze(0),
                            right_link_goal_quat,
                            env.right_arm_cfg,
                        )
                        env.clear_pose_target()
                        env._joint_target = (left_joint_target, right_joint_target)
                        env._active_arm = "left"
                    else:
                        env.set_pose_target(
                            "left",
                            desired_root.unsqueeze(0) + command_pair_target.unsqueeze(0),
                            link_goal_quat,
                        )
                    env.set_gripper(float(args.opening))
                    step(segment_steps, per_step_target_update)
                    if dual_transport and not args.synchronous_dual_transport:
                        right_link_state = env.robot.data.body_state_w[0, right_body_id, :7]
                        right_pair = right_link_state[:3] - env.root_states()["hub"][0, :3]
                        right_pair_target = (
                            right_insertion_pair_offset
                        if dynamic_controlled and args.preserve_lift_grasp_frame
                            else right_pair
                        )
                        env.set_pose_target(
                            "right",
                            desired_root.unsqueeze(0) + right_pair_target.unsqueeze(0),
                            right_insertion_orientation,
                        )
                        env.set_gripper(float(args.opening))
                        step(segment_steps)
                    if orientation_delta is not None and dynamic_controlled and args.preserve_lift_grasp_frame:
                        rotated_pair_target = _qrotate(orientation_delta, rotated_pair_target)
                        if right_pair_target is not None:
                            rotated_right_pair_target = _qrotate(
                                orientation_delta, rotated_right_pair_target
                            )
                    records.append(snapshot(f"{label}_{segment}"))
                    # A dual-arm pinch can remain well-conditioned for most of
                    # the horizontal route but become over-constrained near a
                    # reachable-set boundary.  Allow an explicit, measured
                    # handoff one segment before that boundary: open and
                    # withdraw the right jaw while the left jaw stays closed,
                    # then refresh the surviving left grasp frame.  This is
                    # intentionally inside the ordinary action loop so the
                    # Hub remains dynamic and the handoff is auditable.
                    if (
                        dual_transport
                        and args.release_right_at_transport_segment
                        and segment == int(args.release_right_at_transport_segment)
                        and label.startswith((
                            "closed_loop_direct_above",
                            "closed_loop_side_above",
                            "closed_loop_above",
                        ))
                    ):
                        right_release_body_id = env.right_arm_cfg.body_ids[0]
                        right_release_link_state = robot.data.body_state_w[
                            0, right_release_body_id, :7
                        ].clone()
                        env.set_pose_target(
                            "right",
                            right_release_link_state[:3].unsqueeze(0),
                            right_release_link_state[3:7].unsqueeze(0),
                        )
                        env.set_dual_gripper_targets(
                            float(args.opening), float(args.release_opening)
                        )
                        step(max(1, int(args.release_open_steps)))
                        records.append(snapshot("right_release_at_transport_segment"))
                        if not args.release_right_in_place:
                            # Clear the released tool vertically in short
                            # measured actions. Four 50 mm increments avoid a
                            # single kinematic-looking jump and leave the
                            # left tool as the only load-bearing controller.
                            for clearance_segment in range(1, 5):
                                clearance_fraction = float(clearance_segment) / 4.0
                                if abs(float(args.right_release_clearance_y_m)) > 1.0e-9:
                                    clearance_vector = torch.tensor(
                                        [[0.0, float(args.right_release_clearance_y_m) * clearance_fraction, 0.0]],
                                        device=sim_device,
                                    )
                                else:
                                    clearance_vector = torch.tensor(
                                        [[0.0, 0.0, 0.20 * clearance_fraction]],
                                        device=sim_device,
                                    )
                                right_clearance_target = (
                                    right_release_link_state[:3].unsqueeze(0)
                                    + clearance_vector
                                )
                                env.set_pose_target(
                                    "right",
                                    right_clearance_target,
                                    right_release_link_state[3:7].unsqueeze(0),
                                )
                                env.set_dual_gripper_targets(
                                    float(args.opening), float(args.release_opening)
                                )
                                step(max(1, int(args.release_open_steps / 4)))
                                records.append(
                                    snapshot(
                                        f"right_retract_at_transport_segment_{clearance_segment}"
                                    )
                                )
                        right_released_after_lift = True
                        dual_transport = False
                        right_retract_pending = bool(args.release_right_in_place)
                        handoff_hub = env.root_states()["hub"][0, :3].clone()
                        handoff_left = robot.data.body_state_w[
                            0, env.left_arm_cfg.body_ids[0], :7
                        ].clone()
                        insertion_pair_offset = handoff_left[:3] - handoff_hub
                        insertion_orientation = handoff_left[3:7].unsqueeze(0)
                        records.append(snapshot("right_retracted_at_transport_segment"))
                if position_only_transport:
                    env.set_ik_position_only_override(None)
                final_hub = env.root_states()["hub"][0, :3].clone()
                final_link = env.robot.data.body_state_w[0, body_id, :7].clone()
                if args.rotate_held_orientation and args.controlled_place and args.preserve_lift_grasp_frame:
                    insertion_pair_offset = rotated_pair_target
                if not (args.controlled_place and args.preserve_lift_grasp_frame):
                    insertion_pair_offset = final_link[:3] - final_hub
                    insertion_orientation = final_link[3:7].unsqueeze(0)
                    if dual_transport:
                        right_final_link = env.robot.data.body_state_w[0, right_body_id, :7].clone()
                        right_insertion_pair_offset = right_final_link[:3] - final_hub
                        right_insertion_orientation = right_final_link[3:7].unsqueeze(0)

            if (
                args.correct_orientation_after_lift
                and args.controlled_place
                and (not args.dual_gripper or right_released_after_lift)
            ):
                # Correct the real held assembly while it is still well above
                # the fixture.  In dual-arm mode the second hand must already
                # be released; with one arm, this is the first safe point to
                # correct orientation before crossing toward the socket.  The
                # target root is measured; ordinary IK moves only the left
                # arm and PhysX keeps the Hub dynamic.
                post_lift_orientation_root = env.root_states()["hub"][0, :3].clone().unsqueeze(0)
                args.rotate_held_orientation = True
                _closed_loop_move(
                    post_lift_orientation_root,
                    "closed_loop_post_lift_orientation_correction",
                )
                args.rotate_held_orientation = False
                corrected_link_state = robot.data.body_state_w[
                    0, env.left_arm_cfg.body_ids[0], :7
                ].clone()
                corrected_hub_root = env.root_states()["hub"][0, :3].clone()
                insertion_pair_offset = corrected_link_state[:3] - corrected_hub_root
                insertion_orientation = corrected_link_state[3:7].unsqueeze(0)
                records.append(snapshot("post_lift_orientation_corrected"))

            if args.insert_safe_waypoint and not (args.release_from_preplace or args.preplace_controlled_place):
                above_root = casing_root + torch.tensor(
                    [[0.0, route_socket_y, 0.55]], device=sim_device
                )
                if args.controlled_place and args.closed_loop_insertion:
                    # The held dynamic object must not be dragged diagonally
                    # through the Casing's outer wall.  Execute three
                    # orthogonal legs: rise vertically, translate in X on the
                    # source-Y side, and only then cross Y above the Casing.
                    # Every leg is replanned from measured Hub/link6 states at
                    # short segments, so this remains an ordinary action path.
                    current_root = env.root_states()["hub"][0, :3].clone()
                    # Never command the held object downward while crossing
                    # the casing footprint.  Earlier runs lifted above the
                    # fixture and then lowered by ~67 mm on the side leg,
                    # creating a needless impact that ejected the dynamic Hub.
                    high_z = max(
                        float(current_root[2].item()),
                        float(casing_root[2].item()) + float(args.transport_clearance_z),
                    )
                    # Keep the final over-target waypoint on the same safe
                    # horizontal plane.  The legacy fixed 0.55 m clearance
                    # adds an unnecessary 250 mm vertical excursion when a
                    # lower, validated clearance is requested.
                    above_root[0, 2] = float(high_z)
                    rise_root = torch.tensor(
                        [[float(current_root[0].item()), float(current_root[1].item()), high_z]],
                        device=sim_device,
                    )
                    source_side_y = float(current_root[1].item())
                    side_clearance_y = float(args.transport_side_clearance_y)
                    if abs(side_clearance_y) > 1.0e-9:
                        # Preserve the side of the Casing on which the Hub
                        # started.  This avoids grazing the fixture's outer
                        # rim when the source-side waypoint happens to be
                        # only one Hub radius away from the Casing boundary.
                        source_side_y += math.copysign(
                            abs(side_clearance_y),
                            source_side_y - float(casing_root[1].item()) or 1.0,
                        )
                    side_root = torch.tensor(
                        [[float(casing_root[0].item()) + float(args.transport_side_offset_x), source_side_y, high_z]],
                        device=sim_device,
                    )
                    if args.transport_direct:
                        # At the calibrated lift height the Hub is already
                        # above the fixture's top envelope.  A direct segment
                        # tests whether the orthogonal route's IK corner is
                        # what loses the frictional pinch.  It is deliberately
                        # explicit and opt-in; the default route remains the
                        # collision-avoiding orthogonal sequence.
                        _closed_loop_move(above_root, "closed_loop_direct_above")
                    elif args.transport_x_first:
                        # The left arm has a narrower reachable set at the
                        # high-Z pose.  Move laterally while retaining the
                        # measured lift height, then raise at the side before
                        # crossing Y.  All three legs remain ordinary IK
                        # actions; the Hub stays dynamic throughout.
                        side_low_root = side_root.clone()
                        side_low_root[0, 2] = float(current_root[2].item())
                        rise_side_root = side_root.clone()
                        _closed_loop_move(side_low_root, "closed_loop_side_low")
                        _closed_loop_move(rise_side_root, "closed_loop_rise_side")
                    else:
                        _closed_loop_move(rise_root, "closed_loop_rise")
                        _closed_loop_move(side_root, "closed_loop_side_above")
                    if not args.transport_direct:
                        _closed_loop_move(above_root, "closed_loop_above")
                else:
                    env.set_pose_target("left", above_root + insertion_pair_offset, insertion_orientation)
                    step(180)
                records.append(snapshot("insertion_above"))
                if args.reanchor_at_insertion_above:
                    # The dynamic Hub can rotate or translate by a few
                    # millimetres during the long transport, even while both
                    # gripper contacts remain present.  Reusing the lift-time
                    # frame for the next waypoint then turns that real slip
                    # into an artificial wrench at the socket.  Re-anchor
                    # only from the measured link/Hub state; no Hub pose or
                    # velocity is written.
                    above_hub = env.root_states()["hub"][0, :3].clone()
                    above_left = robot.data.body_state_w[
                        0, env.left_arm_cfg.body_ids[0], :7
                    ].clone()
                    insertion_pair_offset = above_left[:3] - above_hub
                    insertion_orientation = above_left[3:7].unsqueeze(0)
                    if dual_transport:
                        above_right = robot.data.body_state_w[
                            0, env.right_arm_cfg.body_ids[0], :7
                        ].clone()
                        right_insertion_pair_offset = above_right[:3] - above_hub
                        right_insertion_orientation = above_right[3:7].unsqueeze(0)
                    records.append(snapshot("insertion_above_reanchored"))
                if args.correct_orientation_at_insertion_above and args.controlled_place:
                    # Correct the held orientation while the Hub is still
                    # above the fixture.  Performing this after horizontal
                    # transport but before the socket corridor avoids turning
                    # a wall contact into a large frictional wrench.  The
                    # helper applies bounded incremental wrist actions and
                    # the dynamic body remains under PhysX throughout.
                    orientation_root = env.root_states()["hub"][0, :3].clone().unsqueeze(0)
                    args.rotate_held_orientation = True
                    _closed_loop_move(
                        orientation_root,
                        "closed_loop_insertion_above_orientation_correction",
                    )
                    args.rotate_held_orientation = False
                    corrected_link_state = robot.data.body_state_w[
                        0, env.left_arm_cfg.body_ids[0], :7
                    ].clone()
                    corrected_hub_root = env.root_states()["hub"][0, :3].clone()
                    insertion_pair_offset = corrected_link_state[:3] - corrected_hub_root
                    insertion_orientation = corrected_link_state[3:7].unsqueeze(0)
                    if dual_transport:
                        corrected_right = robot.data.body_state_w[
                            0, env.right_arm_cfg.body_ids[0], :7
                        ].clone()
                        right_insertion_pair_offset = corrected_right[:3] - corrected_hub_root
                        right_insertion_orientation = corrected_right[3:7].unsqueeze(0)
                    records.append(snapshot("insertion_above_orientation_corrected"))
                if args.controlled_place and args.rotate_held_orientation:
                    # The transport wrist correction changes the actual
                    # grasp frame.  Refresh the offset at the collision-free
                    # above-socket waypoint before the downward insertion;
                    # this is measured feedback, not a Hub pose write.
                    above_hub = env.root_states()["hub"][0, :3].clone()
                    above_link = env.robot.data.body_state_w[0, env.left_arm_cfg.body_ids[0], :7].clone()
                    insertion_pair_offset = above_link[:3] - above_hub
                    insertion_orientation = above_link[3:7].unsqueeze(0)
                if (
                    (args.release_right_at_insertion_above or right_retract_pending)
                    and args.dual_gripper
                    and args.controlled_place
                    and not args.release_from_preplace
                    and not args.preplace_controlled_place
                ):
                    # Deferred handoff ablation: keep both real pinches
                    # load-bearing through horizontal transport, then release
                    # and withdraw only the right tool while the Hub is above
                    # the socket.  The left jaw remains explicitly closed;
                    # no rigid-body pose or constraint is authored.
                    right_release_body_id = env.right_arm_cfg.body_ids[0]
                    right_release_link_state = robot.data.body_state_w[
                        0, right_release_body_id, :7
                    ].clone()
                    env.set_pose_target(
                        "right",
                        right_release_link_state[:3].unsqueeze(0),
                        right_release_link_state[3:7].unsqueeze(0),
                    )
                    env.set_dual_gripper_targets(
                        float(args.opening), float(args.release_opening)
                    )
                    # The transport-segment variant already opened the jaw;
                    # one tick reasserts the open target before withdrawal.
                    # The ordinary deferred variant still receives the full
                    # opening dwell.
                    step(
                        1 if right_retract_pending
                        else max(1, int(args.release_open_steps))
                    )
                    records.append(snapshot(
                        "right_retract_deferred_release_at_insertion_above"
                        if right_retract_pending
                        else "right_release_at_insertion_above"
                    ))
                    for clearance_segment in range(1, 9):
                        clearance_fraction = float(clearance_segment) / 8.0
                        right_clearance_target = (
                            right_release_link_state[:3].unsqueeze(0)
                            + torch.tensor(
                                [[0.0, 0.0, 0.20 * clearance_fraction]],
                                device=sim_device,
                            )
                        )
                        env.set_pose_target(
                            "right",
                            right_clearance_target,
                            right_release_link_state[3:7].unsqueeze(0),
                        )
                        env.set_dual_gripper_targets(
                            float(args.opening), float(args.release_opening)
                        )
                        step(max(1, int(180 / 8)))
                        records.append(
                            snapshot(f"right_retract_at_insertion_above_{clearance_segment}")
                        )
                    records.append(snapshot("right_retracted_at_insertion_above"))
                    right_released_after_lift = True
                    dual_transport = False
                    right_retract_pending = False
                    # Refresh the surviving left grasp frame after the
                    # handoff; this is measured feedback, not a pose write.
                    handoff_hub = env.root_states()["hub"][0, :3].clone()
                    handoff_left = robot.data.body_state_w[
                        0, env.left_arm_cfg.body_ids[0], :7
                    ].clone()
                    insertion_pair_offset = handoff_left[:3] - handoff_hub
                    insertion_orientation = handoff_left[3:7].unsqueeze(0)
                if args.closed_loop_insertion and not (args.release_from_preplace or args.preplace_controlled_place):
                    if args.controlled_place and args.insert_safe_waypoint:
                        # Align X/Y while remaining on the collision-free
                        # above-socket plane.  A diagonal move toward the
                        # preinsert height can enter the annular wall before the
                        # Hub is centered, even when both endpoints are valid.
                        preinsert_xy_root = preinsert_root.clone()
                        preinsert_xy_root[0, 2] = max(
                            float(env.root_states()["hub"][0, 2].item()),
                            float(casing_root[2].item()) + float(args.transport_clearance_z),
                        )
                        _closed_loop_move(preinsert_xy_root, "closed_loop_preinsert_xy")
                _closed_loop_move(preinsert_root, "closed_loop_preinsert")
            elif not (args.release_from_preplace or args.preplace_controlled_place):
                env.set_pose_target("left", preinsert_root + insertion_pair_offset, insertion_orientation)
                step(180)
            if not (args.release_from_preplace or args.preplace_controlled_place):
                records.append(snapshot("preinsert"))
            if abs(float(args.seat_yaw_correction_deg)) > 1.0e-6 and args.controlled_place:
                # The strict hybrid route arrives above the socket with a
                # small measured yaw residual.  Correct only that residual at
                # the measured preinsert root, using a normal wrist pose
                # target while the dynamic Hub stays in PhysX.  This is kept
                # separate from the larger authored-orientation correction:
                # the latter previously twisted the frictional pinch and
                # ejected the held part.
                yaw_half = math.radians(float(args.seat_yaw_correction_deg)) * 0.5
                yaw_delta = torch.tensor(
                    [math.cos(yaw_half), 0.0, 0.0, math.sin(yaw_half)],
                    device=sim_device,
                )
                current_hub_quat = env.root_states()["hub"][0, 3:7].clone()
                orientation_goal_override = _qmul(
                    yaw_delta.unsqueeze(0), current_hub_quat.unsqueeze(0)
                )[0]
                args.rotate_held_orientation = True
                yaw_root = env.root_states()["hub"][0, :3].clone().unsqueeze(0)
                _closed_loop_move(yaw_root, "closed_loop_seat_yaw_correction")
                args.rotate_held_orientation = False
                orientation_goal_override = None
                corrected_link_state = robot.data.body_state_w[
                    0, env.left_arm_cfg.body_ids[0], :7
                ].clone()
                corrected_hub_root = env.root_states()["hub"][0, :3].clone()
                insertion_pair_offset = corrected_link_state[:3] - corrected_hub_root
                insertion_orientation = corrected_link_state[3:7].unsqueeze(0)
                records.append(snapshot("seat_yaw_corrected"))
            if args.correct_orientation_before_seat and args.controlled_place:
                # Keep the dynamic Hub at its measured preinsert position while
                # the ordinary IK controller corrects only the held assembly's
                # wrist orientation.  This is a separate opt-in stage: the
                # transport route preserves the frictional grasp frame, then
                # the final seat receives the authored Hub orientation needed
                # for the bolt pattern.
                orientation_root = env.root_states()["hub"][0, :3].clone().unsqueeze(0)
                args.rotate_held_orientation = True
                _closed_loop_move(orientation_root, "closed_loop_orientation_correction")
                # The correction stage has already accumulated the physical
                # grasp offset.  Freeze that measured wrist pose for the
                # subsequent axial seat; re-entering the rotate branch would
                # apply the same orientation error a second time while the
                # Hub is close to the Casing wall.
                args.rotate_held_orientation = False
                corrected_link_state = robot.data.body_state_w[
                    0, env.left_arm_cfg.body_ids[0], :7
                ].clone()
                corrected_hub_root = env.root_states()["hub"][0, :3].clone()
                # Contact/friction may have allowed a small real slip during
                # the correction.  Seed the axial seat from that measured
                # frame rather than from the idealized rotated vector.
                insertion_pair_offset = corrected_link_state[:3] - corrected_hub_root
                insertion_orientation = corrected_link_state[3:7].unsqueeze(0)
                records.append(snapshot("orientation_corrected"))
            for correction_index in range(
                max(0, int(args.closed_loop_preinsert_corrections))
                if not (args.release_from_preplace or args.preplace_controlled_place) else 0
            ):
                measured_root = env.root_states()["hub"][0, :3].clone()
                correction = preinsert_root[0] - measured_root
                correction_norm = _norm(correction)
                # A large correction is a controller/geometry failure, not a
                # reason to teleport the object.  Limit each action waypoint
                # to 30 mm and keep the residual visible in the trace.
                if correction_norm > 0.03:
                    correction = correction * (0.03 / correction_norm)
                # Apply the bounded residual from the measured state, not
                # from the already-desired waypoint (which doubles the error).
                # Reuse the closed-loop helper so a synchronous dual-arm route
                # corrects both wrists in the same action.
                corrected_root = measured_root.unsqueeze(0) + correction.unsqueeze(0)
                _closed_loop_move(
                    corrected_root,
                    f"closed_loop_preinsert_correction_{correction_index + 1}",
                )
                records.append(snapshot(f"preinsert_correction_{correction_index + 1}"))
            if args.release_from_preplace:
                # Placement-only baseline: the cover has been contacted while
                # dynamic and the staging pads have been withdrawn, but no
                # pose is written and no transport waypoint is used.  Open at
                # the measured TCP, then withdraw the real arm and let PhysX
                # settle the free cover into the socket under gravity.
                # First correct the small lateral slip caused by withdrawing
                # the pads.  This is an ordinary closed gripper IK waypoint;
                # the Hub remains dynamic and its transform is never written.
                align_root = env.root_states()["hub"][0, :3].clone()
                align_root[0] = casing_root[0]
                align_root[1] = float(args.socket_center_y) + casing_root[1]
                align_body_id = env.left_arm_cfg.body_ids[0]
                align_link_state = env.robot.data.body_state_w[0, align_body_id, :7].clone()
                align_pair_offset = align_link_state[:3] - env.root_states()["hub"][0, :3]
                env.set_pose_target(
                    "left",
                    align_root.unsqueeze(0) + align_pair_offset.unsqueeze(0),
                    align_link_state[3:7].unsqueeze(0),
                )
                env.set_gripper(float(args.opening))
                step(120)
                records.append(snapshot("preplace_align"))
                release_body_id = env.left_arm_cfg.body_ids[0]
                release_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                if args.freeze_release_open:
                    # Hold the measured arm joints while only the gripper
                    # actuator opens. Re-solving pose IK during this dwell
                    # can select a different loaded-wrist branch and drag the
                    # released ring laterally. This is still a normal
                    # actuator target; neither the Hub pose nor its velocity
                    # is authored.
                    left_release_joints = robot.data.joint_pos[
                        :, env.left_arm_cfg.joint_ids
                    ].clone()
                    right_release_joints = robot.data.joint_pos[
                        :, env.right_arm_cfg.joint_ids
                    ].clone()
                    env.clear_pose_target()
                    env._joint_target = (left_release_joints, right_release_joints)
                    env._active_arm = "left"
                else:
                    env.set_pose_target(
                        "left",
                        release_link_state[:3].unsqueeze(0),
                        release_link_state[3:7].unsqueeze(0),
                    )
                env.set_gripper(float(args.release_opening))
                step(max(1, int(args.release_open_steps)))
                if args.dual_gripper and not args.release_right_after_lift:
                    right_release_body_id = env.right_arm_cfg.body_ids[0]
                    right_release_link_state = env.robot.data.body_state_w[0, right_release_body_id, :7].clone()
                    if not args.freeze_release_open:
                        env.set_pose_target(
                            "right",
                            right_release_link_state[:3].unsqueeze(0),
                            right_release_link_state[3:7].unsqueeze(0),
                        )
                    env.set_gripper(float(args.release_opening))
                    step(max(1, int(args.release_open_steps)))
                records.append(snapshot("release"))
                # A single 20 cm Cartesian retract can make the R1
                # differential-IK solution sweep the outer finger through
                # the seated ring before the open jaw has cleared it.  The
                # persistent M0 path therefore uses measured short
                # waypoints. This remains an ordinary arm action: no Hub
                # pose/state is written, and the dynamic body reacts between
                # every segment.
                retract_segments = (
                    max(1, int(args.m0_retract_segments))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 1
                )
                retract_step_z = (
                    float(args.m0_retract_step_z_m)
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 0.20
                )
                retract_segment_steps = (
                    max(1, int(args.m0_retract_segment_steps))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 140
                )
                for _ in range(retract_segments):
                    retract_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                    retract_target = retract_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, 0.0, retract_step_z]], device=sim_device
                    )
                    env.set_pose_target("left", retract_target, retract_link_state[3:7].unsqueeze(0))
                    env.set_gripper(float(args.release_opening))
                    step(retract_segment_steps)
                if args.m0_serve and args.m0_freeze_release_joints and retract_segments > 1:
                    # The jaw is now at the first fully clear tool pose. Keep
                    # this measurement separate from the initial ``release``
                    # sample, which may still show a harmless residual outer
                    # finger contact while the arm is leaving the annulus.
                    records.append(snapshot("release_clearance"))
                if args.dual_gripper and not args.release_right_after_lift:
                    right_retract_target = right_release_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, 0.0, 0.20]], device=sim_device
                    )
                    env.set_pose_target(
                        "right", right_retract_target, right_release_link_state[3:7].unsqueeze(0)
                    )
                    env.set_gripper(float(args.release_opening))
                    step(140)
                step(max(1, int(args.gravity_settle_steps)))
                records.append(snapshot("insert"))
                records.append(snapshot("retract"))
            elif args.preplace_controlled_place:
                # Controlled placement from the preplace state.  The cover is
                # still a dynamic body and remains in the real grasp while a
                # short Cartesian descent closes the lateral/axial error.
                # This deliberately omits the long pick-and-carry route.
                place_start = env.root_states()["hub"][0, :3].clone()
                place_body_id = env.left_arm_cfg.body_ids[0]
                place_link_state = env.robot.data.body_state_w[0, place_body_id, :7].clone()
                place_pair_offset = place_link_state[:3] - place_start
                if args.preplace_correct_orientation:
                    # Correct only the measured grasp-frame rotation toward
                    # the authored Hub orientation. This is a normal wrist
                    # pose target; the dynamic Hub is never rotated by writing
                    # state.
                    place_hub_quat = env.root_states()["hub"][0, 3:7].clone()
                    target_hub_quat = env.hub.data.default_root_state[0, 3:7].clone()
                    place_rotation_delta = _qmul(
                        target_hub_quat.unsqueeze(0), _qinv(place_hub_quat.unsqueeze(0))
                    )
                    place_orientation = _qmul(
                        place_rotation_delta, place_link_state[3:7].unsqueeze(0)
                    )
                    # The link pose is rotated as an ordinary controller
                    # target.  Rotate the measured link-to-Hub offset by the
                    # same delta so the dynamic Hub remains at the commanded
                    # root instead of being translated by the wrist turn.
                    insertion_pair_offset = _qrotate(
                        place_rotation_delta[0], insertion_pair_offset
                    )
                else:
                    # Preserve the observed grasp frame; this was the more
                    # stable single-gripper path in the preceding probe.
                    place_orientation = place_link_state[3:7].unsqueeze(0)
                if float(args.preplace_seat_extra_depth_m) < 0.0:
                    raise ValueError("--preplace-seat-extra-depth-m must be non-negative")
                seat_command_root = seat_root.clone()
                seat_command_root[:, 2] -= float(args.preplace_seat_extra_depth_m)
                if args.preplace_controlled_place:
                    # M0 starts from the calibrated preplace state: the Hub
                    # is already above the registered socket and the measured
                    # grasp frame is close to its target XY.  A full
                    # rise/translate/descend detour is unnecessary here and
                    # made the IK chase an unreachable high-Z pose, dragging
                    # the dynamic part away from the socket.  Keep the
                    # preplace action as a bounded, closed-loop descent.  The
                    # strict full pick-and-carry path below still uses the
                    # orthogonal transport route.
                    if args.m0_serve and getattr(env, "target_blocker", None) is not None:
                        env.set_pose_target(
                            "left",
                            seat_command_root + insertion_pair_offset,
                            insertion_orientation,
                        )
                        env.set_gripper(float(args.opening))
                        if args.m0_precontact_guard:
                            # The obstacle is physically instantiated in the
                            # target socket.  Stop before collision instead of
                            # allowing a kinematic blocker to impart an
                            # impulse to the held Hub.  This is a guarded
                            # physical failure, not a claim of contact.
                            records.append(snapshot("m0_blocked_guard"))
                            m0_event("blocked_guarded_attempt", skill="place", safe_hold=True, physical_contact=False)
                        else:
                            # Optional contact-injection negative control. It
                            # is retained for diagnosing collision response,
                            # but is not the calibrated M0 path.
                            step(max(1, int(args.m0_blocked_attempt_steps)))
                            records.append(snapshot("m0_blocked_attempt"))
                        blocked_link_state = robot.data.body_state_w[
                            0, env.left_arm_cfg.body_ids[0], :7
                        ].clone()
                        env.set_pose_target(
                            "left",
                            blocked_link_state[:3].unsqueeze(0),
                            blocked_link_state[3:7].unsqueeze(0),
                        )
                        env.set_gripper(float(args.m0_hold_opening))
                        m0_event(
                            "blocked_contact_attempt" if not args.m0_precontact_guard else "blocked_guard_waiting_help",
                            skill="place",
                            safe_hold=True,
                            physical_contact=not args.m0_precontact_guard,
                        )
                        m0_wait_for_help()
                        m0_event("post_help_resume", control_owner="robot")
                        if defer_m0_supports:
                            # Re-align the real gripper against the still
                            # supported Hub before withdrawing the pads.  This
                            # is an ordinary IK/close action; the Hub pose is
                            # never written and the support keeps gravity
                            # physically honest during the handoff.
                            supported_hub = env.root_states()["hub"][0, :3].clone()
                            env.set_pose_target(
                                "left",
                                supported_hub.unsqueeze(0) + insertion_pair_offset.unsqueeze(0),
                                insertion_orientation,
                            )
                            env.set_gripper(float(args.opening))
                            step(max(1, int(args.m0_regrasp_steps)))
                            records.append(snapshot("m0_regrasp_on_support"))
                            retract_supports = getattr(env, "retract_hub_staging_supports", None)
                            if not callable(retract_supports):
                                raise RuntimeError("M0 deferred support retraction requested but unavailable")
                            retract_steps = max(1, int(args.staging_support_retract_steps))
                            for _ in range(retract_steps):
                                retract_supports(
                                    drop_m=float(args.staging_support_drop_m) / retract_steps,
                                    mode=args.staging_support_withdraw_mode,
                                )
                                step(2)
                            step(18)
                            records.append(snapshot("m0_supports_retracted_after_help"))
                        if args.m0_precontact_guard:
                            # Re-anchor the Cartesian grasp frame to the
                            # measured post-handoff robot/Hub state.  The
                            # helper may take wall-clock time while gravity
                            # and friction continue; reusing the pre-help
                            # link-to-Hub offset would turn that benign drift
                            # into a large release impulse.
                            post_help_link = robot.data.body_state_w[
                                0, env.left_arm_cfg.body_ids[0], :7
                            ].clone()
                            post_help_hub = env.root_states()["hub"][0, :3].clone()
                            insertion_pair_offset = post_help_link[:3] - post_help_hub
                            insertion_orientation = post_help_link[3:7].unsqueeze(0)
                            if args.dual_gripper:
                                post_help_right = robot.data.body_state_w[
                                    0, env.right_arm_cfg.body_ids[0], :7
                                ].clone()
                                right_insertion_pair_offset = post_help_right[:3] - post_help_hub
                                right_insertion_orientation = post_help_right[3:7].unsqueeze(0)
                            records.append(snapshot("m0_post_help_reanchor"))
                    _closed_loop_move(seat_command_root, "preplace_controlled_seat")
                else:
                    # Full pick-and-carry route: do not drag the dynamic Hub
                    # diagonally through the Casing wall.  Raise first,
                    # translate above the fixture, and then descend.
                    high_z = max(
                        float(place_start[2].item()),
                        float(casing_root[2].item()) + float(args.transport_clearance_z),
                    )
                    rise_root = torch.tensor(
                        [[float(place_start[0].item()), float(place_start[1].item()), high_z]],
                        device=sim_device,
                    )
                    above_root = torch.tensor(
                        [[float(casing_root[0].item()), float(seat_root[0, 1].item()), high_z]],
                        device=sim_device,
                    )
                    _closed_loop_move(rise_root, "preplace_controlled_rise")
                    records.append(snapshot("preplace_controlled_rise_done"))
                    _closed_loop_move(above_root, "preplace_controlled_above")
                    records.append(snapshot("preplace_controlled_above_done"))
                    _closed_loop_move(seat_command_root, "preplace_controlled_seat")
                if args.preplace_correct_orientation:
                    # Apply the measured wrist correction only for the final
                    # socket approach.  Keeping the grasp orientation during
                    # the transport legs avoids twisting the frictional
                    # pinch; the canonical Hub orientation is required for
                    # the socket and bolt pattern to pass.
                    insertion_orientation = place_orientation
                records.append(snapshot("insert"))
                release_body_id = env.left_arm_cfg.body_ids[0]
                release_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                if args.m0_serve and args.m0_freeze_release_joints:
                    left_release_joints = env.robot.data.joint_pos[:, env.left_arm_cfg.joint_ids].clone()
                    right_release_joints = env.robot.data.joint_pos[:, env.right_arm_cfg.joint_ids].clone()
                    env.clear_pose_target()
                    env._joint_target = (left_release_joints, right_release_joints)
                    env._active_arm = "left"
                else:
                    env.set_pose_target(
                        "left",
                        release_link_state[:3].unsqueeze(0),
                        release_link_state[3:7].unsqueeze(0),
                    )
                env.set_gripper(float(args.release_opening))
                step(max(1, int(args.release_open_steps)))
                if args.dual_gripper and not args.release_right_after_lift:
                    right_release_body_id = env.right_arm_cfg.body_ids[0]
                    right_release_link_state = env.robot.data.body_state_w[0, right_release_body_id, :7].clone()
                    env.set_pose_target(
                        "right",
                        right_release_link_state[:3].unsqueeze(0),
                        right_release_link_state[3:7].unsqueeze(0),
                    )
                    env.set_gripper(float(args.release_opening))
                    step(max(1, int(args.release_open_steps)))
                records.append(snapshot("release"))
                # The outer finger can remain on the annulus after the open
                # command.  Clear it in the outward radial direction before
                # the upward retract; this is an ordinary open-gripper motion
                # and is logged separately from the released Hub state.
                clearance_y = float(args.preplace_release_clearance_y)
                clearance_z = float(args.preplace_release_clearance_z)
                if abs(clearance_y) > 1.0e-9 or abs(clearance_z) > 1.0e-9:
                    clearance_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                    clearance_target = clearance_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, clearance_y, clearance_z]], device=sim_device
                    )
                    env.set_pose_target(
                        "left", clearance_target, clearance_link_state[3:7].unsqueeze(0)
                    )
                    env.set_gripper(float(args.release_opening))
                    step(160)
                    records.append(snapshot("release_clearance"))
                # A single 20 cm Cartesian retract can make the R1
                # differential-IK solution sweep the outer finger through
                # the seated ring before the open jaw has cleared it.  The
                # persistent M0 path therefore uses measured short
                # waypoints. This remains an ordinary arm action: no Hub
                # pose/state is written, and the dynamic body reacts between
                # every segment.
                retract_segments = (
                    max(1, int(args.m0_retract_segments))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 1
                )
                retract_step_z = (
                    float(args.m0_retract_step_z_m)
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 0.20
                )
                retract_segment_steps = (
                    max(1, int(args.m0_retract_segment_steps))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 140
                )
                for _ in range(retract_segments):
                    retract_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                    retract_target = retract_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, 0.0, retract_step_z]], device=sim_device
                    )
                    env.set_pose_target("left", retract_target, retract_link_state[3:7].unsqueeze(0))
                    env.set_gripper(float(args.release_opening))
                    step(retract_segment_steps)
                if args.m0_serve and args.m0_freeze_release_joints and retract_segments > 1:
                    # The jaw is now at the first fully clear tool pose. Keep
                    # this measurement separate from the initial ``release``
                    # sample, which may still show residual outer-finger
                    # contact while the arm is leaving the annulus.
                    records.append(snapshot("release_clearance"))
                if args.dual_gripper and not args.release_right_after_lift:
                    right_retract_target = right_release_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, 0.0, 0.20]], device=sim_device
                    )
                    env.set_pose_target(
                        "right", right_retract_target, right_release_link_state[3:7].unsqueeze(0)
                    )
                    env.set_gripper(float(args.release_opening))
                    step(140)
                step(max(1, int(args.gravity_settle_steps)))
                records.append(snapshot("retract"))
            elif args.stop_after_preinsert:
                insertion_verdict = "PREINSERT_ONLY"
            elif args.controlled_place:
                # Strict path: the Hub remains a dynamic body under gravity,
                # while the two real gripper links stay closed throughout the
                # descent.  There is no mid-air drop and no pose write to the
                # Hub.  The only state updates are normal IK joint targets and
                # PhysX integration.
                env.set_gripper(float(args.opening))
                if args.preinsert_hold_steps < 0:
                    raise ValueError("--preinsert-hold-steps must be non-negative")
                if args.freeze_preinsert_hold and args.preinsert_hold_steps:
                    # The Hub is already in real socket contact at the preinsert
                    # waypoint.  Do not issue another Cartesian IK target while
                    # the contact solver settles: that can move the wrist branch
                    # and pull a frictional pinch off the annulus.  Holding the
                    # measured joint positions is still an ordinary actuator
                    # command; it never writes the Hub pose or velocity.
                    left_hold_joints = robot.data.joint_pos[
                        :, env.left_arm_cfg.joint_ids
                    ].clone()
                    right_hold_joints = robot.data.joint_pos[
                        :, env.right_arm_cfg.joint_ids
                    ].clone()
                    env.clear_pose_target()
                    env._joint_target = (left_hold_joints, right_hold_joints)
                    env._active_arm = "left"
                elif args.reanchor_preinsert_hold and args.preinsert_hold_steps:
                    hold_hub = env.root_states()["hub"][0, :3].clone()
                    hold_link = robot.data.body_state_w[
                        0, env.left_arm_cfg.body_ids[0], :7
                    ].clone()
                    insertion_pair_offset = hold_link[:3] - hold_hub
                    insertion_orientation = hold_link[3:7].unsqueeze(0)
                    env.set_pose_target(
                        "left",
                        hold_link[:3].unsqueeze(0),
                        hold_link[3:7].unsqueeze(0),
                    )
                if args.preinsert_hold_steps:
                    step(int(args.preinsert_hold_steps))
                records.append(snapshot("preinsert_grasp_hold"))
                if not args.release_at_preinsert:
                    if args.reanchor_before_seat:
                        seat_hub = env.root_states()["hub"][0, :3].clone()
                        seat_link = robot.data.body_state_w[
                            0, env.left_arm_cfg.body_ids[0], :7
                        ].clone()
                        insertion_pair_offset = seat_link[:3] - seat_hub
                        insertion_orientation = seat_link[3:7].unsqueeze(0)
                        # In the true dual-arm route the right arm is still
                        # carrying the Hub at this point.  Refresh its
                        # measured pair frame as well; reusing the lift-time
                        # offset after real contact slip would make the two
                        # arms command inconsistent TCP targets and can
                        # eject the dynamic part during the seat descent.
                        if args.dual_gripper and dual_transport:
                            seat_right_link = robot.data.body_state_w[
                                0, env.right_arm_cfg.body_ids[0], :7
                            ].clone()
                            right_insertion_pair_offset = seat_right_link[:3] - seat_hub
                            right_insertion_orientation = seat_right_link[3:7].unsqueeze(0)
                    if args.seat_position_only:
                        env.set_ik_position_only_override(True)
                    if float(args.controlled_seat_extra_depth_m) < 0.0:
                        raise ValueError("--controlled-seat-extra-depth-m must be non-negative")
                    controlled_seat_root = seat_root.clone()
                    controlled_seat_root[:, 2] -= float(args.controlled_seat_extra_depth_m)
                    try:
                        _closed_loop_move(controlled_seat_root, "controlled_seat")
                    finally:
                        if args.seat_position_only:
                            env.set_ik_position_only_override(None)
                    if args.post_seat_hold_steps < 0:
                        raise ValueError("--post-seat-hold-steps must be non-negative")
                    if args.post_seat_hold_steps:
                        # Hold the measured arm configuration rather than
                        # re-solving a Cartesian target while the dynamic Hub
                        # is in socket contact. This is an ordinary actuator
                        # command; it never writes the Hub pose or velocity.
                        left_post_seat_joints = robot.data.joint_pos[
                            :, env.left_arm_cfg.joint_ids
                        ].clone()
                        right_post_seat_joints = robot.data.joint_pos[
                            :, env.right_arm_cfg.joint_ids
                        ].clone()
                        env.clear_pose_target()
                        env._joint_target = (
                            left_post_seat_joints,
                            right_post_seat_joints,
                        )
                        env._active_arm = "left"
                        env.set_gripper(float(args.opening))
                        step(int(args.post_seat_hold_steps))
                        records.append(snapshot("post_seat_hold"))
                    records.append(snapshot("insert"))
                    if abs(float(args.seat_yaw_correction_after_seat_deg)) > 1.0e-6:
                        # The in-air yaw ablation can twist a frictional pinch
                        # free.  This variant waits until the dynamic Hub has
                        # reached the socket waypoint, then rotates the wrist
                        # with the part still constrained by real geometry.
                        # The option is a per-segment maximum, rather than a
                        # request to apply the legacy 6-degree held-orientation
                        # step in one shot.  A one-shot correction can drag a
                        # seated annulus laterally before the support contact
                        # has time to settle.
                        # Refresh the measured grasp frame first; no Hub pose
                        # or velocity is authored here.
                        after_seat_hub = env.root_states()["hub"][0]
                        after_seat_link = robot.data.body_state_w[
                            0, env.left_arm_cfg.body_ids[0], :7
                        ].clone()
                        insertion_pair_offset = after_seat_link[:3] - after_seat_hub[:3]
                        insertion_orientation = after_seat_link[3:7].unsqueeze(0)
                        if dual_transport and args.synchronous_dual_transport:
                            after_seat_right_link = robot.data.body_state_w[
                                0, env.right_arm_cfg.body_ids[0], :7
                            ].clone()
                            right_insertion_pair_offset = (
                                after_seat_right_link[:3] - after_seat_hub[:3]
                            )
                            right_insertion_orientation = (
                                after_seat_right_link[3:7].unsqueeze(0)
                            )
                        target_hub_quat = env.hub.data.default_root_state[0, 3:7].clone()
                        orientation_goal_override = target_hub_quat
                        saved_orientation_step_deg = float(args.held_orientation_step_deg)
                        args.held_orientation_step_deg = abs(
                            float(args.seat_yaw_correction_after_seat_deg)
                        )
                        args.rotate_held_orientation = True
                        try:
                            _closed_loop_move(
                                after_seat_hub[:3].clone().unsqueeze(0),
                                "closed_loop_seat_yaw_correction_after",
                            )
                        finally:
                            args.rotate_held_orientation = False
                            args.held_orientation_step_deg = saved_orientation_step_deg
                            orientation_goal_override = None
                        corrected_link_state = robot.data.body_state_w[
                            0, env.left_arm_cfg.body_ids[0], :7
                        ].clone()
                        corrected_hub_root = env.root_states()["hub"][0, :3].clone()
                        insertion_pair_offset = corrected_link_state[:3] - corrected_hub_root
                        insertion_orientation = corrected_link_state[3:7].unsqueeze(0)
                        records.append(snapshot("seat_yaw_corrected_after"))
                    for correction_index in range(max(0, int(args.closed_loop_seat_corrections))):
                        measured_root = env.root_states()["hub"][0, :3].clone()
                        correction = seat_root[0] - measured_root
                        correction_norm = _norm(correction)
                        if correction_norm > 0.02:
                            correction = correction * (0.02 / correction_norm)
                        corrected_root = seat_root + correction.unsqueeze(0)
                        env.set_pose_target("left", corrected_root + insertion_pair_offset, insertion_orientation)
                        step(100)
                        records.append(snapshot(f"insert_correction_{correction_index + 1}"))
                release_blocked_not_seated = False
                if args.release_only_if_seated:
                    # Runtime guard at the actual gripper-release boundary.
                    # D17 released with zero Hub-Casing contact and a 34.5 mm
                    # axial gap; post-run scoring can identify that failure
                    # but cannot prevent the actuator command.
                    seat_check = snapshot("pre_release_seat_check")
                    hub_state = torch.tensor(
                        seat_check["hub_root_state"], dtype=torch.float32, device=sim_device
                    )
                    seat_target = seat_root[0].to(torch.float32)
                    target_quat = env.hub.data.default_root_state[0, 3:7].to(torch.float32)
                    position_delta = hub_state[:3] - seat_target
                    radial_error = _norm(position_delta[:2])
                    axial_error = abs(float(position_delta[2].item()))
                    orientation_error_deg = _quat_angle_deg(hub_state[3:7], target_quat)
                    bolt_local = torch.tensor(
                        [[0.0800, -0.0795, 0.0], [-0.0800, -0.0795, 0.0],
                         [0.0800, 0.0805, 0.0], [-0.0800, 0.0805, 0.0]],
                        dtype=torch.float32, device=sim_device,
                    )
                    target_bolts = seat_target.unsqueeze(0) + _qrotate(
                        target_quat.unsqueeze(0).expand(4, -1), bolt_local
                    )
                    measured_bolts = hub_state[:3].unsqueeze(0) + _qrotate(
                        hub_state[3:7].unsqueeze(0).expand(4, -1), bolt_local
                    )
                    bolt_error_max = float(
                        torch.linalg.vector_norm(measured_bolts - target_bolts, dim=-1).max().item()
                    )
                    casing_force = float(seat_check["hub_casing_force_norm"])
                    hub_speed = float(seat_check["hub_speed_mps"])
                    release_ready = bool(
                        casing_force > 1.0e-3
                        and hub_speed <= 0.01
                        and radial_error <= 0.012
                        and axial_error <= 0.015
                        and orientation_error_deg <= 5.0
                        and bolt_error_max <= 0.010
                    )
                    seat_check["release_readiness"] = {
                        "ready": release_ready,
                        "hub_casing_force_N": casing_force,
                        "hub_speed_mps": hub_speed,
                        "radial_error_m": radial_error,
                        "axial_error_m": axial_error,
                        "orientation_error_deg": orientation_error_deg,
                        "bolt_hole_alignment_max_error_m": bolt_error_max,
                    }
                    records.append(seat_check)
                    release_blocked_not_seated = not release_ready
                    print(
                        "[release] "
                        + ("seat verified; opening grippers" if release_ready
                           else "seat not verified; keeping grippers closed")
                    )
                # Hold both TCPs at their measured poses and open both jaws in
                # one action. Sequential opening leaves one arm load-bearing
                # while the other releases, which moved the Hub in the D17 replay.
                release_body_id = env.left_arm_cfg.body_ids[0]
                release_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                dual_release = args.dual_gripper and not args.release_right_after_lift
                right_release_link_state = None
                if dual_release:
                    right_release_body_id = env.right_arm_cfg.body_ids[0]
                    right_release_link_state = robot.data.body_state_w[
                        0, right_release_body_id, :7
                    ].clone()
                if args.freeze_release_open:
                    left_release_joints = robot.data.joint_pos[
                        :, env.left_arm_cfg.joint_ids
                    ].clone()
                    right_release_joints = robot.data.joint_pos[
                        :, env.right_arm_cfg.joint_ids
                    ].clone()
                    env.clear_pose_target()
                    env._joint_target = (left_release_joints, right_release_joints)
                    env._active_arm = "left"
                elif dual_release:
                    left_release_joints = env._ik_target(
                        release_link_state[:3].unsqueeze(0),
                        release_link_state[3:7].unsqueeze(0),
                        env.left_arm_cfg,
                    )
                    right_release_joints = env._ik_target(
                        right_release_link_state[:3].unsqueeze(0),
                        right_release_link_state[3:7].unsqueeze(0),
                        env.right_arm_cfg,
                    )
                    env.clear_pose_target()
                    env._joint_target = (left_release_joints, right_release_joints)
                    env._active_arm = "left"
                else:
                    env.set_pose_target(
                        "left",
                        release_link_state[:3].unsqueeze(0),
                        release_link_state[3:7].unsqueeze(0),
                    )
                if not release_blocked_not_seated:
                    if dual_release:
                        env.set_dual_gripper_targets(
                            float(args.release_opening), float(args.release_opening)
                        )
                    else:
                        env.set_gripper(float(args.release_opening))
                    step(max(1, int(args.release_open_steps)))
                if release_blocked_not_seated:
                    records.append(snapshot("release_skipped_not_seated"))
                elif args.release_at_preinsert:
                    # Keep the opening event separate from the evaluator's
                    # release sample.  The latter must be measured after the
                    # free dynamic Hub has had time to seat under gravity, so
                    # release drift is not confused with the intended seat
                    # motion.
                    records.append(snapshot("release_open"))
                    # If requested, clear the open fingers before allowing
                    # the Hub to fall.  This is an ordinary Cartesian
                    # withdrawal of the real arm; it avoids leaving an
                    # opened inner/outer finger in the socket corridor while
                    # gravity performs the final seat.
                    pregravity_clearance_z = float(args.preplace_release_clearance_z)
                    if abs(pregravity_clearance_z) > 1.0e-9:
                        clearance_link_state = env.robot.data.body_state_w[
                            0, release_body_id, :7
                        ].clone()
                        clearance_target = clearance_link_state[:3].unsqueeze(0) + torch.tensor(
                            [[0.0, 0.0, pregravity_clearance_z]], device=sim_device
                        )
                        env.set_pose_target(
                            "left", clearance_target, clearance_link_state[3:7].unsqueeze(0)
                        )
                        env.set_gripper(float(args.release_opening))
                        step(160)
                        records.append(snapshot("pregravity_release_clearance"))
                    env.clear_pose_target()
                    step(max(1, int(args.gravity_settle_steps)))
                    records.append(snapshot("insert"))
                    records.append(snapshot("release"))
                else:
                    records.append(snapshot("release"))
                # If an opened finger still carries a small normal load on the
                # seated ring, clear it laterally before the upward retract.
                # This remains a normal Cartesian IK action; the Hub stays
                # dynamic and is never moved by a state write.
                clearance_y = float(args.preplace_release_clearance_y)
                clearance_z = float(args.preplace_release_clearance_z)
                if not release_blocked_not_seated and (
                    abs(clearance_y) > 1.0e-9 or abs(clearance_z) > 1.0e-9
                ):
                    clearance_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                    clearance_target = clearance_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, clearance_y, clearance_z]], device=sim_device
                    )
                    env.set_pose_target(
                        "left", clearance_target, clearance_link_state[3:7].unsqueeze(0)
                    )
                    env.set_gripper(float(args.release_opening))
                    step(160)
                    records.append(snapshot("release_clearance"))
                # Retract only after the open command has had time to clear the
                # Hub.  The object remains under gravity and must be supported
                # by the Casing, not by a hidden attachment.
                retract_segments = (
                    max(1, int(args.m0_retract_segments))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 1
                )
                retract_step_z = (
                    float(args.m0_retract_step_z_m)
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 0.20
                )
                retract_segment_steps = (
                    max(1, int(args.m0_retract_segment_steps))
                    if args.m0_serve and args.m0_freeze_release_joints
                    else 180
                )
                if not release_blocked_not_seated:
                    for _ in range(retract_segments):
                        release_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                        retract_target = release_link_state[:3].unsqueeze(0) + torch.tensor(
                            [[0.0, 0.0, retract_step_z]], device=sim_device
                        )
                        env.set_pose_target("left", retract_target, release_link_state[3:7].unsqueeze(0))
                        env.set_gripper(float(args.release_opening))
                        step(retract_segment_steps)
                    if args.m0_serve and args.m0_freeze_release_joints and retract_segments > 1:
                        records.append(snapshot("release_clearance"))
                if args.post_release_settle_steps < 0:
                    raise ValueError("--post-release-settle-steps must be non-negative")
                if args.post_release_settle_steps and not release_blocked_not_seated:
                    # Let the released dynamic Hub settle on the Casing after
                    # the tool is already clear. This adds no pose write or
                    # hidden attachment; it only changes which final state is
                    # measured for the diagnostic metrics.
                    step(int(args.post_release_settle_steps))
                if args.dual_gripper and not args.release_right_after_lift and not release_blocked_not_seated:
                    right_retract_target = right_release_link_state[:3].unsqueeze(0) + torch.tensor(
                        [[0.0, 0.0, 0.20]], device=sim_device
                    )
                    env.set_pose_target("right", right_retract_target, right_release_link_state[3:7].unsqueeze(0))
                    env.set_gripper(float(args.release_opening))
                    step(180)
                if not release_blocked_not_seated:
                    records.append(snapshot("retract"))
            elif args.release_before_seat:
                if not args.hub_gravity:
                    raise ValueError("--release-before-seat requires --hub-gravity")
                # Open the fingers while gravity is still disabled, then move
                # the tool out of the mating footprint.  This is a normal
                # controller trajectory: it gives the inner/outer fingers a
                # chance to clear the ring without allowing a partially
                # released part to fall into the socket and collide with the
                # Casing.  No Hub pose or attachment is written here.
                #
                # The closed-loop preinsert target can be outside the R1's
                # reachable workspace (the observed dynamic link6 may stop
                # short of it).  Reusing that stale target while opening the
                # fingers makes the arm continue chasing an unreachable pose
                # and drags the Hub.  Freeze the release waypoint at the
                # *measured* link6 pose first; this is ordinary feedback
                # control and does not write either rigid body's pose.
                release_body_id = env.left_arm_cfg.body_ids[0]
                release_link_state = env.robot.data.body_state_w[0, release_body_id, :7].clone()
                env.set_pose_target(
                    "left",
                    release_link_state[:3].unsqueeze(0),
                    release_link_state[3:7].unsqueeze(0),
                )
                env.set_gripper(float(args.release_opening))
                step(max(1, int(args.release_open_steps)))
                if abs(float(args.release_wrist_roll_deg)) > 1.0e-6:
                    # Rotate the already-open tool through the real wrist
                    # joint while the dynamic Hub remains gravity-disabled.
                    # This is a bounded release maneuver, not a pose write;
                    # the Hub can move only if a finger is still contacting it.
                    if env._pose_joint_target is None:
                        raise RuntimeError("release wrist roll requires an active pose target")
                    arm_target = env._pose_joint_target[1].clone()
                    arm_target[:, -1] += torch.deg2rad(
                        torch.tensor(float(args.release_wrist_roll_deg), device=sim_device)
                    )
                    right_target = env.robot.data.joint_pos[:, env.right_arm_cfg.joint_ids].clone()
                    env.clear_pose_target()
                    env._joint_target = (arm_target, right_target)
                    step(120)
                records.append(snapshot("release"))
                release_contacts = records[-1].get("gripper_to_hub_contact_force_norm_N", {})
                release_has_contact = any(
                    isinstance(release_contacts.get(name), (int, float))
                    and float(release_contacts.get(name, 0.0)) > 1.0e-3
                    for name in ("left_gripper_link1_contact", "left_gripper_link2_contact")
                )
                clearance_y = float(args.release_clearance_y)
                clearance_x = float(args.release_clearance_x)
                clearance_z = float(args.release_clearance_z)
                can_attempt_clearance = (
                    abs(clearance_x) > 1.0e-9 or abs(clearance_y) > 1.0e-9 or abs(clearance_z) > 1.0e-9
                ) and (not release_has_contact or abs(clearance_z) > 1.0e-9)
                if can_attempt_clearance:
                    clearance_vec = torch.tensor(
                        [[clearance_x, clearance_y, clearance_z]], device=sim_device
                    )
                    clearance_distance = _norm(clearance_vec[0])
                    clearance_step = max(float(args.release_clearance_step_m), 1.0e-4)
                    clearance_segments = max(1, int(torch.ceil(torch.tensor(
                        clearance_distance / clearance_step, device=sim_device
                    )).item()))
                    steps_per_segment = max(1, int(120 / clearance_segments))
                    for segment in range(1, clearance_segments + 1):
                        fraction = float(segment) / float(clearance_segments)
                        release_clearance_root = preinsert_root + clearance_vec * fraction
                        env.set_pose_target("left", release_clearance_root + insertion_pair_offset, insertion_orientation)
                        step(steps_per_segment)
                    records.append(snapshot("release_clearance"))
                clearance_record = next(
                    (record for record in reversed(records) if record["label"] == "release_clearance"),
                    records[-1],
                )
                clearance_contacts = clearance_record.get("gripper_to_hub_contact_force_norm_N", {})
                clearance_has_contact = any(
                    isinstance(clearance_contacts.get(name), (int, float))
                    and float(clearance_contacts.get(name, 0.0)) > 1.0e-3
                    for name in ("left_gripper_link1_contact", "left_gripper_link2_contact")
                )
                # Enable gravity only after the open tool has cleared both
                # contact sensors.  If a finger is still engaged, preserve a
                # diagnostic failure without injecting a drop or a retract
                # collision into the scene.
                if not clearance_has_contact:
                    set_hub_gravity(True)
                env.clear_pose_target()
                step(max(1, int(args.gravity_settle_steps)))
                records.append(snapshot("insert"))
                insert_now = records[-1]
                insert_root_now = torch.tensor(insert_now["hub_root_state"][:3])
                insert_speed_now = float(insert_now["hub_speed_mps"])
                seat_target_now = torch.tensor(insertion_target)
                stable_seat = (
                    _norm(insert_root_now - seat_target_now) <= 0.015
                    and insert_speed_now <= 0.05
                )
                # Move the open tool away only after the object is plausibly
                # seated.  If it is not stable, keep the scene at the observed
                # state and record a safe diagnostic failure rather than
                # injecting a second collision.
                retract_root = preinsert_root + torch.tensor(
                    [[0.0, 0.0, 0.20]], device=sim_device
                )
                if stable_seat:
                    env.set_pose_target("left", retract_root + insertion_pair_offset, insertion_orientation)
                    step(140)
                records.append(snapshot("retract"))
            else:
                if args.closed_loop_insertion:
                    _closed_loop_move(seat_root, "closed_loop_seat")
                else:
                    env.set_pose_target("left", seat_root + insertion_pair_offset, insertion_orientation)
                    step(140)
                records.append(snapshot("insert"))
                for correction_index in range(max(0, int(args.closed_loop_seat_corrections))):
                    measured_root = env.root_states()["hub"][0, :3].clone()
                    correction = seat_root[0] - measured_root
                    correction_norm = _norm(correction)
                    if correction_norm > 0.03:
                        correction = correction * (0.03 / correction_norm)
                    corrected_root = seat_root + correction.unsqueeze(0)
                    env.set_pose_target("left", corrected_root + insertion_pair_offset, insertion_orientation)
                    step(120)
                    records.append(snapshot(f"insert_correction_{correction_index + 1}"))
                # Release is an ordinary gripper action.  The Hub remains dynamic
                # and is not repositioned after this point.
                env.set_gripper(0.04)
                step(80)
                records.append(snapshot("release"))
                if args.grasp_constraint:
                    env.disable_grasp_constraint()
                env.set_pose_target("left", preinsert_root + insertion_pair_offset, insertion_orientation)
                step(140)
                records.append(snapshot("retract"))

        reset_pos = torch.tensor(records[0]["hub_root_state"][:3])
        lift_record = next((record for record in records if record["label"] == "lift"), records[-1])
        lift_pos = torch.tensor(lift_record["hub_root_state"][:3])
        close_record = next((record for record in reversed(records) if str(record["label"]).startswith("close_")), records[-1])
        close_hub = torch.tensor(close_record["hub_root_state"][:3])
        close_link6 = torch.tensor(close_record["bodies"]["left_arm_link6"][:3])
        lift_link6 = torch.tensor(lift_record["bodies"]["left_arm_link6"][:3])
        hub_lift = lift_pos - close_hub
        link6_lift = lift_link6 - close_link6
        contact_values = []
        for record in records:
            for key, value in record.get("gripper_to_hub_contact_force_norm_N", {}).items():
                if key.endswith("_net") or not isinstance(value, (int, float)):
                    continue
                contact_values.append(float(value))
        contact_names = ["left_gripper_link1_contact", "left_gripper_link2_contact"]
        if args.dual_gripper:
            contact_names.extend(["right_gripper_link1_contact", "right_gripper_link2_contact"])
        both_contact = all(
            max((float(record.get("gripper_to_hub_contact_force_norm_N", {}).get(name, 0.0))
                 for record in records if isinstance(record.get("gripper_to_hub_contact_force_norm_N", {}).get(name, 0.0), (int, float))),
            default=0.0) > 1.0e-3
            for name in contact_names
        )
        follows_lift = _norm(hub_lift) >= 0.5 * max(_norm(link6_lift), 1.0e-6)
        if both_contact and follows_lift:
            candidate_verdict = "HELD_CANDIDATE"
        elif _norm(torch.tensor(records[2]["hub_root_state"][:3]) - reset_pos) > 0.05:
            candidate_verdict = "COLLISION_EJECTION_OR_APPROACH_DRIFT"
        elif not contact_values or max(contact_values, default=0.0) <= 1.0e-3:
            candidate_verdict = "NO_REPORTED_FINGER_CONTACT"
        else:
            candidate_verdict = "CONTACT_WITHOUT_HELD_FOLLOW"
        release_performed = any(record.get("label") == "release" for record in records)
        insertion_verdict = None
        insertion_metrics: dict[str, object] = {}
        if args.insert_after_lift and not args.stop_after_lift:
            if args.stop_after_preinsert:
                preinsert_record = next(record for record in records if record["label"] == "preinsert")
                insertion_metrics = {
                    "preinsert_target_root_m": preinsert_target,
                    "preinsert_hub_root_m": preinsert_record["hub_root_state"][:3],
                    "preinsert_root_error_m": _norm(
                        torch.tensor(preinsert_record["hub_root_state"][:3]) - torch.tensor(preinsert_target)
                    ),
                    "preinsert_hub_casing_force_norm_N": float(preinsert_record["hub_casing_force_norm"]),
                    "preinsert_gripper_contact_force_norm_N": {
                        key: value
                        for key, value in preinsert_record.get("gripper_to_hub_contact_force_norm_N", {}).items()
                        if not key.endswith("_net")
                    },
                }
                insertion_verdict = "PREINSERT_ONLY"
            else:
                insert_record = next(
                    record for record in reversed(records)
                    if record["label"] == "insert" or str(record["label"]).startswith("insert_correction_")
                )
                release_record = next(
                    (record for record in records if record["label"] == "release"),
                    insert_record,
                )
                retract_record = next(
                    (record for record in records if record["label"] == "retract"),
                    release_record,
                )
            if not args.stop_after_preinsert:
                clearance_record = next(
                    (record for record in reversed(records) if record["label"] == "release_clearance"),
                    release_record,
                )
                insert_pos = torch.tensor(insert_record["hub_root_state"][:3])
                release_pos = torch.tensor(release_record["hub_root_state"][:3])
                retract_pos = torch.tensor(retract_record["hub_root_state"][:3])
                seat_target = torch.tensor(insertion_target)
                seat_error = _norm(insert_pos - seat_target)
                release_drift = _norm(release_pos - insert_pos)
                retract_drift = _norm(retract_pos - release_pos)
                seat_contact = float(insert_record["hub_casing_force_norm"]) > 1.0e-3
                insertion_metrics = {
                    "seat_target_root_m": insertion_target,
                    "insert_hub_root_m": _vec(insert_pos),
                    "insert_root_error_m": seat_error,
                    "insert_hub_casing_force_norm_N": float(insert_record["hub_casing_force_norm"]),
                    "release_performed": bool(release_performed),
                    "release_skipped_not_seated": any(
                        record.get("label") == "release_skipped_not_seated" for record in records
                    ),
                    "release_hub_root_m": _vec(release_pos),
                    "release_drift_m": release_drift,
                    "release_to_insert_motion_m": _norm(insert_pos - release_pos),
                    "release_clearance_hub_root_m": _vec(torch.tensor(clearance_record["hub_root_state"][:3])),
                    "release_clearance_gripper_contact_force_norm_N": {
                        key: value
                        for key, value in clearance_record.get("gripper_to_hub_contact_force_norm_N", {}).items()
                        if not key.endswith("_net")
                    },
                    "retract_hub_root_m": _vec(retract_pos),
                    "retract_drift_m": retract_drift,
                    "post_settle_retract_drift_m": _norm(retract_pos - insert_pos),
                }
                stable_insert = float(insert_record["hub_speed_mps"]) <= 0.05
                if seat_error <= 0.01 and seat_contact and stable_insert and (
                    args.release_before_seat or release_drift <= 0.02
                ):
                    insertion_verdict = "SEAT_RELEASE_CANDIDATE"
                elif seat_error <= 0.02 and seat_contact:
                    insertion_verdict = "CONTACTED_SEAT_NOT_STABLE"
                else:
                    insertion_verdict = "INSERTION_NOT_VERIFIED"
                # Strict acceptance is intentionally separate from the legacy
                # candidate verdict.  It checks the physical evidence that a
                # video alone cannot establish: full 6DoF pose, the four
                # canonical top-bolt locations, two-sided tip contact, and a
                # gravity-supported release/retract.
                insert_quat = torch.tensor(insert_record["hub_root_state"][3:7], dtype=torch.float32)
                target_quat = env.hub.data.default_root_state[0, 3:7].detach().cpu().to(torch.float32)
                orientation_error_deg = _quat_angle_deg(insert_quat, target_quat)
                position_delta = insert_pos - seat_target
                radial_error = _norm(position_delta[:2])
                axial_error = abs(float(position_delta[2].item()))
                bolt_local = torch.tensor(
                    [[0.0800, -0.0795, 0.0], [-0.0800, -0.0795, 0.0],
                     [0.0800, 0.0805, 0.0], [-0.0800, 0.0805, 0.0]],
                    dtype=torch.float32,
                )
                target_root_tensor = seat_target.to(torch.float32)
                insert_root_tensor = insert_pos.to(torch.float32)
                target_bolts = target_root_tensor.unsqueeze(0) + _qrotate(
                    target_quat.unsqueeze(0).repeat(4, 1), bolt_local
                )
                insert_bolts = insert_root_tensor.unsqueeze(0) + _qrotate(
                    insert_quat.unsqueeze(0).repeat(4, 1), bolt_local
                )
                bolt_errors = torch.linalg.vector_norm(insert_bolts - target_bolts, dim=-1)
                bolt_error_values = [float(v) for v in bolt_errors.detach().cpu().tolist()]
                # Also expose the pose after the object has been released and
                # the arm has retracted. The strict pre-release checks above
                # remain unchanged for reproducibility, but a gravity-seat
                # trajectory can legitimately settle a few millimetres or
                # degrees after the ``insert`` snapshot. Keeping both poses
                # makes that distinction auditable instead of hiding it in a
                # video-only judgment.
                final_quat = torch.tensor(
                    retract_record["hub_root_state"][3:7], dtype=torch.float32
                )
                final_orientation_error_deg = _quat_angle_deg(final_quat, target_quat)
                final_position_delta = retract_pos - seat_target
                final_target_bolts = target_bolts
                final_bolts = retract_pos.unsqueeze(0) + _qrotate(
                    final_quat.unsqueeze(0).repeat(4, 1), bolt_local
                )
                final_bolt_errors = torch.linalg.vector_norm(
                    final_bolts - final_target_bolts, dim=-1
                )
                insertion_metrics.update({
                    "final_hub_root_m": _vec(retract_pos),
                    "final_root_error_m": _norm(final_position_delta),
                    "final_radial_error_m": _norm(final_position_delta[:2]),
                    "final_axial_error_m": abs(float(final_position_delta[2].item())),
                    "final_hub_casing_force_norm_N": float(retract_record["hub_casing_force_norm"]),
                    "final_orientation_error_deg": final_orientation_error_deg,
                    "final_bolt_hole_alignment_error_m": [
                        float(v) for v in final_bolt_errors.detach().cpu().tolist()
                    ],
                    "final_bolt_hole_alignment_max_error_m": float(
                        final_bolt_errors.max().item()
                    ),
                })
                tip_topology_records = []
                for record in records:
                    geometry = record.get("tip_contact_geometry", {})
                    if not isinstance(geometry, dict):
                        continue
                    link1 = geometry.get("left_gripper_link1_contact", {})
                    link2 = geometry.get("left_gripper_link2_contact", {})
                    if not isinstance(link1, dict) or not isinstance(link2, dict):
                        continue
                    link1_inner = bool(link1.get("inner_wall_candidate", False))
                    link1_outer = bool(link1.get("outer_wall_candidate", False))
                    link2_inner = bool(link2.get("inner_wall_candidate", False))
                    link2_outer = bool(link2.get("outer_wall_candidate", False))
                    left_one_inner_one_outer = bool(
                        (link1_inner and link2_outer) or (link1_outer and link2_inner)
                    )
                    topology_entry = {
                        "label": record.get("label"),
                        "link1_inner": link1_inner,
                        "link1_outer": link1_outer,
                        "link2_inner": link2_inner,
                        "link2_outer": link2_outer,
                        "one_inner_one_outer": left_one_inner_one_outer,
                    }
                    if args.dual_gripper:
                        right_link1 = geometry.get("right_gripper_link1_contact", {})
                        right_link2 = geometry.get("right_gripper_link2_contact", {})
                        right_link1_inner = bool(isinstance(right_link1, dict) and right_link1.get("inner_wall_candidate", False))
                        right_link1_outer = bool(isinstance(right_link1, dict) and right_link1.get("outer_wall_candidate", False))
                        right_link2_inner = bool(isinstance(right_link2, dict) and right_link2.get("inner_wall_candidate", False))
                        right_link2_outer = bool(isinstance(right_link2, dict) and right_link2.get("outer_wall_candidate", False))
                        topology_entry["right_one_inner_one_outer"] = bool(
                            (right_link1_inner and right_link2_outer)
                            or (right_link1_outer and right_link2_inner)
                        )
                    tip_topology_records.append(topology_entry)
                tip_pair_pass = any(
                    item["one_inner_one_outer"]
                    and (not args.dual_gripper or item.get("right_one_inner_one_outer", False))
                    for item in tip_topology_records
                )
                contact_free_record = next(
                    (record for record in reversed(records) if record["label"] == "release_clearance"),
                    release_record,
                )
                release_force_values = contact_free_record.get("gripper_to_hub_contact_force_norm_N", {})
                retract_force_values = retract_record.get("gripper_to_hub_contact_force_norm_N", {})
                release_force_names = ["left_gripper_link1_contact", "left_gripper_link2_contact"]
                if args.dual_gripper:
                    release_force_names.extend(["right_gripper_link1_contact", "right_gripper_link2_contact"])
                release_contact_free = all(
                    not isinstance(values.get(name), (int, float)) or float(values.get(name, 0.0)) <= 1.0e-3
                    for values in (release_force_values, retract_force_values)
                    for name in release_force_names
                )
                bolt_alignment_pass = max(bolt_error_values, default=float("inf")) <= 0.003
                six_dof_pass = (
                    radial_error <= 0.004
                    and axial_error <= 0.008
                    and orientation_error_deg <= 2.0
                )
                strict_checks = {
                    "full_gravity": bool(args.full_gravity),
                    "physical_supports": bool(getattr(cfg, "spawn_physical_supports", False)),
                    "controlled_place": bool(args.controlled_place),
                    "held_contact_and_follow": bool(both_contact and follows_lift),
                    "tip_one_inner_one_outer": bool(tip_pair_pass),
                    "hub_casing_contact": bool(seat_contact),
                    "stable_insert_speed": bool(float(insert_record["hub_speed_mps"]) <= 0.01),
                    "six_dof_pose": bool(six_dof_pass),
                    "bolt_hole_alignment": bool(bolt_alignment_pass),
                    "release_performed": bool(release_performed),
                    "release_contact_free": bool(release_contact_free),
                    "release_drift_m": bool(release_drift <= 0.005),
                    "retract_stable": bool(retract_drift <= 0.02),
                }
                insertion_metrics.update({
                    "radial_error_m": radial_error,
                    "axial_error_m": axial_error,
                    "orientation_error_deg": orientation_error_deg,
                    "bolt_hole_alignment_error_m": bolt_error_values,
                    "bolt_hole_alignment_max_error_m": max(bolt_error_values, default=float("inf")),
                    "tip_contact_topology_records": tip_topology_records,
                    "tip_one_inner_one_outer": bool(tip_pair_pass),
                    "strict_checks": strict_checks,
                })
                strict_success = bool(args.strict_acceptance and all(strict_checks.values()))
                if strict_success:
                    insertion_verdict = "SUCCESS"
        roco_style_score = _roco_style_six_point_score(
            records=records,
            args=args,
            candidate_verdict=candidate_verdict,
            insertion_metrics=insertion_metrics,
        )
        physical_place_score = _physical_place_four_point_score(
            records=records,
            args=args,
            insertion_metrics=insertion_metrics,
        )
        post_release_placement_score = _post_release_placement_score(
            records=records,
            args=args,
            insertion_metrics=insertion_metrics,
        )
        if args.place_acceptance and physical_place_score["total"] == physical_place_score["max"]:
            insertion_verdict = "SUCCESS"
        result = {
            "probe": f"{args.orientation}_inner_wall_candidate",
            "seed": int(args.seed),
            "z_offset_m": float(args.z_offset),
            "radial_offset_m": float(args.radial_offset),
            "opening_target": float(args.opening),
            "safe_approach": bool(args.safe_approach),
            "disable_casing_during_approach": bool(args.disable_casing_during_approach),
            "hub_reset_position_m": records[0]["hub_root_state"][:3],
            "gripper_contact_offset_override_m": args.gripper_contact_offset,
            "hub_displacement_m": _vec(lift_pos - reset_pos),
            "hub_displacement_norm_m": _norm(lift_pos - reset_pos),
            "hub_lift_after_close_m": _vec(hub_lift),
            "hub_lift_after_close_norm_m": _norm(hub_lift),
            "link6_lift_after_close_m": _vec(link6_lift),
            "link6_lift_after_close_norm_m": _norm(link6_lift),
            "max_reported_gripper_contact_force_N": max(contact_values, default=0.0),
            "both_fingers_report_contact": bool(both_contact),
            "candidate_verdict": candidate_verdict,
            "insert_after_lift": bool(args.insert_after_lift),
            "stop_after_lift": bool(args.stop_after_lift),
            "release_before_seat": bool(args.release_before_seat),
            "insert_safe_waypoint": bool(args.insert_safe_waypoint),
            "preserve_lift_grasp_frame": bool(args.preserve_lift_grasp_frame),
            "stop_after_preinsert": bool(args.stop_after_preinsert),
            "release_only_if_seated": bool(args.release_only_if_seated),
            "release_performed": bool(release_performed),
            "release_skipped_not_seated": any(
                record.get("label") == "release_skipped_not_seated" for record in records
            ),
            "closed_loop_preinsert_corrections": int(args.closed_loop_preinsert_corrections),
            "closed_loop_seat_corrections": int(args.closed_loop_seat_corrections),
            "grasp_constraint": bool(args.grasp_constraint),
            "dual_gripper": bool(args.dual_gripper),
            "synchronous_dual_lift": bool(args.synchronous_dual_lift),
            "synchronous_dual_transport": bool(args.synchronous_dual_transport),
            "release_right_after_lift": bool(args.release_right_after_lift),
            "release_right_at_insertion_above": bool(args.release_right_at_insertion_above),
            "release_right_at_transport_segment": int(args.release_right_at_transport_segment),
            "right_release_clearance_y_m": float(args.right_release_clearance_y_m),
            "right_release_deferred_to_insertion_above": bool(
                args.release_right_at_transport_segment and args.release_right_in_place
            ),
            "seat_depth_m": float(args.seat_depth_m),
            "correct_orientation_after_lift": bool(args.correct_orientation_after_lift),
            "correct_orientation_at_insertion_above": bool(
                args.correct_orientation_at_insertion_above
            ),
            "seat_yaw_correction_deg": float(args.seat_yaw_correction_deg),
            "seat_yaw_correction_after_seat_deg": float(
                args.seat_yaw_correction_after_seat_deg
            ),
            "reanchor_preinsert_hold": bool(args.reanchor_preinsert_hold),
            "reanchor_at_insertion_above": bool(args.reanchor_at_insertion_above),
            "freeze_preinsert_hold": bool(args.freeze_preinsert_hold),
            "preinsert_hold_steps": int(args.preinsert_hold_steps),
            "reanchor_before_seat": bool(args.reanchor_before_seat),
            "post_seat_hold_steps": int(args.post_seat_hold_steps),
            "seat_vertical_only": bool(args.seat_vertical_only),
            "seat_position_only": bool(args.seat_position_only),
            "seat_full_pose_ik": bool(args.seat_full_pose_ik),
            "seat_jacobian_position": bool(args.seat_jacobian_position),
            "torso_runtime_override": bool(args.torso_runtime_override),
            "dual_transport": bool(dual_transport),
            "grasp_correction_x_deg": float(args.grasp_correction_x_deg),
            "grasp_correction_y_deg": float(args.grasp_correction_y_deg),
            "closed_loop_insertion": bool(args.closed_loop_insertion),
            "ik_position_only": bool(args.ik_position_only),
            "adaptive_grasp_frame": bool(args.adaptive_grasp_frame),
            "follow_link_orientation": bool(args.follow_link_orientation),
            "rotate_held_orientation": bool(args.rotate_held_orientation),
            "insertion_segments": int(args.insertion_segments),
            "insertion_segment_steps": int(args.insertion_segment_steps),
            "controlled_seat_segments": int(
                args.controlled_seat_segments or args.insertion_segments
            ),
            "controlled_seat_segment_steps": int(
                args.controlled_seat_segment_steps or args.insertion_segment_steps
            ),
            "lift_segments": int(args.lift_segments),
            "lift_segment_steps": int(args.lift_segment_steps),
            "full_gravity": bool(args.full_gravity),
            "physical_supports": bool(getattr(cfg, "spawn_physical_supports", False)),
            "scatter_reset": bool(args.scatter_reset),
            "scatter_reset_report": scatter_report,
            "gripper_contact_static_friction": float(getattr(cfg, "gripper_contact_static_friction", 1.5)),
            "gripper_contact_dynamic_friction": float(getattr(cfg, "gripper_contact_dynamic_friction", 1.5)),
            "gripper_effort_limit_override": args.gripper_effort_limit,
            "gripper_stiffness_override": args.gripper_stiffness,
            "gripper_damping_override": args.gripper_damping,
            "gripper_velocity_limit_override": args.gripper_velocity_limit,
            "arm_effort_limit_override": args.arm_effort_limit,
            "arm_stiffness_override": args.arm_stiffness,
            "arm_damping_override": args.arm_damping,
            "arm_velocity_limit_override": args.arm_velocity_limit,
            "m1_contact_static_friction": float(getattr(cfg, "m1_contact_static_friction", 0.45)),
            "m1_contact_dynamic_friction": float(getattr(cfg, "m1_contact_dynamic_friction", 0.35)),
            "controlled_place": bool(args.controlled_place),
            "release_from_preplace": bool(args.release_from_preplace),
            "preplace_controlled_place": bool(args.preplace_controlled_place),
            "staging_support_drop_m": float(args.staging_support_drop_m),
            "staging_support_retract_steps": int(args.staging_support_retract_steps),
            "staging_support_withdraw_mode": args.staging_support_withdraw_mode,
            "episode_length_s": float(cfg.episode_length_s),
            "preplace_release_clearance_y_m": float(args.preplace_release_clearance_y),
            "preplace_release_clearance_z_m": float(args.preplace_release_clearance_z),
            "preplace_correct_orientation": bool(args.preplace_correct_orientation),
            "track_tip_contact": bool(args.track_tip_contact or args.strict_acceptance),
            "strict_acceptance_requested": bool(args.strict_acceptance),
            "place_acceptance_requested": bool(args.place_acceptance),
            "socket_center_y_m": float(args.socket_center_y),
            "route_socket_y_m": route_socket_y if args.insert_safe_waypoint else None,
            "transport_clearance_z_m": float(args.transport_clearance_z),
            "transport_side_clearance_y_m": float(args.transport_side_clearance_y),
            "transport_side_offset_x_m": float(args.transport_side_offset_x),
            "transport_direct": bool(args.transport_direct),
            "transport_position_only": bool(args.transport_position_only),
            "transport_x_first": bool(args.transport_x_first),
            "preinsert_height_m": float(args.preinsert_height_m),
            "controlled_seat_extra_depth_m": float(args.controlled_seat_extra_depth_m),
            "preinsert_offset_x_m": float(args.preinsert_offset_x_m),
            "preinsert_offset_y_m": float(args.preinsert_offset_y_m),
            # ``--hub-gravity`` is only the legacy release-before-seat toggle.
            # The strict controlled-place route enables gravity at scene
            # construction through ``--full-gravity`` and never calls the
            # mid-air toggle.  Report the physical scene state, not just the
            # legacy CLI flag, so metrics cannot falsely imply that strict
            # full-gravity runs were gravity-disabled.
            "hub_gravity": bool(args.full_gravity or args.hub_gravity),
            "release_opening": float(args.release_opening),
            "release_open_steps": int(args.release_open_steps),
            "freeze_release_open": bool(args.freeze_release_open),
            "gravity_settle_steps": int(args.gravity_settle_steps),
            "post_release_settle_steps": int(args.post_release_settle_steps),
            "release_wrist_roll_deg": float(args.release_wrist_roll_deg),
            "release_clearance_y_m": float(args.release_clearance_y),
            "release_clearance_x_m": float(args.release_clearance_x),
            "release_clearance_z_m": float(args.release_clearance_z),
            "release_clearance_step_m": float(args.release_clearance_step_m),
            "insertion_verdict": insertion_verdict,
            "insertion_metrics": insertion_metrics,
            "roco_style_score": roco_style_score,
            "physical_place_score": physical_place_score,
            "post_release_placement_score": post_release_placement_score,
            "records": records,
            "rgb_frame_names": m0_capture_rgb_frames("post_place") if m0_repair_rgb else [],
            "vader_frame_names": m0_capture_rgb_frames("post_place") if m0_vader_vqa else [],
            "video": str(video_path),
            "video_codec": "H.264/yuv420p",
            "video_frame_size": [int(args.video_width * 2), int(args.video_height * 2)],
            "video_frame_count": int(adapter.video_frame_count),
            "video_error": adapter.video_error,
            "camera_update_stride": int(args.camera_update_stride),
            "render_interval": int(cfg.sim.render_interval),
            "step_scale": float(args.step_scale),
            "video_refresh_semantics": "live_camera_refresh_then_repeat_latest_frame_at_control_rate",
            "interpretation": (
                "SUCCESS in either preplace-only mode means the separate "
                "M1_PHYSICAL_PREPLACE placement baseline passed; it does not "
                "claim pick-and-carry transport. Otherwise SUCCESS is "
                "reserved for the strict full-gravity, supported, "
                "one-inner/one-outer contact, 6DoF-aligned, bolt-aligned, "
                "release-and-retract path. Legacy candidate labels are not "
                "evidence of physical assembly success."
            ),
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("probe", "hub_displacement_m", "hub_displacement_norm_m")}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        # Isaac can propagate SystemExit from a native contact-buffer failure
        # without printing a Python traceback.  Persist a machine-readable
        # artifact so a run that has no metrics is never mistaken for a
        # physical result.
        import traceback

        fatal = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            (args.output_dir / "fatal_error.json").write_text(
                json.dumps(fatal, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        print(json.dumps(fatal, indent=2), flush=True)
        raise
    finally:
        adapter.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
