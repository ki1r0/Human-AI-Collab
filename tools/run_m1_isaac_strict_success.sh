#!/usr/bin/env bash
set -euo pipefail

# Reproducible Isaac Sim strict physics baseline for M1.
# This is deliberately a scripted/oracle controller: ACT is not loaded.  The
# artifact is valid only if metrics.json reports insertion_verdict=SUCCESS.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${M1_ISAAC_STRICT_OUTPUT_DIR:-$ROOT_DIR/validation_logs/m1_isaac_strict_success}"
GPU="${M1_GPU_DEVICE:-0}"
BLOCKED="${M1_BLOCKED:-0}"
STRIDE="${M1_CAMERA_UPDATE_STRIDE:-5}"
RENDER_INTERVAL="${M1_RENDER_INTERVAL:-5}"
VIDEO_WIDTH="${M1_CAMERA_WIDTH:-480}"
VIDEO_HEIGHT="${M1_CAMERA_HEIGHT:-360}"
CAMERA_PROBE="${M1_CAMERA_PROBE_STEPS:-0}"
STEP_SCALE="${M1_STEP_SCALE:-1.0}"
EPISODE_LENGTH_S="${M1_EPISODE_LENGTH_S:-60}"
PREPLACE_RELEASE_CLEARANCE_Y="${M1_PREPLACE_RELEASE_CLEARANCE_Y:-0.0}"
PREPLACE_RELEASE_CLEARANCE_Z="${M1_PREPLACE_RELEASE_CLEARANCE_Z:-0.0}"
PREPLACE_SEAT_EXTRA_DEPTH_M="${M1_PREPLACE_SEAT_EXTRA_DEPTH_M:-0.0}"
CONTROLLED_SEAT_EXTRA_DEPTH_M="${M1_CONTROLLED_SEAT_EXTRA_DEPTH_M:-0.0}"
PREINSERT_HEIGHT_M="${M1_PREINSERT_HEIGHT_M:-0.14}"
PREINSERT_OFFSET_X_M="${M1_PREINSERT_OFFSET_X_M:-0.0}"
PREINSERT_OFFSET_Y_M="${M1_PREINSERT_OFFSET_Y_M:-0.0}"
SOCKET_CENTER_Y_M="${M1_SOCKET_CENTER_Y_M:-0.08683718}"
SEAT_DEPTH_M="${M1_SEAT_DEPTH_M:-0.062}"
Z_OFFSET_M="${M1_Z_OFFSET_M:-0.041}"
PREPLACE_CORRECT_ORIENTATION="${M1_PREPLACE_CORRECT_ORIENTATION:-0}"
GRIP_OPENING="${M1_GRIP_OPENING:-0.005}"
ORIENTATION="${M1_ORIENTATION:-radial}"
RADIAL_OFFSET="${M1_RADIAL_OFFSET:-0.085}"
GRIPPER_STATIC_FRICTION="${M1_GRIPPER_STATIC_FRICTION:-4.0}"
GRIPPER_DYNAMIC_FRICTION="${M1_GRIPPER_DYNAMIC_FRICTION:-3.0}"
GRIPPER_EFFORT_LIMIT="${M1_GRIPPER_EFFORT_LIMIT:-}"
GRIPPER_STIFFNESS="${M1_GRIPPER_STIFFNESS:-}"
GRIPPER_DAMPING="${M1_GRIPPER_DAMPING:-}"
GRIPPER_VELOCITY_LIMIT="${M1_GRIPPER_VELOCITY_LIMIT:-}"
ARM_EFFORT_LIMIT="${M1_ARM_EFFORT_LIMIT:-}"
ARM_STIFFNESS="${M1_ARM_STIFFNESS:-}"
ARM_DAMPING="${M1_ARM_DAMPING:-}"
ARM_VELOCITY_LIMIT="${M1_ARM_VELOCITY_LIMIT:-}"
M1_CONTACT_STATIC_FRICTION="${M1_CONTACT_STATIC_FRICTION:-0.45}"
M1_CONTACT_DYNAMIC_FRICTION="${M1_CONTACT_DYNAMIC_FRICTION:-0.35}"
HUB_MASS_KG="${M1_HUB_MASS_KG:-}"
GRASP_CORRECTION_X="${M1_GRASP_CORRECTION_X_DEG:--4}"
GRASP_CORRECTION_Y="${M1_GRASP_CORRECTION_Y_DEG:--5}"
WRIST_ROLL="${M1_WRIST_ROLL_DEG:-0}"
INSERTION_SEGMENTS="${M1_INSERTION_SEGMENTS:-16}"
INSERTION_STEPS="${M1_INSERTION_SEGMENT_STEPS:-60}"
HELD_ORIENTATION_STEP_DEG="${M1_HELD_ORIENTATION_STEP_DEG:-6.0}"
TRANSPORT_CLEARANCE_Z="${M1_TRANSPORT_CLEARANCE_Z:-0.55}"
TRANSPORT_SIDE_CLEARANCE_Y="${M1_TRANSPORT_SIDE_CLEARANCE_Y:-0.0}"
TRANSPORT_SIDE_OFFSET_X="${M1_TRANSPORT_SIDE_OFFSET_X:-0.0}"
TRANSPORT_X_FIRST="${M1_TRANSPORT_X_FIRST:-0}"
TRANSPORT_DIRECT="${M1_TRANSPORT_DIRECT:-0}"
LIFT_SEGMENTS="${M1_LIFT_SEGMENTS:-8}"
LIFT_STEPS="${M1_LIFT_SEGMENT_STEPS:-60}"
RELEASE_OPEN_STEPS="${M1_RELEASE_OPEN_STEPS:-120}"
RELEASE_OPENING="${M1_RELEASE_OPENING:-0.05}"
FREEZE_RELEASE_OPEN="${M1_FREEZE_RELEASE_OPEN:-0}"
POST_RELEASE_SETTLE_STEPS="${M1_POST_RELEASE_SETTLE_STEPS:-0}"
RELEASE_AT_PREINSERT="${M1_RELEASE_AT_PREINSERT:-0}"
HEADLESS="${M1_HEADLESS:-1}"
NO_VIDEO="${M1_NO_VIDEO:-0}"
STOP_AFTER_LIFT="${M1_STOP_AFTER_LIFT:-0}"
STOP_AFTER_PREINSERT="${M1_STOP_AFTER_PREINSERT:-0}"
FIXED_IK_TARGET="${M1_FIXED_IK_TARGET:-0}"
IK_POSITION_ONLY="${M1_IK_POSITION_ONLY:-0}"
TRANSPORT_POSITION_ONLY="${M1_TRANSPORT_POSITION_ONLY:-0}"
FOLLOW_LINK_ORIENTATION="${M1_FOLLOW_LINK_ORIENTATION:-0}"
ROTATE_HELD_ORIENTATION="${M1_ROTATE_HELD_ORIENTATION:-0}"
CORRECT_ORIENTATION_BEFORE_SEAT="${M1_CORRECT_ORIENTATION_BEFORE_SEAT:-0}"
SEAT_YAW_CORRECTION_DEG="${M1_SEAT_YAW_CORRECTION_DEG:-0}"
SEAT_YAW_CORRECTION_AFTER_SEAT_DEG="${M1_SEAT_YAW_CORRECTION_AFTER_SEAT_DEG:-0}"
REANCHOR_PREINSERT_HOLD="${M1_REANCHOR_PREINSERT_HOLD:-0}"
REANCHOR_AT_INSERTION_ABOVE="${M1_REANCHOR_AT_INSERTION_ABOVE:-0}"
CORRECT_ORIENTATION_AT_INSERTION_ABOVE="${M1_CORRECT_ORIENTATION_AT_INSERTION_ABOVE:-0}"
FREEZE_PREINSERT_HOLD="${M1_FREEZE_PREINSERT_HOLD:-0}"
PREINSERT_HOLD_STEPS="${M1_PREINSERT_HOLD_STEPS:-40}"
POST_SEAT_HOLD_STEPS="${M1_POST_SEAT_HOLD_STEPS:-0}"
CONTROLLED_SEAT_SEGMENTS="${M1_CONTROLLED_SEAT_SEGMENTS:-0}"
CONTROLLED_SEAT_STEPS="${M1_CONTROLLED_SEAT_SEGMENT_STEPS:-0}"
RELEASE_ONLY_IF_SEATED="${M1_RELEASE_ONLY_IF_SEATED:-0}"
PREINSERT_CORRECTIONS="${M1_CLOSED_LOOP_PREINSERT_CORRECTIONS:-0}"
SEAT_CORRECTIONS="${M1_CLOSED_LOOP_SEAT_CORRECTIONS:-0}"
REANCHOR_BEFORE_SEAT="${M1_REANCHOR_BEFORE_SEAT:-0}"
SEAT_VERTICAL_ONLY="${M1_SEAT_VERTICAL_ONLY:-0}"
SEAT_POSITION_ONLY="${M1_SEAT_POSITION_ONLY:-0}"
SEAT_FULL_POSE_IK="${M1_SEAT_FULL_POSE_IK:-0}"
SEAT_JACOBIAN_POSITION="${M1_SEAT_JACOBIAN_POSITION:-0}"
TORSO_RUNTIME_OVERRIDE="${M1_TORSO_RUNTIME_OVERRIDE:-0}"
ADAPTIVE_GRASP_FRAME="${M1_ADAPTIVE_GRASP_FRAME:-0}"
RETRACT_STAGING_SUPPORTS="${M1_RETRACT_STAGING_SUPPORTS:-0}"
STAGING_SUPPORT_DROP_M="${M1_STAGING_SUPPORT_DROP_M:-0.05}"
STAGING_SUPPORT_RETRACT_STEPS="${M1_STAGING_SUPPORT_RETRACT_STEPS:-1}"
STAGING_SUPPORT_WITHDRAW_MODE="${M1_STAGING_SUPPORT_WITHDRAW_MODE:-down}"
RELEASE_FROM_PREPLACE="${M1_RELEASE_FROM_PREPLACE:-0}"
PREPLACE_CONTROLLED_PLACE="${M1_PREPLACE_CONTROLLED_PLACE:-0}"
DUAL_GRIPPER="${M1_DUAL_GRIPPER:-0}"
RELEASE_RIGHT_AFTER_LIFT="${M1_RELEASE_RIGHT_AFTER_LIFT:-0}"
RELEASE_RIGHT_AT_INSERTION_ABOVE="${M1_RELEASE_RIGHT_AT_INSERTION_ABOVE:-0}"
RELEASE_RIGHT_AT_TRANSPORT_SEGMENT="${M1_RELEASE_RIGHT_AT_TRANSPORT_SEGMENT:-0}"
RELEASE_RIGHT_IN_PLACE="${M1_RELEASE_RIGHT_IN_PLACE:-0}"
FREEZE_RIGHT_RELEASE_OPEN="${M1_FREEZE_RIGHT_RELEASE_OPEN:-0}"
HOLD_RIGHT_RELEASE_POSE="${M1_HOLD_RIGHT_RELEASE_POSE:-0}"
RIGHT_RELEASE_CLEARANCE_Y="${M1_RIGHT_RELEASE_CLEARANCE_Y_M:-0.0}"
SYNCHRONOUS_DUAL_LIFT="${M1_SYNCHRONOUS_DUAL_LIFT:-0}"
SYNCHRONOUS_DUAL_TRANSPORT="${M1_SYNCHRONOUS_DUAL_TRANSPORT:-0}"
CORRECT_ORIENTATION_AFTER_LIFT="${M1_CORRECT_ORIENTATION_AFTER_LIFT:-0}"
M0_SERVE="${M1_M0_SERVE:-0}"
M0_VADER_VQA="${M1_M0_VADER_VQA:-0}"
M0_REPAIR_RGB="${M1_M0_REPAIR_RGB:-0}"
M0_INITIAL_RGB_OBSERVE="${M1_M0_INITIAL_RGB_OBSERVE:-0}"
M0_REPLAN_AFTER_HELP="${M1_M0_REPLAN_AFTER_HELP:-0}"
M0_STEPWISE_ACTIONS="${M1_M0_STEPWISE_ACTIONS:-0}"
M0_HELP_TIMEOUT_S="${M1_M0_HELP_TIMEOUT_S:-900}"
M0_BLOCKED_ATTEMPT_STEPS="${M1_M0_BLOCKED_ATTEMPT_STEPS:-40}"
M0_PRECONTACT_GUARD="${M1_M0_PRECONTACT_GUARD:-0}"
M0_HELP_SETTLE_STEPS="${M1_M0_HELP_SETTLE_STEPS:-50}"
M0_HELP_POLL_STEP_INTERVAL="${M1_M0_HELP_POLL_STEP_INTERVAL:-1}"
M0_HOLD_OPENING="${M1_M0_HOLD_OPENING:-0.005}"
M0_DEFER_STAGING_SUPPORTS="${M1_M0_DEFER_STAGING_SUPPORT_RETRACTION:-0}"
M0_REGRASP_STEPS="${M1_M0_REGRASP_STEPS:-60}"
M0_FREEZE_RELEASE_JOINTS="${M1_M0_FREEZE_RELEASE_JOINTS:-0}"
M0_RETRACT_SEGMENTS="${M1_M0_RETRACT_SEGMENTS:-1}"
M0_RETRACT_SEGMENT_STEPS="${M1_M0_RETRACT_SEGMENT_STEPS:-20}"
M0_RETRACT_STEP_Z_M="${M1_M0_RETRACT_STEP_Z_M:-0.02}"
BLOCKER_PROFILE="${HRC_M1_BLOCKER_PROFILE:-default}"
PLACE_ACCEPTANCE="${M1_PLACE_ACCEPTANCE:-1}"
TRACK_TIP_CONTACT="${M1_TRACK_TIP_CONTACT:-1}"
STRICT_ACCEPTANCE="${M1_STRICT_ACCEPTANCE:-1}"
GRASP_CONSTRAINT="${M1_GRASP_CONSTRAINT:-0}"
FILTER_ROBOT_TABLE_COLLISIONS="${M1_FILTER_ROBOT_TABLE_COLLISIONS:-0}"
HUB_RESET_X="${M1_HUB_RESET_X:-}"
HUB_RESET_Y="${M1_HUB_RESET_Y:-}"
HUB_RESET_Z="${M1_HUB_RESET_Z:-}"
SCATTER_RESET="${M1_SCATTER_RESET:-0}"
TABLE_SIZE_X="${M1_TABLE_SIZE_X:-1.8}"
TABLE_SIZE_Y="${M1_TABLE_SIZE_Y:-1.8}"
TABLE_TOP_Z="${M1_TABLE_TOP_Z:-0.934}"
TABLE_CENTER_X="${M1_TABLE_CENTER_X:-1.20}"
TABLE_CENTER_Y="${M1_TABLE_CENTER_Y:-0.0}"
CASING_RESET_X="${M1_CASING_RESET_X:-0.55}"
CASING_RESET_Y="${M1_CASING_RESET_Y:-0.0}"
SCATTER_SPAWN_MARGIN="${M1_SCATTER_SPAWN_MARGIN:-0.002}"
SCATTER_PHYSICAL_SUPPORTS="${M1_SCATTER_PHYSICAL_SUPPORTS:-0}"
SCATTER_SUPPORT_TOP_Z="${M1_SCATTER_SUPPORT_TOP_Z:-0.934}"
SCATTER_HUB_SUPPORT_TOP_Z="${M1_SCATTER_HUB_SUPPORT_TOP_Z:-}"
DOCKER_DISPLAY_ARGS=()
DOCKER_LIFECYCLE_ARGS=(--rm)
DOCKER_IPC_ARGS=(--ipc=host)
if [[ "${M1_ISAAC_IPC_MODE:-host}" == "private" ]]; then
  DOCKER_IPC_ARGS=(--ipc=private --shm-size="${M1_ISAAC_SHM_SIZE:-8g}")
fi
if [[ -n "${M1_DOCKER_NAME:-}" ]]; then
  DOCKER_LIFECYCLE_ARGS=(--rm --name "$M1_DOCKER_NAME")
fi
APP_ARGS=(--headless --enable_cameras)
VIDEO_ARGS=()
PHASE_ARGS=()
RESET_ARGS=()
SCENE_ARGS=()
PHYSICS_ARGS=(--full-gravity)
CALIBRATION_ARGS=()
ROUTE_ARGS=(--insert-safe-waypoint)
PLACE_MODE_ARGS=(--controlled-place)
if [[ "$NO_VIDEO" == "1" ]]; then
  APP_ARGS=(--headless)
  VIDEO_ARGS=(--no-video)
fi
if [[ "$STOP_AFTER_LIFT" == "1" ]]; then
  PHASE_ARGS=(--stop-after-lift)
fi
if [[ "$STOP_AFTER_PREINSERT" == "1" ]]; then
  PHASE_ARGS+=(--stop-after-preinsert)
fi
if [[ "$FIXED_IK_TARGET" == "1" ]]; then
  PHASE_ARGS+=(--fixed-ik-target)
fi
if [[ "$IK_POSITION_ONLY" == "1" ]]; then
  PHASE_ARGS+=(--ik-position-only)
fi
if [[ "$TRANSPORT_POSITION_ONLY" == "1" ]]; then
  PHASE_ARGS+=(--transport-position-only)
fi
if [[ "$FOLLOW_LINK_ORIENTATION" == "1" ]]; then
  PHASE_ARGS+=(--follow-link-orientation)
fi
if [[ "$ROTATE_HELD_ORIENTATION" == "1" ]]; then
  PHASE_ARGS+=(--rotate-held-orientation)
fi
if [[ "$CORRECT_ORIENTATION_BEFORE_SEAT" == "1" ]]; then
  PHASE_ARGS+=(--correct-orientation-before-seat)
fi
if [[ "$SEAT_YAW_CORRECTION_DEG" != "0" ]]; then
  PHASE_ARGS+=(--seat-yaw-correction-deg "$SEAT_YAW_CORRECTION_DEG")
fi
if [[ "$SEAT_YAW_CORRECTION_AFTER_SEAT_DEG" != "0" ]]; then
  PHASE_ARGS+=(--seat-yaw-correction-after-seat-deg "$SEAT_YAW_CORRECTION_AFTER_SEAT_DEG")
fi
if [[ "$REANCHOR_PREINSERT_HOLD" == "1" ]]; then
  PHASE_ARGS+=(--reanchor-preinsert-hold)
fi
if [[ "$REANCHOR_AT_INSERTION_ABOVE" == "1" ]]; then
  PHASE_ARGS+=(--reanchor-at-insertion-above)
fi
if [[ "$CORRECT_ORIENTATION_AT_INSERTION_ABOVE" == "1" ]]; then
  PHASE_ARGS+=(--correct-orientation-at-insertion-above)
fi
if [[ "$FREEZE_PREINSERT_HOLD" == "1" ]]; then
  PHASE_ARGS+=(--freeze-preinsert-hold)
fi
PHASE_ARGS+=(--preinsert-hold-steps "$PREINSERT_HOLD_STEPS")
if [[ "$POST_SEAT_HOLD_STEPS" != "0" ]]; then
  PHASE_ARGS+=(--post-seat-hold-steps "$POST_SEAT_HOLD_STEPS")
fi
if [[ "$CONTROLLED_SEAT_SEGMENTS" != "0" ]]; then
  PHASE_ARGS+=(--controlled-seat-segments "$CONTROLLED_SEAT_SEGMENTS")
fi
if [[ "$CONTROLLED_SEAT_STEPS" != "0" ]]; then
  PHASE_ARGS+=(--controlled-seat-segment-steps "$CONTROLLED_SEAT_STEPS")
fi
if [[ "$RELEASE_ONLY_IF_SEATED" == "1" ]]; then
  PHASE_ARGS+=(--release-only-if-seated)
fi
if [[ "$PREINSERT_CORRECTIONS" != "0" ]]; then
  PHASE_ARGS+=(--closed-loop-preinsert-corrections "$PREINSERT_CORRECTIONS")
fi
if [[ "$SEAT_CORRECTIONS" != "0" ]]; then
  PHASE_ARGS+=(--closed-loop-seat-corrections "$SEAT_CORRECTIONS")
fi
if [[ "$REANCHOR_BEFORE_SEAT" == "1" ]]; then
  PHASE_ARGS+=(--reanchor-before-seat)
fi
if [[ "$SEAT_VERTICAL_ONLY" == "1" ]]; then
  PHASE_ARGS+=(--seat-vertical-only)
fi
if [[ "$SEAT_POSITION_ONLY" == "1" ]]; then
  PHASE_ARGS+=(--seat-position-only)
fi
if [[ "$SEAT_FULL_POSE_IK" == "1" ]]; then
  PHASE_ARGS+=(--seat-full-pose-ik)
fi
if [[ "$SEAT_JACOBIAN_POSITION" == "1" ]]; then
  PHASE_ARGS+=(--seat-jacobian-position)
fi
if [[ "$TORSO_RUNTIME_OVERRIDE" == "1" ]]; then
  PHASE_ARGS+=(--torso-runtime-override)
fi
if [[ "$ADAPTIVE_GRASP_FRAME" == "1" ]]; then
  PHASE_ARGS+=(--adaptive-grasp-frame)
fi
if [[ "$FILTER_ROBOT_TABLE_COLLISIONS" == "1" ]]; then
  PHASE_ARGS+=(--filter-robot-table-collisions)
fi
if [[ "$RETRACT_STAGING_SUPPORTS" == "1" ]]; then
  PHASE_ARGS+=(--retract-staging-supports-after-grasp)
fi
if [[ "$RELEASE_FROM_PREPLACE" == "1" ]]; then
  PLACE_MODE_ARGS=(--release-from-preplace)
fi
if [[ "$PREPLACE_CONTROLLED_PLACE" == "1" ]]; then
  PLACE_MODE_ARGS=(--preplace-controlled-place)
fi
if [[ "$PREPLACE_CORRECT_ORIENTATION" == "1" ]]; then
  PHASE_ARGS+=(--preplace-correct-orientation)
fi
if [[ "$RELEASE_AT_PREINSERT" == "1" ]]; then
  PHASE_ARGS+=(--release-at-preinsert)
fi
if [[ "$FREEZE_RELEASE_OPEN" == "1" ]]; then
  PHASE_ARGS+=(--freeze-release-open)
fi
if [[ "$POST_RELEASE_SETTLE_STEPS" != "0" ]]; then
  PHASE_ARGS+=(--post-release-settle-steps "$POST_RELEASE_SETTLE_STEPS")
fi
if [[ "$DUAL_GRIPPER" == "1" ]]; then
  PHASE_ARGS+=(--dual-gripper)
fi
if [[ "$RELEASE_RIGHT_AFTER_LIFT" == "1" ]]; then
  PHASE_ARGS+=(--release-right-after-lift)
fi
if [[ "$RELEASE_RIGHT_AT_INSERTION_ABOVE" == "1" ]]; then
  PHASE_ARGS+=(--release-right-at-insertion-above)
fi
if [[ "$RELEASE_RIGHT_AT_TRANSPORT_SEGMENT" != "0" ]]; then
  PHASE_ARGS+=(--release-right-at-transport-segment "$RELEASE_RIGHT_AT_TRANSPORT_SEGMENT")
fi
if [[ "$RELEASE_RIGHT_IN_PLACE" == "1" ]]; then
  PHASE_ARGS+=(--release-right-in-place)
fi
if [[ "$FREEZE_RIGHT_RELEASE_OPEN" == "1" ]]; then
  PHASE_ARGS+=(--freeze-right-release-open)
fi
if [[ "$HOLD_RIGHT_RELEASE_POSE" == "1" ]]; then
  PHASE_ARGS+=(--hold-right-release-pose)
fi
if [[ "$RIGHT_RELEASE_CLEARANCE_Y" != "0.0" ]]; then
  PHASE_ARGS+=(--right-release-clearance-y-m "$RIGHT_RELEASE_CLEARANCE_Y")
fi
if [[ "$SYNCHRONOUS_DUAL_LIFT" == "1" ]]; then
  PHASE_ARGS+=(--synchronous-dual-lift)
fi
if [[ "$SYNCHRONOUS_DUAL_TRANSPORT" == "1" ]]; then
  PHASE_ARGS+=(--synchronous-dual-transport)
fi
if [[ "$CORRECT_ORIENTATION_AFTER_LIFT" == "1" ]]; then
  PHASE_ARGS+=(--correct-orientation-after-lift)
fi
if [[ "$M0_SERVE" == "1" ]]; then
  PHASE_ARGS+=(--m0-serve --m0-help-timeout-s "$M0_HELP_TIMEOUT_S" \
    --m0-blocked-attempt-steps "$M0_BLOCKED_ATTEMPT_STEPS" \
    --m0-help-settle-steps "$M0_HELP_SETTLE_STEPS" \
    --m0-help-poll-step-interval "$M0_HELP_POLL_STEP_INTERVAL" \
    --m0-hold-opening "$M0_HOLD_OPENING")
  if [[ "$M0_PRECONTACT_GUARD" == "1" ]]; then
    PHASE_ARGS+=(--m0-precontact-guard)
  fi
  if [[ "$M0_DEFER_STAGING_SUPPORTS" == "1" ]]; then
    PHASE_ARGS+=(--m0-defer-staging-support-retraction --m0-regrasp-steps "$M0_REGRASP_STEPS")
  fi
  if [[ "$M0_FREEZE_RELEASE_JOINTS" == "1" ]]; then
    PHASE_ARGS+=(--m0-freeze-release-joints)
  fi
  PHASE_ARGS+=(--m0-retract-segments "$M0_RETRACT_SEGMENTS" \
    --m0-retract-segment-steps "$M0_RETRACT_SEGMENT_STEPS" \
    --m0-retract-step-z-m "$M0_RETRACT_STEP_Z_M")
fi
if [[ "$PLACE_ACCEPTANCE" == "1" ]]; then
  PHASE_ARGS+=(--place-acceptance)
fi
if [[ -n "$HUB_RESET_X" ]]; then
  RESET_ARGS+=(--hub-reset-x "$HUB_RESET_X")
fi
if [[ -n "$HUB_RESET_Y" ]]; then
  RESET_ARGS+=(--hub-reset-y "$HUB_RESET_Y")
fi
if [[ -n "$HUB_RESET_Z" ]]; then
  RESET_ARGS+=(--hub-reset-z "$HUB_RESET_Z")
fi
if [[ "$SCATTER_RESET" == "1" ]]; then
  SCENE_ARGS+=(
    --scatter-reset
    --table-size-x "$TABLE_SIZE_X" --table-size-y "$TABLE_SIZE_Y"
    --table-top-z "$TABLE_TOP_Z"
    --table-center-x "$TABLE_CENTER_X" --table-center-y "$TABLE_CENTER_Y"
    --casing-reset-x "$CASING_RESET_X" --casing-reset-y "$CASING_RESET_Y"
    --scatter-spawn-margin-m "$SCATTER_SPAWN_MARGIN"
  )
  if [[ "$SCATTER_PHYSICAL_SUPPORTS" == "1" ]]; then
    SCENE_ARGS+=(--scatter-physical-supports --scatter-support-top-z "$SCATTER_SUPPORT_TOP_Z")
    if [[ -n "$SCATTER_HUB_SUPPORT_TOP_Z" ]]; then
      SCENE_ARGS+=(--scatter-hub-support-top-z "$SCATTER_HUB_SUPPORT_TOP_Z")
    fi
    PHYSICS_ARGS+=(--physical-supports)
  fi
else
  PHYSICS_ARGS+=(--physical-supports)
fi
if [[ "${M1_INSERT_SAFE_WAYPOINT:-1}" == "0" ]]; then
  ROUTE_ARGS=()
fi
if [[ "$TRANSPORT_X_FIRST" == "1" ]]; then
  PHASE_ARGS+=(--transport-x-first)
fi
if [[ "$TRANSPORT_DIRECT" == "1" ]]; then
  PHASE_ARGS+=(--transport-direct)
fi
if [[ -n "$HUB_MASS_KG" ]]; then
  CALIBRATION_ARGS+=(--hub-mass-kg "$HUB_MASS_KG")
fi
if [[ -n "$GRIPPER_EFFORT_LIMIT" ]]; then
  CALIBRATION_ARGS+=(--gripper-effort-limit "$GRIPPER_EFFORT_LIMIT")
fi
if [[ -n "$GRIPPER_STIFFNESS" ]]; then
  CALIBRATION_ARGS+=(--gripper-stiffness "$GRIPPER_STIFFNESS")
fi
if [[ -n "$GRIPPER_DAMPING" ]]; then
  CALIBRATION_ARGS+=(--gripper-damping "$GRIPPER_DAMPING")
fi
if [[ -n "$GRIPPER_VELOCITY_LIMIT" ]]; then
  CALIBRATION_ARGS+=(--gripper-velocity-limit "$GRIPPER_VELOCITY_LIMIT")
fi
if [[ -n "$ARM_EFFORT_LIMIT" ]]; then
  CALIBRATION_ARGS+=(--arm-effort-limit "$ARM_EFFORT_LIMIT")
fi
if [[ -n "$ARM_STIFFNESS" ]]; then
  CALIBRATION_ARGS+=(--arm-stiffness "$ARM_STIFFNESS")
fi
if [[ -n "$ARM_DAMPING" ]]; then
  CALIBRATION_ARGS+=(--arm-damping "$ARM_DAMPING")
fi
if [[ -n "$ARM_VELOCITY_LIMIT" ]]; then
  CALIBRATION_ARGS+=(--arm-velocity-limit "$ARM_VELOCITY_LIMIT")
fi
ACCEPTANCE_ARGS=()
if [[ "$TRACK_TIP_CONTACT" == "1" ]]; then
  ACCEPTANCE_ARGS+=(--track-tip-contact)
fi
if [[ "$STRICT_ACCEPTANCE" == "1" ]]; then
  ACCEPTANCE_ARGS+=(--strict-acceptance)
fi
if [[ "$GRASP_CONSTRAINT" == "1" ]]; then
  ACCEPTANCE_ARGS+=(--grasp-constraint)
fi
if [[ "$HEADLESS" == "0" ]]; then
  : "${DISPLAY:?M1_HEADLESS=0 requires DISPLAY}"
  : "${XAUTHORITY:?M1_HEADLESS=0 requires XAUTHORITY}"
  [[ -d /tmp/.X11-unix ]] || { echo "M1_HEADLESS=0 requires /tmp/.X11-unix" >&2; exit 2; }
  [[ -f "$XAUTHORITY" ]] || { echo "XAUTHORITY is not a readable file: $XAUTHORITY" >&2; exit 2; }
  DOCKER_DISPLAY_ARGS=(
    -e "DISPLAY=$DISPLAY"
    -e XAUTHORITY=/tmp/m1.Xauthority
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "$XAUTHORITY:/tmp/m1.Xauthority:ro"
  )
  if [[ "$NO_VIDEO" == "1" ]]; then
    APP_ARGS=()
  else
    APP_ARGS=(--enable_cameras)
  fi
fi
mkdir -p "$OUT"

exec docker run "${DOCKER_LIFECYCLE_ARGS[@]}" --gpus "device=$GPU" "${DOCKER_IPC_ARGS[@]}" --network=host "${DOCKER_DISPLAY_ARGS[@]}" \
  --entrypoint /isaac-sim/python.sh -w /workspace/Human-AI-Collab \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e HRC_M1_ROOT=/workspace/Human-AI-Collab \
  -e "HRC_M1_BLOCKED=$BLOCKED" \
  -e "HRC_M1_BLOCKER_PROFILE=$BLOCKER_PROFILE" \
  -e "M1_M0_VADER_VQA=$M0_VADER_VQA" \
  -e "M1_M0_REPAIR_RGB=$M0_REPAIR_RGB" \
  -e "M1_M0_INITIAL_RGB_OBSERVE=$M0_INITIAL_RGB_OBSERVE" \
  -e "M1_M0_REPLAN_AFTER_HELP=$M0_REPLAN_AFTER_HELP" \
  -e "M1_M0_STEPWISE_ACTIONS=$M0_STEPWISE_ACTIONS" \
  -e PYTHONPATH=/workspace/Human-AI-Collab:/workspace/gearboxAssembly/source/Galaxea_Lab_External \
  -v "$ROOT_DIR:/workspace/Human-AI-Collab" \
  -v /home/sunsiliang/roco_runtime/gearboxAssembly:/workspace/gearboxAssembly:ro \
  nvcr.io/nvidia/isaac-lab:2.3.0 -m hrc_m1.debug_inner_wall_grasp \
  --output-dir "/workspace/Human-AI-Collab/$(realpath --relative-to="$ROOT_DIR" "$OUT")" \
  --orientation "$ORIENTATION" --radial-offset "$RADIAL_OFFSET" --opening "$GRIP_OPENING" --approach-opening 0.045 \
  --gripper-static-friction "$GRIPPER_STATIC_FRICTION" --gripper-dynamic-friction "$GRIPPER_DYNAMIC_FRICTION" \
  --m1-contact-static-friction "$M1_CONTACT_STATIC_FRICTION" --m1-contact-dynamic-friction "$M1_CONTACT_DYNAMIC_FRICTION" \
  --z-offset "$Z_OFFSET_M" --gripper-contact-offset 0.0001 \
  --grasp-correction-x-deg="${GRASP_CORRECTION_X}" --grasp-correction-y-deg="${GRASP_CORRECTION_Y}" \
  --wrist-roll-deg "$WRIST_ROLL" \
  --preserve-lift-grasp-frame --insert-after-lift "${ROUTE_ARGS[@]}" \
  --transport-clearance-z "$TRANSPORT_CLEARANCE_Z" \
  --transport-side-clearance-y "$TRANSPORT_SIDE_CLEARANCE_Y" \
  --transport-side-offset-x "$TRANSPORT_SIDE_OFFSET_X" \
  "${PHYSICS_ARGS[@]}" "${SCENE_ARGS[@]}" "${PLACE_MODE_ARGS[@]}" \
  --staging-support-drop-m "$STAGING_SUPPORT_DROP_M" \
  --staging-support-retract-steps "$STAGING_SUPPORT_RETRACT_STEPS" \
  --staging-support-withdraw-mode "$STAGING_SUPPORT_WITHDRAW_MODE" \
  --preinsert-height-m "$PREINSERT_HEIGHT_M" \
  --preinsert-offset-x-m "$PREINSERT_OFFSET_X_M" --preinsert-offset-y-m "$PREINSERT_OFFSET_Y_M" \
  --socket-center-y "$SOCKET_CENTER_Y_M" \
  --seat-depth-m "$SEAT_DEPTH_M" \
  --release-opening "$RELEASE_OPENING" \
  --lift-segments "$LIFT_SEGMENTS" --lift-segment-steps "$LIFT_STEPS" \
  --closed-loop-insertion --insertion-segments "$INSERTION_SEGMENTS" \
  --insertion-segment-steps "$INSERTION_STEPS" --held-orientation-step-deg "$HELD_ORIENTATION_STEP_DEG" \
  --release-open-steps "$RELEASE_OPEN_STEPS" \
  --step-scale "$STEP_SCALE" --camera-update-stride "$STRIDE" --render-interval "$RENDER_INTERVAL" \
  --episode-length-s "$EPISODE_LENGTH_S" \
  --preplace-release-clearance-y "$PREPLACE_RELEASE_CLEARANCE_Y" \
  --preplace-release-clearance-z "$PREPLACE_RELEASE_CLEARANCE_Z" \
  --preplace-seat-extra-depth-m "$PREPLACE_SEAT_EXTRA_DEPTH_M" \
  --controlled-seat-extra-depth-m "$CONTROLLED_SEAT_EXTRA_DEPTH_M" \
  --video-width "$VIDEO_WIDTH" --video-height "$VIDEO_HEIGHT" \
  --camera-probe "$CAMERA_PROBE" \
  "${RESET_ARGS[@]}" \
  "${CALIBRATION_ARGS[@]}" \
  "${ACCEPTANCE_ARGS[@]}" \
  "${VIDEO_ARGS[@]}" \
  "${PHASE_ARGS[@]}" \
  "${APP_ARGS[@]}"
