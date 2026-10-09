"""Synthetic evaluator unit histories only; these are not physical success evidence."""

import unittest
from dataclasses import replace
from math import cos, sin

from runtime.bolt_harness.evaluator import (
    BoltNativePoseSample,
    BoltPoseWindowTolerances,
    BoltSeatMeasurement,
    BoltSeatStatus,
    BoltSeatTolerances,
    IndependentBoltSeatingEvaluator,
)


TOLERANCES = BoltSeatTolerances(
    max_cap_support_gap_m=0.002,
    min_casing_entry_depth_m=0.01,
    max_shaft_radial_error_m=0.001,
    max_penetration_m=0.0005,
    max_axial_motion_since_release_m=0.001,
    max_relative_linear_speed_mps=0.001,
    max_relative_angular_speed_radps=0.01,
    held_seat_dwell_s=0.25,
    stable_dwell_s=1.0,
    max_sample_gap_s=0.25,
)


def sample(time_s, **changes):
    values = {
        "time_s": time_s,
        "through_selected_cover_hole": True,
        "entered_selected_casing_opening": True,
        "casing_entry_depth_m": 0.02,
        "shaft_radial_error_m": 0.0005,
        "cap_support_gap_m": 0.0005,
        "cap_support_contact": True,
        "penetration_m": 0.0001,
        "premature_bottoming": False,
        "physically_held": True,
        "insertion_provenance_valid": True,
        "controller_stopped": True,
        "robot_contact_or_support": True,
        "axial_position_m": 0.05,
        "relative_linear_speed_mps": 0.0,
        "relative_angular_speed_radps": 0.0,
    }
    return BoltSeatMeasurement(**(values | changes))


def release_sample(time_s, **changes):
    return sample(time_s, **({
        "physically_held": False,
        "robot_contact_or_support": False,
    } | changes))


def held_seat(evaluator, **changes):
    return [evaluator.observe(sample(time_s, **changes)) for time_s in (0.0, 0.125, 0.25)]


def finish_stable_dwell(evaluator, release_time=0.375, axial_position_m=0.05):
    results = []
    for time_s in (release_time + 0.25, release_time + 0.5,
                   release_time + 0.75, release_time + 1.0):
        results.append(evaluator.observe(release_sample(time_s, axial_position_m=axial_position_m)))
    return results


POSE_SEAT_TOLERANCES = replace(
    TOLERANCES,
    held_seat_dwell_s=0.2,
    stable_dwell_s=0.4,
    max_sample_gap_s=0.11,
)
POSE_WINDOW_TOLERANCES = BoltPoseWindowTolerances(
    max_position_excursion_m=0.0001,
    max_orientation_excursion_rad=0.001,
)


def pose_sample(index, time_s, *, position=(0.0, 0.0, 0.0), angle_rad=0.0):
    return BoltNativePoseSample(
        physics_step_index=index,
        time_s=time_s,
        root_pos_w_m=position,
        root_quat_wxyz=(cos(angle_rad / 2.0), 0.0, 0.0, sin(angle_rad / 2.0)),
        native_linear_velocity_w_mps=(0.02, 0.0, 0.0),
        native_angular_velocity_w_radps=(0.0, 0.0, 1.32),
    )


def pose_measurement(
    time_s,
    first_index,
    *,
    mid_angle_rad=0.0,
    start_position=(0.0, 0.0, 0.0),
    end_position=(0.0, 0.0, 0.0),
    mid_position=None,
    **changes,
):
    dt_s = 1.0 / 120.0
    pose_samples = []
    for offset in range(12):
        fraction = offset / 11.0
        position = tuple(
            start_position[axis] + fraction * (end_position[axis] - start_position[axis])
            for axis in range(3)
        )
        angle_rad = 0.0
        if offset == 5:
            angle_rad = mid_angle_rad
            if mid_position is not None:
                position = mid_position
        pose_samples.append(
            pose_sample(
                first_index + offset,
                time_s - (11 - offset) * dt_s,
                position=position,
                angle_rad=angle_rad,
            )
        )
    return sample(
        time_s,
        relative_linear_speed_mps=0.02,
        relative_angular_speed_radps=1.32,
        native_pose_samples=tuple(pose_samples),
        native_pose_dt_s=dt_s,
        **changes,
    )


class BoltHarnessEvaluatorTests(unittest.TestCase):
    def test_pose_window_allows_stable_pose_with_large_native_velocity_diagnostics(self):
        evaluator = IndependentBoltSeatingEvaluator(
            POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
        )
        statuses = [
            evaluator.observe(pose_measurement(time_s, index))
            for time_s, index in ((0.0, 0), (0.1, 12), (0.2, 24))
        ]
        self.assertEqual(statuses, [BoltSeatStatus(False, False), BoltSeatStatus(False, False), BoltSeatStatus(True, False)])

        released = pose_measurement(
            0.3, 36, physically_held=False, robot_contact_or_support=False
        )
        self.assertEqual(evaluator.observe(released), BoltSeatStatus(False, False))
        results = [
            evaluator.observe(
                pose_measurement(
                    time_s,
                    index,
                    physically_held=False,
                    robot_contact_or_support=False,
                )
            )
            for time_s, index in ((0.4, 48), (0.5, 60), (0.6, 72), (0.7, 84))
        ]
        self.assertTrue(results[-1].task_success)
        self.assertEqual(released.relative_linear_speed_mps, 0.02)
        self.assertEqual(released.relative_angular_speed_radps, 1.32)
        self.assertEqual(released.native_pose_samples[-1].native_angular_velocity_w_radps, (0.0, 0.0, 1.32))

    def test_pose_window_catches_intermediate_rotation_hidden_at_control_sample(self):
        evaluator = IndependentBoltSeatingEvaluator(
            POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
        )
        evaluator.observe(pose_measurement(0.0, 0))
        # Both control-step endpoints have the same orientation; the 120 Hz
        # intermediate sample exceeds the configured excursion and restarts dwell.
        hidden_motion = pose_measurement(0.1, 12, mid_angle_rad=0.002)
        self.assertEqual(evaluator.observe(hidden_motion), BoltSeatStatus(False, False))
        self.assertEqual(
            evaluator.observe(pose_measurement(0.2, 24)),
            BoltSeatStatus(False, False),
        )

    def test_pose_window_rejects_cumulative_drift_below_per_tick_velocity_gate(self):
        evaluator = IndependentBoltSeatingEvaluator(
            POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
        )
        evaluator.observe(pose_measurement(0.0, 0))
        evaluator.observe(
            pose_measurement(
                0.1,
                12,
                start_position=(0.0, 0.0, 0.0),
                end_position=(0.00006, 0.0, 0.0),
            )
        )
        status = evaluator.observe(
            pose_measurement(
                0.2,
                24,
                start_position=(0.00006, 0.0, 0.0),
                end_position=(0.00012, 0.0, 0.0),
            )
        )

        self.assertEqual(status, BoltSeatStatus(False, False))

    def test_pose_window_restarts_release_dwell_on_intermediate_motion(self):
        evaluator = IndependentBoltSeatingEvaluator(
            POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
        )
        for time_s, index in ((0.0, 0), (0.1, 12), (0.2, 24)):
            evaluator.observe(pose_measurement(time_s, index))
        evaluator.observe(
            pose_measurement(0.3, 36, physically_held=False, robot_contact_or_support=False)
        )
        hidden_motion = evaluator.observe(
            pose_measurement(
                0.4,
                48,
                mid_angle_rad=0.002,
                physically_held=False,
                robot_contact_or_support=False,
            )
        )
        self.assertEqual(hidden_motion, BoltSeatStatus(False, False))
        for time_s, index in ((0.5, 60), (0.6, 72), (0.7, 84)):
            self.assertEqual(
                evaluator.observe(
                    pose_measurement(
                        time_s,
                        index,
                        physically_held=False,
                        robot_contact_or_support=False,
                    )
                ),
                BoltSeatStatus(False, False),
            )
        self.assertEqual(
            evaluator.observe(
                pose_measurement(
                    0.8,
                    96,
                    physically_held=False,
                    robot_contact_or_support=False,
                )
            ),
            BoltSeatStatus(False, True),
        )

    def test_pose_window_gap_and_missing_samples_cannot_fall_back_to_velocity_gate(self):
        for rows in (
            (pose_measurement(0.0, 0), pose_measurement(0.1, 24), pose_measurement(0.2, 36)),
            (
                sample(0.0, native_pose_samples=(), native_pose_dt_s=1.0 / 120.0),
                sample(0.1, native_pose_samples=(), native_pose_dt_s=1.0 / 120.0),
                sample(0.2, native_pose_samples=(), native_pose_dt_s=1.0 / 120.0),
            ),
        ):
            evaluator = IndependentBoltSeatingEvaluator(
                POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
            )
            statuses = [evaluator.observe(row) for row in rows]
            self.assertTrue(all(not status.seat_ready for status in statuses))

    def test_pose_window_does_not_replace_postrelease_axial_motion_limit(self):
        evaluator = IndependentBoltSeatingEvaluator(
            POSE_SEAT_TOLERANCES, pose_window_tolerances=POSE_WINDOW_TOLERANCES
        )
        for time_s, index in ((0.0, 0), (0.1, 12), (0.2, 24)):
            evaluator.observe(pose_measurement(time_s, index))
        evaluator.observe(
            pose_measurement(0.3, 36, physically_held=False, robot_contact_or_support=False)
        )
        status = evaluator.observe(
            pose_measurement(
                0.4,
                48,
                physically_held=False,
                robot_contact_or_support=False,
                axial_position_m=0.052,
            )
        )
        self.assertEqual(status, BoltSeatStatus(False, False))

    def test_synthetic_valid_history_reports_seat_ready_then_final_success(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held = held_seat(evaluator)
        self.assertEqual(
            held,
            [BoltSeatStatus(False, False), BoltSeatStatus(False, False), BoltSeatStatus(True, False)],
        )
        self.assertIs(type(held[-1].seat_ready), bool)
        self.assertFalse(hasattr(held[-1], "metrics"))

        released = evaluator.observe(release_sample(0.375))
        self.assertEqual(released, BoltSeatStatus(False, False))
        results = finish_stable_dwell(evaluator)
        self.assertEqual([result.task_success for result in results], [False, False, False, True])
        self.assertTrue(all(not result.seat_ready for result in results))

    def test_valid_geometry_does_not_require_an_exact_nominal_axial_position(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        statuses = [
            evaluator.observe(sample(time_s, axial_position_m=0.137))
            for time_s in (0.0, 0.125, 0.25)
        ]
        self.assertEqual(statuses[-1], BoltSeatStatus(True, False))
        self.assertEqual(
            evaluator.observe(release_sample(0.375, axial_position_m=0.137)),
            BoltSeatStatus(False, False),
        )
        results = finish_stable_dwell(evaluator, axial_position_m=0.137)
        self.assertTrue(results[-1].task_success)

    def test_hover_above_cover_without_support_contact_is_not_seated(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        statuses = held_seat(
            evaluator,
            cap_support_gap_m=TOLERANCES.max_cap_support_gap_m + 0.0001,
            cap_support_contact=False,
        )
        self.assertTrue(all(not status.seat_ready for status in statuses))
        self.assertEqual(evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False))

    def test_wrong_hole_or_missing_casing_entry_prevents_held_seat(self):
        for invalid_geometry in (
            {"through_selected_cover_hole": False},
            {"entered_selected_casing_opening": False},
            {"casing_entry_depth_m": TOLERANCES.min_casing_entry_depth_m - 0.001},
        ):
            with self.subTest(invalid_geometry=invalid_geometry):
                evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
                statuses = held_seat(evaluator, **invalid_geometry)
                self.assertTrue(all(not status.seat_ready for status in statuses))
                self.assertEqual(
                    evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False)
                )

    def test_cap_underside_support_without_casing_entry_is_not_seated(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        statuses = held_seat(
            evaluator,
            through_selected_cover_hole=True,
            entered_selected_casing_opening=False,
            casing_entry_depth_m=0.0,
            cap_support_gap_m=0.0,
            cap_support_contact=True,
        )
        self.assertTrue(all(not status.seat_ready for status in statuses))
        self.assertEqual(evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False))

    def test_release_before_held_seat_dwell_latches_failure(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        self.assertEqual(evaluator.observe(sample(0.0)), BoltSeatStatus(False, False))
        self.assertEqual(evaluator.observe(sample(0.125)), BoltSeatStatus(False, False))
        self.assertEqual(evaluator.observe(release_sample(0.25)), BoltSeatStatus(False, False))
        self.assertEqual(
            [result.task_success for result in finish_stable_dwell(evaluator, release_time=0.25)],
            [False] * 4,
        )

    def test_unheld_initial_samples_are_not_mistaken_for_release(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        self.assertEqual(evaluator.observe(release_sample(0.0)), BoltSeatStatus(False, False))
        self.assertEqual(evaluator.observe(release_sample(0.125)), BoltSeatStatus(False, False))

    def test_wrong_hole_casing_hover_gripper_support_penetration_and_bottoming_fail(self):
        invalid_seats = (
            {"through_selected_cover_hole": False},
            {"entered_selected_casing_opening": False},
            {"casing_entry_depth_m": 0.0},
            {"cap_support_gap_m": 0.01},
            {"cap_support_contact": False},
            {"penetration_m": 0.001},
            {"premature_bottoming": True},
        )
        for invalid_seat in invalid_seats:
            with self.subTest(invalid_seat=invalid_seat):
                evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
                held_seat(evaluator, **invalid_seat)
                self.assertEqual(
                    evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False)
                )
                self.assertEqual([r.task_success for r in finish_stable_dwell(evaluator)], [False] * 4)

    def test_early_release_cannot_be_repaired_by_falling_into_seat(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held_seat(evaluator, insertion_provenance_valid=False)
        self.assertEqual(evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False))
        self.assertEqual([r.task_success for r in finish_stable_dwell(evaluator)], [False] * 4)

    def test_bad_post_release_geometry_or_robot_support_latches_failure(self):
        for invalid_release in (
            {"through_selected_cover_hole": False},
            {"entered_selected_casing_opening": False},
            {"cap_support_contact": False},
            {"premature_bottoming": True},
        ):
            with self.subTest(invalid_release=invalid_release):
                evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
                held_seat(evaluator)
                self.assertEqual(
                    evaluator.observe(release_sample(0.375, **invalid_release)),
                    BoltSeatStatus(False, False),
                )
                self.assertEqual([r.task_success for r in finish_stable_dwell(evaluator)], [False] * 4)

    def test_residual_gripper_contact_delays_but_does_not_poison_release_dwell(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held_seat(evaluator)
        self.assertEqual(
            evaluator.observe(release_sample(0.375, robot_contact_or_support=True)),
            BoltSeatStatus(False, False),
        )
        still_contact = evaluator.observe(
            release_sample(0.625, robot_contact_or_support=True)
        )
        self.assertFalse(still_contact.task_success)
        results = [
            evaluator.observe(release_sample(time_s))
            for time_s in (0.875, 1.125, 1.375, 1.625, 1.875)
        ]
        self.assertEqual([result.task_success for result in results], [False, False, False, False, True])

    def test_axial_motion_between_last_held_seat_and_release_is_bounded(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held_seat(evaluator)
        self.assertEqual(
            evaluator.observe(release_sample(0.375, axial_position_m=0.052)),
            BoltSeatStatus(False, False),
        )
        self.assertEqual([r.task_success for r in finish_stable_dwell(evaluator)], [False] * 4)

    def test_release_requires_a_current_active_seat_and_real_insertion_provenance(self):
        for held_changes in (
            {"controller_stopped": False},
            {"relative_linear_speed_mps": 0.01},
            {"insertion_provenance_valid": False},
        ):
            with self.subTest(held_changes=held_changes):
                evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
                held_seat(evaluator, **held_changes)
                self.assertEqual(evaluator.observe(release_sample(0.375)), BoltSeatStatus(False, False))
                self.assertEqual([r.task_success for r in finish_stable_dwell(evaluator)], [False] * 4)

    def test_unstable_motion_restarts_dwell_but_large_sample_gap_fails(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held_seat(evaluator)
        evaluator.observe(release_sample(0.375))
        evaluator.observe(release_sample(0.625))
        evaluator.observe(release_sample(0.875, relative_linear_speed_mps=0.01))
        self.assertFalse(evaluator.observe(release_sample(1.125)).task_success)
        self.assertFalse(evaluator.observe(release_sample(1.375)).task_success)

        gap_evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        held_seat(gap_evaluator)
        self.assertEqual(gap_evaluator.observe(release_sample(0.625)), BoltSeatStatus(False, False))
        self.assertEqual(
            [r.task_success for r in finish_stable_dwell(gap_evaluator, release_time=0.625)],
            [False] * 4,
        )

    def test_tolerances_are_required_and_measurements_reject_non_finite_values(self):
        with self.assertRaises(TypeError):
            BoltSeatTolerances()
        with self.assertRaises(ValueError):
            replace(TOLERANCES, stable_dwell_s=0.0)
        with self.assertRaises(ValueError):
            sample(0.0, axial_position_m=float("nan"))

    def test_measurement_time_must_increase(self):
        evaluator = IndependentBoltSeatingEvaluator(TOLERANCES)
        evaluator.observe(sample(1.0))
        with self.assertRaises(ValueError):
            evaluator.observe(sample(1.0))


if __name__ == "__main__":
    unittest.main()
