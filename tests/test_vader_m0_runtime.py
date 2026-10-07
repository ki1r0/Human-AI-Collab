import unittest

from hrc_repair.contracts import Observation
from vader_m0.run import _available, _detected_observation, _json_object


class VaderM0RuntimeTests(unittest.TestCase):
    def test_json_response_parser_handles_fenced_json(self):
        self.assertEqual(_json_object('```json\n{"action":"pick"}\n```'), {"action": "pick"})

    def test_visual_detection_is_not_overruled_by_contact_sensor(self):
        observation = Observation(
            "episode", "obs_0000", 0, 1.0,
            held="yes", placed="no", release_observed="no",
            recent_skill={"public_sensor": {"placement_observed": False}},
        )
        detected = _detected_observation(
            observation,
            phase="place",
            result={"verdict": "SUCCESS", "evidence": "Flange seated; gripper clear."},
            frame_paths=[],
            expected_outcome="Cover seated and released.",
        )
        self.assertEqual((detected.held, detected.placed, detected.release_observed), ("no", "yes", "yes"))
        self.assertEqual(_available(detected, "place", False), ["finish", "stop"])

    def test_uncertain_initial_affordance_requests_observation_not_motion(self):
        observation = Observation("episode", "obs_0000", 0, 1.0, frames={"head_rgb": "/tmp/head_rgb.png"})
        detected = _detected_observation(
            observation,
            phase="initial",
            result={"verdict": "FAILED", "evidence": "The cover may be held."},
            frame_paths=["/tmp/head_rgb.png"],
            expected_outcome="Cover rests separately.",
        )
        self.assertEqual(_available(detected, "", False), ["observe", "stop"])


if __name__ == "__main__":
    unittest.main()
