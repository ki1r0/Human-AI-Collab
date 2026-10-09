import unittest
from enum import Enum
from types import SimpleNamespace

from runtime.bolt_harness.contacts import (
    BOLT_ACTOR_PATH,
    CAP_COLLIDER_PATH,
    FINGER_ACTOR_PATHS,
    decode_bolt_finger_contacts,
)


class EventType(Enum):
    CONTACT_FOUND = 0
    CONTACT_LOST = 1
    CONTACT_PERSIST = 2


class BoltContactDecoderTests(unittest.TestCase):
    def setUp(self):
        self.paths = {
            1: BOLT_ACTOR_PATH,
            2: FINGER_ACTOR_PATHS["left"],
            3: FINGER_ACTOR_PATHS["right"],
            4: CAP_COLLIDER_PATH,
            5: "/World/envs/env_0/Bolt/collision_shaft",
            6: "/World/envs/env_0/Robot/panda_leftfinger/collision",
            7: "/World/envs/env_0/Robot/panda_rightfinger/collision",
            8: "/World/envs/env_0/Robot/panda_hand",
        }

    def decode(self, headers, data, dt_s=0.01):
        return decode_bolt_finger_contacts(
            headers,
            data,
            dt_s=dt_s,
            path_decoder=self.paths.__getitem__,
        )

    @staticmethod
    def header(
        *,
        event=EventType.CONTACT_FOUND,
        actor0=1,
        actor1=2,
        collider0=4,
        collider1=6,
        offset=0,
        count=1,
    ):
        return SimpleNamespace(
            type=event,
            actor0=actor0,
            actor1=actor1,
            collider0=collider0,
            collider1=collider1,
            contact_data_offset=offset,
            num_contact_data=count,
        )

    @staticmethod
    def point(*, position=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 1.0), impulse=(0.0, 0.0, 0.2), separation=-0.001):
        return SimpleNamespace(
            position=position,
            normal=normal,
            impulse=impulse,
            separation=separation,
        )

    def test_exact_cap_contact_returns_per_finger_raw_and_bolt_oriented_data(self):
        left = self.header()
        # For this pair ordering, PhysX normal/impulse point from Bolt (shape 1)
        # toward the finger (shape 0); force on Bolt is the negated raw vector.
        right = self.header(
            event=EventType.CONTACT_PERSIST,
            actor0=3,
            actor1=1,
            collider0=7,
            collider1=4,
            offset=1,
        )
        result = self.decode([left, right], [self.point(), self.point(position=(4, 5, 6))])

        self.assertEqual(len(result["left"]), 1)
        self.assertEqual(len(result["right"]), 1)
        left_contact = result["left"][0]
        self.assertEqual(left_contact.event_type, "CONTACT_FOUND")
        self.assertEqual(left_contact.raw_impulse_w_ns, (0.0, 0.0, 0.2))
        self.assertEqual(left_contact.force_on_bolt_w_n, (0.0, 0.0, 20.0))
        self.assertEqual(left_contact.normal_on_bolt_w, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(left_contact.normal_compression_n, 20.0)
        self.assertEqual(left_contact.collider0_path, CAP_COLLIDER_PATH)
        self.assertEqual(left_contact.actor1_path, FINGER_ACTOR_PATHS["left"])
        self.assertEqual(left_contact.contact_data_index, 0)
        self.assertEqual(left_contact.contact_data_offset, 0)
        self.assertEqual(left_contact.contact_data_count, 1)

        right_contact = result["right"][0]
        self.assertEqual(right_contact.raw_impulse_w_ns, (0.0, 0.0, 0.2))
        self.assertEqual(right_contact.impulse_on_bolt_w_ns, (0.0, 0.0, -0.2))
        self.assertEqual(right_contact.force_on_bolt_w_n, (0.0, 0.0, -20.0))
        self.assertEqual(right_contact.normal_on_bolt_w, (0.0, 0.0, -1.0))
        self.assertAlmostEqual(right_contact.normal_compression_n, 20.0)
        self.assertEqual(right_contact.actor0_path, FINGER_ACTOR_PATHS["right"])
        self.assertEqual(right_contact.collider1_path, CAP_COLLIDER_PATH)
        self.assertEqual(right_contact.contact_data_index, 1)

    def test_preserves_signed_impulse_without_absolute_value(self):
        result = self.decode(
            [self.header()],
            [self.point(normal=(1.0, 0.0, 0.0), impulse=(-0.03, 0.0, 0.0))],
        )
        contact = result["left"][0]
        self.assertEqual(contact.raw_impulse_w_ns, (-0.03, 0.0, 0.0))
        self.assertEqual(contact.force_on_bolt_w_n, (-3.0, 0.0, 0.0))
        self.assertAlmostEqual(contact.normal_compression_n, -3.0)

    def test_injected_decoder_accepts_sdf_path_like_results(self):
        class DecodedPath:
            def __init__(self, path_string):
                self.pathString = path_string

        result = decode_bolt_finger_contacts(
            [self.header()],
            [self.point()],
            dt_s=0.01,
            path_decoder=lambda encoded: DecodedPath(self.paths[encoded]),
        )
        self.assertEqual(result["left"][0].collider0_path, CAP_COLLIDER_PATH)

    def test_rejects_shaft_collider_and_nonfinger_actor(self):
        shaft = self.header(collider0=5)
        other_actor = self.header(actor1=8)
        result = self.decode([shaft, other_actor], [self.point(), self.point()])
        self.assertEqual(result, {"left": (), "right": ()})

    def test_ignores_lost_event_without_decoding_paths_or_data(self):
        header = self.header(event=EventType.CONTACT_LOST, offset=-10, count=-3)

        def should_not_decode(_):
            self.fail("lost events must be ignored before decoding their paths")

        result = decode_bolt_finger_contacts(
            [header], [], dt_s=0.01, path_decoder=should_not_decode
        )
        self.assertEqual(result, {"left": (), "right": ()})

    def test_rejects_unknown_event_type(self):
        with self.assertRaisesRegex(ValueError, "unsupported PhysX contact event"):
            self.decode([self.header(event="CONTACT_CCD")], [self.point()])

    def test_rejects_nonfinite_contact_fields(self):
        bad_points = (
            self.point(position=(0.0, float("nan"), 0.0)),
            self.point(normal=(0.0, float("inf"), 1.0)),
            self.point(impulse=(0.0, 0.0, float("nan"))),
            self.point(separation=float("-inf")),
        )
        for point in bad_points:
            with self.subTest(point=point), self.assertRaisesRegex(ValueError, "finite"):
                self.decode([self.header()], [point])

    def test_rejects_zero_normal_and_contact_buffer_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            self.decode([self.header()], [self.point(normal=(0.0, 0.0, 0.0))])
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            self.decode([self.header(offset=1)], [self.point()])

    def test_rejects_invalid_offsets_counts_and_timestep(self):
        for header in (self.header(offset=-1), self.header(count=-1), self.header(offset=0.0)):
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, "contact_data"):
                self.decode([header], [self.point()])
        for dt_s in (0.0, -0.01, float("nan"), float("inf")):
            with self.subTest(dt_s=dt_s), self.assertRaises(ValueError):
                self.decode([], [], dt_s=dt_s)

    def test_rejects_nonfinite_derived_force(self):
        with self.assertRaisesRegex(ValueError, "derived Bolt contact force"):
            self.decode([self.header()], [self.point(impulse=(1e308, 0.0, 0.0))], dt_s=1e-308)


if __name__ == "__main__":
    unittest.main()
