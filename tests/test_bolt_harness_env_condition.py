import json
import tempfile
import unittest
from pathlib import Path

import yaml

try:
    from scripts.run_bolt_harness import _write_private_condition_manifest
    from runtime.bolt_harness.env_cfg import (
        _condition_blocker_geometry,
        make_env_cfg,
    )
    from runtime.bolt_harness.measurement import (
        ASSEMBLY_FRAME_POS_M,
        CASING_ENTRY_PLANE_Z_M,
        CASING_MIN_RADIUS_M,
        SHANK_RADIUS_M,
    )
except ModuleNotFoundError as exc:
    if exc.name != "isaaclab.app":
        raise
    raise unittest.SkipTest("Isaac Lab's isaaclab.app module is unavailable") from exc


class BoltEnvironmentConditionTests(unittest.TestCase):
    def test_private_condition_manifest_records_flag_source_and_collider_readback(self):
        evidence = {
            "condition": "S2",
            "blocker_authored": True,
            "blocker": {"prim_path": "/World/envs/env_0/Casing/TaskLayerS2Blocker"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write_private_condition_manifest(
                Path(temp_dir),
                condition="S2",
                selection_source="command_line",
                collider_evidence=evidence,
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "private_condition_manifest.json")
        self.assertEqual(manifest["condition"], "S2")
        self.assertEqual(manifest["selection_source"], "command_line")
        self.assertEqual(manifest["input_flag"], "--condition")
        self.assertEqual(manifest["condition_collider_evidence"], evidence)

    def test_default_config_and_environment_are_s0_without_blocker(self):
        config_path = Path(__file__).resolve().parents[1] / "config/bolt_insertion.yaml"
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg = make_env_cfg(device="cpu")

        self.assertEqual(settings["condition"], "S0")
        self.assertEqual(cfg.condition, "S0")
        self.assertIsNone(_condition_blocker_geometry(cfg))

    def test_s2_blocker_fits_socket_and_stops_before_seating_depth(self):
        cfg = make_env_cfg(device="cpu", condition="S2")
        local_position, local_size = _condition_blocker_geometry(cfg)
        scale = cfg.cad_stage_scale
        world_center = tuple(
            cfg.assembly_frame_pos_m[index] + local_position[index] * scale
            for index in range(3)
        )
        world_size = tuple(dimension * scale for dimension in local_size)
        casing_entry_z = (
            cfg.assembly_frame_pos_m[2]
            + CASING_ENTRY_PLANE_Z_M
            - ASSEMBLY_FRAME_POS_M[2]
        )
        blocker_top_depth = casing_entry_z - (world_center[2] + world_size[2] / 2.0)

        self.assertAlmostEqual(world_center[0], cfg.bolt_seat_root_pos_m[0])
        self.assertAlmostEqual(world_center[1], cfg.bolt_seat_root_pos_m[1])
        self.assertGreater(world_size[0], 2.0 * SHANK_RADIUS_M)
        self.assertLess((world_size[0] / 2.0) ** 2 + (world_size[1] / 2.0) ** 2, CASING_MIN_RADIUS_M**2)
        self.assertGreater(blocker_top_depth, 0.0)
        self.assertLess(blocker_top_depth, 0.015)

    def test_rejects_conditions_outside_m0_s0_s2(self):
        with self.assertRaises(ValueError):
            make_env_cfg(device="cpu", condition="S1")


if __name__ == "__main__":
    unittest.main()
