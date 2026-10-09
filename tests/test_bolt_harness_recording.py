import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from runtime.bolt_harness.recording import EpisodeRGBVideoRecorder


class _Writer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


class EpisodeRGBVideoRecorderTests(unittest.TestCase):
    def test_writes_camera_arrays_and_simulation_timestamps(self):
        writer = _Writer()
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "episode.mp4"
            with patch("runtime.bolt_harness.recording._open_video_writer", return_value=writer):
                recorder = EpisodeRGBVideoRecorder(video_path, fps=10)
                first = np.full((4, 6, 3), 23, dtype=np.uint8)
                second = np.full((4, 6, 3), 91, dtype=np.uint8)
                recorder.append(first, 2.0)
                recorder.append(second, 2.1)
                self.assertIsNone(recorder.close())

            self.assertEqual(recorder.frame_count, 2)
            self.assertTrue(writer.closed)
            np.testing.assert_array_equal(writer.frames, [first, second])
            manifest = json.loads(video_path.with_suffix(".timestamps.json").read_text())
            self.assertEqual(manifest["fps"], 10.0)
            self.assertEqual(manifest["sim_time_s"], [2.0, 2.1])
            self.assertEqual(manifest["frame_count"], 2)
            self.assertTrue(manifest["video_finalized"])

    def test_rejects_invalid_frames_and_non_cadence_timestamps(self):
        writer = _Writer()
        with tempfile.TemporaryDirectory() as directory:
            with patch("runtime.bolt_harness.recording._open_video_writer", return_value=writer):
                recorder = EpisodeRGBVideoRecorder(Path(directory) / "episode.mp4", fps=10)
                with self.assertRaisesRegex(ValueError, "shape .* dtype uint8"):
                    recorder.append(np.zeros((4, 6, 4), dtype=np.uint8), 0.0)
                recorder.append(np.zeros((4, 6, 3), dtype=np.uint8), 0.0)
                with self.assertRaisesRegex(ValueError, "cadence"):
                    recorder.append(np.zeros((4, 6, 3), dtype=np.uint8), 0.15)
                self.assertIn("uint8", recorder.close())
            self.assertEqual(len(writer.frames), 1)
            manifest = json.loads(recorder.timestamps_path.read_text())
            self.assertFalse(manifest["video_finalized"])

    def test_context_manager_finalizes_evidence_on_episode_exception(self):
        writer = _Writer()
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "episode.mp4"
            with patch("runtime.bolt_harness.recording._open_video_writer", return_value=writer):
                recorder = EpisodeRGBVideoRecorder(video_path, fps=10)
                with self.assertRaisesRegex(RuntimeError, "episode failure"):
                    with recorder:
                        recorder.append(np.zeros((2, 3, 3), dtype=np.uint8), 0.0)
                        raise RuntimeError("episode failure")

            self.assertTrue(writer.closed)
            self.assertEqual(recorder.frame_count, 1)
            manifest = json.loads(video_path.with_suffix(".timestamps.json").read_text())
            self.assertEqual(manifest["sim_time_s"], [0.0])
            with self.assertRaisesRegex(RuntimeError, "closed"):
                recorder.append(np.zeros((2, 3, 3), dtype=np.uint8), 0.1)

    def test_empty_episode_is_reported_in_timestamp_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRGBVideoRecorder(Path(directory) / "episode.mp4", fps=12)
            self.assertEqual(recorder.close(), "no camera RGB frames were recorded")
            manifest = json.loads(recorder.timestamps_path.read_text())
            self.assertEqual(manifest["frame_count"], 0)
            self.assertFalse(manifest["video_finalized"])


if __name__ == "__main__":
    unittest.main()
