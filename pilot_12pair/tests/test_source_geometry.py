import unittest

from pilot_12pair.oracle.constrained_tasks import load_config
from pilot_12pair.oracle.validate_source_geometry import audit


class SourceGeometryTests(unittest.TestCase):
    def test_recorded_bounds_match_repository_stl(self):
        rows = audit(load_config())
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["max_abs_error_mm"] <= 0.02 for row in rows))


if __name__ == "__main__":
    unittest.main()
