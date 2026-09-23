import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "scenes" / "recipes"


class SceneRecipeTests(unittest.TestCase):
    def test_ten_recipes_exist_and_share_parameter_source(self):
        files = sorted(RECIPES.glob("*.json"))
        self.assertEqual(len(files), 10)
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["render_collision_contract"]["parameter_source"], "shared_mutation_parameters")
            self.assertIn("hidden blocker", data["render_collision_contract"]["forbidden"])
            self.assertGreaterEqual(data["calibration"]["required_margin_mm"], 2 * data["calibration"]["tolerance_mm"])

    def test_each_recipe_references_source_measurements(self):
        for path in RECIPES.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(data["source_visual_parts"])
            self.assertEqual(len(data["source_visual_parts"]), len(data["source_measurements"]))


if __name__ == "__main__":
    unittest.main()
