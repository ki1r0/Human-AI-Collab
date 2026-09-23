import json
import unittest
from pathlib import Path

from pilot_12pair.oracle.constrained_tasks import (
    DEFAULT_CONFIG,
    evaluate_sequence,
    expected_table,
    load_config,
    task_index,
    validate_config,
)


class ConstrainedTaskContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG)
        cls.tasks = task_index(cls.config)

    def test_config_has_no_contract_errors(self):
        self.assertEqual(validate_config(self.config), [])

    def test_exactly_five_tasks_use_existing_parts(self):
        self.assertEqual(set(self.tasks), {"HCF-01", "WSG-01", "KEY-01", "CAS-01", "DOW-01"})
        for task in self.config["tasks"]:
            for part in task["parts"]:
                self.assertIn(part, self.config["source_assets"])

    def test_each_pair_has_opposite_reverse_feasibility(self):
        for task in self.config["tasks"]:
            hard = evaluate_sequence(task, "HARD", ("B", "A"))
            comm = evaluate_sequence(task, "COMMUTABLE", ("B", "A"))
            self.assertFalse(hard.valid, task["task_id"])
            self.assertTrue(comm.valid, task["task_id"])

    def test_forward_order_is_valid_and_single_actions_are_incomplete(self):
        for task in self.config["tasks"]:
            for variant in ("HARD", "COMMUTABLE"):
                self.assertTrue(evaluate_sequence(task, variant, ("A", "B")).valid)
                self.assertFalse(evaluate_sequence(task, variant, ("A",)).valid)
                self.assertFalse(evaluate_sequence(task, variant, ("B",)).valid)

    def test_geometry_inequalities_are_explicit(self):
        hcf = self.tasks["HCF-01"]["geometry"]
        self.assertLess(hcf["fastener_shaft_diameter_mm"], hcf["round_hole_diameter_mm"])
        self.assertLess(hcf["round_hole_diameter_mm"], hcf["fastener_head_diameter_mm"])
        self.assertLess(hcf["fastener_head_diameter_mm"], hcf["keyhole_lobe_diameter_mm"])
        wsg = self.tasks["WSG-01"]["geometry"]
        self.assertGreater(wsg["side_slot_width_mm"], wsg["required_side_insertion_clearance_mm"])
        key = self.tasks["KEY-01"]["geometry"]
        self.assertGreater(key["side_access_width_mm"], key["key_width_mm"])

    def test_expected_table_contains_ten_domains(self):
        table = expected_table(self.config)
        self.assertEqual(len(table), 10)
        for row in table:
            self.assertTrue(row["AB"]["valid"])
            self.assertFalse(row["A_only"]["valid"])
            self.assertFalse(row["B_only"]["valid"])

    def test_public_contract_does_not_require_evaluator_labels(self):
        excluded = set(self.config["shared_contract"]["model_observation_excludes"])
        self.assertTrue({"variant", "relation_label", "feasible_procedures", "oracle_events"} <= excluded)


if __name__ == "__main__":
    unittest.main()
