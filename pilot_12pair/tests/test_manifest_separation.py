import json
import unittest
from pathlib import Path


MANIFEST_DIR = Path(__file__).resolve().parents[1] / "tasks" / "manifests"


class ManifestSeparationTests(unittest.TestCase):
    def test_public_manifest_has_no_evaluator_labels(self):
        forbidden = {"variant", "relation", "can_b_before_a", "feasible_procedures", "intervention", "oracle_evidence"}
        for line in (MANIFEST_DIR / "public_manifest.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            serialized = json.dumps(record)
            self.assertFalse(any(f'"{key}"' in serialized for key in forbidden), record["domain_id"])

    def test_evaluator_manifest_contains_the_pair_labels(self):
        records = [json.loads(line) for line in (MANIFEST_DIR / "evaluator_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 10)
        self.assertEqual({record["variant"] for record in records}, {"HARD", "COMMUTABLE"})


if __name__ == "__main__":
    unittest.main()
