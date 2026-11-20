import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastap_methods.single_teammate import compare_outcomes, independent_test


class SingleTeammateTests(unittest.TestCase):
    def test_pairing_distinguishes_partial_from_full_completion(self):
        partial = compare_outcomes([1/3] * 6, [0] * 6)
        self.assertEqual(partial["any_food"]["p_one_sided"], 1/64)
        self.assertEqual(partial["all_food"]["p_one_sided"], 1)
        full = compare_outcomes([1, 2/3], [2/3, 1])
        self.assertEqual(full["all_food"]["trained_only"], 1)
        self.assertEqual(full["all_food"]["control_only"], 1)

    def test_only_the_third_role_changes_and_both_controls_must_pass(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".tmp") as tmp:
            root = Path(tmp)
            initial, selected = (root / "checkpoints" / k for k in ("0", "50000"))
            selected.mkdir(parents=True)
            for filename in ("agent.th", "agent.json"):
                (selected / filename).write_bytes(b"selected")
            calls = []
            def evaluate(ego, teammate, seeds, *, random_teammate):
                calls.append((ego, teammate, random_teammate, tuple(seeds)))
                values = [float(not random_teammate)] * len(seeds)
                return {"episode_returns": values, "mean_return": values[0]}
            with patch("fastap_methods.single_teammate.evaluate_pair", side_effect=evaluate):
                result = independent_test(root, selected, range(8))
            self.assertEqual([c[0] for c in calls], [selected, selected, selected, initial])
            self.assertEqual([c[1] for c in calls], [root/"T1", root/"T1", initial, initial])
            self.assertEqual([c[2] for c in calls], [False, True, False, False])
            self.assertTrue(all(c[3] == tuple(range(8)) for c in calls))
            self.assertFalse(result["evidence_for_full_completion"])
            self.assertEqual(json.loads((root/"T1"/"role.json").read_text())["slot"], 2)
