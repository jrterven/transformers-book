import json
from pathlib import Path
import unittest

from examples.ch18_behavior.data import PAIRS, SLICES
from examples.ch18_behavior.model import patch_effect, route_policy, summarize

try:
    import torch
    from examples.ch18_behavior.run_pretrained import replace_rows
except ImportError:
    torch = None


RECORD = Path(__file__).resolve().parents[1] / "examples/ch18_behavior/PRETRAINED_RUN.json"


class ModelBehaviorTests(unittest.TestCase):
    def test_recovery_has_sign_and_can_overshoot(self):
        self.assertEqual(patch_effect(-2, 2, 3), (5, 1.25))
        self.assertEqual(patch_effect(2, -2, 3), (1, -.25))
        self.assertEqual(patch_effect(-2, -2, 3), (5, None))

    def test_threshold_can_defer_a_clear_winner(self):
        self.assertEqual(route_policy((2, 1, 0)), "A")
        self.assertEqual(route_policy((2, 1, 0), .8), "C")
        self.assertEqual(route_policy((102, 101, 100), .6), "A")
        self.assertEqual(route_policy((0, 0, 1), .9), "C")

    def test_invalid_token_counts_as_an_error(self):
        self.assertEqual(summarize("ABC", ("B", "invalid", "C")),
                         {"n": 3, "valid": 2, "correct": 1, "right_to_clarify": 1,
                          "ambiguous": 1, "incorrect_write": 1})

    @unittest.skipIf(torch is None, "PyTorch is optional")
    def test_patch_replaces_only_selected_rows_without_mutating_input(self):
        output = torch.arange(12).reshape(2, 3, 2)
        snapshot = output.clone()
        patched = replace_rows(output, torch.tensor([1, 2]), torch.tensor([[20, 21], [22, 23]]))
        expected = torch.tensor([[[0, 1], [20, 21], [4, 5]], [[6, 7], [8, 9], [22, 23]]])
        torch.testing.assert_close(patched, expected)
        torch.testing.assert_close(output, snapshot)

    def test_published_controls_and_outcomes(self):
        report = json.loads(RECORD.read_text())
        self.assertEqual([tuple(p["requests"]) for p in report["pairs"]], list(PAIRS))
        self.assertEqual([len(p["patches"]) for p in report["pairs"]], [24, 5, 5])
        self.assertEqual(list(report["routing"]), list(SLICES))
        for pair in report["pairs"]:
            for block in pair["patches"]:
                self.assertLess(block["self_patch_max_logit_difference"], 3e-4)
                for restored, native in zip(block["self_patch_logits"], pair["logits"], strict=True):
                    for got, want in zip(restored, native, strict=True):
                        self.assertAlmostEqual(got, want, delta=3e-4)
                for key in ("read_recipient", "write_recipient"):
                    self.assertLess(abs(block[key][0]["shift"]), 6e-4)
                    for result in block[key]:
                        a, b, _ = result["logits"]
                        self.assertAlmostEqual(result["gap"], b-a, places=5)
            final = pair["patches"][-1]
            for index, key in enumerate(("read_recipient", "write_recipient")):
                for result in final[key][:-1]:
                    self.assertLess(abs(result["shift"]), 6e-4)
                self.assertAlmostEqual(final[key][-1]["gap"], pair["gaps"][1 - index], places=3)
        for group, cases in SLICES.items():
            rows = report["routing"][group]["cases"]
            self.assertEqual([(r["request"], r["target"]) for r in rows], list(cases))
            for row in rows:
                logits = row["candidate_logits"]
                self.assertEqual(route_policy(logits), row["predictions"]["Constrained"])
                predicted = route_policy(logits, report["min_confidence"])
                self.assertEqual(predicted, row["predictions"]["Defer below 0.8"])
