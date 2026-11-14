from collections import Counter
import json
import math
from pathlib import Path
import unittest

from examples.ch18_behavior.feedback import expected_cost, preference_probability, reward_curves, reward_tilt
from examples.ch18_behavior.method_data import TEST_FAMILIES, TRAIN_FAMILIES, probe_requests

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from examples.ch18_behavior.interpretability import integrated_gradients, ridge_probe

DATA = Path(__file__).resolve().parents[1] / "examples/ch18_behavior"


class FeedbackTests(unittest.TestCase):
    def test_preferences_are_invariant_to_a_common_shift(self):
        self.assertAlmostEqual(preference_probability(1.5, .5), 1/(1+math.exp(-1)))
        self.assertEqual(preference_probability(1.5, .5), preference_probability(15.5, 14.5))
        self.assertEqual(preference_probability(-1000, 0), 0)

    def test_reward_tilt_preserves_reference_without_reward_differences(self):
        ref = [math.log(p) for p in (.5, .3, .2)]
        for got, want in zip(reward_tilt(ref, (2, 2, 2), 1), (.5, .3, .2), strict=True):
            self.assertAlmostEqual(got, want)
        p = reward_tilt(ref, (0, 0, 1), 0.5)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertAlmostEqual(p[2] / p[0], (.2 / .5) * math.exp(2))
        self.assertEqual(reward_tilt([0, -math.inf], [0, 10], 1), [1.0, 0.0])
        with self.assertRaises(ValueError):
            reward_tilt(ref, (0, 0, 1), 0)

    def test_costs_use_the_true_intent_axis(self):
        values = expected_cost((.60, .25, .15), ((0, 3, 2), (8, 0, 8), (1, 1, 0)))
        for got, want in zip(values, (1.05, 6.0, .85), strict=True):
            self.assertAlmostEqual(got, want)
        self.assertEqual(values.index(min(values)), 2)

    def test_rescoring_the_real_record_exposes_proxy_tradeoff(self):
        rows = reward_curves(json.loads((DATA/"PRETRAINED_RUN.json").read_text()))
        self.assertAlmostEqual(rows[0]["correct"], .5285, places=4)
        self.assertAlmostEqual(rows[4]["clarify_if_ambiguous"], .0297, places=4)
        self.assertGreater(rows[-1]["immediate_action"], rows[0]["immediate_action"])
        self.assertLess(rows[-1]["correct"], rows[0]["correct"])


class InterpretabilityTests(unittest.TestCase):
    def test_probe_example_split_is_balanced_and_disjoint(self):
        train = probe_requests(TRAIN_FAMILIES, (31, 57, 93))
        test = probe_requests(TEST_FAMILIES, (12, 44, 85))
        self.assertEqual(Counter(row[-1] for row in train), {0:18, 1:18})
        self.assertEqual(Counter(row[-1] for row in test), {0:9, 1:9})
        self.assertTrue({row[0] for row in train}.isdisjoint(row[0] for row in test))

    @unittest.skipIf(torch is None, "PyTorch is optional")
    def test_ridge_probe_matches_the_hand_example(self):
        features = torch.tensor([[-1.], [1.]])
        target = torch.tensor([0., 1.])
        test = torch.tensor([[.8]])
        torch.testing.assert_close(ridge_probe(features, target, test, 2), torch.tensor([.7]))

    @unittest.skipIf(torch is None, "PyTorch is optional")
    def test_gradients_obey_completeness_and_unchanged_coordinates(self):
        start, finish = torch.tensor([0., 3.]), torch.tensor([1., 3.])
        result = integrated_gradients(lambda x: 3*x[0]-2*x[0]**2+x[1], start, finish, 2)
        torch.testing.assert_close(result, torch.tensor([1., 0.]))
        with self.assertRaises(ValueError):
            integrated_gradients(lambda x: x.sum(), start, finish, 0)

    def test_record_contains_measured_controls(self):
        record = json.loads((DATA/"METHODS_RUN.json").read_text())
        self.assertEqual(record["training_requests"], 36)
        self.assertEqual(record["probes"]["n"], 18)
        for row in record["test_requests"]:
            for a,b in zip(row["lens"][-1], row["logits"], strict=True):
                self.assertAlmostEqual(a,b,delta=3e-4)
        for row in record["head"]["attention_rows"]:
            self.assertAlmostEqual(sum(row), 1.0, places=5)
        attribution = record["attribution"]
        self.assertAlmostEqual(sum(attribution["ig"]), attribution["ig_sum"], places=5)
        self.assertLess(abs(attribution["ig_sum"]-attribution["endpoint_change"]), .1)
        no_edit = next(row for row in record["steering"]["cases"] if row["multiplier"]==0)
        self.assertEqual(no_edit["contrast"], no_edit["random"])
        for row in record["dictionary"]:
            self.assertGreater(row["test_rows"], 0)
            self.assertLessEqual(row["mean_active"], row["features"])


if __name__=="__main__":
    unittest.main()
