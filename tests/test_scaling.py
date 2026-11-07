"""Check the Chapter 9 budget, router, and expert computations."""

import json
import unittest

import numpy as np

from examples.ch09_scaling.model import (
    BILLION,
    allocation_sweep,
    expert_branch,
    route_top1,
    topk_gates,
    tokens_at_budget,
    training_flops,
)
from examples.ch09_scaling.run_examples import (
    BUDGET,
    MODEL_SIZES_B,
    capacity_example,
    balance_diagnostic_example,
    top2_example,
    token_example,
)


class BudgetTests(unittest.TestCase):
    def test_equal_budget_and_larger_run(self):
        for n, d in ((10, 400), (20, 200), (40, 100)):
            with self.subTest(parameters_b=n, tokens_b=d):
                self.assertEqual(training_flops(n * BILLION, d * BILLION), BUDGET)
                self.assertEqual(tokens_at_budget(BUDGET, n * BILLION), d * BILLION)
        self.assertEqual(training_flops(20 * BILLION, 800 * BILLION), 4 * BUDGET)

    def test_numpy_counts_preserve_large_integer_flop_estimate(self):
        n, d = 10 * BILLION, 400 * BILLION
        expected = 6 * n * d
        for parameters, tokens in ((np.int64(n), d), (n, np.int64(d)),
                                   (np.int64(n), np.int64(d))):
            with self.subTest(parameters=type(parameters), tokens=type(tokens)):
                self.assertEqual(training_flops(parameters, tokens), expected)

    def test_sweep_preserves_budget_and_compares_synthetic_losses(self):
        rows = allocation_sweep(BUDGET, MODEL_SIZES_B)
        np.testing.assert_allclose(rows, [
            (5, 800, 1.249596116318638),
            (10, 400, 1.2264911064067352),
            (20, 200, 1.230864075337301),
            (40, 100, 1.2632455532033675),
            (80, 50, 1.327564072024615),
        ], rtol=0, atol=1e-14)
        self.assertEqual(min(rows, key=lambda row: row[2])[:2], (10, 400))
        larger_rows = allocation_sweep(4 * BUDGET, MODEL_SIZES_B)
        self.assertEqual(min(larger_rows, key=lambda row: row[2])[:2], (20, 800))

    def test_rejects_invalid_budget_inputs(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    training_flops(value, 1)
                with self.assertRaises(ValueError):
                    training_flops(1, value)
                with self.assertRaises(ValueError):
                    tokens_at_budget(value, 1)
                with self.assertRaises(ValueError):
                    tokens_at_budget(1, value)


class RoutingTests(unittest.TestCase):
    def test_one_token_gate_and_residual(self):
        record = token_example()
        np.testing.assert_allclose(record["probabilities"], [0.1, 0.6, 0.2, 0.1])
        self.assertEqual(record["expert"], 2)
        self.assertAlmostEqual(record["gate"], 0.6)
        self.assertEqual(record["expert_output"], [2, -1, 0, 3])
        np.testing.assert_allclose(record["branch"], [1.2, -0.6, 0, 1.8])
        np.testing.assert_allclose(record["after_residual"], [2.2, -0.6, 0, 1.8])
        self.assertEqual(record["stored_expert_weights"], 256)
        self.assertEqual(record["selected_expert_weights"], 64)
        self.assertEqual(record["router_weights"], 16)

    def test_capacity_counts_and_bypassed_rows(self):
        record = capacity_example()
        self.assertEqual(record["requests"], [1, 2, 2, 2, 2, 2, 3, 4])
        self.assertEqual(record["requested_per_expert"], [1, 5, 1, 1])
        self.assertEqual(record["processed_per_expert"], [1, 3, 1, 1])
        self.assertEqual(record["overflow_rows"], [5, 6])
        np.testing.assert_array_equal(np.array(record["branch"])[4:6], 0)
        np.testing.assert_array_equal(record["after_residual"][4:6], record["input"][4:6])
        np.testing.assert_allclose(record["mean_probability"], [0.1625, 0.45, 0.225, 0.1625])
        self.assertAlmostEqual(record["raw_balance_term"], 1.4)

    def test_more_capacity_accepts_more_rows_without_changing_requests(self):
        for capacity, expected in (
            (0, [0, 0, 0, 0]), (2, [1, 2, 1, 1]), (5, [1, 5, 1, 1]),
        ):
            with self.subTest(capacity=capacity):
                record = capacity_example(capacity)
                self.assertEqual(record["requested_per_expert"], [1, 5, 1, 1])
                self.assertEqual(record["processed_per_expert"], expected)
                self.assertAlmostEqual(record["raw_balance_term"], 1.4)
        zero_capacity = capacity_example(0)
        self.assertEqual(zero_capacity["input"], zero_capacity["after_residual"])

    def test_numpy_capacity_has_json_serializable_record(self):
        record = json.loads(json.dumps(capacity_example(np.int64(2))))
        self.assertEqual(record["capacity"], 2)
        self.assertEqual(record["overflow_rows"], [4, 5, 6])

    def test_softmax_rows_ties_and_capacity_order(self):
        base = [[2, 2], [1, 2], [2, 1], [1, 2]]
        shifted = base + np.array([[10000.0], [-10000.0], [3.0], [4.0]])
        probabilities, selected, gates, accepted = route_top1(shifted, 1)
        np.testing.assert_allclose(probabilities, route_top1(base, 1)[0])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(selected, [0, 1, 0, 1])
        np.testing.assert_allclose(gates, probabilities[np.arange(4), selected])
        np.testing.assert_array_equal(accepted, [True, True, False, False])

    def test_largest_logit_wins_when_probabilities_round_to_a_tie(self):
        probabilities, selected, gates, accepted = route_top1([[0, 1e-20]], 1)
        np.testing.assert_array_equal(probabilities, [[0.5, 0.5]])
        np.testing.assert_array_equal(selected, [1])
        np.testing.assert_array_equal(gates, [0.5])
        np.testing.assert_array_equal(accepted, [True])

    def test_expert_rows_use_own_weights_and_return_to_input_order(self):
        x = [[1, -2], [3, 4], [5, 6], [-1, 2]]
        logits = [[0, 1], [1, 0], [1, 0], [0, 1]]
        _, selected, gates, accepted = route_top1(logits, 2)
        w_up = [[[1], [0]], [[0], [1]]]
        w_down = [[[1, 0]], [[0, 2]]]
        branch = expert_branch(x, selected, gates, accepted, w_up, w_down)
        np.testing.assert_allclose(
            branch,
            np.array([[0, 0], [3, 0], [5, 0], [0, 4]]) / (1 + np.exp(-1)),
        )

    def test_invalid_router_inputs(self):
        for logits in ([], [1, 2], [[]], [[np.inf, 0]], [[np.nan, 0]]):
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                route_top1(logits, 1)
        for capacity in (-1, 1.5, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                route_top1([[0, 1]], capacity)

    def test_top_two_combines_given_experts_before_one_residual_addition(self):
        result = top2_example()
        self.assertEqual(result["experts"], [2, 3])
        np.testing.assert_allclose(result["gates"], [0.75, 0.25], atol=1e-14)
        np.testing.assert_allclose(result["branch"], [1.5, -0.5, 1, 2], atol=1e-14)
        np.testing.assert_allclose(result["after_residual"], [2.5, -0.5, 1, 2])

    def test_selected_gate_normalization_differs_from_switch_and_ignores_offset(self):
        scores = np.log([[0.1, 0.6, 0.2, 0.1]])
        ids, gates = topk_gates(scores, 1)
        np.testing.assert_array_equal(ids, [[1]])
        np.testing.assert_array_equal(gates, [[1.]])
        self.assertAlmostEqual(route_top1(scores, 1)[2][0], 0.6)
        a_ids, a_gates = topk_gates(scores, 2)
        b_ids, b_gates = topk_gates(scores + 1000, 2)
        np.testing.assert_array_equal(a_ids, b_ids)
        np.testing.assert_allclose(a_gates, b_gates, rtol=0, atol=1e-12)

    def test_topk_resolves_ties_in_expert_order_and_validates_selection(self):
        ids, gates = topk_gates([[0, 0, 0, 0]], 2)
        np.testing.assert_array_equal(ids, [[0, 1]])
        np.testing.assert_array_equal(gates, [[0.5, 0.5]])
        for k in (0, 5, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                topk_gates([[0, 1, 2, 3]], k)
        with self.assertRaises(ValueError):
            topk_gates([[np.nan, 0]], 1)

    def test_auxiliary_reference_value_does_not_certify_equal_request_counts(self):
        result = balance_diagnostic_example()
        for record in result.values():
            self.assertAlmostEqual(record["raw_balance_term"], 1.)
        self.assertEqual(result["tied"]["requests_per_expert"], [4, 0, 0, 0])
        self.assertEqual(result["tied"]["accepted_rows"], [1])
        self.assertEqual(result["tied"]["overflow_fraction"], 0.75)
        self.assertEqual(result["balanced"]["requests_per_expert"], [1, 1, 1, 1])
        self.assertEqual(result["balanced"]["accepted_rows"], [1, 2, 3, 4])
        self.assertEqual(result["balanced"]["overflow_fraction"], 0.)


if __name__ == "__main__":
    unittest.main()
