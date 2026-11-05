"""Check the Chapter 9 budget, router, and expert computations."""

import unittest

import numpy as np

from examples.ch09_scaling.model import (
    BILLION,
    allocation_sweep,
    expert_branch,
    route_top1,
    tokens_at_budget,
    training_flops,
)
from examples.ch09_scaling.run_examples import (
    BUDGET,
    MODEL_SIZES_B,
    capacity_example,
    token_example,
)


class BudgetTests(unittest.TestCase):
    def test_equal_budget_and_larger_run(self):
        for n, d in ((10, 400), (20, 200), (40, 100)):
            with self.subTest(parameters_b=n, tokens_b=d):
                self.assertEqual(training_flops(n * BILLION, d * BILLION), BUDGET)
                self.assertEqual(tokens_at_budget(BUDGET, n * BILLION), d * BILLION)
        self.assertEqual(training_flops(20 * BILLION, 800 * BILLION), 4 * BUDGET)

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

    def test_softmax_rows_ties_and_capacity_order(self):
        base = [[2, 2], [1, 2], [2, 1], [1, 2]]
        shifted = base + np.array([[10000.0], [-10000.0], [3.0], [4.0]])
        probabilities, selected, gates, accepted = route_top1(shifted, 1)
        np.testing.assert_allclose(probabilities, route_top1(base, 1)[0])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(selected, [0, 1, 0, 1])
        np.testing.assert_allclose(gates, probabilities[np.arange(4), selected])
        np.testing.assert_array_equal(accepted, [True, True, False, False])

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


if __name__ == "__main__":
    unittest.main()
