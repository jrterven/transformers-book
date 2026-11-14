"""Check Chapter 31's return targets, action counts, and grid rollouts."""

import unittest

import numpy as np

from examples.ch31_decisions.model import START, fit_count_policies, returns_to_go, rollout
from examples.ch31_decisions.run_examples import TRAJECTORIES, grid_example


class ReturnTests(unittest.TestCase):
    def test_suffix_includes_current_reward_and_keeps_episode_end(self):
        np.testing.assert_array_equal(returns_to_go([1, -2, 3]), [2, 1, 3])
        np.testing.assert_array_equal(returns_to_go([0, 0, 0, 1]), [1, 1, 1, 1])
        np.testing.assert_array_equal(returns_to_go([0, 0.25, 0.75]), [1, 1, 0.75])
        self.assertEqual(returns_to_go([]).shape, (0,))

    def test_invalid_shape_dtype_and_nonfinite_return(self):
        for rewards in (2, [[1, 2]], ["1"], [1 + 2j]):
            with self.subTest(rewards=rewards), self.assertRaisesRegex(ValueError, "real sequence"):
                returns_to_go(rewards)
        for rewards in ([np.inf], [np.nan], [1e308, 1e308]):
            with self.subTest(rewards=rewards), self.assertRaisesRegex(ValueError, "finite"):
                returns_to_go(rewards)


class CountPolicyTests(unittest.TestCase):
    def test_majority_and_return_key_choose_different_start_actions(self):
        state_policy, return_policy = fit_count_policies(TRAJECTORIES)
        self.assertEqual(state_policy[START], "R")
        self.assertEqual(return_policy[(START, -1)], "R")
        self.assertEqual(return_policy[(START, 1)], "D")
        self.assertEqual(return_policy[((1, 2), 1)], "U")
        self.assertNotIn((START, 0), return_policy)

    def test_ties_use_fixed_action_order_independent_of_row_order(self):
        choices = [[(START, "D", 0)], [(START, "R", 0)]]
        for order in (choices, choices[::-1]):
            state_policy, return_policy = fit_count_policies(order)
            self.assertEqual(state_policy[START], "R")
            self.assertEqual(return_policy[(START, 0)], "R")

    def test_trajectory_rows_have_a_state_an_action_and_a_reward(self):
        with self.assertRaisesRegex(ValueError, "unknown action"):
            fit_count_policies([[(START, "X", 0)]])
        with self.assertRaisesRegex(ValueError, "outside"):
            fit_count_policies([[((2, 0), "U", 0)]])
        with self.assertRaises(ValueError):
            fit_count_policies([[(START, "R", 0, "extra"), (START, "D", 0)]])


class RolloutTests(unittest.TestCase):
    def test_original_grid_reproduces_both_routes(self):
        example = grid_example()
        baseline = example["state_only"]
        self.assertEqual(baseline["states"], [(0, 0), (0, 1)])
        self.assertEqual(baseline["actions"], ["R"])
        self.assertEqual(baseline["stop_reason"], "trap")
        self.assertEqual(baseline["total_reward"], -1)
        self.assertIsNone(baseline["remaining_targets"])

        conditioned = example["return_conditioned"]
        self.assertEqual(conditioned["states"], [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)])
        self.assertEqual(conditioned["actions"], ["D", "R", "R", "U"])
        self.assertEqual(conditioned["rewards"], [0, 0, 0, 1])
        self.assertEqual(conditioned["remaining_targets"], [1, 1, 1, 1, 0])
        self.assertEqual(conditioned["total_reward"], 1)
        self.assertEqual(conditioned["stop_reason"], "goal")

    def test_shifted_trap_reuses_policy_and_subtracts_negative_reward(self):
        _, policy = fit_count_policies(TRAJECTORIES)
        before = policy.copy()
        result = rollout(policy, trap=(1, 1), target_return=1)
        self.assertEqual(policy, before)
        self.assertEqual(result["states"], [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(result["actions"], ["D", "R"])
        self.assertEqual(result["rewards"], [0, -1])
        self.assertEqual(result["remaining_targets"], [1, 1, 2])
        self.assertEqual(result["total_reward"], -1)
        self.assertEqual(result["stop_reason"], "trap")

    def test_missing_target_and_missing_next_state_stop_without_an_extra_action(self):
        missing_target = grid_example()["unseen_return"]
        self.assertEqual(missing_target["actions"], [])
        self.assertEqual(missing_target["remaining_targets"], [0])
        self.assertEqual(missing_target["stop_reason"], "unsupported")

        missing_state = rollout({START: "D"})
        self.assertEqual(missing_state["states"], [(0, 0), (1, 0)])
        self.assertEqual(missing_state["actions"], ["D"])
        self.assertEqual(missing_state["stop_reason"], "unsupported")

    def test_wall_move_counts_toward_step_cap_and_target_zero_is_conditioned(self):
        result = rollout({(START, 0): "U"}, target_return=0, max_steps=2)
        self.assertEqual(result["states"], [START, START, START])
        self.assertEqual(result["actions"], ["U", "U"])
        self.assertEqual(result["remaining_targets"], [0, 0, 0])
        self.assertEqual(result["stop_reason"], "step_limit")

    def test_invalid_environment_and_policy_inputs_raise(self):
        with self.assertRaisesRegex(ValueError, "start and goal"):
            rollout({}, trap=START)
        with self.assertRaisesRegex(ValueError, "outside"):
            rollout({}, trap=(3, 0))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            rollout({}, max_steps=0)
        with self.assertRaisesRegex(ValueError, "finite real"):
            rollout({}, target_return=np.nan)
        with self.assertRaisesRegex(ValueError, "unknown action"):
            rollout({START: "X"})


if __name__ == "__main__":
    unittest.main()
