"""Check Chapter 26's callables, behavioral checks, and pass@k."""

from itertools import combinations, product
import unittest

from examples.ch26_code_models.model import (
    Case,
    pass_at_k,
    positive_length,
    score_candidate,
    serialize_fim,
    strict_overlap,
)
from examples.ch26_code_models.run_examples import CANDIDATES, SMOKE, benchmark_example


class BehaviorTests(unittest.TestCase):
    def test_worked_benchmark_counts_and_failure_modes(self):
        report = benchmark_example()
        rows = report["candidates"]
        self.assertEqual([row["name"] for row in rows],
                         [candidate.__name__ for candidate in CANDIDATES])
        self.assertEqual([row["smoke"]["passed_cases"] for row in rows], [2, 2, 2, 2])
        self.assertEqual([row["complete_passed_cases"] for row in rows], [4, 6, 5, 6])
        self.assertEqual(report["smoke_correct"], 4)
        self.assertEqual(report["complete_correct"], 2)
        self.assertEqual(report["complete_pass_at_1"], 0.5)
        self.assertEqual(report["complete_pass_at_2"], 5 / 6)
        failures = [[result["case"] for result in row["edges"]["per_case"]
                     if not result["passed"]] for row in rows]
        self.assertEqual(failures, [["touching", "empty"], [], ["length_one"], []])

    def test_boolean_output_and_exception_records(self):
        cases = (Case("overlap", (0, 3), (1, 5), True),)
        wrong_type = score_candidate(lambda a, b: 1, cases)
        self.assertFalse(wrong_type["passed_all"])
        self.assertEqual(wrong_type["per_case"][0]["output_type"], "int")
        self.assertIsNone(wrong_type["per_case"][0]["observed"])

        def raises(a, b):
            raise RuntimeError("a fixed test exception")

        exception = score_candidate(raises, cases)
        self.assertEqual(exception["passed_cases"], 0)
        self.assertEqual(exception["per_case"][0]["error"], "RuntimeError")

    def test_cases_validate_intervals_and_store_immutable_bounds(self):
        mutable = [0, 3]
        case = Case("overlap", mutable, (1, 5), True)
        mutable[0] = 4
        self.assertEqual(case.a, (0, 3))
        with self.assertRaisesRegex(ValueError, "start"):
            Case("invalid", (4, 2), (1, 3), False)
        with self.assertRaisesRegex(TypeError, "Python bool"):
            Case("invalid", (1, 2), (2, 3), 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            score_candidate(strict_overlap, [case, case])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            score_candidate(strict_overlap, [])
        with self.assertRaisesRegex(TypeError, "callable"):
            score_candidate("candidate source text", SMOKE)

    def test_correct_candidates_match_discrete_interval_sets(self):
        intervals = [(a, b) for a in range(-3, 4) for b in range(a, 4)]
        cases = []
        for i, (a, b) in enumerate(product(intervals, repeat=2)):
            expected = bool(set(range(*a)) & set(range(*b)))
            cases.append(Case(str(i), a, b, expected))
        for candidate in (strict_overlap, positive_length):
            self.assertTrue(score_candidate(candidate, cases)["passed_all"])


class EstimationTests(unittest.TestCase):
    def test_pass_at_k_matches_uniform_subsets(self):
        for n in range(1, 8):
            for correct in range(n + 1):
                for k in range(1, n + 1):
                    groups = tuple(combinations(range(n), k))
                    expected = sum(any(i < correct for i in g) for g in groups) / len(groups)
                    self.assertEqual(pass_at_k(n, correct, k), expected)

    def test_pass_at_k_handles_small_nonzero_probabilities_and_validation(self):
        self.assertEqual(pass_at_k(10**20, 1, 1), 1e-20)
        for arguments in ((4, 2, 0), (4, 2, 5), (4, 5, 2), (0, 0, 1), (4, -1, 2)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                pass_at_k(*arguments)
        with self.assertRaises(TypeError):
            pass_at_k(4.0, 2, 2)

    def test_fim_format_preserves_prefix_and_suffix_whitespace(self):
        self.assertEqual(
            serialize_fim("def f():\n    ", "\n# footer"),
            "<fim_prefix>def f():\n    <fim_suffix>\n# footer<fim_middle>",
        )
        with self.assertRaisesRegex(TypeError, "strings"):
            serialize_fim("prefix", None)


if __name__ == "__main__":
    unittest.main()
