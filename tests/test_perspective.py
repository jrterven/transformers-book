"""Check Chapter 32's exact-answer stores and causal connectivity counts."""

import unittest

from examples.ch32_perspective.model import causal_pair_counts, compare_memory, make_history
from examples.ch32_perspective.run_examples import accuracy_sweep, memory_examples


class MemoryTests(unittest.TestCase):
    def test_worked_histories_keep_different_answers_when_the_code_is_retained(self):
        first, second = memory_examples()
        self.assertEqual(first["events"], (("code", 731), ("route", "A"),
                                           ("route", "B"), ("route", "C")))
        for name, retained, unit in (("full_archive", 4, "event"),
                                     ("code_and_route", 2, "field")):
            self.assertEqual(first["answers"][name], {"first_code": 731, "current_route": "C",
                                                     "retained_units": retained, "unit": unit})
            self.assertEqual(second["answers"][name]["first_code"], 954)

    def test_different_codes_produce_identical_window_and_route_only_answers(self):
        first, second = memory_examples()
        for name, retained, unit in (("recent_window", 3, "event"), ("route_only", 1, "field")):
            expected = {"first_code": None, "current_route": "C",
                        "retained_units": retained, "unit": unit}
            self.assertEqual(first["answers"][name], expected)
            self.assertEqual(second["answers"][name], expected)

    def test_recent_window_can_still_answer_code_until_an_event_evicts_it(self):
        for window, code, units in ((1, None, 1), (2, None, 2), (3, 0, 3), (8, 0, 3)):
            with self.subTest(window=window):
                events = make_history(0, ["north", "west"])
                answers = compare_memory(iter(events), window=window)["recent_window"]
                self.assertEqual(answers["first_code"], code)
                self.assertEqual(answers["current_route"], "west")
                self.assertEqual(answers["retained_units"], units)

    def test_code_only_history_has_no_current_route_or_route_field(self):
        answers = compare_memory(make_history(731, []), window=1)
        for name, result in answers.items():
            with self.subTest(name=name):
                self.assertIsNone(result["current_route"])
                self.assertEqual(result["retained_units"], 0 if name == "route_only" else 1)
        self.assertIsNone(answers["route_only"]["first_code"])
        self.assertEqual(answers["code_and_route"]["first_code"], 731)

    def test_repeated_route_assignment_is_an_event_but_keeps_one_field(self):
        answers = compare_memory(make_history(731, ["east", "east", "east"]), window=2)
        self.assertEqual(answers["full_archive"]["retained_units"], 4)
        self.assertEqual(answers["route_only"]["retained_units"], 1)
        self.assertEqual(answers["recent_window"]["current_route"], "east")

    def test_invalid_history_and_window_are_rejected(self):
        for events in ([], [("route", "A")], [("code", "731")], [("code", True)]):
            with self.subTest(events=events), self.assertRaisesRegex(ValueError, "code event"):
                compare_memory(events, window=2)
        for event in (("code", 954), ("route", None), ("route", ""), ("stop", "A")):
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, "route label"):
                compare_memory([("code", 731), event], window=2)
        with self.assertRaisesRegex(ValueError, "route label"):
            make_history(731, ["A", ""])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compare_memory(make_history(731, []), window=0)

    def test_sweep_scores_code_and_route_independently(self):
        rows = accuracy_sweep()
        self.assertEqual([row["route_updates"] for row in rows], [1, 2, 4, 8])
        self.assertEqual([row["cases"] for row in rows], [2, 2, 2, 2])
        for name, expected_scores, expected_units in (
                ("full_archive", [1, 1, 1, 1], [2, 3, 5, 9]),
                ("recent_window", [1, 1, 0, 0], [2, 3, 3, 3]),
                ("route_only", [0, 0, 0, 0], [1, 1, 1, 1]),
                ("code_and_route", [1, 1, 1, 1], [2, 2, 2, 2])):
            reports = [row["methods"][name] for row in rows]
            self.assertEqual([report["first_code_accuracy"] for report in reports], expected_scores)
            self.assertEqual([report["current_route_accuracy"] for report in reports], [1, 1, 1, 1])
            self.assertEqual([report["retained_units"] for report in reports], expected_units)


class CausalPairTests(unittest.TestCase):
    def test_chapter_counts_and_degenerate_prefixes(self):
        self.assertEqual(causal_pair_counts(8, 3), (36, 21))
        self.assertEqual(causal_pair_counts(16, 3), (136, 45))
        self.assertEqual(causal_pair_counts(0, 3), (0, 0))
        self.assertEqual(causal_pair_counts(16, 1), (136, 16))
        self.assertEqual(causal_pair_counts(16, 100), (136, 136))

    def test_closed_form_matches_enumerating_allowed_query_key_pairs(self):
        for length in range(15):
            for window in range(1, 18):
                full_pairs = [(q, k) for q in range(length) for k in range(q + 1)]
                local_pairs = [(q, k) for q, k in full_pairs if q - k < window]
                self.assertEqual(causal_pair_counts(length, window),
                                 (len(full_pairs), len(local_pairs)))

    def test_invalid_lengths_and_windows_are_rejected(self):
        for length in (-1, True, 5.0):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "length"):
                causal_pair_counts(length, 3)
        for window in (0, -1, True, 3.0):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window"):
                causal_pair_counts(5, window)


if __name__ == "__main__":
    unittest.main()
