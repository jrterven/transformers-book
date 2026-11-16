"""Check Chapter 32's tool contracts, recorded decisions and restart behavior."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.ch32_agents import fixed_workflow, resume_agent
from examples.ch32_agents.campus import CASES, dispatch
from examples.ch32_agents.run_agent import grade


RECORDS = Path(resume_agent.__file__).with_name("records")


def recorded(name):
    return json.loads((RECORDS / name).read_text(encoding="utf-8"))


class AgentTests(unittest.TestCase):
    def test_quote_schema_and_integer_arithmetic(self):
        valid = {"party_size": 12, "rate_cents": 875, "fee_cents": 200}
        self.assertEqual(dispatch("quote_total", valid, False), {"total_cents": 10700})
        for changes in ({"party_size": True}, {"party_size": "12"}, {"x": 3}):
            with self.subTest(changes=changes):
                self.assertIn("error", dispatch("quote_total", {**valid, **changes}, False))

    def test_fixed_workflow_reproduces_the_saved_baseline(self):
        actual = [fixed_workflow.run_case(case) for case in CASES]
        self.assertEqual(actual, recorded("FIXED_WORKFLOW.json")["runs"])

    def test_native_observations_and_grades_replay(self):
        cases = {case.name: case for case in CASES}
        for filename in ("MEASURED_RUN.json", "BUDGET_RUN.json", "CLARIFICATION_RUN.json"):
            for row in recorded(filename)["runs"]:
                with self.subTest(file=filename, case=row["case"]):
                    case = cases[row["case"]]
                    for step in row["trace"]:
                        if step["event"] == "tool_result":
                            self.assertEqual(
                                step["observation"],
                                dispatch(step["tool"], step["arguments"], case.cancelled_camera),
                            )
                    checks, success = row["checks"], row["success"]
                    grade(case, row)
                    self.assertEqual((checks, success), (row["checks"], row["success"]))

    def test_waiting_gate_rejects_an_old_room_proposal(self):
        data = recorded("HORIZON_RUN.json")
        waiting = data["waiting"]
        self.assertEqual(waiting["wake_at"], 180)
        self.assertEqual(
            resume_agent.check_proposal(waiting, "room_status", {"room_id": "R-OAK"}),
            {"error": "read_current_notice_first", "notice_id": "N-17"},
        )

    def test_recovered_quote_depends_on_current_room_evidence(self):
        data = recorded("HORIZON_RUN.json")
        for name, expected in (("baseline", None), ("guarded", 10700)):
            with self.subTest(name=name):
                state = data[name]
                self.assertEqual(state["trace"][:3], data["waiting"]["trace"])
                self.assertEqual(resume_agent.verified_records(state)[-1], expected)
                self.assertEqual(state["checks"]["current_room_before_quote"], expected is not None)

    def test_resume_before_release_or_after_completion_is_a_noop(self):
        data = recorded("HORIZON_RUN.json")
        with TemporaryDirectory() as directory, patch.object(resume_agent, "model_info") as model:
            path = Path(directory) / "checkpoint.json"
            for name, minute in (("waiting", 120), ("guarded", 185)):
                with self.subTest(name=name):
                    resume_agent.save_checkpoint(path, data[name])
                    previous = path.read_bytes()
                    self.assertEqual(resume_agent.run(path, minute), data[name])
                    self.assertEqual(path.read_bytes(), previous)
            model.assert_not_called()

    def test_expiration_preserves_committed_calls(self):
        data = recorded("HORIZON_RUN.json")
        with TemporaryDirectory() as directory, patch.object(resume_agent, "model_info") as model:
            path = Path(directory) / "checkpoint.json"
            resume_agent.save_checkpoint(path, data["waiting"])
            expired = resume_agent.run(path, minute=240)
            self.assertEqual(expired, data["expired"])
            self.assertEqual(expired["executed_calls"], 2)
            model.assert_not_called()
