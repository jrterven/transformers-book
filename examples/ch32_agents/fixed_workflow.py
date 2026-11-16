"""Conditional baseline when a session title and group size are supplied."""

import argparse
import json
from pathlib import Path

from .campus import CASES, dispatch
from .run_agent import grade


def decision(case, call):
    match = call("find_session", query=case.title)
    if match["status"] == "ambiguous":
        choices = match["matches"]
        titles = " or ".join(row["title"] for row in choices)
        return "ask", None, None, [row["id"] for row in choices], titles + "?"
    if match["status"] == "not_found":
        return "insufficient", None, None, [], "Please check the workshop title."

    session = match["session"]
    evidence = [session["id"]]
    room = session["room_id"]
    rate = session["rate_cents"]
    # snippet:workflow_revision
    if session["revision_id"] is not None:
        notice = call("read_notice", notice_id=session["revision_id"])
        if "error" in notice:
            return "insufficient", None, None, evidence, "I cannot verify the revision."
        evidence.append(notice["id"])
        if notice["status"] == "cancelled":
            return "cancelled", None, None, evidence, "The workshop is cancelled."
        room = notice.get("room_id", room)
        rate = notice.get("rate_cents", rate)
    # end-snippet:workflow_revision

    available = call("room_status", room_id=room)
    if "error" in available:
        return "insufficient", None, None, evidence, "I cannot verify the room."
    evidence.append(available["id"])
    if not available["open"]:
        return "unavailable", room, None, evidence, "The room is closed."
    if case.party > available["capacity"]:
        return "too_large", room, None, evidence, "The group exceeds the room capacity."
    quote = call("quote_total", party_size=case.party, rate_cents=rate,
                 fee_cents=session["fee_cents"])
    return "fits", room, quote["total_cents"], evidence, "The group fits."


def run_case(case):
    trace = []

    def call(tool, **arguments):
        observation = dispatch(tool, arguments, case.cancelled_camera)
        trace.append({"event": "tool_result", "tool": tool,
                      "arguments": arguments, "observation": observation})
        return observation

    status, room, cents, evidence, message = decision(case, call)
    final = {"status": status, "room_id": room, "total_cents": cents,
             "evidence": evidence, "message": message}
    row = {"case": case.name, "trace": trace, "explained_inputs": {
        "title": case.title, "party_size": case.party}, "final": final,
        "executed_calls": len(trace), "model_turns": 0}
    grade(case, row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"input": "supplied title and group size", "runs": [run_case(case) for case in CASES]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(sum(row["success"] for row in report["runs"]), "of", len(CASES), "passed")


if __name__ == "__main__":
    main()
