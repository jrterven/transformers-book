"""Pause and resume the campus agent using a file and a simulated clock."""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .campus import ASK_CAMERA, CASES, SYSTEM, dispatch
from .run_agent import (
    MAX_TURNS, final_object, grade, model_info, next_response, public_message,
    tool_message,
)

PUBLICATION_MINUTE = 180
DEADLINE_MINUTE = 240
TOOL_BUDGET = 6
GOAL = ASK_CAMERA + " If the updated notice is still pending, wait for its release."


def new_run(model, guard=False):
    return {
        "version": 1, "goal": GOAL, "model": model, "minute": 0,
        "deadline": DEADLINE_MINUTE, "status": "running", "wake_at": None, "guard": guard,
        "max_tool_attempts": TOOL_BUDGET, "max_model_turns": MAX_TURNS,
        "tool_attempts": 0, "executed_calls": 0, "model_turns": 0, "final": None,
        "trace": [],
        "public_messages": [{"role": "user", "content": GOAL}],
    }


def save_checkpoint(path, state):
    file = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    staging = Path(file.name)
    try:
        with file:
            json.dump(state, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def read_tool(tool, arguments, minute):
    if tool == "read_notice" and arguments == {"notice_id": "N-17"} and minute < PUBLICATION_MINUTE:
        return {"id": "N-17", "event_id": "E-CAM", "status": "pending",
                "available_at_minute": PUBLICATION_MINUTE}
    return dispatch(tool, arguments, cancelled_camera=False)


def verified_records(state):
    """Rebuild the currently supported notice, room and quote from observations."""
    notice, room, quote = {}, {}, None
    for step in state["trace"]:
        if step["event"] != "tool_result":
            continue
        name, result = step["tool"], step["observation"]
        if name == "read_notice" and result.get("id") == "N-17":
            notice = result if result.get("status") == "active" else {}
            room, quote = {}, None
        if name == "room_status" and notice and result.get("id") == notice["room_id"]:
            room, quote = result, None
        if name == "quote_total" and room.get("open") and room["capacity"] >= 12:
            expected = {"party_size": 12, "rate_cents": notice["rate_cents"], "fee_cents": 200}
            if step["arguments"] == expected:
                quote = result.get("total_cents")
    return notice, room, quote


def check_proposal(state, tool, arguments):
    if tool not in {"room_status", "quote_total"}:
        return None
    notice, room, _ = verified_records(state)
    if not notice:
        return {"error": "read_current_notice_first", "notice_id": "N-17"}
    if tool == "room_status" and arguments != {"room_id": notice["room_id"]}:
        return {"error": "use_current_room", "room_id": notice["room_id"]}
    if tool == "quote_total" and not room:
        return {"error": "verify_current_room_first", "room_id": notice["room_id"]}
    return None


def run_turn(state, guard):
    minute = state["minute"]
    instructions = (
        SYSTEM + f"\nThe current simulated minute is {minute}. "
        "If a previous notice was pending and its release time has arrived, "
        "obtain the current notice before making the room or quote decision."
    )
    response = next_response([{"role": "system", "content": instructions}, *state["public_messages"]])
    assistant = public_message(response["message"])
    state["public_messages"].append(assistant)
    state["model_turns"] += 1
    entry = {"minute": minute, "turn": state["model_turns"],
             "model_seconds": round(response.get("total_duration", 0) / 1e9, 3)}
    calls = assistant.get("tool_calls") or []
    if not calls:
        state["final"] = final_object(assistant.get("content", ""))
        state["trace"].append({**entry, "event": "final"})
        grade(CASES[0], state)
        _, _, supported_quote = verified_records(state)
        state["checks"]["current_room_before_quote"] = supported_quote == 10700
        state["success"] = all(state["checks"].values())
        state["status"] = "complete" if state["success"] else "unverified"
        return
    if len(calls) != 1 or state["tool_attempts"] >= state["max_tool_attempts"]:
        state["status"] = "stopped"
        state["trace"].append({**entry, "event": "call_limit", "proposed_calls": calls})
        return

    call = calls[0]["function"]
    tool, arguments = call["name"], call.get("arguments", {})
    state["tool_attempts"] += 1
    observation = check_proposal(state, tool, arguments) if guard else None
    event = "proposal_rejected"
    if observation is None:
        observation = read_tool(tool, arguments, minute)
        state["executed_calls"] += 1
        event = "tool_result"
    state["public_messages"].append(tool_message(tool, observation))
    state["trace"].append({**entry, "event": event, "tool": tool,
                           "arguments": arguments, "observation": observation})
    if observation.get("status") == "pending":
        state["status"] = "waiting"
        state["wake_at"] = observation["available_at_minute"]
        state["trace"].append({"event": "waiting", "minute": minute, "wake_at": state["wake_at"]})


def run(path, minute, guard=None):
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["version"] != 1 or state["goal"] != GOAL:
            raise ValueError("This checkpoint belongs to a different demo revision.")
        if minute < state["minute"]:
            raise ValueError("The clock cannot move behind the last committed minute.")
        if state["status"] not in {"running", "waiting"}:
            return state
        if minute >= state["deadline"]:
            state.update(minute=minute, status="expired")
            state["trace"].append({"event": "deadline", "minute": minute})
            save_checkpoint(path, state)
            return state
        if state["status"] == "waiting" and minute < state["wake_at"]:
            return state
        if model_info() != state["model"]:
            raise ValueError("Restore the model and serving configuration used to start the run.")
        state.update(minute=minute, status="running", wake_at=None)
        if guard is not None:
            state["guard"] = guard
        state["trace"].append({"event": "resumed", "minute": minute, "guard": state["guard"],
                               "saved_calls": state["executed_calls"]})
    else:
        if minute != 0:
            raise ValueError("Start a new experiment at minute 0.")
        state = new_run(model_info(), bool(guard))
    save_checkpoint(path, state)

    while state["status"] == "running":
        if state["model_turns"] >= state["max_model_turns"]:
            state["status"] = "stopped"
            state["trace"].append({"event": "turn_limit", "minute": minute})
        else:
            run_turn(state, state["guard"])
        save_checkpoint(path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--minute", type=int, required=True)
    parser.add_argument("--guard", action="store_true", default=None,
                        help="Require the current notice and room before quoting.")
    args = parser.parse_args()
    state = run(args.checkpoint, args.minute, args.guard)
    print(json.dumps({key: state[key] for key in (
        "status", "minute", "wake_at", "tool_attempts", "executed_calls", "model_turns",
        "final")}, indent=2))


if __name__ == "__main__":
    main()
