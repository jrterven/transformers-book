"""Run a native tool-calling LLM against the campus desk over local Ollama."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request

from .campus import CASES, SYSTEM, TOOLS, dispatch

ENDPOINT = "http://127.0.0.1:11434"
TAG = "gpt-oss:20b"
MODEL_DIGEST = "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
OPTIONS = {"temperature": 0, "seed": 32, "num_ctx": 8192, "num_predict": 1024}
MAX_TURNS = 10


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("The demonstration only connects to the local model server.")


HTTP = build_opener(ProxyHandler({}), RejectRedirect())


def api(path, payload=None):
    request = Request(ENDPOINT + path)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(payload).encode("utf-8")
    with HTTP.open(request, timeout=240) as response:
        return json.load(response)


def model_info():
    tags = api("/api/tags")["models"]
    matches = [row for row in tags if row["name"] == TAG]
    if len(matches) != 1 or matches[0]["digest"] != MODEL_DIGEST:
        raise RuntimeError(f"Install the local {TAG} artifact with digest {MODEL_DIGEST}.")
    return {"tag": TAG, "digest": matches[0]["digest"], "details": matches[0]["details"],
            "server": api("/api/version")["version"], "thinking": "low", "options": OPTIONS}


# snippet:model_request
def next_response(messages):
    response = api("/api/chat", {
        "model": TAG, "messages": messages, "tools": TOOLS, "stream": False,
        "think": "low", "options": OPTIONS, "keep_alive": "3m",
    })
    return response
# end-snippet:model_request


def final_object(content):
    content = content.strip()
    fence = "\x60" * 3
    if content.startswith(fence + "json\n") and content.endswith(fence):
        content = content[8:-3].strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def public_message(message):
    # Keep the native call messages for replay. Do not save internal thinking.
    return {key: message[key] for key in ("role", "content", "tool_calls") if key in message}


# snippet:result_message
def tool_message(name, observation):
    return {
        "role": "tool", "tool_name": name,
        "content": json.dumps(observation, ensure_ascii=True),
    }
# end-snippet:result_message


def run_case(case, max_calls, clarify_evidence=False):
    instructions = SYSTEM
    if clarify_evidence:
        instructions += ("\nFor status ask, put both candidate IDs returned by find_session "
                         "in evidence, even while the session choice is unresolved.")
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": case.request}]
    trace, public = [], list(messages)
    model_turns, tool_count = 0, 0
    final, terminal = None, "turn_limit"
    started = perf_counter()
    for turn in range(1, MAX_TURNS + 1):
        response = next_response(messages)
        assistant = response["message"]
        model_turns += 1
        messages.append(assistant)
        public.append(public_message(assistant))
        stats = {"turn": turn, "prompt_tokens": response.get("prompt_eval_count"),
                 "generated_tokens": response.get("eval_count"),
                 "model_seconds": round(response.get("total_duration", 0) / 1e9, 3)}
        calls = assistant.get("tool_calls") or []
        if not calls:
            content = assistant.get("content", "")
            final = final_object(content)
            terminal = "final" if final is not None else "non_json_final"
            trace.append({**stats, "event": terminal, "assistant_content": content})
            break
        if len(calls) != 1:
            terminal = "one_call_per_turn"
            trace.append({**stats, "event": terminal, "proposed_calls": calls})
            break
        # snippet:dispatch_update
        call = calls[0]["function"]
        tool, args = call["name"], call.get("arguments", {})
        if tool_count == max_calls:
            terminal = "tool_limit"
            trace.append({**stats, "event": terminal, "tool": tool, "arguments": args})
            break
        obs = dispatch(tool, args, case.cancelled_camera)
        result = tool_message(tool, obs)
        tool_count += 1
        messages.append(result)
        public.append(result)
        trace.append({**stats, "event": "tool_result", "tool": tool,
                      "arguments": args, "observation": obs})
        # end-snippet:dispatch_update
    run = {"case": case.name, "request": case.request, "cancelled_fixture": case.cancelled_camera,
           "max_tool_calls": max_calls, "model_turns": model_turns, "executed_calls": tool_count,
           "clarification_evidence_instruction": clarify_evidence,
           "terminal": terminal, "seconds": round(perf_counter() - started, 3),
           "final": final, "trace": trace, "public_messages": public}
    grade(case, run)
    return run


def grade(case, record):
    final = record["final"] or {}
    observations = [step for step in record["trace"] if step["event"] == "tool_result"]
    call_names = [step["tool"] for step in observations]
    exact_fields = (final.get("status") == case.expected_status and
                    "room_id" in final and final["room_id"] == case.expected_room and
                    "total_cents" in final and final["total_cents"] == case.expected_cents)
    evidence = final.get("evidence")
    observed_ids = set()
    for step in observations:
        result = step["observation"]
        if "id" in result:
            observed_ids.add(result["id"])
        if "session" in result:
            observed_ids.add(result["session"]["id"])
        observed_ids.update(match["id"] for match in result.get("matches", ()))
    exact_evidence = (isinstance(evidence, list) and all(isinstance(x, str) for x in evidence)
                      and set(evidence) == set(case.evidence) and set(evidence) <= observed_ids)
    required = {"fits": ("find_session", "room_status", "quote_total"),
                "too_large": ("find_session", "room_status"),
                "cancelled": ("find_session", "read_notice"), "ask": ("find_session",)}
    required_names = list(required[case.expected_status])
    if any(x.startswith("N-") for x in case.evidence) and "read_notice" not in required_names:
        required_names.insert(1, "read_notice")
    history = iter(call_names)
    legal_order = all(any(got == need for got in history) for need in required_names)
    if case.expected_status == "fits":
        quote_correct = any(s["tool"] == "quote_total" and
                            s["observation"].get("total_cents") == case.expected_cents
                            for s in observations)
    else:
        quote_correct = "quote_total" not in call_names
    no_tool_errors = all("error" not in step["observation"] for step in observations)
    ask_complete = (case.expected_status != "ask" or
                    all(option in str(final.get("message", "")).casefold()
                        for option in ("morning", "evening")))
    record["checks"] = {"fields": exact_fields, "evidence": exact_evidence,
                        "ordered_calls": legal_order, "quote": quote_correct,
                        "no_tool_errors": no_tool_errors, "clarification": ask_complete}
    record["success"] = all(record["checks"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--max-calls", type=int, default=6, choices=range(1, 9))
    parser.add_argument("--clarification-evidence", action="store_true")
    args = parser.parse_args()
    report = {"model": model_info(), "runs": []}
    for case in CASES:
        if args.include and case.name not in args.include:
            continue
        print(case.name, flush=True)
        record = run_case(case, args.max_calls, args.clarification_evidence)
        report["runs"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(record["terminal"], record["success"],
              [step.get("tool") for step in record["trace"] if "tool" in step],
              record["final"], flush=True)


if __name__ == "__main__":
    main()
