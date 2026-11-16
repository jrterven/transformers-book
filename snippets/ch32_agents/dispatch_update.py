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
