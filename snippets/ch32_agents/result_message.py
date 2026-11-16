def tool_message(name, observation):
    return {
        "role": "tool", "tool_name": name,
        "content": json.dumps(observation, ensure_ascii=True),
    }
