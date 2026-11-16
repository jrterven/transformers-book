def next_response(messages):
    response = api("/api/chat", {
        "model": TAG, "messages": messages, "tools": TOOLS, "stream": False,
        "think": "low", "options": OPTIONS, "keep_alive": "3m",
    })
    return response
