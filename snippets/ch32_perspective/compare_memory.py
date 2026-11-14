def compare_memory(events, window):
    """Read the same two answers from four independently updated stores."""
    _check_window(window)
    archive = []
    recent = deque(maxlen=window)
    route_only = {}
    code_and_route = {}

    for index, (kind, value) in enumerate(events):
        _check_event(index, kind, value)
        event = (kind, value)
        archive.append(event)
        recent.append(event)  # Drops the oldest event when the window is full.
        if kind == "route":
            route_only["route"] = value
        code_and_route[kind] = value

    if not archive:
        raise ValueError("history must start with one integer code event")
    return {
        "full_archive": _answers(archive, "event"),
        "recent_window": _answers(recent, "event"),
        "route_only": _answers(route_only, "field"),
        "code_and_route": _answers(code_and_route, "field"),
    }
