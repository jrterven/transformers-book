"""Explicit history buffers and summaries for a delivery toy task."""

from collections import deque


def _check_window(window):
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")


def _check_event(index, kind, value):
    if index == 0:
        if kind != "code" or not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("history must start with one integer code event")
    elif kind != "route" or not isinstance(value, str) or not value:
        raise ValueError("after the code, each event must assign a nonempty route label")


def make_history(first_code, routes):
    """Build a history from one integer code and an iterable of route labels."""
    _check_event(0, "code", first_code)
    events = [("code", first_code)]
    for route in routes:
        _check_event(1, "route", route)
        events.append(("route", route))
    return tuple(events)


def _answers(retained, unit):
    # Turning ordered events into a dictionary keeps the latest route assignment.
    fields = dict(retained)
    return {
        "first_code": fields.get("code"),
        "current_route": fields.get("route"),
        "retained_units": len(retained),
        "unit": unit,
    }


# snippet:compare_memory
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
# end-snippet:compare_memory


# snippet:causal_pair_counts
def causal_pair_counts(length, window):
    """Count full and windowed causal pairs, including each self-pair."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    _check_window(window)
    width = min(length, window)
    full = length * (length + 1) // 2
    local = length * width - width * (width - 1) // 2
    return full, local
# end-snippet:causal_pair_counts
