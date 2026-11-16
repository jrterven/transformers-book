"""Fictional campus records and read-only tools for the textbook demonstration."""

from dataclasses import dataclass
import re

CATALOG = (
    {"id": "E-CAM", "title": "Camera Lab", "room_id": "R-OAK", "rate_cents": 800,
     "fee_cents": 200, "revision_id": "N-17"},
    {"id": "E-PAP", "title": "Paper Making", "room_id": "R-ELM", "rate_cents": 600,
     "fee_cents": 450, "revision_id": None},
    {"id": "E-INK", "title": "Ink Workshop", "room_id": "R-MAPLE", "rate_cents": 750,
     "fee_cents": 300, "revision_id": None},
    {"id": "E-SK1", "title": "Morning Sketch Club", "room_id": "R-OAK", "rate_cents": 400,
     "fee_cents": 0, "revision_id": None},
    {"id": "E-SK2", "title": "Evening Sketch Club", "room_id": "R-MAPLE", "rate_cents": 500,
     "fee_cents": 0, "revision_id": None},
    {"id": "E-DYE", "title": "Dye Studio", "room_id": "R-OAK", "rate_cents": 1100,
     "fee_cents": 0, "revision_id": "N-24"},
)
DIRECTORY = {
    "R-OAK": {"capacity": 14, "open": True},
    "R-ELM": {"capacity": 8, "open": True},
    "R-MAPLE": {"capacity": 16, "open": True},
}
NOTICES = {
    "N-17": {"event_id": "E-CAM", "status": "active", "room_id": "R-MAPLE", "rate_cents": 875},
    "N-24": {"event_id": "E-DYE", "status": "active", "room_id": "R-ELM", "rate_cents": 1000},
}

SYSTEM = """You are a campus workshop information desk. All records are fictional.
Answer whether the group can attend and the total if it can. Use the available tools as evidence.
1. Find the session using the title or short title in the request. If the search result is ambiguous, ask which title they mean. If missing, report insufficient evidence.
2. If the catalog lists revision_id, read that notice. Its status and any supplied room or rate override the catalog. If cancelled, stop.
3. Read the current room record before saying a group can attend. If closed, report unavailable. If capacity is smaller than the group, stop without calculating a price.
4. For a group that fits, use quote_total with the current rate and the catalog fee.
Call at most one tool per turn. Never invent records, quote outputs, or IDs. Only cite IDs returned in tools you actually used.
When you have enough evidence, return one plain JSON object with:
status = fits, too_large, cancelled, ask, unavailable, or insufficient
room_id = current room ID, or null for cancelled, ask, insufficient
total_cents = exact group price for fits, or null otherwise
evidence = array of applicable catalog, revision, and room IDs
message = short readable answer. Include both candidate titles if asking which session.
The quote response is arithmetic. Use catalog, notice, and room IDs in evidence."""
TEXT = {"type": "string"}
INTEGER = {"type": "integer", "minimum": 0, "maximum": 100000}
GUESTS = {"type": "integer", "minimum": 1, "maximum": 1000}


def function_tool(name, description, fields):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": fields,
                "required": list(fields), "additionalProperties": False,
            },
        },
    }


TOOLS = [
    function_tool("find_session", "Search the workshop catalog by title.", {"query": TEXT}),
    function_tool("read_notice", "Read the update identified by a catalog revision_id.",
                  {"notice_id": TEXT}),
    function_tool("room_status", "Read opening and capacity for one current room.",
                  {"room_id": TEXT}),
    function_tool("quote_total", "Return exact cents: party size times rate, plus one group fee.",
                  {"party_size": GUESTS, "rate_cents": INTEGER, "fee_cents": INTEGER}),
]
SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"]["properties"]
           for tool in TOOLS}


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def find_session(query):
    query_terms = words(query)
    candidates = [row for row in CATALOG if query_terms and
                  (query_terms <= words(row["title"]) or words(row["title"]) <= query_terms)]
    if len(candidates) > 1:
        return {"status": "ambiguous", "matches": [
            {"id": row["id"], "title": row["title"]} for row in candidates]}
    if not candidates:
        return {"status": "not_found"}
    return {"status": "found", "session": dict(candidates[0])}


def read_notice(notice_id, cancelled_camera):
    if notice_id not in NOTICES:
        return {"error": "notice_not_found"}
    if notice_id == "N-17" and cancelled_camera:
        return {"id": notice_id, "event_id": "E-CAM", "status": "cancelled",
                "detail": "The main camera rig is being serviced. The workshop will not run."}
    return {"id": notice_id, **NOTICES[notice_id]}


def room_status(room_id):
    if room_id not in DIRECTORY:
        return {"error": "room_not_found"}
    return {"id": room_id, **DIRECTORY[room_id]}


# snippet:quote
def quote_total(party_size, rate_cents, fee_cents):
    """Compute exact cents after the runner validates the integers."""
    return {"total_cents": party_size * rate_cents + fee_cents}
# end-snippet:quote


def validate(name, arguments):
    fields = SCHEMAS.get(name)
    if fields is None:
        return "unknown_tool"
    if not isinstance(arguments, dict) or arguments.keys() != fields.keys():
        return "wrong_fields"
    for key, field in fields.items():
        value = arguments[key]
        if field["type"] == "string":
            if not isinstance(value, str) or not value.strip() or len(value) > 80:
                return "invalid_text"
        elif type(value) is not int or not field["minimum"] <= value <= field["maximum"]:
            return "integer_out_of_range"
    return None


def dispatch(name, arguments, cancelled_camera):
    error = validate(name, arguments)
    if error:
        return {"error": error}
    local_functions = {"find_session": find_session, "room_status": room_status,
                       "quote_total": quote_total}
    if name == "read_notice":
        return read_notice(cancelled_camera=cancelled_camera, **arguments)
    return local_functions[name](**arguments)


@dataclass(frozen=True)
class Case:
    name: str
    request: str
    title: str
    party: int
    expected_status: str
    expected_room: str | None
    expected_cents: int | None
    evidence: tuple[str, ...]
    cancelled_camera: bool = False


ASK_CAMERA = "Can our group of 12 attend Camera Lab, and what would it cost in total?"
CASES = (
    Case("camera_updated", ASK_CAMERA, "Camera Lab", 12, "fits", "R-MAPLE", 10700,
         ("E-CAM", "N-17", "R-MAPLE")),
    Case("camera_cancelled", ASK_CAMERA, "Camera Lab", 12, "cancelled", None, None,
         ("E-CAM", "N-17"), True),
    Case("paper_too_large", "Can 12 of us attend Paper Making? If so, what is the total?",
         "Paper Making", 12, "too_large", "R-ELM", None, ("E-PAP", "R-ELM")),
    Case("ink_fits", "We are five people for the Ink Workshop. Can we fit, and what will we pay altogether?",
         "Ink Workshop", 5, "fits", "R-MAPLE", 4050, ("E-INK", "R-MAPLE")),
    Case("sketch_ambiguous", "Can six of us attend Sketch Club and what is the group charge?",
         "Sketch Club", 6, "ask", None, None, ("E-SK1", "E-SK2")),
    Case("dye_relocated", "Check whether nine people can attend Dye Studio and give us the total if they can.",
         "Dye Studio", 9, "too_large", "R-ELM", None, ("E-DYE", "N-24", "R-ELM")),
)
