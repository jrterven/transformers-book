"""Fixed inputs and bounded evaluator for the Chapter 27 model demonstrations."""

import ast
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile


FIM_PREFIX = 'def overlaps(a, b):\n    """Return whether the two integer intervals overlap."""\n    '
FIM_SUFFIX = (
    "\n\nassert overlaps((0, 2), (2, 5)) is False\n"
    "assert overlaps((0, 3), (2, 5)) is True\n"
)
FIM_MARKERS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
OPERATOR_PREFIX = "return max(a[0], b[0])"

GENERATE = """Write a Python function with the signature def peak_bookings(bookings):
Return the largest number of bookings active at any one integer time. Each
booking is a pair (start, end) with integer endpoints and start <= end. It is
active from start inclusive to end exclusive. Return 0 if no booking is active.
Duplicate bookings count separately. Do not change the input list.
Endpoints can be far apart, so runtime must not scale with their distance.
Use built-ins only. Return exactly one Python function, without imports."""

BASE = """def conflicts(a, b):
    start_a, stop_a = bounds(a)
    start_b, stop_b = bounds(b)
    return max(start_a, start_b) <= min(stop_a, stop_b)
"""
POLICY = '''def bounds(booking, *, occupied=False):
    """Return displayed times, or the interval that consumes capacity."""
    start, end, cleanup, cancelled = booking
    if not occupied:
        return start, end
    if cancelled or start == end:
        return start, start
    return start, end + cleanup
'''
ISSUE = """Replace conflicts(a, b) in calendar.py. The function must return a bool.
Adjacent bookings without cleanup are reported as a conflict. A conflict
means the two bookings share time when both consume capacity. A booking's
displayed times can differ from the time that consumes capacity.
Each booking is (start, end, cleanup, cancelled), with ordered integer times
and nonnegative cleanup. The existing bounds helper is already imported.
Use it to read the bookings. Return exactly one replacement Python function,
without imports. Keep the same signature.

calendar.py:
from .policy import bounds

"""


def repair_prompt(with_policy):
    extra = "\npolicy.py:\n" + POLICY if with_policy else ""
    return ISSUE + BASE + extra


def overlap_cases():
    pairs = (
        ("interior", (-1, 5), (1, 4)),
        ("gap", (1, 3), (5, 7)),
        ("touching", (5, 7), (2, 5)),
        ("equal endpoints inside", (-1, 2), (0, 0)),
        ("one shared", (-4, -1), (-2, 1)),
        ("both equal endpoints", (3, 3), (3, 3)),
    )
    return [[name, [a, b], bool(set(range(*a)) & set(range(*b)))] for name, a, b in pairs]


def generation_cases():
    # The three intervals share time 2. This is the only named peak above two.
    cases = [
        ["no bookings", [[]], 0],
        ["three active", [[(0, 5), (1, 3), (2, 4)]], 3],
        ["touching", [[(0, 2), (2, 4)]], 1],
        ["empty inside", [[(4, 4), (3, 5), (4, 4)]], 1],
        ["unsorted duplicates", [[(2, 7), (-4, 3), (2, 7), (-2, 1)]], 3],
        ["wide time span", [[(-10**9, 10**9), (-4, -2), (-2, 8)]], 2],
    ]
    rng = random.Random(2626)
    for i in range(48):
        bookings = [tuple(sorted(rng.choices(range(-4, 6), k=2))) for _ in range(rng.randrange(9))]
        # Enumerate active bookings independently of any endpoint-sorting implementation.
        peak = max(sum(t in range(a, b) for a, b in bookings) for t in range(-4, 6))
        cases.append([f"enumerated {i + 1}", [bookings], peak])
    return cases


def repair_cases():
    def slot(start, end, cleanup=0, cancelled=False):
        return (start, end, cleanup, cancelled)

    # The first four cases fail on BASE. The rest already pass.
    return [
        ["touching", [slot(5, 7), slot(2, 5)], False],
        ["empty inside", [slot(0, 4), slot(2, 2, 4)], False],
        ["cleanup overlaps", [slot(0, 2, 3), slot(3, 4)], True],
        ["cancelled", [slot(0, 4, 3, True), slot(2, 3)], False],
        ["interior overlap", [slot(0, 6), slot(2, 4)], True],
        ["one shared", [slot(1, 5), slot(4, 6)], True],
        ["separate", [slot(0, 2, 1), slot(4, 5)], False],
        ["negative endpoints", [slot(-4, -1), slot(-2, 2)], True],
    ]


BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "max": max, "min": min,
    "range": range, "reversed": reversed, "set": set, "sorted": sorted, "sum": sum,
    "tuple": tuple, "zip": zip,
}
METHODS = {"append", "copy", "extend", "get", "items", "keys", "pop", "setdefault", "sort", "values"}
UNAVAILABLE_NAMES = {"open", "exec", "eval", "compile", "input", "globals", "locals", "getattr", "setattr", "vars"}
DISALLOWED = (ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Global, ast.Import,
              ast.ImportFrom, ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFor)
PROFILE = """
(version 1)
(allow default)
(deny file-write*)
(deny network*)
(deny process-fork)
(deny file-read*
  (require-all
    (subpath "/Users")
    (require-not
      (require-any (subpath (param "PYTHON_ROOT")) (literal (param "CHILD_SOURCE"))))))
"""


def validate(source, function):
    if len(source) > 7000:
        raise ValueError("source exceeds the teaching evaluator's limit")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != function:
        raise ValueError("expected exactly the requested function")
    for node in ast.walk(tree):
        if isinstance(node, DISALLOWED):
            raise ValueError(f"{type(node).__name__} is outside this exercise")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.args.defaults or node.args.kw_defaults):
            raise ValueError("function decorators and defaults are outside this exercise")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in UNAVAILABLE_NAMES):
            raise ValueError(f"name {node.id} is outside this exercise")
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            raise ValueError(f"attribute {node.attr} is outside this exercise")
    return tree


def child(request):
    from copy import deepcopy
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    source, function, cases = request["source"], request["function"], request["cases"]
    env = {"__builtins__": BUILTINS}
    if function == "conflicts":
        exec(compile(POLICY, "<checked-in policy>", "exec"), env)
    exec(compile(validate(source, function), "<candidate>", "exec"), env)
    candidate = env[function]
    results = []
    for name, arguments, expected in cases:
        args = [list(map(tuple, arguments[0]))] if function == "peak_bookings" else list(map(tuple, arguments))
        before = deepcopy(args)
        record = {"case": name, "expected": expected}
        try:
            observed = candidate(*args)
            valid = type(observed) is type(expected)
            record.update(observed=observed if valid else None, output_type=type(observed).__name__,
                          input_preserved=args == before, passed=valid and observed == expected and args == before)
        except Exception as error:
            record.update(observed=None, error=type(error).__name__, passed=False)
        results.append(record)
    return results


def evaluate(source, function, cases):
    """Evaluate a single fixture submission in a fresh, restricted macOS child."""
    try:
        validate(source, function)
    except (SyntaxError, ValueError) as error:
        return {"status": "format", "detail": str(error), "passed": False, "checks": []}
    if sys.platform != "darwin":
        raise RuntimeError("This demonstration's subprocess profile requires macOS")
    path = str(Path(__file__).resolve())
    cmd = ["/usr/bin/sandbox-exec", "-D", f"PYTHON_ROOT={Path(sys.base_prefix).resolve()}",
           "-D", f"CHILD_SOURCE={path}", "-p", PROFILE, sys.executable, "-I", "-S", path]
    request = json.dumps({"source": source, "function": function, "cases": cases})
    with tempfile.TemporaryDirectory(prefix="ch27-eval-") as directory:
        try:
            result = subprocess.run(cmd, input=request, capture_output=True, text=True,
                                    timeout=4, cwd=directory, env={"LANG": "C.UTF-8"})
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": False, "checks": []}
    if result.returncode and result.stderr.startswith("sandbox-exec:"):
        raise RuntimeError(result.stderr.strip())
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        return {"status": "runtime", "passed": False, "checks": [],
                "detail": lines[-1] if lines else f"exit {result.returncode}"}
    checks = json.loads(result.stdout)
    return {"status": "tested", "passed": all(check["passed"] for check in checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(child(json.load(sys.stdin)), allow_nan=False))
