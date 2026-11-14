"""A deterministic benchmark with checked-in Python callables."""

from dataclasses import dataclass
from math import comb
from operator import index


def _interval(bounds):
    start, end = bounds
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeError("interval endpoints must be integers")
    start, end = index(start), index(end)
    if start > end:
        raise ValueError("interval start must be at most its end")
    return start, end


@dataclass(frozen=True)
class Case:
    name: str
    a: tuple[int, int]
    b: tuple[int, int]
    expected: bool

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("each case needs a name")
        if type(self.expected) is not bool:
            raise TypeError("expected must be a Python bool")
        object.__setattr__(self, "a", _interval(self.a))
        object.__setattr__(self, "b", _interval(self.b))


# snippet:interval_candidates
def touching_counts(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def strict_overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def needs_two(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) >= 2


def positive_length(a, b):
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return end - start > 0
# end-snippet:interval_candidates


def _checked_cases(cases):
    cases = tuple(cases)
    if not cases or any(not isinstance(case, Case) for case in cases):
        raise ValueError("provide a nonempty sequence of Case objects")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("case names must be unique")
    return cases


# snippet:score_candidate
def score_candidate(candidate, cases):
    """Run a known callable on a fixed suite and retain each outcome."""
    if not callable(candidate):
        raise TypeError("candidate must be a callable")
    cases = _checked_cases(cases)
    results = []
    for case in cases:
        record = {"case": case.name, "expected": case.expected, "observed": None}
        try:
            actual = candidate(case.a, case.b)
        except Exception as error:
            record["error"] = type(error).__name__
            record["passed"] = False
        else:
            valid = type(actual) is bool
            record["observed"] = actual if valid else None
            record["passed"] = valid and actual == case.expected
            if not valid:
                record["output_type"] = type(actual).__name__
        results.append(record)
    passed = sum(record["passed"] for record in results)
    return {"passed_cases": passed, "total_cases": len(cases),
            "passed_all": passed == len(cases), "per_case": results}
# end-snippet:score_candidate


# snippet:pass_at_k
def pass_at_k(n, correct, k):
    """Probability a uniform size-k subset includes a passing candidate."""
    if any(isinstance(value, bool) for value in (n, correct, k)):
        raise TypeError("n, correct, and k must be integer counts")
    n, correct, k = map(index, (n, correct, k))
    if not 1 <= k <= n or not 0 <= correct <= n:
        raise ValueError("use 1 <= k <= n and 0 <= correct <= n")
    total = comb(n, k)
    misses = comb(n - correct, k)
    return (total - misses) / total
# end-snippet:pass_at_k


# snippet:serialize_fim
def serialize_fim(prefix, suffix):
    """Show a generic prefix-suffix-middle layout with teaching markers."""
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise TypeError("prefix and suffix must be strings")
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
# end-snippet:serialize_fim
