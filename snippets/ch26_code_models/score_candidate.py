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
