"""Export marked canonical code for the book's LaTeX listings."""

import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "examples/ch04_attention/attention.py",
    ROOT / "examples/ch04_attention/run_examples.py",
)
DESTINATION = ROOT / "snippets/ch04_attention"
EXPECTED = {
    "projections", "stable_softmax", "attention_core", "batch_heads",
    "causal_example", "backward",
}
MARKER = re.compile(r"^\s*# (snippet|end-snippet):([a-z][a-z0-9_]*)\s*$")


def collect_snippets():
    snippets = {}
    for source in SOURCES:
        active = None
        lines = []
        for number, line in enumerate(source.read_text().splitlines(keepends=True), 1):
            match = MARKER.match(line)
            if match is None:
                if active is not None:
                    lines.append(line)
                continue
            kind, name = match.groups()
            if kind == "snippet":
                if active is not None or name in snippets:
                    raise ValueError(f"nested or duplicate snippet at {source}:{number}")
                active, lines = name, []
            else:
                if active != name:
                    raise ValueError(f"unmatched end marker at {source}:{number}")
                snippets[name] = "".join(lines).rstrip() + "\n"
                active = None
        if active is not None:
            raise ValueError(f"unclosed snippet {active} in {source}")
    if set(snippets) != EXPECTED:
        raise ValueError(f"expected {sorted(EXPECTED)}, found {sorted(snippets)}")
    return snippets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without writing")
    args = parser.parse_args()
    snippets = collect_snippets()
    expected_paths = {DESTINATION / f"{name}.py" for name in snippets}
    existing_paths = set(DESTINATION.glob("*.py"))
    stale_paths = existing_paths - expected_paths
    if stale_paths:
        raise SystemExit(f"unexpected snippet files: {sorted(map(str, stale_paths))}")
    if args.check:
        mismatches = [path for path in sorted(expected_paths)
                      if not path.exists() or path.read_text() != snippets[path.stem]]
        if mismatches:
            raise SystemExit(f"snippets need export: {sorted(map(str, mismatches))}")
        print(f"Checked {len(snippets)} snippets: all current.")
        return
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, contents in sorted(snippets.items()):
        path = DESTINATION / f"{name}.py"
        # Avoid touching unchanged files so repeated exports are idempotent.
        if not path.exists() or path.read_text() != contents:
            path.write_text(contents)
    print(f"Exported {len(snippets)} snippets to {DESTINATION.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
