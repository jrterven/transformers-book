"""Export marked canonical code for the book's LaTeX listings."""

import argparse
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = {
    "ch01": {
        "sources": ("examples/ch01_language/model.py",
                    "examples/ch01_language/run_examples.py"),
        "destination": "snippets/ch01_language",
        "expected": {"embedding_lookup", "context_model", "stable_softmax",
                     "shifted_pairs", "example"},
    },
    "ch02": {
        "sources": ("examples/ch02_recurrence/model.py",
                    "examples/ch02_recurrence/data.py"),
        "destination": "snippets/ch02_recurrence",
        "expected": {"rnn_step", "lstm_update", "encode_decode",
                     "teacher_forcing", "token_loss", "greedy_decode"},
    },
    "ch03": {
        "sources": ("examples/ch03_alignment/model.py",
                    "examples/ch03_alignment/run_examples.py"),
        "destination": "snippets/ch03_alignment",
        "expected": {"additive_scores", "masked_context", "encoder_annotations",
                     "decode_step", "teacher_forward", "greedy_decode",
                     "numerical_example"},
    },
    "ch04": {
        "sources": ("examples/ch04_attention/attention.py",
                    "examples/ch04_attention/run_examples.py"),
        "destination": "snippets/ch04_attention",
        "expected": {"projections", "one_query_example", "stable_softmax", "attention_core",
                     "batch_heads", "causal_example", "backward"},
    },
    "ch05": {
        "sources": ("examples/ch05_transformer/model.py",
                    "examples/ch05_transformer/run_examples.py"),
        "destination": "snippets/ch05_transformer",
        "expected": {"positional_encoding", "multihead", "feed_forward",
                     "encoder_layer", "decoder_layer", "mask_building",
                     "model_forward", "vocabulary_readout", "example"},
    },
    "ch06": {
        "sources": ("examples/ch06_training/data.py",
                    "examples/ch06_training/training.py",
                    "examples/ch06_training/decoding.py",
                    "examples/ch06_training/run_experiment.py"),
        "destination": "snippets/ch06_training",
        "expected": {"data_shift", "token_loss", "loss_sanity", "train_step", "evaluate",
                     "checkpoint", "cache_step", "greedy_decode", "experiment"},
    },
    "ch07": {
        "sources": ("examples/ch07_pretraining/data.py",
                    "examples/ch07_pretraining/model.py",
                    "examples/ch07_pretraining/run_examples.py"),
        "destination": "snippets/ch07_pretraining",
        "expected": {"causal_pairs", "bert_corruption", "span_corruption",
                     "token_stack", "selected_loss", "visibility_probe", "objective_step"},
    },
    "ch08": {
        "sources": ("examples/ch08_modern/model.py",
                    "examples/ch08_modern/run_examples.py"),
        "destination": "snippets/ch08_modern",
        "expected": {"rms_norm", "swiglu", "rope", "rotary_attention",
                     "modern_block", "modern_decoder", "example_checks"},
    },
    "ch09": {
        "sources": ("examples/ch09_scaling/model.py",),
        "destination": "snippets/ch09_scaling",
        "expected": {"compute_budget", "allocation_sweep", "top1_routing",
                     "expert_branch"},
    },
    "ch10": {
        "sources": ("examples/ch10_kv/model.py", "examples/ch10_kv/latent.py"),
        "destination": "snippets/ch10_kv",
        "expected": {"causal_mask", "grouped_attention", "attention_step",
                     "cache_bytes", "latent_attention"},
    },
}
MARKER = re.compile(r"^\s*# (snippet|end-snippet):([a-z][a-z0-9_]*)\s*$")


def collect_snippets(chapter="ch04"):
    config = CHAPTERS[chapter]
    snippets = {}
    for relative_source in config["sources"]:
        source = ROOT / relative_source
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
                snippets[name] = textwrap.dedent("".join(lines)).rstrip() + "\n"
                active = None
        if active is not None:
            raise ValueError(f"unclosed snippet {active} in {source}")
    if set(snippets) != config["expected"]:
        raise ValueError(f"expected {sorted(config['expected'])}, found {sorted(snippets)}")
    return snippets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without writing")
    parser.add_argument("--chapter", choices=(*CHAPTERS, "all"), default="all")
    args = parser.parse_args()
    chapters = tuple(CHAPTERS) if args.chapter == "all" else (args.chapter,)
    count = 0
    for chapter in chapters:
        snippets = collect_snippets(chapter)
        destination = ROOT / CHAPTERS[chapter]["destination"]
        expected_paths = {destination / f"{name}.py" for name in snippets}
        stale_paths = set(destination.glob("*.py")) - expected_paths
        if stale_paths:
            raise SystemExit(f"unexpected snippet files: {sorted(map(str, stale_paths))}")
        if args.check:
            mismatches = [path for path in sorted(expected_paths)
                          if not path.exists() or path.read_text() != snippets[path.stem]]
            if mismatches:
                raise SystemExit(f"snippets need export: {sorted(map(str, mismatches))}")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            for name, contents in sorted(snippets.items()):
                path = destination / f"{name}.py"
                # Preserve the bytes and mtimes of unchanged excerpts.
                if not path.exists() or path.read_text() != contents:
                    path.write_text(contents)
        count += len(snippets)
    verb = "Checked" if args.check else "Exported"
    print(f"{verb} {count} snippets ({', '.join(chapters)}): all current.")


if __name__ == "__main__":
    main()
