"""Print objectives, check information paths, optionally memorize one example."""

import argparse
import json
from pathlib import Path

import torch

from .data import (
    BOS, EOS, IGNORE, VOCAB, bert_corrupt, causal_pairs, encode,
    span_corrupt, token_names,
)
from .model import TinyTokenModel, make_model, selected_loss


def prepared_examples():
    sentence = encode("the red car stopped")
    gpt_input, gpt_labels = causal_pairs(sentence)
    bert_input, bert_labels = bert_corrupt(
        sentence, [1], generator=torch.Generator().manual_seed(7)
    )
    t5_source, t5_input, t5_labels = span_corrupt(
        encode("the red car stopped near the old bridge today"), [(1, 3), (6, 8)]
    )
    return {
        "gpt": ((gpt_input[None],), gpt_labels[None]),
        "bert": ((bert_input[None],), bert_labels[None]),
        "t5": ((t5_source[None], t5_input[None]), t5_labels[None]),
    }


# snippet:visibility_probe
@torch.no_grad()
def visibility_probe():
    original = torch.tensor([encode("the red car stopped")])
    changed = torch.tensor([encode("the red car moved")])
    torch.manual_seed(71)
    causal = TinyTokenModel(causal=True).double().eval()
    bidirectional = TinyTokenModel(causal=False).double().eval()
    bidirectional.load_state_dict(causal.state_dict())
    differences = {}
    for name, model in (("causal", causal), ("bidirectional", bidirectional)):
        before = model(original)[0, 1]   # vocabulary logits at "red"'s position
        after = model(changed)[0, 1]     # only a later token changed
        differences[name] = (after - before).abs().max().item()
    return differences
# end-snippet:visibility_probe


# snippet:objective_step
def objective_step(model, optimizer, inputs, labels):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(*inputs)
    loss = selected_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
# end-snippet:objective_step


def overfit_example(family, steps=200):
    """A wiring check on one fixed example, with no held-out generalization claim."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(72)
    model = make_model(family)
    inputs, labels = prepared_examples()[family]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.eval()
    with torch.no_grad():
        initial = selected_loss(model(*inputs), labels).item()
    for _ in range(steps):
        objective_step(model, optimizer, inputs, labels)
    model.eval()
    with torch.no_grad():
        logits = model(*inputs)
        final = selected_loss(logits, labels).item()
        predictions = logits.argmax(dim=-1)
    selected = labels != IGNORE
    result = {"initial_nll": initial, "final_nll": final,
              "correct_selected": int((predictions[selected] == labels[selected]).sum()),
              "selected_labels": int(selected.sum()),
              "selected_predictions": token_names(predictions[selected]),
              "selected_targets": token_names(labels[selected]), "steps": steps}
    if family == "t5":
        # Greedy free generation checks more than teacher-forced token choices.
        prefix, generated = torch.tensor([[BOS]]), []
        with torch.no_grad():
            for _ in range(labels.shape[1]):
                token = model(inputs[0], prefix)[:, -1].argmax(dim=-1)
                generated.append(int(token.item()))
                prefix = torch.cat((prefix, token[:, None]), dim=1)
                if token.item() == EOS:
                    break
        result["generated"] = token_names(generated)
        result["exact_generated"] = generated == labels[0].tolist()
    return result


def run(train=False, steps=200):
    torch.set_num_threads(1)
    result = {"vocabulary": list(VOCAB), "objectives": {},
              "visibility_max_abs_logit_difference": visibility_probe()}
    for family, (inputs, labels) in prepared_examples().items():
        result["objectives"][family] = {
            "inputs": [token_names(row[0]) for row in inputs],
            "labels": token_names(labels[0]),
            "selected_labels": int((labels != IGNORE).sum()),
        }
    if train:
        result["fixed_example_checks"] = {
            family: overfit_example(family, steps) for family in ("gpt", "bert", "t5")
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    result = run(args.train, args.steps)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
