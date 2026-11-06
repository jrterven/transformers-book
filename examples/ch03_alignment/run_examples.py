"""Run additive-attention arithmetic and an optional color-reversal experiment."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

from examples.ch02_recurrence.data import EOS, PAD, collate, make_splits, words
from examples.ch02_recurrence.model import token_loss
from .model import AttentiveSeq2Seq, additive_scores, greedy_decode, weighted_read


# snippet:numerical_example
def numerical_example():
    annotations = np.array([[1., 0.], [0., 1.], [1., 1.]])
    given_scores = np.log([0.05, 0.10, 0.85])
    weights, context = weighted_read(given_scores, annotations)
    query = np.array([0.2, -0.1])
    scores = additive_scores(query, annotations, np.eye(2), np.eye(2),
                             np.array([1., -1.]))
    scorer_weights, scorer_context = weighted_read(scores, annotations)
    return weights, context, scores, scorer_weights, scorer_context
# end-snippet:numerical_example


def arithmetic_demo():
    weights, context, scores, calculated, calculated_context = numerical_example()
    np.testing.assert_allclose(weights, [.05, .10, .85], atol=1e-15)
    np.testing.assert_allclose(context, [.90, .95], atol=1e-15)
    print("Given-score attention:", np.array2string(weights, precision=12))
    print("Given-score context:", np.array2string(context, precision=12))
    print("Hand-parameter additive scores:", np.array2string(scores, precision=12))
    print("Hand-parameter probabilities:", np.array2string(calculated, precision=12))
    print("Hand-parameter context:", np.array2string(calculated_context, precision=12))

    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = AttentiveSeq2Seq()
    batch = collate([(3, 4, 5), (8, 3)])
    logits, attention = model(batch.source, batch.lengths, batch.decoder_input)
    token_loss(logits, batch.target).backward()
    assert logits.shape == (2, 4, 9)
    assert attention.shape == (2, 4, 3)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 4))
    assert torch.equal(attention[1, :, 2], torch.zeros(4))
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    print("Teacher-forced logits:", tuple(logits.shape))
    print("Attention [batch, target step, source position]:", tuple(attention.shape))
    print("Parameter tensor entries:", sum(p.numel() for p in model.parameters()))
    print("Fixed PAD embedding entries:", 2 * model.source_embedding.embedding_dim)
    print("Arithmetic, source masking, and forward/backward checks passed.")


def _until_eos(row):
    result = []
    for token in row:
        result.append(int(token))
        if int(token) == EOS:
            break
    return result


@torch.no_grad()
def evaluate(model, sequences, generation_cap=7):
    """Grade reference-prefix probabilities and independent greedy generation."""
    model.eval()
    batch = collate(sequences)
    logits, _ = model(batch.source, batch.lengths, batch.decoder_input)
    valid = batch.target != PAD
    correct = ((logits.argmax(-1) == batch.target) & valid).sum().item()
    generated, _ = greedy_decode(model, batch.source, batch.lengths, generation_cap)
    predictions = [_until_eos(row) for row in generated.tolist()]
    matches = [p == list(reversed(s)) + [EOS]
               for p, s in zip(predictions, sequences)]
    ended = [EOS in p for p in predictions]
    by_length = {}
    for length in sorted(set(map(len, sequences))):
        positions = [j for j, s in enumerate(sequences) if len(s) == length]
        by_length[str(length)] = {
            "sequences": len(positions),
            "token_nll": token_loss(logits[positions], batch.target[positions]).item(),
            "exact_matches": sum(matches[j] for j in positions),
            "ended_in_eos": sum(ended[j] for j in positions),
        }
    return {"token_nll": token_loss(logits, batch.target).item(),
            "nonpadding_tokens": valid.sum().item(),
            "teacher_forced_correct_tokens": correct,
            "sequences": len(sequences), "exact_matches": sum(matches),
            "ended_in_eos": sum(ended), "by_length": by_length,
            "predictions": predictions}


@torch.no_grad()
def alignment_example(model, test, predictions, generation_cap=7):
    """Pick a correct length-six example for illustration after evaluation.

    Among correct length-six examples, prefer more distinct source colors;
    ties use test order. This choice does not select the model checkpoint.
    If none is correct, display the first length-six example and its errors.
    """
    candidates = [i for i, (s, p) in enumerate(zip(test, predictions))
                  if len(s) == 6 and p == list(reversed(s)) + [EOS]]
    index = (max(candidates, key=lambda i: len(set(test[i]))) if candidates else
             next(i for i, s in enumerate(test) if len(s) == 6))
    selection = (
        "correct length-six test example with most distinct colors; ties in test order"
        if candidates else
        "first length-six test example; no correct length-six prediction"
    )
    source = test[index]
    batch = collate([source])
    logits, teacher_attention = model(batch.source, batch.lengths, batch.decoder_input)
    tokens, attention = greedy_decode(model, batch.source, batch.lengths, generation_cap)
    generated = _until_eos(tokens[0].tolist())
    ended_in_eos = EOS in generated
    mode = ("greedy generation; rows end at the EOS prediction" if ended_in_eos else
            "greedy generation; generation cap reached without EOS")
    return {"test_index": index, "source": list(source),
            "target": list(reversed(source)) + [EOS], "generated": generated,
            "attention": attention[0, :len(generated), :len(source)].tolist(),
            "teacher_forced_attention": teacher_attention[0, :, :len(source)].tolist(),
            "teacher_forced_choices": logits.argmax(-1)[0].tolist(),
            "selection": selection, "mode": mode, "ended_in_eos": ended_in_eos}


def run_training(steps=750, seed=2026, output=Path("output/ch03")):
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train, validation, test = make_splits(seed)
    model = AttentiveSeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    rng = random.Random(seed)
    best_nll, best_step, best = float("inf"), None, None
    history = []
    for step in range(1, steps + 1):
        model.train()
        batch = collate(rng.sample(train, 32))
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.source, batch.lengths, batch.decoder_input)
        loss = token_loss(logits, batch.target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 50 == 0 or step == steps:
            result = evaluate(model, validation)
            history.append({"step": step, "train_batch_nll": loss.item(),
                            "validation_token_nll": result["token_nll"],
                            "validation_exact_matches": result["exact_matches"]})
            print(f"Step {step:4d}: validation NLL={result['token_nll']:.6f}; "
                  f"greedy exact={result['exact_matches']}/{len(validation)}")
            if result["token_nll"] < best_nll:
                best_nll, best_step = result["token_nll"], step
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    configuration = {"seed": seed, "updates": steps, "batch_size": 32,
                     "embedding_width": 16, "hidden_width": 64, "alignment_width": 32,
                     "encoder_layers": 1, "encoder_directions": 1, "dropout": 0.,
                     "learning_rate": .003, "gradient_clip_norm": 1.,
                     "validation_interval": 50, "generation_cap": 7,
                     "split_sizes": [192, 48, 48], "source_lengths": [3, 4, 5, 6],
                     "checkpoint_selection": "minimum validation token NLL"}
    results = {"configuration": configuration,
               "environment": {"python": platform.python_version(),
                               "torch": torch.__version__, "numpy": np.__version__,
                               "device": "cpu", "threads": torch.get_num_threads()},
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "selected_step": best_step, "history": history,
               "split_sha256": {name: hashlib.sha256(json.dumps(split).encode()).hexdigest()
                                for name, split in zip(("train", "validation", "test"),
                                                       (train, validation, test))},
               "validation": evaluate(model, validation),
               "test": evaluate(model, test)}
    results["alignment_example"] = alignment_example(
        model, test, results["test"]["predictions"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best, "configuration": configuration,
                "selected_step": best_step}, output / "best.pt")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Selected update:", best_step)
    for name in ("validation", "test"):
        r = results[name]
        print(f"{name}: NLL={r['token_nll']:.12f}; "
              f"greedy exact={r['exact_matches']}/{r['sequences']}")
    trace = results["alignment_example"]
    print("Illustrated source:", words(trace["source"]))
    print("Illustrated generation:", words(trace["generated"]))
    print("Results saved in", output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("output/ch03"))
    args = parser.parse_args()
    arithmetic_demo()
    if args.train:
        run_training(args.steps, args.seed, args.output)


if __name__ == "__main__":
    main()
