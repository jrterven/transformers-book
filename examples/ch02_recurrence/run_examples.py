"""Run the chapter arithmetic; optionally train the small reversal model.

From the repository root:
    python -m examples.ch02_recurrence.run_examples
    python -m examples.ch02_recurrence.run_examples --train --steps 750
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch

from .data import EOS, PAD, collate, make_splits, words
from .model import (RecurrentSeq2Seq, gated_cell_update, greedy_decode,
                    rnn_trace, token_loss)


def arithmetic_demo():
    trace = rnn_trace([0.2, 0.4, -0.1])
    reverse_trace = rnn_trace([-0.1, 0.4, 0.2])
    hidden, cell = gated_cell_update(0.5, 0.8, 0.5, -0.4, 0.6)
    np.testing.assert_allclose(cell, 0.2, atol=1e-15)
    np.testing.assert_allclose(hidden, 0.6 * np.tanh(0.2), atol=1e-15)
    print("Scalar recurrent states:", np.array2string(trace, precision=12))
    print("Reversed input states:   ", np.array2string(reverse_trace, precision=12))
    print(f"LSTM cell={cell:.12f}; hidden={hidden:.12f}")

    torch.manual_seed(2026)
    torch.set_num_threads(1)
    model = RecurrentSeq2Seq()
    batch = collate([(3, 4, 5), (8, 3)])
    logits = model(batch.source, batch.lengths, batch.decoder_input)
    loss = token_loss(logits, batch.target)
    loss.backward()
    assert logits.shape == (2, 4, 9)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    model.eval()
    generated = greedy_decode(model, batch.source, batch.lengths, 5)
    print("Teacher-forced logits shape:", tuple(logits.shape))
    print("Encoder state shapes:", [tuple(s.shape) for s in
                                      model.encode(batch.source, batch.lengths)])
    print("Parameter tensor entries:", sum(p.numel() for p in model.parameters()))
    print("Fixed PAD embedding entries:", 2 * model.source_embedding.embedding_dim)
    print(f"Untrained demonstration loss: {loss.item():.12f}")
    print("Untrained greedy IDs:", generated.tolist())
    print("Forward/backward checks passed; these are not trained predictions.")


def _tokens_until_eos(row):
    result = []
    for token in row:
        result.append(int(token))
        if int(token) == EOS:
            break
    return result


@torch.no_grad()
def evaluate(model, sequences, generation_cap=7):
    """One full split: token-weighted NLL and complete greedy exact match."""
    model.eval()
    batch = collate(sequences)
    logits = model(batch.source, batch.lengths, batch.decoder_input)
    valid = batch.target != PAD
    nll = token_loss(logits, batch.target).item()
    correct = ((logits.argmax(-1) == batch.target) & valid).sum().item()
    generated = greedy_decode(model, batch.source, batch.lengths, generation_cap)
    matches = []
    ended = []
    predictions = []
    for source, row in zip(sequences, generated.tolist()):
        prediction = _tokens_until_eos(row)
        matches.append(prediction == list(reversed(source)) + [EOS])
        ended.append(EOS in prediction)
        predictions.append(prediction)
    by_length = {}
    for length in sorted(set(map(len, sequences))):
        selected = [j for j, row in enumerate(sequences) if len(row) == length]
        target = batch.target[selected]
        length_nll = token_loss(logits[selected], target).item()
        by_length[str(length)] = {
            "sequences": len(selected),
            "token_nll": length_nll,
            "exact_matches": sum(matches[j] for j in selected),
            "ended_in_eos": sum(ended[j] for j in selected),
        }
    return {"token_nll": nll, "nonpadding_tokens": valid.sum().item(),
            "teacher_forced_correct_tokens": correct,
            "exact_matches": sum(matches), "sequences": len(sequences),
            "ended_in_eos": sum(ended), "by_length": by_length,
            "predictions": predictions}


def run_training(steps=750, seed=2026, output=Path("output/ch02")):
    """Train from scratch; choose a checkpoint using validation NLL only."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train, validation, test = make_splits(seed)
    model = RecurrentSeq2Seq()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = random.Random(seed)
    history = []
    best_nll = float("inf")
    best = None
    best_step = None
    for step in range(1, steps + 1):
        model.train()
        batch = collate(rng.sample(train, 32))
        # snippet:training_update
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.source, batch.lengths, batch.decoder_input)
        loss = token_loss(logits, batch.target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # end-snippet:training_update
        if step % 50 == 0 or step == steps:
            metrics = evaluate(model, validation)
            record = {"step": step, "train_batch_nll": loss.item(),
                      "validation_token_nll": metrics["token_nll"],
                      "validation_exact_matches": metrics["exact_matches"]}
            history.append(record)
            print(f"Step {step:4d}: validation NLL={metrics['token_nll']:.6f}; "
                  f"greedy exact={metrics['exact_matches']}/{len(validation)}")
            if metrics["token_nll"] < best_nll:
                best_nll, best_step = metrics["token_nll"], step
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"seed": seed, "updates": steps, "batch_size": 32,
                     "embedding_width": 16, "hidden_width": 64,
                     "layers": 1, "dropout": 0.0, "learning_rate": 0.003,
                     "gradient_clip_norm": 1.0, "validation_interval": 50,
                     "split_sizes": [192, 48, 48], "source_lengths": [3, 4, 5, 6],
                     "generation_cap": 7, "checkpoint_selection": "validation token NLL"}
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
    torch.save({"model": best, "configuration": configuration,
                "selected_step": best_step}, output / "best.pt")
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Selected update:", best_step)
    for name in ("validation", "test"):
        metric = results[name]
        print(f"{name}: NLL={metric['token_nll']:.12f}; "
              f"greedy exact={metric['exact_matches']}/{metric['sequences']}")
    for length, metric in results["test"]["by_length"].items():
        print(f"Test length {length}: NLL={metric['token_nll']:.12f}; "
              f"greedy exact={metric['exact_matches']}/{metric['sequences']}")
    for source, prediction in zip(test[:3], results["test"]["predictions"][:3]):
        print("Source:", words(source), "| generated:", words(prediction))
    print("Results saved in", output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("output/ch02"))
    args = parser.parse_args()
    arithmetic_demo()
    if args.train:
        run_training(args.steps, args.seed, args.output)


if __name__ == "__main__":
    main()
