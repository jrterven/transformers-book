"""Train a small reversal model and report held-out generation and cache checks.

Run from the repository root:
    python -m examples.ch06_training.run_experiment --output output/ch06
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import torch

from examples.ch05_transformer.model import Transformer
from .data import VOCAB, collate, make_splits, words
from .decoding import CachedDecoder, benchmark, decode_prefix, encode_source, greedy_decode
from .training import evaluate, load_checkpoint, save_checkpoint, token_loss, train_step


# snippet:loss_sanity
def loss_sanity_example():
    """Uniform next-token probabilities in a two-example padded batch."""
    batch = collate([(3, 4, 5), (3, 5)])
    logits = torch.zeros(*batch.target.shape, len(VOCAB), dtype=torch.float64)
    total, count = token_loss(logits, batch.target)
    nll = total / count
    return {"loss_sum": total.item(), "tokens": count.item(),
            "nll": nll.item(), "perplexity": torch.exp(nll).item()}
# end-snippet:loss_sanity


def split_fingerprint(split):
    payload = json.dumps(split, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run(output, steps=2000, seed=2026, data_seed=2026, batch_size=64,
        validation_every=100):
    if steps < 1 or batch_size < 1 or validation_every < 1:
        raise ValueError("steps, batch_size and validation_every must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train, validation, test = make_splits(seed=data_seed)
    if batch_size > len(train):
        raise ValueError("batch_size cannot exceed the training split size")
    model_config = dict(source_vocab_size=len(VOCAB), target_vocab_size=len(VOCAB),
                        d_model=64, n_heads=4, d_ff=128, n_encoder_layers=2,
                        n_decoder_layers=2, dropout=0.0, pad_id=0, max_length=64)
    model = Transformer(**model_config)
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.xavier_uniform_(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    experiment = {"seed": seed, "data_seed": data_seed, "steps": steps,
                  "batch_size": batch_size, "learning_rate": 0.001,
                  "optimizer": "Adam", "betas": [0.9, 0.999],
                  "gradient_clip_norm": 1.0, "validation_every": validation_every,
                  "matrix_initialization": "xavier_uniform",
                  "split_sizes": [len(train), len(validation), len(test)],
                  "min_length": 3, "max_length": 6,
                  "split_sha256": {name: split_fingerprint(split)
                                   for name, split in zip(("train", "validation", "test"),
                                                          (train, validation, test))}}
    history = []
    initial = evaluate(model, validation)
    history.append({"step": 0, "validation": initial})
    best_nll = initial["nll"]
    checkpoint_path = output / "best.pt"
    save_checkpoint(checkpoint_path, model, model_config, experiment, 0, initial)
    started = perf_counter()
    # snippet:experiment
    for step in range(1, steps + 1):
        batch = collate(rng.sample(train, batch_size))
        training_nll = train_step(model, optimizer, batch)
        if step % validation_every == 0 or step == steps:
            metrics = evaluate(model, validation)
            history.append({"step": step, "training_batch_nll": training_nll,
                            "validation": metrics})
            print(f"step={step:4d} validation_nll={metrics['nll']:.5f} "
                  f"exact_match={metrics['exact_match']:.3f}", flush=True)
            if metrics["nll"] < best_nll:
                best_nll = metrics["nll"]
                save_checkpoint(checkpoint_path, model, model_config,
                                experiment, step, metrics)
    model, checkpoint = load_checkpoint(checkpoint_path)
    test_metrics = evaluate(model, test)
    # end-snippet:experiment
    by_length = {str(length): evaluate(model, [s for s in test if len(s) == length])
                 for length in range(3, 7)}
    elapsed = perf_counter() - started
    example = collate(test[:8])
    cached = greedy_decode(model, example.source, cached=True)
    full = greedy_decode(model, example.source, cached=False)
    if not torch.equal(cached, full):
        raise AssertionError("cached and full-prefix generated IDs differ")
    with torch.no_grad():
        memory, allowed = encode_source(model, example.source)
        cache = CachedDecoder(model, memory, allowed)
        max_error = 0.0
        for t in range(example.decoder_input.shape[1]):
            # Repeated EOS is harmless in a finished row; PAD is not a cache input.
            token = example.decoder_input[:, t:t + 1].clone()
            prefix = example.decoder_input[:, :t + 1].clone()
            token[token == 0] = 2
            prefix[prefix == 0] = 2
            got = cache.step(token)
            reference = decode_prefix(model, prefix, memory, allowed)[:, -1]
            max_error = max(max_error, (got - reference).abs().max().item())
    timing_batch = collate([test[0]])
    timing = benchmark(model, timing_batch.source,
                       timing_batch.decoder_input[:, :5], repeats=25)
    examples = [{"source": words(s), "expected": words(t, include_special=True),
                 "generated": words(g, include_special=True)}
                for s, t, g in zip(example.source, example.target, cached)]
    result = {"experiment": experiment, "model_config": model_config,
              "parameters": sum(p.numel() for p in model.parameters()),
              "selected_step": checkpoint["step"],
              "validation": checkpoint["validation"], "test": test_metrics,
              "test_by_length": by_length,
              "cache_max_abs_error_float32": max_error,
              "cached_generation_matches_full": True,
              "timing": timing, "examples": examples, "history": history,
              "elapsed_seconds": elapsed,
              "environment": {"python": platform.python_version(),
                              "torch": str(torch.__version__),
                              "platform": platform.platform(), "device": "cpu",
                              "threads": torch.get_num_threads()}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("parameters", "selected_step", "test", "test_by_length",
                       "cache_max_abs_error_float32", "timing", "examples")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/ch06")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=100)
    args = parser.parse_args()
    print("Uniform nine-class loss sanity check:", loss_sanity_example())
    run(**vars(args))


if __name__ == "__main__":
    main()
