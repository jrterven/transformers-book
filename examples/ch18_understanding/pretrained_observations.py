"""Measure sentence and word decisions with three pinned, released encoders."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from ..ch16_adaptation.real_data import examples
from .model import bio_spans, first_piece_labels, first_piece_predictions, span_metrics


SEED = 18
CHECKPOINTS = {
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    "nli": ("cross-encoder/nli-distilroberta-base", "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd"),
    "ner": ("dslim/distilbert-NER", "dfa2838a127384aabb82ed7719e16dab84c42a2a"),
}
BANK_SENTENCES = (
    "We walked to the bank by the river.",
    "We walked to the bank beside the stream.",
    "We walked to the bank to deposit cash.",
    "We walked to the bank to open an account.",
)
NLI_PAIRS = (
    ("Drop destination", "Ari moved to New Harbor.", "Ari moved.", "entailment"),
    ("Reverse the pair", "Ari moved.", "Ari moved to New Harbor.", "neutral"),
    ("Negate the move", "Ari moved to New Harbor.", "Ari did not move.", "contradiction"),
    ("Add a date", "Ari moved to New Harbor.", "Ari moved last Tuesday.", "neutral"),
    ("Swap the callers", "Ari called Sam.", "Sam called Ari.", "neutral"),
    ("Paraphrase the action", "The clerk updated the address.", "The address was changed.", "entailment"),
)
NER_CASES = (
    ("Ari moved to New Harbor .", ((0, 1, "PER"), (3, 5, "LOC"))),
    ("Maya Bhattacharyya moved to New Harbor .", ((0, 2, "PER"), (4, 6, "LOC"))),
    ("Jordan Lee joined Acme Research in San Francisco .", ((0, 2, "PER"), (3, 5, "ORG"), (6, 8, "LOC"))),
    ("New Harbor Transit hired Riya Banerjee .", ((0, 3, "ORG"), (4, 6, "PER"))),
    ("Sam moved from Lake City to North Harbor .", ((0, 1, "PER"), (3, 5, "LOC"), (6, 8, "LOC"))),
    ("Morgan met Casey in Oaxaca .", ((0, 1, "PER"), (2, 3, "PER"), (4, 5, "LOC"))),
    ("Núria Vidal called from São Paulo .", ((0, 2, "PER"), (4, 6, "LOC"))),
    ("Ximena Hernández joined Banco del Norte .", ((0, 2, "PER"), (3, 6, "ORG"))),
    ("Sam lives near New York City .", ((0, 1, "PER"), (3, 6, "LOC"))),
    ("The meeting starts at noon .", ()),
    ("Jordan signed the report .", ((0, 1, "PER"),)),
    ("The shipment crossed Jordan .", ((3, 4, "LOC"),)),
    ("Ari moved to Bellwether Quay .", ((0, 1, "PER"), (3, 5, "LOC"))),
    ("The New Harbor office opened .", ((1, 3, "LOC"),)),
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(key, model_class, cache_dir, offline):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    name, revision = CHECKPOINTS[key]
    common = dict(revision=revision, cache_dir=str(cache_dir), local_files_only=offline, token=False)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=False, **common)
    model, loading = model_class.from_pretrained(name, use_safetensors=True, trust_remote_code=False,
                                                 attn_implementation="eager", output_loading_info=True, **common)
    if any(loading.get(field) for field in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"incomplete load for {name}: {loading}")
    unused = loading.get("unexpected_keys", [])
    if any(key != "bert" or not parameter.startswith("cls.") for parameter in unused):
        raise RuntimeError(f"unexpected parameters for {name}: {unused}")
    model.eval().float().requires_grad_(False)
    weight_file = hf_hub_download(name, "model.safetensors", **common)
    source = inspect.getsourcefile(type(model))
    return model, tokenizer, {
        "model": name, "revision": revision, "weights_sha256": sha256(weight_file),
        "runtime_source_sha256": sha256(source), "unused_checkpoint_parameters": sorted(unused),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


@torch.inference_mode()
def contextual_geometry(model, tokenizer):
    encoded = tokenizer(list(BANK_SENTENCES), padding=True, return_tensors="pt")
    bank_id = tokenizer.convert_tokens_to_ids("bank")
    positions = (encoded.input_ids == bank_id).nonzero()
    if len(positions) != len(BANK_SENTENCES) or len(set(positions[:, 1].tolist())) != 1:
        raise ValueError("the shared word must appear once in the same position")
    states = model(**encoded, output_hidden_states=True).hidden_states
    vectors = torch.stack([state[positions[:, 0], positions[:, 1]] for state in states]).double()
    if not torch.equal(vectors[0], vectors[0, :1].expand_as(vectors[0])):
        raise ValueError("the embedding comparison must start from identical rows")
    lengths = vectors.norm(dim=-1)
    directions = vectors / lengths[..., None]
    cosines = (directions @ directions.transpose(-1, -2)).clamp(-1, 1)
    return {"sentences": list(BANK_SENTENCES), "token_id": bank_id, "position": int(positions[0, 1]),
            "cosines": cosines.tolist(), "norms": lengths.tolist(),
            "same_embedding_rows": True, "width": int(vectors.shape[-1])}


def encoder_features(model, tokenizer, requests):
    cls_rows, mean_rows = [], []
    with torch.no_grad():
        for start in range(0, len(requests), 8):
            encoded = tokenizer(requests[start:start + 8], padding=True, return_tensors="pt",
                                return_special_tokens_mask=True)
            content = encoded.attention_mask.bool() & ~encoded.pop("special_tokens_mask").bool()
            states = model(**encoded).last_hidden_state
            cls_rows.append(states[:, 0])
            mean_rows.append((states * content[..., None]).sum(dim=1) / content.sum(dim=1, keepdim=True))
    return {"cls": torch.cat(cls_rows), "mean": torch.cat(mean_rows)}


def fit_head(train, truth, held, held_truth):
    center = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    x = (train - center) / scale
    head = nn.Linear(x.shape[-1], 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.001)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(x), truth)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_logits = head(x)
        logits = head((held - center) / scale)
        record = {"parameters": sum(p.numel() for p in head.parameters()),
                  "train_cross_entropy": float(nn.functional.cross_entropy(train_logits, truth)),
                  "train_correct": int((train_logits.argmax(-1) == truth).sum()),
                  "held_correct": int((logits.argmax(-1) == held_truth).sum()),
                  "held_logits": logits.tolist(), "held_update_p": logits.softmax(-1)[:, 1].tolist()}
    return (head, center, scale), record


@torch.no_grad()
def padding_probe(model, tokenizer, fitted, text):
    head, center, scale = fitted
    plain = tokenizer(text, return_tensors="pt", return_special_tokens_mask=True)
    long = tokenizer(text, padding="max_length", max_length=plain.input_ids.shape[1] + 24,
                     return_tensors="pt", return_special_tokens_mask=True)
    valid = plain.attention_mask.bool() & ~plain.pop("special_tokens_mask").bool()
    content = long.attention_mask.bool() & ~long.pop("special_tokens_mask").bool()
    pad = ~long.attention_mask.bool()
    plain_states = model(**plain).last_hidden_state
    long_states = model(**long).last_hidden_state
    unmasked = model(**{**long, "attention_mask": torch.ones_like(long.attention_mask)}).last_hidden_state
    paths = {
        "short_correct": (plain_states, valid),
        "padded_correct": (long_states, content),
        "pool_pad_rows": (long_states, content | pad),
        "attend_pad_keys": (unmasked, content),
    }
    results = {}
    for name, (states, mask) in paths.items():
        pooled = states[mask].mean(dim=0)
        scores = head((pooled - center) / scale)
        results[name] = {"pooled_rows": int(mask.sum()), "logits": scores.tolist(),
                         "update_margin": float(scores[1] - scores[0])}
    return {"text": text, "input_pieces": tokenizer.convert_ids_to_tokens(plain.input_ids[0].tolist()),
            "extra_padding": int(pad.sum()), "first_pad_l2_norm": float(long_states[pad][0].norm()),
            "valid_state_max_difference": float((plain_states[valid] - long_states[content]).abs().max()),
            "paths": results}


def intent_experiment(model, tokenizer):
    train, held = examples("train"), examples("held")
    for key in ("family", "account", "request"):
        if {x[key] for x in train} & {x[key] for x in held}:
            raise ValueError(f"train and held sets overlap in {key}")
    train_vectors = encoder_features(model, tokenizer, [row["request"] for row in train])
    held_vectors = encoder_features(model, tokenizer, [row["request"] for row in held])
    train_y = torch.tensor([row["label"] for row in train])
    held_y = torch.tensor([row["label"] for row in held])
    heads, runs = {}, {}
    for pool in ("cls", "mean"):
        heads[pool], runs[pool] = fit_head(train_vectors[pool], train_y, held_vectors[pool], held_y)
    return {"train_count": len(train), "held": held, "probes": runs,
            "optimizer": {"updates": 100, "learning_rate": 0.005, "weight_decay": 0.001},
            "padding": padding_probe(model, tokenizer, heads["mean"], held[16]["request"])}


@torch.inference_mode()
def nli_experiment(model, tokenizer):
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    rows = []
    for name, premise, hypothesis, gold in NLI_PAIRS:
        inputs = tokenizer(premise, hypothesis, return_tensors="pt")
        logits = model(**inputs).logits[0].double()
        ids = inputs.input_ids[0].tolist()
        rows.append({"case": name, "premise": premise, "hypothesis": hypothesis, "gold": gold,
                     "input_ids": ids, "pieces": tokenizer.convert_ids_to_tokens(ids),
                     "sequence_ids": inputs.sequence_ids(), "logits": logits.tolist(),
                     "probabilities": logits.softmax(0).tolist(), "prediction": labels[int(logits.argmax())]})
    return {"labels": labels, "rows": rows}


def constrained_iob2(logits, labels):
    """Find the highest sum of word logits among legal IOB2 paths."""
    values = np.asarray(logits, dtype=np.float64)
    allowed = np.ones((len(labels), len(labels)), dtype=bool)
    may_start = np.ones(len(labels), dtype=bool)
    for following, tag in enumerate(labels):
        if tag.startswith("I-"):
            may_start[following] = False
            allowed[:, following] = [previous in (tag, "B-" + tag[2:]) for previous in labels]
    best = np.where(may_start, values[0], -np.inf)
    history = []
    for row in values[1:]:
        candidates = np.where(allowed, best[:, None], -np.inf)
        previous = candidates.argmax(axis=0)
        history.append(previous)
        best = row + candidates[previous, np.arange(len(labels))]
    path = [int(best.argmax())]
    for previous in reversed(history):
        path.append(int(previous[path[-1]]))
    return [labels[i] for i in reversed(path)]


def summarize_ner(rows, constrained=False):
    key = "constrained_tags" if constrained else "first_tags"
    metrics = [span_metrics(row["gold_tags"], row[key], repair_predictions=not constrained) for row in rows]
    m = sum(v["correct_spans"] for v in metrics)
    p = sum(v["predicted_spans"] for v in metrics)
    g = sum(v["gold_spans"] for v in metrics)
    words = sum(len(row["words"]) for row in rows)
    word_matches = sum(a == b for row in rows for a, b in zip(row[key], row["gold_tags"], strict=True))
    by_type = {}
    for entity in ("PER", "LOC", "ORG", "MISC"):
        matches, gold, predicted = 0, 0, 0
        for row in rows:
            truth = {s for s in bio_spans(row["gold_tags"]) if s[2] == entity}
            found = {s for s in bio_spans(row[key], repair=not constrained) if s[2] == entity}
            matches += len(truth & found)
            gold += len(truth)
            predicted += len(found)
        by_type[entity] = {"matches": matches, "gold": gold, "predicted": predicted}
    return {"matches": m, "predicted": p, "gold": g, "precision": m / p if p else None,
            "recall": m / g if g else None, "f1": 2 * m / (p + g) if p + g else None,
            "word_matches": word_matches, "words": words, "word_accuracy": word_matches / words,
            "by_type": by_type}


@torch.inference_mode()
def ner_experiment(model, tokenizer):
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    rows = []
    for case, (sentence, spans) in enumerate(NER_CASES, 1):
        original = sentence.split()
        gold_tags = ["O"] * len(original)
        for start, end, kind in spans:
            gold_tags[start:end] = ["B-" + kind] + ["I-" + kind] * (end - start - 1)
        if bio_spans(gold_tags) != set(spans):
            raise ValueError("gold span fixture failed to round-trip")
        for casing, words in (("original", original), ("lower", [word.lower() for word in original])):
            encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
            ids = [-1 if i is None else i for i in encoded.word_ids()]
            targets = first_piece_labels(ids, [labels.index(tag) for tag in gold_tags])
            first = np.flatnonzero(targets >= 0)
            logits = model(**encoded).logits[0].double()
            first_tags = [labels[i] for i in first_piece_predictions(ids, logits.argmax(-1).numpy())]
            constrained = constrained_iob2(logits[first].numpy(), labels)
            prediction = sorted(bio_spans(first_tags, repair=True))
            rows.append({"case": case, "casing": casing, "words": words, "gold_tags": gold_tags,
                         "gold_spans": sorted(spans), "input_ids": encoded.input_ids[0].tolist(),
                         "pieces": tokenizer.convert_ids_to_tokens(encoded.input_ids[0].tolist()),
                         "word_ids": ids, "first_positions": first.tolist(), "targets": targets.tolist(),
                         "logits": logits.tolist(), "first_tags": first_tags,
                         "first_spans": prediction, "constrained_tags": constrained,
                         "constrained_spans": sorted(bio_spans(constrained))})
    summary = {}
    for casing in ("original", "lower"):
        subset = [row for row in rows if row["casing"] == casing]
        summary[casing] = {"first_piece": summarize_ner(subset),
                           "constrained": summarize_ner(subset, constrained=True),
                           "paths_changed": sum(row["first_tags"] != row["constrained_tags"] for row in subset)}
    return {"labels": labels, "rows": rows, "summary": summary}


def main():
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("output/model-cache"))
    parser.add_argument("--bert-cache-dir", type=Path, help="separate cache for the base BERT checkpoint")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    result = {"seed": SEED, "device": "cpu", "threads": 4, "dtype": "float32",
              "attention_backend": "eager", "platform": platform.platform(),
              "versions": {name: importlib.metadata.version(name) for name in (
                  "numpy", "torch", "transformers", "tokenizers", "safetensors")}, "checkpoints": {}}
    model, tokenizer, result["checkpoints"]["bert"] = load_model(
        "bert", AutoModel, args.bert_cache_dir or args.cache_dir, args.offline)
    print("Measuring BERT and fitting two frozen-feature probes", flush=True)
    result["context"] = contextual_geometry(model, tokenizer)
    result["intent"] = intent_experiment(model, tokenizer)
    del model, tokenizer
    model, tokenizer, result["checkpoints"]["nli"] = load_model(
        "nli", AutoModelForSequenceClassification, args.cache_dir, args.offline)
    result["nli"] = nli_experiment(model, tokenizer)
    del model, tokenizer
    model, tokenizer, result["checkpoints"]["ner"] = load_model(
        "ner", AutoModelForTokenClassification, args.cache_dir, args.offline)
    result["ner"] = ner_experiment(model, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved real BERT, inference, and entity observations to {args.output}")


if __name__ == "__main__":
    main()
