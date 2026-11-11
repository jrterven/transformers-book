"""Record English-to-Spanish translations and English summaries on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import sacrebleu
from sacrebleu.metrics import BLEU
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from transformers.utils.hub import cached_file

from .model import rouge_l


CHECKPOINTS = {
    "marian": {
        "name": "Helsinki-NLP/opus-mt-en-es",
        "revision": "5bc4493d463cf000c1f0b50f8d56886a392ed4ab",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "befce0f620e8e63f83bbb7f0818eb51d541cba28057a7cc14b00466d85c4994c",
    },
    "bart": {
        "name": "facebook/bart-large-cnn",
        "revision": "37f520fa929c961707657b28798b30c003dd100b",
        "weight_file": "model.safetensors",
        "weight_sha256": "40041830399afb5348525ef8354b007ecec4286fdf3524f7e6b54377e17096cb",
    },
}

TRANSLATION_PAIRS = (
    ("blue_bus", "The blue bus arrived late.", "El autobús azul llegó tarde."),
    ("platform", "The blue bus arrived late and stopped at platform three.",
     "El autobús azul llegó tarde y paró en el andén tres."),
    ("departed", "The ferry departed at 7:10.", "El ferry salió a las 7:10."),
    ("negated", "The ferry did not depart at 7:10.", "El ferry no salió a las 7:10."),
    ("counts", "Only two of the five ferries left at noon.",
     "Solo dos de los cinco ferris salieron al mediodía."),
    ("river", "She sat on the bank and watched the river.",
     "Se sentó en la orilla y miró el río."),
    ("loan", "She walked into the bank and asked for a loan.",
     "Entró en el banco y pidió un préstamo."),
    ("order", "The last bus left at six after the gate closed.",
     "El último autobús salió a las seis después de que cerrara la puerta."),
)
TRANSLATION_SETTINGS = {
    "greedy": {"beams": 1, "length_penalty": 0.0, "max_new_tokens": 48},
    "beam_raw": {"beams": 4, "length_penalty": 0.0, "max_new_tokens": 48},
    "beam_mean": {"beams": 4, "length_penalty": 1.0, "max_new_tokens": 48},
}

REPORT = (
    "On Tuesday, the Lakeside ferry was delayed for 20 minutes.",
    "A crew shut the north gate for a routine ramp inspection.",
    "Staff gave waiting passengers an update while the crew finished the inspection.",
    "Riders at the terminal waited under the covered walkway.",
    "After a check with the harbor office, the operator said thick fog caused the delay.",
    "Passengers boarded through the south gate at 8:40 a.m.",
    "No sailings were canceled.",
)
SUMMARY_REFERENCE = (
    "Thick fog delayed the Lakeside ferry for 20 minutes. "
    "Passengers boarded at the south gate at 8:40 a.m., and no sailings were canceled."
)
SUMMARY_SETTINGS = {"beams": 4, "length_penalty": 1.0}
SUMMARY_CAPS = (18, 72)
THREADS = 4


def load_checkpoint(key, *, cache_dir, offline):
    spec = CHECKPOINTS[key]
    name = spec["name"]
    common = dict(revision=spec["revision"], cache_dir=cache_dir,
                  local_files_only=offline, token=False)
    weights = cached_file(name, spec["weight_file"], **common)
    with Path(weights).open("rb") as handle:
        actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_hash != spec["weight_sha256"]:
        raise ValueError(f"Unexpected weight checksum for {name}: {actual_hash}")
    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=key == "bart", trust_remote_code=False, **common)
    model, loading = AutoModelForSeq2SeqLM.from_pretrained(
        name, use_safetensors=spec["weight_file"].endswith(".safetensors"),
        weights_only=True, trust_remote_code=False, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True, **common)
    if any(loading.get(field) for field in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete model load for {name}: {loading}")
    model.eval()
    metadata = {
        **spec,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer": type(tokenizer).__name__,
        "start_id": model.config.decoder_start_token_id,
        "bos_id": model.config.bos_token_id,
        "pad_id": model.config.pad_token_id,
        "eos_id": model.config.eos_token_id,
        "loading": loading,
    }
    return model, tokenizer, metadata


def source_tokens(tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    ids = inputs["input_ids"][0].tolist()
    return inputs, {"text": text, "ids": ids, "pieces": tokenizer.convert_ids_to_tokens(ids)}


def score_path(model, tokenizer, inputs, ids, *, attention=False, top_candidates=False):
    """Recompute each prediction from the supplied path and the same source."""
    start = model.config.decoder_start_token_id
    decoder_inputs = torch.tensor([[start, *ids[:-1]]], dtype=torch.long)
    output = model(**inputs, decoder_input_ids=decoder_inputs, use_cache=False,
                   output_attentions=attention)
    log_p = output.logits[0].log_softmax(dim=-1)
    chosen = log_p.gather(1, torch.tensor(ids, dtype=torch.long)[:, None]).squeeze(1)
    trace = {
        "ids": ids,
        "pieces": tokenizer.convert_ids_to_tokens(ids),
        "text": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        "token_log_p": chosen.tolist(),
        "sum_log_p": float(chosen.double().sum()),
    }
    if top_candidates:
        values, tokens = log_p.topk(3, dim=-1)
        trace["top3"] = {"ids": tokens.tolist(), "log_p": values.tolist()}
    if attention:
        weights = output.cross_attentions[-1][0].mean(dim=0)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(weights.shape[0]))
        trace["cross_attention"] = weights.tolist()
    return trace


def generate_path(model, tokenizer, inputs, *, beams, length_penalty, max_new_tokens,
                  attention=False, top_candidates=False):
    config = GenerationConfig(
        num_beams=beams, num_return_sequences=1, do_sample=False,
        early_stopping="never" if beams > 1 else False,
        max_new_tokens=max_new_tokens, use_cache=True,
        decoder_start_token_id=model.config.decoder_start_token_id,
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
    )
    if beams > 1:
        config.length_penalty = length_penalty
    result = model.generate(**inputs, generation_config=config, use_model_defaults=False,
                            return_dict_in_generate=True, output_scores=True)
    start, *ids = result.sequences[0].tolist()
    if start != config.decoder_start_token_id or not ids:
        raise ValueError("Generation must return the decoder start followed by a nonempty path")
    trace = score_path(model, tokenizer, inputs, ids, attention=attention,
                       top_candidates=top_candidates)
    if beams == 1:
        # Greedy stores logits. Normalize each vocabulary row before selecting its token.
        transitions = torch.stack([scores[0].log_softmax(dim=-1)[token]
                                   for scores, token in zip(result.scores, ids, strict=True)])
    else:
        # Beam search stores log-probabilities. Gather along the selected beam ancestry.
        transitions = model.compute_transition_scores(
            result.sequences, result.scores, result.beam_indices, normalize_logits=False)[0]
    replay_error = (transitions - torch.tensor(trace["token_log_p"])).abs().max().item()
    if replay_error > 2e-3:
        raise ValueError(f"Cached generation and full-path replay differ by {replay_error}")
    trace["full_path_replay_max_abs_log_p_difference"] = replay_error
    trace["token_log_p"] = transitions.tolist()
    trace["sum_log_p"] = float(transitions.double().sum())
    normalized = trace["sum_log_p"] / len(ids)**length_penalty
    native_score = None
    if beams > 1:
        native_score = float(result.sequences_scores[0])
        if abs(native_score - normalized) > 2e-4:
            raise ValueError(f"Native beam score {native_score} differs from {normalized}")
    trace.update(
        length=len(ids),
        content_tokens=sum(token not in tokenizer.all_special_ids for token in ids),
        ended=ids[-1] == config.eos_token_id,
        rank_score=normalized,
        native_beam_score=native_score,
    )
    return trace


@torch.inference_mode()
def translations(model, tokenizer):
    cases = []
    for key, source, reference in TRANSLATION_PAIRS:
        inputs, encoded = source_tokens(tokenizer, source)
        outputs = {
            setting: generate_path(model, tokenizer, inputs, **config,
                                   attention=key == "blue_bus" and setting == "beam_raw",
                                   top_candidates=setting in ("greedy", "beam_raw"))
            for setting, config in TRANSLATION_SETTINGS.items()
        }
        row = {"id": key, "source": encoded, "reference": reference, "outputs": outputs}
        if key == "blue_bus":
            targets = [reference, "El autobús rojo llegó tarde."]
            row["supplied_targets"] = []
            for target in targets:
                ids = tokenizer(text_target=target)["input_ids"]
                trace = score_path(model, tokenizer, inputs, ids)
                row["supplied_targets"].append({"target": target, **trace})
        cases.append(row)

    # The corpus uses the same eight source/reference pairs in each condition.
    scores = {}
    for setting in TRANSLATION_SETTINGS:
        metric = BLEU(tokenize="13a", smooth_method="exp", effective_order=False)
        score = metric.corpus_score([row["outputs"][setting]["text"] for row in cases],
                                    [[row["reference"] for row in cases]])
        scores[setting] = {"bleu": score.score, "matches": score.counts,
                           "totals": score.totals, "candidate_length": score.sys_len,
                           "reference_length": score.ref_len, "brevity_penalty": score.bp,
                           "signature": str(metric.get_signature())}
    return {"settings": TRANSLATION_SETTINGS, "cases": cases, "corpus_scores": scores}


def words(text):
    """Case-fold and extract Unicode letter/digit runs for the LCS example."""
    return re.findall(r"[^\W_]+", text.casefold())


@torch.inference_mode()
def summaries(model, tokenizer):
    sources = {}
    for key, sentences in (("full", REPORT), ("lead", REPORT[:4])):
        inputs, encoded = source_tokens(tokenizer, " ".join(sentences))
        reference_words = words(SUMMARY_REFERENCE)
        outputs = {}
        for cap in SUMMARY_CAPS:
            trace = generate_path(model, tokenizer, inputs, **SUMMARY_SETTINGS, max_new_tokens=cap)
            candidate_words = words(trace["text"])
            trace["overlap"] = {"candidate_words": len(candidate_words),
                                "reference_words": len(reference_words),
                                **rouge_l(candidate_words, reference_words)}
            outputs[str(cap)] = trace
        sources[key] = {"source": encoded, "outputs": outputs}
    return {"settings": SUMMARY_SETTINGS, "caps": list(SUMMARY_CAPS),
            "report_sentences": list(REPORT), "reference": SUMMARY_REFERENCE, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ch19-pretrained.json"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true", help="Read weights from the local cache.")
    args = parser.parse_args()

    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(19)
    record = {"environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "sacrebleu": sacrebleu.__version__, "device": "cpu",
                              "dtype": "float32", "threads": THREADS, "seed": 19},
              "checkpoints": {}}
    for key, name, experiment in (("marian", "translation", translations),
                                  ("bart", "summary", summaries)):
        model, tokenizer, metadata = load_checkpoint(key, cache_dir=args.cache_dir, offline=args.offline)
        record["checkpoints"][key] = metadata
        record[name] = experiment(model, tokenizer)
        print(f"Recorded {name} with {metadata['name']} ({metadata['parameters']:,} parameters)",
              flush=True)
        del model, tokenizer

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
