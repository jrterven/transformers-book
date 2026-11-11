"""Run retrieval, reranking, and grounded generation on a fictional archive."""

import argparse
from collections import defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re

import numpy as np
import torch
import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3_5ForConditionalGeneration,
)
from transformers.utils.hub import cached_file

from .model import bm25_scores, dense_scores, evidence_recall, rank_passages, rerank_candidates


CHECKPOINTS = {
    "embedding": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "weight_file": "model.safetensors",
        "weight_sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    },
    "reranker": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "weight_file": "model.safetensors",
        "weight_sha256": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
    },
    "generator": {
        "name": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "weight_file": "model.safetensors-00001-of-00001.safetensors",
        "weight_sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
}
FIXTURE = Path(__file__).with_name("CAMPUS_NOTICES.json")
TASK = "Given a campus services question, retrieve passages that provide the answer."
RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PLAIN_SYSTEM = (
    "Answer the question using only the supplied campus passages. Give one short sentence "
    "and cite the supporting passage IDs in square brackets. Copy the names of places exactly. "
    "If the passages do not provide the answer, reply exactly: Insufficient evidence."
)
SYSTEM = (
    "Answer campus questions using the given passages. Write the answer in one sentence "
    "followed by the supporting passage ID in brackets. If the passages do not answer "
    "the question, write Insufficient evidence."
)
EXAMPLE = (
    {"role": "user", "content": "Passages:\n[P90] The community music rehearsal takes place in "
                                "Room 7 of Spruce Hall.\n\nQuestion: Where is the community music rehearsal?"},
    {"role": "assistant", "content": "The community music rehearsal takes place in Room 7 of Spruce Hall [P90]."},
)
SHORTLIST = 5
RERANK_LIMIT = 6
RRF_OFFSET = 10
INPUT_CAP = 360
ANSWER_CAP = 64
THREADS = 4
FOCUS = "renew_paraphrase"
EVIDENCE = "P01"


def load_model(role, model_class, cache_dir, offline):
    spec = CHECKPOINTS[role]
    name = spec["name"]
    common = dict(revision=spec["revision"], cache_dir=cache_dir,
                  local_files_only=offline, token=False)
    weight = Path(cached_file(name, spec["weight_file"], **common))
    with weight.open("rb") as handle:
        actual_hash = file_digest(handle, "sha256").hexdigest()
    if actual_hash != spec["weight_sha256"]:
        raise ValueError(f"Unexpected weight checksum for {name}: {actual_hash}")
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left", trust_remote_code=False,
                                             **common)
    model, loading = model_class.from_pretrained(
        name, dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
        trust_remote_code=False, output_loading_info=True, **common)
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete checkpoint load: {loading}")
    unused = loading.get("unexpected_keys") or []
    if role == "generator":
        expected_unused = all(k.startswith("mtp.") or ".mtp." in k for k in unused)
    elif role == "embedding":
        expected_unused = set(unused) <= {"lm_head.weight"}
    else:
        expected_unused = not unused
    if not expected_unused:
        raise ValueError(f"Unexpected unused tensors in {name}: {unused}")
    metadata = {
        **spec,
        "parameters": sum(param.numel() for param in model.parameters()),
        "unused_weights": sorted(unused),
        "tokenizer_class": type(tokenizer).__name__,
    }
    model.eval()
    return model, tokenizer, metadata


def embed(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    # Select the last visible position in each row, including in a padded batch.
    mask = inputs["attention_mask"].bool()
    positions = torch.arange(mask.shape[1]).expand_as(mask)
    last = positions.masked_fill(~mask, -1).amax(dim=1)
    hidden = model(**inputs, use_cache=False).last_hidden_state
    vectors = hidden[torch.arange(len(texts)), last].double()
    norms = torch.linalg.vector_norm(vectors, dim=-1)
    normalized = vectors / norms[:, None]
    return normalized.numpy(), norms.tolist(), mask.sum(dim=-1).tolist()


def fused_candidates(lexical, dense):
    scores = defaultdict(float)
    for ranking in (lexical, dense):
        for rank, pid in enumerate(ranking[:SHORTLIST], start=1):
            scores[pid] += 1 / (RRF_OFFSET + rank)
    return scores, rank_passages(list(scores), list(scores.values()), RERANK_LIMIT)


def retrieve(model, tokenizer, fixture):
    passages = fixture["passages"]
    questions = fixture["questions"]
    passage_vectors, passage_norms, passage_tokens = embed(model, tokenizer, list(passages.values()))
    query_texts = [f"Instruct: {TASK}\nQuery: {row['text']}" for row in questions]
    query_vectors, query_norms, query_tokens = embed(model, tokenizer, query_texts)
    rows = {}
    for question, q in zip(questions, query_vectors, strict=True):
        lexical = dict(zip(passages, bm25_scores(question["text"], passages.values()).tolist(), strict=True))
        dense = dict(zip(passages, dense_scores(q, passage_vectors).tolist(), strict=True))
        lexical_order = [pid for pid in rank_passages(passages, list(lexical.values())) if lexical[pid] > 0]
        dense_order = list(rank_passages(passages, list(dense.values())))
        fused, candidates = fused_candidates(lexical_order, dense_order)
        rows[question["id"]] = {
            "bm25_scores": lexical,
            "cosine_scores": dense,
            "rrf_scores": fused,
            "bm25": lexical_order,
            "dense": dense_order,
            "fused": candidates,
        }
    qi = [row["id"] for row in questions].index(FOCUS)
    pi = list(passages).index(EVIDENCE)
    q, p = query_vectors[qi], passage_vectors[pi]
    diagnostics = {
        "width": int(q.shape[0]),
        "passage_norms": dict(zip(passages, passage_norms, strict=True)),
        "passage_lengths": dict(zip(passages, passage_tokens, strict=True)),
        "query_norms": dict(zip((row["id"] for row in questions), query_norms, strict=True)),
        "query_lengths": dict(zip((row["id"] for row in questions), query_tokens, strict=True)),
        "focus_dot_product": {
            "question": FOCUS, "passage": EVIDENCE,
            "query_first_four": q[:4].tolist(), "passage_first_four": p[:4].tolist(),
            "first_four_dot": float(q[:4] @ p[:4]),
            "remaining_dot": float(q[4:] @ p[4:]),
        },
    }
    return rows, diagnostics


def rerank(model, tokenizer, fixture, rows):
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("no", "yes")]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Expected single-token relevance labels")
    label_ids = [ids[0] for ids in labels]
    for q in fixture["questions"]:
        row = rows[q["id"]]
        candidates = row["fused"]
        prompts = [f"{RERANK_PREFIX}<Instruct>: {TASK}\n<Query>: {q['text']}\n"
                   f"<Document>: {fixture['passages'][pid]}{RERANK_SUFFIX}" for pid in candidates]
        inputs = tokenizer(prompts, padding=True, add_special_tokens=False,
                           truncation=False, return_tensors="pt")
        logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, label_ids].double()
        probabilities = logits.softmax(dim=-1)[:, 1]
        scores = {}
        for pid, pair, prob in zip(candidates, logits.tolist(), probabilities.tolist(), strict=True):
            scores[pid] = {"no_logit": pair[0], "yes_logit": pair[1], "p_yes": prob}
        row["pair_scores"] = scores
        row["reranked"] = rerank_candidates(candidates, {pid: scores[pid]["p_yes"] for pid in candidates})
    return label_ids


def format_prompt(tokenizer, question, ids, passages, *, demonstrate=True):
    formatted = "\n\n".join(f"[{pid}] {passages[pid]}" for pid in ids)
    if demonstrate:
        messages = [
            {"role": "system", "content": SYSTEM}, *EXAMPLE,
            {"role": "user", "content": f"Passages:\n{formatted}\n\nQuestion: {question}"},
        ]
    else:
        messages = [
            {"role": "system", "content": PLAIN_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n{formatted}"},
        ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=True)["input_ids"]


def pack_prompt(tokenizer, question, candidates, passages, input_cap, *, demonstrate=True):
    kept, decisions = [], []
    ids = format_prompt(tokenizer, question, kept, passages, demonstrate=demonstrate)
    empty = len(ids)
    if empty > input_cap:
        raise ValueError("The input cap must fit the formatted question and instructions")
    for pid in candidates:
        trial = format_prompt(tokenizer, question, [*kept, pid], passages, demonstrate=demonstrate)
        fits = len(trial) <= input_cap
        decisions.append({"id": pid, "trial_tokens": len(trial), "kept": fits})
        if fits:
            kept.append(pid)
            ids = trial
    return {"candidates": list(candidates), "input_cap": input_cap, "empty_prompt_tokens": empty,
            "packed": kept, "prompt_tokens": len(ids), "packing": decisions}, ids


def generate_answer(model, tokenizer, ids):
    inputs = torch.tensor([ids], dtype=torch.long)
    eos = model.generation_config.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else eos
    config = GenerationConfig(
        max_new_tokens=ANSWER_CAP, do_sample=False, use_cache=True,
        eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
    )
    result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            generation_config=config)
    tail = result[0, len(ids):].tolist()
    text = tokenizer.decode(tail, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    citations = list(dict.fromkeys(re.findall(r"\[(P\d{2})\]", text)))
    return {"text": text, "generated_ids": tail, "ended": tail[-1] in stop_ids,
            "citations": citations}


def run_condition(model, tokenizer, question, candidates, passages, input_cap, *, demonstrate=True):
    run, ids = pack_prompt(tokenizer, question, candidates, passages, input_cap, demonstrate=demonstrate)
    run["answer"] = generate_answer(model, tokenizer, ids)
    return run


def answer_questions(model, tokenizer, fixture, rows):
    passages, questions = fixture["passages"], fixture["questions"]
    runs = {}
    for q in questions:
        candidates = rows[q["id"]]["reranked"]
        runs[q["id"]] = run_condition(model, tokenizer, q["text"], candidates, passages, INPUT_CAP)
        print(f"Generated answer for {q['id']}", flush=True)

    q = next(q["text"] for q in questions if q["id"] == FOCUS)
    row = rows[FOCUS]
    normal = runs[FOCUS]
    first_lexical = row["bm25"][0]
    tight = len(format_prompt(tokenizer, q, [first_lexical], passages))
    if first_lexical == EVIDENCE or tight >= len(format_prompt(tokenizer, q, [EVIDENCE], passages)):
        raise ValueError("The controlled contrast needs a shorter lexical distractor")
    replacement = passages[EVIDENCE].replace("Alder Hall", "Juniper Hall")
    if replacement == passages[EVIDENCE] or EVIDENCE not in normal["packed"]:
        raise ValueError("The location intervention requires the original passage in the input")
    changed_passages = {**passages, EVIDENCE: replacement}
    evidence_only = run_condition(model, tokenizer, q, [EVIDENCE], passages, INPUT_CAP)
    modified = run_condition(model, tokenizer, q, [EVIDENCE], changed_passages, INPUT_CAP)
    if modified["packed"] != evidence_only["packed"] or modified["packed"] != [EVIDENCE]:
        raise ValueError("Both location probes must see the same passage ID")
    modified["passage_edit"] = {"id": EVIDENCE, "text": replacement}
    controls = {
        "question": FOCUS,
        "lexical_top_one": run_condition(model, tokenizer, q, [first_lexical], passages, INPUT_CAP),
        "tight_input": run_condition(model, tokenizer, q, row["reranked"], passages, tight),
        "plain_evidence_only": run_condition(model, tokenizer, q, [EVIDENCE], passages, INPUT_CAP,
                                            demonstrate=False),
        "evidence_only": evidence_only,
        "location_edited": modified,
    }
    return runs, controls


def evaluate_retrieval(questions, rows, answers):
    included = [q for q in questions if q["evidence"]]
    scores = {}
    for stage in ("bm25", "dense", "fused", "reranked", "packed"):
        stage_rows = []
        for q in included:
            key, gold = q["id"], q["evidence"]
            ids = answers[key]["packed"] if stage == "packed" else rows[key][stage]
            ks = [len(ids)] if stage == "packed" else [1, 3]
            if stage in ("fused", "reranked"):
                ks.append(RERANK_LIMIT)
            recalls = {str(k): evidence_recall(ids, gold, k) for k in ks}
            if stage == "packed":
                recalls = {"all": next(iter(recalls.values()))}
            stage_rows.append(recalls)
        scores[stage] = {}
        for key in stage_rows[0]:
            per_query = [r[key] for r in stage_rows]
            scores[stage][key] = {
                "mean_recall": sum(per_query) / len(per_query),
                "complete_questions": sum(recall == 1 for recall in per_query),
                "question_count": len(per_query),
            }
    return scores


@torch.inference_mode()
def observe(fixture, cache_dir, offline):
    record = {
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "numpy": np.__version__, "device": "cpu", "dtype": "float32",
                        "threads": THREADS, "seed": 20},
        "checkpoints": {}, "fixture": fixture,
        "settings": {"instruction": TASK, "bm25": {"k1": 1.2, "b": 0.75},
                     "shortlist": SHORTLIST, "rerank_limit": RERANK_LIMIT, "rrf_offset": RRF_OFFSET,
                     "system_message": SYSTEM, "format_example": EXAMPLE, "plain_system_message": PLAIN_SYSTEM,
                     "input_cap": INPUT_CAP, "answer_cap": ANSWER_CAP},
    }
    model, tokenizer, record["checkpoints"]["embedding"] = load_model("embedding", AutoModel, cache_dir, offline)
    rows, record["embedding_diagnostics"] = retrieve(model, tokenizer, fixture)
    print("Recorded lexical and dense rankings", flush=True)
    del model, tokenizer

    model, tokenizer, record["checkpoints"]["reranker"] = load_model("reranker", AutoModelForCausalLM, cache_dir, offline)
    record["settings"]["reranker_label_ids_no_yes"] = rerank(model, tokenizer, fixture, rows)
    record["retrieval"] = rows
    print("Recorded relevance scores and reranked candidates", flush=True)
    del model, tokenizer

    model, tokenizer, record["checkpoints"]["generator"] = load_model(
        "generator", Qwen3_5ForConditionalGeneration, cache_dir, offline)
    record["settings"]["chat_template_sha256"] = sha256(tokenizer.chat_template.encode()).hexdigest()
    record["answers"], record["controls"] = answer_questions(model, tokenizer, fixture, rows)
    record["retrieval_metrics"] = evaluate_retrieval(fixture["questions"], rows, record["answers"])
    print("Recorded answers and evidence controls", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ch20-pretrained.json"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true", help="Read from the local model cache.")
    args = parser.parse_args()
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data = observe(fixture, args.cache_dir, args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
