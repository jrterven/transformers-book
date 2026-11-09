"""Measure GPT-2 and BERT inputs, predictions, and context dependencies.

Uses pinned public safetensors checkpoints in eval mode on CPU. No training,
sampling, remote Python, or task-specific classifier is used. The chapter's
small objective examples remain independent of these optional dependencies.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np


GPT_MODEL = "openai-community/gpt2"
GPT_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
BERT_MODEL = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
GPT_TEXTS = ("The capital of France is", "The capital of Italy is")
BERT_TEXTS = ("The [MASK] can bark.", "The [MASK] can fly.")
PAIR = ("An unaffordable car.", "It stopped.")
STAGES = (0, 1, 4, 8, 12)  # Fixed before inspection; all stages are checked/saved.
FEATURES = tuple(range(8))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def piece(tokenizer, token_id, gpt=False):
    text = tokenizer.decode([int(token_id)]) if gpt else tokenizer.convert_ids_to_tokens(int(token_id))
    return text.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")


def top_tokens(tokenizer, probabilities, count=8, gpt=False):
    ids = np.argsort(-probabilities, kind="stable")[:count]
    return {"ids": ids.tolist(), "pieces": [piece(tokenizer, i, gpt) for i in ids],
            "probabilities": probabilities[ids].tolist(),
            "remaining_probability": float(1-probabilities[ids].sum(dtype=np.float64))}


def linear64(row, weight, bias=None):
    """Independent row projection, chunked to avoid a large float64 copy."""
    row = np.asarray(row, dtype=np.float64)
    output = np.empty(weight.shape[0], dtype=np.float64)
    for start in range(0, len(output), 1024):
        output[start:start+1024] = weight[start:start+1024].astype(np.float64) @ row
    return output if bias is None else output + bias.astype(np.float64)


def normalize64(values, module):
    mean = values.mean(-1, keepdims=True)
    variance = values.var(-1, keepdims=True)
    return ((values-mean)/np.sqrt(variance+module.eps)
            * module.weight.detach().numpy() + module.bias.detach().numpy())


def load_model(model_id, revision, cls, cache_dir, offline):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    common = dict(revision=revision, cache_dir=str(cache_dir), local_files_only=offline,
                  trust_remote_code=False, token=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model, info = cls.from_pretrained(model_id, **common, use_safetensors=True,
                                      attn_implementation="eager", output_loading_info=True)
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise AssertionError("the requested predictor has missing or mismatched weights")
    assert model.config._commit_hash == revision
    model.eval().float().requires_grad_(False)
    weights = hf_hub_download(model_id, "model.safetensors", revision=revision,
                              cache_dir=str(cache_dir), local_files_only=True, token=False)
    report = {"model": model_id, "revision": revision, "weights_sha256": sha256(weights),
              "loaded_predictor_parameters": sum(p.numel() for p in model.parameters()),
              "missing_keys": info["missing_keys"], "unused_checkpoint_keys": info["unexpected_keys"],
              "vocabulary_size": model.config.vocab_size}
    return tokenizer, model, report


def observe_gpt(cache_dir, offline):
    import torch
    from transformers import GPT2LMHeadModel
    tokenizer, model, report = load_model(GPT_MODEL, GPT_REVISION, GPT2LMHeadModel, cache_dir, offline)
    assert not report["unused_checkpoint_keys"]
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    traces, records = [], []
    with torch.inference_mode():
        for text in GPT_TEXTS:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            assert tokenizer.decode(encoded["input_ids"][0]) == text
            result = model(**encoded, output_attentions=True, output_hidden_states=True, use_cache=False)
            hidden = np.stack([state[0].numpy() for state in result.hidden_states])
            attention = np.stack([a[0].numpy() for a in result.attentions])
            probabilities = result.logits[0, -1].softmax(-1).numpy()
            reconstructed = linear64(hidden[-1, -1], model.lm_head.weight.numpy())
            np.testing.assert_allclose(reconstructed, result.logits[0, -1].numpy(), atol=2e-4, rtol=3e-6)
            reconstructed_p = np.exp(reconstructed-reconstructed.max())
            reconstructed_p /= reconstructed_p.sum()
            np.testing.assert_allclose(reconstructed_p, probabilities, atol=1e-5, rtol=2e-4)
            np.testing.assert_allclose(attention.sum(-1), 1, atol=3e-7, rtol=0)
            assert np.count_nonzero(np.triu(attention, 1)) == 0
            ids = encoded["input_ids"][0].numpy()
            traces.append({"ids": ids, "hidden": hidden, "attention": attention,
                           "lookup": model.transformer.wte(encoded["input_ids"])[0].numpy(),
                           "logits": result.logits[0].numpy(), "probabilities": probabilities})
            records.append({"text": text, "ids": ids.tolist(),
                            "pieces": [piece(tokenizer, i, True) for i in ids],
                            "top": top_tokens(tokenizer, probabilities, gpt=True),
                            "readout_max_error": float(np.abs(reconstructed-result.logits[0, -1].numpy()).max()),
                            "softmax_max_error": float(np.abs(reconstructed_p-probabilities).max())})
        original, changed = traces
        assert len(original["ids"]) == len(changed["ids"]) == 5
        assert np.array_equal(np.flatnonzero(original["ids"] != changed["ids"]), [3])
        np.testing.assert_allclose(original["hidden"][:, :3], changed["hidden"][:, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(original["logits"][:3], changed["logits"][:3], atol=1e-5, rtol=0)
        differences = np.linalg.norm(changed["hidden"].astype(np.float64)-original["hidden"], axis=-1)

        ids = tokenizer(GPT_TEXTS[0], return_tensors="pt", add_special_tokens=False)["input_ids"]
        generation, generation_logits = [], []
        for step in range(3):
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            probabilities = result.logits[0, -1].softmax(-1).numpy()
            chosen = int(result.logits[0, -1].argmax())
            generation.append({"step": step+1, "prefix": tokenizer.decode(ids[0]),
                               "prefix_ids": ids[0].tolist(), "selected_id": chosen,
                               "selected_piece": piece(tokenizer, chosen, True),
                               "selected_probability": float(probabilities[chosen]),
                               "top": top_tokens(tokenizer, probabilities, count=5, gpt=True)})
            generation_logits.append(result.logits[0, -1].numpy())
            ids = torch.cat((ids, ids.new_tensor([[chosen]])), -1)
            if chosen == tokenizer.eos_token_id:
                break
        full = tokenizer(GPT_TEXTS[0]+" Paris.", return_tensors="pt", add_special_tokens=False)
        assert torch.equal(full["input_ids"][:, :5], torch.tensor(original["ids"])[None])
        scored = model(**full, labels=full["input_ids"], use_cache=False)
        logp = scored.logits[0, :-1].log_softmax(-1)
        selected = logp.gather(-1, full["input_ids"][0, 1:, None])[:, 0]
        torch.testing.assert_close(-selected.mean(), scored.loss, atol=1e-6, rtol=1e-6)
        continuation = selected[4:]
        report.update(layers=model.config.n_layer, heads=model.config.n_head, width=model.config.n_embd,
                      maximum_positions=model.config.n_positions, tied_input_output_embeddings=True,
                      inputs=records, generation=generation, generated_text=tokenizer.decode(ids[0]),
                      generation_limit=3, stopped_on_eos=generation[-1]["selected_id"] == tokenizer.eos_token_id,
                      prefix_max_hidden_difference=float(np.abs(changed["hidden"][:, :3]-original["hidden"][:, :3]).max()),
                      hidden_l2_difference_by_stage_and_position=differences.tolist(),
                      scored_text=tokenizer.decode(full["input_ids"][0]),
                      scored_ids=full["input_ids"][0].tolist(),
                      scored_next_token_probabilities=selected.exp().tolist(), mean_scored_nll=float(scored.loss),
                      continuation_pieces=[piece(tokenizer, i, True) for i in full["input_ids"][0, 5:]],
                      continuation_probabilities=continuation.exp().tolist(),
                      continuation_nll=float(-continuation.mean()),
                      continuation_probability=float(continuation.sum().exp()))
    arrays = {f"gpt_{prefix}_{name}": value for prefix, trace in zip(("original", "changed"), traces)
              for name, value in trace.items()}
    arrays.update(gpt_hidden_difference=differences, gpt_generation_logits=np.stack(generation_logits))
    del model
    gc.collect()
    return report, arrays


def observe_bert(cache_dir, offline):
    import torch
    from transformers import BertForMaskedLM
    tokenizer, model, report = load_model(BERT_MODEL, BERT_REVISION, BertForMaskedLM, cache_dir, offline)
    assert set(report["unused_checkpoint_keys"]) <= {
        "bert.pooler.dense.bias", "bert.pooler.dense.weight",
        "cls.seq_relationship.bias", "cls.seq_relationship.weight"}
    embeddings = model.bert.embeddings
    head = model.cls.predictions
    assert embeddings.word_embeddings.weight.data_ptr() == head.decoder.weight.data_ptr()
    assert model.config.hidden_act == "gelu"
    with torch.inference_mode():
        pair = tokenizer(*PAIR, return_tensors="pt")
        n = pair["input_ids"].shape[1]
        positions = torch.arange(n)[None]
        word = embeddings.word_embeddings(pair["input_ids"])[0].numpy()
        position = embeddings.position_embeddings(positions)[0].numpy()
        segment = embeddings.token_type_embeddings(pair["token_type_ids"])[0].numpy()
        summed = word.astype(np.float64) + position + segment
        reconstructed = normalize64(summed, embeddings.LayerNorm)
        native_embedding = embeddings(input_ids=pair["input_ids"],
                                       token_type_ids=pair["token_type_ids"], position_ids=positions)[0].numpy()
        np.testing.assert_allclose(reconstructed, native_embedding, atol=5e-6, rtol=3e-6)
        selected_position = int((pair["token_type_ids"][0] == 1).nonzero()[0])
        pair_report = {"segments": list(PAIR), "ids": pair["input_ids"][0].tolist(),
                       "pieces": tokenizer.convert_ids_to_tokens(pair["input_ids"][0]),
                       "segment_ids": pair["token_type_ids"][0].tolist(),
                       "positions": list(range(n)), "display_position": selected_position,
                       "display_piece": tokenizer.convert_ids_to_tokens(int(pair["input_ids"][0, selected_position])),
                       "embedding_max_error": float(np.abs(reconstructed-native_embedding).max()),
                       "embedding_first_feature": {"word": float(word[selected_position, 0]),
                                                   "position": float(position[selected_position, 0]),
                                                   "segment": float(segment[selected_position, 0]),
                                                   "sum": float(summed[selected_position, 0]),
                                                   "after_layernorm": float(native_embedding[selected_position, 0])}}
        traces, records = [], []
        for text in BERT_TEXTS:
            encoded = tokenizer(text, return_tensors="pt")
            mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()[:, 0]
            assert mask_positions.numel() == 1
            mask_position = int(mask_positions[0])
            result = model(**encoded, output_attentions=True, output_hidden_states=True)
            hidden = np.stack([state[0].numpy() for state in result.hidden_states])
            attention = np.stack([a[0].numpy() for a in result.attentions])
            h = hidden[-1, mask_position]
            dense = linear64(h, head.transform.dense.weight.numpy(), head.transform.dense.bias.numpy())
            gelu = .5*dense*(1+np.array([math.erf(v/math.sqrt(2)) for v in dense]))
            transformed = normalize64(gelu, head.transform.LayerNorm)
            native_transform = head.transform(result.hidden_states[-1][0, mask_position]).numpy()
            np.testing.assert_allclose(transformed, native_transform, atol=6e-6, rtol=3e-6)
            reconstructed_logits = linear64(transformed, head.decoder.weight.numpy(), head.bias.numpy())
            native_logits = result.logits[0, mask_position].numpy()
            np.testing.assert_allclose(reconstructed_logits, native_logits, atol=2e-4, rtol=3e-6)
            np.testing.assert_allclose(attention.sum(-1), 1, atol=3e-7, rtol=0)
            probabilities = result.logits[0, mask_position].softmax(-1).numpy()
            ids = encoded["input_ids"][0].numpy()
            traces.append({"ids": ids, "hidden": hidden, "attention": attention,
                           "mask_logits": native_logits, "probabilities": probabilities,
                           "mask_transformed": native_transform})
            records.append({"text": text, "ids": ids.tolist(), "pieces": tokenizer.convert_ids_to_tokens(ids),
                            "mask_position": mask_position, "top": top_tokens(tokenizer, probabilities),
                            "dog_probability": float(probabilities[tokenizer.convert_tokens_to_ids("dog")]),
                            "bird_probability": float(probabilities[tokenizer.convert_tokens_to_ids("bird")]),
                            "mlm_transform_max_error": float(np.abs(transformed-native_transform).max()),
                            "readout_max_error": float(np.abs(reconstructed_logits-native_logits).max())})
        original, changed = traces
        assert len(original["ids"]) == len(changed["ids"]) == 7
        assert np.array_equal(np.flatnonzero(original["ids"] != changed["ids"]), [4])
        assert records[0]["mask_position"] == records[1]["mask_position"] == 2
        np.testing.assert_array_equal(original["hidden"][0, :4], changed["hidden"][0, :4])
        differences = np.linalg.norm(changed["hidden"].astype(np.float64)-original["hidden"], axis=-1)
        assert differences[-1, 2] > 1e-5  # Right-context change reaches the earlier mask.
        report.update(layers=model.config.num_hidden_layers, heads=model.config.num_attention_heads,
                      width=model.config.hidden_size, maximum_positions=model.config.max_position_embeddings,
                      tied_input_output_embeddings=True, pair=pair_report, inputs=records,
                      hidden_l2_difference_by_stage_and_position=differences.tolist(),
                      mask_input_embedding_difference=float(differences[0, 2]),
                      mask_final_hidden_difference=float(differences[-1, 2]))
    arrays = {f"bert_{prefix}_{name}": value for prefix, trace in zip(("original", "changed"), traces)
              for name, value in trace.items()}
    arrays.update(bert_pair_ids=pair["input_ids"][0].numpy(), bert_pair_word=word,
                  bert_pair_position=position, bert_pair_segment=segment, bert_pair_sum=summed,
                  bert_pair_embedding=native_embedding, bert_hidden_difference=differences)
    del model
    gc.collect()
    return report, arrays


def draw_figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    paths = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    def heatmap(ax, values, rows, columns, title, signed=False, limit=None, decimals=2):
        limit = float(np.abs(values).max()) if limit is None else limit
        im = ax.imshow(values, aspect="auto", cmap="PuOr_r" if signed else "Blues",
                       vmin=-limit if signed else 0, vmax=limit)
        ax.set_xticks(range(values.shape[1]), columns)
        ax.set_yticks(range(values.shape[0]), rows)
        ax.set_title(title, loc="left", pad=10)
        for (r, c), value in np.ndenumerate(values):
            displayed = 0.0 if round(float(value), decimals) == 0 else value
            ax.text(c, r, f"{displayed:.{decimals}f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value)>.65*limit else "#222222")
        return im

    def probability_bars(ax, top, title, maximum=None):
        y = np.arange(len(top["ids"]))
        ax.barh(y, top["probabilities"], color="#2468a0", height=.65)
        ax.set_yticks(y, top["pieces"])
        ax.invert_yaxis()
        bound = maximum or max(top["probabilities"])*1.3
        ax.set_xlim(0, bound)
        for i, value in enumerate(top["probabilities"]):
            ax.text(value+bound*.015, i, f"{value:.4f}", va="center", fontsize=8)
        ax.set(title=title, xlabel="Probability in the full vocabulary")
        ax.text(1, -.25, f"All unshown tokens: {top['remaining_probability']:.4f}",
                ha="right", va="top", transform=ax.transAxes, fontsize=9)

    gpt = report["gpt"]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.5), layout="constrained",
                             gridspec_kw={"height_ratios": [.7, 1, 2.2]})
    axes[0].axis("off")
    table = axes[0].table(cellText=[gpt["inputs"][0]["pieces"], gpt["inputs"][0]["ids"]],
                          rowLabels=["Piece", "ID"], colLabels=[str(i) for i in range(5)],
                          cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    axes[0].set_title("(a) GPT-2 tokenizes the supplied prefix (zero-based positions)", loc="left")
    values = np.stack((arrays["gpt_original_lookup"][-1, :8], arrays["gpt_original_hidden"][-1, -1, :8]))
    im = heatmap(axes[1], values, ["Token lookup", "Final context"], range(1, 9),
                 "(b) The last input token: first 8 of 768 coordinates", signed=True)
    fig.colorbar(im, ax=axes[1], shrink=.8)
    probability_bars(axes[2], gpt["inputs"][0]["top"], "(c) Vocabulary scores become next-token probabilities")
    save(fig, "ch07_gpt_prediction.png")

    fig, axes = plt.subplots(len(gpt["generation"]), 1, figsize=(8.4, 6.3), layout="constrained")
    maximum = max(max(row["top"]["probabilities"]) for row in gpt["generation"])*1.18
    for ax, row in zip(np.atleast_1d(axes), gpt["generation"]):
        probability_bars(ax, row["top"], f"Step {row['step']}: {row['prefix']!r}", maximum)
        ax.patches[0].set_facecolor("#168477")
    save(fig, "ch07_gpt_generation.png")

    def context_figure(kind, titles, labels, changed_label, name):
        fig = plt.figure(figsize=(8.4, 6.3), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
        for i, prefix in enumerate(("original", "changed")):
            ax = fig.add_subplot(grid[0, i])
            current_labels = report[kind]["inputs"][i]["pieces"]
            im = heatmap(ax, arrays[f"{kind}_{prefix}_attention"][0, 0], current_labels, current_labels,
                         titles[i], limit=1)
            ax.tick_params(axis="x", rotation=25)
            ax.set(xlabel="Key position", ylabel="Query position")
            fig.colorbar(im, ax=ax, shrink=.75)
        ax = fig.add_subplot(grid[1, :])
        values = arrays[f"{kind}_hidden_difference"][list(STAGES)]
        im = heatmap(ax, values, list(STAGES), labels,
                     "Hidden-vector difference after changing " + changed_label, decimals=2)
        ax.set(xlabel="Input position (labels identify the original input)",
               ylabel="Returned stage (0 = input)")
        fig.colorbar(im, ax=ax, shrink=.8, label="L2 difference, all 768 features")
        save(fig, name)

    context_figure("gpt", ["GPT-2: France prefix", "GPT-2: Italy prefix"],
                   gpt["inputs"][0]["pieces"], "France to Italy",
                   "ch07_gpt_causal_context.png")

    bert = report["bert"]
    pair = bert["pair"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.7), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.5]})
    axes[0].axis("off")
    table = axes[0].table(cellText=[pair["pieces"], pair["ids"], pair["segment_ids"]],
                          rowLabels=["WordPiece", "ID", "Segment"],
                          colLabels=pair["positions"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if c >= 0 and r > 0:
            cell.set_facecolor("#e9f0f6" if pair["segment_ids"][c] == 0 else "#fff0df")
    axes[0].set_title("(a) BERT encodes two segments, including WordPieces and boundaries", loc="left")
    row = pair["display_position"]
    names = ("word", "position", "segment", "sum", "embedding")
    values = np.stack([arrays["bert_pair_"+name][row, :8] for name in names])
    im = heatmap(axes[1], values, ["Word lookup", "Position", "Segment", "Sum", "After LayerNorm"],
                 range(1, 9), f"(b) Input at position {row}: {pair['display_piece']!r} (first 8 of 768 features)", signed=True)
    axes[1].set_xlabel("Feature coordinate; LayerNorm uses all 768 features")
    fig.colorbar(im, ax=axes[1], shrink=.8)
    save(fig, "ch07_bert_inputs.png")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), layout="constrained")
    maximum = max(max(row["top"]["probabilities"]) for row in bert["inputs"])*1.3
    for ax, row in zip(axes, bert["inputs"]):
        probability_bars(ax, row["top"], row["text"], maximum)
    save(fig, "ch07_bert_mask_predictions.png")
    context_figure("bert", ["BERT: can bark", "BERT: can fly"],
                   bert["inputs"][0]["pieces"], "bark to fly",
                   "ch07_bert_bidirectional_context.png")
    return {path.name: sha256(path) for path in paths}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/ch07-pretrained"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpt, gpt_arrays = observe_gpt(args.cache_dir, args.offline)
    bert, bert_arrays = observe_bert(args.cache_dir, args.offline)
    report = {"gpt": gpt, "bert": bert,
              "mode": "frozen CPU float32 models in eval mode; eager attention; no training or sampling",
              "selection": "fixed prompts, first block/head, first eight features, and stages 0/1/4/8/12 chosen before inspection",
              "software": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy", "matplotlib", "Pillow")}}}
    arrays = {**gpt_arrays, **bert_arrays}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "observations.npz"
    np.savez_compressed(archive, **arrays)
    report["artifacts"] = {archive.name: sha256(archive), **draw_figures(report, arrays, args.output_dir)}
    report["source_sha256"] = sha256(__file__)
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
