"""Observe equivalent attention execution with real SmolLM2 arrays.

CPU SDPA is identified from the operator trace, not inferred from its API
name. The figures do not benchmark a CUDA FlashAttention implementation.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from .model import online_update, tiled_attention


MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TEXT = ("The red car stopped near the bridge. "
        "The driver checked the map before continuing. ") * 40
PREFILL = 512
DECODE = 2
PROBE = 383
KEY_TILE = 64
LENGTHS = (64, 128, 256, 512)
BACKENDS = ("eager", "sdpa")
WARMUPS = 3
REPETITIONS = 11


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def softmax64(scores):
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = dict(revision=REVISION, cache_dir=str(cache_dir), token=False,
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **common)
    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, **common, use_safetensors=True, attn_implementation="eager",
        output_loading_info=True)
    assert not any(loading.get(k) for k in
                   ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), loading
    assert model.config._commit_hash == REVISION
    model.eval().float().requires_grad_(False)
    assert (model.config.hidden_size, model.config.num_attention_heads,
            model.config.num_key_value_heads, model.config.num_hidden_layers) == (576, 9, 3, 30)
    complete = tokenizer(TEXT, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert complete.shape[1] >= PREFILL + DECODE
    ids = complete[:, :PREFILL + DECODE].clone()
    attention = model.model.layers[0].self_attn
    arrays, captured, native_kwargs, runs = {}, {}, {}, {}
    native_shapes = {}

    def before(module, args, kwargs):
        captured["input"] = kwargs["hidden_states"].detach().clone()
        captured["cos"], captured["sin"] = [x.detach().clone() for x in kwargs["position_embeddings"]]
        captured["mask"] = (None if kwargs["attention_mask"] is None
                            else kwargs["attention_mask"].detach().clone())

    def before_output(module, args):
        captured["heads"] = args[0].detach().numpy().copy()

    def after(module, args, result):
        if result[1] is not None:
            captured["weights"] = result[1][0, 0].detach().numpy().copy()

    with torch.inference_mode():
        for backend in BACKENDS:
            model.set_attn_implementation(backend)
            cache, logits = None, []
            for step in range(DECODE + 1):
                start, stop = (0, PREFILL) if step == 0 else (PREFILL + step - 1, PREFILL + step)
                handles = [attention.register_forward_pre_hook(before, with_kwargs=True),
                           attention.o_proj.register_forward_pre_hook(before_output),
                           attention.register_forward_hook(after)]
                try:
                    result = model(
                        input_ids=ids[:, start:stop], past_key_values=cache,
                        attention_mask=torch.ones((1, stop), dtype=torch.long),
                        position_ids=torch.arange(start, stop)[None],
                        cache_position=torch.arange(start, stop),
                        use_cache=True, output_attentions=False, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                cache = result.past_key_values
                logits.append(result.logits[0].detach().numpy().copy())
                if step == 0:
                    native_kwargs[backend] = {k: captured[k] for k in ("input", "cos", "sin", "mask")}
                    native_shapes[backend] = {
                        "input": list(captured["input"].shape),
                        "mask": None if captured["mask"] is None else list(captured["mask"].shape),
                        "heads_before_output_projection": list(captured["heads"].shape),
                    }
                    if backend == "eager":
                        arrays["prefill_probabilities"] = captured["weights"]
                        arrays["native_prefill_head_output"] = captured["heads"][0, :, :64]
                        arrays["native_k"] = cache[0][0][0, 0].detach().numpy().copy()
                        arrays["native_v"] = cache[0][1][0, 0].detach().numpy().copy()
                if backend == "eager" and step == 1:
                    arrays["decode_probabilities"] = captured["weights"]
                del result
            runs[backend] = np.concatenate(logits)
            arrays[backend + "_logits"] = runs[backend]
            arrays[backend + "_final_cache"] = np.stack([
                np.stack([x.detach().numpy().copy() for x in pair]) for pair in cache])
            print(backend, "complete-model passes finished", flush=True)
            del cache

        logit_error = np.max(np.abs(runs["eager"] - runs["sdpa"]), axis=-1)
        np.testing.assert_allclose(runs["eager"], runs["sdpa"], atol=5e-4, rtol=0)
        argmax_agreement = bool(np.array_equal(runs["eager"].argmax(-1), runs["sdpa"].argmax(-1)))
        cache_error = float(np.max(np.abs(arrays["eager_final_cache"] - arrays["sdpa_final_cache"])))
        np.testing.assert_allclose(arrays["eager_final_cache"], arrays["sdpa_final_cache"], atol=2e-4, rtol=0)
        arrays["logit_max_abs_error_by_position"] = logit_error

        # Reconstruct the first query head using every input coordinate and
        # the captured native RoPE arrays. Read native cached K/V head zero.
        data = native_kwargs["eager"]
        x = data["input"][0].numpy().astype(np.float64)
        q = x @ attention.q_proj.weight[:64].detach().numpy().astype(np.float64).T
        half = np.concatenate([-q[:, 32:], q[:, :32]], axis=-1)
        q = q * data["cos"][0].numpy() + half * data["sin"][0].numpy()
        k, v = arrays["native_k"].astype(np.float64), arrays["native_v"].astype(np.float64)
        scores = q @ k.T / 8
        allowed = np.arange(PREFILL)[None, :] <= np.arange(PREFILL)[:, None]
        p = softmax64(np.where(allowed, scores, -np.inf))
        np.testing.assert_allclose(p, arrays["prefill_probabilities"], atol=2e-6, rtol=0)
        np.testing.assert_allclose(p @ v, arrays["native_prefill_head_output"], atol=2e-6, rtol=0)
        assert np.count_nonzero(arrays["prefill_probabilities"][~allowed]) == 0
        tiled = tiled_attention(q, k, v, 37, 64, causal=True)
        np.testing.assert_allclose(tiled, p @ v, atol=5e-13, rtol=0)
        arrays.update(q=q, scores=scores, allowed=allowed, reconstructed_probabilities=p,
                      reconstructed_head_output=p @ v, tiled_head_output=tiled)

        state = np.full(1, -np.inf), np.zeros(1), np.zeros((1, 64))
        history = []
        for start in range(0, PREFILL, KEY_TILE):
            stop = min(start + KEY_TILE, PREFILL)
            old = [a.copy() for a in state]
            legal = np.arange(start, stop)[None, :] <= PROBE
            state = online_update(*state, scores[PROBE:PROBE+1, start:stop], v[start:stop], legal)
            if not legal.any():
                for a, b in zip(old, state):
                    np.testing.assert_array_equal(a, b)
            maximum, denominator, numerator = state
            partial = numerator[0] / denominator[0]
            history.append({"last_key": stop-1, "allowed_keys_in_tile": int(legal.sum()),
                            "maximum": float(maximum[0]), "denominator": float(denominator[0]),
                            "numerator_coordinate_zero": float(numerator[0, 0]),
                            "partial_output_coordinate_zero": float(partial[0])})
        np.testing.assert_allclose(partial, arrays["native_prefill_head_output"][PROBE], atol=2e-6, rtol=0)
        arrays["online_states"] = np.array([[h[k] for k in
            ("last_key", "allowed_keys_in_tile", "maximum", "denominator",
             "numerator_coordinate_zero", "partial_output_coordinate_zero")] for h in history])

        def kwargs_for(backend, length):
            d = native_kwargs[backend]
            mask = None if d["mask"] is None else d["mask"][:, :, :length, :length]
            return {"hidden_states": d["input"][:, :length],
                    "position_embeddings": (d["cos"][:, :length], d["sin"][:, :length]),
                    "attention_mask": mask, "past_key_values": None,
                    "cache_position": torch.arange(length)}

        profiles, timings = {}, {}
        for length in LENGTHS:
            profiles[str(length)] = {}
            timings[str(length)] = {b: [] for b in BACKENDS}
            outputs = {}
            for backend in BACKENDS:
                model.set_attn_implementation(backend)
                kw = kwargs_for(backend, length)
                for _ in range(WARMUPS):
                    attention(**kw)
                with torch.autograd.profiler.profile(use_cpu=True, record_shapes=True, profile_memory=True) as prof:
                    output = attention(**kw)
                outputs[backend] = output[0].detach().numpy().copy()
                events = prof.function_events
                positive = [e for e in events if e.self_cpu_memory_usage > 0]
                operators = sorted(set(e.name for e in events if "scaled_dot_product" in e.name))
                top = sorted(positive, key=lambda e: e.self_cpu_memory_usage, reverse=True)[:8]
                record = {
                    "sdpa_operators": operators,
                    "positive_self_cpu_memory_bytes": int(sum(e.self_cpu_memory_usage for e in positive)),
                    "largest_positive_self_cpu_memory_bytes": int(max(e.self_cpu_memory_usage for e in positive)),
                    "largest_events": [{"operator": e.name, "bytes": int(e.self_cpu_memory_usage),
                                        "input_shapes": e.input_shapes} for e in top],
                    "separate_softmax_events": [{"operator": e.name, "bytes": int(e.self_cpu_memory_usage),
                                                 "input_shapes": e.input_shapes} for e in events if e.name == "aten::_softmax"],
                }
                if backend == "sdpa":
                    assert "aten::_scaled_dot_product_flash_attention_for_cpu" in operators, operators
                else:
                    softmax_events = record["separate_softmax_events"]
                    assert len(softmax_events) == 1
                    assert softmax_events[0]["bytes"] == 9 * length * length * 4
                profiles[str(length)][backend] = record
                del output
            np.testing.assert_allclose(outputs["eager"], outputs["sdpa"], atol=3e-6, rtol=0)
            # Alternate order to reduce systematic order effects. Backend
            # switching and cleanup happen outside each timed interval.
            for repetition in range(REPETITIONS):
                order = BACKENDS if repetition % 2 == 0 else BACKENDS[::-1]
                for backend in order:
                    model.set_attn_implementation(backend)
                    kw = kwargs_for(backend, length)
                    begin = time.perf_counter_ns()
                    output = attention(**kw)
                    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
                    timings[str(length)][backend].append(elapsed_ms)
                    del output
            print("profiled and timed", length, flush=True)

    files = {}
    for filename in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
        file = hf_hub_download(MODEL_ID, filename, revision=REVISION, cache_dir=str(cache_dir),
                              local_files_only=True, token=False)
        files[filename] = {"sha256": sha256(file), "bytes": Path(file).stat().st_size}
    report = {
        "model": MODEL_ID, "revision": REVISION, "license": "Apache-2.0", "loading_info": loading,
        "loaded_parameters": sum(p.numel() for p in model.parameters()), "checkpoint_files": files,
        "text_before_token_truncation": TEXT, "input_ids": ids[0].tolist(),
        "token_pieces": tokenizer.convert_ids_to_tokens(ids[0]),
        "prefill": PREFILL, "cached_calls": DECODE, "probe_zero_based": PROBE,
        "head_zero_based": 0, "block_zero_based": 0, "key_tile": KEY_TILE,
        "native_shapes": native_shapes, "model_layers": 30, "query_heads": 9, "kv_heads": 3,
        "head_width": 64, "native_probability_max_abs_error": float(np.max(np.abs(p - arrays["prefill_probabilities"]))),
        "native_head_output_max_abs_error": float(np.max(np.abs(p @ v - arrays["native_prefill_head_output"]))),
        "tiled_vs_float64_materialized_max_abs_error": float(np.max(np.abs(tiled - p @ v))),
        "online_vs_native_probe_max_abs_error": float(np.max(np.abs(partial - arrays["native_prefill_head_output"][PROBE]))),
        "complete_model_max_abs_logit_error": float(logit_error.max()),
        "complete_model_argmax_agreement": argmax_agreement,
        "complete_final_cache_max_abs_error": cache_error,
        "online_state_history": history, "profiles": profiles, "timings_ms": timings,
        "timing_protocol": {"module": "block 1 self-attention including projections, RoPE, and output projection",
                            "prefix_lengths": LENGTHS, "warmups_per_backend_and_length": WARMUPS,
                            "samples_per_backend_and_length": REPETITIONS, "order": "alternating",
                            "timer": "perf_counter_ns; synchronous CPU calls; profiling disabled during timing"},
    }
    return arrays, report


def draw(arrays, report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "figure.dpi": 160, "savefig.dpi": 190,
                         "axes.spines.top": False, "axes.spines.right": False})
    files = []

    def save(fig, name):
        fig.savefig(output / name, facecolor="white", bbox_inches="tight", pad_inches=.12)
        plt.close(fig)
        files.append(name)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    fig = plt.figure(figsize=(10.5, 6), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, .06], height_ratios=[5, 1])
    ax = fig.add_subplot(grid[0, 0])
    p = arrays["prefill_probabilities"]
    im = ax.imshow(np.ma.masked_equal(p, 0), cmap=cmap, norm=LogNorm(1e-6, 1),
                   interpolation="nearest", aspect="auto")
    for boundary in range(KEY_TILE, PREFILL, KEY_TILE):
        ax.axvline(boundary-.5, color="white", alpha=.4, lw=.5)
        ax.axhline(boundary-.5, color="white", alpha=.4, lw=.5)
    ax.axhline(PROBE, color="#f97316", lw=1.2)
    ax.set(title="(a) Native prefill: 512 queries × 512 keys", xlabel="Key position (zero-based)",
           ylabel="Query position (zero-based)", xticks=[0, 128, 256, 383, 511], yticks=[0, 128, 256, 383, 511])
    ax2 = fig.add_subplot(grid[1, 0])
    row = arrays["decode_probabilities"]
    ax2.imshow(np.ma.masked_equal(row, 0), cmap=cmap, norm=LogNorm(1e-6, 1), aspect="auto", interpolation="nearest")
    ax2.set(title="(b) First cached call: 1 query × 513 keys", xlabel="Key position (zero-based)",
            yticks=[0], yticklabels=["Query 512"], xticks=[0, 128, 256, 383, 512])
    fig.colorbar(im, cax=fig.add_subplot(grid[:, 1]), label="Native attention probability (log scale)")
    fig.suptitle("SmolLM2 · block 1, head 1 · same weights and causal rule", fontsize=14)
    save(fig, "ch12_pretrained_stage_reads.png")

    history = report["online_state_history"]
    x = np.arange(1, len(history)+1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.7), layout="constrained", sharex=True)
    specs = [("maximum", "(a) Running maximum m", "Score reference", "#2563eb"),
             ("denominator", "(b) Shifted denominator ℓ", "Total shifted weight", "#0f766e"),
             ("numerator_coordinate_zero", "(c) Numerator coordinate u₀", "Signed weighted sum", "#7c3aed"),
             ("partial_output_coordinate_zero", "(d) Partial output u₀ / ℓ", "Head coordinate 0", "#ea580c")]
    for ax, (key, title, ylabel, color) in zip(axes.flat, specs):
        ax.plot(x, [h[key] for h in history], "o-", color=color)
        ax.axvspan(6.5, 8.5, color="#e5e7eb", zorder=-1)
        ax.set(title=title, ylabel=ylabel, xlim=(.7, 8.3), xticks=x)
        ax.grid(alpha=.2)
    axes[1, 1].axhline(arrays["native_prefill_head_output"][PROBE, 0], color="black", ls="--", lw=1, label="Native full-row result")
    axes[1, 1].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Key tile visited (64 keys per tile)")
    fig.suptitle("Real query 383 · tiles 7–8 contain only future keys and change nothing", fontsize=13)
    save(fig, "ch12_pretrained_online_state.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for backend, color, label in (("eager", "#2563eb", "Eager"), ("sdpa", "#ea580c", "PyTorch SDPA on CPU")):
        for ax, field in zip(axes, ("largest_positive_self_cpu_memory_bytes", "positive_self_cpu_memory_bytes")):
            values = [report["profiles"][str(n)][backend][field] / 2**20 for n in LENGTHS]
            ax.plot(LENGTHS, values, "o-", color=color, label=label)
    for ax, title in zip(axes, ("(a) Largest positive operator allocation", "(b) Sum of positive operator allocations")):
        ax.set(title=title, xlabel="Actual prefix positions n", ylabel="MiB", xticks=LENGTHS)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Operator allocation activity · native first attention module · CPU float32", fontsize=13)
    save(fig, "ch12_pretrained_allocations.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    axes[0].plot(arrays["logit_max_abs_error_by_position"], color="#7c3aed", lw=1)
    axes[0].axvline(PREFILL-.5, color="black", ls="--", lw=1)
    axes[0].set(title="(a) Complete model: eager versus SDPA", xlabel="Consumed input position (zero-based)",
                ylabel="Max absolute vocabulary-logit difference")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for backend, color, label in (("eager", "#2563eb", "Eager"), ("sdpa", "#ea580c", "PyTorch SDPA on CPU")):
        values = np.array([report["timings_ms"][str(n)][backend] for n in LENGTHS])
        q25, median, q75 = np.percentile(values, [25, 50, 75], axis=1)
        axes[1].errorbar(LENGTHS, median, yerr=np.stack([median-q25, q75-median]),
                         fmt="o-", color=color, label=label, capsize=4)
    axes[1].set(title="(b) CPU timing: median and interquartile range", xlabel="Actual prefix positions n",
                ylabel="First attention module time (ms)", xticks=LENGTHS)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Output agreement and timing are separate measurements · 1 CPU thread, float32", fontsize=12)
    save(fig, "ch12_pretrained_equivalence_timing.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, report = observe(args.cache_dir, args.offline)
    files = draw(arrays, report, args.output_dir)
    np.savez_compressed(args.output_dir / "observations.npz", **arrays)
    processor = platform.processor() or platform.machine()
    report.update({"observed_on": "2026-11-09", "device": "cpu", "processor": processor,
                   "dtype": "float32", "eval_mode": True, "weights_frozen": True,
                   "threads": 1, "sampling": False, "training": False,
                   "python": platform.python_version(), "platform": platform.platform(),
                   "software": {p: importlib.metadata.version(p) for p in
                                ("torch", "transformers", "tokenizers", "numpy", "matplotlib", "huggingface_hub")},
                   "source_sha256": sha256(__file__),
                   "outputs": {f: {"sha256": sha256(args.output_dir / f),
                                   "bytes": (args.output_dir / f).stat().st_size}
                               for f in files + ["observations.npz"]}})
    (args.output_dir / "PRETRAINED_FIGURES_RUN.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
        ("complete_model_max_abs_logit_error", "complete_model_argmax_agreement",
         "native_probability_max_abs_error", "native_head_output_max_abs_error",
         "tiled_vs_float64_materialized_max_abs_error", "online_state_history")}, indent=2))


if __name__ == "__main__":
    main()
