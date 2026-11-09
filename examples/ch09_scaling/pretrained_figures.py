"""Measure routing, expert computation, and capacity in pretrained Switch-base-8.

Loads the published encoder with PyTorch's restricted weights-only loader.
The reduced-capacity experiment reuses one captured layer input and changes no
weights; it is distinct from the checkpoint's original capacity setting.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np


MODEL = "google/switch-base-8"
REVISION = "92fe2d22b024d9937146fe097ba3d3a7ba146e1b"
TEXT = "The red car stopped near the old bridge. The blue car moved toward the river."
POSITION = 2
FEATURES = tuple(range(8))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def softmax64(logits):
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits-logits.max(-1, keepdims=True))
    return exp/exp.sum(-1, keepdims=True)


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, SwitchTransformersEncoderModel
    from transformers.models.switch_transformers.modeling_switch_transformers import load_balancing_loss_func

    common = dict(revision=REVISION, cache_dir=str(cache_dir), token=False,
                  local_files_only=offline, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, **common)
    model, loading = SwitchTransformersEncoderModel.from_pretrained(
        MODEL, **common, use_safetensors=False, weights_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
    unused = loading["unexpected_keys"]
    assert all(k.startswith(("decoder.", "lm_head.")) for k in unused)
    assert model.config._commit_hash == REVISION
    model.eval().float().requires_grad_(False)
    encoded = tokenizer(TEXT, return_tensors="pt", padding=False)
    ids = encoded["input_ids"][0].numpy()
    assert not np.any(ids == tokenizer.pad_token_id)
    assert ids[-1] == tokenizer.eos_token_id
    raw_pieces = tokenizer.convert_ids_to_tokens(ids)
    assert raw_pieces[POSITION] == "▁car"
    pieces = [p.replace("▁", "·") for p in raw_pieces]
    count, width = len(ids), model.config.d_model
    layers = [(i, block.layer[-1]) for i, block in enumerate(model.encoder.block) if block.layer[-1].is_sparse]
    assert len(layers) == 6
    layer_index, wrapper = layers[0]
    branch = wrapper.mlp
    capacity, expert_count = branch.router.expert_capacity, model.config.num_experts
    assert capacity == 64 and expert_count == 8 and count <= capacity
    assert wrapper.dropout.training is False and branch.router.training is False
    captured, routes, expert_outputs, handles = {}, [], {}, []

    def capture_wrapper(module, args, output):
        captured["residual"] = args[0].detach().numpy().copy()
        captured["residual_output"] = output[0].detach().numpy().copy()

    def capture_branch(module, args, output):
        captured["router_input"] = args[0].detach().numpy().copy()
        captured["branch_output"] = output[0].detach().numpy().copy()

    def capture_router(module, args, output):
        captured["mask"] = output[0].detach().numpy().copy()
        captured["gates"] = output[1].detach().numpy().copy()
        captured["logits"] = output[2].detach().numpy().copy()

    handles.extend([wrapper.register_forward_hook(capture_wrapper), branch.register_forward_hook(capture_branch),
                    branch.router.register_forward_hook(capture_router)])
    for _, sparse in layers:
        handles.append(sparse.mlp.router.register_forward_hook(
            lambda module, args, output: routes.append(tuple(t.detach().numpy().copy() for t in output))))
    for i, expert in enumerate(branch.experts.values()):
        def capture_expert(module, args, output, index=i):
            expert_outputs[index] = (args[0].detach().numpy().copy(), output.detach().numpy().copy())
        handles.append(expert.register_forward_hook(capture_expert))
    with torch.inference_mode():
        native = model(**encoded, output_router_logits=True, return_dict=True)
    for handle in handles:
        handle.remove()
    arrays = {key: value[0] for key, value in captured.items()}
    arrays["ids"] = ids
    arrays["all_router_logits"] = np.stack([route[2][0] for route in routes])
    arrays["encoder_output"] = native.last_hidden_state[0].numpy()
    for mask, gate, logits in routes:
        np.testing.assert_array_equal(mask.sum(-1), 1)
        np.testing.assert_allclose(gate[..., 0], softmax64(logits).max(-1), atol=3e-6, rtol=2e-5)

    x, h, logits = arrays["router_input"], arrays["residual"], arrays["logits"]
    independent_logits = x.astype(np.float64) @ branch.router.classifier.weight.numpy().astype(np.float64).T
    np.testing.assert_allclose(independent_logits, logits, atol=5e-5, rtol=3e-5)
    with torch.inference_mode():
        probabilities, _ = branch.router._compute_router_probabilities(torch.from_numpy(captured["router_input"]))
    p = probabilities[0].numpy()
    np.testing.assert_allclose(softmax64(independent_logits), p, atol=5e-6, rtol=3e-5)
    requests = p.argmax(-1)
    np.testing.assert_array_equal(requests, logits.argmax(-1))
    masks = np.eye(expert_count, dtype=np.int64)[requests]
    np.testing.assert_array_equal(masks, arrays["mask"])
    np.testing.assert_array_equal(p.max(-1), arrays["gates"][:, 0])
    arrays["probabilities"], arrays["requests"] = p, requests
    full_expert_output = np.zeros_like(x)
    for index, (observed_input, output) in expert_outputs.items():
        rows = np.flatnonzero(requests == index)
        np.testing.assert_array_equal(observed_input, x[rows])
        full_expert_output[rows] = output
    np.testing.assert_allclose(full_expert_output*arrays["gates"], arrays["branch_output"], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(h+arrays["branch_output"], arrays["residual_output"])
    arrays["expert_output"] = full_expert_output

    selected = int(requests[POSITION])
    expert = branch.experts[f"expert_{selected}"]
    projected = x[POSITION].astype(np.float64) @ expert.wi.weight.numpy().astype(np.float64).T
    activated = np.maximum(projected, 0)
    reconstructed = activated @ expert.wo.weight.numpy().astype(np.float64).T
    np.testing.assert_allclose(reconstructed, full_expert_output[POSITION], atol=2e-4, rtol=5e-5)
    with torch.inference_mode():
        actual_projected = expert.wi(torch.from_numpy(x[POSITION]))
        actual_activated = expert.act(actual_projected)
    arrays["expert_hidden_projection"] = actual_projected.numpy()
    arrays["expert_hidden_relu"] = actual_activated.numpy()
    np.testing.assert_allclose(projected, actual_projected.numpy(), atol=5e-5, rtol=4e-5)
    np.testing.assert_allclose(activated, actual_activated.numpy(), atol=5e-5, rtol=4e-5)

    requests_count = np.bincount(requests, minlength=expert_count)
    fractions = requests_count/count
    mean_p = p.astype(np.float64).mean(0)
    balance = float(expert_count*np.dot(fractions, mean_p))
    with torch.inference_mode():
        native_balance_helper = load_balancing_loss_func(probabilities, torch.from_numpy(requests[None]))
    np.testing.assert_allclose(balance, native_balance_helper.item(), atol=2e-6, rtol=0)
    arrays["request_counts"], arrays["request_fractions"], arrays["mean_probabilities"] = requests_count, fractions, mean_p

    reduced_capacity = math.ceil(count/expert_count)
    priority = np.cumsum(masks, axis=0)
    reduced_mask = masks*(priority <= reduced_capacity)
    admitted = reduced_mask.sum(-1).astype(bool)
    branch.router.expert_capacity = reduced_capacity
    try:
        with torch.inference_mode():
            native_mask, native_gates, reduced_logits = branch.router(torch.from_numpy(captured["router_input"]))
            reduced_output, _ = wrapper(torch.from_numpy(captured["residual"]), output_router_logits=True)
            reduced_branch, _ = branch(torch.from_numpy(captured["router_input"]))
    finally:
        branch.router.expert_capacity = capacity
    np.testing.assert_array_equal(native_mask[0].numpy(), reduced_mask)
    np.testing.assert_array_equal(reduced_logits[0].numpy(), logits)
    np.testing.assert_array_equal(native_gates[0].numpy(), arrays["gates"])
    np.testing.assert_array_equal(reduced_branch[0].numpy()[~admitted], np.zeros_like(x[~admitted]))
    np.testing.assert_array_equal(reduced_output[0].numpy()[~admitted], h[~admitted])
    np.testing.assert_allclose(reduced_branch[0].numpy(), arrays["branch_output"]*admitted[:, None], atol=2e-4, rtol=6e-5)
    arrays.update(reduced_mask=reduced_mask, reduced_admitted=admitted,
                  reduced_branch=reduced_branch[0].numpy(), reduced_residual_output=reduced_output[0].numpy())
    per_expert = sum(p.numel() for p in expert.parameters())
    bank_weights = sum(p.numel() for p in branch.experts.parameters())
    router_weights = sum(p.numel() for p in branch.router.parameters())
    assert per_expert == 2*width*model.config.d_ff and bank_weights == expert_count*per_expert
    assert router_weights == width*expert_count
    weights = hf_hub_download(MODEL, "pytorch_model.bin", revision=REVISION,
                              cache_dir=str(cache_dir), local_files_only=True, token=False)
    report = {"model": MODEL, "revision": REVISION, "weight_file": "pytorch_model.bin",
        "weights_sha256": sha256(weights), "loader": "weights_only=True; no remote Python; encoder weights only",
        "loaded_encoder_parameters": sum(p.numel() for p in model.parameters()),
        "loading": {"missing_keys": loading["missing_keys"], "unused_decoder_or_head_key_count": len(unused)},
        "text": TEXT, "pieces": pieces, "input_ids": ids.tolist(), "tokens_including_eos": count,
        "probe": {"array_position": POSITION, "encoder_block_index": layer_index,
                  "encoder_block_label": layer_index+1, "expert_index": selected, "expert_label": selected+1,
                  "displayed_feature_indices": list(FEATURES)},
        "configuration": {name: getattr(model.config, name) for name in
            ("d_model", "d_ff", "num_experts", "expert_capacity", "num_layers", "num_sparse_encoder_layers", "vocab_size")},
        "routing": {"requests_indices": requests.tolist(), "request_counts": requests_count.tolist(),
            "request_fractions": fractions.tolist(), "mean_probabilities": mean_p.tolist(), "raw_balance": balance,
            "probe_logits": logits[POSITION].tolist(), "probe_probabilities": p[POSITION].tolist(),
            "probe_gate": float(arrays["gates"][POSITION, 0]),
            "logit_reconstruction_max_error": float(np.max(np.abs(independent_logits-logits))),
            "probability_reconstruction_max_error": float(np.max(np.abs(softmax64(independent_logits)-p)))},
        "expert": {"first_output": float(full_expert_output[POSITION, 0]),
            "first_gated_update": float(arrays["branch_output"][POSITION, 0]),
            "first_residual_input": float(h[POSITION, 0]), "first_residual_output": float(arrays["residual_output"][POSITION, 0]),
            "relu_zero_count": int((actual_activated.numpy() == 0).sum()),
            "output_reconstruction_max_error": float(np.max(np.abs(reconstructed-full_expert_output[POSITION]))),
            "residual_addition_matches_exactly": True},
        "capacity_experiment": {"published_capacity": capacity, "reduced_capacity": reduced_capacity,
            "input_and_router_logits_unchanged": True, "admitted_counts": reduced_mask.sum(0).tolist(),
            "overflow_array_positions": np.flatnonzero(~admitted).tolist(), "overflow_fraction": float((~admitted).mean()),
            "reserved_slots": reduced_capacity*expert_count,
            "slot_utilization": float(admitted.sum()/(reduced_capacity*expert_count)),
            "overflow_branch_exactly_zero": True, "overflow_residual_exactly_preserved": True},
        "parameters": {"one_expert": per_expert, "router": router_weights, "stored_sparse_branch": bank_weights+router_weights,
            "one_accepted_token": per_expert+router_weights, "distinct_requested_experts": len(expert_outputs),
            "this_input_active_matrix_entries": len(expert_outputs)*per_expert+router_weights}}
    del model
    return report, arrays


def figures(report, arrays, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    paths, count = [], report["tokens_including_eos"]

    def save(fig, name):
        path = output/name
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.12, facecolor="white")
        plt.close(fig)
        paths.append(path)

    def feature_matrix(ax, data, labels, title):
        scale = np.max(np.abs(data)) or 1
        ax.imshow(data, cmap="PuOr_r", vmin=-scale, vmax=scale, aspect="auto")
        ax.set_xticks(range(8), FEATURES)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Feature array index (first eight shown)")
        ax.set_title(title, loc="left", fontsize=10, pad=12)
        for i in range(len(labels)):
            for j in range(8):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(data[i,j]) > .67*scale else "#222222")

    fig, ax = plt.subplots(figsize=(8.4, 7.0), layout="constrained")
    p = arrays["probabilities"]
    im = ax.imshow(p, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(8), range(1,9))
    ax.set_yticks(range(count), [f"{i}: {piece}" for i,piece in enumerate(report["pieces"])])
    ax.set_xlabel("Expert label (native array index + 1)")
    ax.set_ylabel("Input array position and SentencePiece")
    ax.set_title("Switch-base-8: actual router probabilities in encoder block 2", loc="left", fontsize=12)
    for i in range(count):
        ax.add_patch(Rectangle((arrays["requests"][i]-.5, i-.5),1,1,fill=False,edgecolor="#d98a23",linewidth=1.5))
        for j in range(8):
            ax.text(j,i,f"{p[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if p[i,j]>.55 else "#222222")
    fig.colorbar(im, ax=ax, label="Probability across eight experts", shrink=.65)
    save(fig,"ch09_pretrained_router.png")

    fig, axes = plt.subplots(2,1,figsize=(8.4,5.4),layout="constrained")
    feature_matrix(axes[0],np.stack((arrays["expert_hidden_projection"][:8],arrays["expert_hidden_relu"][:8])),
                   ["Up projection", "After ReLU"],"(a) Selected expert: first 8 of 3,072 intermediate features")
    row = POSITION
    feature_matrix(axes[1],np.stack([arrays[k][row,:8] for k in ("expert_output","branch_output","residual","residual_output")]),
                   ["Expert output", "Gated update", "Residual input", "Residual output"],
                   f"(b) Expert {report['probe']['expert_label']}; gate = {report['routing']['probe_gate']:.6f}; width 768")
    save(fig,"ch09_pretrained_expert.png")

    fig, axes = plt.subplots(2,1,figsize=(8.4,5.5),gridspec_kw={"height_ratios":[1.5,1]},layout="constrained")
    idx=np.arange(8); cap=report["capacity_experiment"]["reduced_capacity"]
    axes[0].bar(idx-.18,arrays["request_counts"],width=.36,color="#64748b",label="Requests (and admissions at capacity 64)")
    bars=axes[0].bar(idx+.18,arrays["reduced_mask"].sum(0),width=.36,color="#178473",label=f"Admissions at capacity {cap}")
    axes[0].bar_label(bars,padding=3,fontsize=8)
    axes[0].axhline(cap,color="#cb8129",linestyle="--",linewidth=1.2,label=f"Reduced per-expert limit: {cap}")
    axes[0].set_xticks(idx,idx+1)
    axes[0].set_xlabel("Expert label")
    axes[0].set_ylabel("Token rows")
    axes[0].set_title("(a) Same learned router, same input; change only the capacity limit",loc="left",fontsize=11)
    axes[0].legend(fontsize=8)
    accepted=np.stack((np.ones(count),arrays["reduced_admitted"]))
    from matplotlib.colors import ListedColormap
    axes[1].imshow(accepted,cmap=ListedColormap(["#e6a052","#4e9f8b"]),vmin=0,vmax=1,aspect="auto")
    axes[1].set_xticks(range(count),range(count),fontsize=8)
    axes[1].set_yticks([0,1],["Published limit 64",f"Reduced limit {cap}"])
    axes[1].set_xlabel("Input array position; orange = overflow, green = admitted")
    axes[1].set_title("(b) Overflow skips this expert update and retains the residual",loc="left",fontsize=11)
    save(fig,"ch09_pretrained_capacity.png")

    fig, axes=plt.subplots(2,1,figsize=(8.4,5.8),layout="constrained")
    axes[0].bar(idx-.18,arrays["request_fractions"],width=.36,color="#2985ad",label="Hard request fraction f")
    axes[0].bar(idx+.18,arrays["mean_probabilities"],width=.36,color="#cf8a39",label="Mean router probability P")
    axes[0].set_xticks(idx,idx+1)
    axes[0].set_xlabel("Expert label")
    axes[0].set_ylabel("Fraction / mean probability")
    axes[0].set_title(f"(a) Before filtering: raw balance term = {report['routing']['raw_balance']:.5f}",loc="left",fontsize=11)
    axes[0].legend(fontsize=8)
    params=report["parameters"]
    values=np.array([params[k] for k in ("stored_sparse_branch","one_accepted_token","this_input_active_matrix_entries")])/1e6
    bars=axes[1].barh(["Stored branch", "One accepted token", "This complete input"],values,color=["#64748b","#178473","#2985ad"])
    axes[1].invert_yaxis(); axes[1].bar_label(bars,fmt="%.4f",padding=4,fontsize=9)
    axes[1].set_xlim(0,values.max()*1.2)
    axes[1].set_xlabel("Million matrix weights in this sparse branch (including router)")
    axes[1].set_title("(b) Stored weights, per-token use, and batch use are different counts",loc="left",fontsize=11)
    save(fig,"ch09_pretrained_balance_storage.png")
    return{p.name:sha256(p) for p in paths}


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path("output/ch09-pretrained"))
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    report,arrays=observe(args.cache_dir,args.offline)
    report["execution"]="Frozen CPU float32 encoder inference; one thread; eval mode; no training or sampling"
    report["selection"]="Fixed text, first sparse encoder block, array position 2, and first eight features chosen before inspection"
    report["software"]={"python":platform.python_version(),**{n:importlib.metadata.version(n) for n in
        ("torch","transformers","tokenizers","huggingface_hub","numpy","matplotlib","Pillow")}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    archive=args.output_dir/"observations.npz";np.savez_compressed(archive,**arrays)
    report["artifacts"]={archive.name:sha256(archive),**figures(report,arrays,args.output_dir)}
    report["source_sha256"]=sha256(__file__)
    (args.output_dir/"PRETRAINED_FIGURES_RUN.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
