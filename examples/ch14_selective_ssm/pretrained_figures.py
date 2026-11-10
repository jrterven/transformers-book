"""Observe selective memory and SSD with two released Mamba checkpoints.

Use the installed Transformers CPU reference paths, rather than a CUDA
kernel, and compare Mamba-2 with the authors' hash-checked SSD function.
All probes, channels, heads, and boundaries are fixed before inspection.
"""

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform

import numpy as np

from .model import causal_decay, selective_scan, ssd_head_chunked, ssd_head_recurrent

MAMBA_ID = "state-spaces/mamba-130m-hf"
MAMBA_REV = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"
MAMBA2_ID = "state-spaces/mamba2-130m"
MAMBA2_REV = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
CODE_REV = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
SSD_SHA256 = "6442a9fe25a6792258df657392ab1fab4dd873ee777d9dd5783cc76030218e07"
TEXT = ("The parcel arrived in Paris. The weather was cloudy. "
        "The parcel moved to Rome. The driver checked the map. ") * 40
LENGTH = 384
DECODE = 2
MUTATION = 191
PROBE = 95
CHANNELS = (0, 1, 2)
HEAD = 0
CHUNK = 32


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def official_ssd(source_dir, offline):
    """Import only the two reviewed mathematical functions, without CUDA imports."""
    import requests
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat

    path = source_dir / "ssd_minimal.py"
    source_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if offline:
            raise FileNotFoundError("Cache the pinned ssd_minimal.py before --offline")
        url = f"https://raw.githubusercontent.com/state-spaces/mamba/{CODE_REV}/mamba_ssm/modules/ssd_minimal.py"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        assert hashlib.sha256(response.content).hexdigest() == SSD_SHA256
        path.write_bytes(response.content)
    assert sha256(path) == SSD_SHA256, "Upstream SSD source differs from reviewed code"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"segsum", "ssd_minimal_discrete"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    namespace = {"torch": torch, "F": F, "rearrange": rearrange, "repeat": repeat}
    # The function bodies are unchanged. Omit only module imports and the
    # unrelated CUDA comparison test; no repository-wide import is executed.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ssd_minimal_discrete"]


def cache_payload(cache):
    return {"ssm": sum(x.numel() * x.element_size() for x in cache.ssm_states),
            "conv": sum(x.numel() * x.element_size() for x in cache.conv_states)}


def observe(cache_dir, source_dir, offline):
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mamba2Config, Mamba2ForCausalLM
    from transformers.models.mamba import modeling_mamba
    from transformers.models.mamba2 import modeling_mamba2

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    ssd_reference = official_ssd(source_dir, offline)
    common = dict(cache_dir=str(cache_dir), token=False, local_files_only=offline)
    tokenizer = AutoTokenizer.from_pretrained(MAMBA_ID, revision=MAMBA_REV,
                                             trust_remote_code=False, **common)
    all_ids = tokenizer(TEXT, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = all_ids[:, :LENGTH+DECODE].clone()
    replacement = tokenizer.encode(" blue", add_special_tokens=False)
    assert len(replacement) == 1 and replacement[0] != int(ids[0, MUTATION])
    arrays, checks, models = {}, {}, {}
    arrays["ids"] = ids[0].numpy()
    for name in ("mamba", "mamba2"):
        if name == "mamba":
            model, loading = AutoModelForCausalLM.from_pretrained(
                MAMBA_ID, revision=MAMBA_REV, trust_remote_code=False,
                use_safetensors=True, output_loading_info=True, **common)
            assert not any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
            path = hf_hub_download(MAMBA_ID, "model.safetensors", revision=MAMBA_REV, **common)
            code_path = inspect.getfile(modeling_mamba)
        else:
            path = hf_hub_download(MAMBA2_ID, "pytorch_model.bin", revision=MAMBA2_REV, **common)
            weights = torch.load(path, map_location="cpu", weights_only=True)
            assert torch.equal(weights["backbone.embedding.weight"], weights["lm_head.weight"])
            # The released config uses the original repository's field names.
            # Match its default Mamba2 dimensions, verified against every tensor.
            config = Mamba2Config(hidden_size=768, num_hidden_layers=24,
                                  num_heads=24, head_dim=64, state_size=128,
                                  n_groups=1, vocab_size=50288, conv_kernel=4,
                                  chunk_size=CHUNK, tie_word_embeddings=True)
            model = Mamba2ForCausalLM(config)
            # The library's built-in load hook renames embedding -> embeddings.
            loading = model.load_state_dict(weights, strict=True)
            assert not loading.missing_keys and not loading.unexpected_keys
            assert model.lm_head.weight is model.backbone.embeddings.weight
            assert torch.equal(weights["lm_head.weight"], model.lm_head.weight.to(weights["lm_head.weight"].dtype))
            del weights
            code_path = inspect.getfile(modeling_mamba2)
        model.eval().float().requires_grad_(False)
        mixer = model.backbone.layers[0].mixer
        captured = {}
        hooks = [mixer.register_forward_pre_hook(
            lambda mod, args: captured.update(input=args[0].detach().clone())),
            mixer.out_proj.register_forward_pre_hook(
                lambda mod, args: captured.update(before_output=args[0].detach().clone()))]
        if name == "mamba2":
            hooks.append(mixer.norm.register_forward_pre_hook(
                lambda mod, args: captured.update(before_norm=args[0].detach().clone())))
        with torch.inference_mode():
            try:
                result = model(ids[:, :LENGTH], use_cache=True)
            finally:
                for handle in hooks:
                    handle.remove()
            baseline_logits = result.logits[0].numpy().copy()
            cache = result.cache_params
            payloads = [{"length": LENGTH, **cache_payload(cache)}]
            native_last_state = cache.ssm_states[0].clone()
            cached_logits = []
            for t in range(LENGTH, LENGTH+DECODE):
                step = model(ids[:, t:t+1], cache_params=cache,
                             cache_position=torch.tensor([t]), use_cache=True)
                cached_logits.append(step.logits[0].numpy().copy())
                payloads.append({"length": t+1, **cache_payload(cache)})
            whole = model(ids, use_cache=False).logits[0].numpy().copy()
            cached_logits = np.concatenate(cached_logits)
            arrays[name+"_cached_logit_error"] = np.max(np.abs(cached_logits - whole[LENGTH:]), axis=-1)
            checks[name+"_cached_vs_full_logits"] = float(arrays[name+"_cached_logit_error"].max())
            np.testing.assert_allclose(cached_logits, whole[LENGTH:], atol=1e-3, rtol=0)
            checks[name+"_cached_argmax_agreement"] = bool(np.array_equal(cached_logits.argmax(-1), whole[LENGTH:].argmax(-1)))
            assert checks[name+"_cached_argmax_agreement"]
            changed_ids = ids[:, :LENGTH].clone()
            changed_ids[0, MUTATION] = replacement[0]
            changed_logits = model(changed_ids, use_cache=False).logits[0].numpy().copy()
            arrays[name+"_logit_change"] = np.max(np.abs(changed_logits - baseline_logits), axis=-1)
            checks[name+"_earlier_prefix_change"] = float(arrays[name+"_logit_change"][:MUTATION].max())
            checks[name+"_affected_suffix_change"] = float(arrays[name+"_logit_change"][MUTATION:].max())
            assert checks[name+"_earlier_prefix_change"] < 1e-4
            assert checks[name+"_affected_suffix_change"] > 1e-3
            for length in (64, 128):
                short = model.backbone(ids[:, :length], use_cache=True)
                payloads.append({"length": length, **cache_payload(short.cache_params)})
            payloads.sort(key=lambda row: row["length"])
            assert len({(r["ssm"],r["conv"]) for r in payloads}) == 1
            if name == "mamba":
                projected = mixer.in_proj(captured["input"]).transpose(1, 2)
                branch, gate = projected.chunk(2, dim=1)
                u = F.silu(mixer.conv1d(branch)[..., :LENGTH])
                parameters = mixer.x_proj(u.transpose(1, 2))
                low, b, c = parameters.split([48, 16, 16], dim=-1)
                dt = F.softplus(mixer.dt_proj(low))[0].T
                rates = -torch.exp(mixer.A_log.float())
                data_u = u[0, 0].numpy().astype(np.float64)
                data_b, data_c = b[0].numpy().astype(np.float64), c[0].numpy().astype(np.float64)
                delta = dt[0].numpy().astype(np.float64)
                a = np.exp(delta[:, None] * rates[0].numpy().astype(np.float64))
                beff = delta[:, None] * data_b
                core, states = selective_scan(data_u, a, beff, data_c)
                full_before_projection = ((torch.tensor(core, dtype=torch.float64)
                    + data_u * float(mixer.D[0])) * F.silu(gate)[0, 0].double()).numpy()
                checks["mamba_first_channel_output_reconstruction"] = float(np.max(np.abs(
                    full_before_projection - captured["before_output"][0, :, 0].numpy())))
                np.testing.assert_allclose(full_before_projection, captured["before_output"][0, :, 0].numpy(), atol=3e-5, rtol=0)
                checks["mamba_final_state_reconstruction"] = float(np.max(np.abs(states[-1] - native_last_state[0, 0].numpy())))
                np.testing.assert_allclose(states[-1], native_last_state[0, 0].numpy(), atol=3e-5, rtol=0)
                matrix = np.zeros((LENGTH, LENGTH))
                for j in range(16):
                    matrix += data_c[:, j, None] * causal_decay(a[:, j]) * beff[None, :, j]
                np.testing.assert_allclose(matrix @ data_u, core, atol=1e-11, rtol=0)
                arrays.update(mamba_steps=dt[list(CHANNELS)].numpy(), mamba_retentions=a,
                              mamba_state=states, mamba_matrix=matrix, mamba_input=data_u,
                              mamba_core_output=core, mamba_probe_contributions=matrix[PROBE]*data_u)
            else:
                projection = mixer.in_proj(captured["input"])
                gate, branch, raw_dt = projection.split([1536, 1792, 24], dim=-1)
                convolved = F.silu(mixer.conv1d(branch.transpose(1, 2))[..., :LENGTH].transpose(1, 2))
                u, b, c = convolved.split([1536, 128, 128], dim=-1)
                u = u.reshape(1, LENGTH, 24, 64)
                dt = F.softplus(raw_dt + mixer.dt_bias)
                loga = dt * (-torch.exp(mixer.A_log.float()))
                bb, cc = [x.reshape(1, LENGTH, 1, 128).expand(-1, -1, 24, -1).contiguous() for x in (b, c)]
                reference, final = ssd_reference(u * dt[..., None], loga, bb, cc, block_len=CHUNK)
                before_norm = captured["before_norm"].reshape(1, LENGTH, 24, 64)
                reference_with_D = reference + u * mixer.D[None, None, :, None]
                checks["mamba2_official_ssd_vs_library_all_heads"] = float((reference_with_D - before_norm).abs().max())
                np.testing.assert_allclose(reference_with_D.numpy(), before_norm.numpy(), atol=2e-4, rtol=0)
                checks["mamba2_official_ssd_final_state"] = float((final - native_last_state).abs().max())
                np.testing.assert_allclose(final.numpy(), native_last_state.numpy(), atol=2e-4, rtol=0)
                uu = u[0, :, HEAD].numpy().astype(np.float64)
                aa = np.exp(loga[0, :, HEAD].numpy().astype(np.float64))
                beff = dt[0, :, HEAD, None].numpy().astype(np.float64) * b[0].numpy().astype(np.float64)
                cc64 = c[0].numpy().astype(np.float64)
                recurrence, states = ssd_head_recurrent(uu, aa, beff, cc64)
                chunked, last = ssd_head_chunked(uu, aa, beff, cc64, chunk_size=37)
                decay = causal_decay(aa)
                content = cc64 @ beff.T
                matrix = decay * content
                np.testing.assert_allclose(matrix @ uu, recurrence, atol=1e-10, rtol=0)
                np.testing.assert_allclose(chunked, recurrence, atol=1e-10, rtol=0)
                np.testing.assert_allclose(last, states[-1], atol=1e-10, rtol=0)
                native_core = before_norm[0, :, HEAD].numpy().astype(np.float64) - uu * float(mixer.D[HEAD])
                checks["mamba2_numpy_head_vs_library"] = float(np.max(np.abs(recurrence - native_core)))
                np.testing.assert_allclose(recurrence, native_core, atol=2e-4, rtol=0)
                local = np.zeros_like(recurrence)
                for start in range(0, LENGTH, CHUNK):
                    sl = slice(start, start+CHUNK)
                    local[sl] = matrix[sl, sl] @ uu[sl]
                boundary_rms = np.sqrt(np.mean(states[CHUNK-1::CHUNK]**2, axis=2)).T
                arrays.update(mamba2_decay=decay, mamba2_content=content, mamba2_matrix=matrix,
                              mamba2_head_output=recurrence, mamba2_head_native=native_core,
                              mamba2_head_local=local, mamba2_head_incoming=recurrence-local,
                              mamba2_boundary_rms=boundary_rms, mamba2_steps=dt[0].numpy().T)
            models[name] = {
                "repository": MAMBA_ID if name=="mamba" else MAMBA2_ID,
                "revision": MAMBA_REV if name=="mamba" else MAMBA2_REV,
                "weights_sha256": sha256(path), "runtime_source_sha256": sha256(code_path),
                "unique_parameters": sum(p.numel() for p in model.parameters()),
                "state_width": model.config.state_size, "features": 1536, "layers": 24,
                "vocab_size": model.config.vocab_size, "cache_payloads": payloads,
                "cache_ssm_layer_shape": list(native_last_state.shape),
                "cache_conv_layer_shape": list(cache.conv_states[0].shape),
                "loaded_without_missing_or_unused_weights": True,
                "original_checkpoint_port": name=="mamba2",
            }
        print(name, "native model and reconstruction checks passed", flush=True)
        del model, mixer, result, cache, native_last_state, whole, baseline_logits, changed_logits, short
        gc.collect()
    record = {
        "models": models, "checks": checks, "prefill_tokens": LENGTH, "decode_tokens": DECODE,
        "channels_zero_based": CHANNELS, "head_zero_based": HEAD, "probe_zero_based": PROBE,
        "native_cpu_chunk_size": CHUNK, "independent_chunk_size": 37,
        "mutation_zero_based": MUTATION, "replacement_id": replacement[0],
        "original_piece": tokenizer.convert_ids_to_tokens(int(ids[0,MUTATION])),
        "replacement_piece": tokenizer.convert_ids_to_tokens(replacement[0]),
        "input_ids": ids[0].tolist(), "tokenizer_pieces": tokenizer.convert_ids_to_tokens(ids[0].tolist()),
        "input_text_recipe": TEXT[:len(TEXT)//40], "input_recipe_repetitions": 40,
        "tokenizer_repository": MAMBA_ID, "tokenizer_revision": MAMBA_REV,
        "source_revision": CODE_REV, "official_ssd_sha256": SSD_SHA256,
        "device": "cpu", "threads": 1, "native_dtype": "float32", "independent_dtype": "float64",
        "additional_training": False, "timing_benchmark": False,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch","transformers","numpy","matplotlib","einops","safetensors","huggingface-hub")},
        "platform": platform.platform(),
    }
    return arrays, record


def figures(data, record, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

    plt.rcParams.update({"font.size":11,"axes.titlesize":12,"axes.spines.top":False,
                         "axes.spines.right":False,"font.family":"DejaVu Sans","savefig.dpi":220})
    blue,orange,green="#24689b","#c66a2b","#41866d"
    def save(fig,name):
        fig.savefig(output/name,facecolor="white",bbox_inches="tight");plt.close(fig)
    pos=np.arange(1,LENGTH+1)
    fig,ax=plt.subplots(2,1,figsize=(10,4.7),layout="constrained",sharex=True)
    for i,col in enumerate([blue,orange,green]):ax[0].plot(pos[:96],data['mamba_steps'][i,:96],color=col,label=f"Channel {i+1}",lw=1.2)
    ax[0].set(ylabel="Learned step Δ",yscale="log",title="(a) Mamba block 1: input-dependent steps in three fixed channels");ax[0].legend(frameon=False,ncol=3)
    im=ax[1].imshow(data['mamba_retentions'][:96].T,aspect="auto",origin="lower",extent=(.5,96.5,.5,16.5),cmap="viridis",vmin=0,vmax=1)
    ax[1].set(xlabel="Token position",ylabel="State coordinate",title="(b) Channel 1: retention exp(Δ A) for all 16 coordinates")
    fig.colorbar(im,ax=ax[1],label="Retained fraction",fraction=.025)
    save(fig,"ch14_pretrained_mamba_controls.png")

    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    m=data['mamba_matrix'][:64,:64];bound=np.abs(m).max()
    signed_map=plt.get_cmap("RdBu_r").copy();signed_map.set_bad("#e4e7eb")
    future=np.triu(np.ones((64,64),dtype=bool),k=1)
    im=ax[0].imshow(np.ma.masked_where(future,m),cmap=signed_map,norm=SymLogNorm(linthresh=1e-3,vmin=-bound,vmax=bound),extent=(.5,64.5,64.5,.5),aspect="auto")
    ax[0].set(xlabel="Source token position",ylabel="Read position",title="(a) Frozen-control coefficients, channel 1")
    cb=fig.colorbar(im,ax=ax[0],label="Signed coefficient",fraction=.04)
    cb.set_ticks([-bound,-.01,0,.01,bound]);cb.set_ticklabels([f"{-bound:.2g}","−0.01","0","0.01",f"{bound:.2g}"])
    contribution=data['mamba_probe_contributions'][:PROBE+1]
    ax[1].bar(np.arange(1,PROBE+2),contribution,color=np.where(contribution>=0,blue,orange),width=.9)
    ax[1].axhline(0,color="0.6",lw=.6)
    ax[1].set(xlabel="Source token position",ylabel="Contribution to the SSM core output",title=f"(b) Source contributions to read {PROBE+1}")
    ax[1].text(.04,.94,f"Sum = {contribution.sum():.5g}",transform=ax[1].transAxes,va="top")
    save(fig,"ch14_pretrained_mamba_paths.png")

    fig,ax=plt.subplots(1,3,figsize=(10.8,3.7),layout="constrained")
    selections=[('mamba2_decay','(a) Causal decay L'),('mamba2_content','(b) Content products G'),('mamba2_matrix','(c) Coefficients M = L ⊙ G')]
    signed=max(np.abs(data[key][:64,:64]).max() for key in ['mamba2_content','mamba2_matrix'])
    for a,(key,title) in zip(ax,selections):
        m=data[key][:64,:64]
        cm=plt.get_cmap("viridis").copy() if key.endswith('decay') else signed_map
        cm.set_bad("#e4e7eb")
        display=m if key.endswith('content') else np.ma.masked_where(future,m)
        im=a.imshow(display,aspect="equal",extent=(.5,64.5,64.5,.5),cmap=cm,norm=None if key.endswith('decay') else SymLogNorm(linthresh=1e-3,vmin=-signed,vmax=signed),vmin=0 if key.endswith('decay') else None,vmax=1 if key.endswith('decay') else None)
        a.set(title=title,xlabel="Source position",ylabel="Read position")
        cb=fig.colorbar(im,ax=a,fraction=.045,orientation="horizontal",pad=.18)
        if key.endswith('decay'):
            cb.set_ticks([0,.5,1])
        else:
            cb.set_ticks([-signed,0,signed]);cb.set_ticklabels([f"{-signed:.2g}","0",f"{signed:.2g}"])
    save(fig,"ch14_pretrained_ssd_matrices.png")

    fig,ax=plt.subplots(2,1,figsize=(10,5),layout="constrained")
    sl=slice(0,96)
    ax[0].plot(pos[sl],data['mamba2_head_native'][sl,0],color="0.15",label="Native core total",lw=1.7)
    ax[0].plot(pos[sl],data['mamba2_head_local'][sl,0],color=blue,label="Local chunk from zero",lw=1.1)
    ax[0].plot(pos[sl],data['mamba2_head_incoming'][sl,0],color=orange,label="Incoming-state contribution",lw=1.1)
    for b in [32.5,64.5]:ax[0].axvline(b,color="0.6",ls="--",lw=.8)
    ax[0].set(xlabel="Token position",ylabel="Head 1, feature 1",title="(a) Mamba-2: local work plus carried history (32-token chunks)");ax[0].legend(frameon=False,fontsize=9,ncol=3)
    z=data['mamba2_boundary_rms']
    im=ax[1].imshow(z,origin="lower",aspect="auto",extent=(.5,12.5,.5,128.5),cmap="magma",norm=LogNorm(vmin=max(1e-8,z.min()),vmax=z.max()))
    ax[1].set(xlabel="Chunk boundary (after 32, 64, …, 384 tokens)",ylabel="State coordinate",title="(b) Carried state in head 1: RMS magnitude across its 64 features")
    fig.colorbar(im,ax=ax[1],label="RMS magnitude (log scale)",fraction=.025)
    save(fig,"ch14_pretrained_ssd_chunks.png")

    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    labels=[];ssm=[];conv=[]
    for name,label in [('mamba','Mamba'),('mamba2','Mamba-2')]:
        for row in record['models'][name]['cache_payloads']:
            if row['length'] in [64,128,384]:
                labels.append(str(row['length']));ssm.append(row['ssm']/2**20);conv.append(row['conv']/2**20)
    xx=np.array([0,1,2,4,5,6]);ax[0].bar(xx,ssm,color=blue,label="SSM state");ax[0].bar(xx,conv,bottom=ssm,color=green,label="Short-convolution buffer")
    ax[0].set(xticks=xx,xticklabels=labels,ylabel="Cache tensor payload (MiB)",title="(a) Measured persistent cache arrays")
    ax[0].text(1,-.14,'Mamba',transform=ax[0].get_xaxis_transform(),ha='center')
    ax[0].text(5,-.14,'Mamba-2',transform=ax[0].get_xaxis_transform(),ha='center')
    ax[0].set_xlabel("Consumed prefix length (tokens)",labelpad=30)
    ax[0].legend(frameon=False,fontsize=9)
    for i,(name,label,col) in enumerate([('mamba','Mamba',blue),('mamba2','Mamba-2',orange)]):
        ax[1].bar(np.arange(2)+(i-.5)*.3,data[name+'_cached_logit_error'],width=.3,color=col,label=label)
    ax[1].set(xticks=[0,1],xticklabels=['Token 385','Token 386'],ylabel="Maximum vocabulary-logit discrepancy",title="(b) Cached step versus full-prefix calculation")
    ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax[1].legend(frameon=False)
    save(fig,"ch14_pretrained_cache_equivalence.png")

    fig,ax=plt.subplots(2,2,figsize=(10,5.5),layout="constrained")
    for row,(name,label,col) in enumerate([('mamba','Mamba',blue),('mamba2','Mamba-2',orange)]):
        change=data[name+'_logit_change'];ax[row,0].plot(pos,change,color=col,lw=1.1)
        ax[row,0].axvline(MUTATION+1,color="0.35",ls="--",lw=1);ax[row,0].axvspan(1,MUTATION,color=green,alpha=.1)
        ax[row,0].set(title=label+": one changed token at position 192",ylabel="Maximum absolute logit change",xlabel="Output position")
        prefix=change[:MUTATION];ax[row,1].plot(pos[:MUTATION],prefix,color=col,lw=1)
        if np.max(prefix)==0:ax[row,1].set_ylim(-1e-6,1e-6);ax[row,1].text(.08,.82,'Exact zero in this run',transform=ax[row,1].transAxes)
        ax[row,1].set(title="Earlier prefix, expanded numerical scale",ylabel="Maximum absolute logit change",xlabel="Output position")
        ax[row,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    save(fig,"ch14_pretrained_causality.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    arrays,record=observe(args.cache_dir,args.source_dir,args.offline)
    np.savez_compressed(args.output_dir/'observations.npz',**arrays)
    figures(arrays,record,args.output_dir)
    record['figures']={p.name:sha256(p) for p in sorted(args.output_dir.glob('ch14_*.png'))}
    (args.output_dir/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record['checks'],indent=2))


if __name__=='__main__':
    main()
