"""Inspect published S4 and HyenaDNA weights with their pinned native code.

Only the four hash-checked source files listed below are imported. The S4
adapter replaces a distributed-logging decorator with an identity decorator;
its numerical implementation is unchanged. No model is trained by this run.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import types

import numpy as np

S4_ID = "krandiash/sashimi-release"
S4_REV = "83302f50c90d48d4b9ed4896683eacc4b47a824f"
S4_CODE = "e757cef57d89e448c413de7325ed5601aceaac13"
HYENA_ID = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
HYENA_REV = "e8c1effa8673814e257e627d2e1eda9ea5a373f6"
SOURCE_HASHES = {
    "s4.py": "9f3418b396ec9ccbea30d1f1de92b55e631a3380ebbb19257a73f0219e2a23a6",
    "configuration_hyena.py": "29cb6cae67f5010e279d67349e688663043f0bd6791a09c36d96da8cac3668a6",
    "modeling_hyena.py": "d78029cacea2c9259f1293fdb50b60f90872089469185e27d5c769086c475607",
    "tokenization_hyena.py": "d424da2f794958b0360e033bbf5edc9ee4e3e3126e883093621348f07761e408",
}
CHANNELS = (0, 1, 2)
LENGTH = 512
DNA = ("ACGTTGCAAGTCGATCCGTA" * 26)[:LENGTH]
MUTATION = 255  # zero-based: printed position 256


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources(cache, source_dir, offline):
    from huggingface_hub import hf_hub_download
    import requests

    source_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCE_HASHES.items():
        target = source_dir / name
        if not target.exists():
            if name == "s4.py":
                if offline:
                    raise FileNotFoundError("Cache the pinned s4.py before using --offline")
                url = f"https://raw.githubusercontent.com/state-spaces/s4/{S4_CODE}/models/s4/s4.py"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                data = response.content
            else:
                data = Path(hf_hub_download(
                    HYENA_ID, name, revision=HYENA_REV, cache_dir=str(cache),
                    token=False, local_files_only=offline)).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, name
            target.write_bytes(data)
        assert sha256(target) == digest, (name, "Source differs from reviewed revision")
    text = (source_dir / "s4.py").read_text()
    old = "from pytorch_lightning.utilities import rank_zero_only"
    assert text.count(old) == 1
    adapter = source_dir / "s4_logging_adapter.py"
    adapter.write_text(text.replace(old, "def rank_zero_only(fn):\n    return fn"))
    spec = importlib.util.spec_from_file_location("_ch13_audited_s4", adapter)
    s4 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = s4
    spec.loader.exec_module(s4)
    package = types.ModuleType("_ch13_audited_hyena")
    package.__path__ = [str(source_dir)]
    sys.modules[package.__name__] = package
    config = importlib.import_module(package.__name__ + ".configuration_hyena")
    hyena = importlib.import_module(package.__name__ + ".modeling_hyena")
    tokenizer = importlib.import_module(package.__name__ + ".tokenization_hyena")
    return s4, config, hyena, tokenizer, sha256(adapter)


def observe(cache, source_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    s4, config_module, hyena_module, token_module, adapter_hash = sources(cache, source_dir, offline)
    common = dict(cache_dir=str(cache), token=False, local_files_only=offline)
    s4_path = hf_hub_download(S4_ID, "checkpoints/sashimi_sc09_v4.pt", revision=S4_REV, **common)
    whole = torch.load(s4_path, map_location="cpu", weights_only=True)
    prefix = "model.c_layers.0.layer.layer.kernel."
    weights = {k[len(prefix):]: v for k, v in whole.items() if k.startswith(prefix)}
    kernel = s4.SSMKernelDPLR(
        d_model=256, d_state=64, n_ssm=1, channels=1,
        backend="naive", disc="bilinear", dt_transform="exp", verbose=False)
    kernel.load_state_dict(weights, strict=True)
    kernel.eval().requires_grad_(False)
    arrays, checks = {}, {}
    u = np.zeros(LENGTH)
    u[[0, 128, 256, 384]] = [1, -0.5, 0.75, 1.25]
    with torch.inference_mode():
        native_k, _ = kernel(L=LENGTH)
        arrays["s4_kernel"] = native_k[0, list(CHANNELS)].numpy().copy()
        kernel._setup_step(mode="dense")
        state = kernel.default_state(1)
        outputs, states = [], []
        for value in u:
            y, state = kernel.step(torch.full((1, 256), float(value)), state)
            outputs.append(y[0, 0, list(CHANNELS)].numpy().copy())
            states.append(state[0, 0, :32].numpy().copy())
        arrays["s4_native_output"] = np.stack(outputs).T
        arrays["s4_native_state"] = np.stack(states)
        arrays["s4_input"] = u

        # Independent complex128 reconstruction of the full conjugate-pair
        # system and its finite-length-corrected readout, for channels 0--2.
        dt, diag, b, c_tilde, p, _ = kernel._get_params()
        dt = dt.numpy()[:, 0].astype(np.float64)
        full = lambda a: np.concatenate([a, a.conj()], axis=-1).astype(np.complex128)
        lam = full(diag.numpy()[0])
        pp = full(p.numpy()[0, 0])
        aa = np.diag(lam) - np.outer(pp, pp.conj())
        bb = full(b.numpy()[0, 0])
        eigs, direct, independent = [], [], []
        for channel in CHANNELS:
            left = np.eye(64) - dt[channel] * aa / 2
            abar = np.linalg.solve(left, np.eye(64) + dt[channel] * aa / 2)
            bbar = np.linalg.solve(left, dt[channel] * bb)
            c = np.linalg.solve(
                (np.eye(64) - np.linalg.matrix_power(abar, int(weights["l_kernel"]))).T,
                full(c_tilde.numpy()[0, channel]))
            eigs.append(np.linalg.eigvals(abar))
            x, taps = bbar.copy(), []
            for _ in range(LENGTH):
                taps.append(c @ x)
                x = abar @ x
            taps = np.asarray(taps)
            assert np.max(np.abs(taps.imag)) < 1e-10
            independent.append(taps.real)
            direct.append(np.convolve(u, taps.real)[:LENGTH])
        arrays["s4_poles"] = np.stack(eigs)
        arrays["s4_reconstructed_kernel"] = np.stack(independent)
        arrays["s4_direct_output"] = np.stack(direct)
        arrays["s4_dt"] = dt[list(CHANNELS)]
        fft_size = 2 * LENGTH - 1
        arrays["s4_fft_output"] = np.fft.irfft(
            np.fft.rfft(u, n=fft_size)[None] *
            np.fft.rfft(arrays["s4_reconstructed_kernel"], n=fft_size), n=fft_size)[:, :LENGTH]
        for name, lhs, rhs, tol in [
            ("s4_native_vs_reconstructed_filter", arrays["s4_kernel"], arrays["s4_reconstructed_kernel"], 1e-5),
            ("s4_native_scan_vs_convolution", arrays["s4_native_output"], arrays["s4_direct_output"], 1e-5),
            ("s4_fft_vs_direct", arrays["s4_fft_output"], arrays["s4_direct_output"], 1e-12),
        ]:
            checks[name] = float(np.max(np.abs(lhs - rhs)))
            np.testing.assert_allclose(lhs, rhs, atol=tol, rtol=0)
        assert np.all(np.abs(arrays["s4_poles"]) < 1)

    config_path = hf_hub_download(HYENA_ID, "config.json", revision=HYENA_REV, **common)
    model_path = hf_hub_download(HYENA_ID, "model.safetensors", revision=HYENA_REV, **common)
    config = config_module.HyenaConfig.from_json_file(config_path)
    model = hyena_module.HyenaDNAForCausalLM(config)
    loaded = load_file(model_path, device="cpu")
    loading = model.load_state_dict(loaded, strict=False)
    expected_aliases = [f"hyena.backbone.layers.{i}.mixer.filter_fn.implicit_filter.{j}.freq"
                        for i in range(2) for j in (3, 5)]
    assert sorted(loading.missing_keys) == sorted(expected_aliases)
    assert not loading.unexpected_keys
    for layer in model.hyena.backbone.layers:
        net = layer.mixer.filter_fn.implicit_filter
        assert net[1].freq is net[3].freq is net[5].freq
    model.eval().float().requires_grad_(False)
    tokenizer = token_module.HyenaDNATokenizer(model_max_length=1026)
    ids = torch.tensor([tokenizer.encode(DNA, add_special_tokens=False)])
    changed = ids.clone()
    changed[0, MUTATION] = 7 if ids[0, MUTATION] != 7 else 8
    mixer = model.hyena.backbone.layers[0].mixer
    captured = {}
    hooks = [
        mixer.short_filter.register_forward_hook(
            lambda mod, args, out: captured.update(short=out[..., :LENGTH].detach().clone())),
        mixer.out_proj.register_forward_pre_hook(
            lambda mod, args: captured.update(before_out=args[0].detach().clone())),
    ]
    with torch.inference_mode():
        try:
            logits = model(ids).logits[0].numpy().copy()
        finally:
            for h in hooks:
                h.remove()
        a, g, v = captured["short"].split(128, dim=1)
        filt = mixer.filter_fn.filter(LENGTH)[0].T
        bias = mixer.filter_fn.bias
        arrays["hyena_kernel"] = filt[list(CHANNELS)].numpy().copy()
        arrays["hyena_a"] = a[0, 0].numpy().copy()
        arrays["hyena_g"] = g[0, 0].numpy().copy()
        arrays["hyena_v"] = v[0, 0].numpy().copy()
        arrays["hyena_gated"] = arrays["hyena_g"] * arrays["hyena_v"]
        arrays["hyena_native_before_projection"] = captured["before_out"][0].T.numpy().copy()
        av, gv, vv = [x[0].numpy().astype(np.float64) for x in (a, g, v)]
        kk = filt.numpy().astype(np.float64)
        dd = bias.numpy().astype(np.float64)
        ww = np.stack([np.convolve(gv[i] * vv[i], kk[i])[:LENGTH]
                       + dd[i] * gv[i] * vv[i] for i in range(128)])
        reconstructed = av * ww
        arrays["hyena_filtered"] = ww[0]
        arrays["hyena_output"] = reconstructed[0]
        checks["hyena_all_channels_native_vs_direct"] = float(np.max(np.abs(
            reconstructed - arrays["hyena_native_before_projection"])))
        np.testing.assert_allclose(reconstructed, arrays["hyena_native_before_projection"], atol=3e-5, rtol=0)
        # A frozen-gate coefficient matrix acts on v, not on token IDs and
        # not on all input-derived branches simultaneously.
        n = 48
        lag = np.arange(n)[:, None] - np.arange(n)[None, :]
        toeplitz = np.where(lag >= 0, kk[0, np.maximum(lag, 0)], 0.0)
        toeplitz += np.eye(n) * dd[0]
        arrays["hyena_frozen_matrix"] = av[0, :n, None] * toeplitz * gv[0, None, :n]
        np.testing.assert_allclose(arrays["hyena_frozen_matrix"] @ vv[0, :n], reconstructed[0, :n], atol=1e-10, rtol=0)
        mutated_logits = model(changed).logits[0].numpy().copy()
        arrays["hyena_logit_change"] = np.max(np.abs(logits - mutated_logits), axis=-1)
        checks["hyena_prefix_max_logit_change"] = float(arrays["hyena_logit_change"][:MUTATION].max())
        checks["hyena_suffix_max_logit_change"] = float(arrays["hyena_logit_change"][MUTATION:].max())
        assert checks["hyena_prefix_max_logit_change"] < 2e-5
        assert checks["hyena_suffix_max_logit_change"] > 1e-3
        # An independent shorter prompt checks length cropping and normalization.
        short_logits = model(ids[:, :MUTATION]).logits[0].numpy()
        checks["hyena_short_vs_long_prefix"] = float(np.max(np.abs(short_logits - logits[:MUTATION])))
        np.testing.assert_allclose(short_logits, logits[:MUTATION], atol=2e-5, rtol=0)
        arrays["hyena_logits"] = logits
        arrays["hyena_mutated_logits"] = mutated_logits
        arrays["hyena_ids"] = ids[0].numpy()
    record = {
        "s4": {"repository": S4_ID, "revision": S4_REV, "file": "checkpoints/sashimi_sc09_v4.pt",
               "sha256": sha256(s4_path), "source_revision": S4_CODE,
               "layer": "model.c_layers.0.layer.layer.kernel", "features": 256, "real_state_dimension": 64,
               "stored_complex_coordinates": 32, "n_ssm": 1, "stored_readout_length": int(weights["l_kernel"]),
               "direct_D_omitted": True, "whole_model_executed": False,
               "adapter_sha256": adapter_hash, "dt_selected": arrays["s4_dt"].tolist()},
        "hyena": {"repository": HYENA_ID, "revision": HYENA_REV, "sha256": sha256(model_path),
                  "unique_parameters": sum(p.numel() for p in model.parameters()), "layers": 2, "width": 128,
                  "order": 2, "whole_model_executed": True, "missing_keys_are_verified_shared_aliases": expected_aliases,
                  "sequence": DNA, "ids": ids[0].tolist(), "changed_position_zero_based": MUTATION,
                  "original_base": DNA[MUTATION], "changed_base": tokenizer.convert_ids_to_tokens(int(changed[0, MUTATION])),
                  "channel_zero_D": float(dd[0])},
        "channels_zero_based": CHANNELS, "length": LENGTH, "training": False, "device": "cpu",
        "threads": 1, "native_dtype": "float32", "independent_dtype": "float64/complex128",
        "reviewed_source_sha256": SOURCE_HASHES, "checks": checks,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "numpy", "matplotlib", "scipy", "einops", "safetensors", "huggingface-hub")},
        "platform": platform.platform(),
    }
    return arrays, record


def figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220, "font.family": "DejaVu Sans"})
    colors = ["#24689b", "#c66a2b", "#41866d"]
    def save(fig, name):
        fig.savefig(output / name, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    lag = np.arange(LENGTH)
    fig, ax = plt.subplots(3, 2, figsize=(10, 6.2), layout="constrained", sharex="col")
    for i, color in enumerate(colors):
        ax[i,0].plot(lag, data["s4_kernel"][i], color=color, lw=1.2)
        ax[i,1].plot(lag[:48], data["s4_kernel"][i, :48], color=color, marker=".", lw=1.2)
        for a in ax[i]: a.axhline(0, color="0.7", lw=.7);a.set_ylabel(f"Channel {i+1}\ncoefficient")
    for a in ax[-1]: a.set_xlabel("Lag (updates)")
    ax[0,0].set_title("(a) Published S4 layer: 512 filter taps")
    ax[0,1].set_title("(b) First 48 taps (each row has its own scale)")
    save(fig, "ch13_pretrained_s4_filters.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    angle = np.linspace(0, 2*np.pi, 512)
    ax[0].plot(np.cos(angle), np.sin(angle), color="0.65", ls="--", label="Unit circle")
    for i, color in enumerate(colors):
        p = data["s4_poles"][i]
        ax[0].scatter(p.real, p.imag, s=13, color=color, alpha=.75, label=f"Channel {i+1}")
        radii = np.sort(np.abs(p))
        ax[1].plot(np.arange(1, 65), -np.log(2)/np.log(radii), color=color, label=f"Δ = {data['s4_dt'][i]:.4f}")
    ax[0].set(xlabel="Real part of discrete pole", ylabel="Imaginary part", title="(a) Poles of the full coupled transition", aspect="equal", xlim=(-1.12,1.12), ylim=(-1.12,1.12))
    ax[0].legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(.5,-.2), ncol=2)
    ax[1].set(xlabel="Mode sorted by pole magnitude", ylabel="Envelope half-life (updates)", title="(b) One learned step size per channel", yscale="log")
    ax[1].legend(frameon=False)
    save(fig, "ch13_pretrained_s4_poles.png")

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), layout="constrained", sharex=True)
    ax[0].plot(lag+1, data["s4_native_output"][0], color=colors[0], label="Native recurrent S4 layer", lw=1.6)
    ax[0].plot(lag[::8]+1, data["s4_direct_output"][0, ::8], "o", ms=3, mfc="none", color=colors[1], label="Independent convolution")
    ax[0].set(ylabel="Output, channel 1", title="(a) The same learned response, evaluated two ways")
    ax[0].legend(frameon=False, ncol=2, fontsize=10)
    for p, val in zip([1,129,257,385],[1,-.5,.75,1.25]):
        ax[0].axvline(p, color="0.8", ls=":", lw=.9)
    magnitude = np.abs(data["s4_native_state"][:, :16]).T
    heat = ax[1].imshow(magnitude, aspect="auto", origin="lower", extent=(.5,512.5,.5,16.5), cmap="magma", norm=LogNorm(vmin=1e-7, vmax=magnitude.max()))
    ax[1].set(xlabel="Input position (four impulses at 1, 129, 257, 385)", ylabel="Complex coordinate", title="(b) Magnitudes of 16 carried state coordinates, channel 1")
    fig.colorbar(heat, ax=ax[1], label="State magnitude (log scale)", fraction=.025)
    save(fig, "ch13_pretrained_s4_execution.png")

    fig, ax = plt.subplots(1, 2, figsize=(10,3.4), layout="constrained")
    for i, color in enumerate(colors):
        ax[0].plot(lag, data["hyena_kernel"][i], color=color, lw=1.3, label=f"Channel {i+1}")
        ax[1].plot(lag[:48], data["hyena_kernel"][i,:48], color=color, marker=".", lw=1.2)
    for a in ax: a.axhline(0,color="0.7",lw=.7);a.set_xlabel("Lag (nucleotide positions)");a.set_ylabel("Long-filter coefficient")
    ax[0].set_title("(a) HyenaDNA block 1: an implicit long filter");ax[0].legend(frameon=False)
    ax[1].set_title("(b) First 48 taps, before the separate D term")
    save(fig, "ch13_pretrained_hyena_filters.png")

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    pos = np.arange(1,49)
    ax[0,0].plot(pos,data["hyena_g"][:48],color=colors[0],label="Input gate g")
    ax[0,0].plot(pos,data["hyena_a"][:48],color=colors[1],label="Output gate a")
    ax[0,0].set(title="(a) Native gates, block 1 / channel 1",ylabel="Gate value");ax[0,0].legend(frameon=False,fontsize=9)
    ax[0,1].plot(pos,data["hyena_v"][:48],color="0.6",label="Projected feature v")
    ax[0,1].plot(pos,data["hyena_gated"][:48],color=colors[2],label="After input gate: g × v")
    ax[0,1].set(title="(b) What enters the long filter",ylabel="Feature value");ax[0,1].legend(frameon=False,fontsize=9)
    ax[1,0].plot(pos,data["hyena_filtered"][:48],color="0.6",label="After convolution + D")
    ax[1,0].plot(pos,data["hyena_output"][:48],color=colors[0],label="After output gate")
    ax[1,0].set(title="(c) Gating after the convolution",ylabel="Feature value");ax[1,0].legend(frameon=False,fontsize=9)
    matrix=data["hyena_frozen_matrix"]
    bound=np.max(np.abs(matrix))
    hm=ax[1,1].imshow(matrix,cmap="RdBu_r",norm=TwoSlopeNorm(0,-bound,bound),origin="upper",extent=(.5,48.5,48.5,.5),aspect="auto")
    ax[1,1].set(title="(d) Frozen-gate coefficients acting on v",xlabel="Source position",ylabel="Output position")
    fig.colorbar(hm,ax=ax[1,1],label="Signed coefficient",fraction=.04)
    for a in ax.flat:
        if a is not ax[1,1]: a.axhline(0,color="0.8",lw=.6);a.set_xlabel("Nucleotide position")
    save(fig, "ch13_pretrained_hyena_gates.png")

    fig, ax = plt.subplots(1,2,figsize=(10,3.3),layout="constrained")
    delta=data["hyena_logit_change"]
    ax[0].plot(lag+1,delta,color=colors[0],lw=1.3)
    ax[0].axvline(MUTATION+1,color=colors[1],ls="--",label="One changed nucleotide")
    ax[0].axvspan(1,MUTATION,color=colors[2],alpha=.1,label="Unchanged earlier prefix")
    ax[0].set(title="(a) Full model: change at position 256",xlabel="Output position",ylabel="Maximum absolute logit change")
    ax[0].legend(frameon=False,fontsize=9)
    ax[1].plot(np.arange(1,MUTATION+1),delta[:MUTATION],color=colors[2],lw=1)
    ax[1].set(title="(b) Earlier prefix, expanded numerical scale",xlabel="Output position",ylabel="Maximum absolute logit change")
    ax[1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    save(fig, "ch13_pretrained_hyena_causality.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--source-dir",type=Path,required=True,help="Cache of four reviewed source files")
    parser.add_argument("--offline",action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    arrays,record=observe(args.cache_dir,args.source_dir,args.offline)
    np.savez_compressed(args.output_dir/"observations.npz",**arrays)
    figures(arrays,args.output_dir)
    record["figures"]={p.name:sha256(p) for p in sorted(args.output_dir.glob("ch13_*.png"))}
    (args.output_dir/"run.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record["checks"],indent=2))


if __name__ == "__main__":
    main()
