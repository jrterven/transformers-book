"""Inspect a released small Gated DeltaNet/full-attention hybrid on a CPU.

The published multimodal checkpoint is used with text inputs only. No vision
features, additional training, sampling, or remote model code are used.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform

import numpy as np

from .model import delta_chunked, delta_sequence, delta_step

MODEL = 'Qwen/Qwen3.5-0.8B-Base'
REVISION = 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68'
TEXT = ('The parcel arrived in Paris. The weather was cloudy. '
        'The parcel moved to Rome. The driver checked the map. ') * 20
LENGTH, DECODE, MUTATION, WRITE, HEAD, CHUNK = 128, 2, 63, 47, 1, 32


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cache_record(cache, length):
    rows = []
    for layer, store in enumerate(cache.layers):
        row = {'layer': layer + 1, 'state': 0, 'conv': 0, 'kv': 0, 'shapes': {}}
        for attribute, component in [('recurrent_states', 'state'), ('conv_states', 'conv'),
                                     ('keys', 'kv'), ('values', 'kv')]:
            tensor = getattr(store, attribute, None)
            if tensor is not None:
                row[component] += tensor.numel() * tensor.element_size()
                row['shapes'][attribute] = list(tensor.shape)
        rows.append(row)
    return {'length': length, 'layers': rows,
            **{kind: sum(row[kind] for row in rows) for kind in ('state', 'conv', 'kv')}}


def observe(cache_dir, offline):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    common = dict(cache_dir=str(cache_dir), local_files_only=offline, token=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                             trust_remote_code=False, **common)
    ids = tokenizer(TEXT, add_special_tokens=False, return_tensors='pt')['input_ids'][:, :LENGTH+DECODE]
    replacement = tokenizer.encode(' blue', add_special_tokens=False)
    assert len(replacement) == 1 and replacement[0] != int(ids[0, MUTATION])
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation='eager',
        trust_remote_code=False, output_loading_info=True, **common)
    assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'], loading
    # A next-token text replay does not use the checkpoint's optional MTP branch.
    assert all(key.startswith('mtp.') or '.mtp.' in key for key in loading['unexpected_keys']), loading
    loading = {key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()}
    model.eval().requires_grad_(False)
    backbone = model.model.language_model
    config = backbone.config
    assert config.layer_types == (['linear_attention'] * 3 + ['full_attention']) * 6
    mixer, attention = backbone.layers[0].linear_attn, backbone.layers[3].self_attn
    captured = {}
    original = mixer.chunk_gated_delta_rule

    def capture_core(query, key, value, **kwargs):
        result = original(query, key, value, **kwargs)
        for name, tensor in [('q', query), ('k', key), ('v', value),
                             ('g', kwargs['g']), ('beta', kwargs['beta']),
                             ('core', result[0]), ('last', result[1])]:
            captured[name] = tensor.detach().clone()
        return result

    def capture_attention(module, args, kwargs, result):
        captured['attention'] = result[1].detach().clone()
        captured['attention_input'] = kwargs['hidden_states'].detach().clone()

    mixer.chunk_gated_delta_rule = capture_core
    hooks = [attention.register_forward_hook(capture_attention, with_kwargs=True),
             attention.o_proj.register_forward_pre_hook(
                 lambda mod, args: captured.update(attention_before_output=args[0].detach().clone()))]
    with torch.inference_mode():
        try:
            baseline = model(ids[:, :LENGTH], use_cache=True)
        finally:
            mixer.chunk_gated_delta_rule = original
            for handle in hooks:
                handle.remove()
        logits = baseline.logits[0].numpy().copy()
        cache = baseline.past_key_values
        payloads = [cache_record(cache, LENGTH)]
        first_state = cache.layers[0].recurrent_states.detach().numpy().copy()
        attention_values = cache.layers[3].values.detach().clone()
        cached_logits = []
        for position in range(LENGTH, LENGTH+DECODE):
            step = model(ids[:, position:position+1], past_key_values=cache, use_cache=True)
            cached_logits.append(step.logits[0].numpy().copy())
            payloads.append(cache_record(cache, position+1))
        full = model(ids, use_cache=False).logits[0].numpy().copy()
        cached_logits = np.concatenate(cached_logits)
        cached_error = np.max(np.abs(cached_logits - full[LENGTH:]), axis=-1)
        np.testing.assert_allclose(cached_logits, full[LENGTH:], rtol=0, atol=2e-3)
        assert np.array_equal(cached_logits.argmax(-1), full[LENGTH:].argmax(-1))
        changed = ids[:, :LENGTH].clone()
        changed[0, MUTATION] = replacement[0]
        changed_logits = model(changed, use_cache=False).logits[0].numpy().copy()
        causal_error = np.max(np.abs(changed_logits - logits), axis=-1)
        assert causal_error[:MUTATION].max() < 1e-4
        assert causal_error[MUTATION:].max() > 1e-3
        for length in (32, 64):
            shorter = model(ids[:, :length], use_cache=True, logits_to_keep=1)
            payloads.append(cache_record(shorter.past_key_values, length))
        # Normalize in native float32, with its exact epsilon, then independently
        # evaluate the captured core in NumPy float64. The library uses [key,value]
        # state orientation, the transpose of the book's [value,key] convention.
        q = (native.l2norm(captured['q'].float()) / np.sqrt(config.linear_key_head_dim))[0].numpy().astype(np.float64)
        k = native.l2norm(captured['k'].float())[0].numpy().astype(np.float64)
        v = captured['v'][0].numpy().astype(np.float64)
        alpha = captured['g'][0].exp().numpy().astype(np.float64)
        beta = captured['beta'][0].numpy().astype(np.float64)
        native_recurrent, native_last = native.torch_recurrent_gated_delta_rule(
            captured['q'], captured['k'], captured['v'], captured['g'], captured['beta'],
            initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
        core_error = float((native_recurrent - captured['core']).abs().max())
        native_state_error = float((native_last - captured['last']).abs().max())
        assert core_error < 2e-4 and native_state_error < 2e-4
        attn = captured['attention']
        assert float(attn.triu(1).abs().max()) == 0
        torch.testing.assert_close(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=2e-6, rtol=0)
        grouped_values = attention_values.repeat_interleave(attention.num_key_value_groups, dim=1)
        raw_read = (attn @ grouped_values).transpose(1, 2).reshape(1, LENGTH, -1)
        gate = attention.q_proj(captured['attention_input']).view(1, LENGTH, config.num_attention_heads, -1).chunk(2, dim=-1)[1]
        reconstructed_read = raw_read * gate.reshape(1, LENGTH, -1).sigmoid()
        attn_error = float((reconstructed_read - captured['attention_before_output']).abs().max())
        assert attn_error < 2e-5

    q, k, v, a, b = q[:, HEAD], k[:, HEAD], v[:, HEAD], alpha[:, HEAD], beta[:, HEAD]
    independent, last = delta_sequence(q, k, v, a, b)
    chunked, chunk_last = delta_chunked(q, k, v, a, b, 31)
    core = captured['core'][0, :, HEAD].numpy()
    numpy_error = float(np.max(np.abs(independent - core)))
    numpy_state_error = float(np.max(np.abs(last - first_state[0, HEAD].T)))
    assert numpy_error < 2e-4 and numpy_state_error < 2e-4
    np.testing.assert_allclose(chunked, independent, rtol=0, atol=1e-11)
    np.testing.assert_allclose(chunk_last, last, rtol=0, atol=1e-11)
    state = np.zeros_like(last)
    predictions, updated_reads, errors = [], [], []
    local, carried = [], []
    for t in range(LENGTH):
        prediction = (a[t] * state) @ k[t]
        predictions.append(prediction.copy())
        errors.append(v[t] - prediction)
        _, state = delta_step(state, q[t], k[t], v[t], a[t], b[t])
        updated_reads.append(state @ k[t])
    state = np.zeros_like(last)
    for start in range(0, LENGTH, CHUNK):
        rows = slice(start, start+CHUNK)
        zero, _ = delta_sequence(q[rows], k[rows], v[rows], a[rows], b[rows])
        total, state = delta_sequence(q[rows], k[rows], v[rows], a[rows], b[rows], state)
        local.append(zero)
        carried.append(total - zero)
    coefficients, _ = delta_chunked(q, k, np.eye(LENGTH), a, b, 31)
    np.testing.assert_allclose(coefficients @ v, independent, rtol=0, atol=1e-11)
    predicted, updated = np.array(predictions), np.array(updated_reads)
    interpolation = predicted + b[:, None] * np.sum(k*k, axis=-1)[:, None] * (v-predicted)
    np.testing.assert_allclose(updated, interpolation, atol=1e-11, rtol=0)
    payloads.sort(key=lambda row: row['length'])
    assert len({p['state'] for p in payloads}) == len({p['conv'] for p in payloads}) == 1
    for payload in payloads:
        assert payload['kv'] == payload['length'] * 6 * 2 * 2 * 256 * 4
    weight_path = hf_hub_download(MODEL, 'model.safetensors-00001-of-00001.safetensors', revision=REVISION, **common)
    record = {'model': MODEL, 'revision': REVISION, 'weights_sha256': sha256(weight_path),
              'runtime_source_sha256': sha256(inspect.getfile(native)),
              'parameters': sum(p.numel() for p in model.parameters()),
              'text_parameters': sum(p.numel() for p in backbone.parameters()),
              'loading': loading, 'layer_types': config.layer_types, 'history': payloads,
              'checks': {'cached_vs_full_logits': float(cached_error.max()),
                         'cached_argmax_agreement': True,
                         'earlier_prefix_change': float(causal_error[:MUTATION].max()),
                         'affected_suffix_change': float(causal_error[MUTATION:].max()),
                         'native_recurrent_vs_chunked_core': core_error,
                         'native_recurrent_vs_chunked_final_state': native_state_error,
                         'numpy_vs_native_core': numpy_error,
                         'numpy_vs_native_final_state': numpy_state_error,
                         'attention_read_reconstruction': attn_error},
              'prefill_tokens': LENGTH, 'decode_tokens': DECODE, 'mutation_zero_based': MUTATION,
              'write_zero_based': WRITE, 'head_zero_based': HEAD, 'teaching_chunk': CHUNK,
              'independent_chunk': 31, 'native_chunk': 64,
              'replacement_id': replacement[0], 'original_piece': tokenizer.convert_ids_to_tokens(int(ids[0, MUTATION])),
              'replacement_piece': tokenizer.convert_ids_to_tokens(replacement[0]),
              'input_ids': ids[0].tolist(), 'input_recipe': TEXT, 'device': 'cpu',
              'dtype': 'float32', 'independent_dtype': 'float64', 'threads': 4,
              'training': False, 'sampling': False, 'timing_benchmark': False,
              'vision_inputs': False, 'versions': {name: importlib.metadata.version(name)
                  for name in ('torch', 'transformers', 'numpy', 'matplotlib', 'huggingface_hub', 'safetensors')},
              'platform': platform.platform()}
    arrays = dict(alpha=alpha, beta=beta, k=k, q=q, v=v, predictions=predicted,
                  updated_reads=updated, core=core, independent=independent,
                  coefficients=coefficients, attention=attn[0, 0].numpy(),
                  local=np.concatenate(local), carried=np.concatenate(carried),
                  causal_error=causal_error, cached_error=cached_error)
    return arrays, record


def plot(arrays, record, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FuncFormatter

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 150})
    blue, orange, green = '#2466a2', '#d07a29', '#298263'

    def save(fig, stem):
        fig.savefig(destination / (stem + '.png'), dpi=190, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.3), sharex=True, layout='constrained')
    for head in range(3):
        axes[0].plot(np.arange(1, 97), arrays['alpha'][:96, head], label=f'Head {head+1}', lw=1)
        axes[1].plot(np.arange(1, 97), arrays['beta'][:96, head], label=f'Head {head+1}', lw=1)
    axes[0].set(title='(a) Learned retention in Gated DeltaNet block 1', ylabel=r'Retention $\alpha_t$', ylim=(0, 1.03))
    axes[1].set(title='(b) Learned correction strength', ylabel=r'Correction $\beta_t$', xlabel='Input position', ylim=(0, 1.03))
    axes[0].legend(ncols=3, loc='center left')
    save(fig, 'ch15_pretrained_controls')

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.9), layout='constrained')
    x = np.arange(8)
    for offset, name, color, label in [(-.25, 'predictions', blue, 'Decayed prediction'),
                                      (0, 'v', orange, 'Projected value'),
                                      (.25, 'updated_reads', green, 'Read at key after correction')]:
        axes[0].bar(x+offset, arrays[name][WRITE, :8], width=.24, color=color, label=label)
    axes[0].set(ylabel='Feature value')
    axes[0].set_title(f'(a) A captured write: block 1, head {HEAD+1}, input position {WRITE+1}', pad=32)
    axes[0].legend(ncols=3, fontsize=8, loc='lower left', bbox_to_anchor=(0,1))
    error = arrays['v'][WRITE, :8] - arrays['predictions'][WRITE, :8]
    scaled = arrays['beta'][WRITE, HEAD] * error
    observed = arrays['updated_reads'][WRITE, :8] - arrays['predictions'][WRITE, :8]
    axes[1].bar(x-.14, scaled, width=.28, color=orange, label=r'Correction vector $\beta_t e_t$')
    axes[1].bar(x+.14, observed, width=.28, color=blue, label='Actual change in the key read')
    axes[1].set(ylabel='Change', xlabel='Displayed value-coordinate index')
    axes[1].set_title('(b) The change in the read includes the squared key norm', pad=32)
    axes[1].legend(ncols=2, fontsize=8, loc='lower left', bbox_to_anchor=(0,1))
    for ax in axes:
        ax.set_xticks(x, x+1)
        ax.axhline(0, color='#888888', lw=.5)
    save(fig, 'ch15_pretrained_delta_write')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout='constrained')
    matrix = arrays['coefficients'][:48, :48]
    bound = float(np.max(np.abs(matrix)))
    signed = np.ma.array(matrix, mask=np.triu(np.ones_like(matrix, dtype=bool), 1))
    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#dddddd')
    im = axes[0].imshow(signed, origin='upper', extent=(.5,48.5,48.5,.5), cmap=cmap,
                        norm=SymLogNorm(linthresh=1e-4, vmin=-bound, vmax=bound))
    fig.colorbar(im, ax=axes[0], fraction=.045, ticks=[-bound, 0, bound], format=FuncFormatter(lambda x, _: f'{x:.2g}'))
    axes[0].set(title=f'(a) Frozen-control GDN coefficients\nblock 1, head {HEAD+1}', xlabel='Source position', ylabel='Read position')
    weights = np.ma.array(arrays['attention'][:48, :48], mask=np.triu(np.ones((48,48), dtype=bool),1))
    cmap2 = plt.get_cmap('viridis').copy(); cmap2.set_bad('#dddddd')
    im = axes[1].imshow(weights, origin='upper', extent=(.5,48.5,48.5,.5), cmap=cmap2, vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[1], fraction=.045, ticks=[0,.5,1])
    axes[1].set(title='(b) Native full-attention weights\nblock 4, query head 1', xlabel='Source position', ylabel='Read position')
    save(fig, 'ch15_pretrained_reads')

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), layout='constrained')
    positions = np.arange(1,97)
    for name, color, label, style in [('local', blue, 'Chunk from zero', '-'),
                                     ('carried', orange, 'Response to incoming state', '-'),
                                     ('core', '#222222', 'Native total', '--')]:
        axes[0].plot(positions, arrays[name][:96,0], color=color, lw=1, label=label, linestyle=style)
    for boundary in (32.5,64.5):
        axes[0].axvline(boundary, color='#999999', ls=':', lw=.8)
    axes[0].set(ylabel='Feature 1', xlabel='Input position')
    axes[0].set_title('(a) GDN read decomposed into local and carried contributions', pad=32)
    axes[0].legend(ncols=3, fontsize=8, loc='lower left', bbox_to_anchor=(0,1))
    k, a, b = arrays['k'][:CHUNK], arrays['alpha'][:CHUNK,HEAD], arrays['beta'][:CHUNK,HEAD]
    from examples.ch14_selective_ssm.model import causal_decay
    system = np.eye(CHUNK) + np.tril(causal_decay(a) * (k@k.T) * b[None,:], -1)
    masked_system = np.ma.array(system, mask=np.triu(np.ones_like(system, dtype=bool), 1))
    system_cmap = plt.get_cmap('RdBu_r').copy(); system_cmap.set_bad('#dddddd')
    im = axes[1].imshow(masked_system, cmap=system_cmap, norm=SymLogNorm(linthresh=1e-3,vmin=-1,vmax=1), aspect='auto', extent=(.5,CHUNK+.5,CHUNK+.5,.5))
    axes[1].set(title='(b) Error-dependency system in the first 32-position chunk', xlabel='Earlier error index', ylabel='Current error index')
    fig.colorbar(im, ax=axes[1], fraction=.03, ticks=[-1,0,1], format=FuncFormatter(lambda x, _: f'{x:g}'))
    save(fig, 'ch15_pretrained_chunks')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    rows = [row for row in record['history'] if row['length'] in (32,64,128)]
    bottom = np.zeros(len(rows))
    for name, label, color in [('state','GDN matrix states',blue),('conv','Short-convolution buffers',green),('kv','Full-attention KV histories',orange)]:
        height = np.array([row[name] for row in rows])/2**20
        axes[0].bar(np.arange(len(rows)),height,bottom=bottom,label=label,color=color)
        bottom += height
    axes[0].set(xticks=np.arange(3),xticklabels=[32,64,128],ylabel='Persistent history payload (MiB)',xlabel='Consumed prefix (tokens)',ylim=(0,27),title='(a) Actual carried arrays across all 24 blocks')
    axes[0].legend(fontsize=8,loc='upper left')
    final = rows[-1]['layers']
    axes[1].bar([r['layer'] for r in final],[(r['state']+r['conv']+r['kv'])/2**20 for r in final],color=[orange if r['kv'] else blue for r in final])
    axes[1].set(title='(b) Separate stores at a 128-token prefix',xlabel='Block index',ylabel='Payload per block (MiB)',xticks=[1,4,8,12,16,20,24],ylim=(0,1.4))
    save(fig, 'ch15_pretrained_history')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout='constrained')
    axes[0].plot(np.arange(1,LENGTH+1),arrays['causal_error'],color=blue,lw=1)
    axes[0].axvspan(.5,MUTATION+.5,color='#eeeeee')
    axes[0].axvline(MUTATION+1,color=orange,lw=.9,ls=':')
    axes[0].set(title=f'(a) One changed token at position {MUTATION+1}',xlabel='Output position',ylabel='Maximum absolute logit change',ylim=(0,None))
    axes[1].plot(np.arange(1,MUTATION+1),arrays['causal_error'][:MUTATION],color=blue)
    axes[1].set(title='(b) Earlier prefix, expanded scale',xlabel='Output position',ylim=(-1e-6,1e-6))
    if np.max(arrays['causal_error'][:MUTATION])==0:
        axes[1].text(.5,.8,'Exact zero in this run',transform=axes[1].transAxes,ha='center',fontsize=8)
    axes[2].bar([1,2],arrays['cached_error'],color=[blue,orange])
    axes[2].set(title='(c) Cached versus full-prefix replay',xlabel='Next consumed token position',ylabel='Maximum absolute logit discrepancy',xticks=[1,2],xticklabels=[LENGTH+1,LENGTH+2],ylim=(0,None))
    save(fig, 'ch15_pretrained_causality')
    record['figures'] = {p.name: sha256(p) for p in sorted(destination.glob('ch15_pretrained_*.png'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path('output/model-cache'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/ch15-pretrained'))
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        arrays = dict(np.load(args.output_dir / 'observations.npz'))
        record = json.loads((args.output_dir / 'run.json').read_text())
    else:
        arrays, record = observe(args.cache_dir, args.offline)
        np.savez_compressed(args.output_dir / 'observations.npz', **arrays)
    plot(arrays, record, args.output_dir)
    (args.output_dir / 'run.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record['checks'], indent=2), flush=True)
    print('Saved six figures from the published Qwen3.5-0.8B-Base text pathway.', flush=True)


if __name__ == '__main__':
    main()
