"""Actually adapt a pinned pretrained SmolLM2 model on a small fictional task.

The fixed run trains a frozen-feature classifier, rank-four SFT adapters, and
then format-preference DPO adapters. No held-out outcome selects a run.
"""

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
from torch.nn import functional as F

from .model import (LoRALinear, attach_lora, completion_logprob, dpo_loss,
                    merge_lora, response_labels, response_loss)
from .real_data import LABELS, examples

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
SEED, RANK, ALPHA, SFT_STEPS, DPO_STEPS = 16, 4, 8, 64, 16
SFT_LR, DPO_LR, BETA, PAD = 2e-3, 2e-4, .1, 1


def file_sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def base_digest(model):
    digest = hashlib.sha256()
    tensors = [(name.replace('.base.', '.'), p) for name,p in model.named_parameters()
               if not (name.endswith('.a') or name.endswith('.b'))]
    for name, p in sorted(tensors):
        digest.update(name.encode()); digest.update(p.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def packed(tokenizer, rows, *, rejected=False):
    sequences, masks = [], []
    for row in rows:
        prompt = tokenizer.encode(row['prompt'], add_special_tokens=False)
        response = tokenizer.encode(row['rejected'] if rejected else row['response'], add_special_tokens=False)
        assert tokenizer.encode(row['prompt'] + (row['rejected'] if rejected else row['response']),
                                add_special_tokens=False) == prompt + response
        sequence = prompt + response + [tokenizer.eos_token_id]
        assert PAD not in sequence and PAD != tokenizer.eos_token_id
        sequences.append(sequence)
        masks.append([False]*len(prompt) + [True]*(len(response)+1))
    width = max(map(len,sequences))
    ids = torch.full((len(rows),width),PAD,dtype=torch.long)
    response_mask = torch.zeros_like(ids,dtype=torch.bool)
    for index,(sequence,mask) in enumerate(zip(sequences,masks)):
        ids[index,:len(sequence)] = torch.tensor(sequence)
        response_mask[index,:len(sequence)] = torch.tensor(mask)
    return ids, ids.ne(PAD), response_labels(ids,response_mask,pad_id=PAD), response_mask


def scores(model, tokenizer, rows, *, rejected=False, batch_size=8):
    totals, counts = [], []
    with torch.no_grad():
        for start in range(0,len(rows),batch_size):
            ids,mask,labels,_ = packed(tokenizer,rows[start:start+batch_size],rejected=rejected)
            logits = model(ids,attention_mask=mask,use_cache=False).logits
            totals.append(completion_logprob(logits,labels))
            counts.append(labels.ne(-100).sum(-1))
    return torch.cat(totals), torch.cat(counts)


def report_scores(model,tokenizer,rows):
    correct,counts = scores(model,tokenizer,rows)
    alternatives = [{**row,'response':' '+LABELS[1-row['label']]} for row in rows]
    other,_ = scores(model,tokenizer,alternatives)
    return {'token_nll':float(-correct.sum()/counts.sum()),
            'candidate_label_accuracy':float((correct>other).float().mean()),
            'correct_logp':correct.tolist(),'other_label_logp':other.tolist(),
            'selected_counts':counts.tolist()}


def generated(model,tokenizer,rows):
    records=[]
    with torch.no_grad():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]
            prompts=[tokenizer.encode(row['prompt'],add_special_tokens=False) for row in batch]
            width=max(map(len,prompts)); ids=torch.full((len(batch),width),PAD,dtype=torch.long)
            for index,values in enumerate(prompts):ids[index,-len(values):]=torch.tensor(values)
            outputs=model.generate(ids,attention_mask=ids.ne(PAD),do_sample=False,max_new_tokens=8,
                                   pad_token_id=PAD,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            for row,output in zip(batch,outputs[:,width:]):
                values=output.tolist()
                if tokenizer.eos_token_id in values:
                    values=values[:values.index(tokenizer.eos_token_id)+1]
                text=tokenizer.decode(values,skip_special_tokens=True)
                records.append({'text':text,'token_ids':values,
                                'exact_label':text.strip()==LABELS[row['label']],
                                'stopped_on_eos':bool(values and values[-1]==tokenizer.eos_token_id)})
    return records


def snapshot(adapters):
    return {name+'.'+factor:getattr(layer,factor).detach().clone()
            for name,layer in adapters.items() for factor in ('a','b')}


@torch.no_grad()
def restore(adapters,saved):
    expected={name+'.'+factor for name in adapters for factor in ('a','b')}
    assert set(saved)==expected
    for name,layer in adapters.items():
        for factor in ('a','b'):
            value=saved[name+'.'+factor]
            assert value.shape==getattr(layer,factor).shape
            getattr(layer,factor).copy_(value)


def run(cache_dir,output_dir,offline):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    torch.set_num_threads(4);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    train,held=examples('train'),examples('held')
    for field in ('family','account','prompt'):
        assert not {row[field] for row in train}&{row[field] for row in held}
    output_dir.mkdir(parents=True,exist_ok=True)
    common=dict(revision=REVISION,cache_dir=str(cache_dir),local_files_only=offline,
                trust_remote_code=False,token=False)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,use_fast=True,**common)
    assert tokenizer.eos_token_id==0 and tokenizer.convert_tokens_to_ids('<|im_start|>')==PAD
    model,loading=AutoModelForCausalLM.from_pretrained(MODEL,attn_implementation='sdpa',
                                                    output_loading_info=True,**common)
    assert not any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
    model.eval().float().requires_grad_(False)
    base_parameters=sum(p.numel() for p in model.parameters()); initial_digest=base_digest(model)
    arrays={};checks={};stages={}
    print('Loaded published base; extracting real frozen features.',flush=True)

    def features(rows):
        results=[]
        with torch.no_grad():
            for start in range(0,len(rows),8):
                prompts=[tokenizer.encode(row['prompt'],add_special_tokens=False) for row in rows[start:start+8]]
                ids=torch.full((len(prompts),max(map(len,prompts))),PAD,dtype=torch.long)
                lengths=torch.tensor([len(p) for p in prompts])
                for index,p in enumerate(prompts):ids[index,:len(p)]=torch.tensor(p)
                hidden=model.model(ids,attention_mask=ids.ne(PAD),use_cache=False).last_hidden_state
                results.append(hidden[torch.arange(len(prompts)),lengths-1].detach())
        return torch.cat(results)

    train_x,held_x=features(train),features(held)
    feature_mean=train_x.mean(0)
    feature_scale=train_x.std(0,unbiased=False).clamp_min(1e-4)
    train_x=(train_x-feature_mean)/feature_scale
    held_x=(held_x-feature_mean)/feature_scale
    train_y=torch.tensor([r['label'] for r in train]);held_y=torch.tensor([r['label'] for r in held])
    head=nn.Linear(model.config.hidden_size,2)
    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    arrays['head_before']=head(held_x).detach().softmax(-1).numpy()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    head_losses=[]
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(head(train_x),train_y);loss.backward();optimizer.step()
        head_losses.append(float(loss.detach()))
    arrays['head_after']=head(held_x).detach().softmax(-1).numpy();arrays['head_losses']=np.array(head_losses)
    arrays['held_labels']=held_y.numpy()
    head_accuracy=float((head(held_x).argmax(-1)==held_y).float().mean())
    assert np.isfinite(head_losses).all() and head_losses[-1]<head_losses[0]
    assert base_digest(model)==initial_digest
    save_file({**{name:p.detach() for name,p in head.state_dict().items()},
               'feature_mean':feature_mean,'feature_scale':feature_scale},str(output_dir/'classifier.safetensors'))

    # Observe the loss's direct logit gradient and its separate context path.
    short={'prompt':'Request: Please change my address.\nIntent:','response':' UPDATE'}
    ids,mask,labels,response_mask=packed(tokenizer,[short])
    embedded=model.get_input_embeddings()(ids).detach().requires_grad_()
    logits=model(inputs_embeds=embedded,attention_mask=mask,use_cache=False).logits
    logits.retain_grad();loss=response_loss(logits,labels);loss.backward()
    direct=logits.grad.norm(dim=-1)[0].detach().numpy()
    contextual=embedded.grad.norm(dim=-1)[0].detach().numpy()
    selected=np.r_[labels[0].ne(-100).numpy(),False]
    assert np.max(direct[~selected])==0 and np.min(direct[selected])>0
    prompt_length=int((~response_mask[0]).sum())
    assert np.max(contextual[:prompt_length-1])>0
    arrays.update(mask_ids=ids[0].numpy(),mask_selected=selected,
                  mask_direct_gradient=direct,mask_context_gradient=contextual,
                  mask_response_targets=response_mask[0].numpy())
    mask_pieces=tokenizer.convert_ids_to_tokens(ids[0].tolist())
    checks['ignored_direct_logit_gradient_max']=float(np.max(direct[~selected]))
    checks['earlier_prompt_embedding_gradient_max']=float(np.max(contextual[:prompt_length-1]))
    del embedded,logits,loss

    print('Measuring unchanged pretrained label probabilities and generation.',flush=True)
    stages['base']=report_scores(model,tokenizer,held)
    stages['base']['generated']=generated(model,tokenizer,held)
    probe_prompt=tokenizer.encode(held[0]['prompt'],add_special_tokens=False)
    probe_ids=torch.tensor([probe_prompt])
    with torch.no_grad():
        before_logits=model(probe_ids,use_cache=False).logits.detach().clone()
    arrays['next_token_before']=before_logits[0,-1].softmax(-1).numpy()
    targets=tuple(f'model.layers.{layer}.self_attn.{projection}' for layer in range(30)
                  for projection in ('q_proj','v_proj'))
    adapters=attach_lora(model,targets,RANK,ALPHA)
    model.eval()
    assert base_digest(model)==initial_digest
    with torch.no_grad():
        fresh=model(probe_ids,use_cache=False).logits
    checks['zero_adapter_initial_logit_difference']=float((fresh-before_logits).abs().max())
    assert checks['zero_adapter_initial_logit_difference']==0
    trainable=[p for p in model.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in trainable)
    assert trainable_count==230400
    optimizer=torch.optim.AdamW(trainable,lr=SFT_LR,weight_decay=0)
    generator=torch.Generator().manual_seed(SEED)
    order=torch.randperm(len(train),generator=generator).tolist();losses=[];gradient_norms=[]
    print(f'Training {SFT_STEPS} SFT steps on {len(train)} messages, {trainable_count} adapter parameters.',flush=True)
    for step in range(SFT_STEPS):
        if step and step%(len(train)//8)==0:order=torch.randperm(len(train),generator=generator).tolist()
        start=(step*8)%len(train);batch=[train[i] for i in order[start:start+8]]
        ids,mask,labels,_=packed(tokenizer,batch)
        optimizer.zero_grad(set_to_none=True)
        logits=model(ids,attention_mask=mask,use_cache=False).logits
        loss=response_loss(logits,labels);loss.backward()
        norms=[float(torch.sqrt(sum((getattr(layer,factor).grad.detach()**2).sum()
                                  for layer in adapters.values()))) for factor in ('a','b')]
        gradient_norms.append(norms)
        if step==0:assert norms[0]==0 and norms[1]>0
        if step==1:assert norms[0]>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step();losses.append(float(loss.detach()))
        if (step+1)%16==0:print(f'SFT step {step+1}: batch token loss {losses[-1]:.6f}',flush=True)
    stages['sft']=report_scores(model,tokenizer,held);stages['sft']['generated']=generated(model,tokenizer,held)
    arrays['sft_losses']=np.array(losses);arrays['sft_gradient_norms']=np.array(gradient_norms)
    with torch.no_grad():
        arrays['next_token_sft']=model(probe_ids,use_cache=False).logits[0,-1].softmax(-1).numpy()
    sft_saved=snapshot(adapters);save_file(sft_saved,str(output_dir/'sft-adapter.safetensors'))
    observed=adapters['model.layers.29.self_attn.q_proj']
    update=observed.scale*(observed.b.detach().double().numpy()@observed.a.detach().double().numpy())
    singular=np.linalg.svd(update,compute_uv=False)
    arrays['sft_update_corner']=update[:24,:24];arrays['sft_update_singular_values']=singular
    checks['update_numerical_rank_at_relative_1e-10']=int(np.sum(singular>singular[0]*1e-10))
    assert checks['update_numerical_rank_at_relative_1e-10']<=RANK

    # The reference is the frozen SFT checkpoint. Cache exact pair scores once.
    preference_rows=[row for row in train if row['account']<4]
    ref_train_plus,_=scores(model,tokenizer,preference_rows)
    ref_train_minus,_=scores(model,tokenizer,preference_rows,rejected=True)
    ref_held_plus,_=scores(model,tokenizer,held)
    ref_held_minus,_=scores(model,tokenizer,held,rejected=True)
    ref_digest=hashlib.sha256(torch.stack([ref_train_plus,ref_train_minus]).numpy().tobytes()).hexdigest()
    optimizer=torch.optim.AdamW(trainable,lr=DPO_LR,weight_decay=0)
    generator=torch.Generator().manual_seed(SEED+1)
    order=torch.randperm(len(preference_rows),generator=generator).tolist();dpo_losses=[]
    print(f'Training {DPO_STEPS} DPO steps on {len(preference_rows)} authored format preferences.',flush=True)
    for step in range(DPO_STEPS):
        if step and step%(len(preference_rows)//4)==0:order=torch.randperm(len(preference_rows),generator=generator).tolist()
        start=(step*4)%len(preference_rows);indices=order[start:start+4]
        batch=[preference_rows[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        ids,mask,labels,_=packed(tokenizer,batch)
        chosen=completion_logprob(model(ids,attention_mask=mask,use_cache=False).logits,labels)
        ids,mask,labels,_=packed(tokenizer,batch,rejected=True)
        rejected=completion_logprob(model(ids,attention_mask=mask,use_cache=False).logits,labels)
        loss=dpo_loss(chosen,rejected,ref_train_plus[indices],ref_train_minus[indices],BETA)
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
        dpo_losses.append(float(loss.detach()))
        if (step+1)%4==0:print(f'DPO step {step+1}: pair loss {dpo_losses[-1]:.6f}',flush=True)
    policy_plus,_=scores(model,tokenizer,held)
    policy_minus,_=scores(model,tokenizer,held,rejected=True)
    arrays.update(reference_chosen=ref_held_plus.numpy(),reference_rejected=ref_held_minus.numpy(),
                  policy_chosen=policy_plus.numpy(),policy_rejected=policy_minus.numpy(),dpo_losses=np.array(dpo_losses))
    adjusted=(policy_plus-policy_minus)-(ref_held_plus-ref_held_minus)
    stages['dpo']=report_scores(model,tokenizer,held);stages['dpo']['generated']=generated(model,tokenizer,held)
    stages['dpo']['mean_reference_adjusted_gap']=float(adjusted.mean())
    stages['dpo']['mean_held_pair_loss']=float(F.softplus(-BETA*adjusted).mean())
    checks['reference_scores_unchanged']=ref_digest==hashlib.sha256(torch.stack([ref_train_plus,ref_train_minus]).numpy().tobytes()).hexdigest()
    assert checks['reference_scores_unchanged'] and base_digest(model)==initial_digest
    checks['base_parameters_unchanged_after_both_training_stages']=True
    dpo_saved=snapshot(adapters);save_file(dpo_saved,str(output_dir/'dpo-adapter.safetensors'))
    # A real file round trip and a whole-model merge check use all logit coordinates.
    check_rows=held[:2];ids,mask,_,_=packed(tokenizer,check_rows)
    with torch.no_grad():expected=model(ids,attention_mask=mask,use_cache=False).logits.detach().clone()
    restore(adapters,load_file(str(output_dir/'sft-adapter.safetensors')))
    with torch.no_grad():restored_sft=model(probe_ids,use_cache=False).logits[0,-1].softmax(-1).numpy()
    checks['sft_reload_next_token_probability_error']=float(np.max(abs(restored_sft-arrays['next_token_sft'])))
    assert checks['sft_reload_next_token_probability_error']<1e-7
    restore(adapters,load_file(str(output_dir/'dpo-adapter.safetensors')))
    with torch.no_grad():restored=model(ids,attention_mask=mask,use_cache=False).logits
    checks['dpo_reload_max_logit_error']=float((expected-restored).abs().max())
    assert checks['dpo_reload_max_logit_error']==0
    merge_lora(model,targets)
    with torch.no_grad():merged=model(ids,attention_mask=mask,use_cache=False).logits
    arrays['merged_logit_errors']=(expected-merged).abs().amax(dim=-1).numpy()
    checks['whole_model_merged_max_logit_error']=float(arrays['merged_logit_errors'].max())
    assert checks['whole_model_merged_max_logit_error']<2e-3
    assert torch.equal(expected.argmax(-1),merged.argmax(-1))
    checks['merged_argmax_all_positions_agree']=True
    pieces=tokenizer.convert_ids_to_tokens(np.arange(model.config.vocab_size).tolist())
    weights=hf_hub_download(MODEL,'model.safetensors',**{k:v for k,v in common.items() if k!='trust_remote_code'})
    record={'model':MODEL,'revision':REVISION,'weights_sha256':file_sha256(weights),
            'runtime_source_sha256':file_sha256(inspect.getfile(modeling_llama)),
            'base_parameters':base_parameters,'base_parameter_digest':initial_digest,
            'classifier_parameters':sum(p.numel() for p in head.parameters()),
            'adapter_parameters':trainable_count,'adapter_rank':RANK,'adapter_alpha':ALPHA,
            'target_modules':list(targets),'seed':SEED,'threads':4,'device':'cpu','dtype':'float32',
            'backend':'native CPU SDPA','dropout_mode':'eval throughout, autograd enabled during updates',
            'train':train,'held':held,'preference_train_indices':[train.index(row) for row in preference_rows],
            'sft_steps':SFT_STEPS,'sft_learning_rate':SFT_LR,'sft_batch_size':8,
            'dpo_steps':DPO_STEPS,'dpo_learning_rate':DPO_LR,'dpo_batch_size':4,'dpo_beta':BETA,
            'gradient_clip_norm':1.,'optimizer':'AdamW, weight_decay=0',
            'head_steps':100,'head_learning_rate':.01,'head_weight_decay':.001,
            'head_optimizer':'AdamW','head_normalization':'Training-coordinate mean and population standard deviation, floored at 1e-4',
            'head_fixed_normalization_scalars':1152,
            'head_accuracy':head_accuracy,'head_train_accuracy':float((head(train_x).argmax(-1)==train_y).float().mean()),
            'stages':stages,'checks':checks,'mask_token_pieces':mask_pieces,
            'label_token_ids':{name:tokenizer.encode(' '+name,add_special_tokens=False) for name in LABELS},
            'pad_id':PAD,'eos_id':tokenizer.eos_token_id,'mask_example':short,
            'probe_held_index':0,'native_vocabulary_pieces':pieces,
            'checkpoints':{path.name:{'sha256':file_sha256(path),'bytes':path.stat().st_size} for path in sorted(output_dir.glob('*.safetensors'))},
            'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','matplotlib','safetensors','huggingface_hub')},
            'platform':platform.platform(),'test_used_for_selection':False,
            'scope':'Illustrative fictional intent and authored-format task, one seed; no broad alignment or benchmark claim.'}
    # Vocabulary pieces support plotting; keep the compact public record separate.
    np.savez_compressed(output_dir/'observations.npz',**arrays)
    (output_dir/'run.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'classifier_held_accuracy':head_accuracy,'stages':{name:{k:v for k,v in stage.items() if k in ('token_nll','candidate_label_accuracy','mean_reference_adjusted_gap','mean_held_pair_loss')} for name,stage in stages.items()},'checks':checks},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=Path('output/model-cache'))
    parser.add_argument('--output-dir',type=Path,default=Path('output/ch16-pretrained'))
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args();run(args.cache_dir,args.output_dir,args.offline)


if __name__=='__main__':main()
