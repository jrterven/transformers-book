"""Plot six observations from actual frozen-head, SFT/LoRA, and DPO training.

Run real_adaptation first, or use --train to run it before plotting.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FuncFormatter

    arrays=dict(np.load(directory/'observations.npz'))
    record=json.loads((directory/'run.json').read_text())
    blue,orange,green='#28669e','#cc792b','#2b8868'
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})

    def save(fig,name):
        fig.savefig(directory/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white')
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9,3.5),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
    im=axes[0].imshow(np.stack([arrays['head_before'][:,1],arrays['head_after'][:,1]]),
                      cmap='viridis',vmin=0,vmax=1,aspect='auto',extent=(.5,32.5,1.5,-.5))
    axes[0].axvline(16.5,color='white',ls=':',lw=1)
    axes[0].set(title='(a) A trained head on frozen SmolLM2 features',xlabel='Held-out message index',yticks=[0,1],yticklabels=['Initial head','Trained head'],xticks=[1,8,16,24,32])
    fig.colorbar(im,ax=axes[0],fraction=.045,label='Head probability of UPDATE',ticks=[0,.5,1])
    prediction=arrays['head_after'].argmax(-1);truth=arrays['held_labels']
    confusion=np.array([[np.sum((truth==i)&(prediction==j)) for j in range(2)] for i in range(2)])
    axes[1].imshow(confusion,cmap='Blues',vmin=0,vmax=16)
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(confusion[i,j]),ha='center',va='center',color='white' if confusion[i,j]>8 else '#222222',fontsize=16)
    axes[1].set(title='(b) Held-out confusion counts',xlabel='Head prediction',ylabel='Supplied class',xticks=[0,1],yticks=[0,1],xticklabels=['LOOKUP','UPDATE'],yticklabels=['LOOKUP','UPDATE'])
    save(fig,'ch16_pretrained_head')

    fig,axes=plt.subplots(2,1,figsize=(9.5,4.8),sharex=True,layout='constrained')
    positions=np.arange(1,len(arrays['mask_ids'])+1)
    axes[0].bar(positions,arrays['mask_direct_gradient'],color=[green if s else '#aaaaaa' for s in arrays['mask_selected']])
    axes[0].set(title='(a) Direct loss gradient at each native logit row',ylabel='Logit-gradient L2 norm')
    axes[1].bar(positions,arrays['mask_context_gradient'],color=blue)
    axes[1].set(title='(b) Gradient reaching each input embedding',ylabel='Input-gradient L2 norm',xlabel='Input token position and tokenizer piece')
    pieces=[p.replace('Ġ','·').replace('Ċ',r'\n') for p in record['mask_token_pieces']]
    axes[1].set_xticks(positions,pieces,rotation=55,ha='right',fontsize=8)
    for ax in axes:ax.set_ylim(bottom=0)
    save(fig,'ch16_pretrained_mask_gradients')

    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    axes[0].plot(np.arange(1,len(arrays['sft_losses'])+1),arrays['sft_losses'],color=blue,lw=1.2)
    axes[0].set(title='(a) Actual LoRA SFT optimization',xlabel='Optimizer step',ylabel='Current training-batch token NLL',ylim=(0,None))
    values=[record['stages'][stage]['token_nll'] for stage in ('base','sft')]
    axes[1].bar(np.arange(2),values,color=[blue,green])
    for index,value in enumerate(values):axes[1].text(index,value+max(values)*.025,f'{value:.3g}',ha='center',fontsize=9)
    axes[1].set(title='(b) Fixed held-out response targets',xticks=np.arange(2),xticklabels=['Pretrained','After SFT'],ylabel='Mean supervised-token NLL',ylim=(0,max(values)*1.18))
    save(fig,'ch16_pretrained_sft')

    fig,axes=plt.subplots(1,3,figsize=(10,3.3),layout='constrained')
    matrix=arrays['sft_update_corner'];bound=float(abs(matrix).max())
    im=axes[0].imshow(matrix,cmap='RdBu_r',norm=SymLogNorm(linthresh=bound/100,vmin=-bound,vmax=bound),extent=(.5,24.5,24.5,.5))
    fig.colorbar(im,ax=axes[0],fraction=.045,ticks=[-bound,0,bound],format=FuncFormatter(lambda x,_:f'{x:.2g}'))
    axes[0].set(title='(a) Learned update, block 30 query\nfirst 24 output/input coordinates',xlabel='Input coordinate',ylabel='Output coordinate')
    singular=arrays['sft_update_singular_values'][:12]
    axes[1].plot(np.arange(1,13),singular,'o-',color=blue,ms=3)
    axes[1].set_yscale('log');axes[1].axvline(4.5,color=orange,ls=':',lw=1)
    axes[1].set(title='(b) Singular values of the full update',xlabel='Singular-value index',ylabel='Singular value (log scale)',xticks=[1,4,8,12])
    grad=arrays['sft_gradient_norms'][:8]
    axes[2].plot(np.arange(1,9),grad[:,0],'o-',color=blue,label='A factors',ms=3)
    axes[2].plot(np.arange(1,9),grad[:,1],'s-',color=orange,label='B factors',ms=3)
    axes[2].set_yscale('symlog',linthresh=1e-5)
    axes[2].set(title='(c) Factor gradients before clipping',xlabel='Optimizer step',ylabel='Combined L2 norm (symlog)')
    axes[2].legend(fontsize=8)
    save(fig,'ch16_pretrained_lora')

    before,after=arrays['next_token_before'],arrays['next_token_sft']
    target=record['label_token_ids'][record['held'][0]['response'].strip()][0]
    selected=set(np.argsort(before)[-6:].tolist()+np.argsort(after)[-6:].tolist()+[target])
    selected=sorted(selected,key=lambda i:max(before[i],after[i]),reverse=True)
    vocabulary=record['native_vocabulary_pieces']
    labels=[vocabulary[i].replace('Ġ','·').replace('Ċ',r'\n')+(' [target]' if i==target else '') for i in selected]
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
    y=np.arange(len(selected))
    ax.barh(y-.18,before[selected],height=.35,color=blue,label='Published base')
    ax.barh(y+.18,after[selected],height=.35,color=green,label='After actual LoRA SFT')
    ax.set(yticks=y,yticklabels=labels,xlabel='Probability over the full 49,152-token vocabulary',title='One held-out prompt: the first response-token distribution',xlim=(0,None))
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    save(fig,'ch16_pretrained_token_probabilities')
    record['displayed_next_token_ids']=selected

    fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
    axes[0].plot(np.arange(1,len(arrays['dpo_losses'])+1),arrays['dpo_losses'],'o-',color=orange,ms=3)
    axes[0].axhline(np.log(2),color='#888888',ls=':',lw=.8)
    axes[0].set(title='(a) Actual DPO optimization',xlabel='Optimizer step',ylabel='Current training-batch pair loss',ylim=(0,None))
    plus=arrays['policy_chosen']-arrays['reference_chosen'];minus=arrays['policy_rejected']-arrays['reference_rejected']
    xlow,xhigh=min(float(plus.min()),0),max(float(plus.max()),0)
    ylow,yhigh=min(float(minus.min()),0),max(float(minus.max()),0)
    xpad=max((xhigh-xlow)*.08,.01);ypad=max((yhigh-ylow)*.08,.01)
    for label,color in [(0,blue),(1,green)]:
        which=truth==label;axes[1].scatter(plus[which],minus[which],color=color,s=18,label=['LOOKUP','UPDATE'][label])
    axes[1].set(title='(b) Held-out response score changes',xlabel='Chosen log-probability change',ylabel='Rejected log-probability change',xlim=(xlow-xpad,xhigh+xpad),ylim=(ylow-ypad,yhigh+ypad))
    axes[1].axvline(0,color='#888888',ls=':',lw=.8)
    axes[1].legend(fontsize=7,loc='upper left')
    margin=record['dpo_beta']*(plus-minus)
    axes[2].bar(np.arange(1,33),margin,color=[blue if t==0 else green for t in truth])
    axes[2].axhline(0,color='#777777',lw=.7)
    axes[2].set(title='(c) Reference-adjusted held-out margin',xlabel='Held-out pair index',ylabel=r'$\beta\,[g_{policy}-g_{reference}]$',xticks=[1,16,32])
    save(fig,'ch16_pretrained_dpo')

    record['figures']={p.name:sha256(p) for p in sorted(directory.glob('ch16_pretrained_*.png'))}
    (directory/'run.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    compact={key:value for key,value in record.items() if key!='native_vocabulary_pieces'}
    compact['displayed_next_token_pieces']=[vocabulary[i] for i in selected]
    (directory/'public-run.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'figures':record['figures'],'checks':record['checks']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('output/ch16-pretrained'))
    parser.add_argument('--cache-dir',type=Path,default=Path('output/model-cache'))
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    if args.train:
        from .real_adaptation import run
        run(args.cache_dir,args.output_dir,args.offline)
    plot(args.output_dir)


if __name__=='__main__':main()
