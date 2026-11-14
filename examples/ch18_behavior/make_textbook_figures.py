"""Vector diagrams and measured plots for the expanded Chapter 18."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .feedback import preference_probability, reward_curves, reward_tilt
from .make_figures import BLUE, GRID, INK, ORANGE, STYLE, TEAL

MUTED = "#5d7082"
PALE = ("#eaf3fa", "#e9f4ef", "#fcf0e7", "#f2f0f8")


def diagram(width, height):
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", accent=BLUE, fill="#f7fafc", title_size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008,rounding_size=.017",
                                facecolor=fill, linewidth=1.0, edgecolor=accent))
    offset = 0.32 if body else 0.50
    ax.text(x+w/2, y+h*(1-offset), title, ha="center", va="center",
            color=accent, weight="bold", fontsize=title_size)
    if body:
        ax.text(x+w/2, y+h*.31, body, ha="center", va="center", fontsize=10)


def arrow(ax, start, end, color=MUTED, connection="arc3"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.35,
                                shrinkA=0, shrinkB=0, connectionstyle=connection))


def question_map():
    fig, ax = diagram(8.8, 3.4)
    headings = ("Question about the desk", "Experiment", "Evidence we obtain")
    for x, title in zip((.025, .38, .69), headings, strict=True):
        ax.text(x, .93, title, weight="bold", fontsize=11, color=INK)
    entries = (
        ("Does wording change the action?", "Compare matched prompts", "Dependence on an input edit"),
        ("Which inputs move a score?", "Attribute the score difference", "Effects for a stated baseline"),
        ("Can we recover the intent?", "Train a probe", "Information a readout can decode"),
        ("Which computation matters?", "Intervene and rerun", "Effect of one specified change"),
    )
    for i, (q, action, evidence) in enumerate(entries):
        y = .71-i*.195
        ax.add_patch(FancyBboxPatch((.009,y), .98,.158, boxstyle="round,pad=.004,rounding_size=.018",
                                    linewidth=0, facecolor=PALE[i]))
        ax.text(.025,y+.077,q,va="center", weight="bold", fontsize=10.4)
        ax.text(.38,y+.077,action,va="center",fontsize=10.2)
        ax.text(.69,y+.077,evidence,va="center",fontsize=10.2)
    return fig


def attention_figure(head):
    weights = np.asarray(head["attention_rows"])
    tokens = head["tokens"]
    example_a = next(i for i, t in enumerate(tokens) if t == " A")
    verb = tokens.index(" show")
    address = max(i for i, t in enumerate(tokens) if t == " address")
    colon = len(tokens)-1
    picked = weights[:, [example_a, verb, address, colon]]
    selected = np.column_stack((picked, 1-picked.sum(axis=1)))
    n = len(selected)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.85), width_ratios=(2.25, 1.2, 1.35),
                             sharey=True, layout="constrained")
    left, mid, right = axes
    heat = left.imshow(selected, cmap="Blues", norm=PowerNorm(.6, vmin=0, vmax=1),
                       aspect="auto")
    left.set_xticks(range(5), ["Demo A", "show", "address", "Colon", "Rest"], fontsize=9)
    left.set_yticks(range(n), range(1, n+1))
    left.set(ylabel="Query head", title="(a) Attention weight")
    for row in range(n):
        for col in range(selected.shape[1]):
            value = selected[row, col]
            left.text(col, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if value>.51 else INK, fontsize=9)
    fig.colorbar(heat, ax=left, shrink=.8, pad=.025, ticks=[0,.25,.5,1])
    mid.barh(np.arange(n), head["write_norms"], color=BLUE, height=.6)
    mid.set(title="(b) Gated write", xlabel="Norm")
    effect = head["remove_effects"]
    right.barh(np.arange(n), effect, color=[ORANGE if x>0 else TEAL for x in effect], height=.6)
    right.axvline(0, linewidth=.9, color=INK)
    right.set(title="(c) Head contribution", xlabel="Native − ablated (nats)",
              xticks=[-.8, -.4, 0, .4])
    for ax in (mid, right):
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(left=False, labelleft=False)
    return fig


def attribution_figure(row):
    fig, (left,right) = plt.subplots(1,2, figsize=(8.8,3.6), layout="constrained",
                                      gridspec_kw={"width_ratios":[1,1.35]})
    x = np.linspace(0,1,120)
    left.plot(x,2*x-x*x,color=BLUE,lw=2)
    left.plot([.58,1.05],[1,1],":",color=ORANGE)
    left.scatter([0,1],[0,1],color=BLUE,zorder=3)
    left.annotate("Slope = 0 at endpoint",(.95,1),xytext=(.43,.62),
                  arrowprops=dict(arrowstyle="->", color=MUTED),fontsize=9)
    left.set(title="(a) Follow the full path",xlabel="Interpolation fraction",ylabel="Teaching score",
             ylim=(-.03,1.13))
    offsets = np.arange(len(row["positions"])+1)
    left_tokens, right_tokens = row["tokens"]
    labels = [f"{a.strip()}\n→ {b.strip()}" for a,b in zip(left_tokens,right_tokens,strict=True)]
    right.bar(offsets-.18,[*row["local"],row["local_sum"]],.34,color=ORANGE,label="Endpoint gradient")
    right.bar(offsets+.18,[*row["ig"],row["ig_sum"]],.34,color=TEAL,label="Integrated gradients")
    right.hlines(row["endpoint_change"],offsets[-1]-.42,offsets[-1]+.42,
                 color=INK,linestyles="--",linewidth=1.2,label="Measured total")
    right.axhline(0,color=INK,linewidth=.8)
    right.set_xticks(offsets,labels+["Sum"])
    right.set(title="(b) Two edits in Qwen",ylabel="Attribution (nats)",
              ylim=(-.5, max(6.1,row["endpoint_change"]+1.1)))
    right.legend(frameon=False,loc="upper left",fontsize=8.6)
    right.grid(axis="y",color=GRID)
    right.set_axisbelow(True)
    return fig


def probe_figure(report):
    fig, ax = plt.subplots(figsize=(8.6,3.25), layout="constrained")
    rows = report["layers"]
    blocks = [row["block"] for row in rows]
    ax.plot(blocks,[row["correct"] for row in rows], "o-",lw=2.1,color=BLUE,label="Fitted probe")
    ax.plot(blocks,[row["shuffled_mean_correct"] for row in rows],"s:",lw=1.7,
            color=ORANGE,label="Shuffled-label mean")
    ax.axhline(report["native_correct"],color=TEAL,ls="--",lw=1.35,label="Native model (A versus B)")
    ax.set(xlabel="Block whose output is used",ylabel="Correct out of 18",
           xticks=blocks,yticks=[0,3,6,9,12,15,18],ylim=(0,19.2))
    ax.legend(loc="lower right",frameon=False)
    ax.grid(axis="y",color=GRID)
    ax.set_axisbelow(True)
    return fig


def lens_figure(report):
    fig, ax = plt.subplots(figsize=(8.6,3.4), layout="constrained")
    depths = report["lens_depths"]
    for label, name, color in ((0,"Intended read",BLUE),(1,"Intended update",ORANGE)):
        scores = np.asarray([[b-a for a,b,_ in row["lens"]] for row in report["test_requests"]
                            if row["label"] == label])
        ax.plot(depths,scores.T,color=color,alpha=.19,lw=.8)
        ax.plot(depths,scores.mean(0),"o-",color=color,lw=2,label=name)
    ax.axhline(0,color=INK,lw=.9,linestyle="--")
    ax.set(xlabel="Block whose output is read",ylabel="Update minus read gap (nats)",xticks=depths)
    ax.legend(frameon=False,loc="upper left")
    ax.grid(axis="y",color=GRID)
    ax.set_axisbelow(True)
    return fig


def circuit_figure():
    fig, ax = diagram(8.8,3.1)
    ax.text(.02,.94,"(a) Test a connection",weight="bold",fontsize=12)
    box(ax,.02,.42,.16,.21,"Sender")
    box(ax,.29,.65,.21,.18,"Receiver 1",accent=TEAL)
    box(ax,.29,.23,.21,.18,"Receiver 2",accent=MUTED)
    arrow(ax,(.18,.55),(.29,.72),TEAL)
    arrow(ax,(.18,.49),(.29,.31),MUTED)
    ax.text(.03,.06,"Edit one route. Hold the other route fixed.",fontsize=10)
    ax.text(.57,.94,"(b) Measure redundancy",weight="bold",fontsize=12)
    ax.text(.60,.73,"Components present",weight="bold",fontsize=10)
    ax.text(.89,.73,"Gap",weight="bold",fontsize=10)
    for i,(label,val) in enumerate((("Both","2"),("Only first","2"),("Only second","2"),("Neither","−2"))):
        y=.59-.117*i
        ax.add_patch(FancyBboxPatch((.59,y-.043),.37,.092,boxstyle="round,pad=.006,rounding_size=.01",
                                    linewidth=0,facecolor=PALE[i]))
        ax.text(.61,y,label,va="center",fontsize=10.5)
        ax.text(.90,y,val,ha="center",va="center",weight="bold")
    return fig


def dictionary_figure(rows):
    fig, (left,right) = plt.subplots(1,2,figsize=(8.8,3.55), layout="constrained",
                                     gridspec_kw={"width_ratios":[1,1.45]})
    for phi,color,tag in zip((0,120,240),(BLUE,TEAL,ORANGE),(1,2,3),strict=True):
        x,y = np.cos(np.deg2rad(phi)),np.sin(np.deg2rad(phi))
        arrow(left,(0,0),(x,y),color)
        left.text(x*1.14,y*1.17,fr"$d_{tag}$",color=color,ha="center",va="center",fontsize=13)
    left.axhline(0,color=GRID,lw=.7,zorder=0)
    left.axvline(0,color=GRID,lw=.7,zorder=0)
    left.set(xlim=(-1.3,1.34),ylim=(-1.18,1.18),aspect="equal",xticks=[],yticks=[],
             title="(a) Three directions in a plane")
    for spine in left.spines.values():
        spine.set_visible(False)
    right.plot([x["mean_active"] for x in rows],[x["relative_squared_error"] for x in rows],
               "o-", color=TEAL, linewidth=1.8)
    for i,row in enumerate(rows):
        offset = (10, 7) if i != 1 else (12,-20)
        right.annotate(f"λ = {row['l1_weight']:.3f}",
                       (row["mean_active"],row["relative_squared_error"]),fontsize=9,
                       textcoords="offset points",xytext=offset)
    right.axhline(1,color=MUTED,ls="--",lw=.9)
    right.set(title="(b) Sparse dictionaries fit to Qwen",xlabel="Mean active features per token",
              ylabel="Relative squared error",xlim=(55,300),ylim=(.43,1.15))
    right.grid(axis="y",color=GRID)
    right.set_axisbelow(True)
    return fig


def steering_figure(record):
    fig, ax = plt.subplots(figsize=(8.6,3.35),layout="constrained")
    cases = record["cases"]
    strength = [x["multiplier"] for x in cases]
    for key,name,color in (("contrast","Mean update − read direction",BLUE),("random","Matched random direction",ORANGE)):
        values = np.asarray([x[key] for x in cases])
        ax.plot(strength,values.mean(1),"o-",color=color,lw=2,label=name)
        ax.fill_between(strength,values.min(1),values.max(1),color=color,alpha=.14)
    ax.axhline(0,color=INK,ls="--",lw=.8)
    ax.set(xlabel="Multiplier on the added direction", ylabel="Update minus read gap (nats)", xticks=strength)
    ax.legend(loc="upper left",frameon=False)
    ax.grid(axis="y",color=GRID)
    ax.set_axisbelow(True)
    return fig


def alignment_system():
    fig, ax = diagram(8.8,2.9)
    xs = (.015,.265,.515,.765)
    nodes = (("Request","Intent and available\nevidence",BLUE),("Route proposal","Read, update,\nor clarify",BLUE),
             ("Independent check","Permissions and\nconfirmation",TEAL),("Outcome","Execute, defer,\nor ask",ORANGE))
    for x,(title,body,c) in zip(xs,nodes,strict=True):
        box(ax,x,.44,.215,.43,title,body,c)
    for x,z in zip(xs[:-1],xs[1:],strict=True):
        arrow(ax,(x+.22,.655),(z-.004,.655))
    arrow(ax,(.873,.425),(.873,.225),ORANGE)
    ax.text(.873,.12,"Log and evaluate outcomes",va="center",ha="center",fontsize=10,weight="bold",color=ORANGE)
    ax.text(.03,.12,"Feedback can train the proposal model",va="center",fontsize=10,color=BLUE)
    return fig


def preference_figure():
    fig, (left,right) = plt.subplots(1,2,figsize=(8.8,3.55),layout="constrained",
                                      gridspec_kw={"width_ratios":[1.1,1]})
    gap=np.linspace(-3,3,121)
    probability=[preference_probability(x,0) for x in gap]
    left.plot(gap,probability,color=BLUE,lw=2,label="Preference probability")
    scale=left.twinx()
    scale.plot(gap,-np.log(probability),color=ORANGE,lw=1.6,ls="--",label="Loss")
    left.set(title="(a) One preferred response", xlabel="Preferred minus rejected reward score",
             ylabel="Preference probability",ylim=(0,1.02))
    scale.set(ylabel="Negative log likelihood",ylim=(0,3.2))
    scale.spines.right.set_visible(True)
    left.legend(left.lines+scale.lines,["Preference probability","Loss"],fontsize=8.4,
                frameon=False,loc="upper center")
    beta=np.geomspace(.18,5,85)
    probabilities=np.array([reward_tilt(np.log([.5,.3,.2]),(0,0,1),b) for b in beta])
    for i,(label,color) in enumerate(zip(("Read","Update","Clarify"),(BLUE,ORANGE,TEAL),strict=True)):
        right.plot(beta,probabilities[:,i],label=label,color=color,lw=2)
    right.axvline(1,color=MUTED,ls=":",lw=.9)
    right.set(title="(b) Three-action optimum", xlabel="Reference penalty β",ylabel="Policy probability",
              xscale="log",ylim=(0,1.03),xticks=[.2,.5,1,2,5],xticklabels=["0.2","0.5","1","2","5"])
    right.minorticks_off()
    right.legend(frameon=False,fontsize=9)
    right.grid(axis="y",color=GRID)
    return fig


def feedback_routes():
    fig, ax = diagram(8.8,3.55)
    columns=(.025,.36,.69)
    rows=(("Demonstrations","A desired answer","Supervised fine-tuning"),
          ("Human comparisons","Preferred and rejected replies","Reward model and RL, or DPO"),
          ("AI comparisons","Judgments under a rubric","AI feedback and preferences"),
          ("Verifiable feedback","Checks on steps or outcomes","Reward or selection"))
    for x,title in zip(columns,("Source","What it produces","How it can train the model"),strict=True):
        ax.text(x,.94,title,weight="bold",fontsize=11)
    for i,row in enumerate(rows):
        y=.72-.19*i
        ax.add_patch(FancyBboxPatch((.013,y),.974,.151,boxstyle="round,pad=.004,rounding_size=.017",
                                    linewidth=0,facecolor=PALE[i]))
        for j,(x,v) in enumerate(zip(columns,row,strict=True)):
            ax.text(x,y+.075,v,va="center",fontsize=10,
                    weight="bold" if j==0 else "normal")
    return fig


def proxy_figure(record):
    rows=reward_curves(record)
    fig, axes = plt.subplots(1,2,figsize=(8.8,3.5),layout="constrained")
    x=[row["strength"] for row in rows]
    for ax,key,name,color,limits in (
        (axes[0],"immediate_action","Immediate action",BLUE,(.83,1.015)),
        (axes[1],"correct","Correct action",TEAL,(0,1.015)),
        (axes[1],"clarify_if_ambiguous","Clarify if ambiguous",ORANGE,(0,1.015)),
    ):
        ax.plot(x,[row[key] for row in rows],"o-",color=color,lw=1.8,label=name)
        ax.set(xlabel="Extra reward for immediate action",xticks=[0,1,2,4,8],ylim=limits)
        ax.grid(axis="y",color=GRID)
        ax.set_axisbelow(True)
    axes[0].set(title="(a) What the dashboard rewards",ylabel="Expected fraction")
    axes[1].set(title="(b) What the task requires",ylabel="Expected fraction")
    for ax in axes:
        ax.legend(frameon=False,fontsize=9)
    return fig


def oversight_grid():
    fig, ax=diagram(8.8,3.75)
    labels = ("Trusted record,\npermission granted", "Trusted record,\nwrite forbidden", "Evidence missing\nor inconsistent")
    values = (
        ("A · return saved address", "A · read if permitted", "A · verify evidence first"),
        ("B · confirm, then update", "B · block write", "B · verify, then check permission"),
        ("C · ask what is wanted", "C · ask what is wanted", "C · ask or hand off"),
    )
    columns=(.16,.44,.72)
    for x, label in zip(columns,labels,strict=True):
        ax.text(x+.125,.90,label,ha="center",weight="bold",fontsize=10)
    for i,(action,cells) in enumerate(zip(("Read","Update","Unclear"),values,strict=True)):
        y=.66-.22*i
        ax.text(.016,y+.076,action,va="center",weight="bold")
        for j,(x,cell) in enumerate(zip(columns,cells,strict=True)):
            c=PALE[1] if (i,j) in ((0,0),(1,0)) else "#eef0f2"
            box(ax,x,y,.25,.145,cell,fill=c,accent=INK,title_size=8.8)
    ax.text(.16,.08,"Label = intent. Text after the label = action in the system.",fontsize=10,color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods",type=Path,required=True)
    parser.add_argument("--routing",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    methods=json.loads(args.methods.read_text())
    routing=json.loads(args.routing.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        figures={
            "evidence_questions":question_map(),
            "attention_evidence":attention_figure(methods["head"]),
            "gradient_path":attribution_figure(methods["attribution"]),
            "probe_depth":probe_figure(methods["probes"]),
            "logit_lens":lens_figure(methods),
            "circuit_tests":circuit_figure(),
            "sparse_dictionary":dictionary_figure(methods["dictionary"]),
            "steering_controls":steering_figure(methods["steering"]),
            "alignment_system":alignment_system(),
            "preference_objectives":preference_figure(),
            "feedback_routes":feedback_routes(),
            "proxy_tradeoff":proxy_figure(routing),
            "oversight_grid":oversight_grid(),
        }
        for name,figure in figures.items():
            figure.savefig(args.output/f"ch18_{name}.pdf",metadata={"CreationDate":None})
            plt.close(figure)


if __name__=="__main__":
    main()
