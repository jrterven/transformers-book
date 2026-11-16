"""Render textbook diagrams and the native tool-call traces as vector PDFs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK = "#172e46"
BLUE = "#245a80"
TEAL = "#08736a"
ORANGE = "#a94924"
GRAY = "#526476"
PALE_BLUE = "#edf4fa"
PALE_TEAL = "#eaf5f0"
PALE_ORANGE = "#fbefe7"
LINE = "#c7d5df"


def canvas(height=3.1):
    fig, ax = plt.subplots(figsize=(8.7, height))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.subplots_adjust(left=.018, right=.982, bottom=.025, top=.975)
    return fig, ax


def heading(ax, x, y, title):
    ax.text(x, y, title, weight="bold", fontsize=12, color=INK, va="center")


def label(ax, x, y, value, color=GRAY, ha="left", size=10):
    ax.text(x, y, value, color=color, fontsize=size, ha=ha, va="center", linespacing=1.2)


def card(ax, x, y, w, h, title, details="", accent=BLUE, fill=PALE_BLUE, size=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                         linewidth=1, edgecolor=accent, facecolor=fill)
    ax.add_patch(box)
    center = x + w / 2
    if details:
        ax.text(center, y + .72 * h, title, ha="center", va="center", color=accent,
                fontsize=size, weight="bold")
        label(ax, center, y + .29 * h, details, color=INK, ha="center", size=9.8)
    else:
        ax.text(center, y + .5 * h, title, ha="center", va="center", color=accent,
                fontsize=size, weight="bold", linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, color=GRAY, curve=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.18,
                            "shrinkA": 1, "shrinkB": 1, "connectionstyle": f"arc3,rad={curve}"})


def flow(ax, x, y, w, h, items, gap=.023, palette=None):
    width = (w - (len(items)-1)*gap) / len(items)
    if palette is None:
        palette = [(BLUE, PALE_BLUE)] * len(items)
    for i, (item, (color, fill)) in enumerate(zip(items, palette, strict=True)):
        args = item if isinstance(item, tuple) else (item, "")
        start = x+i*(width+gap)
        card(ax, start, y, width, h, *args, accent=color, fill=fill)
        if i:
            arrow(ax, start-gap, y+h/2, start, y+h/2)
    return width


def workflow_loop():
    fig, ax = canvas(3.55)
    heading(ax, .03, .92, "(a) Fixed retrieval workflow")
    flow(ax, .03, .64, .94, .17, ("Question", "Search", "Generate", "Reply"),
         palette=[(BLUE, PALE_BLUE), (TEAL, PALE_TEAL), (BLUE, PALE_BLUE), (ORANGE, PALE_ORANGE)])
    heading(ax, .03, .52, "(b) Loop with a model-selected action")
    flow(ax, .03, .24, .94, .17, ("Model proposes", "Runner checks", "Tool returns", "Observation"),
         palette=[(BLUE, PALE_BLUE), (TEAL, PALE_TEAL), (TEAL, PALE_TEAL), (BLUE, PALE_BLUE)])
    ax.plot([.85, .85, .14], [.24, .10, .10], color=TEAL, lw=1.2)
    arrow(ax, .14, .10, .14, .24, TEAL)
    label(ax, .51, .047, "The model sees the observation before deciding again.", TEAL, "center")
    return fig


def tool_contract():
    fig, ax = canvas(2.6)
    flow(ax, .025, .26, .95, .54, [
        ("Model proposal", "quote_total\n(12, 875, 200)"),
        ("Runner", "Check name, keys\nand integer bounds"),
        ("Python function", "12 × 875 + 200"),
        ("Tool result", "total_cents = 10700"),
    ], gap=.044, palette=[(BLUE, PALE_BLUE), (TEAL, PALE_TEAL),
                         (TEAL, PALE_TEAL), (ORANGE, PALE_ORANGE)])
    label(ax, .025, .09, "Arguments are a request. The returned integer is an observed result.")
    return fig


def after_notice():
    fig, ax = canvas(3.05)
    heading(ax, .035, .93, "Same request and six available calls")
    card(ax, .035, .41, .20, .32, "Read catalog", "N-17 pointer\nFive calls left")
    card(ax, .30, .41, .20, .32, "Read notice", "Four calls left")
    arrow(ax, .235, .57, .30, .57)
    card(ax, .63, .57, .33, .27, "Status active", "Read R-MAPLE, then quote", TEAL, PALE_TEAL)
    card(ax, .63, .19, .33, .27, "Status cancelled", "Tell the visitor and stop", ORANGE, PALE_ORANGE)
    arrow(ax, .50, .59, .63, .71, TEAL)
    arrow(ax, .50, .54, .63, .33, ORANGE)
    label(ax, .30, .12, "The history changes at the notice result.")
    return fig


def architecture_patterns():
    fig, ax = canvas(4.3)
    heading(ax, .02, .92, "(a) Pipeline")
    flow(ax, .02, .66, .45, .16, ("Extract", "Check", "Reply"))
    heading(ax, .53, .92, "(b) Routing")
    card(ax, .54, .67, .15, .16, "Route")
    card(ax, .77, .77, .20, .115, "Workshop", accent=TEAL, fill=PALE_TEAL)
    card(ax, .77, .62, .20, .115, "Registration", accent=ORANGE, fill=PALE_ORANGE)
    arrow(ax, .69, .76, .77, .82)
    arrow(ax, .69, .73, .77, .67)
    ax.axhline(.53, xmin=.025, xmax=.972, color=LINE, lw=1)
    ax.axvline(.50, ymin=.07, ymax=.98, color=LINE, lw=1)
    heading(ax, .02, .43, "(c) Parallel reads")
    card(ax, .03, .27, .16, .115, "Room")
    card(ax, .03, .12, .16, .115, "Rate")
    card(ax, .28, .19, .18, .13, "Combine", accent=TEAL, fill=PALE_TEAL)
    arrow(ax, .19, .33, .28, .26)
    arrow(ax, .19, .175, .28, .24)
    heading(ax, .53, .43, "(d) Single-agent loop")
    card(ax, .55, .22, .16, .14, "Model")
    card(ax, .79, .22, .16, .14, "Tool", accent=TEAL, fill=PALE_TEAL)
    arrow(ax, .71, .32, .79, .32)
    arrow(ax, .79, .26, .71, .26, TEAL)
    label(ax, .75, .10, "Observe before the next decision", TEAL, "center", 9.6)
    return fig


def delegation():
    fig, ax = canvas(3.9)
    heading(ax, .035, .93, "(a) Lead delegates")
    card(ax, .08, .64, .18, .18, "Lead")
    card(ax, .04, .26, .125, .16, "Room", accent=TEAL, fill=PALE_TEAL)
    card(ax, .185, .26, .125, .16, "Price", accent=TEAL, fill=PALE_TEAL)
    arrow(ax, .14, .64, .105, .42)
    arrow(ax, .20, .64, .24, .42)
    label(ax, .165, .113, "Lead assembles results", ha="center", size=9.6)
    heading(ax, .365, .93, "(b) Handoff")
    card(ax, .395, .64, .20, .18, "Desk")
    card(ax, .395, .26, .20, .16, "Specialist", accent=TEAL, fill=PALE_TEAL)
    arrow(ax, .495, .64, .495, .42)
    label(ax, .495, .51, "Task and evidence", ha="center", size=9)
    label(ax, .495, .113, "Specialist acts next", ha="center", size=9.6)
    heading(ax, .69, .93, "(c) Check and revise")
    card(ax, .72, .64, .22, .18, "Draft")
    card(ax, .72, .26, .22, .16, "Verifier", accent=ORANGE, fill=PALE_ORANGE)
    arrow(ax, .80, .64, .80, .42)
    arrow(ax, .89, .42, .89, .64, ORANGE)
    label(ax, .83, .113, "Check records and claims", ha="center", size=9.6)
    ax.axvline(.335, ymin=.05, ymax=.98, color=LINE)
    ax.axvline(.66, ymin=.05, ymax=.98, color=LINE)
    return fig


def state_memory():
    fig, ax = canvas(2.9)
    heading(ax, .025, .92, "Input for the next model call")
    flow(ax, .025, .53, .95, .23,
         ("User request", "Tool schemas", "Verified facts", "Latest result"),
         palette=[(BLUE, PALE_BLUE), (BLUE, PALE_BLUE), (TEAL, PALE_TEAL), (TEAL, PALE_TEAL)])
    box = FancyBboxPatch((.025, .12), .95, .22, boxstyle="round,pad=0.008,rounding_size=.012",
                         lw=1, edgecolor=LINE, facecolor="#f7f9fa")
    ax.add_patch(box)
    heading(ax, .045, .23, "Audit log")
    label(ax, .25, .23, "Entire call history, record versions, results and limits")
    arrow(ax, .63, .34, .63, .53, TEAL)
    return fig


def runner_checks():
    fig, ax = canvas(3.3)
    heading(ax, .025, .92, "One decision in the runner")
    card(ax, .025, .49, .18, .26, "Model result")
    card(ax, .272, .57, .211, .15, "Final reply", accent=BLUE)
    card(ax, .272, .30, .211, .16, "Proposed call", accent=TEAL, fill=PALE_TEAL)
    card(ax, .552, .30, .19, .16, "Check budget", accent=ORANGE, fill=PALE_ORANGE)
    card(ax, .810, .30, .16, .16, "Dispatch", accent=TEAL, fill=PALE_TEAL)
    arrow(ax, .205, .65, .272, .65)
    arrow(ax, .205, .56, .272, .38)
    arrow(ax, .483, .38, .552, .38)
    arrow(ax, .742, .38, .810, .38, TEAL)
    label(ax, .647, .18, "Stop at limit", ORANGE, "center", 9.7)
    label(ax, .890, .18, "Validate arguments", TEAL, "center", 9.7)
    label(ax, .025, .085, "The runner records the result or a validation error.")
    return fig


def display_step(step, final):
    if step["event"] == "final":
        if final["status"] == "cancelled":
            return "Final reply", "Cancelled"
        return "Final reply", "$" + f"{final['total_cents'] / 100:,.2f}"
    tool = step["tool"]
    args = step["arguments"]
    obs = step["observation"]
    if tool == "find_session":
        row = obs["session"]
        return "Find session", f"{row['id']}\nPoints to {row['revision_id']}"
    if tool == "read_notice":
        text = ("Cancelled" if obs["status"] == "cancelled"
                else f"{obs['room_id']}\n{obs['rate_cents']} cents/person")
        return f"Read {args['notice_id']}", text
    if tool == "room_status":
        return f"Read {args['room_id']}", f"Open, capacity {obs['capacity']}"
    if tool == "quote_total":
        return "Calculate quote", f"{args['party_size']} × {args['rate_cents']} + {args['fee_cents']}\n{obs['total_cents']:,} cents"
    raise ValueError(f"unexpected measured tool {tool}")


def measured_traces(rows):
    fig, ax = canvas(3.6)
    for row, y, name, col, fill in (
        (rows["camera_updated"], .55, "Notice reports a room and rate change", TEAL, PALE_TEAL),
        (rows["camera_cancelled"], .12, "Notice reports a cancellation", ORANGE, PALE_ORANGE),
    ):
        heading(ax, .028, y+.34, name)
        for i, step in enumerate(row["trace"]):
            x = .028+i*.192
            title, detail = display_step(step, row["final"])
            card(ax, x, y, .172, .227, title, detail,
                 col if i>=1 else BLUE, fill if i>=1 else PALE_BLUE, size=10)
            if i:
                arrow(ax, x-.02, y+.113, x, y+.113)
    label(ax, .028, .04, "Both runs begin with the same visitor request and catalog record.", size=10)
    return fig


def run_summary(rows):
    fig, ax = canvas(3.8)
    heading(ax, .035, .93, "Visitor request")
    heading(ax, .40, .93, "Calls")
    heading(ax, .74, .93, "Full rubric")
    names = ("Camera Lab, moved", "Camera Lab, cancelled", "Paper Making, 12 guests",
             "Ink Workshop, five guests", "Sketch Club, two matches", "Dye Studio, nine guests")
    for i, (name, row) in enumerate(zip(names, rows, strict=True)):
        y = .79-.124*i
        if i % 2 == 0:
            ax.axhspan(y-.05, y+.05, color="#f3f6f8", zorder=0)
        label(ax, .035, y, name, color=INK)
        calls = row["executed_calls"]
        color = TEAL if row["success"] else ORANGE
        ax.barh(y, .055*calls, height=.034, left=.40, color=color)
        label(ax, .41+.055*calls, y, str(calls), color=color)
        label(ax, .74, y, "Pass" if row["success"] else "Evidence missing", color=color)
    return fig


def permissions_recovery():
    fig, ax = canvas(3.35)
    heading(ax, .025, .93, "A write requires a separate authorization step")
    flow(ax, .025, .48, .95, .28, [
        ("Proposal", "Tool, record,\narguments"),
        ("Runner checks", "Permission and\nmatching approval"),
        ("Service", "Execute under\napproved scope"),
        ("Outcome", "Receipt or state\nconfirmation"),
    ], gap=.035, palette=[(BLUE, PALE_BLUE), (ORANGE, PALE_ORANGE),
                         (TEAL, PALE_TEAL), (TEAL, PALE_TEAL)])
    arrow(ax, .88, .46, .88, .27, ORANGE)
    label(ax, .97, .15, "If the outcome is uncertain, check state before retrying.", ORANGE, "right")
    return fig


def evaluation():
    fig, ax = canvas(3.4)
    heading(ax, .022, .93, "Evidence for scoring a complete episode")
    for i, (title, criterion, color, fill) in enumerate((
        ("Request and reference", "What result would satisfy the task?", BLUE, PALE_BLUE),
        ("Action and observation trace", "What did the system actually do?", TEAL, PALE_TEAL),
        ("Final reply or changed state", "What outcome reached the user?", ORANGE, PALE_ORANGE),
    )):
        y=.66-i*.22
        card(ax, .025, y, .35, .15, title, accent=color, fill=fill, size=10.5)
        arrow(ax, .375, y+.07, .425, y+.07, color)
        label(ax, .445, y+.075, criterion, color=INK, size=11)
    return fig


def plan_dependencies():
    fig, ax = canvas(3.65)
    heading(ax, .025, .92, "Independent work starts after the dependency is resolved")
    card(ax, .028, .51, .18, .22, "Read notice", "0.3 s")
    card(ax, .37, .63, .21, .17, "Venue directory", "0.4 s", TEAL, PALE_TEAL)
    card(ax, .37, .36, .21, .17, "Parking record", "0.7 s", TEAL, PALE_TEAL)
    card(ax, .76, .51, .19, .22, "Assemble", "0.2 s")
    arrow(ax, .208, .65, .37, .715, TEAL)
    arrow(ax, .208, .60, .37, .445, TEAL)
    arrow(ax, .58, .715, .76, .66, TEAL)
    arrow(ax, .58, .445, .76, .56, TEAL)
    label(ax, .315, .75, "Active", TEAL, "right", 9.5)
    card(ax, .37, .075, .25, .135, "Report cancellation", accent=ORANGE, fill=PALE_ORANGE)
    ax.plot([.12, .12, .31], [.51, .145, .145], color=ORANGE, lw=1.2)
    arrow(ax, .31, .145, .37, .145, ORANGE)
    label(ax, .16, .19, "Cancelled", ORANGE, size=9.4)
    return fig


def token_trace(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.32), layout="constrained")
    for suffix, name, col, form in (
        ("updated", "Room and rate change", TEAL, "o-"),
        ("cancelled", "Cancellation", ORANGE, "s--"),
    ):
        trace = rows["camera_"+suffix]["trace"]
        for ax, key in zip(axes, ("prompt_tokens", "generated_tokens"), strict=True):
            ax.plot([step["turn"] for step in trace], [step[key] for step in trace],
                    form, color=col, lw=2, label=name)
    for ax, title in zip(axes, ("Input tokens", "Generated tokens"), strict=True):
        ax.set(title=title, xlabel="Model turn", xticks=[1, 2, 3, 4, 5], xlim=(.85, 5.15))
        ax.tick_params(labelcolor=INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=LINE, linewidth=.75)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.1)
    return fig


def horizon_timeline():
    fig, ax = canvas(2.85)
    heading(ax, .04, .91, "Workshop report due at 12:45")
    segments = (
        (0, 15, BLUE, PALE_BLUE),
        (15, 180, ORANGE, PALE_ORANGE),
        (180, 215, TEAL, PALE_TEAL),
        (215, 225, BLUE, PALE_BLUE),
    )
    ticks = {minute: .045 + .9 * minute / 225 for minute in (0, 15, 180, 215, 225)}
    for start, end, edge, fill in segments:
        ax.add_patch(Rectangle((ticks[start], .49), ticks[end] - ticks[start], .18,
                               edgecolor=edge, facecolor=fill, linewidth=1.2))
    label(ax, ticks[0], .76, "Prepare", BLUE, size=10.5)
    label(ax, (ticks[15] + ticks[180]) / 2, .58, "Wait for facilities notice · 165 min",
          ORANGE, "center", 11.2)
    label(ax, (ticks[180] + ticks[215]) / 2, .76, "Verify", TEAL, "center", 10.5)
    label(ax, .98, .76, "Assemble", BLUE, "right", 10.5)
    for minute, clock, align, height in (
        (0, "9:00", "right", .38), (15, "9:15", "center", .38),
        (180, "12:00", "center", .38), (215, "12:35", "right", .38),
        (225, "12:45", "center", .27),
    ):
        x = ticks[minute]
        ax.plot([x, x], [.48, height + .065], color=LINE, lw=1)
        label(ax, x, height, clock, INK, align, 9.2)
    label(ax, ticks[15], .13, "Save checkpoint", BLUE, "center")
    label(ax, ticks[180], .13, "Resume with current records", TEAL, "center")
    return fig


def horizon_resume(report):
    fig, ax = canvas(4.55)
    waiting = report["waiting"]
    heading(ax, .025, .935, f"Same checkpoint at simulated minute {waiting['wake_at']}")
    label(ax, .025, .865, f"Catalog and pending notice already saved · {waiting['executed_calls']} calls")
    for key, title, y in (
        ("baseline", "(a) Original continuation", .51),
        ("guarded", "(b) Current-room rule enabled", .13),
    ):
        run = report[key]
        heading(ax, .025, y + .26, title)
        items, colors = [], []
        for step in run["trace"]:
            if step["minute"] != run["minute"]:
                continue
            if step["event"] == "proposal_rejected":
                items.append(("Room denied", f"Proposed {step['arguments']['room_id']}\nRead current notice"))
                colors.append((ORANGE, PALE_ORANGE))
            elif step["event"] == "tool_result":
                observation = step["observation"]
                if step["tool"] == "room_status":
                    room = observation["id"]
                    current = room == run["final"]["room_id"]
                    items.append(("Check room", f"{room}\nCapacity {observation['capacity']}"))
                    colors.append((TEAL, PALE_TEAL) if current else (ORANGE, PALE_ORANGE))
                elif step["tool"] == "read_notice":
                    items.append(("Read notice", f"{observation['id']} active\nRoom {observation['room_id']}"))
                    colors.append((TEAL, PALE_TEAL))
                elif step["tool"] == "quote_total":
                    items.append(("Quote total", f"{observation['total_cents'] / 100:.2f} USD"))
                    colors.append((TEAL, PALE_TEAL))
            elif step["event"] == "final":
                if run["success"]:
                    items.append(("Verified", "Current room\nand quote supported"))
                    colors.append((TEAL, PALE_TEAL))
                else:
                    items.append(("Unverified", "Missing current\nroom evidence"))
                    colors.append((ORANGE, PALE_ORANGE))
        flow(ax, .025, y, .19 * len(items) - .016, .205, items, gap=.021, palette=colors)
    return fig


def horizon_thresholds():
    fig, ax = plt.subplots(figsize=(8.7, 3.0))
    minutes = list(range(1, 181))
    ax.plot(minutes, [1 / (1 + (minute / 60) ** 2) for minute in minutes],
            color=BLUE, lw=2.2)
    for minute, probability, color in ((30, .8, TEAL), (60, .5, ORANGE)):
        ax.plot((0, minute, minute), (probability, probability, 0), linestyle="--",
                color=color, alpha=.72, lw=1.1)
        ax.scatter([minute], [probability], color=color, s=28, zorder=3)
        ax.annotate(f"{probability:.0%} at {minute} min", (minute, probability),
                    xytext=(10, 9), textcoords="offset points", fontsize=11,
                    color=color, weight="bold")
    ax.set(xlim=(0, 180), ylim=(0, 1.04), xticks=[0, 30, 60, 90, 120, 150, 180],
           xlabel="Human reference time (minutes)", ylabel="Agent success probability")
    ax.set_yticks([0, .2, .5, .8, 1], ["0%", "20%", "50%", "80%", "100%"])
    ax.tick_params(labelcolor=INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=LINE, linewidth=.7, alpha=.65)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--horizon-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.run.read_text())["runs"]
    rows = {row["case"]: row for row in data}
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.unicode_minus": False}
    with plt.rc_context(settings):
        figures = {
            "workflow-and-loop": workflow_loop(), "tool-call-contract": tool_contract(),
            "branch-after-notice": after_notice(), "workflow-patterns": architecture_patterns(),
            "delegation-and-review": delegation(), "agent-memory": state_memory(),
            "runner-checks": runner_checks(), "measured-traces": measured_traces(rows),
            "pilot-results": run_summary(data), "permissions-and-recovery": permissions_recovery(),
            "evaluate-agent": evaluation(), "plan-dependencies": plan_dependencies(),
            "reported-tokens": token_trace(rows),
            "horizon-timeline": horizon_timeline(),
            "horizon-thresholds": horizon_thresholds(),
        }
        if args.horizon_run:
            figures["horizon-resume"] = horizon_resume(json.loads(args.horizon_run.read_text()))
        for name, fig in figures.items():
            fig.savefig(args.output/f"ch32_{name}.pdf", metadata={"CreationDate": None})
            plt.close(fig)


if __name__ == "__main__":
    main()
