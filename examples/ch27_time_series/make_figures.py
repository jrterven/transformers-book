"""Draw the Chapter 27 figures from the saved CPU forecasts and EIA snapshot."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator, PercentFormatter
import numpy as np

from .demand_data import HOUR, SNAPSHOT, load_snapshot
from .run_pretrained import file_hash


INK = "#223a4f"
MUTED = "#526777"
GRID = "#dde5ea"
BLUE = "#27628e"
ORANGE = "#a34f28"
GRAY = "#74818b"
GREEN = "#236953"
AREA_NAMES = {"NYIS": "New York (NYIS)", "ERCO": "Texas (ERCO)"}
NAMES = {"timesfm3": "TimesFM 3", "timestral": "Timestral-28M",
         "weekly": "Weekly repeat", "daily": "Daily repeat"}
COLORS = {"timesfm3": BLUE, "timestral": ORANGE, "weekly": GRAY, "daily": GREEN}
STYLES = {"timesfm3": "-", "timestral": "-", "weekly": "--", "daily": ":"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "axes.edgecolor": GRID, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, output, name):
    fig.savefig(output / name, bbox_inches="tight", pad_inches=0.07,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def date_at(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.7, color=MUTED)


def draw_forecast_day(report, hours, demand, output):
    fig, axes = plt.subplots(2, 2, figsize=(8.55, 5.25), sharex=True, sharey="row")
    for row, (area, name) in enumerate(AREA_NAMES.items()):
        record = report["areas"][area]
        issue = date_at(record["test_issues_utc"][0])
        position = hours.index(issue)
        actual = np.asarray(record["actual_mw"][0]) / 1000
        future = np.arange(1, len(actual) + 1)
        past = np.arange(-24, 0)
        observed = demand[area][position + past] / 1000
        for col, model in enumerate(("timesfm3", "timestral")):
            ax = axes[row, col]
            f = record["forecasts"][model]
            lower, median, upper = (np.asarray(f[key][0]) / 1000 for key in ("lower_mw", "median_mw", "upper_mw"))
            miss = (actual < lower) | (actual > upper)
            color = COLORS[model]
            ax.axvspan(-1, 1, color=GRID, alpha=0.45, zorder=0, linewidth=0)
            ax.axvline(0, color=MUTED, linewidth=0.9, linestyle="--")
            ax.fill_between(future, lower, upper, color=color, alpha=0.16, linewidth=0)
            ax.plot(past, observed, color=INK, linewidth=1.55)
            ax.plot(future, actual, color=INK, linewidth=1.55)
            ax.plot(future, median, color=color, linewidth=1.85, linestyle=(0, (5, 2)))
            ax.scatter(future[miss], actual[miss], color=color, s=28, marker="x", zorder=5, linewidth=1.2)
            ax.set_title(NAMES[model] if row == 0 else "", loc="left", fontweight="bold", pad=6)
            ax.set_xlim(-24, 24.5)
            ax.xaxis.set_major_locator(FixedLocator([-24, -12, 0, 12, 24]))
            style_axis(ax)
        axes[row, 0].set_ylabel(f"{name}\nDemand (GW)")
    fig.supxlabel("Hours relative to issue at 00:00 UTC on October 30, 2026", y=0.025, fontsize=10.5)
    legend = [Line2D([], [], color=INK, lw=1.6, label="Reported demand"),
              Line2D([], [], color=MUTED, lw=1.6, ls="--", label="Median forecast"),
              Line2D([], [], color=MUTED, lw=7, alpha=0.23, label="Central 80% interval"),
              Line2D([], [], color=MUTED, marker="x", ls="", label="Interval miss")]
    fig.legend(handles=legend, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=.125, right=.99, bottom=.12, top=.85, wspace=.13, hspace=.22)
    save(fig, output, "ch27_demand_forecast.pdf")


def draw_horizon_scores(report, output):
    fig, axes = plt.subplots(2, 2, figsize=(8.55, 4.7), sharex=True)
    h = np.arange(1, report["horizon_hours"] + 1)
    for col, (area, name) in enumerate(AREA_NAMES.items()):
        top, bottom = axes[:, col]
        forecasts = report["areas"][area]["forecasts"]
        for key in ("daily", "weekly", "timesfm3", "timestral"):
            stats = forecasts[key]["metrics"]
            top.plot(h, np.asarray(stats["mae_by_hour_mw"]) / 1000, color=COLORS[key],
                     lw=1.85, ls=STYLES[key], label=NAMES[key])
            if "coverage_by_hour" in stats:
                bottom.plot(h, stats["coverage_by_hour"], color=COLORS[key], lw=1.65, ls=STYLES[key])
        top.set_title(name, loc="left", fontweight="bold", pad=6)
        top.set_ylabel("Mean absolute error (GW)")
        top.set_ylim(bottom=0)
        bottom.axhline(0.8, color=INK, lw=0.85, ls=(0, (2, 3)))
        bottom.set_ylabel("Empirical coverage")
        bottom.set_ylim(-0.02, 1.04)
        bottom.yaxis.set_major_locator(MultipleLocator(0.25))
        bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        for ax in (top, bottom):
            ax.set_xlim(0.8, 24.2)
            ax.xaxis.set_major_locator(FixedLocator([1, 6, 12, 18, 24]))
            style_axis(ax)
    fig.supxlabel("Scored horizon (hours after issue)", y=0.025, fontsize=10.5)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.53, 1.0))
    fig.subplots_adjust(left=.107, right=.99, bottom=.135, top=.84, wspace=.28, hspace=.22)
    save(fig, output, "ch27_demand_horizons.pdf")


def draw_width_and_coverage(report, output):
    fig, axes = plt.subplots(1, 2, figsize=(8.55, 3.3), sharey=True)
    for ax, (area, name) in zip(axes, AREA_NAMES.items()):
        results = report["areas"][area]["forecasts"]
        for key, marker in (("weekly", "s"), ("timesfm3", "o"), ("timestral", "D")):
            base = results[key]["metrics"]
            ax.scatter(base["width_mw"] / 1000, base["coverage"], color=COLORS[key], marker=marker,
                       edgecolor="white", linewidth=1.0, s=75, zorder=4)
            if key + "_calendar" in results:
                with_calendar = results[key + "_calendar"]["metrics"]
                before = (base["width_mw"] / 1000, base["coverage"])
                after = (with_calendar["width_mw"] / 1000, with_calendar["coverage"])
                ax.plot([before[0], after[0]], [before[1], after[1]], color=COLORS[key], lw=1.1, alpha=.8)
                ax.scatter(*after, facecolor="none", edgecolor=COLORS[key], marker=marker,
                           linewidth=1.3, s=54, zorder=5)
        ax.set_title(name, loc="left", fontweight="bold", pad=6)
        ax.margins(x=.13)
        ax.set_ylim(0.55, 1.0)
        ax.axhline(.8, color=INK, lw=.85, ls=(0, (2, 3)))
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Mean interval width (GW)")
        style_axis(ax)
    axes[0].set_ylabel("Empirical coverage")
    handles = [Line2D([], [], color=COLORS[k], marker=m, ls="", label=NAMES[k])
               for k, m in (("weekly", "s"), ("timesfm3", "o"), ("timestral", "D"))]
    handles.append(Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", ls="", label="With calendar"))
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.53, 1.02))
    fig.subplots_adjust(left=.105, right=.99, top=.77, bottom=.18, wspace=.16)
    save(fig, output, "ch27_demand_interval_tradeoff.pdf")


def draw_calendar_change(report, output):
    data = report["areas"]["NYIS"]
    issue = "2026-11-01T00:00:00Z"
    row = data["test_issues_utc"].index(issue)
    local = ZoneInfo(data["timezone"])
    h = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(8.55, 3.2))
    actual = np.array(data["actual_mw"][row]) / 1000
    base, calendar = (np.asarray(data["forecasts"][key]["median_mw"][row]) / 1000
                      for key in ("timesfm3", "timesfm3_calendar"))
    ax.axvspan(5.5, 7.5, color="#eaf1f5", lw=0)
    ax.plot(h, actual, color=INK, lw=1.9, label="Reported demand")
    ax.plot(h, base, color=BLUE, lw=1.65, ls="--", label="History only")
    ax.plot(h, calendar, color=GREEN, lw=1.8, ls=(0, (5, 2, 1, 2)), label="History and calendar")
    ax.set(xlim=(1, 24), xlabel="UTC hour ending", ylabel="Demand (GW)", xticks=[1, 6, 12, 18, 24])
    ax.set_xticklabels(["01:00", "06:00", "12:00", "18:00", "00:00"])
    local_start = [(date_at(issue) + (i - 1) * HOUR).astimezone(local) for i in (6, 7)]
    for end_hour, local_hour, offset, align in zip((6, 7), local_start, (-8, 8), ("right", "left")):
        label = f"{end_hour:02d}:00 UTC\n{local_hour:%H:%M %Z} start"
        ax.annotate(label, xy=(end_hour, 1), xycoords=("data", "axes fraction"), xytext=(offset, 10),
                    textcoords="offset points", ha=align, va="bottom", fontsize=8.7, color=MUTED,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": .65})
    style_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.59, 1.38), ncol=3)
    assert [(x.hour, x.fold) for x in local_start] == [(1, 0), (1, 1)]
    fig.subplots_adjust(left=.11, right=.98, top=.75, bottom=.17)
    save(fig, output, "ch27_demand_calendar.pdf")


def main(args):
    report = json.loads(args.record.read_text(encoding="utf-8"))
    if file_hash(args.data) != report["snapshot_sha256"]:
        raise ValueError("Forecasts must be plotted with the snapshot used at inference")
    hours, demand = load_snapshot(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    draw_forecast_day(report, hours, demand, args.output)
    draw_horizon_scores(report, args.output)
    draw_width_and_coverage(report, args.output)
    draw_calendar_change(report, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
