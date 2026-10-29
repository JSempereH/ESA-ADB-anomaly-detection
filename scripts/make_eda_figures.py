"""Generate clean, minimal EDA figures for the paper (anomaly vs. rare-event examples
over time, and event-duration distributions), in a monochromatic-first, colorblind-safe
pastel style. Saves PDF (for LaTeX \\includegraphics) and PNG (for quick viewing) into
paper/figures/.
"""
from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")
sys.path.insert(0, "src")
from esa_adb.benchmark import load_benchmark_data  # noqa: E402
from esa_adb.evaluation import anomalies_only  # noqa: E402

INK = "#2b2b33"  # near-black, warm charcoal (monochrome signal line)
GRID = "#e7e7ea"
ANOMALY_FILL = "#f6b8ac"  # pastel vermillion (Okabe-Ito, lightened) -- colorblind-safe
ANOMALY_EDGE = "#d55e00"
RARE_FILL = "#bfe3f0"  # pastel sky blue (Okabe-Ito, lightened) -- colorblind-safe
RARE_EDGE = "#0072b2"
FONT = "DejaVu Sans"  # portable sans-serif stand-in for a Helvetica/Inter-like face

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 9,
        "axes.edgecolor": "#55555f",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999aa3")
    ax.spines["bottom"].set_color("#999aa3")
    ax.tick_params(colors="#55555f", length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def fig_examples_over_time(data, anoms, rare):
    channel = "channel_13"
    anomaly_row = anoms.filter(pl.col("ID") == "id_619").filter(pl.col("Channel") == channel).row(0, named=True)
    rare_row = rare.filter(pl.col("ID") == "id_5").filter(pl.col("Channel") == channel).row(0, named=True)

    df = data.lf.select(["time", channel]).collect().sort("time")
    t = df["time"].to_numpy()
    v = df[channel].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.2), sharex=False)

    for ax, row, fill, edge, label in (
        (axes[0], anomaly_row, ANOMALY_FILL, ANOMALY_EDGE, "Anomaly"),
        (axes[1], rare_row, RARE_FILL, RARE_EDGE, "Rare event"),
    ):
        start, end = np.datetime64(row["StartTime"]), np.datetime64(row["EndTime"])
        span = (end - start) / np.timedelta64(1, "s")
        pad = np.timedelta64(int(max(span * 2.5, 1800)), "s")
        lo, hi = start - pad, end + pad
        mask = (t >= lo) & (t <= hi)
        tt, vv = t[mask], v[mask]
        tt_hours = (tt - lo) / np.timedelta64(1, "h")
        ax.plot(tt_hours, vv, color=INK, linewidth=1.1, zorder=3)
        x0 = (start - lo) / np.timedelta64(1, "h")
        x1 = (end - lo) / np.timedelta64(1, "h")
        ax.axvspan(x0, x1, color=fill, alpha=0.9, zorder=1, lw=0)
        ax.axvline(x0, color=edge, linewidth=0.9, zorder=2)
        ax.axvline(x1, color=edge, linewidth=0.9, zorder=2)
        ax.text(
            0.02, 0.92, f"{label}  ({span/60:.0f} min)", transform=ax.transAxes,
            fontsize=8, color=edge, fontweight="bold", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )
        ax.set_xlabel("hours", fontsize=7.5, color="#55555f")
        ax.set_ylabel(channel, fontsize=7.5, color="#55555f")
        _clean_axes(ax)

    fig.suptitle("A real anomaly runs longer than a rare event", fontsize=9.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("paper/figures/eda_examples.pdf", bbox_inches="tight")
    fig.savefig("paper/figures/eda_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_duration_distribution(anoms, rare):
    rows_anom = (anoms["duration_seconds"].to_numpy() / 18.0)
    rows_rare = (rare["duration_seconds"].to_numpy() / 18.0)
    rows_anom = rows_anom[rows_anom > 0]
    rows_rare = rows_rare[rows_rare > 0]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bins = np.logspace(0, np.log10(max(rows_anom.max(), rows_rare.max())), 26)
    ax.hist(rows_rare, bins=bins, color=RARE_FILL, edgecolor=RARE_EDGE, linewidth=0.8,
            alpha=0.85, density=True, label=f"rare event (n={len(rows_rare)})", zorder=2)
    ax.hist(rows_anom, bins=bins, color=ANOMALY_FILL, edgecolor=ANOMALY_EDGE, linewidth=0.8,
            alpha=0.85, density=True, label=f"anomaly (n={len(rows_anom)})", zorder=3)
    ax.axvline(24, color="#55555f", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("event length (rows, 18s each, log scale)", fontsize=8, color="#55555f")
    ax.set_ylabel("density", fontsize=8, color="#55555f")
    ax.set_title("Real anomalies tend to be longer, not shorter", fontsize=9.5, color=INK)
    ax.text(0.98, 0.72, "dashed line:\n24-row window\n(RoCAFull's\nwindow length)",
            transform=ax.transAxes, fontsize=6.3, color="#55555f", ha="right", va="top")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig("paper/figures/eda_duration_hist.pdf", bbox_inches="tight")
    fig.savefig("paper/figures/eda_duration_hist.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_generalization_slope():
    # Mission 2 -> Mission 1 anomalies-only macro F0.5, nine models checked this round.
    rows = [
        ("KalmanGlobalMultivariate", 0.073, 0.096, "kalman"),
        ("KalmanLocalTrend", 0.070, 0.092, "kalman"),
        ("KalmanBlockMultivariate", 0.075, 0.093, "kalman"),
        ("PyOD-ECOD", 0.104, 0.097, "other"),
        ("HeterogeneousFusion", 0.125, 0.132, "other"),
        ("Autoencoder", 0.057, 0.036, "other"),
        ("ADDAEILForest", 0.089, 0.041, "other"),
        ("MatrixProfileDetector", 0.074, 0.059, "other"),
        ("TCopulaDetector", 0.160, 0.056, "copula"),
        ("GaussianCopulaDetector", 0.152, 0.043, "copula"),
    ]
    colors = {"kalman": "#0072b2", "copula": "#d55e00", "other": "#999aa3"}
    fig, ax = plt.subplots(figsize=(3.6, 4.0))
    for name, m2, m1, fam in rows:
        c = colors[fam]
        ax.plot([0, 1], [m2, m1], color=c, linewidth=1.6, alpha=0.85, zorder=2,
                marker="o", markersize=4)

    # Greedy label de-collision: sort by endpoint value, enforce a minimum vertical gap,
    # keep a thin leader line from the true data point to the (possibly nudged) label.
    y_span = max(m1 for _, _, m1, _ in rows) - min(m1 for _, _, m1, _ in rows)
    min_gap = y_span * 0.072
    ordered = sorted(rows, key=lambda r: r[2])
    label_y = [ordered[0][2]]
    for _, _, m1, _ in ordered[1:]:
        label_y.append(max(m1, label_y[-1] + min_gap))
    for (name, m2, m1, fam), ly in zip(ordered, label_y):
        c = colors[fam]
        if abs(ly - m1) > 1e-6:
            ax.plot([1.0, 1.04], [m1, ly], color=c, linewidth=0.6, alpha=0.6, zorder=1)
        ax.text(1.06, ly, name, fontsize=6.6, color=c, va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mission 2", "Mission 1"], fontsize=8.5, color="#55555f")
    ax.set_xlim(-0.08, 2.05)
    ax.set_ylabel("anomalies-only macro F0.5", fontsize=8, color="#55555f")
    ax.set_title("Kalman models hold up out of domain;\ncopula models do not", fontsize=9.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#999aa3")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(colors="#55555f", length=3)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("paper/figures/mission1_slope.pdf", bbox_inches="tight")
    fig.savefig("paper/figures/mission1_slope.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_benchmark_data(mission=2)
    anoms = anomalies_only(data.events)
    rare = data.events.filter(pl.col("Category") == "rare event")
    fig_examples_over_time(data, anoms, rare)
    fig_duration_distribution(anoms, rare)
    fig_generalization_slope()
    print("wrote paper/figures/{eda_examples,eda_duration_hist,mission1_slope}.{pdf,png}")


if __name__ == "__main__":
    main()
