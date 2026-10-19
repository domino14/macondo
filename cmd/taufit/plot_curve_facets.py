"""Faceted tau log-likelihood curves by tiles-in-bag bucket, from taufit's curve.csv."""
import csv
import math
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

CURVE_CSV = sys.argv[1]
OUT_PNG = sys.argv[2]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"
GRID = "#e4e3e0"

TAU_DEFAULT = 0.05
Y_BOTTOM = -5.0

curves = defaultdict(lambda: {"tau": [], "ll": [], "n": 0, "uniform": None})
with open(CURVE_CSV) as f:
    for row in csv.DictReader(f):
        ll = float(row["mean_ll"])
        if not math.isfinite(ll):
            continue
        c = curves[row["bucket"]]
        c["tau"].append(float(row["tau"]))
        c["ll"].append(ll)
        c["n"] = int(row["n"])
        c["uniform"] = float(row["uniform_mean_ll"])

FACETS = ["bag 46+", "bag 31-45", "bag 11-30", "bag 2-10"]  # early → late

fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

tau_lo = min(min(c["tau"]) for b, c in curves.items() if b != "all")
tau_hi = max(max(c["tau"]) for b, c in curves.items() if b != "all")
ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2)
         if tau_lo <= t <= tau_hi]

for ax, bucket in zip(axes.flat, FACETS):
    c = curves[bucket]
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.plot(c["tau"], c["ll"], color=SERIES_1, linewidth=2, solid_capstyle="round")

    ax.axhline(c["uniform"], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)))
    ax.axvline(TAU_DEFAULT, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (2, 3)))

    best = max(range(len(c["ll"])), key=lambda i: c["ll"][i])
    tau_star, ll_star = c["tau"][best], c["ll"][best]
    ax.plot([tau_star], [ll_star], marker="o", markersize=7, color=SERIES_1,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"τ* = {tau_star:.3f}", xy=(tau_star, ll_star),
                xytext=(0, 8), textcoords="offset points",
                color=TEXT_PRIMARY, fontsize=9, ha="center", va="bottom")

    ax.set_title(f"{bucket} tiles   (n = {c['n']:,})", color=TEXT_PRIMARY,
                 fontsize=10, loc="left")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

# One explanatory label for the shared reference lines, in the first facet.
ax0 = axes.flat[0]
c0 = curves[FACETS[0]]
ax0.annotate("uniform choice", xy=(tau_hi, c0["uniform"]),
             xytext=(0, -6), textcoords="offset points",
             color=TEXT_SECONDARY, fontsize=8, ha="right", va="top")
ax0.annotate("default τ = 0.05", xy=(TAU_DEFAULT, Y_BOTTOM),
             xytext=(3, 6), textcoords="offset points",
             color=TEXT_SECONDARY, fontsize=8, ha="left", va="bottom")

top = max(max(c["ll"]) for b, c in curves.items() if b != "all")
axes.flat[0].set_ylim(Y_BOTTOM, top + 0.55)

fig.suptitle("Likelihood of BestBot's real moves vs τ, by tiles remaining in bag\n"
             "16,028 positions from 2,000 NWL23 games — shallow-sim softmax model",
             color=TEXT_PRIMARY, fontsize=11.5, x=0.02, ha="left")
fig.supxlabel("τ (softmax temperature, log scale)", color=TEXT_PRIMARY, fontsize=10)
fig.supylabel("mean log-likelihood per position", color=TEXT_PRIMARY, fontsize=10)

fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
fig.savefig(OUT_PNG, facecolor=SURFACE)
print(OUT_PNG)
