"""Plot the tau log-likelihood curve from taufit's curve.csv."""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

CURVE_CSV = sys.argv[1]
OUT_PNG = sys.argv[2]

# Reference dataviz palette (light mode)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"  # categorical slot 1 (blue)
GRID = "#e4e3e0"

TAU_STAR = 0.1194
TAU_DEFAULT = 0.05
LL_UNIFORM = -2.3105  # mean log-likelihood of a uniform guess over candidates

import math

taus, mean_ll = [], []
with open(CURVE_CSV) as f:
    for row in csv.DictReader(f):
        t, ll = float(row["tau"]), float(row["mean_ll"])
        # At very small tau some position's likelihood underflows to 0 and
        # its log to -inf; those grid points carry no plottable information.
        if math.isfinite(ll):
            taus.append(t)
            mean_ll.append(ll)


def interp_ll(x):
    for i in range(len(taus) - 1):
        if taus[i] <= x <= taus[i + 1]:
            frac = (x - taus[i]) / (taus[i + 1] - taus[i])
            return mean_ll[i] + frac * (mean_ll[i + 1] - mean_ll[i])
    return None


fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.set_xscale("log")
ax.plot(taus, mean_ll, color=SERIES_1, linewidth=2, solid_capstyle="round")

# Uniform-choice reference line
ax.axhline(LL_UNIFORM, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 4)))
ax.annotate("uniform choice among candidates", xy=(taus[-1], LL_UNIFORM),
            xytext=(0, -6), textcoords="offset points",
            color=TEXT_SECONDARY, fontsize=8.5, ha="right", va="top")

# Selective direct labels: the MLE and the shipped default
for x, label, va_off in [
    (TAU_STAR, f"MLE τ* = {TAU_STAR:.3f}", 10),
    (TAU_DEFAULT, f"current default τ = {TAU_DEFAULT:.2f}", 10),
]:
    y = interp_ll(x)
    ax.plot([x], [y], marker="o", markersize=8, color=SERIES_1,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"{label}\nmean LL = {y:.2f}", xy=(x, y),
                xytext=(0, va_off), textcoords="offset points",
                color=TEXT_PRIMARY, fontsize=9, ha="center", va="bottom")

ax.set_xlabel("τ (softmax temperature, log scale)", color=TEXT_PRIMARY, fontsize=10)
ax.set_ylabel("mean log-likelihood per position", color=TEXT_PRIMARY, fontsize=10)
ax.set_title("Likelihood of BestBot's real moves under the shallow-sim softmax model\n"
             "16,028 positions from 2,000 NWL23 games",
             color=TEXT_PRIMARY, fontsize=11, loc="left", pad=12)

ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2)
         if taus[0] <= t <= taus[-1]]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", color=GRID, linewidth=0.7)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)

# Headroom so the τ* label doesn't collide with the top
lo, hi = min(mean_ll), max(mean_ll)
ax.set_ylim(lo - 0.1, hi + 0.35)

fig.tight_layout()
fig.savefig(OUT_PNG, facecolor=SURFACE)
print(OUT_PNG)
