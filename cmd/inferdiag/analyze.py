#!/usr/bin/env python3
"""
Tier 1 + Tier 2 Bayesian inference analyzer.

Reads a JSONL file produced by the inferdiag harness and computes:
  1. Bag-consistency assertion (hard fail if violated)
  2. Top-k hit rate (k=1,3,5,10) — overall and by rack_length bucket
  3. Log-loss vs uniform-prior baseline
  4. Calibration (reliability diagram + ECE)
  5. Entropy reduction (bits learned vs uniform)
  6. Tile-level marginal accuracy (Brier score per tile)
  7. Leave-value error (requires --klv-json mapping file)
  8. Tier 2 move-quality evaluation (if --tier2 records present):
       - Move agreement rates (inferred vs oracle, no-info vs oracle)
       - Mean win% by assumption (no-info, inferred, oracle)
       - Regret analysis on informative positions (where oracle ≠ no-info)
       - Cases where inference hurts vs helps

Usage:
    python3 analyze.py results.jsonl
    python3 analyze.py results.jsonl --klv-json leaves.json
    python3 analyze.py results.jsonl --plots          # save PNG plots
    python3 analyze.py results.jsonl --csv out/       # save CSV files
"""

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Combinatorics helper
# ---------------------------------------------------------------------------

def log_comb(n: int, k: int) -> float:
    """log C(n, k) using Stirling-free computation via lgamma."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def log_multiset_count(bag_unseen: dict, rack_length: int) -> float:
    """
    log C(total_tiles, rack_length) for a multiset draw (multivariate
    hypergeometric denominator), which bounds the uniform prior.
    Returns the log of the number of distinct leaves of length rack_length
    drawable from the unseen pool (approximated as combinations without
    replacement from a pool of total_tiles distinct slots).

    This is an *upper bound* on the true count of distinct multiset leaves;
    the true count is the multiset coefficient over the tile types.
    For the baseline log-loss we use the actual count via counting
    multisets via product formula when the pool is small, falling back to
    C(total, k) as an approximation.
    """
    total = sum(bag_unseen.values())
    return log_comb(total, rack_length)


# ---------------------------------------------------------------------------
# KLV loader (optional)
# ---------------------------------------------------------------------------

def load_klv(path: str) -> dict:
    """
    Parse a .klv2 file and return a dict mapping sorted-leave-string → float.
    KLV2 format: 4-byte little-endian uint32 count N, then N × (uint32 key, float32 value).
    The key encodes the leave as a packed integer; we need the alphabet to decode.
    Since decoding from machine letters to strings requires the alphabet and the
    leave values are indexed by a hash, the simplest approach is to trust the
    JSONL posterior's leave strings and look up their value via a sorted-tile key.

    KLV2 actual format (from the word-golib source):
    - 4 bytes: number of entries N
    - N × (4 bytes uint32 key + 4 bytes float32 value)

    The key is the leave encoded as a sorted sequence of machine letter values
    packed into a uint32. For ≤7 tiles with values ≤31, this fits in a uint32
    via packing 5 bits each. Without the alphabet mapping we can't use this file
    directly from Python. So this function returns None and leave-value metric
    is skipped unless the user has an alternative mapping.

    In practice, please use --klv-json instead (see README).
    """
    print(f"  WARNING: .klv2 binary loading not yet implemented. "
          f"Provide --klv-json instead.", file=sys.stderr)
    return {}


def load_klv_json(path: str) -> dict:
    """
    Load leave values from a JSON file mapping leave-string → float.
    Generate this with: ./bin/shell leavevalues --output leaves.json
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    records = []
    errors = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors += 1
                if errors <= 5:
                    print(f"  JSON parse error line {lineno}: {e}", file=sys.stderr)
    if errors:
        print(f"  WARNING: {errors} malformed lines skipped", file=sys.stderr)
    return records


# ---------------------------------------------------------------------------
# Metric 1: Bag-consistency assertion
# ---------------------------------------------------------------------------

def check_bag_consistency(records: list[dict]) -> int:
    """
    Assert every rack in every posterior is a multiset-subset of bag_unseen.
    Returns the number of violations found.
    """
    violations = 0
    for rec in records:
        bag = defaultdict(int, rec.get("bag_unseen", {}))
        for entry in rec.get("posterior", []):
            leave_str = entry["leave"]
            leave_counts = defaultdict(int)
            # count individual characters, handling '?' for blank
            for ch in leave_str:
                leave_counts[ch.upper()] += 1
            for tile, cnt in leave_counts.items():
                if bag[tile] < cnt:
                    violations += 1
                    if violations <= 3:
                        print(f"  BAG VIOLATION game={rec['game_id']} turn={rec['turn']}: "
                              f"leave '{leave_str}' needs {cnt}×{tile} but bag has {bag[tile]}",
                              file=sys.stderr)
    return violations


# ---------------------------------------------------------------------------
# Metric 2: Top-k hit rate
# ---------------------------------------------------------------------------

def topk_hit_rate(records: list[dict], ks=(1, 3, 5, 10)) -> pd.DataFrame:
    """Returns a DataFrame with columns: rack_length, k, hit_rate, n."""
    rows = []
    for rec in records:
        if not rec.get("true_opp_leave_found"):
            continue
        true_leave = rec["true_opp_leave"]
        posterior = rec.get("posterior", [])
        # posterior is already sorted by weight descending from the harness
        found_at = None
        for rank, entry in enumerate(posterior, 1):
            if entry["leave"] == true_leave:
                found_at = rank
                break

        rl = rec.get("rack_length", -1)
        for k in ks:
            hit = 1 if (found_at is not None and found_at <= k) else 0
            rows.append({"rack_length": rl, "k": k, "hit": hit, "game_id": rec["game_id"]})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    result = df.groupby(["rack_length", "k"])["hit"].agg(hit_rate="mean", n="count").reset_index()
    return result


# ---------------------------------------------------------------------------
# Metric 3: Log-loss vs uniform baseline
# ---------------------------------------------------------------------------

def log_loss_analysis(records: list[dict]) -> dict:
    """
    Computes per-record log-loss and the uniform-prior baseline.
    Returns summary dict.
    """
    eps = 1e-12
    losses = []
    baselines = []

    for rec in records:
        if not rec.get("true_opp_leave_found"):
            continue
        true_leave = rec["true_opp_leave"]
        posterior = rec.get("posterior", [])
        total_weight = rec.get("posterior_sum_all_weights", 0.0)
        if total_weight <= 0:
            continue

        # Find true rack probability
        true_weight = 0.0
        for entry in posterior:
            if entry["leave"] == true_leave:
                true_weight = entry["weight"]
                break
        # If truncated, the true leave might not be in the top-K.
        # We use eps (treated as absent) so truncation penalties are visible.
        p_true = max(true_weight / total_weight, eps)
        losses.append(-math.log(p_true))

        # Uniform baseline: 1 / C(total_unseen, rack_length)
        bag = rec.get("bag_unseen", {})
        rl = rec.get("rack_length", 0)
        total_unseen = sum(bag.values())
        if total_unseen > 0 and rl > 0:
            baseline_ll = log_multiset_count(bag, rl)
            baselines.append(baseline_ll)
        else:
            baselines.append(None)

    valid_baselines = [b for b in baselines if b is not None]
    return {
        "n": len(losses),
        "mean_log_loss": float(np.mean(losses)) if losses else None,
        "median_log_loss": float(np.median(losses)) if losses else None,
        "mean_uniform_baseline": float(np.mean(valid_baselines)) if valid_baselines else None,
        "improvement_vs_uniform": (
            float(np.mean(valid_baselines)) - float(np.mean(losses))
        ) if losses and valid_baselines else None,
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Metric 4: Calibration (reliability diagram + ECE)
# ---------------------------------------------------------------------------

def calibration_analysis(records: list[dict], n_bins: int = 10) -> dict:
    """
    Bins all (rack, normalized_weight) pairs from all posteriors by weight.
    Within each bin, measures empirical frequency of being the true rack.
    Returns bin data and ECE.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_sum_conf = defaultdict(float)
    bin_sum_acc = defaultdict(float)
    bin_n = defaultdict(int)

    for rec in records:
        if not rec.get("true_opp_leave_found"):
            continue
        true_leave = rec["true_opp_leave"]
        posterior = rec.get("posterior", [])
        total_weight = rec.get("posterior_sum_all_weights", 0.0)
        if total_weight <= 0 or not posterior:
            continue

        for entry in posterior:
            p = entry["weight"] / total_weight
            correct = 1.0 if entry["leave"] == true_leave else 0.0
            # Find bin
            b = min(int(p * n_bins), n_bins - 1)
            bin_sum_conf[b] += p
            bin_sum_acc[b] += correct
            bin_n[b] += 1

    bins = []
    total_samples = sum(bin_n.values())
    ece = 0.0
    for b in range(n_bins):
        n = bin_n[b]
        if n == 0:
            continue
        avg_conf = bin_sum_conf[b] / n
        avg_acc = bin_sum_acc[b] / n
        ece += (n / total_samples) * abs(avg_conf - avg_acc)
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bins.append({
            "bin_lo": lo, "bin_hi": hi,
            "avg_confidence": avg_conf,
            "empirical_accuracy": avg_acc,
            "n": n,
        })

    return {"bins": bins, "ece": ece, "total_samples": total_samples}


# ---------------------------------------------------------------------------
# Metric 5: Entropy reduction
# ---------------------------------------------------------------------------

def entropy_analysis(records: list[dict]) -> dict:
    """
    Computes H(posterior) and H(uniform) per record, returns mean bits learned.
    H(uniform) = log C(total_unseen, rack_length) in nats / log(2) in bits.
    """
    bits_learned = []
    for rec in records:
        posterior = rec.get("posterior", [])
        total_weight = rec.get("posterior_sum_all_weights", 0.0)
        if total_weight <= 0 or not posterior:
            continue

        # Posterior entropy
        h_post = 0.0
        for entry in posterior:
            p = entry["weight"] / total_weight
            if p > 0:
                h_post -= p * math.log2(p)

        # Uniform baseline entropy
        bag = rec.get("bag_unseen", {})
        rl = rec.get("rack_length", 0)
        total_unseen = sum(bag.values())
        if total_unseen > 0 and rl > 0:
            h_uniform = log_multiset_count(bag, rl) / math.log(2)
            bits_learned.append(h_uniform - h_post)

    return {
        "n": len(bits_learned),
        "mean_bits_learned": float(np.mean(bits_learned)) if bits_learned else None,
        "median_bits_learned": float(np.median(bits_learned)) if bits_learned else None,
    }


# ---------------------------------------------------------------------------
# Metric 6: Tile-level marginal accuracy (Brier score)
# ---------------------------------------------------------------------------

def tile_marginal_analysis(records: list[dict]) -> dict:
    """
    For each tile letter, computes predicted P(tile in rack) and ground truth.
    Returns per-tile Brier score and overall mean.
    """
    tile_pred = defaultdict(list)  # tile → [predicted_prob, ...]
    tile_true = defaultdict(list)  # tile → [0/1, ...]

    for rec in records:
        if not rec.get("true_opp_leave_found"):
            continue
        true_leave = rec["true_opp_leave"]
        posterior = rec.get("posterior", [])
        total_weight = rec.get("posterior_sum_all_weights", 0.0)
        if total_weight <= 0:
            continue

        # Find all tiles in the bag
        bag = rec.get("bag_unseen", {})
        all_tiles = set(bag.keys())
        # Also include true leave tiles (should already be in bag)
        for ch in true_leave:
            all_tiles.add(ch.upper())

        # Compute P(tile in rack) = sum of weights of racks containing tile / total
        tile_weight = defaultdict(float)
        for entry in posterior:
            w = entry["weight"]
            leave_tiles = set(ch.upper() for ch in entry["leave"])
            for tile in leave_tiles:
                tile_weight[tile] += w

        # True tiles in leave
        true_tiles = set(ch.upper() for ch in true_leave)

        for tile in all_tiles:
            p = tile_weight[tile] / total_weight
            y = 1.0 if tile in true_tiles else 0.0
            tile_pred[tile].append(p)
            tile_true[tile].append(y)

    results = {}
    all_brier = []
    for tile in sorted(tile_pred.keys()):
        preds = np.array(tile_pred[tile])
        truths = np.array(tile_true[tile])
        brier = float(np.mean((preds - truths) ** 2))
        results[tile] = {"brier_score": brier, "n": len(preds), "mean_pred": float(preds.mean())}
        all_brier.append(brier)

    return {
        "per_tile": results,
        "overall_mean_brier": float(np.mean(all_brier)) if all_brier else None,
    }


# ---------------------------------------------------------------------------
# Metric 7: Leave-value error (optional, requires leave values dict)
# ---------------------------------------------------------------------------

def leave_value_error(records: list[dict], leave_values: dict) -> dict:
    """
    Computes E[leave_value | posterior] − leave_value(true).
    Requires a dict mapping leave-string (sorted, uppercase) → float.
    """
    if not leave_values:
        return {}

    signed_errors = []
    abs_errors = []

    def normalize_leave(leave_str: str) -> str:
        return "".join(sorted(leave_str.upper()))

    for rec in records:
        if not rec.get("true_opp_leave_found"):
            continue
        true_leave_norm = normalize_leave(rec["true_opp_leave"])
        if true_leave_norm not in leave_values:
            continue
        true_val = leave_values[true_leave_norm]

        posterior = rec.get("posterior", [])
        total_weight = rec.get("posterior_sum_all_weights", 0.0)
        if total_weight <= 0:
            continue

        expected_val = sum(
            entry["weight"] / total_weight * leave_values.get(normalize_leave(entry["leave"]), 0.0)
            for entry in posterior
        )
        err = expected_val - true_val
        signed_errors.append(err)
        abs_errors.append(abs(err))

    return {
        "n": len(signed_errors),
        "mean_signed_error": float(np.mean(signed_errors)) if signed_errors else None,
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
        "median_signed_error": float(np.median(signed_errors)) if signed_errors else None,
    }


# ---------------------------------------------------------------------------
# Metric 8: Tier 2 move-quality evaluation
# ---------------------------------------------------------------------------

def tier2_analysis(records: list[dict]) -> dict | None:
    """
    Analyzes Tier 2 move-quality fields (no_info / inferred / oracle sims).

    Returns a dict with:
      - n_total: records with tier2 data
      - n_oracle: records where oracle was available (true leave found)
      - agreement rates overall and by rack_length
      - mean win% by assumption
      - regret on "informative" positions (oracle ≠ no-info)
      - counts of inference helping / hurting / neutral
    """
    t2_records = [r for r in records if r.get("tier2")]
    if not t2_records:
        return None

    n_total = len(t2_records)
    n_oracle = sum(1 for r in t2_records if r["tier2"]["oracle_available"])

    # Per-record stats
    no_info_wins, inferred_wins, oracle_wins = [], [], []
    no_info_oracle_agree = []
    inferred_oracle_agree = []

    # By rack_length
    by_rl: dict[int, dict] = defaultdict(lambda: {
        "n": 0, "n_oracle": 0,
        "no_info_oracle_agree": 0, "inferred_oracle_agree": 0,
    })

    # Informative positions: oracle chose differently from no-info
    informative = []

    for r in t2_records:
        t2 = r["tier2"]
        rl = r.get("rack_length", -1)

        no_info_wins.append(t2["no_info"]["win_pct"])
        inferred_wins.append(t2["inferred"]["win_pct"])

        by_rl[rl]["n"] += 1

        if not t2["oracle_available"]:
            continue

        by_rl[rl]["n_oracle"] += 1
        oracle_wins.append(t2["oracle"]["win_pct"])

        ni_agree = t2["no_info_oracle_agree"]
        inf_agree = t2["inferred_oracle_agree"]
        no_info_oracle_agree.append(ni_agree)
        inferred_oracle_agree.append(inf_agree)

        if ni_agree:
            by_rl[rl]["no_info_oracle_agree"] += 1
        if inf_agree:
            by_rl[rl]["inferred_oracle_agree"] += 1

        # Informative: oracle picked a different move than no-info
        if not ni_agree:
            informative.append({
                "rack_length": rl,
                "no_info_win":   t2["no_info"]["win_pct"],
                "inferred_win":  t2["inferred"]["win_pct"],
                "oracle_win":    t2["oracle"]["win_pct"],
                "inferred_oracle_agree": inf_agree,
            })

    # Agreement rates (oracle-available records only)
    n_oa = len(no_info_oracle_agree)
    ni_agree_rate  = sum(no_info_oracle_agree) / n_oa if n_oa else None
    inf_agree_rate = sum(inferred_oracle_agree) / n_oa if n_oa else None

    # Mean win% by assumption
    mean_no_info  = float(np.mean(no_info_wins))  if no_info_wins  else None
    mean_inferred = float(np.mean(inferred_wins)) if inferred_wins else None
    mean_oracle   = float(np.mean(oracle_wins))   if oracle_wins   else None

    # Regret on informative positions
    inf_regret = None
    inf_recovery_pct = None
    inf_helps = inf_hurts = inf_neutral = 0
    if informative:
        gaps_oracle  = [x["oracle_win"]   - x["no_info_win"] for x in informative]
        gaps_inferred= [x["inferred_win"] - x["no_info_win"] for x in informative]
        inf_regret = float(np.mean(gaps_inferred))  # >0 = inference helped on avg

        # Recovery: fraction of oracle gain captured by inference
        recoveries = []
        for oracle_gap, inferred_gap in zip(gaps_oracle, gaps_inferred):
            if abs(oracle_gap) > 1e-6:
                recoveries.append(inferred_gap / oracle_gap)
        if recoveries:
            inf_recovery_pct = float(np.mean(recoveries)) * 100

        for x in informative:
            d = x["inferred_win"] - x["no_info_win"]
            if d > 0.005:
                inf_helps += 1
            elif d < -0.005:
                inf_hurts += 1
            else:
                inf_neutral += 1

    # By rack_length table
    rl_rows = []
    for rl in sorted(by_rl.keys()):
        v = by_rl[rl]
        n_oa_rl = v["n_oracle"]
        rl_rows.append({
            "rack_length": rl,
            "n": v["n"],
            "n_oracle": n_oa_rl,
            "no_info_oracle_agree_rate": v["no_info_oracle_agree"] / n_oa_rl if n_oa_rl else None,
            "inferred_oracle_agree_rate": v["inferred_oracle_agree"] / n_oa_rl if n_oa_rl else None,
        })

    return {
        "n_total": n_total,
        "n_oracle": n_oracle,
        "n_informative": len(informative),
        "no_info_oracle_agree_rate":   ni_agree_rate,
        "inferred_oracle_agree_rate":  inf_agree_rate,
        "mean_win_pct_no_info":   mean_no_info,
        "mean_win_pct_inferred":  mean_inferred,
        "mean_win_pct_oracle":    mean_oracle,
        "informative_mean_inferred_gain": inf_regret,
        "informative_recovery_pct":       inf_recovery_pct,
        "informative_inference_helps":    inf_helps,
        "informative_inference_hurts":    inf_hurts,
        "informative_inference_neutral":  inf_neutral,
        "by_rack_length": rl_rows,
    }


def plot_tier2_winpct(t2: dict, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = ["No info", "Inferred", "Oracle"]
    vals = [t2["mean_win_pct_no_info"], t2["mean_win_pct_inferred"], t2["mean_win_pct_oracle"]]
    if any(v is None for v in vals):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    ax.bar(labels, vals, color=colors, width=0.5)
    ax.set_ylabel("Mean win% of chosen move")
    ax.set_title("Tier 2: move quality by rack assumption")
    ax.set_ylim(min(vals) * 0.98, max(vals) * 1.02)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved Tier 2 win% bar chart → {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reliability_diagram(cal: dict, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots", file=sys.stderr)
        return
    bins = cal["bins"]
    if not bins:
        return
    conf = [b["avg_confidence"] for b in bins]
    acc = [b["empirical_accuracy"] for b in bins]
    ns = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    scatter = ax.scatter(conf, acc, c=ns, cmap="viridis", s=80, zorder=3)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram  (ECE={cal['ece']:.4f})")
    fig.colorbar(scatter, ax=ax, label="# samples in bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved reliability diagram → {out_path}")


def plot_log_loss_histogram(losses: list[float], mean_uniform: float, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(losses, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(float(np.mean(losses)), color="red", linewidth=1.5, label=f"Mean={np.mean(losses):.2f}")
    if mean_uniform is not None:
        ax.axvline(mean_uniform, color="orange", linewidth=1.5, linestyle="--",
                   label=f"Uniform baseline={mean_uniform:.2f}")
    ax.set_xlabel("−log p(true rack)")
    ax.set_ylabel("Count")
    ax.set_title("Log-loss distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved log-loss histogram → {out_path}")


def plot_topk_by_rack_length(topk_df: pd.DataFrame, out_path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if topk_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, grp in topk_df.groupby("k"):
        grp_sorted = grp.sort_values("rack_length")
        ax.plot(grp_sorted["rack_length"], grp_sorted["hit_rate"], marker="o", label=f"top-{k}")
    ax.set_xlabel("Rack length (tiles opponent kept)")
    ax.set_ylabel("Hit rate")
    ax.set_title("Top-k hit rate by rack length")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved top-k plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tier 1 inference analyzer")
    parser.add_argument("jsonl", help="JSONL file from inferdiag harness")
    parser.add_argument("--klv-json", help="Leave values JSON (leave_str → float)")
    parser.add_argument("--plots", action="store_true", help="Save PNG plots")
    parser.add_argument("--csv", metavar="DIR", help="Save CSV files to DIR")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\nLoading {args.jsonl}...")
    records = load_records(args.jsonl)
    if not records:
        print("No records found. Exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(records)} records loaded")

    n_with_leave = sum(1 for r in records if r.get("true_opp_leave_found"))
    print(f"  {n_with_leave} records have a recoverable true opp leave")

    # ---- Metric 1: Bag consistency ----
    print("\n[1] Bag-consistency assertion...")
    violations = check_bag_consistency(records)
    if violations > 0:
        print(f"  FAIL: {violations} bag-consistency violations found. "
              f"Fix the inference before continuing.", file=sys.stderr)
        sys.exit(2)
    else:
        print("  PASS: all posterior racks are bag-consistent")

    # ---- Metric 2: Top-k hit rate ----
    print("\n[2] Top-k hit rate...")
    topk_df = topk_hit_rate(records)
    if not topk_df.empty:
        overall = topk_df.groupby("k").apply(
            lambda g: pd.Series({"hit_rate": (g["hit_rate"] * g["n"]).sum() / g["n"].sum(),
                                  "n": g["n"].sum()}),
            include_groups=False
        ).reset_index()
        print(f"  Overall (all rack lengths):")
        for _, row in overall.iterrows():
            print(f"    top-{int(row['k'])}: {row['hit_rate']:.3f}  (n={int(row['n'])})")
        if args.verbose:
            print(f"\n  By rack_length:")
            print(topk_df.to_string(index=False))
        if args.csv:
            Path(args.csv).mkdir(parents=True, exist_ok=True)
            topk_df.to_csv(Path(args.csv) / "topk_by_racklen.csv", index=False)
        if args.plots:
            plot_topk_by_rack_length(topk_df, "topk_by_racklen.png")

    # ---- Metric 3: Log-loss ----
    print("\n[3] Log-loss vs uniform baseline...")
    ll = log_loss_analysis(records)
    if ll["mean_log_loss"] is not None:
        print(f"  Mean log-loss (inference):  {ll['mean_log_loss']:.4f}")
        if ll["mean_uniform_baseline"] is not None:
            print(f"  Mean log-loss (uniform):    {ll['mean_uniform_baseline']:.4f}")
            improvement = ll["improvement_vs_uniform"]
            sign = "BETTER" if improvement and improvement > 0 else "WORSE"
            print(f"  Improvement vs uniform:     {improvement:+.4f}  [{sign}]")
        if args.plots and ll["losses"]:
            plot_log_loss_histogram(ll["losses"], ll["mean_uniform_baseline"], "log_loss_hist.png")

    # ---- Metric 4: Calibration ----
    print("\n[4] Calibration...")
    cal = calibration_analysis(records)
    print(f"  ECE: {cal['ece']:.4f}  (total_samples={cal['total_samples']})")
    if cal["ece"] > 0.1:
        print("  WARNING: ECE > 0.10 — inference may be significantly miscalibrated")
    if args.plots:
        plot_reliability_diagram(cal, "reliability_diagram.png")
    if args.csv and cal["bins"]:
        Path(args.csv).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cal["bins"]).to_csv(Path(args.csv) / "calibration_bins.csv", index=False)

    # ---- Metric 5: Entropy ----
    print("\n[5] Entropy reduction...")
    ent = entropy_analysis(records)
    if ent["mean_bits_learned"] is not None:
        print(f"  Mean bits learned vs uniform: {ent['mean_bits_learned']:.3f} bits")
        print(f"  Median bits learned:          {ent['median_bits_learned']:.3f} bits")
        if ent["mean_bits_learned"] < 0:
            print("  WARNING: negative bits learned — inference is *increasing* entropy (bug?)")

    # ---- Metric 6: Tile marginals ----
    print("\n[6] Tile-level marginal accuracy (Brier score)...")
    tile = tile_marginal_analysis(records)
    if tile["overall_mean_brier"] is not None:
        print(f"  Overall mean Brier score: {tile['overall_mean_brier']:.5f}")
        if args.verbose and tile["per_tile"]:
            print(f"  {'Tile':<6} {'Brier':>8} {'Mean pred':>10} {'N':>8}")
            for t, v in sorted(tile["per_tile"].items()):
                print(f"  {t:<6} {v['brier_score']:>8.5f} {v['mean_pred']:>10.3f} {v['n']:>8}")
        if args.csv and tile["per_tile"]:
            Path(args.csv).mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(tile["per_tile"], orient="index").to_csv(
                Path(args.csv) / "tile_marginals.csv")

    # ---- Metric 7: Leave-value error ----
    leave_values = {}
    if args.klv_json:
        print(f"\n[7] Leave-value error (using {args.klv_json})...")
        leave_values = load_klv_json(args.klv_json)
        lv = leave_value_error(records, leave_values)
        if lv.get("n"):
            print(f"  n: {lv['n']}")
            print(f"  Mean signed error:  {lv['mean_signed_error']:+.4f}  "
                  f"(positive = inference over-estimates leave value)")
            print(f"  Mean absolute error: {lv['mean_abs_error']:.4f}")
        else:
            print("  No records with leave values found.")
    else:
        print("\n[7] Leave-value error: skipped (pass --klv-json to enable)")

    # ---- Metric 8: Tier 2 move-quality ----
    t2 = tier2_analysis(records)
    if t2 is None:
        print("\n[8] Tier 2 move-quality: no tier2 data (rerun with --tier2 flag)")
    else:
        print(f"\n[8] Tier 2 move-quality evaluation...")
        print(f"  Positions evaluated:  {t2['n_total']}")
        print(f"  Oracle available:     {t2['n_oracle']}")
        print(f"  Informative positions (oracle ≠ no-info): {t2['n_informative']}")
        print()
        if t2["mean_win_pct_no_info"] is not None:
            print(f"  Mean win% of chosen move:")
            print(f"    No-info:  {t2['mean_win_pct_no_info']:.4f}")
            print(f"    Inferred: {t2['mean_win_pct_inferred']:.4f}")
            if t2["mean_win_pct_oracle"] is not None:
                print(f"    Oracle:   {t2['mean_win_pct_oracle']:.4f}")
        print()
        if t2["no_info_oracle_agree_rate"] is not None:
            print(f"  Move agreement with oracle:")
            print(f"    No-info agrees:  {t2['no_info_oracle_agree_rate']:.1%}")
            print(f"    Inferred agrees: {t2['inferred_oracle_agree_rate']:.1%}")
        print()
        if t2["n_informative"] > 0:
            print(f"  On informative positions (oracle chose differently from no-info):")
            g = t2["informative_mean_inferred_gain"]
            sign = "+" if g and g >= 0 else ""
            print(f"    Mean inferred gain vs no-info: {sign}{g:.4f} win%")
            if t2["informative_recovery_pct"] is not None:
                print(f"    Recovery of oracle advantage:  {t2['informative_recovery_pct']:.1f}%")
            print(f"    Inference helped:  {t2['informative_inference_helps']}")
            print(f"    Inference hurt:    {t2['informative_inference_hurts']}")
            print(f"    Inference neutral: {t2['informative_inference_neutral']}")
        if args.verbose and t2["by_rack_length"]:
            print(f"\n  By rack_length:")
            print(f"  {'RL':>4} {'N':>6} {'N_oracle':>9} {'NoInfo==Oracle':>15} {'Inf==Oracle':>12}")
            for row in t2["by_rack_length"]:
                ni = f"{row['no_info_oracle_agree_rate']:.1%}" if row["no_info_oracle_agree_rate"] is not None else "n/a"
                inf = f"{row['inferred_oracle_agree_rate']:.1%}" if row["inferred_oracle_agree_rate"] is not None else "n/a"
                print(f"  {row['rack_length']:>4} {row['n']:>6} {row['n_oracle']:>9} {ni:>15} {inf:>12}")
        if args.plots:
            plot_tier2_winpct(t2, "tier2_winpct.png")
        if args.csv and t2["by_rack_length"]:
            Path(args.csv).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(t2["by_rack_length"]).to_csv(Path(args.csv) / "tier2_by_racklen.csv", index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
