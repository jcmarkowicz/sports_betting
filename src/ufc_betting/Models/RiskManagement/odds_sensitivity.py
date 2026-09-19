"""Historical payout sensitivity of close1/close2 choices versus opening picks.

One unit is staked per settled choice. Picks, outcomes, and bin membership stay
fixed; this is an ex-post price sensitivity calculation, not a refitted model or
an estimate of future returns. Opening returns are matched to the SAME fights
in each closing-price bin, using the opening model's own selected fighter.

By default, find the smallest common favorable moneyline shift that reaches
opening ROI. Shifts use a continuous American-odds axis: -150 -> -100/+100 ->
+150 is 100 points, avoiding the artificial 200-point jump at even money.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ufc_betting.config import settings


KEY = ["date", "fighter_red", "fighter_blue"]


def american_to_decimal(odds):
    odds = np.asarray(odds, dtype=float)
    if not np.isfinite(odds).all() or (np.abs(odds) < 100).any():
        raise ValueError("American odds must be finite and have magnitude >= 100")
    return np.where(odds > 0, 1 + odds / 100, 1 + 100 / np.abs(odds))


def decimal_to_american(odds):
    odds = np.asarray(odds, dtype=float)
    if not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Decimal odds must be finite and > 1")
    return np.where(odds >= 2, 100 * (odds - 1), -100 / (odds - 1))


def improve_prices(american, shift):
    """Apply a favorable shift through even money and return decimal payouts."""
    if shift < 0:
        raise ValueError("shift must be nonnegative")
    american = np.asarray(american, dtype=float)
    american_to_decimal(american)  # Validate before constructing the axis.
    axis = np.where(american < 0, american + 100, american - 100) + shift
    return np.where(axis >= 0, 2 + axis / 100, 1 + 100 / (100 + np.abs(axis)))


def roi(decimal, wins):
    return float(100 * np.mean(np.where(wins, np.asarray(decimal) - 1, -1)))


def load_paired(data_dir, split, odds_type):
    """Join by fight identity, rejecting mismatched cohorts and outcomes."""
    if odds_type not in ("close1", "close2") or split not in ("train", "test"):
        raise ValueError("Expected close1/close2 and train/test")
    data_dir = Path(data_dir)
    opening = pd.read_csv(data_dir / f"{split}_logit_open.csv")
    closing = pd.read_csv(data_dir / f"{split}_logit_{odds_type}.csv")
    for frame, stage in ((opening, "open"), (closing, odds_type)):
        required = KEY + ["pred_winner", "winner", f"{stage}_red", f"{stage}_blue"]
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{stage}: missing {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError(f"{stage}: missing or duplicate fight keys")
        if not frame.pred_winner.isin([0, 1]).all():
            raise ValueError(f"{stage}: invalid pred_winner")
    left = closing[KEY + ["pred_winner", "winner", f"{odds_type}_red", f"{odds_type}_blue"]]
    right = opening[KEY + ["pred_winner", "winner", "open_red", "open_blue"]]
    paired = left.merge(right, on=KEY, how="outer", suffixes=("_close", "_open"),
                        validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Opening and closing CSVs must contain the same fights")
    same_outcome = paired.winner_close.eq(paired.winner_open) | (
        paired.winner_close.isna() & paired.winner_open.isna())
    if not same_outcome.all():
        raise ValueError("Opening and closing fight outcomes disagree")
    settled = paired.winner_close.isin([0, 1])
    excluded = int((~settled).sum())
    paired = paired.loc[settled].copy()
    if paired.empty:
        raise ValueError("No settled fights")
    red = paired.pred_winner_close.eq(1)
    paired["choice_fighter"] = np.where(red, paired.fighter_red, paired.fighter_blue)
    paired["close_american"] = np.where(red, paired[f"{odds_type}_red"], paired[f"{odds_type}_blue"])
    paired["same_choice_open_american"] = np.where(red, paired.open_red, paired.open_blue)
    paired["open_pick_american"] = np.where(paired.pred_winner_open.eq(1), paired.open_red, paired.open_blue)
    for source, target in (("close_american", "close_decimal"),
                           ("same_choice_open_american", "same_choice_open_decimal"),
                           ("open_pick_american", "open_pick_decimal")):
        paired[target] = american_to_decimal(paired[source])
    paired["close_win"] = paired.pred_winner_close.eq(paired.winner_close)
    paired["open_win"] = paired.pred_winner_open.eq(paired.winner_open)
    paired["odds_type"] = odds_type
    return paired.drop(columns="_merge"), excluded


def solve_bin(group, max_shift=5000, step=10):
    """Solve the smallest allowed adjustment meeting matched opening ROI."""
    if not np.isfinite([max_shift, step]).all() or max_shift <= 0 or step <= 0:
        raise ValueError("Maximum shift and step must be positive and finite")
    prices = group.close_decimal.to_numpy()
    wins = group.close_win.to_numpy(dtype=bool)
    target = roi(group.open_pick_decimal, group.open_win)
    initial = roi(prices, wins)
    gap = target - initial
    limit = float(max_shift)
    adjusted = lambda amount: improve_prices(group.close_american, amount)
    grid = np.unique(np.append(np.arange(0, limit, step), limit))
    final_roi = roi(adjusted(limit), wins)
    if gap <= 1e-9:
        amount, status = 0.0, "already_at_or_above_open"
    elif not wins.any():
        amount, status = np.nan, "unreachable_no_wins"
    elif final_roi < target - 1e-9:
        amount, status = np.nan, "unreachable_within_limit"
    else:
        amount = float(brentq(lambda value: roi(adjusted(value), wins) - target,
                              0, limit, xtol=1e-10))
        status = "matched"
    solved = adjusted(amount) if np.isfinite(amount) else None
    if np.isfinite(amount):
        grid = np.unique(np.append(grid, amount))
    curve = pd.DataFrame({"adjustment": grid,
                          "adjusted_roi_pct": [roi(adjusted(value), wins) for value in grid]})
    curve["remaining_gap_pp"] = target - curve.adjusted_roi_pct
    curve["gap_closed_pct"] = (100 * (curve.adjusted_roi_pct - initial) / gap
                               if gap > 1e-9 else np.nan)
    summary = {
        "n_bets": len(group), "n_events": group.date.nunique(),
        "close_wins": int(wins.sum()), "open_wins": int(group.open_win.sum()),
        "different_picks": int(group.pred_winner_close.ne(group.pred_winner_open).sum()),
        "mean_close_american": group.close_american.mean(),
        "close_roi_pct": initial, "open_target_roi_pct": target,
        "initial_gap_pp": gap,
        "same_close_picks_at_open_roi_pct": roi(group.same_choice_open_decimal, wins),
        "required_adjustment": amount, "status": status,
        "matched_roi_pct": roi(solved, wins) if solved is not None else np.nan,
        "mean_decimal_increase": float(np.mean(solved - prices)) if solved is not None else np.nan,
        "mean_raw_american_change": float(np.mean(decimal_to_american(solved) - group.close_american))
                                    if solved is not None else np.nan,
        "mean_adjusted_american": float(np.mean(decimal_to_american(solved))) if solved is not None else np.nan,
        "roi_at_limit_pct": final_roi,
        "roi_gain_pp_first_step": roi(adjusted(min(step, limit)), wins) - initial,
        "average_gap_closed_pp_per_point": gap / amount if np.isfinite(amount) and amount > 0 else np.nan,
    }
    return summary, curve


def run_sensitivity(data_dir=None, output_dir=None, split="test", bin_width=50,
                    max_shift=5000, step=10):
    if bin_width <= 0 or bin_width % 1:
        raise ValueError("bin_width must be a positive integer")
    data_dir = Path(data_dir) if data_dir else settings.data_dir / "model_results"
    output_dir = Path(output_dir) if output_dir else data_dir / "odds_sensitivity"
    summaries, curves, pairs, exclusions = [], [], [], {}
    for odds_type in ("close1", "close2"):
        paired, exclusions[odds_type] = load_paired(data_dir, split, odds_type)
        # Original closing choice prices define fixed, half-open bins [low, high).
        paired["bin_low"] = np.floor(paired.close_american / bin_width).astype(int) * bin_width
        pairs.append(paired)
        for low, group in paired.groupby("bin_low", sort=True):
            summary, curve = solve_bin(group, max_shift, step)
            identifiers = {"odds_type": odds_type, "bin_low": int(low),
                           "bin_high_exclusive": int(low + bin_width)}
            summaries.append({**identifiers, **summary})
            for name, value in identifiers.items():
                curve[name] = value
            curves.append(curve)
    summary = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "sensitivity_by_moneyline_bin.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(output_dir / "convergence_curves.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(output_dir / "paired_fights.csv", index=False)
    metadata = {
        "split": split, "bin_width": bin_width,
        "max_shift": max_shift, "step": step, "excluded_unsettled": exclusions,
        "bin_definition": "[low, high), based on unadjusted closing choice American odds",
        "return_definition": "100 * total net profit / total stake; one unit per choice",
        "target": "Opening model picks at opening prices on the same fights in each bin",
        "adjustment_units": "continuous American moneyline points; -150 to +150 is 100 points",
        "interpretation": "Historical fixed-pick payout sensitivity; not a forecast or retrained betting strategy",
    }
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Odds sensitivity\n\n"
        "Each row groups settled closing choices by their original American odds "
        "in a fixed [low, high) bin. Opening ROI uses the opening model's own "
        "choices on those same fights. Both strategies stake one unit per fight.\n\n"
        "`required_adjustment` is the smallest common favorable shift applied to "
        "every closing choice in the bin that reaches the opening ROI. Picks and "
        "outcomes stay fixed. Zero means the closing ROI already meets the target. "
        "Missing means no solution within the allowed adjustment.\n\n"
        "The shift removes the artificial discontinuity at even money: -150 to "
        "+150 is 100 continuous points. `mean_raw_american_change` also reports "
        "the literal signed-number difference; it includes that discontinuity. "
        "`mean_decimal_increase` reports the average payout increase.\n\n"
        "`average_gap_closed_pp_per_point` measures the average ROI improvement "
        "per continuous point up to the target. The convergence CSV contains "
        "the full adjustment path, remaining gap, and percentage of gap closed. "
        "Favorite payout changes are nonlinear.\n\n"
        "Use `n_bets` and `n_events` to assess sample support. These are historical "
        "sensitivity estimates using observed outcomes, not prospective thresholds "
        "validated on another holdout. No models are retrained or stakes recalculated.\n",
        encoding="utf-8",
    )
    print(summary[["odds_type", "bin_low", "n_bets", "close_roi_pct", "open_target_roi_pct",
                   "required_adjustment", "status"]].round(3).to_string(index=False))
    return summary


def plot_convergence(output_dir=None):
    """Export ROI/shift curves from saved analysis, including every populated bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir) if output_dir else settings.data_dir / "model_results" / "odds_sensitivity"
    summary = pd.read_csv(output_dir / "sensitivity_by_moneyline_bin.csv")
    curves = pd.read_csv(output_dir / "convergence_curves.csv")
    colors = {"close1": "#2366af", "close2": "#c25916"}
    bins = sorted(summary.bin_low.unique())

    def page(selected, title):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        fig.suptitle(title, fontsize=19)
        for ax, low in zip(axes.flat, selected):
            rows = summary.loc[summary.bin_low.eq(low)]
            thresholds = rows.required_adjustment.dropna()
            xmax = max(100, float(thresholds.max()) * 1.4) if len(thresholds) else 500
            xmax = min(float(curves.adjustment.max()), np.ceil(xmax / 50) * 50)
            for row in rows.itertuples():
                curve = curves.loc[curves.bin_low.eq(low) & curves.odds_type.eq(row.odds_type)].sort_values("adjustment")
                # Include an interpolated endpoint so both lines share the axis extent.
                x = np.unique(np.append(curve.loc[curve.adjustment.le(xmax), "adjustment"], xmax))
                y = np.interp(x, curve.adjustment, curve.adjusted_roi_pct)
                color = colors[row.odds_type]
                ax.plot(x, y, color=color, linewidth=2.2,
                        label=f"{row.odds_type} ROI (n={row.n_bets})")
                ax.axhline(row.open_target_roi_pct, color=color, linestyle="--", alpha=.7,
                           label=f"{row.odds_type} opening target")
                shift = row.required_adjustment
                if np.isfinite(shift) and shift <= xmax:
                    level = row.matched_roi_pct
                    ax.plot(shift, level, "o", color=color, markersize=6)
                    ax.annotate(f"{shift:.1f} pts" if shift > 0 else "Already meets target",
                                (shift, level), xytext=(7, 10 if row.odds_type == "close1" else -20),
                                textcoords="offset points", fontsize=9, color=color)
            high = int(rows.bin_high_exclusive.iloc[0])
            ax.set_title(f"Moneylines [{int(low):+d}, {high:+d})", fontsize=12)
            ax.set_xlabel("Favorable moneyline shift (points)")
            ax.set_ylabel("ROI (%)")
            ax.set_xlim(0, xmax * 1.04)
            ax.grid(alpha=.18)
            ax.legend(fontsize=8, loc="best")
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.supxlabel("Fixed picks and one-unit stakes. Dashed targets use opening picks on the same fights within each closing bin.\n"
                       "Bins exclude the upper boundary. Shift is continuous through even money: -150 to +150 = 100 points. Small bins are descriptive only.", fontsize=10)
        return fig

    common = summary.groupby("bin_low").n_bets.sum().nlargest(6).index.sort_values().tolist()
    overview = page(common, "How ROI increases as closing prices improve — six largest bins")
    overview.savefig(output_dir / "roi_convergence_overview.png", dpi=160)
    plt.close(overview)
    with PdfPages(output_dir / "roi_convergence_all_bins.pdf") as pdf:
        for start in range(0, len(bins), 6):
            fig = page(bins[start:start + 6], "Closing ROI versus price improvement — all moneyline bins")
            pdf.savefig(fig)
            plt.close(fig)
    return output_dir / "roi_convergence_overview.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--bin-width", type=int, default=50)
    parser.add_argument("--max-shift", type=float, default=5000)
    parser.add_argument("--step", type=float, default=10)
    parser.add_argument("--plot-only", action="store_true", help="Plot previously saved convergence curves")
    args = parser.parse_args()
    plot_only = vars(args).pop("plot_only")
    if not plot_only:
        run_sensitivity(**vars(args))
    plot_convergence(args.output_dir if args.output_dir else
                     (args.data_dir / "odds_sensitivity" if args.data_dir else None))
