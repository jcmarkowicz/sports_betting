"""Compound saved model-implied event returns at each odds-shift percentage."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ufc_betting.config import config, settings


def compound(curves, starting_bankroll):
    if not np.isfinite(starting_bankroll) or starting_bankroll <= 0:
        raise ValueError("Starting bankroll must be positive and finite")
    curves = curves.copy()
    curves["date"] = pd.to_datetime(curves.date, errors="raise")
    if curves.duplicated(["date", "shift_pct"]).any():
        raise ValueError("Duplicate event/shift rows")
    matrix = curves.pivot(index="date", columns="shift_pct", values="expected_bankroll_return_pct").sort_index()
    if not np.isfinite(matrix.to_numpy()).all() or (matrix <= -100).any().any():
        raise ValueError("Every shift must cover the same events with finite returns above -100%")
    paths = []
    for shift in matrix.columns:
        returns = matrix[shift]
        multiplier = (1 + returns / 100).cumprod()
        end = starting_bankroll * multiplier
        begin = end.shift(1, fill_value=starting_bankroll)
        paths.append(pd.DataFrame({"date": matrix.index, "shift_pct": shift,
                     "event_return_pct": returns.to_numpy(), "bankroll_before_event": begin.to_numpy(),
                     "expected_event_profit": (end - begin).to_numpy(), "bankroll_after_event": end.to_numpy(),
                     "cumulative_return_pct": ((multiplier - 1) * 100).to_numpy()}))
    paths = pd.concat(paths, ignore_index=True)
    ending = paths.groupby("shift_pct", sort=True).tail(1).copy()
    ending = ending[["shift_pct", "bankroll_after_event", "cumulative_return_pct"]].rename(columns={"bankroll_after_event": "ending_bankroll"})
    ending["growth_multiple"] = ending.ending_bankroll / starting_bankroll
    ending["geometric_mean_event_return_pct"] = 100 * (ending.growth_multiple ** (1 / len(matrix)) - 1)
    baseline = ending.loc[ending.shift_pct.eq(0), "ending_bankroll"].iloc[0]
    ending["increase_over_unshifted_pct"] = 100 * (ending.ending_bankroll / baseline - 1)
    ending["ending_bankroll_change_per_shift_point"] = ending.ending_bankroll.diff() / ending.shift_pct.diff()
    return paths, ending.reset_index(drop=True)


def run(input_dir=None, output_dir=None, starting_bankroll=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    input_dir = Path(input_dir) if input_dir else settings.data_dir / "model_results" / "event_odds_sensitivity"
    output_dir = Path(output_dir) if output_dir else input_dir / "compounded"
    starting_bankroll = config.bankroll if starting_bankroll is None else starting_bankroll
    curves = pd.read_csv(input_dir / "event_curves.csv", parse_dates=["date"])
    upstream = json.loads((input_dir / "methodology.json").read_text(encoding="utf-8"))
    paths, ending = compound(curves, starting_bankroll)
    targets = curves.groupby("date").open_target_pct
    if not targets.nunique().eq(1).all():
        raise ValueError("Opening benchmark changes with shift")
    opening = targets.first().sort_index()
    opening_bankroll = starting_bankroll * (1 + opening / 100).cumprod()
    if opening.sum() <= 0:
        raise ValueError("Percent of opening EV requires a positive aggregate opening return")
    ev_ratio = 100 * curves.groupby("shift_pct").expected_bankroll_return_pct.sum() / opening.sum()
    ending["pct_of_opening_ev"] = ending.shift_pct.map(ev_ratio)
    paths["pct_of_opening_ev_test_set"] = paths.shift_pct.map(ev_ratio)
    paths["pct_of_opening_ev_event"] = 100 * paths.event_return_pct / paths.date.map(opening).replace(0, np.nan)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output_dir / "bankroll_paths_all_shifts.csv", index=False)
    ending.to_csv(output_dir / "ending_bankroll_by_shift.csv", index=False)
    ending.to_csv(output_dir / "ending_bankroll_by_opening_ev.csv", index=False)
    opening_bankroll.rename("opening_expected_bankroll").to_csv(output_dir / "opening_bankroll.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    available = np.sort(paths.shift_pct.unique())
    selected = np.unique([available[np.abs(available - value).argmin()] for value in [0, 25, 50, 75, 100]])
    for shift in selected:
        group = paths.loc[paths.shift_pct.eq(shift)].sort_values("date")
        axes[0].plot(np.arange(len(group) + 1), np.r_[starting_bankroll, group.bankroll_after_event],
                     label=f"Close1: {ev_ratio.loc[shift]:.1f}% of opening EV", lw=2)
    axes[0].plot(np.arange(len(opening) + 1), np.r_[starting_bankroll, opening_bankroll],
                 color="black", ls="--", label="Opening: 100% EV", lw=2)
    axes[0].set_xlabel("Completed events (chronological order)")
    axes[0].set_ylabel("Projected bankroll ($, logarithmic scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("Cumulative bankroll paths")
    axes[0].legend(fontsize=9)
    axes[1].plot(ending.pct_of_opening_ev, ending.ending_bankroll, color="#2366af", lw=2, label="Close1 scenarios")
    axes[1].scatter([100], [opening_bankroll.iloc[-1]], color="black", marker="D", label="Opening benchmark")
    axes[1].annotate(f"Opening: ${opening_bankroll.iloc[-1]:,.0f}", (100, opening_bankroll.iloc[-1]),
                     xytext=(-10, -20), textcoords="offset points", ha="right")
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Expected event return as % of opening EV (test-set aggregate)")
    axes[1].set_ylabel("Ending projected bankroll ($)")
    axes[1].set_title("Ending bankroll versus percentage of opening EV")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    fig.suptitle(f"Compounded model-implied returns | {len(opening)} events | ${starting_bankroll:,.0f} starting bankroll", fontsize=16)
    fig.supxlabel("Opening EV % = 100 × sum(close1 expected event returns) / sum(opening expected event returns). Event ratios vary.\n"
                  "Model-implied projections, not realized returns. "
                  f"Uncapped inherited stakes: maximum event exposure is {100 * curves.stake_fraction.max():.1f}% of bankroll.", fontsize=10)
    fig.savefig(output_dir / "compounded_bankroll.png", dpi=150)
    plt.close(fig)
    metadata = {"starting_bankroll": starting_bankroll, "events": len(opening),
                "risk_parameters": {key: upstream[key] for key in ("mdd_open", "mdd_close1", "N_open", "N_close1")},
                "maximum_event_stake_fraction": float(curves.stake_fraction.max()),
                "opening_ending_bankroll": float(opening_bankroll.iloc[-1]),
                "opening_ev_percent_definition": "100 * sum(close1 expected bankroll return percentages) / sum(opening expected bankroll return percentages); aggregate across identical test events, not a uniform per-event percentage",
                "increment": "0-100% close1-to-open price interpolation; 1% steps inherited from event curves",
                "formula": "B_n = B_0 * product(1 + event_expected_bankroll_return_pct / 100)",
                "notes": ["Same shift used for every event in each scenario; probabilities and fstar were recomputed upstream.",
                          "Multiplying model means is a deterministic projection, not a realized backtest or expected log growth.",
                          "Each event uses its current projected bankroll as the staking base; no within-event compounding.",
                          "Risk parameters inherited from event analysis; see risk_parameters. No exposure cap added."]}
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(ending.loc[ending.shift_pct.isin(selected)].to_string(index=False))
    print(f"Opening ending bankroll: {opening_bankroll.iloc[-1]:,.2f}")
    print(f"Maximum event stake fraction in inputs: {curves.stake_fraction.max():.4f}")
    return paths, ending


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--starting-bankroll", type=float)
    run(**vars(parser.parse_args()))
