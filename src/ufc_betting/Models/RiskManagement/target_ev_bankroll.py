"""Expected bankroll paths held to per-event fractions of opening EV."""
import json
import numpy as np
import pandas as pd
from ufc_betting.config import settings


def build_paths(targets, starting_bankroll=500):
    if starting_bankroll <= 0 or not np.isfinite(starting_bankroll):
        raise ValueError("Invalid initial bankroll")
    targets = targets.copy()
    targets["date"] = pd.to_datetime(targets.date)
    if targets.duplicated(["date", "target_pct_opening_ev"]).any():
        raise ValueError("Duplicate event targets")
    records = []
    expected_dates = set(targets.date)
    for target, group in targets.groupby("target_pct_opening_ev"):
        group = group.sort_values("date").copy()
        if set(group.date) != expected_dates:
            raise ValueError("All targets must cover identical events")
        requested = group.opening_expected_return_pct * target / 100
        achieved = group.opening_expected_return_pct * group.achieved_pct_opening_ev / 100
        zero = group.opening_expected_return_pct.abs() < 1e-10
        if ((~zero) & (achieved.isna() | (achieved + 1e-8 < requested))).any():
            raise ValueError("Some positive EV targets are unreachable")
        group["stake_multiplier_to_hold_target"] = np.where(zero, 0, requested / achieved)
        group["held_event_expected_return_pct"] = requested
        group["bankroll_after_event"] = starting_bankroll * (1 + requested / 100).cumprod()
        group["bankroll_before_event"] = group.bankroll_after_event.shift(1, fill_value=starting_bankroll)
        group["cumulative_return_pct"] = 100 * (group.bankroll_after_event / starting_bankroll - 1)
        records.append(group)
    return pd.concat(records, ignore_index=True)


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    root = settings.data_dir / "model_results" / "opening_model_ev_targets"
    targets = pd.read_csv(root / "required_movement_by_event.csv")
    paths = build_paths(targets)
    out = root / "target_bankroll"
    out.mkdir(parents=True, exist_ok=True)
    paths.to_csv(out / "event_bankroll_paths.csv", index=False)
    ending = paths.groupby("target_pct_opening_ev").tail(1)[["target_pct_opening_ev", "bankroll_after_event", "cumulative_return_pct"]]
    ending.to_csv(out / "ending_bankroll_by_ev_target.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for target, group in paths.groupby("target_pct_opening_ev"):
        x = np.arange(len(group) + 1)
        for ax in axes:
            ax.plot(x, np.r_[500, group.bankroll_after_event], lw=2, label=f"{target:g}% opening EV")
    # The 100% held path and actual opening benchmark have identical event EVs.
    benchmark = paths.loc[paths.target_pct_opening_ev.eq(100)]
    for ax in axes:
        ax.plot(np.arange(len(benchmark)+1), np.r_[500,benchmark.bankroll_after_event],
                color="black", linestyle="--", linewidth=1.5, label="Opening benchmark (=100%)")
        ax.set_xlabel("Completed test events, chronological order")
        ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    axes[0].set(title="Cumulative expected bankroll", ylabel="Bankroll ($)")
    axes[1].set(title="Same paths on logarithmic scale", ylabel="Bankroll ($, log scale)", yscale="log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Event EV held to 50–100% of opening EV | {len(benchmark)} test events | $500 start",fontsize=16)
    fig.supxlabel("B(next) = B(current) × [1 + target fraction × opening event expected return]. Excess EV is removed by proportional stake reduction.\n"
                  "Model-implied projections, not realized returns or typical wealth. Opening model; MDD 0.40, N 250, z 0.5; no parlays.",fontsize=10)
    fig.savefig(out / "cumulative_bankroll_by_ev_target.png",dpi=150)
    plt.close(fig)
    (out / "methodology.json").write_text(json.dumps({"starting_bankroll":500,"events":len(benchmark),
        "target_levels":[50,60,70,80,90,100],"formula":"product(1 + target_fraction * opening_expected_event_return/100)",
        "implementation":"First qualifying movement per event, then proportionally reduce all event stakes by target EV / achieved EV. No stake increases.",
        "zero_opening_ev":"No bet; bankroll unchanged. Ratio and required movement remain undefined.",
        "risk":"Inherits notebook MDD .4, N 250, z .5, and 100% event exposure cap; target scaling only reduces exposure.",
        "interpretation":"Exact model-EV targets; not an independently validated prediction of realized or typical compounded growth. 100% path equals opening benchmark by construction."},indent=2),encoding="utf-8")
    print(ending.to_string(index=False))


if __name__ == "__main__":
    run()
