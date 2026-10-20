"""Daily event moneyline EV relative to opening, from committed snapshots."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .six_day_odds_ev import snapshot_moneyline_ev


def plot_intermediate_moneyline_ev(snapshot_dir, opening_params, closing_params,
                                  max_days=60, output_dir=None,
                                  exclude_from_average=None, max_event_ev_pct=300):
    """Return one plot of daily mean EV ratios and their arithmetic midpoint.

    Latest commit per UTC event/day, no fill of unobserved days. Opening EV is
    recomputed from each snapshot's features and recorded opening prices, on the
    same eligible fights as its closing EV. Both stages use the saved open model.
    """
    snapshot_dir = Path(snapshot_dir)
    out = Path(output_dir) if output_dir else snapshot_dir / "ev_over_time"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    manifest["days_before"] = (pd.to_datetime(manifest.event_date, utc=True)
                               - manifest.timestamp.dt.floor("D")).dt.days
    selected = manifest.loc[manifest.days_before.ge(1)]
    if max_days is not None:
        selected = selected.loc[selected.days_before.le(max_days)]
    selected = selected.sort_values(["timestamp", "commit", "original_path"]).drop_duplicates(
        ["event_date", "days_before"], keep="last")
    if selected.empty:
        raise ValueError("No pre-event snapshots in the requested window.")
    events = snapshot_moneyline_ev(snapshot_dir, opening_params, closing_params, selected)
    events = events.rename(columns={"six_day_ev_pct": "closing_ev_pct"}).merge(
        selected[["filename", "days_before", "commit_datetime_utc"]],
        on="filename", validate="many_to_one")
    excluded = set(exclude_from_average or [])
    if max_event_ev_pct is not None:
        excluded.update(events.loc[events.pct_ev_of_open.gt(max_event_ev_pct), "event_date"])
    valid = events.loc[np.isfinite(events.pct_ev_of_open)
                       & ~events.event_date.isin(excluded)]
    average = valid.groupby(["odds_type", "days_before"]).agg(
        mean_pct_ev=("pct_ev_of_open", "mean"),
        median_pct_ev=("pct_ev_of_open", "median"),
        events=("event_date", "nunique"),
    ).reset_index()
    days = np.arange(1, int(selected.days_before.max()) + 1)
    average_path, midpoint = plot_average_ev(average, days, out, excluded)
    events.to_csv(out / "event_ev_over_time.csv", index=False)
    average.to_csv(out / "average_ev_over_time.csv", index=False)
    selected.to_csv(out / "selected_snapshots.csv", index=False)
    excluded_events = events.loc[events.event_date.isin(excluded)]
    excluded_events.to_csv(out / "excluded_event_ev.csv", index=False)
    return {"events": events, "average": average, "selected_snapshots": selected,
            "midpoint": midpoint, "average_plot_path": average_path,
            "excluded_events": excluded_events}


def plot_average_ev(average, days, output_dir, excluded=()):
    """Plot both daily means and (close1 mean + close2 mean) / 2."""
    means = average.pivot(index="days_before", columns="odds_type", values="mean_pct_ev")
    means = means.reindex(index=days, columns=["close1", "close2"])
    midpoint = ((means["close1"] + means["close2"]) / 2).rename("midpoint_pct_ev")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for stage, color in (("close1", "#3579ad"), ("close2", "#df8332")):
        ax.plot(days, means[stage], color=color, marker=".", label=f"{stage} average")
    ax.plot(days, midpoint, color="#228765", linestyle="--", linewidth=2.5,
            label="Midpoint of close1 and close2 averages")
    ax.axhline(100, color="gray", linestyle=":", label="Opening EV (100%)")
    note = f" | excluded: {', '.join(sorted(excluded))}" if excluded else ""
    ax.set(title=f"Average moneyline EV relative to opening{note}",
           ylabel="Opening EV (%)", xlabel="Days before event (UTC commit date)",
           xlim=(max(days)+.5, .5))
    ax.legend(); ax.grid(alpha=.2)
    path = Path(output_dir) / "average_ev_over_time.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    midpoint.reset_index().to_csv(Path(output_dir) / "midpoint_ev_over_time.csv", index=False)
    return path, midpoint
