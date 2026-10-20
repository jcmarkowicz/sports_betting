"""Select exact six-day archive snapshots and evaluate moneyline event EV."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def select_six_day_snapshots(source_dir, destination_dir):
    source, destination = Path(source_dir), Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source / "manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    manifest["days_before"] = (pd.to_datetime(manifest.event_date, utc=True)
                               - manifest.timestamp.dt.floor("D")).dt.days
    selected = manifest.loc[manifest.days_before.eq(6)].sort_values(
        ["timestamp", "commit", "original_path"]).drop_duplicates("event_date", keep="last")
    for row in selected.itertuples():
        target = destination / row.filename
        if target.exists() and target.read_bytes() != (source / row.filename).read_bytes():
            raise ValueError(f"Existing snapshot differs: {target}")
        shutil.copyfile(source / row.filename, target)
    missing = pd.DataFrame({"event_date": sorted(set(manifest.event_date) - set(selected.event_date))})
    selected.to_csv(destination / "manifest.csv", index=False)
    missing.to_csv(destination / "missing_six_day_events.csv", index=False)
    return selected, missing



def plot_six_day_moneyline_ev(snapshot_dir, opening_params, closing_params, bins=15,
                            exclude_event_dates=None):
    """Plot six-day event EV ratios; exclusions affect only the histogram."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    snapshot_dir = Path(snapshot_dir)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv")
    events = snapshot_moneyline_ev(snapshot_dir, opening_params, closing_params, manifest)
    excluded_dates = set(exclude_event_dates or [])
    plot_events = events.loc[~events.event_date.isin(excluded_dates)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, layout="constrained")
    finite = plot_events.loc[np.isfinite(plot_events.pct_ev_of_open), "pct_ev_of_open"]
    edges = np.histogram_bin_edges(finite, bins=bins) if len(finite) else np.linspace(0, 100, bins+1)
    for ax, stage in zip(axes, ("close1", "close2")):
        group = events.loc[events.odds_type.eq(stage)]
        ratios = group.loc[np.isfinite(group.pct_ev_of_open)
                           & ~group.event_date.isin(excluded_dates), "pct_ev_of_open"]
        ax.hist(ratios, bins=edges, edgecolor="white", color="#3579ad")
        ax.axvline(100, color="gray", linestyle="--", label="Opening EV (100%)")
        if len(ratios):
            ax.axvline(ratios.median(), color="#c44e52", label=f"Median: {ratios.median():.1f}%")
        ax.set(title=f"{stage}: {len(ratios)} events | {len(group)-len(ratios)} excluded",
               xlabel="Six-day moneyline event EV / opening event EV (%)", ylabel="Event count")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Moneyline EV six days before event | recorded prices, fixed opening model")
    events.to_csv(snapshot_dir / "moneyline_event_ev.csv", index=False)
    fig.savefig(snapshot_dir / "moneyline_pct_open_ev_histogram.png", dpi=150)
    return {"events": events, "figure": fig, "manifest": manifest,
            "excluded_events": events.loc[events.event_date.isin(excluded_dates)]}
