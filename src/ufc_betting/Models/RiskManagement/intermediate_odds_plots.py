"""Historical opening-relative price movements from timestamped Git snapshots."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_first_snapshot_regressions(snapshot_dir, output_dir=None):
    """OLS of closing on opening decimal prices, first committed CSV per event.

    Includes both fighter sides. Selection precedes price validation: missing
    prices never cause a later snapshot to replace an event's first snapshot.
    """
    import statsmodels.api as sm

    snapshot_dir = Path(snapshot_dir)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    first = manifest.sort_values(
        ["timestamp", "commit", "original_path"]
    ).drop_duplicates("event_date", keep="first")
    frames = []
    for row in first.itertuples():
        data = pd.read_csv(snapshot_dir / row.filename)
        for side, other in (("red", "blue"), ("blue", "red")):
            frames.append(pd.DataFrame({
                "event_date": row.event_date,
                "filename": row.filename,
                "fighter": data[f"fighter_{side}"].str.strip().str.lower(),
                "opponent": data[f"fighter_{other}"].str.strip().str.lower(),
                **{stage: pd.to_numeric(data[f"dec_{stage}_{side}"], errors="coerce")
                   for stage in ("open", "close1", "close2")},
            }))
    observations = pd.concat(frames, ignore_index=True).dropna(subset=["fighter", "opponent"])
    keys = ["event_date", "fighter", "opponent"]
    observations = observations.drop_duplicates()
    if observations.duplicated(keys).any():
        raise ValueError("Conflicting duplicate fighter prices in first snapshots.")
    figures, models, summaries = {}, {}, []
    for stage in ("close1", "close2"):
        valid = observations[np.isfinite(observations.open) & np.isfinite(observations[stage])
                             & observations.open.gt(1) & observations[stage].gt(1)].copy()
        if len(valid) < 3 or valid.open.nunique() < 2:
            raise ValueError(f"Insufficient valid prices for {stage} regression.")
        model = sm.OLS(valid[stage], sm.add_constant(valid[["open"]], has_constant="add")).fit()
        models[stage] = model
        intercept, slope = model.params["const"], model.params["open"]
        summaries.append({"odds_type": stage, "samples": len(valid),
                          "events": valid.event_date.nunique(), "intercept": intercept,
                          "slope": slope, "r_squared": model.rsquared,
                          "excluded_price_rows": len(observations) - len(valid)})
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        ax.scatter(valid.open, valid[stage], alpha=.4, s=25, label="Fighter prices")
        low = min(valid.open.min(), valid[stage].min())
        high = max(valid.open.max(), valid[stage].max())
        ax.plot([low, high], [low, high], color="gray", linestyle="--", label="Unchanged price (y = x)")
        x = np.linspace(valid.open.min(), valid.open.max(), 200)
        ax.plot(x, intercept + slope * x, color="#c44e52", linewidth=2,
                label=f"OLS: y = {intercept:.3f} + {slope:.3f}x | R² = {model.rsquared:.3f}")
        ax.set(xlabel="Opening decimal odds", ylabel=f"{stage} decimal odds",
               title=f"First committed CSV per event: opening vs {stage}\n"
                     f"{len(valid):,} fighter samples | {valid.event_date.nunique()} events")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        figures[stage] = fig
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / f"first_snapshot_open_vs_{stage}.png", dpi=150)
    return {"selected_snapshots": first, "observations": observations,
            "summary": pd.DataFrame(summaries), "models": models, "figures": figures}


def plot_intermediate_odds_movements(snapshot_dir, max_days=60, output_dir=None):
    """Plot four daily movement curves; count each fighter/matchup once per UTC day.

    Uses the latest committed observation on each pre-event calendar day, without
    filling missing days. Opening prices are those recorded in the same snapshot.
    Event-day and later commits are excluded because event start times are unknown.
    Percent change is 100 * (closing_decimal / opening_decimal - 1).
    """
    snapshot_dir = Path(snapshot_dir)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    manifest["event_day"] = pd.to_datetime(manifest.event_date, utc=True)
    manifest["days_before"] = (
        manifest.event_day - manifest.timestamp.dt.floor("D")
    ).dt.days
    eligible = manifest[manifest.days_before.ge(1)].copy()
    if max_days is not None:
        eligible = eligible[eligible.days_before.le(max_days)]
    frames = []
    for row in eligible.sort_values(["timestamp", "commit", "original_path"]).itertuples():
        columns = ["fighter_red", "fighter_blue"] + [
            f"dec_{stage}_{side}" for stage in ("open", "close1", "close2")
            for side in ("red", "blue")
        ]
        data = pd.read_csv(snapshot_dir / row.filename, usecols=columns)
        data = data.dropna(subset=["fighter_red", "fighter_blue"]).copy()
        for side, opponent in (("red", "blue"), ("blue", "red")):
            part = pd.DataFrame({
                "event_date": row.event_date, "days_before": row.days_before,
                "timestamp": row.timestamp, "commit": row.commit,
                "fighter": data[f"fighter_{side}"].str.strip().str.lower(),
                "opponent": data[f"fighter_{opponent}"].str.strip().str.lower(),
                **{stage: pd.to_numeric(data[f"dec_{stage}_{side}"], errors="coerce")
                   for stage in ("open", "close1", "close2")},
            })
            frames.append(part)
    if not frames:
        raise ValueError("No pre-event snapshots within the requested time window.")
    daily = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["event_date", "fighter", "opponent", "days_before"], keep="last"
    )
    observations, summaries, figures = [], [], {}
    for stage in ("close1", "close2"):
        valid = daily[np.isfinite(daily[stage]) & np.isfinite(daily.open)
                      & daily[stage].gt(1) & daily.open.gt(1)].copy()
        valid["movement_pct"] = 100 * (valid[stage] / valid.open - 1)
        valid["odds_type"] = stage
        valid["direction"] = np.where(
            np.isclose(valid.movement_pct, 0, atol=1e-9), "unchanged",
            np.where(valid.movement_pct < 0, "decay", "gain"))
        observations.append(valid)
        for direction, color in (("decay", "#c44e52"), ("gain", "#228765")):
            subset = valid[valid.direction.eq(direction)]
            summary = subset.groupby("days_before").agg(
                mean_pct=("movement_pct", "mean"),
                median_pct=("movement_pct", "median"),
                q25=("movement_pct", lambda x: x.quantile(.25)),
                q75=("movement_pct", lambda x: x.quantile(.75)),
                samples=("movement_pct", "size"),
                events=("event_date", "nunique"),
            ).reindex(range(1, int(eligible.days_before.max()) + 1))
            summary[["samples", "events"]] = summary[["samples", "events"]].fillna(0).astype(int)
            summary["odds_type"], summary["direction"] = stage, direction
            summaries.append(summary.reset_index())
            fig, (ax, count_ax) = plt.subplots(
                2, 1, figsize=(12, 6), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
            x = summary.index.to_numpy()
            ax.fill_between(x, summary.q25, summary.q75, color=color, alpha=.16,
                            label="Middle 50% of observations")
            ax.plot(x, summary.mean_pct, color=color, marker=".", label="Mean")
            ax.plot(x, summary.median_pct, color=color, linestyle="--", label="Median")
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set_ylabel("Change from opening decimal odds (%)")
            n_fighters = len(subset.drop_duplicates(["event_date", "fighter", "opponent"]))
            ax.set_title(f"Opening to {stage}: price {direction}\n"
                         f"{len(subset):,} fighter-day samples | {n_fighters:,} fighter/matchups | "
                         f"{subset.event_date.nunique()} events")
            ax.legend(loc="best", fontsize=9)
            ax.grid(alpha=.2)
            count_ax.bar(x, summary.samples, color=color, alpha=.7)
            count_ax.set_ylabel("Samples")
            count_ax.set_xlabel("Days before event (UTC commit date; latest observation per day)")
            count_ax.set_xlim(max(x) + .75, .25)
            count_ax.grid(axis="y", alpha=.2)
            figures[f"{stage}_{direction}"] = fig
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(output_dir) / f"{stage}_{direction}.png", dpi=150)
    observations = pd.concat(observations, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    counts = observations.groupby(["odds_type", "direction"]).agg(
        fighter_day_samples=("fighter", "size"), events=("event_date", "nunique")
    ).reset_index()
    return {"observations": observations, "daily_summary": summary,
            "counts": counts, "figures": figures}
