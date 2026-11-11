"""Dataframe-only Brier history, pre-event trends and reduction-only stake filters.

Dates identify events (one event per date). Fight rows retain their input order
within an event. Supply complete settled events, not partially settled cards.
History must contain predictions from the intended model fit; call separately
for each fit/refit, with its matching warmup. No files are loaded or plots made.

Typical use::

    scored = calculate_brier_scores(settled_fights, odds_type="open")
    history = calculate_event_history(scored)
    scale = fit_trend_scale(history, training_end="2024-05-04")
    past = calculate_event_trends(past_events, history, trend_scale=scale)
    upcoming = calculate_event_trends(upcoming_events, history, trend_scale=scale)
    result = apply_event_filters(upcoming, past, strategy="combined")

Events have a date column (or DatetimeIndex); historical return_pct is measured
in percentage points. Upcoming events need no outcome or return. To transfer
opening filters to closing returns, calculate trends and historical scalers from
opening data, but supply target event rows containing closing stakes/returns.
The filter output preserves target rows; skipped events receive zero multipliers.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalingConfig:
    fight_window: int = 100
    smoothing: str = "ewma"
    smoothing_events: int = 5
    slope_events: int = 5
    flat_cutoff: float = 0.02
    mean_years: int = 5
    volatility_events: int = 10
    volatility_target: str = "above_mean"  # or current_mean_group
    brier_min_events: int = 20
    brier_ridge: float = 10.0

    def __post_init__(self):
        for name in ("fight_window", "smoothing_events", "slope_events",
                     "mean_years", "volatility_events", "brier_min_events"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.slope_events < 2 or self.volatility_events < 2:
            raise ValueError("Slope and volatility windows must be at least 2")
        if self.smoothing not in {"ewma", "rolling"}:
            raise ValueError("smoothing must be ewma or rolling")
        if self.volatility_target not in {"above_mean", "current_mean_group"}:
            raise ValueError("Unknown volatility_target")
        if not np.isfinite(self.flat_cutoff) or self.flat_cutoff < 0:
            raise ValueError("flat_cutoff must be finite and nonnegative")
        if not np.isfinite(self.brier_ridge) or self.brier_ridge < 0:
            raise ValueError("brier_ridge must be finite and nonnegative")


def _dated(frame, *, unique=False):
    out = frame.copy()
    if "date" not in out:
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("Provide a date column or DatetimeIndex")
        out["date"] = out.index
    dates = pd.to_datetime(out["date"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Dates must be nonmissing timezone-naive event dates")
    out["date"] = dates.dt.normalize()
    if unique and out.date.duplicated().any():
        raise ValueError("Expected one event per date")
    return out


def calculate_brier_scores(fights, *, odds_type="open", probability_col="proba_red",
                           outcome_col="winner", config=ScalingConfig()):
    """Score eligible settled fights and average the last N fights (equal weights).

    winner=1 means red wins. No-vig red probability is normalized inverse odds.
    Unsettled/invalid rows are excluded. Dataframe attrs record excluded count.
    Expected Vegas Brier is q*(1-q); excess is actual minus expected;
    advantage is Vegas minus model Brier. Positive advantage favors the model.
    """
    out = _dated(fights).reset_index(drop=True).sort_values("date", kind="stable").reset_index(drop=True)
    p = pd.to_numeric(out[probability_col], errors="coerce")
    y = pd.to_numeric(out[outcome_col], errors="coerce")
    red = pd.to_numeric(out[f"dec_{odds_type}_red"], errors="coerce")
    blue = pd.to_numeric(out[f"dec_{odds_type}_blue"], errors="coerce")
    valid = y.isin([0, 1]) & p.between(0, 1) & np.isfinite(red) & np.isfinite(blue) & red.gt(1) & blue.gt(1)
    q = (1/red) / (1/red + 1/blue)
    out = out.loc[valid].copy()
    out["vegas_probability"] = q.loc[valid]
    out["vegas_actual"] = (q.loc[valid] - y.loc[valid])**2
    out["vegas_expected"] = q.loc[valid] * (1-q.loc[valid])
    out["model_brier"] = (p.loc[valid] - y.loc[valid])**2
    out["excess"] = out.vegas_actual - out.vegas_expected
    out["advantage"] = out.vegas_actual - out.model_brier
    for col in ["vegas_actual", "vegas_expected", "model_brier", "excess", "advantage"]:
        out[f"rolling_{col}"] = out[col].rolling(config.fight_window, min_periods=config.fight_window).mean()
    out.attrs["excluded_fights"] = int((~valid).sum())
    return out.reset_index(drop=True)


def calculate_event_history(scored_fights, *, config=ScalingConfig()):
    """Aggregate completed events and calculate unshifted smoothed advantage/slope.

    Event Brier is the mean of that event's fights; rolling fields are the final
    fight's trailing-window value. Use calculate_event_trends before betting.
    """
    fights = _dated(scored_fights).reset_index(drop=True).sort_values("date", kind="stable")
    cols = ["vegas_actual", "vegas_expected", "model_brier", "excess", "advantage"]
    grouped = fights.groupby("date", sort=True)
    out = grouped[cols].mean()
    out["fights"] = grouped.size()
    for col in cols:
        out[f"rolling_{col}"] = grouped[f"rolling_{col}"].last()
    ends = out.rolling_advantage.dropna()
    smooth = (ends.ewm(span=config.smoothing_events, adjust=False).mean()
              if config.smoothing == "ewma" else
              ends.rolling(config.smoothing_events, min_periods=config.smoothing_events).mean())
    t = np.arange(config.slope_events, dtype=float)
    t -= t.mean()
    out["smoothed_advantage"] = smooth
    out["advantage_slope"] = smooth.rolling(config.slope_events).apply(
        lambda values: np.dot(t, values)/np.dot(t, t), raw=True)
    return out


def fit_trend_scale(event_history, *, training_end):
    """Training-period std of smoothed level, frozen before evaluating events.

    Only use a training_end preceding evaluation dates. Returned scalar can be
    persisted with model parameters; no future-dependent normalization is fitted
    inside calculate_event_trends.
    """
    history = _dated(event_history, unique=True)
    scale = history.loc[history.date <= pd.Timestamp(training_end).normalize(), "smoothed_advantage"].std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training history must provide positive finite trend scale")
    return float(scale)


def calculate_event_trends(events, event_history, *, trend_scale, config=ScalingConfig()):
    """Attach strictly pre-event features, preserving input order and index.

    Uses the last completed event strictly before each target date, not a shift
    of the target dataframe. Missing history is Unknown (never silently Flat).
    Multiple upcoming dates use only supplied settled history, no forecasts.
    """
    if not np.isfinite(trend_scale) or trend_scale <= 0:
        raise ValueError("trend_scale must be positive and finite")
    out = _dated(events, unique=True)
    history = _dated(event_history, unique=True).reset_index(drop=True).sort_values("date").reset_index(drop=True)
    records = []
    for date in out.date:
        prior = history.loc[history.date < date]
        reference = prior.loc[prior.date >= date-pd.DateOffset(years=config.mean_years), "rolling_advantage"].dropna()
        last = prior.iloc[-1] if len(prior) else None
        advantage = last.rolling_advantage if last is not None else np.nan
        slope = last.advantage_slope/trend_scale if last is not None else np.nan
        mean = reference.mean()
        ready = bool(np.isfinite([advantage, slope, mean]).all())
        trend = ("Uptrend" if slope > config.flat_cutoff else "Downtrend" if slope < -config.flat_cutoff else "Flat") if ready else "Unknown"
        records.append(dict(source_date=last.date if last is not None else pd.NaT,
            prior_advantage=advantage, prior_brier=last.vegas_actual if last is not None else np.nan,
            slope=slope, five_year_mean=mean, reference_events=len(reference),
            regime=("Above mean" if advantage > mean else "At/below mean") if ready else "Unknown",
            trend=trend, trend_ready=ready))
    for col in ["source_date", "prior_advantage", "prior_brier", "slope", "five_year_mean", "reference_events", "regime", "trend", "trend_ready"]:
        out[col] = [r[col] for r in records]
    return out


def apply_event_filters(events, historical_events, *, strategy="combined", config=ScalingConfig(),
                        return_col="return_pct", stake_col="stake_fraction"):
    """Apply flat gate and reduction-only scalers to pre-featured event rows.

    strategy: raw, flat_only, volatility_only, prior_brier_only, combined.
    historical_events must have date, pre-event trend/regime/prior_brier and
    ORIGINAL realized return_pct. Source history may be opening or closing;
    target rows may contain another market's return/stake. Flat prior events
    remain in denominator volatility but not its target or Brier regression.
    Insufficient scaler history leaves multiplier=1, with readiness flags;
    insufficient trend history skips the event (except raw strategy).

    Returns every target row, with keep_event, stake_multiplier and audit fields.
    Adds scaled_return_pct / scaled_stake_fraction only when corresponding input
    exists. Missing upcoming returns remain NaN. Flat skipped historical returns
    are zero. Total input event stake must be within [0,1]; no increases allowed.
    """
    if strategy not in {"raw", "flat_only", "volatility_only", "prior_brier_only", "combined"}:
        raise ValueError("Unknown strategy")
    out = _dated(events, unique=True)
    history = _dated(historical_events, unique=True).reset_index(drop=True).sort_values("date")
    required = {"trend", "regime", "prior_brier"}
    if not required.issubset(out) or not required.union({return_col}).issubset(history):
        raise ValueError("Calculate pre-event trends first; history also needs original returns")
    returns = pd.to_numeric(history[return_col], errors="raise")
    if not np.isfinite(returns).all() or returns.lt(-100).any():
        raise ValueError("Historical returns must be finite and at least -100 percent")
    history[return_col] = returns
    if not out.trend.isin(["Uptrend", "Downtrend", "Flat", "Unknown"]).all():
        raise ValueError("Unrecognized trend label")
    if stake_col in out:
        if not out[stake_col].between(0, 1).all():
            raise ValueError("Total event stake must be a fraction in [0,1]")
    rows = []
    for row in out.itertuples(index=False):
        prior = history.loc[history.date < row.date]
        nonflat = prior.trend.isin(["Uptrend", "Downtrend"])
        target_group = "Above mean" if config.volatility_target == "above_mean" else row.regime
        target = prior.loc[nonflat & prior.regime.eq(target_group)].tail(config.volatility_events)
        own = prior.loc[prior.regime.eq(row.regime)].tail(config.volatility_events)
        target_sd = target[return_col].std(ddof=1) if len(target)==config.volatility_events else np.nan
        own_sd = own[return_col].std(ddof=1) if len(own)==config.volatility_events else np.nan
        vol_ready = bool(np.isfinite([target_sd, own_sd]).all())
        vol = min(1., target_sd/own_sd) if vol_ready and own_sd > 1e-12 else 1.
        training = prior.loc[nonflat & np.isfinite(prior.prior_brier)]
        brier_ready = len(training)>=config.brier_min_events and np.isfinite(row.prior_brier)
        brier = 1.;baseline = prediction = np.nan
        if brier_ready:
            x = training.prior_brier.to_numpy(float);y = training[return_col].to_numpy(float)
            baseline = prediction = y.mean()
            if x.std() > 1e-12:
                z = (x-x.mean())/x.std()
                coef = max(0., np.dot(z, y-baseline)/(np.dot(z, z)+config.brier_ridge))
                prediction = baseline+coef*(row.prior_brier-x.mean())/x.std()
            brier = float(np.clip(prediction/baseline, 0, 1)) if baseline > 1e-12 else float(prediction > 0)
        keep = strategy == "raw" or row.trend in {"Uptrend", "Downtrend"}
        multiplier = {"raw": 1., "flat_only": 1., "volatility_only": vol,
                      "prior_brier_only": brier, "combined": vol*brier}[strategy] if keep else 0.
        rows.append(dict(keep_event=keep, volatility_multiplier=vol, brier_multiplier=brier,
            stake_multiplier=multiplier, target_volatility=target_sd, comparison_volatility=own_sd,
            target_count=len(target), comparison_count=len(own), volatility_ready=vol_ready,
            brier_count=len(training), brier_ready=bool(brier_ready), predicted_return_pct=prediction,
            baseline_return_pct=baseline))
    cols = ["keep_event", "volatility_multiplier", "brier_multiplier", "stake_multiplier",
            "target_volatility", "comparison_volatility", "target_count", "comparison_count",
            "volatility_ready", "brier_count", "brier_ready", "predicted_return_pct", "baseline_return_pct"]
    for col in cols:
        out[col] = [r[col] for r in rows]
    if return_col in out:
        out["scaled_return_pct"] = out[return_col]*out.stake_multiplier
    if stake_col in out:
        out["scaled_stake_fraction"] = out[stake_col]*out.stake_multiplier
    return out
