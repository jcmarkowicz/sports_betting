"""Chronological selected-bet Brier shortfall for parameter-search scoring."""
import numpy as np
import pandas as pd


def rolling_brier_shortfall(dates, gaps, *, window_bets=100, target=.01):
    """Mean positive target shortfall over complete trailing whole-event windows.

    gap = Vegas Brier - model Brier, so positive favors the model. Each event
    endpoint receives equal weight; fights within a window receive equal weight.
    Early incomplete windows are excluded. No bootstrap/synthetic dates are used.
    """
    dates = pd.to_datetime(np.asarray(dates), errors='raise')
    gaps = np.asarray(gaps, dtype=float)
    if dates.isna().any() or not np.isfinite(gaps).all():
        raise ValueError('Brier diagnostics require finite gaps and valid dates')
    if isinstance(window_bets, bool) or not isinstance(window_bets, (int, np.integer)) or window_bets < 2:
        raise ValueError('window_bets must be an integer >= 2')
    if not np.isfinite(target) or target < 0:
        raise ValueError('target must be finite and nonnegative')
    events = pd.DataFrame({'date': dates, 'gap': gaps}).groupby('date', sort=True).agg(
        total=('gap', 'sum'), count=('gap', 'size'))
    if len(gaps) < window_bets:
        raise ValueError(f'Brier penalty requires at least {window_bets} selected settled moneylines')
    counts = events['count'].to_numpy()
    totals = events.total.to_numpy()
    start = 0
    count = 0
    total = 0.
    rolling = []
    for stop in range(len(events)):
        count += counts[stop]
        total += totals[stop]
        while start < stop and count - counts[start] >= window_bets:
            count -= counts[start]
            total -= totals[start]
            start += 1
        if count >= window_bets:
            rolling.append(total / count)
    rolling = np.asarray(rolling)
    return dict(brier_mean_gap=float(rolling.mean()),
                brier_mean_shortfall=float(np.maximum(target - rolling, 0).mean()),
                brier_window_count=len(rolling), brier_selected_count=len(gaps))
