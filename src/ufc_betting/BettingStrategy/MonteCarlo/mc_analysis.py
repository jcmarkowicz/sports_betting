import numpy as np
import pandas as pd


def _sharpe(returns):
    """Unannualized arithmetic mean / sample SD, with zero risk-free return."""
    if len(returns) < 2:
        return np.nan
    sd = np.std(returns, ddof=1)
    return float(np.mean(returns) / sd) if sd > 0 else np.nan


def summarize_simulations(results, *, return_path_statistics=False, batch_size=1, event_count=None):
    """Summarize {label: simulate_result}, optionally returning per-path statistics.

    Drawdown is the decline from the running bankroll peak (initial balance
    included). Frozen losses below zero can produce drawdowns above 100%.
    Duration counts consecutive recorded settlements below the previous peak,
    including unrecovered drawdowns at the end of the path.

    Sharpe is mean(path-average event returns) / sample SD(path averages).
    No-bet and frozen post-ruin events contribute zero. For batched settlement,
    supply event_count: batch returns sum constituent event fractions, so their
    sum / event_count recovers each path's mean event return. Intrabatch balances
    are unavailable; ulcer and underwater statistics use recorded settlements.
    Ulcer index is RMS percentage drawdown, excluding the initial balance.
    Undefined Sharpe (fewer than two paths or zero SD) is NaN.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if event_count is not None and (isinstance(event_count, bool) or
            not isinstance(event_count, (int, np.integer)) or event_count < 0):
        raise ValueError('event_count must be a nonnegative integer')
    if batch_size > 1 and event_count is None:
        raise ValueError('event_count is required for batched settlement')
    if not results:
        raise ValueError('results must contain at least one simulation result')
    rows, path_frames = [], []
    for label, result in results.items():
        paths = np.asarray(result[2], dtype=float)
        if (paths.ndim != 2 or 0 in paths.shape or not np.isfinite(paths).all()
                or (paths[:, 0] <= 0).any()):
            raise ValueError('Bankroll paths must be nonempty, finite, and start positive')
        returns = 100 * (paths[:, -1] / paths[:, 0] - 1)
        peak = np.maximum.accumulate(paths, axis=1)
        drawdown = (peak - paths) / peak
        maximum_drawdown = 100 * drawdown.max(axis=1)
        current_duration = np.zeros(len(paths), dtype=int)
        longest_duration = np.zeros(len(paths), dtype=int)
        severe_episodes = np.zeros(len(paths), dtype=int)
        episode_counted = np.zeros(len(paths), dtype=bool)
        for step in range(1, paths.shape[1]):
            # Reset only at full recovery, not when crossing back above -50%.
            episode_counted &= drawdown[:, step] > 0
            newly_severe = (drawdown[:, step] >= .5) & ~episode_counted
            severe_episodes += newly_severe
            episode_counted |= newly_severe
            current_duration = np.where(paths[:, step] < peak[:, step], current_duration + 1, 0)
            longest_duration = np.maximum(longest_duration, current_duration)
        before, after = paths[:, :-1], paths[:, 1:]
        active = before > 0
        settlement_returns = np.full(before.shape, np.nan)
        np.divide(after - before, before, out=settlement_returns, where=active)
        n_settlements = paths.shape[1] - 1
        n_events = n_settlements if event_count is None else event_count
        if (n_events + batch_size - 1) // batch_size != n_settlements:
            raise ValueError('event_count and batch_size do not match bankroll history')
        mean_event_returns = (np.nan_to_num(settlement_returns, nan=0).sum(axis=1) / n_events
                              if n_events else np.full(len(paths), np.nan))
        underwater = (drawdown[:, 1:] > 0).sum(axis=1)
        underwater_pct = (100 * underwater / n_settlements if n_settlements
                           else np.full(len(paths), np.nan))
        ulcer = (100 * np.sqrt(np.mean(drawdown[:, 1:] ** 2, axis=1)) if n_settlements
                 else np.full(len(paths), np.nan))
        ruined = (paths <= 0).any(axis=1)
        rows.append({
            'Prices': label,
            'Simulations': len(paths),
            'Mean ending bankroll': np.mean(paths[:, -1]),
            'P05 ending bankroll': np.quantile(paths[:, -1], .05),
            'P25 ending bankroll': np.quantile(paths[:, -1], .25),
            'Median ending bankroll': np.median(paths[:, -1]),
            'P75 ending bankroll': np.quantile(paths[:, -1], .75),
            'Mean total return (%)': returns.mean(),
            'Median total return (%)': np.median(returns),
            'SD total return (%)': returns.std(ddof=1) if len(returns) > 1 else np.nan,
            'Profitable paths (%)': 100 * (returns > 0).mean(),
            'Mean max drawdown (%)': maximum_drawdown.mean(),
            'Median max drawdown (%)': np.median(maximum_drawdown),
            'P95 max drawdown (%)': np.quantile(maximum_drawdown, .95),
            'Worst max drawdown (%)': maximum_drawdown.max(),
            'Paths with drawdown >= 50% (%)': 100 * (maximum_drawdown >= 50).mean(),
            'Mean drawdown episodes >= 50%': severe_episodes.mean(),
            'Ruined paths (%)': 100 * ruined.mean(),
            'Mean longest drawdown (settlements)': longest_duration.mean(),
            'Sharpe (path-average event returns)': _sharpe(mean_event_returns),
            'Mean path-average event return (%)': 100 * mean_event_returns.mean(),
            'SD path-average event return (%)': 100 * mean_event_returns.std(ddof=1) if len(paths) > 1 else np.nan,
            'Mean ulcer index (%)': ulcer.mean(),
            'Mean time underwater (settlements)': underwater.mean(),
            'Mean time underwater (%)': underwater_pct.mean(),
        })
        path_frames.append(pd.DataFrame({
            'Prices': label, 'Simulation': np.arange(1, len(paths) + 1),
            'Starting bankroll': paths[:, 0], 'Ending bankroll': paths[:, -1],
            'Total profit': paths[:, -1] - paths[:, 0], 'Total return (%)': returns,
            'Max drawdown (%)': maximum_drawdown,
            'Drawdown episodes >= 50%': severe_episodes,
            'Longest drawdown (settlements)': longest_duration,
            'Average event return (%)': 100 * mean_event_returns,
            'Ulcer index (%)': ulcer,
            'Time underwater (settlements)': underwater,
            'Time underwater (%)': underwater_pct,
            'Ruined': ruined,
        }))
    summary = pd.DataFrame(rows)
    return (summary, pd.concat(path_frames, ignore_index=True)) if return_path_statistics else summary


def analyze_bankroll_targets(target_paths, targets, event_count, *, batch_size=1):
    """Return (summary, reach_by_path) for {label: bankroll_paths}; no file writes.

    Reaching a target means a recorded balance >= target, including event zero.
    Unreached paths have NaN First event count. Timing statistics use hits only.
    event_count includes the final partial settlement batch.
    """
    for name, value, minimum in [('event_count', event_count, 0), ('batch_size', batch_size, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    targets = list(targets)
    if not target_paths or not targets or not np.isfinite(targets).all() or len(set(targets)) != len(targets):
        raise ValueError('Supply paths and unique finite targets')
    steps = (event_count + batch_size - 1) // batch_size
    event_axis = np.minimum(np.arange(steps + 1) * batch_size, event_count)
    summary_rows, path_frames = [], []
    for label, paths in target_paths.items():
        for target in targets:
            first_events = bankroll_target_distribution(
                paths, target, event_counts=event_axis)['events_to_target'].to_numpy()
            hit = np.isfinite(first_events)
            observed = first_events[hit]
            quantiles = np.quantile(observed, [.05, .5, .95]) if len(observed) else [np.nan] * 3
            summary_rows.append({
                'Prices': label, 'Target ($)': target, 'Paths': len(hit),
                'Mean ending bankroll': np.asarray(paths)[:, -1].mean(),
                'Mean total return (%)': 100 * (np.asarray(paths)[:, -1] / np.asarray(paths)[:, 0] - 1).mean(),
                'Reached (%)': 100 * hit.mean(), 'Not reached (%)': 100 * (~hit).mean(),
                'Mean events (reached only)': observed.mean() if len(observed) else np.nan,
                'P05 events (reached only)': quantiles[0],
                'Median events (reached only)': quantiles[1],
                'P95 events (reached only)': quantiles[2],
            })
            path_frames.append(pd.DataFrame({
                'Prices': label, 'Target ($)': target,
                'Simulation': np.arange(1, len(hit) + 1),
                'Reached': hit, 'First event count': first_events,
            }))
    return pd.DataFrame(summary_rows), pd.concat(path_frames, ignore_index=True)


def bankroll_target_distribution(bankroll_paths, target_bankroll, *, event_counts=None):
    """First passage to >= target, including the initial balance at event zero.

    Pass simulate's bankroll paths (third return value). By default each step
    is one event; for batched settlement supply counts such as [0, 3, 6, 8].
    Hits are measured at recorded balances only. Unreached paths get NaN.
    Distribution probabilities use all paths; conditional probabilities use
    only paths that reach the target within the simulated horizon.
    """
    balances = np.asarray(bankroll_paths, dtype=float)
    if balances.ndim != 2 or 0 in balances.shape:
        raise ValueError("bankroll_paths must be a nonempty simulations-by-steps array")
    if not np.isfinite(balances).all() or not np.isfinite(target_bankroll):
        raise ValueError("Bankrolls and target must be finite")

    events = np.arange(balances.shape[1]) if event_counts is None else np.asarray(event_counts)
    if (events.shape != (balances.shape[1],) or not np.isfinite(events).all()
            or events[0] != 0 or (np.diff(events) <= 0).any()
            or (events != np.floor(events)).any()):
        raise ValueError("event_counts must start at zero and increase by whole events")

    hits = balances >= target_bankroll
    reached = hits.any(axis=1)
    first_hit = hits.argmax(axis=1)
    events_to_target = pd.Series(
        np.where(reached, events[first_hit], np.nan), name="events_to_target"
    )

    counts = np.bincount(first_hit[reached], minlength=len(events))
    distribution = pd.DataFrame({
        "events": events,
        "count": counts,
        "probability": counts / len(balances),
        "conditional_probability": counts / reached.sum() if reached.any() else np.nan,
    })
    distribution["cumulative_probability"] = distribution.probability.cumsum()

    return {
        "probability_reached": float(reached.mean()),
        "events_to_target": events_to_target,
        "distribution": distribution,
    }


def geometric_growth_by_path(results, event_count):
    """Terminal geometric return per scheduled event; negative wealth is undefined."""
    if isinstance(event_count, bool) or not isinstance(event_count, (int, np.integer)) or event_count < 1:
        raise ValueError('Geometric growth requires a positive event count')
    frames = []
    for label, result in results.items():
        paths = np.asarray(result[2], dtype=float)
        if (paths.ndim != 2 or paths.shape[1] < 2 or not np.isfinite(paths).all()
                or (paths[:, 0] <= 0).any()):
            raise ValueError('Need finite bankroll paths with positive starting wealth')
        final = paths[:, -1]
        growth = np.full(len(paths), np.nan)
        positive = final > 0
        growth[positive] = 100 * np.expm1((np.log(final[positive]) - np.log(paths[positive, 0])) / event_count)
        growth[final == 0] = -100.
        frames.append(pd.DataFrame({'Prices': label, 'Simulation': np.arange(1, len(paths) + 1),
                                   'Geometric growth per event (%)': growth,
                                   'Negative final bankroll': final < 0}))
    return pd.concat(frames, ignore_index=True)


def mean_event_exposure(results):
    """Average event bankroll fractions within each path; one observation per path.

    No-bet/post-ruin observations count as zero. Old archives cannot supply
    exposure from P/L alone; return missing labels separately.
    """
    frames, missing = [], []
    for label, result in results.items():
        metadata = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
        if 'event_exposure' not in metadata:
            missing.append(label)
            continue
        exposure = np.asarray(metadata['event_exposure'], dtype=float)
        if (exposure.ndim != 2 or exposure.shape[0] != len(result[2]) or not exposure.shape[1]
                or not np.isfinite(exposure).all() or (exposure < 0).any()):
            raise ValueError('Need finite nonnegative exposure with equal event counts across paths')
        frames.append(pd.DataFrame({'Prices': label, 'Simulation': np.arange(1, exposure.shape[0] + 1),
                                   'Mean bankroll wagered (%)': 100 * exposure.mean(axis=1),
                                   'Events': exposure.shape[1]}))
    columns = ['Prices', 'Simulation', 'Mean bankroll wagered (%)', 'Events']
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)), missing
