"""Observed performance of placed choices versus no-vig Vegas probabilities."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def rolling_edge_summary(rows, *, window_bets=100, bootstrap_samples=2000, seed=0):
    """Trailing whole-event windows; expand until window_bets bets are available.

    Resample event dates with replacement, retaining every bet in each sampled
    event. Percentile intervals are pointwise 95% intervals, not simultaneous
    bands. A single-event window has no uncertainty estimate.
    """
    if not isinstance(window_bets, int) or window_bets < 1:
        raise ValueError('window_bets must be a positive integer')
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 2:
        raise ValueError('bootstrap_samples must be at least 2')
    data = rows.assign(predicted=rows.model - rows.vegas,
                       realized=rows.outcome - rows.vegas,
                       gap=rows.outcome - rows.model)
    events = data.groupby('date', sort=True).agg(
        predicted=('predicted', 'sum'), realized=('realized', 'sum'),
        gap=('gap', 'sum'), count=('outcome', 'size'))
    rng = np.random.default_rng(seed)
    records = []
    for stop in range(1, len(events) + 1):
        start = stop - 1
        count = events['count'].iloc[start]
        while start > 0 and count < window_bets:
            start -= 1
            count += events['count'].iloc[start]
        window = events.iloc[start:stop]
        totals = window.sum()
        record = dict(date=events.index[stop - 1], count=int(count), events=len(window))
        for name in ('predicted', 'realized', 'gap'):
            record[name] = 100 * totals[name] / count
        for name in ('realized', 'gap'):
            record[name + '_low'] = record[name + '_high'] = np.nan
        if len(window) >= 2:
            sampled = rng.integers(0, len(window), size=(bootstrap_samples, len(window)))
            denominator = window['count'].to_numpy()[sampled].sum(axis=1)
            for name in ('realized', 'gap'):
                draws = 100 * window[name].to_numpy()[sampled].sum(axis=1) / denominator
                record[name + '_low'], record[name + '_high'] = np.quantile(draws, [.025, .975])
        records.append(record)
    return pd.DataFrame(records)


def brier_decomposition(outcome, probability, n_bins=10):
    """Uniform-bin Murphy terms plus residual reconciling the raw Brier score."""
    data = pd.DataFrame({'y': np.asarray(outcome), 'p': np.asarray(probability)})
    if data.empty:
        return dict(n=0, brier=np.nan, reliability=np.nan, resolution=np.nan,
                    uncertainty=np.nan, residual=np.nan)
    data['bin'] = pd.cut(data.p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    groups = data.groupby('bin', observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    weights = groups.n / len(data)
    base = data.y.mean()
    reliability = (weights * (groups.p - groups.y) ** 2).sum()
    resolution = (weights * (groups.y - base) ** 2).sum()
    uncertainty = base * (1 - base)
    brier = ((data.p - data.y) ** 2).mean()
    return dict(n=len(data), brier=brier, reliability=reliability, resolution=resolution,
                uncertainty=uncertainty, residual=brier - (reliability - resolution + uncertainty))


def rolling_brier_summary(rows, window_bets=100):
    """Equal-fight raw scores in trailing whole-event windows, expanding initially."""
    data = rows.assign(model_score=(rows.model - rows.outcome) ** 2,
                       vegas_score=(rows.vegas - rows.outcome) ** 2)
    events = data.groupby('date', sort=True).agg(model=('model_score', 'sum'),
                vegas=('vegas_score', 'sum'), count=('outcome', 'size'))
    records = []
    for stop in range(1, len(events) + 1):
        start = stop - 1
        count = events['count'].iloc[start]
        while start > 0 and count < window_bets:
            start -= 1
            count += events['count'].iloc[start]
        totals = events.iloc[start:stop].sum()
        records.append(dict(date=events.index[stop - 1], count=count,
                            model=totals.model / count, vegas=totals.vegas / count))
    return pd.DataFrame(records)


def plot_comparison_calibration(moneylines, *, odds_type, n_bins=10, show=True,
                                window_bets=100, bootstrap_samples=2000, period_label="Selected period"):
    """Choice reliability plus rolling edge and calibration gap (percentage points).

    Only settled, positive-stake moneylines before ruin are included. Windows
    retain whole events, covering at least window_bets bets where available;
    earlier windows expand. Bets receive equal weight. Vegas is no-vig.
    All-fight Brier scores use red outcomes once per fight (equivalent to
    scoring both complementary sides). Decomposition includes a binning residual.
    Returns (figure, one-dimensional array of five axes).
    """
    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown odds market')
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError('n_bins must be an integer of at least 2')
    if 'choice_proba' not in moneylines and {'proba_red', 'proba_blue', 'pred_winner'} <= set(moneylines):
        moneylines = moneylines.copy()
        moneylines['choice_proba'] = moneylines.proba_red.where(
            moneylines.pred_winner.eq(1), moneylines.proba_blue)
    red, blue = f'dec_{odds_type}_red', f'dec_{odds_type}_blue'
    if not {red, blue} <= set(moneylines):
        red, blue = 'real_odds_red', 'real_odds_blue'
    columns = ['winner', 'pred_winner', 'choice_proba', 'choice_fstar', red, blue]
    missing = set(columns + ['date']) - set(moneylines)
    if missing:
        raise ValueError(f'Missing calibration columns: {sorted(missing)}')
    frame = moneylines[columns].apply(pd.to_numeric, errors='coerce')
    valid = (frame.winner.isin([0, 1]) & frame.pred_winner.isin([0, 1])
             & frame.choice_proba.between(0, 1)
             & np.isfinite(frame[red]) & np.isfinite(frame[blue])
             & frame[red].gt(1) & frame[blue].gt(1))
    selected = frame.choice_fstar.gt(0) & np.isfinite(frame.choice_fstar)
    balances = ['bankroll_postevent', 'event_payout_money_line', 'parlay_net']
    if set(balances) <= set(moneylines):
        post, singles, parlays = (pd.to_numeric(moneylines[c], errors='coerce') for c in balances)
        selected &= (post - singles - parlays).gt(0)
    vegas_red = (1 / frame[red]) / (1 / frame[red] + 1 / frame[blue])
    all_rows = pd.DataFrame({
        'date': pd.to_datetime(moneylines.loc[valid, 'date'], errors='raise').dt.normalize(),
        'vegas': vegas_red.loc[valid],
        'model': frame.choice_proba.where(frame.pred_winner.eq(1), 1 - frame.choice_proba).loc[valid],
        'outcome': frame.winner.loc[valid],
    })
    if all_rows.date.isna().any():
        raise ValueError('Eligible fights must have valid dates')
    mask = valid & selected
    rows = pd.DataFrame({
        'date': pd.to_datetime(moneylines.loc[mask, 'date'], errors='raise').dt.normalize(),
        'vegas': vegas_red.where(frame.pred_winner.eq(1), 1 - vegas_red).loc[mask],
        'model': frame.choice_proba.loc[mask],
        'outcome': frame.winner.eq(frame.pred_winner).astype(int).loc[mask],
    })
    if rows.date.isna().any():
        raise ValueError('Placed choices must have valid dates')
    summary = rolling_edge_summary(rows, window_bets=window_bets, bootstrap_samples=bootstrap_samples)
    fig, axes = plt.subplots(5, 1, figsize=(12, 23), constrained_layout=True)
    reliability, timeline, gap_ax, brier_ax, decomposition_ax = axes
    reliability.plot([0, 1], [0, 1], '--', color='gray', label='Win rate equals Vegas probability')
    timeline.axhline(0, linestyle='--', color='gray', label='No realized edge')
    gap_ax.axhline(0, linestyle='--', color='gray', label='Calibrated on average')
    if not rows.empty:
        rows['bin'] = pd.cut(rows.vegas, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
        grouped = rows.groupby('bin', observed=True).agg(
            vegas=('vegas', 'mean'), actual=('outcome', 'mean'), count=('outcome', 'size'))
        reliability.plot(grouped.vegas, grouped.actual, 'o-', label='Actual chosen-fighter win rate')
        for x, y, count in zip(grouped.vegas, grouped.actual, grouped['count']):
            reliability.annotate(f'n={count}', (x, y), xytext=(0, 7),
                                 textcoords='offset points', ha='center', fontsize=8)
        timeline.plot(summary.date, summary.predicted, 'o--', label='Predicted edge: model âˆ’ Vegas')
        timeline.plot(summary.date, summary.realized, 'o-', label='Realized edge: actual âˆ’ Vegas')
        gap_ax.plot(summary.date, summary.gap, 'o-', label='Calibration gap: actual âˆ’ model')
        for ax, metric in ((timeline, 'realized'), (gap_ax, 'gap')):
            ax.fill_between(summary.date, summary[metric + '_low'], summary[metric + '_high'],
                            alpha=.2, label='95% event-bootstrap interval')
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        timeline.text(.02, .02, f'Latest window: {summary["count"].iloc[-1]} bets / '
                      f'{summary.events.iloc[-1]} events', transform=timeline.transAxes, fontsize=9)
    else:
        for ax in axes[:3]:
            ax.text(.5, .4, 'No eligible settled bets', ha='center', transform=ax.transAxes)
    reliability.set(title=f'Actual wins versus Vegas (n={len(rows):,})',
                    xlabel='Vegas implied chosen-fighter win probability (no vig)',
                    ylabel='Actual chosen-fighter win rate', xlim=(0, 1), ylim=(-.03, 1.08))
    window_label = f'Trailing {window_bets} bets, whole events; expanding initially'
    timeline.set(title=f'Predicted versus realized edge (n={len(rows):,})\n{window_label}',
                 xlabel='Event date', ylabel='Edge (percentage points)')
    gap_ax.set(title=f'Calibration gap (n={len(rows):,})', xlabel='Event date',
               ylabel='Realized âˆ’ predicted edge (percentage points)')
    decomposition = []
    for cohort, cohort_rows, style in [('Choices', rows, '-'), ('All fighters', all_rows, '--')]:
        rolling = rolling_brier_summary(cohort_rows, window_bets)
        for source, color in [('model', 'tab:blue'), ('vegas', 'tab:orange')]:
            if not rolling.empty:
                brier_ax.plot(rolling.date, rolling[source], linestyle=style, color=color,
                              label=f'{cohort} · {source.capitalize()}')
            decomposition.append({'cohort': cohort, 'source': source,
                                  **brier_decomposition(cohort_rows.outcome, cohort_rows[source], n_bins)})
    brier_ax.set(title=f'Rolling Brier score — lower is better\n{window_label} per cohort',
                 xlabel='Event date', ylabel='Mean squared probability error')
    locator = mdates.AutoDateLocator()
    brier_ax.xaxis.set_major_locator(locator)
    brier_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if all_rows.empty:
        brier_ax.text(.5, .4, 'No eligible settled fights', ha='center', transform=brier_ax.transAxes)
    decomposition_ax.axis('off')
    dates = (f'{all_rows.date.min():%Y-%m-%d} to {all_rows.date.max():%Y-%m-%d}'
             if not all_rows.empty else 'No eligible fights')
    decomposition_ax.set_title(f'{period_label}: full-period Brier decomposition\n{dates}')
    fields = ['brier', 'reliability', 'resolution', 'uncertainty', 'residual']
    cells = [[d['cohort'], d['source'].capitalize(), str(d['n'])] +
             [f'{d[k]:.4f}' if np.isfinite(d[k]) else '—' for k in fields] for d in decomposition]
    table = decomposition_ax.table(cellText=cells,
        colLabels=['Cohort', 'Source', 'N', 'Brier ↓', 'Reliability ↓', 'Resolution ↑', 'Uncertainty', 'Residual'],
        cellLoc='center', loc='center', colWidths=[.16, .1, .07, .11, .14, .14, .14, .13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    decomposition_ax.text(.5, .12,
        f'Brier = reliability − resolution + uncertainty + binning residual ({n_bins} uniform bins).\n'
        'All fighters: one red-side outcome per fight; choices: placed fighter only. Vegas: no vig.',
        ha='center', transform=decomposition_ax.transAxes, fontsize=10)
    for ax in axes[:4]:
        ax.grid(alpha=.2)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle(f'{odds_type}: {period_label} probability diagnostics')
    if show:
        plt.show()
    return fig, axes
