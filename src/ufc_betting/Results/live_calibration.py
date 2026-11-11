"""Calibration of archived live probabilities; never refit or regenerate picks."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def calibration_rows(history, market):
    """Return paired model/Vegas probabilities on settled, valid fights."""
    if market not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown odds market')
    columns = ['winner_bool', f'pred_winner_{market}', f'choice_proba_{market}',
               f'{market}_red', f'{market}_blue', f'net_stake_{market}']
    missing = set(columns) - set(history.columns)
    if missing:
        raise ValueError(f'Missing calibration columns: {sorted(missing)}')
    values = history[columns].apply(pd.to_numeric, errors='coerce')
    winner, side, probability, red, blue, stake = (values[c] for c in columns)
    valid = (winner.isin([0, 1]) & side.isin([0, 1]) & probability.between(0, 1)
             & np.isfinite(red) & np.isfinite(blue)
             & red.abs().ge(100) & blue.abs().ge(100))
    values = values.loc[valid]
    def implied(american):
        return pd.Series(np.where(american > 0, 100 / (100 + american.abs()),
                                  american.abs() / (100 + american.abs())), index=american.index)
    red_implied, blue_implied = implied(red.loc[valid]), implied(blue.loc[valid])
    return pd.DataFrame({
        'outcome': winner.loc[valid],
        'model': probability.loc[valid].where(side.loc[valid].eq(1), 1 - probability.loc[valid]),
        'vegas': red_implied / (red_implied + blue_implied),
        'selected': stake.loc[valid].gt(0),
    })


def plot_live_calibration(history, market, *, include_selected=True):
    """Two reliability panels, comparing identical fights within each panel."""
    rows = calibration_rows(history, market)
    panels = [('All predictions', rows)]
    if include_selected:
        panels.insert(0, ('Selected moneyline picks', rows.loc[rows.selected]))
    fig, axes = plt.subplots(1, len(panels), figsize=(11 if include_selected else 8, 4.5), constrained_layout=True)
    for ax, (title, data) in zip(np.atleast_1d(axes), panels):
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        for source, label, color in [('model', 'Model', '#2563eb'), ('vegas', 'Vegas (no vig)', '#ea580c')]:
            if not data.empty:
                observed, predicted = calibration_curve(data.outcome, data[source], n_bins=10, strategy='uniform')
                brier = np.mean((data[source] - data.outcome) ** 2)
                ax.plot(predicted, observed, 'o-', color=color, label=f'{label} · Brier {brier:.3f}')
        if data.empty:
            ax.text(.5, .45, 'No eligible settled fights', ha='center', transform=ax.transAxes)
        ax.set(title=f'{title} (n={len(data):,})', xlabel='Predicted red-win probability',
               ylabel='Observed red-win frequency', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(loc='upper left', fontsize=8)
    fig.suptitle(f'{market.capitalize()} calibration')
    return fig
