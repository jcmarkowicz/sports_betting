"""Read notebook inference reports without loading held-out test data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve

REPORTS = {
    'Training — train fit': ('train', 'train'),
    'Validation — train fit': ('train', 'val'),
    'Training + validation — combined fit': ('trainval', 'train'),
}


def report_paths(root, mode, split):
    if (mode, split) not in REPORTS.values():
        raise ValueError('Only training and validation inference reports are supported')
    root = Path(root)
    return root / f'{mode}_{split}_inference_metrics.csv'


def load_report(root, mode, split):
    metrics = pd.read_csv(report_paths(root, mode, split))
    curves = []
    for row in metrics.itertuples(index=False):
        if row.model not in ('open', 'close1', 'close2') or row.odds not in ('open', 'close1', 'close2'):
            raise ValueError('Unknown model or odds in inference report')
        path = Path(root) / 'report_predictions' / mode / row.model / 'inference' / f'{split}_with_{row.odds}.csv'
        frame = pd.read_csv(path, usecols=['winner', 'proba_red', 'probability_model'])
        if not frame.probability_model.eq('inference').all():
            raise ValueError(f'Expected inference probabilities: {path.name}')
        frame = frame.loc[frame.winner.isin([0, 1])]
        if not frame.proba_red.between(0, 1).all():
            raise ValueError('Invalid inference probabilities')
        if len(frame):
            actual, predicted = calibration_curve(frame.winner, frame.proba_red, n_bins=10, strategy='uniform')
            curves.append((f'{row.model} → {row.odds}', predicted, actual, len(frame)))
    return metrics.set_index(['model', 'odds']), curves


def calibration_figure(curves, title):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, predicted, actual, count in curves:
        ax.plot(predicted, actual, 'o-', label=f'{label} (n={count:,})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.set(xlabel='Mean predicted red-win probability', ylabel='Observed red-win rate',
           title=title, xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    return fig


def inference_coefficients(path):
    model = sm.load(path)
    ci = np.asarray(model.conf_int())
    table = pd.DataFrame({'feature': model.model.exog_names,
        'coefficient': np.asarray(model.params), 'std_error': np.asarray(model.bse),
        'z': np.asarray(model.tvalues), 'p_value': np.asarray(model.pvalues),
        'ci_lower': ci[:, 0], 'ci_upper': ci[:, 1]})
    table['significance'] = np.select(
        [table.p_value < .001, table.p_value < .01, table.p_value < .05],
        ['***', '**', '*'], default='')
    return table
