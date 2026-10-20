"""Saved elastic-net GLM probabilities and training-event bootstrap uncertainty."""

import numpy as np
import pandas as pd

from ufc_betting.Models.LogisticRegression.regularized import (
    fit_regularized_probability, predict_regularized_probability,
)
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.utils import save_results


def fit_glm_bundle(X, y, dates, *, alpha=0.1, L1_wt=0.5, n_bootstrap=50, seed=42):
    """Fit raw features and bootstrap whole training events at fixed parameters.

    Bootstrap SD measures fitted probability variability, conditional on the
    selected features/alpha/L1_wt. It is not an outcome SD or a calibrated CI.
    Each bootstrap refits its own scaler/encoder. Failed fits are surfaced.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError('n_bootstrap must be an integer >= 2')
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y)
    dates = pd.to_datetime(pd.Series(np.asarray(dates)), errors='raise').dt.normalize()
    if len({len(X), len(y), len(dates)}) != 1 or not len(X):
        raise ValueError('X, y, and dates must have equal nonzero lengths')
    if dates.isna().any() or X.isna().any().any() or not np.isin(y, [0, 1]).all():
        raise ValueError('Complete dates/features and binary labels are required')
    categorical = list(X.select_dtypes(include=['category', 'object', 'string', 'bool']).columns)
    numeric = [name for name in X if name not in categorical]

    def fit(indices):
        if len(np.unique(y[indices])) != 2:
            raise ValueError('Each GLM fit needs both outcome classes')
        raw = X.iloc[indices].reset_index(drop=True)
        scaled = TrainTestBuilder.scale_encode(raw, raw, categorical, numeric)
        model = fit_regularized_probability(scaled['X_train'], y[indices],
                                            model_type='glm', alpha=alpha, L1_wt=L1_wt)
        return dict(model=model, scaler=scaled['scaler'], encoder=scaled['encoder'])

    bundle = fit(np.arange(len(X)))
    events = dates.unique()
    if len(events) < 2:
        raise ValueError('Bootstrap uncertainty requires at least two training events')
    groups = [np.flatnonzero(dates.eq(event)) for event in events]
    rng = np.random.default_rng(seed)
    bundle['bootstrap_models'] = [
        fit(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
        for _ in range(n_bootstrap)
    ]
    bundle.update(features=list(X.columns), alpha=alpha, L1_wt=L1_wt,
                  training_last_date=dates.max(), n_bootstrap=n_bootstrap,
                  uncertainty_method='training_event_bootstrap', seed=seed)
    return bundle


def predict_glm_bundle(rows, bundle, *, odds_type='open', fp=None):
    """Apply an opening GLM with opening or substituted closing odds.

    Retains original prices and fight metadata for simulation. Never refits.
    """
    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('odds_type must be open, close1, or close2')

    def predict(fitted):
        design = TrainTestBuilder.prepare_inference_data(
            rows, fitted['scaler'], fitted['encoder'],
            switch_open_odds=odds_type != 'open', odds_type=odds_type)
        return predict_regularized_probability(fitted['model'], design)

    p = predict(bundle)
    bootstrap = np.stack([predict(fitted) for fitted in bundle['bootstrap_models']])
    output = rows.copy()
    output['proba_se'] = bootstrap.std(axis=0, ddof=1)
    output['prediction_se'] = output['proba_se']
    output['risk_red_proba'] = p
    output['probability_model'] = 'glm_elastic_net'
    output['uncertainty_method'] = bundle['uncertainty_method']
    output['model_alpha'] = bundle['alpha']
    output['model_L1_wt'] = bundle['L1_wt']
    return save_results(output, p, fp)
