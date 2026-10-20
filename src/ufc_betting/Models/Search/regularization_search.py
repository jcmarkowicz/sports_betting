"""Optuna Brier tuning with event-grouped expanding-window validation.

Based on RiskManagement/delayed_walk_forward_logit.py. Preprocessing is fit
only on each training fold. The file runner tunes within the supplied dataset;
the optional walk_forward_regularization helper adds monthly outer evaluation. Input features must already be available at prediction time.
"""

import json
from pathlib import Path
import sys

# Allow running this file directly from an editor or terminal.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ufc_betting.Models.LogisticRegression.regularized import (
    fit_regularized_probability, predict_regularized_probability, BuilderPreprocessor,
)


def chronological_folds(dates, n_splits=5, gap=0):
    """Split sorted unique event dates, then expand to positional fight indices.

    gap is measured in event dates, not fights or calendar days.
    """
    dates = pd.to_datetime(pd.Series(dates), errors='raise').dt.normalize().reset_index(drop=True)
    if dates.isna().any():
        raise ValueError('Event dates cannot be missing')
    events = np.sort(dates.unique())
    for train, valid in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(events):
        yield (np.flatnonzero(dates.isin(events[train])),
               np.flatnonzero(dates.isin(events[valid])))


def _preprocessor(X):
    return BuilderPreprocessor()


def _inputs(X, y, dates):
    X = X.copy().reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(np.asarray(dates)), errors='raise').dt.normalize()
    if not len(X) or not X.shape[1] or len({len(X), len(y), len(dates)}) != 1:
        raise ValueError('Nonempty X, y, and dates must have equal lengths')
    if dates.isna().any() or X.isna().any().any() or not y.isin([0, 1]).all():
        raise ValueError('Complete features, valid dates, and binary outcomes are required')
    numeric = X.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Numeric features must be finite')
    return X, y, dates


def search_regularization(X, y, dates, *, model_type='logit', n_trials=50,
                          n_splits=5, gap=0, alpha_bounds=None,
                          l1_wt_bounds=(0.0, 1.0), seed=42):
    """Minimize equal-weight mean fold Brier, then refit on supplied history.

    Returns study, model, preprocessor, and fold audit. X must be unscaled;
    X/y/dates align positionally. This CV score is a tuning score, not an
    independent test score. Use walk_forward_regularization for outer testing.
    """
    if model_type not in ('logit', 'glm'):
        raise ValueError("model_type must be 'logit' or 'glm'")
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    low, high = alpha_bounds if alpha_bounds is not None else (
        (0.2, 7.0) if model_type == 'logit' else (1e-5, 1.0))
    if not np.isfinite([low, high]).all() or not 0 < low <= high:
        raise ValueError('alpha_bounds must be finite, positive, and ordered')
    l1_low, l1_high = l1_wt_bounds
    if not 0 <= l1_low <= l1_high <= 1:
        raise ValueError('l1_wt_bounds must be ordered within [0, 1]')
    X, y, dates = _inputs(X, y, dates)
    folds, audit = [], []
    for train, valid in chronological_folds(dates, n_splits, gap):
        if y.iloc[train].nunique() != 2:
            raise ValueError('Each training fold must contain both outcome classes')
        preprocessor = _preprocessor(X.iloc[train])
        folds.append((preprocessor.fit_transform(X.iloc[train]), y.iloc[train],
                      preprocessor.transform(X.iloc[valid]), y.iloc[valid]))
        audit.append(dict(train_end=str(dates.iloc[train].max().date()),
                          validation_start=str(dates.iloc[valid].min().date()),
                          validation_end=str(dates.iloc[valid].max().date()),
                          train_rows=len(train), validation_rows=len(valid)))
    study = optuna.create_study(direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=seed))
    study.set_user_attr('folds', audit)
    study.set_user_attr('model_type', model_type)
    study.set_user_attr('objective', 'mean_fold_brier')

    def objective(trial):
        params = {'alpha': trial.suggest_float('alpha', low, high, log=True)}
        if model_type == 'glm':
            params['L1_wt'] = trial.suggest_float('L1_wt', l1_low, l1_high)
        scores = []
        for train_X, train_y, valid_X, valid_y in folds:
            try:
                model = fit_regularized_probability(train_X, train_y, model_type=model_type, **params)
                scores.append(float(brier_score_loss(
                    valid_y, predict_regularized_probability(model, valid_X))))
            except (ValueError, np.linalg.LinAlgError, ConvergenceWarning) as exc:
                raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr('fold_brier', scores)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError('No trial completed; inspect fold sizes, features, and convergence')
    preprocessor = _preprocessor(X)
    model = fit_regularized_probability(preprocessor.fit_transform(X), y,
                                        model_type=model_type, **study.best_params)
    return dict(study=study, model=model, preprocessor=preprocessor,
                folds=pd.DataFrame(audit))


def walk_forward_regularization(builder, *, n_trials=50, n_splits=5, gap=0,
                                logit_alpha_bounds=(0.2, 7.0),
                                glm_alpha_bounds=(1e-5, 1.0),
                                l1_wt_bounds=(0.0, 1.0), seed=42):
    """Tune both models on past events, refit, and predict each holdout month.

    Uses builder.all_processed_df, selected_feats, target_col, date_col, and
    start_date, as in RiskManagement/logit_walk_forward.py. Returns long-form
    predictions, monthly_results, summary, and studies keyed by (model, month).
    Earlier holdouts become eligible training history in subsequent months.
    """
    df = builder.all_processed_df.copy().reset_index(drop=True)
    X, y, dates = _inputs(df[builder.selected_feats], df[builder.target_col], df[builder.date_col])
    if builder.target_col in X or builder.date_col in X:
        raise ValueError('Target and date columns cannot be model features')
    start = pd.Timestamp(builder.start_date).normalize()
    months = sorted(dates.loc[dates >= start].dt.to_period('M').unique())
    if not months:
        raise ValueError('No holdout events on or after builder.start_date')
    predictions, results, studies = [], [], {}
    for month in months:
        cutoff = max(start, month.start_time)
        train = np.flatnonzero(dates < cutoff)
        test = np.flatnonzero((dates >= cutoff) & (dates < (month + 1).start_time))
        for model_type, bounds in [('logit', logit_alpha_bounds), ('glm', glm_alpha_bounds)]:
            bundle = search_regularization(
                X.iloc[train], y.iloc[train], dates.iloc[train], model_type=model_type,
                n_trials=n_trials, n_splits=n_splits, gap=gap, alpha_bounds=bounds,
                l1_wt_bounds=l1_wt_bounds, seed=seed)
            study = bundle['study']
            p = predict_regularized_probability(bundle['model'],
                                                bundle['preprocessor'].transform(X.iloc[test]))
            part = df.iloc[test].copy()
            part['source_index'] = test
            part['model_type'] = model_type
            part['month'] = str(month)
            part['training_cutoff'] = cutoff
            part['proba_red'] = p
            part['proba_blue'] = 1 - p
            for name, value in study.best_params.items():
                part[f'best_{name}'] = value
            predictions.append(part)
            results.append(dict(model_type=model_type, month=str(month),
                                training_cutoff=cutoff, train_rows=len(train), test_rows=len(test),
                                cv_brier=study.best_value, holdout_brier=brier_score_loss(y.iloc[test], p),
                                **study.best_params))
            studies[(model_type, str(month))] = study
    predictions = pd.concat(predictions, ignore_index=True)
    summary = pd.DataFrame([
        dict(model_type=name, holdout_brier=brier_score_loss(part[builder.target_col], part.proba_red),
             holdout_rows=len(part)) for name, part in predictions.groupby('model_type')])
    return dict(predictions=predictions, monthly_results=pd.DataFrame(results),
                summary=summary, studies=studies)


def run_from_csv(
    input_path, output_dir, features, *, target_col='winner', date_col='date',
    categorical_cols=('math_red', 'math_blue', 'elo_pred', 'womens_fight'),
    n_trials=50, n_splits=5, gap=0, seed=42,
    logit_alpha_bounds=(0.2, 7.0), glm_alpha_bounds=(1e-5, 1.0),
    l1_wt_bounds=(0.0, 1.0),
):
    """Tune both models directly on an unscaled TrainTestBuilder split CSV.

    There is no separate outer holdout. Each trial fits on earlier events and
    scores later events; the objective is mean fold Brier. Selected models are
    refit on the complete supplied dataset for future inference.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError('Supply a nonempty list of distinct feature names')
    if target_col in features or date_col in features:
        raise ValueError('Target and date columns cannot be model features')

    # Split exports already contain the chosen rows. Do not filter or split again.
    data = pd.read_csv(input_path)
    required = set(features + [target_col, date_col])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    data[date_col] = pd.to_datetime(data[date_col], errors='raise').dt.normalize()
    data = data.sort_values(date_col, kind='stable').reset_index(drop=True)
    # CSV does not preserve category dtypes used by TrainTestBuilder.
    for name in set(categorical_cols) & set(features):
        data[name] = data[name].astype('category')
    X, y, dates = _inputs(data[features], data[target_col], data[date_col])
    bundles, summaries = {}, []
    for model_type, bounds in [('logit', logit_alpha_bounds), ('glm', glm_alpha_bounds)]:
        bundle = search_regularization(
            X, y, dates, model_type=model_type, n_trials=n_trials,
            n_splits=n_splits, gap=gap, seed=seed,
            alpha_bounds=bounds, l1_wt_bounds=l1_wt_bounds,
        )
        bundles[model_type] = bundle
        study = bundle['study']
        summaries.append(dict(model_type=model_type, cv_brier=study.best_value,
                              **study.best_params))

    output_dir.mkdir(parents=True, exist_ok=True)
    best_params = {}
    for model_type, bundle in bundles.items():
        best_params[model_type] = bundle['study'].best_params.copy()
        params_path = output_dir / f'{model_type}_best_params.json'
        params_path.write_text(
            json.dumps(best_params[model_type], indent=2) + '\n', encoding='utf-8')
    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))
    print(f'Results saved to {output_dir.resolve()}')
    return dict(best_params=best_params, summary=summary, bundles=bundles)


if __name__ == '__main__':
    from ufc_betting.config import config

    # EDIT THESE PATHS. Walk-forward tuning uses only this supplied dataset.
    input_path = Path(r"C:/Users/jcmar/my_files/SportsBetting/Data/train_val_test/splits/train.csv")
    output_dir = Path(r"C:/Users/jcmar/my_files/SportsBetting/Data/search_results/regularization")

    # Replace with your own list, e.g. ['proba_fair_open_diff', 'reach_diff'].
    features = [
        'proba_fair_open_diff', 'reach_diff', 
        
        'sub_att_pm_red', 'sub_att_pm_blue',

        'td_attempted_pm_diff', 'td_defended_pm_diff',
        
        'adjusted_td_diff', 'adjusted_control_diff', 'adjusted_sig_str_diff',

        'sig_str_absorbed_pm_diff', 'kd_total_diff',

        'sig_str_accuracy_pct_diff',

        'sig_str_defense_pct_diff',
        
        'win_pct_red', 'win_pct_blue',
        'win_streak_diff', 'lose_streak_diff',
        'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue', 'opp_age_red', 'opp_age_blue'
    ]


    results = run_from_csv(
        input_path=input_path,
        output_dir=output_dir,
        features=features,
        target_col='winner',
        date_col='date',
        n_trials=50,                # Per model; each trial evaluates all folds.
        n_splits=5,                 # Expanding training windows, later validation events.
        gap=0,
        seed=42,
        logit_alpha_bounds=(0.2, 7.0),
        glm_alpha_bounds=(1e-5, 7.0),
        l1_wt_bounds=(0.0, 1.0),
    )
