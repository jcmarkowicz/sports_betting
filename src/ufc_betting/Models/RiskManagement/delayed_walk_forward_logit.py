"""Opening Logit -> three-month betting calibration -> three-month holdout.

Run with --help for the CSV entry point. Model CV uses opening features only;
frozen inference replaces their opening-odds inputs with close1 equivalents.
All date intervals are left-inclusive, right-exclusive. Event dates are groups.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ufc_betting.config import config
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import fair_probabilities
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import FrozenBettingSearch, log_growth


def make_preprocessor(frame):
    categorical = list(frame.select_dtypes(include=['category', 'object', 'bool']).columns)
    numeric = [name for name in frame if name not in categorical]
    return ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore',
                                     sparse_output=False), categorical),
    ], sparse_threshold=0)


def fit_model(X, y, alpha):
    model = sm.Logit(np.asarray(y), sm.add_constant(X, has_constant='add')).fit_regularized(
        method='l1', alpha=alpha, disp=False, maxiter=1000,
    )
    if not model.mle_retvals.get('converged', True):
        raise ValueError("Opening Logit did not converge")
    return model


def opening_feature_frame(frame, features):
    X = frame[list(features)].copy()
    for name in ('math_red', 'math_blue', 'elo_pred', 'womens_fight'):
        if name in X:
            X[name] = X[name].astype('category')
    return X


def train_opening_bundle(frame, cutoff, *, features=None, n_splits=5,
                         n_trials=30, seed=42):
    """Choose L1 alpha by pooled OOF Brier, then fit all rows before cutoff.

    The supplied frame must already be restricted to the training interval.
    Preprocessors are independently fitted inside each event-grouped fold.
    """
    features = list(config.open_feats if features is None else features)
    dates = pd.to_datetime(frame.date).dt.normalize()
    if frame.empty or (dates >= pd.Timestamp(cutoff)).any():
        raise ValueError("Training rows must precede the three-month tuning window")
    if n_splits < 2 or n_splits > dates.nunique() or n_trials < 1:
        raise ValueError("Need positive trials and 2 <= folds <= training event count")
    X = opening_feature_frame(frame, features)
    y = frame.winner.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train, valid in splitter.split(X, y, groups=dates):
        if len(np.unique(y[train])) < 2:
            raise ValueError("Each training fold needs both outcome classes")
        preprocessor = make_preprocessor(X.iloc[train])
        folds.append((preprocessor.fit_transform(X.iloc[train]), y[train],
                      preprocessor.transform(X.iloc[valid]), valid))
    study = optuna.create_study(direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        alpha = trial.suggest_float('alpha', .2, 7.0, log=True)
        predictions = np.empty(len(y))
        for train_X, train_y, valid_X, valid in folds:
            model = fit_model(train_X, train_y, alpha)
            predictions[valid] = model.predict(sm.add_constant(valid_X, has_constant='add'))
        return float(brier_score_loss(y, predictions))
    study.optimize(objective, n_trials=n_trials)
    preprocessor = make_preprocessor(X)
    model = fit_model(preprocessor.fit_transform(X), y, study.best_params['alpha'])
    bundle = dict(model=model, preprocessor=preprocessor, features=features,
                  training_cutoff=str(pd.Timestamp(cutoff).date()),
                  training_last_date=str(dates.max().date()),
                  training_rows=len(frame), alpha=study.best_params['alpha'],
                  cv_brier=study.best_value)
    return bundle, study


def predict_close1(frame, bundle):
    """Use frozen coefficients/preprocessing, with close1 prices in opening slots."""
    result = frame.copy().reset_index(drop=True)
    prices = result[['dec_close1_red', 'dec_close1_blue']].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 1).any():
        raise ValueError("close1 decimal odds must be finite and exceed 1")
    fair = fair_probabilities(prices)
    result['dec_fair_close1_red'] = 1 / fair[:, 0]
    result['dec_fair_close1_blue'] = 1 / fair[:, 1]
    result['proba_fair_close1_diff'] = fair[:, 0] - fair[:, 1]
    inputs = result.copy()
    for name in bundle['features']:
        if 'open' in name:
            close_name = name.replace('open', 'close1')
            if close_name not in inputs:
                raise ValueError(f"No close1 replacement for opening feature {name}")
            inputs[name] = inputs[close_name]
    X = opening_feature_frame(inputs, bundle['features'])
    design = bundle['preprocessor'].transform(X)
    p = np.asarray(bundle['model'].predict(sm.add_constant(design, has_constant='add')))
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid frozen-model probabilities")
    result['proba_red'] = p
    result['proba_blue'] = 1 - p
    result['pred_winner'] = (p >= .5).astype(int)
    result['prediction_source'] = 'delayed_open_logit_close1_inputs'
    return result


def window_masks(dates, test_start, test_end):
    dates = pd.to_datetime(dates).dt.normalize()
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    tuning_start = test_start - pd.DateOffset(months=3)
    return (dates < tuning_start,
            (dates >= tuning_start) & (dates < test_start),
            (dates >= test_start) & (dates < test_end))


def run_delayed_walk_forward(frame, test_start, test_end, output_dir, *,
                             model_trials=30, betting_trials=100, n_splits=5,
                             initial_bankroll=500.0, seed=42, features=None):
    """Refit every three months; never refit on the betting tuning interval.

    test_end is exclusive. A final shorter holdout is explicitly marked partial.
    Saves separate model bundles; never overwrites the production opening model.
    """
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError("initial_bankroll must be positive and finite")
    features = list(config.open_feats if features is None else features)
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date, errors='raise').dt.normalize()
    data = data.sort_values('date', kind='stable').reset_index(drop=True)
    required = list(dict.fromkeys(features + ['date', 'winner']))
    # Do not discard historical opening rows merely because closing odds are absent.
    eligible = data.winner.isin([0, 1]) & data[required].notna().all(axis=1)
    data = data.loc[eligible].reset_index(drop=True)
    start, end = pd.Timestamp(test_start).normalize(), pd.Timestamp(test_end).normalize()
    if start >= end:
        raise ValueError("test_start must precede test_end")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries, event_tables, predictions = [], [], []
    bankroll = float(initial_bankroll)
    while start < end:
        natural_end = start + pd.DateOffset(months=3)
        stop = min(natural_end, end)
        cutoff = start - pd.DateOffset(months=3)
        train_mask, tune_mask, test_mask = window_masks(data.date, start, stop)
        train, tune, test = (data.loc[mask].reset_index(drop=True)
                             for mask in (train_mask, tune_mask, test_mask))
        if train.empty or tune.empty or test.empty:
            raise ValueError(f"Empty training, tuning, or test interval for {start.date()}")
        window = output / str(start.date())
        window.mkdir(exist_ok=True)
        print(f'{start.date()}: fit before {cutoff.date()}, tune preceding 3 months, test until {stop.date()}')
        bundle, model_study = train_opening_bundle(
            train, cutoff, features=features, n_splits=n_splits,
            n_trials=model_trials, seed=seed,
        )
        joblib.dump(bundle, window / 'opening_model.joblib')
        # The persisted opening model is loaded once, then frozen for both windows.
        bundle = joblib.load(window / 'opening_model.joblib')
        tune_predictions = predict_close1(tune, bundle)
        search = FrozenBettingSearch(tune_predictions)
        betting_study = search.run(betting_trials, seed)
        params = betting_study.best_params
        test_predictions = predict_close1(test, bundle)
        events = FrozenBettingSearch(test_predictions).event_returns(params)
        events['window_start'] = str(start.date())
        events['bankroll'] = bankroll * (1 + events.event_return).cumprod()
        bankroll = float(events.bankroll.iloc[-1])
        test_predictions['window_start'] = str(start.date())
        for name, value in params.items():
            test_predictions[f'best_{name}'] = value
        summary = dict(test_start=str(start.date()), test_end=str(stop.date()),
                       tuning_start=str(cutoff.date()), partial_test_window=stop < natural_end,
                       training_rows=len(train), tuning_events=tune.date.nunique(),
                       test_events=test.date.nunique(), alpha=bundle['alpha'],
                       cv_brier=bundle['cv_brier'],
                       tuning_compounded_return=float(np.expm1(betting_study.best_value)),
                       test_compounded_return=float(np.expm1(log_growth(events.event_return))),
                       ending_bankroll=bankroll, **params)
        (window / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        model_study.trials_dataframe().to_csv(window / 'model_trials.csv', index=False)
        betting_study.trials_dataframe().to_csv(window / 'betting_trials.csv', index=False)
        tune_predictions.to_csv(window / 'tuning_predictions.csv', index=False)
        search.event_returns(params).to_csv(window / 'tuning_events.csv', index=False)
        test_predictions.to_csv(window / 'test_predictions.csv', index=False)
        events.to_csv(window / 'test_events.csv', index=False)
        summaries.append(summary)
        event_tables.append(events)
        predictions.append(test_predictions)
        start = stop
    result = dict(windows=pd.DataFrame(summaries), events=pd.concat(event_tables, ignore_index=True),
                  predictions=pd.concat(predictions, ignore_index=True))
    for name, table in result.items():
        table.to_csv(output / f'{name}.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Historical fight/features CSV')
    parser.add_argument('--test-start', required=True)
    parser.add_argument('--test-end', required=True, help='Exclusive date bound')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-trials', type=int, default=30)
    parser.add_argument('--betting-trials', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--bankroll', type=float, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_delayed_walk_forward(pd.read_csv(args.data), args.test_start, args.test_end,
                            args.output_dir, model_trials=args.model_trials,
                            betting_trials=args.betting_trials, n_splits=args.folds,
                            initial_bankroll=args.bankroll, seed=args.seed)


if __name__ == '__main__':
    main()
