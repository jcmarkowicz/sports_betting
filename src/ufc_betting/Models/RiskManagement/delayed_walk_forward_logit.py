"""Opening-only alpha search and delayed Monte Carlo betting search.

Default: TrainTestBuilder(.85), dates strictly after 2010-02-26, refit every
three calendar months. Alpha minimizes mean event-grouped TimeSeriesSplit
validation Brier. Betting maximizes mean/std of bootstrap terminal returns.

For a deployment boundary T: predict [T-n,T) with models available then, tune
bets on that block, and deploy on [T,T+n) with the model refitted at T. Neither
the historical alpha nor its fitted model can use a prediction's outcome.
All intervals are left-inclusive/right-exclusive. Dates identify event groups.
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ufc_betting.config import config
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import fair_probabilities
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    FrozenBettingSearch, ODDS_TYPES,
)


def opening_features(features=None):
    features = list(config.open_feats if features is None else features)
    if not features or any('close1' in name or 'close2' in name for name in features):
        raise ValueError('Model training accepts opening features only')
    return features


def make_preprocessor(frame):
    categorical = list(frame.select_dtypes(include=['category', 'object', 'bool', 'str']).columns)
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
        raise ValueError('Opening Logit did not converge')
    return model


def opening_feature_frame(frame, features):
    X = frame[list(features)].copy()
    for name in ('math_red', 'math_blue', 'elo_pred', 'womens_fight'):
        if name in X:
            X[name] = X[name].astype('category')
    return X


def training_frame(frame, cutoff, features):
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date, errors='raise').dt.normalize()
    data = data.sort_values('date', kind='stable').reset_index(drop=True)
    if data.empty or data.date.isna().any() or (data.date >= pd.Timestamp(cutoff)).any():
        raise ValueError('All training rows must precede the model training cutoff')
    if not data.winner.isin([0, 1]).all() or data[features].isna().any().any():
        raise ValueError('Training requires complete opening features and binary outcomes')
    return data


def chronological_folds(dates, n_splits=5):
    """Use sklearn TimeSeriesSplit on unique events, then expand to fight rows."""
    dates = pd.to_datetime(pd.Series(dates)).dt.normalize().reset_index(drop=True)
    events = np.sort(dates.unique())
    for train_events, valid_events in TimeSeriesSplit(n_splits=n_splits).split(events):
        train = np.flatnonzero(dates.isin(events[train_events]))
        valid = np.flatnonzero(dates.isin(events[valid_events]))
        yield train, valid


def fit_opening_bundle(frame, cutoff, alpha, *, features=None, cv_brier=None):
    """Fit prediction and inference logits on the same opening-only history."""
    features = opening_features(features)
    data = training_frame(frame, cutoff, features)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite')
    X = opening_feature_frame(data, features)
    preprocessor = make_preprocessor(X)
    design = preprocessor.fit_transform(X)
    model = fit_model(design, data.winner, alpha)
    full_design = sm.add_constant(design, has_constant='add')

    # Post-selection inference: retain an intercept and independent active terms.
    active = np.flatnonzero(np.abs(np.asarray(model.params)) > 1e-8)
    selected = [0]
    for column in active:
        if column != 0 and np.linalg.matrix_rank(full_design[:, selected + [column]]) > len(selected):
            selected.append(int(column))
    inference = sm.Logit(data.winner.to_numpy(), full_design[:, selected]).fit(
        disp=False, maxiter=1000
    )
    if (not inference.mle_retvals.get('converged', True)
            or not np.isfinite(inference.cov_params()).all()):
        raise ValueError('Opening inference Logit has no valid converged covariance')
    return dict(model=model, inference_model=inference, inference_columns=selected,
                preprocessor=preprocessor, features=features, training_odds_type='open',
                training_cutoff=str(pd.Timestamp(cutoff).date()),
                training_start=str(data.date.min().date()),
                training_last_date=str(data.date.max().date()), training_rows=len(data),
                alpha=float(alpha), cv_brier=cv_brier)


def train_opening_bundle(frame, cutoff, *, features=None, n_splits=5,
                         n_trials=30, seed=42):
    """Search only alpha, minimizing the equal-weight mean fold Brier score."""
    features = opening_features(features)
    data = training_frame(frame, cutoff, features)
    if n_trials < 1:
        raise ValueError('n_trials must be positive')
    X = opening_feature_frame(data, features)
    y = data.winner.to_numpy()
    folds, audit = [], []
    for train, valid in chronological_folds(data.date, n_splits):
        if len(np.unique(y[train])) < 2:
            raise ValueError('Each chronological training fold needs both outcome classes')
        preprocessor = make_preprocessor(X.iloc[train])
        folds.append((preprocessor.fit_transform(X.iloc[train]), y[train],
                      preprocessor.transform(X.iloc[valid]), y[valid]))
        audit.append(dict(train_end=str(data.date.iloc[train].max().date()),
                          validation_start=str(data.date.iloc[valid].min().date()),
                          validation_end=str(data.date.iloc[valid].max().date()),
                          train_rows=len(train), validation_rows=len(valid)))
    study = optuna.create_study(direction='minimize',
                               sampler=optuna.samplers.TPESampler(seed=seed))
    study.set_user_attr('folds', audit)
    study.set_user_attr('training_odds_type', 'open')

    def objective(trial):
        alpha = trial.suggest_float('alpha', .2, 7.0, log=True)
        scores = []
        for train_X, train_y, valid_X, valid_y in folds:
            try:
                model = fit_model(train_X, train_y, alpha)
                p = model.predict(sm.add_constant(valid_X, has_constant='add'))
                scores.append(float(brier_score_loss(valid_y, p)))
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise optuna.TrialPruned(str(exc)) from exc
        trial.set_user_attr('fold_brier', scores)
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError('No alpha trial completed; inspect training features and fold sizes')
    bundle = fit_opening_bundle(data, cutoff, study.best_params['alpha'],
                                features=features, cv_brier=study.best_value)
    return bundle, study


def predict_odds(frame, bundle, odds_type='open'):
    """Selected market prices enter opening-feature slots; never retrain here."""
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    result = frame.copy().reset_index(drop=True)
    result['date'] = pd.to_datetime(result.date).dt.normalize()
    if (result.date < pd.Timestamp(bundle['training_cutoff'])).any():
        raise ValueError('Cannot use a model on dates before its training cutoff')
    prices = result[[f'dec_{odds_type}_red', f'dec_{odds_type}_blue']].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 1).any():
        raise ValueError(f'{odds_type} decimal odds must be finite and exceed 1')
    fair = fair_probabilities(prices)
    result[f'dec_fair_{odds_type}_red'] = 1 / fair[:, 0]
    result[f'dec_fair_{odds_type}_blue'] = 1 / fair[:, 1]
    result[f'proba_fair_{odds_type}_diff'] = fair[:, 0] - fair[:, 1]
    inputs = result.copy()
    for name in bundle['features']:
        if 'open' in name:
            replacement = name.replace('open', odds_type)
            if replacement not in inputs:
                raise ValueError(f'Missing {odds_type} replacement for {name}')
            inputs[name] = inputs[replacement]
    design = bundle['preprocessor'].transform(opening_feature_frame(inputs, bundle['features']))
    design = sm.add_constant(design, has_constant='add')
    p = np.asarray(bundle['model'].predict(design))
    infer_design = design[:, bundle['inference_columns']]
    inference = bundle['inference_model']
    risk_p = np.asarray(inference.predict(infer_design))
    # Delta-method probability SE for the inference model (not coefficient SE).
    gradient = infer_design * (risk_p * (1 - risk_p))[:, None]
    variance = np.einsum('ij,jk,ik->i', gradient, inference.cov_params(), gradient)
    se = np.sqrt(np.maximum(variance, 0))
    if not np.isfinite(np.column_stack((p, risk_p, se))).all():
        raise ValueError('Nonfinite model probability or inference SE')
    result['proba_red'], result['proba_blue'] = p, 1 - p
    result['risk_red_proba'], result['prediction_se'] = risk_p, se
    result['pred_winner'] = (p >= .5).astype(int)
    result['odds_type'] = odds_type
    result['model_alpha'] = bundle['alpha']
    result['model_training_cutoff'] = bundle['training_cutoff']
    result['prediction_source'] = f'delayed_open_logit_{odds_type}_inputs'
    return result


def predict_close1(frame, bundle):
    """Compatibility wrapper; new callers should specify predict_odds odds_type."""
    return predict_odds(frame, bundle, 'close1')


def prepare_history(frame, *, features=None, train_size=.85):
    features = opening_features(features)
    builder = TrainTestBuilder(
        frame, features, target_col='winner', date_col='date', odds_type='open',
        year=2010, month=2, day=26, required_features_only=True,
    )
    split = builder.prepare_train_test(train_size, scale=False, group_dates=True)
    return split['filtered_df'], split


def validate_alpha_history(alpha_history):
    """effective_date is when selection was available; cutoff excludes that day."""
    required = {'effective_date', 'training_cutoff', 'alpha'}
    if not required <= set(alpha_history.columns) or alpha_history.empty:
        raise ValueError(f'alpha_history needs nonempty columns {sorted(required)}')
    history = alpha_history.copy()
    for name in ('effective_date', 'training_cutoff'):
        history[name] = pd.to_datetime(history[name], errors='raise').dt.normalize()
    if (history[['effective_date', 'training_cutoff']].isna().any().any()
            or history.effective_date.duplicated().any()
            or (history.training_cutoff > history.effective_date).any()):
        raise ValueError('Alpha dates must be unique and training_cutoff <= effective_date')
    history['alpha'] = pd.to_numeric(history.alpha, errors='raise')
    if not np.isfinite(history.alpha).all() or (history.alpha <= 0).any():
        raise ValueError('Historical alphas must be positive and finite')
    if 'training_start' in history:
        history['training_start'] = pd.to_datetime(history.training_start, errors='raise')
        if history.training_start.isna().any() or (history.training_start >= history.training_cutoff).any():
            raise ValueError('training_start must precede training_cutoff')
    return history.sort_values('effective_date').reset_index(drop=True)


def search_alpha_history(frame, start, end, output_dir, *, n_months=3,
                         features=None, n_splits=5, n_trials=30, seed=42):
    """Independent model search; outputs a reusable dated alpha DataFrame."""
    if not isinstance(n_months, int) or n_months < 1:
        raise ValueError('n_months must be a positive integer')
    features = opening_features(features)
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date).dt.normalize()
    cutoff, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if cutoff >= end:
        raise ValueError('Alpha search start must precede end')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, bundles = [], {}
    while cutoff < end:
        train = data.loc[data.date < cutoff]
        bundle, study = train_opening_bundle(
            train, cutoff, features=features, n_splits=n_splits, n_trials=n_trials, seed=seed,
        )
        directory = output / 'models' / str(cutoff.date())
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, directory / 'opening_model.joblib')
        study.trials_dataframe().to_csv(directory / 'alpha_trials.csv', index=False)
        (directory / 'folds.json').write_text(json.dumps(study.user_attrs['folds'], indent=2))
        rows.append(dict(effective_date=cutoff, training_cutoff=cutoff,
                         training_start=bundle['training_start'], alpha=bundle['alpha'],
                         cv_brier=bundle['cv_brier'], training_rows=bundle['training_rows']))
        bundles[cutoff] = bundle
        pd.DataFrame(rows).to_csv(output / 'alpha_history.csv', index=False)
        cutoff += pd.DateOffset(months=n_months)
    return validate_alpha_history(pd.DataFrame(rows)), bundles


def predict_from_alpha_history(frame, training_history, alpha_history, *,
                               odds_type='open', features=None, bundles=None):
    """As-of alpha lookup; fit each historical opening bundle once, before trials."""
    history = validate_alpha_history(alpha_history)
    features = opening_features(features)
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date).dt.normalize()
    data = data.sort_values('date', kind='stable').reset_index(drop=True)
    if data.empty:
        raise ValueError('Prediction interval contains no events')
    assignments = np.searchsorted(history.effective_date.to_numpy(),
                                  data.date.to_numpy(), side='right') - 1
    if (assignments < 0).any():
        raise ValueError('Alpha history does not cover the earliest prediction date')
    train_data = training_history.copy()
    train_data['date'] = pd.to_datetime(train_data.date).dt.normalize()
    bundles = {} if bundles is None else bundles
    predicted = []
    for index in np.unique(assignments):
        row = history.iloc[index]
        key = row.effective_date
        if key not in bundles:
            mask = train_data.date < row.training_cutoff
            if 'training_start' in history:
                mask &= train_data.date >= row.training_start
            bundles[key] = fit_opening_bundle(
                train_data.loc[mask], row.training_cutoff, row.alpha,
                features=features, cv_brier=row.get('cv_brier', None),
            )
        bundle = bundles[key]
        if (bundle['alpha'] != row.alpha
                or pd.Timestamp(bundle['training_cutoff']) != row.training_cutoff
                or bundle['features'] != features):
            raise ValueError('Cached model does not match alpha history')
        part = predict_odds(data.loc[assignments == index], bundle, odds_type)
        part['alpha_effective_date'] = str(key.date())
        predicted.append(part)
    return pd.concat(predicted, ignore_index=True)


def window_masks(dates, test_start, test_end, n_months=3):
    dates = pd.to_datetime(dates).dt.normalize()
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    tuning_start = test_start - pd.DateOffset(months=n_months)
    return (dates < tuning_start,
            (dates >= tuning_start) & (dates < test_start),
            (dates >= test_start) & (dates < test_end))


def kelly_simulation_input(predictions, odds_type='open', initial_bankroll=500.0):
    """Return a fight-level DataFrame and matching simulate_kelly arguments.

    Keep original prices, fighters, dates, and per-window best_* parameters.
    Column lists are blue first, red second, matching winner encoding 0/1.
    """
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    data = predictions.copy().sort_values('date', kind='stable').reset_index(drop=True)
    required = ['fighter_red', 'fighter_blue', 'winner', 'pred_winner',
                'proba_blue', 'proba_red', 'risk_red_proba', 'prediction_se']
    required += [f'{prefix}_{odds_type}_{side}' for prefix in ('dec', 'dec_fair')
                 for side in ('blue', 'red')]
    required += [f'best_{name}' for name in ('mdd', 'N', 'mdd_parlay', 'N_parlay', 'z',
                                          'moneyline_min_american', 'moneyline_max_american')]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing Kelly simulation input columns: {missing}')
    if 'odds_type' in data and not data.odds_type.eq(odds_type).all():
        raise ValueError('Kelly simulation odds must match the prediction odds_type')
    data['risk_proba_red'] = data.risk_red_proba
    data['risk_proba_blue'] = 1 - data.risk_red_proba
    data['se'] = data.prediction_se
    data['proba_se'] = data.prediction_se
    # The simulator's parameter resolver requires these even with heavy-favorite
    # parlays disabled. They are compatibility defaults, not optimized settings.
    data['best_heavy_parlay_max_legs'] = 4
    data['best_heavy_favorite_cutoff'] = -300
    data['best_heavy_parlay_mdd'] = data.best_mdd_parlay
    # The existing simulator includes original American opening odds in its
    # reports, even when execution uses closing prices.
    for side in ('blue', 'red'):
        if f'open_{side}' not in data:
            decimal = data[f'dec_open_{side}'].to_numpy(float)
            if not np.isfinite(decimal).all() or (decimal <= 1).any():
                raise ValueError('Valid opening decimal odds are required for Kelly reports')
            data[f'open_{side}'] = np.where(decimal >= 2, 100 * (decimal - 1),
                                           -100 / (decimal - 1))
    kwargs = dict(
        prob_cols=['proba_blue', 'proba_red'],
        risk_prob_cols=['risk_proba_blue', 'risk_proba_red'],
        fair_decimal_cols=[f'dec_fair_{odds_type}_blue', f'dec_fair_{odds_type}_red'],
        real_decimal_cols=[f'dec_{odds_type}_blue', f'dec_{odds_type}_red'],
        pred_winner_col='pred_winner', prediction_se_col='se',
        init_bankroll=initial_bankroll, calc_parlay=True,
        calc_heavy_favorite_parlay=False, use_walk_forward_params=True,
    )
    return data, kwargs


def run_delayed_walk_forward(frame, test_start=None, test_end=None, output_dir=None, *,
                             model_trials=30, betting_trials=100, n_splits=5,
                             initial_bankroll=500.0, seed=42, features=None,
                             n_months=3, k=1000, odds_type='open', train_size=.85,
                             alpha_history=None):
    """Two independent searches; supplied alpha_history skips model optimization.

    The initial preceding block is a warm-up prediction block trained before it,
    never in-sample predictions from the initial 85% fit. At each later boundary
    the current opening model may incorporate previously settled test outcomes.
    """
    if not isinstance(n_months, int) or n_months < 1 or k < 2:
        raise ValueError('Need positive integer n_months and k >= 2')
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be positive and finite')
    if output_dir is None:
        raise ValueError('output_dir is required')
    features = opening_features(features)
    data, split = prepare_history(frame, features=features, train_size=train_size)
    start = pd.Timestamp(test_start if test_start is not None else split['dates_test'].min()).normalize()
    end = pd.Timestamp(test_end if test_end is not None else data.date.max() + pd.Timedelta(days=1)).normalize()
    if start >= end:
        raise ValueError('test_start must precede test_end')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tuning_start = start - pd.DateOffset(months=n_months)
    bundles = {}
    if alpha_history is None:
        # Train warm-up model first. Generate deployment models separately so
        # month-end subtraction/addition cannot shift the refit schedule.
        warm_history, warm_bundles = search_alpha_history(
            data, tuning_start, tuning_start + pd.Timedelta(days=1), output / 'warmup',
            n_months=n_months, features=features, n_splits=n_splits,
            n_trials=model_trials, seed=seed,
        )
        alpha_history, bundles = search_alpha_history(
            data, start, end, output, n_months=n_months, features=features,
            n_splits=n_splits, n_trials=model_trials, seed=seed,
        )
        alpha_history = pd.concat([warm_history, alpha_history], ignore_index=True)
        bundles.update(warm_bundles)
    alpha_history = validate_alpha_history(alpha_history)
    alpha_history.to_csv(output / 'alpha_history.csv', index=False)
    (output / 'run_config.json').write_text(json.dumps(dict(
        odds_type=odds_type, training_odds_type='open', n_months=n_months, k=k,
        train_size=train_size, split_date=str(split['dates_test'].min().date()),
        test_start=str(start.date()), test_end=str(end.date()), seed=seed,
        initial_bankroll=initial_bankroll, features=features,
        objective='mean_terminal_cumulative_return / sample_std',
        settlement='after_each_event', date_filter='date > 2010-02-26',
    ), indent=2), encoding='utf-8')
    summaries, event_tables, predictions = [], [], []
    bankroll = float(initial_bankroll)
    # Use the previous actual block boundary, avoiding month-end date drift.
    previous_start = tuning_start
    while start < end:
        natural_end = start + pd.DateOffset(months=n_months)
        stop = min(natural_end, end)
        tune = data.loc[(data.date >= previous_start) & (data.date < start)]
        test = data.loc[(data.date >= start) & (data.date < stop)]
        if tune.empty or test.empty:
            raise ValueError(f'Empty tuning or test interval for {start.date()}')
        window = output / str(start.date())
        window.mkdir(exist_ok=True)
        print(f'{start.date()}: tune {previous_start.date()} to {start.date()}, test until {stop.date()}')
        tune_predictions = predict_from_alpha_history(
            tune, data, alpha_history, odds_type=odds_type, features=features, bundles=bundles)
        search = FrozenBettingSearch(tune_predictions, odds_type, k=k,
                                     initial_bankroll=initial_bankroll, seed=seed)
        study = search.run(betting_trials, seed)
        params = study.best_params
        test_predictions = predict_from_alpha_history(
            test, data, alpha_history, odds_type=odds_type, features=features, bundles=bundles)
        events = FrozenBettingSearch(test_predictions, odds_type, k=k,
                                     initial_bankroll=initial_bankroll, seed=seed).event_returns(params)
        events['window_start'] = str(start.date())
        before = bankroll * np.r_[1.0, (1 + events.event_return.to_numpy()).cumprod()[:-1]]
        events['moneyline_profit'] = before * events.moneyline_return
        events['parlay_profit'] = before * events.parlay_return
        events['bankroll'] = bankroll * (1 + events.event_return).cumprod()
        window_return = float(events.bankroll.iloc[-1] / bankroll - 1) if bankroll > 0 else 0.0
        bankroll = float(events.bankroll.iloc[-1])
        test_predictions['window_start'] = str(start.date())
        for name, value in params.items():
            test_predictions[f'best_{name}'] = value
        paths = search.simulate(params)
        terminal = paths['terminal_returns']
        summary = dict(test_start=str(start.date()), test_end=str(stop.date()),
                       tuning_start=str(previous_start.date()), partial_test_window=stop < natural_end,
                       tuning_events=tune.date.nunique(), test_events=test.date.nunique(),
                       odds_type=odds_type, k=k, tuning_terminal_ratio=study.best_value,
                       tuning_mean_terminal_return=float(terminal.mean()),
                       tuning_std_terminal_return=float(terminal.std(ddof=1)),
                       test_brier=float(brier_score_loss(test_predictions.winner, test_predictions.proba_red)),
                       test_compounded_return=window_return, ending_bankroll=bankroll, **params)
        (window / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        study.trials_dataframe().to_csv(window / 'betting_trials.csv', index=False)
        tune_predictions.to_csv(window / 'tuning_predictions.csv', index=False)
        search.event_returns(params).to_csv(window / 'tuning_events.csv', index=False)
        np.savez_compressed(window / 'bootstrap_paths.npz', event_indices=search.path_indices, **paths)
        pd.DataFrame(dict(path_id=np.arange(k), cumulative_return=terminal,
                          final_bankroll=paths['bankroll'][:, -1])).to_csv(
                              window / 'bootstrap_terminal_returns.csv', index=False)
        test_predictions.to_csv(window / 'test_predictions.csv', index=False)
        events.to_csv(window / 'test_events.csv', index=False)
        summaries.append(summary)
        event_tables.append(events)
        predictions.append(test_predictions)
        previous_start, start = start, stop
    # Also persist opening models reconstructed from a supplied alpha history.
    for effective_date, bundle in bundles.items():
        directory = output / 'models' / str(effective_date.date())
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, directory / 'opening_model.joblib')
    result = dict(windows=pd.DataFrame(summaries), events=pd.concat(event_tables, ignore_index=True),
                  predictions=pd.concat(predictions, ignore_index=True), alpha_history=alpha_history)
    result['betting_data'], kelly_kwargs = kelly_simulation_input(
        result['predictions'], odds_type, initial_bankroll)
    for name, table in result.items():
        table.to_csv(output / f'{name}.csv', index=False)
    result['kelly_sim_kwargs'] = kelly_kwargs
    (output / 'kelly_sim_kwargs.json').write_text(json.dumps(kelly_kwargs, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Historical fight/features CSV')
    parser.add_argument('--test-start', help='Defaults to the event-aligned 85%% split')
    parser.add_argument('--test-end', help='Exclusive; defaults to day after final event')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['both', 'alpha', 'betting'], default='both')
    parser.add_argument('--alpha-history', help='CSV: effective_date,training_cutoff,alpha; optional training_start')
    parser.add_argument('--odds-type', choices=ODDS_TYPES, default='open')
    parser.add_argument('--months', type=int, default=3)
    parser.add_argument('--paths', type=int, default=1000)
    parser.add_argument('--train-size', type=float, default=.85)
    parser.add_argument('--model-trials', type=int, default=30)
    parser.add_argument('--betting-trials', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--bankroll', type=float, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    if args.mode == 'betting' and not args.alpha_history:
        parser.error('--mode betting requires --alpha-history')
    if args.mode == 'alpha':
        # Same warm-up and deployment schedule as the full pipeline.
        data, split = prepare_history(frame, train_size=args.train_size)
        start = pd.Timestamp(args.test_start or split['dates_test'].min())
        end = pd.Timestamp(args.test_end or data.date.max() + pd.Timedelta(days=1))
        tuning_start = start - pd.DateOffset(months=args.months)
        warm, _ = search_alpha_history(
            data, tuning_start, tuning_start + pd.Timedelta(days=1),
            Path(args.output_dir) / 'warmup', n_months=args.months,
            n_splits=args.folds, n_trials=args.model_trials, seed=args.seed)
        history, _ = search_alpha_history(data, start, end, args.output_dir, n_months=args.months,
                                         n_splits=args.folds, n_trials=args.model_trials, seed=args.seed)
        pd.concat([warm, history], ignore_index=True).to_csv(
            Path(args.output_dir) / 'alpha_history.csv', index=False)
    else:
        history = pd.read_csv(args.alpha_history) if args.alpha_history else None
        run_delayed_walk_forward(
            frame, args.test_start, args.test_end, args.output_dir,
            model_trials=args.model_trials, betting_trials=args.betting_trials,
            n_splits=args.folds, initial_bankroll=args.bankroll, seed=args.seed,
            n_months=args.months, k=args.paths, odds_type=args.odds_type,
            train_size=args.train_size, alpha_history=history,
        )


if __name__ == '__main__':
    main()
