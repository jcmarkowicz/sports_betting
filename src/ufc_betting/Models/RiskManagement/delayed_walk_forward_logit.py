"""Opening-only alpha search and delayed Monte Carlo betting search.

Default: TrainTestBuilder(.70), dates strictly after 2010-02-26, model refit every
calendar month. Alpha minimizes mean event-grouped TimeSeriesSplit
validation Brier. Betting maximizes mean bootstrap terminal return minus a
configurable penalty times its sample standard deviation.

Hold out the first n months of test predictions for betting calibration. Tune
on trailing n months every k months; opening models update monthly. Neither
the historical alpha nor its fitted model can use a prediction's outcome.
All intervals are left-inclusive/right-exclusive. Dates identify event groups.
"""
import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Support an editor's "Run Python File" without an editable package install.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ufc_betting.config import config, settings
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import fair_probabilities
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    FrozenBettingSearch, ODDS_TYPES,
)

DEFAULT_OUTPUT_DIR = settings.data_dir / 'search_results'


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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
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
    prediction = inference.get_prediction(infer_design).summary_frame()
    risk_p = prediction['predicted'].to_numpy()
    se = prediction['se'].to_numpy()
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


def prepare_history(frame, *, features=None, train_size=.7):
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


def calendar_boundaries(start, end, months):
    """Anchor every boundary to the original date, including month ends."""
    if not isinstance(months, int) or months < 1:
        raise ValueError('Month intervals must be positive integers')
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    index = 0
    while start + pd.DateOffset(months=index * months) < end:
        yield start + pd.DateOffset(months=index * months)
        index += 1


def search_alpha_history(frame, start, end, output_dir=None, *, n_months=1,
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
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    rows, bundles = [], {}
    for cutoff in calendar_boundaries(start, end, n_months):
        train = data.loc[data.date < cutoff]
        bundle, study = train_opening_bundle(
            train, cutoff, features=features, n_splits=n_splits, n_trials=n_trials, seed=seed,
        )
        directory = output / 'models' / str(cutoff.date())
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, directory / 'opening_model.joblib')
        pd.DataFrame([dict(effective_date=cutoff, training_cutoff=cutoff,
                           training_start=bundle['training_start'],
                           alpha=bundle['alpha'], cv_brier=bundle['cv_brier'])]).to_csv(
                               directory / 'best_alpha.csv', index=False)
        print(f'[alpha retrained] training [{bundle["training_start"]}, {cutoff.date()}); '
              f'last training event {bundle["training_last_date"]}; alpha={bundle["alpha"]:.6g}')
        (directory / 'folds.json').write_text(json.dumps(study.user_attrs['folds'], indent=2))
        rows.append(dict(effective_date=cutoff, training_cutoff=cutoff,
                         training_start=bundle['training_start'], alpha=bundle['alpha'],
                         cv_brier=bundle['cv_brier'], training_rows=bundle['training_rows']))
        bundles[cutoff] = bundle
        pd.DataFrame(rows).to_csv(output / 'alpha_history.csv', index=False)
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


def log_alpha_periods(predictions, bundles, odds_type, role, start, stop):
    """Print actual historical model cutoffs, including when alphas are reused."""
    for effective, group in predictions.groupby('alpha_effective_date', sort=True):
        bundle = bundles[pd.Timestamp(effective)]
        print(f'[{odds_type}] {role} [{start.date()}, {stop.date()}): '
              f'prediction dates {group.date.min().date()} to {group.date.max().date()}; '
              f'alpha effective {effective}; opening training '
              f'[{bundle["training_start"]}, {bundle["training_cutoff"]}); '
              f'last training event {bundle["training_last_date"]}; '
              f'alpha={bundle["alpha"]:.6g}')


def kelly_simulation_input(predictions, odds_type='open', initial_bankroll=500.0):
    """Return a fight-level DataFrame and matching simulate_kelly arguments.

    Keep only Kelly inputs and prediction provenance; exclude raw model features.
    Column lists are blue first, red second, matching winner encoding 0/1.
    """
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    data = predictions.copy().sort_values('date', kind='stable').reset_index(drop=True)
    # No upper moneyline limit is searched. Keep the simulator's required column;
    # preserve explicit bounds when adapting historical results.
    if 'best_moneyline_max_american' not in data:
        data['best_moneyline_max_american'] = np.inf
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
    # This workflow does not search or simulate heavy-favorite parlays.
    data = data.drop(columns=['best_heavy_parlay_max_legs',
                              'best_heavy_favorite_cutoff', 'best_heavy_parlay_mdd'],
                     errors='ignore')
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
        adjust_mdd_by_edge=False,
    )
    columns = list(dict.fromkeys([
        'date', *required, 'open_blue', 'open_red',
        'risk_proba_blue', 'risk_proba_red', 'se', 'proba_se',
    ]))
    columns += [name for name in (
        'odds_type', 'refit_start', 'window_start', 'prediction_source',
        'model_alpha', 'alpha_effective_date', 'model_training_cutoff', 'alpha_source',
        'bet_probability_source',
    ) if name in data]
    return data.loc[:, columns].copy(), kwargs


def run_delayed_walk_forward(frame, test_start=None, test_end=None, output_dir=None, *,
                             model_trials=30, betting_trials=100, n_splits=5,
                             initial_bankroll=500.0, seed=42, features=None,
                             model_refit_months=1, betting_lookback_months=3,
                             betting_refit_months=3, n_paths=1000,
                             odds_type='open', train_size=.7,
                             alpha_history=None, risk_penalty=.25, saved_model_dir=None,
                             best_params_dir=None,
                             n_months=None, k=None):
    """Monthly expanding-history models, then delayed rolling betting searches.

    The first betting_lookback_months of TEST data supply calibration predictions
    only. Betting parameters update every betting_refit_months thereafter using
    the trailing lookback. Model predictions are cached once for the whole test
    interval and never regenerated by later models. Legacy n_months sets both
    betting intervals; legacy k specifies path count, never months.
    """
    if n_months is not None:
        betting_lookback_months = betting_refit_months = n_months
    if k is not None:
        n_paths = k
    if not np.isfinite(risk_penalty) or risk_penalty < 0:
        raise ValueError('risk_penalty must be finite and nonnegative')
    if any(not isinstance(value, int) or value < 1 for value in
           (model_refit_months, betting_lookback_months, betting_refit_months)):
        raise ValueError('Model and betting month intervals must be positive integers')
    if not isinstance(n_paths, int) or n_paths < 2:
        raise ValueError('n_paths must be an integer >= 2')
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be positive and finite')
    features = opening_features(features)
    data, split = prepare_history(frame, features=features, train_size=train_size)
    start = pd.Timestamp(test_start if test_start is not None else split['dates_test'].min()).normalize()
    end = pd.Timestamp(test_end if test_end is not None else data.date.max() + pd.Timedelta(days=1)).normalize()
    if start >= end:
        raise ValueError('test_start must precede test_end')
    test_origin = start
    betting_origin = test_origin + pd.DateOffset(months=betting_lookback_months)
    if betting_origin >= end:
        raise ValueError('Test interval must extend beyond the initial betting calibration holdout')
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    run_started_utc = pd.Timestamp.now(tz='UTC').isoformat()
    bundles = {}
    if saved_model_dir is not None and alpha_history is None:
        raise ValueError('saved_model_dir requires alpha_history')
    if alpha_history is None:
        alpha_history, bundles = search_alpha_history(
            data, start, end, output, n_months=model_refit_months, features=features,
            n_splits=n_splits, n_trials=model_trials, seed=seed,
        )
    alpha_history = validate_alpha_history(alpha_history)
    required_dates = pd.DatetimeIndex(list(calendar_boundaries(start, end, model_refit_months)))
    active_history = alpha_history.loc[
        (alpha_history.effective_date >= start) & (alpha_history.effective_date < end)]
    if (list(active_history.effective_date) != list(required_dates)
            or not active_history.training_cutoff.eq(active_history.effective_date).all()):
        raise ValueError('alpha_history must match every model refit boundary and its training cutoff; '
                         'regenerate alpha history for the new split/schedule')
    if saved_model_dir is not None:
        # Require every saved model; never silently refit a missing model.
        for boundary in required_dates:
            model_path = Path(saved_model_dir) / str(boundary.date()) / 'opening_model.joblib'
            if not model_path.is_file():
                raise FileNotFoundError(f'Saved opening model missing: {model_path}')
            bundles[boundary] = joblib.load(model_path)
    alpha_history.to_csv(output / 'alpha_history.csv', index=False)
    all_predictions = predict_from_alpha_history(
        data.loc[(data.date >= start) & (data.date < end)], data, alpha_history,
        odds_type=odds_type, features=features, bundles=bundles)
    model_periods = []
    for index, boundary in enumerate(required_dates):
        next_boundary = test_origin + pd.DateOffset(months=(index + 1) * model_refit_months)
        period_end = min(next_boundary, end)
        period = all_predictions.loc[(all_predictions.date >= boundary) &
                                     (all_predictions.date < period_end)]
        if period.empty:
            continue
        log_alpha_periods(period, bundles, odds_type, 'model validation', boundary, period_end)
        bundle = bundles[boundary]
        model_periods.append(dict(
            prediction_start=str(boundary.date()), prediction_end=str(period_end.date()),
            training_start=bundle['training_start'], training_cutoff=bundle['training_cutoff'],
            training_last_date=bundle['training_last_date'], training_rows=bundle['training_rows'],
            alpha=bundle['alpha'], cv_brier=bundle['cv_brier'], odds_type=odds_type,
            validation_brier=float(brier_score_loss(period.winner, period.proba_red))))
    calibration_predictions = all_predictions.loc[all_predictions.date < betting_origin].copy()
    (output / 'run_config.json').write_text(json.dumps(dict(
        odds_type=odds_type, training_odds_type='open',
        model_refit_months=model_refit_months, betting_lookback_months=betting_lookback_months,
        betting_refit_months=betting_refit_months, n_paths=n_paths,
        train_size=train_size, split_date=str(split['dates_test'].min().date()),
        test_start=str(start.date()), test_end=str(end.date()), seed=seed,
        betting_start=str(betting_origin.date()),
        initial_bankroll=initial_bankroll, features=features,
        objective='mean_terminal_return - risk_penalty * std_terminal_return',
        risk_penalty=risk_penalty,
        settlement='after_each_event', date_filter='date > 2010-02-26',
    ), indent=2), encoding='utf-8')
    summaries, event_tables, predictions, best_parameters = [], [], [], []
    bankroll = float(initial_bankroll)
    for index, start in enumerate(calendar_boundaries(betting_origin, end, betting_refit_months)):
        natural_end = betting_origin + pd.DateOffset(months=(index + 1) * betting_refit_months)
        stop = min(natural_end, end)
        previous_start = max(test_origin, start - pd.DateOffset(months=betting_lookback_months))
        tune_predictions = all_predictions.loc[
            (all_predictions.date >= previous_start) & (all_predictions.date < start)].copy()
        test_predictions = all_predictions.loc[
            (all_predictions.date >= start) & (all_predictions.date < stop)].copy()
        if tune_predictions.empty or test_predictions.empty:
            raise ValueError(f'Empty tuning or test interval for {start.date()}')
        window = output / str(start.date())
        window.mkdir(exist_ok=True)
        log_alpha_periods(tune_predictions, bundles, odds_type, 'betting tuning', previous_start, start)
        log_alpha_periods(test_predictions, bundles, odds_type, 'validation', start, stop)
        search = FrozenBettingSearch(tune_predictions, odds_type, k=n_paths,
                                     initial_bankroll=initial_bankroll, seed=seed,
                                     risk_penalty=risk_penalty)
        study = search.run(betting_trials, seed)
        params = study.best_params
        events = FrozenBettingSearch(test_predictions, odds_type, k=n_paths,
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
                       tuning_events=tune_predictions.date.nunique(), test_events=test_predictions.date.nunique(),
                       odds_type=odds_type, n_paths=n_paths, risk_penalty=risk_penalty,
                       tuning_objective_score=study.best_value,
                       tuning_mean_terminal_return=float(terminal.mean()),
                       tuning_std_terminal_return=float(terminal.std(ddof=1)),
                       test_brier=float(brier_score_loss(test_predictions.winner, test_predictions.proba_red)),
                       test_compounded_return=window_return, ending_bankroll=bankroll, **params)
        (window / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        best_parameters.append(dict(
            run_id=run_id, run_started_utc=run_started_utc,
            output_dir=str(output.resolve()), best_trial_number=study.best_trial.number,
            test_start=str(start.date()), test_end=str(stop.date()),
            tuning_start=str(previous_start.date()), odds_type=odds_type,
            betting_lookback_months=betting_lookback_months,
            betting_refit_months=betting_refit_months, betting_trials=betting_trials,
            n_paths=n_paths, seed=seed, initial_bankroll=initial_bankroll,
            risk_penalty=risk_penalty, score=study.best_value, **params))
        pd.DataFrame([best_parameters[-1]]).to_csv(window / 'best_params.csv', index=False)
        pd.DataFrame(best_parameters).to_csv(output / 'best_params.csv', index=False)
        tune_predictions.to_csv(window / 'tuning_predictions.csv', index=False)
        search.event_returns(params).to_csv(window / 'tuning_events.csv', index=False)
        np.savez_compressed(window / 'bootstrap_paths.npz', event_indices=search.path_indices, **paths)
        pd.DataFrame(dict(path_id=np.arange(n_paths), cumulative_return=terminal,
                          final_bankroll=paths['bankroll'][:, -1])).to_csv(
                              window / 'bootstrap_terminal_returns.csv', index=False)
        test_predictions.to_csv(window / 'test_predictions.csv', index=False)
        events.to_csv(window / 'test_events.csv', index=False)
        summaries.append(summary)
        event_tables.append(events)
        predictions.append(test_predictions)
    # Also persist opening models reconstructed from a supplied alpha history.
    for effective_date, bundle in bundles.items():
        directory = output / 'models' / str(effective_date.date())
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, directory / 'opening_model.joblib')
    result = dict(windows=pd.DataFrame(summaries), events=pd.concat(event_tables, ignore_index=True),
                  predictions=pd.concat(predictions, ignore_index=True), alpha_history=alpha_history,
                  best_params=pd.DataFrame(best_parameters), model_periods=pd.DataFrame(model_periods),
                  model_predictions=all_predictions, calibration_predictions=calibration_predictions)
    result['betting_data'], kelly_kwargs = kelly_simulation_input(
        result['predictions'], odds_type, initial_bankroll)
    for name, table in result.items():
        table.to_csv(output / f'{name}.csv', index=False)
    result['kelly_sim_kwargs'] = kelly_kwargs
    (output / 'kelly_sim_kwargs.json').write_text(json.dumps(kelly_kwargs, indent=2), encoding='utf-8')
    # One history CSV per odds type, retaining the winners from earlier runs.
    summary_dir = Path(best_params_dir) if best_params_dir is not None else output
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'{odds_type}_best_params.csv'
    winners = result['best_params']
    if summary_path.exists():
        winners = pd.concat([pd.read_csv(summary_path), winners], ignore_index=True)
    winners.to_csv(summary_path, index=False)
    return result


def run_opening_model(data=None, *, output_dir=DEFAULT_OUTPUT_DIR,
                      test_start=None, test_end=None, train_size=.7,
                      model_refit_months=1, model_trials=30, n_splits=15,
                      seed=42, features=None):
    """Search alpha and save expanding-history opening models, without betting.

    Outputs can be reused through run_search(opening_results_dir=output_dir).
    No training rows are exported.
    """
    if data is None:
        data = settings.data_dir / 'training_data' / 'entire_odds_stats_2026-03-09.csv'
    frame = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    features = opening_features(features)
    history, split = prepare_history(frame, features=features, train_size=train_size)
    start = pd.Timestamp(test_start if test_start is not None else split['dates_test'].min()).normalize()
    end = pd.Timestamp(test_end if test_end is not None else history.date.max() + pd.Timedelta(days=1)).normalize()
    output = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    alpha_history, bundles = search_alpha_history(
        history, start, end, output, n_months=model_refit_months,
        features=features, n_splits=n_splits, n_trials=model_trials, seed=seed)
    (output / 'run_config.json').write_text(json.dumps(dict(
        mode='model_only', odds_type='open', training_odds_type='open',
        train_size=train_size, split_date=str(split['dates_test'].min().date()),
        test_start=str(start.date()), test_end=str(end.date()),
        model_refit_months=model_refit_months, model_trials=model_trials,
        n_splits=n_splits, seed=seed, features=features,
        objective='mean_fold_brier', date_filter='date > 2010-02-26',
    ), indent=2), encoding='utf-8')
    return dict(alpha_history=alpha_history, models=bundles)


def run_search(data=None, *, alpha_history=None, opening_results_dir=None, **kwargs):
    """Run from Python using DataFrames or CSV paths; return all result tables.

    Additional keyword arguments go to run_delayed_walk_forward, e.g.
    odds_type, output_dir, betting_lookback_months, betting_refit_months,
    model_refit_months, n_paths, model_trials, and betting_trials.
    opening_results_dir reuses saved models without fitting or alpha search;
    its saved configuration supplies the model schedule and feature selection.
    """
    if opening_results_dir is not None:
        source = Path(opening_results_dir)
        fixed = {'train_size', 'test_start', 'test_end', 'model_refit_months',
                 'features', 'model_trials', 'n_splits', 'saved_model_dir'}
        conflicts = fixed.intersection(kwargs)
        if alpha_history is not None or conflicts:
            raise ValueError('With opening_results_dir, omit alpha_history and model settings: '
                             + ', '.join(sorted(conflicts)))
        destination = Path(kwargs.get('output_dir') or DEFAULT_OUTPUT_DIR)
        if destination.resolve() == source.resolve():
            raise ValueError('Use a separate output_dir to preserve the opening results')
        config = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
        if config['training_odds_type'] != 'open':
            raise ValueError('Saved models must have been trained on opening odds')
        for name in ('train_size', 'test_start', 'test_end', 'model_refit_months', 'features'):
            kwargs[name] = config[name]
        kwargs['saved_model_dir'] = source / 'models'
        kwargs.setdefault('best_params_dir', source)
        alpha_history = source / 'alpha_history.csv'
    if data is None:
        data = settings.data_dir / 'training_data' / 'entire_odds_stats_2026-03-09.csv'
    frame = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    if alpha_history is not None and not isinstance(alpha_history, pd.DataFrame):
        alpha_history = pd.read_csv(alpha_history, float_precision='round_trip')
    return run_delayed_walk_forward(frame, alpha_history=alpha_history, **kwargs)


# EDIT THESE SETTINGS, THEN RUN THIS FILE FROM YOUR EDITOR (no arguments needed).
# 'betting_only': concurrent open/close1/close2 searches using saved models;
# 'model_then_closing': opening models only, then both closing betting searches;
# 'model_only': opening models only; 'closing': saved models + closing searches;
# 'all': opening model/betting search followed by both closing searches.
EDITOR_MODE = 'betting_only'
OPEN_MODEL_SETTINGS = dict(
    data=settings.data_dir / 'training_data' / 'entire_odds_stats_2026-03-09.csv',
    output_dir=DEFAULT_OUTPUT_DIR,
    test_start=None,
    test_end=None,
    train_size=.7,
    model_refit_months=1,
    model_trials=30,
    n_splits=15,
    seed=42,
)

# Settings for the optional combined opening model + betting search.
RUN_SETTINGS = dict(
    data=settings.data_dir / 'training_data' / 'entire_odds_stats_2026-03-09.csv',
    output_dir=DEFAULT_OUTPUT_DIR,
    odds_type='open',                  # 'open', 'close1', or 'close2'
    alpha_history=None,                # DataFrame or saved alpha_history.csv path
    test_start=None,                   # None uses the event-aligned 70% split
    test_end=None,                     # None uses all remaining history
    train_size=.7,
    model_refit_months=1,
    betting_lookback_months=10,         # n: initial test holdout and trailing calibration history
    betting_refit_months=1,            # k: months between betting parameter searches
    n_paths=1000,
    risk_penalty=.6,                 # 0 = mean return only; larger = more volatility penalty
    model_trials=30,
    betting_trials=100,
    n_splits=15,
    initial_bankroll=500.0,
    seed=42,
)

# Opening run searches alpha and betting parameters.
OPEN_RUN_SETTINGS = {**RUN_SETTINGS, 'odds_type': 'open', 'alpha_history': None}
OPEN_BETTING_RUN_SETTINGS = dict(
    data=RUN_SETTINGS['data'],
    opening_results_dir=DEFAULT_OUTPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR / 'open',
    odds_type='open',
    betting_lookback_months=10,
    betting_refit_months=1,
    n_paths=1000,
    risk_penalty=.2,
    betting_trials=100,
    initial_bankroll=500.0,
    seed=42,
)
# Edit each closing search independently. Model settings come from the saved
# opening run; keep data pointed at the same historical input used by that run.
CLOSE1_RUN_SETTINGS = dict(
    data=RUN_SETTINGS['data'],
    opening_results_dir=DEFAULT_OUTPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR / 'close1',
    odds_type='close1',
    betting_lookback_months=10,
    betting_refit_months=1,
    n_paths=1000,
    risk_penalty=.35,
    betting_trials=100,
    initial_bankroll=500.0,
    seed=42,
)
CLOSE2_RUN_SETTINGS = dict(
    data=RUN_SETTINGS['data'],
    opening_results_dir=DEFAULT_OUTPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR / 'close2',
    odds_type='close2',
    betting_lookback_months=10,
    betting_refit_months=1,
    n_paths=1000,
    risk_penalty=.35,
    betting_trials=100,
    initial_bankroll=500.0,
    seed=42,
)


def run_all_betting_searches(open_settings=None, close1_settings=None, close2_settings=None):
    """Run three independent betting searches concurrently, using saved models only."""
    from concurrent.futures import ThreadPoolExecutor

    configurations = {
        'open': dict(OPEN_BETTING_RUN_SETTINGS if open_settings is None else open_settings),
        'close1': dict(CLOSE1_RUN_SETTINGS if close1_settings is None else close1_settings),
        'close2': dict(CLOSE2_RUN_SETTINGS if close2_settings is None else close2_settings),
    }
    outputs, sources = [], []
    for odds_type, config in configurations.items():
        if config.get('opening_results_dir') is None:
            raise ValueError(f'{odds_type} requires opening_results_dir; model training is disabled')
        if config.get('odds_type') != odds_type:
            raise ValueError(f'{odds_type} settings must use odds_type={odds_type!r}')
        outputs.append(Path(config.get('output_dir') or DEFAULT_OUTPUT_DIR).resolve())
        sources.append(Path(config['opening_results_dir']).resolve())
    if len(set(outputs)) != 3 or any(output in sources for output in outputs):
        raise ValueError('Use three distinct output folders, separate from saved opening results')
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {name: executor.submit(run_search, **config)
                   for name, config in configurations.items()}
        return {name: future.result() for name, future in futures.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Historical fight/features CSV')
    parser.add_argument('--test-start', help='Initial test/calibration start; defaults to the 70%% split')
    parser.add_argument('--test-end', help='Exclusive; defaults to day after final event')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR),
                        help='Export folder (default: Data/search_results)')
    parser.add_argument('--mode', choices=['both', 'alpha', 'betting'], default='both')
    parser.add_argument('--alpha-history', help='CSV: effective_date,training_cutoff,alpha; optional training_start')
    parser.add_argument('--odds-type', choices=ODDS_TYPES, default='open')
    parser.add_argument('--months', type=int, help='Legacy: sets both betting lookback and refit months')
    parser.add_argument('--model-refit-months', type=int, default=1)
    parser.add_argument('--betting-lookback-months', type=int, default=3)
    parser.add_argument('--betting-refit-months', type=int, default=3)
    parser.add_argument('--paths', type=int, default=1000)
    parser.add_argument('--risk-penalty', type=float, default=.25)
    parser.add_argument('--train-size', type=float, default=.7)
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
        run_opening_model(
            frame, output_dir=args.output_dir, test_start=args.test_start,
            test_end=args.test_end, train_size=args.train_size,
            model_refit_months=args.model_refit_months, model_trials=args.model_trials,
            n_splits=args.folds, seed=args.seed)
    else:
        history = pd.read_csv(args.alpha_history) if args.alpha_history else None
        run_delayed_walk_forward(
            frame, args.test_start, args.test_end, args.output_dir,
            model_trials=args.model_trials, betting_trials=args.betting_trials,
            n_splits=args.folds, initial_bankroll=args.bankroll, seed=args.seed,
            n_months=args.months, n_paths=args.paths, odds_type=args.odds_type,
            model_refit_months=args.model_refit_months,
            betting_lookback_months=args.betting_lookback_months,
            betting_refit_months=args.betting_refit_months,
            train_size=args.train_size, alpha_history=history, risk_penalty=args.risk_penalty,
        )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    elif EDITOR_MODE == 'betting_only':
        betting_results = run_all_betting_searches()
        open_results = betting_results['open']
        close1_results = betting_results['close1']
        close2_results = betting_results['close2']
        open_betting_data = open_results['betting_data']
        close1_betting_data = close1_results['betting_data']
        close2_betting_data = close2_results['betting_data']
    elif EDITOR_MODE == 'model_only':
        model_results = run_opening_model(**OPEN_MODEL_SETTINGS)
    elif EDITOR_MODE in ('model_then_closing', 'closing', 'all'):
        close1_settings = CLOSE1_RUN_SETTINGS.copy()
        close2_settings = CLOSE2_RUN_SETTINGS.copy()
        if EDITOR_MODE == 'model_then_closing':
            model_results = run_opening_model(**OPEN_MODEL_SETTINGS)
            # Reuse exactly the models and input data from the run just completed.
            for closing_settings in (close1_settings, close2_settings):
                closing_settings['opening_results_dir'] = OPEN_MODEL_SETTINGS['output_dir']
                closing_settings['data'] = OPEN_MODEL_SETTINGS['data']
        if EDITOR_MODE == 'all':
            results = run_search(**OPEN_RUN_SETTINGS)
            betting_data = results['betting_data']
        close1_results = run_search(**close1_settings)
        close2_results = run_search(**close2_settings)
        close1_betting_data = close1_results['betting_data']
        close2_betting_data = close2_results['betting_data']
    else:
        raise ValueError("EDITOR_MODE must be 'betting_only', 'model_then_closing', 'model_only', 'closing', or 'all'")
