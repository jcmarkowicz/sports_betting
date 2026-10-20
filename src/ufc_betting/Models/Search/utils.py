from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
import numpy as np 
import pandas as pd 
import statsmodels.api as sm

import joblib
from pathlib import Path 

from ufc_betting.config import settings, config 
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.odds_features import devig_two_way

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ODDS_TYPES = config.odds_types 

def discover_alpha_history(search_dir=None):
    """Combine prior runs' alpha histories; conflicting selections need an explicit file."""
    root = Path(search_dir) if search_dir is not None else settings.data_dir / 'search_results'
    histories = []
    for path in sorted(root.rglob('alpha_history.csv')):
        history = validate_alpha_history(pd.read_csv(path, float_precision='round_trip'))
        history['alpha_source'] = str(path)
        histories.append(history)
    if not histories:
        return pd.DataFrame(columns=['effective_date', 'training_cutoff', 'alpha'])
    combined = pd.concat(histories, ignore_index=True)
    for date, group in combined.groupby('effective_date'):
        columns = ['alpha', 'training_cutoff']
        if 'training_start' in combined:
            columns.append('training_start')
        if len(group[columns].drop_duplicates()) > 1:
            raise ValueError(f'Conflicting saved alphas for {date.date()}; supply --alpha-history')
    return validate_alpha_history(combined.drop_duplicates('effective_date'))

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


def load_validation_predictions(input_path, odds_type='close1', *, alpha_history=None,
                                training_input=None, search_dir=None,
                                use_monthly_models=True, model_refit_months=1):
    """Recompute predictions using the notebook's saved preprocessing artifacts.

    The input is an already selected validation cohort; every row is retained.
    By default discover other runs' monthly alphas and refit when available.
    alpha_history accepts a DataFrame or CSV path; an empty DataFrame means
    no monthly selections. Set use_monthly_models=False for saved-model only.
    Close prices replace only the opening market-probability feature, as in
    the notebook. Inference probabilities and SE use the separate saved model.
    """
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    frame = pd.read_csv(input_path)
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
    frame = frame.sort_values('date', kind='stable').reset_index(drop=True)
    keys = ['date', 'fighter_red', 'fighter_blue']
    if frame.empty or frame.date.isna().any() or frame.duplicated(keys).any():
        raise ValueError('Validation requires nonempty, dated, unique fights')

    # Fail rather than silently removing rows from the requested full cohort.
    training = pd.read_csv(settings.data_dir / 'model_results/train_logit_open.csv')
    training['date'] = pd.to_datetime(training.date).dt.normalize()
    if not frame.merge(training[keys], on=keys, how='inner').empty:
        raise ValueError('Validation contains opening-model training fights')

    model = sm.load(config.model_open_path)
    inference = sm.load(config.model_open_se_path)
    scaler = joblib.load(config.scaler_open_path)
    encoder = joblib.load(config.encoder_open_path)
    features = replace_opening_odds(frame, odds_type)[list(config.open_feats)]
    numeric = list(scaler.feature_names_in_)
    categorical = list(encoder.feature_names_in_)
    scaled = pd.DataFrame(scaler.transform(features[numeric]),
                          columns=numeric, index=frame.index)
    encoded = pd.DataFrame(encoder.transform(features[categorical]),
                           columns=encoder.get_feature_names_out(categorical),
                           index=frame.index)
    design = sm.add_constant(pd.concat([scaled, encoded], axis=1),
                             has_constant='add')
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError('Validation model features must be finite')
    frame['proba_red'] = np.asarray(model.predict(design[model.model.exog_names]))
    prediction = inference.get_prediction(design[inference.model.exog_names]).summary_frame()
    frame['risk_red_proba'] = prediction['predicted'].to_numpy()
    frame['prediction_se'] = prediction['se'].to_numpy()
    frame['proba_blue'] = 1 - frame.proba_red
    frame['pred_winner'] = (frame.proba_red >= .5).astype(int)
    frame['odds_type'] = odds_type
    frame['prediction_source'] = 'saved_open_logit'
    if use_monthly_models:
        if alpha_history is None:
            alpha_history = discover_alpha_history(search_dir)
        elif not isinstance(alpha_history, pd.DataFrame):
            path = Path(alpha_history)
            alpha_history = pd.read_csv(path, float_precision='round_trip')
            alpha_history['alpha_source'] = str(path)
        training = pd.DataFrame()
        if not alpha_history.empty:
            training_path = (Path(training_input) if training_input is not None else
                             settings.data_dir / 'training_data/entire_odds_stats_2026-03-07.csv')
            training, _ = prepare_history(pd.read_csv(training_path))
        frame = apply_monthly_logit_models(frame, training, alpha_history, odds_type,
                                           model_refit_months=model_refit_months)
    return frame

def opening_features(features=None):
    features = list(config.open_feats if features is None else features)
    if not features or any('close1' in name or 'close2' in name for name in features):
        raise ValueError('Model training accepts opening features only')
    return features


def prepare_history(frame, *, features=None, train_size=.7):
    features = opening_features(features)
    builder = TrainTestBuilder(
        frame, features, target_col='winner', date_col='date', odds_type='open',
        year=2010, month=2, day=26, required_features_only=True,
    )
    split = builder.prepare_train_test(train_size, scale=False, group_dates=True)
    return split['filtered_df'], split


def apply_monthly_logit_models(predictions, training_data, alpha_history, odds_type='close1',
                               *, model_refit_months=1):
    """Refit in every Nth calendar month, anchored to the first validation month.

    Fits each eligible selection once, before betting trials, using the saved
    training interval. Hold that model through the remaining months of its
    interval. Use the first saved selection in each scheduled month, only
    from its effective date onward. Missing selections (and events before
    availability) use the opening fallback. Fit failures are surfaced.
    """
    if (isinstance(model_refit_months, bool)
            or not isinstance(model_refit_months, (int, np.integer))
            or model_refit_months < 1):
        raise ValueError('model_refit_months must be a positive integer')
    data = predictions.copy()
    data['date'] = pd.to_datetime(data.date).dt.normalize()
    data['prediction_source'] = 'saved_open_logit'
    data['model_alpha'] = np.nan
    data['alpha_effective_date'] = pd.NaT
    data['model_training_cutoff'] = pd.NaT
    data['alpha_source'] = ''
    if alpha_history.empty:
        return data.sort_values('date', kind='stable').reset_index(drop=True)
    history = validate_alpha_history(alpha_history)
    assignments = np.full(len(data), -1, dtype=int)
    origin = data.date.min().to_period('M')
    selected_months = set()
    for index, row in history.iterrows():
        month = row.effective_date.to_period('M')
        offset = month.ordinal - origin.ordinal
        if offset < 0 or offset % model_refit_months or month in selected_months:
            continue
        selected_months.add(month)
        end = (month + int(model_refit_months)).start_time
        eligible = (data.date >= row.effective_date) & (data.date < end)
        assignments[eligible.to_numpy()] = index
    parts = [data.loc[assignments == -1].copy()]
    train = training_data.copy()
    train['date'] = pd.to_datetime(train.date).dt.normalize()
    for index in np.unique(assignments[assignments >= 0]):
        row = history.iloc[index]
        mask = train.date < row.training_cutoff
        if 'training_start' in history:
            mask &= train.date >= row.training_start
        print(f'{row.effective_date.date()}: refitting opening Logit with saved alpha '
              f'{row.alpha:.6g} on {int(mask.sum())} prior fights', flush=True)
        bundle = fit_opening_bundle(train.loc[mask], row.training_cutoff, row.alpha)
        part = predict_odds(data.loc[assignments == index], bundle, odds_type)
        part['prediction_source'] = 'refitted_monthly_open_logit'
        part['alpha_effective_date'] = row.effective_date
        part['model_training_cutoff'] = row.training_cutoff
        part['alpha_source'] = row.get('alpha_source', 'supplied_alpha_history')
        parts.append(part)
    return pd.concat([part for part in parts if not part.empty], ignore_index=True).sort_values(
        'date', kind='stable').reset_index(drop=True)


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


def make_preprocessor(frame):
    categorical = list(frame.select_dtypes(include=['category', 'object', 'bool', 'str']).columns)
    numeric = [name for name in frame if name not in categorical]
    return ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore',
                                     sparse_output=False), categorical),
    ], sparse_threshold=0)


def predict_odds(frame, bundle, odds_type='open'):
    """Selected market prices enter opening-feature slots; never retrain here."""
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    result = frame.copy().reset_index(drop=True)
    result['date'] = pd.to_datetime(result.date).dt.normalize()
    if (result.date < pd.Timestamp(bundle['training_cutoff'])).any():
        raise ValueError('Cannot use a model on dates before its training cutoff')
    inputs = replace_opening_odds(result, odds_type, features=bundle['features'])
    # Keep execution-market fields in the output without overwriting opening history.
    for name in inputs:
        if odds_type in name and name.startswith(('dec_', 'proba_fair_')):
            result[name] = inputs[name]
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


def fair_probabilities(prices):
    return np.asarray([devig_two_way(red, blue, method="power")[:2]
                       for red, blue in prices])

def training_frame(frame, cutoff, features):
    data = frame.copy()
    data['date'] = pd.to_datetime(data.date, errors='raise').dt.normalize()
    data = data.sort_values('date', kind='stable').reset_index(drop=True)
    if data.empty or data.date.isna().any() or (data.date >= pd.Timestamp(cutoff)).any():
        raise ValueError('All training rows must precede the model training cutoff')
    if not data.winner.isin([0, 1]).all() or data[features].isna().any().any():
        raise ValueError('Training requires complete opening features and binary outcomes')
    return data


def opening_feature_frame(frame, features):
    X = frame[list(features)].copy()
    for name in ('math_red', 'math_blue', 'elo_pred', 'womens_fight'):
        if name in X:
            X[name] = X[name].astype('category')
    return X


def fit_model(X, y, alpha):
    model = sm.Logit(np.asarray(y), sm.add_constant(X, has_constant='add')).fit_regularized(
        method='l1', alpha=alpha, disp=False, maxiter=1000,
    )
    if not model.mle_retvals.get('converged', True):
        raise ValueError('Opening Logit did not converge')
    return model