"""Load saved fight-level paths without resampling their cards or predictions."""
from pathlib import Path

import pandas as pd
import numpy as np


def load_sampled_paths(value, *, n_paths=None, random_state=None):
    """Accept a list of DataFrames or a trusted pandas pickle containing it.

    Each frame is one path. Dates and sampled_event_id must identify the same
    cards one-to-one. Donor repetitions across cards/paths are allowed.
    n_paths selects whole paths uniformly without replacement before copying
    and validating them. None retains every path in its original order.
    """
    from .compact_paths import CompactPaths
    if isinstance(value, (str, Path)) and Path(value).is_dir():
        value = CompactPaths(value)
    if isinstance(value, (str, Path)):
        value = pd.read_pickle(value)  # Only load files from a trusted source.
    if not isinstance(value, (list, tuple, CompactPaths)) or not len(value):
        raise ValueError('sampled_paths must be a nonempty list of DataFrames or its pickle path')
    if n_paths is not None:
        if (isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer))
                or not 1 <= n_paths <= len(value)):
            raise ValueError(f'n_paths must be an integer between 1 and {len(value)}')
        indices = np.random.default_rng(random_state).choice(len(value), size=n_paths, replace=False)
        value = [value[i] for i in indices]
    paths = []
    for number, frame in enumerate(value):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f'Sampled path {number} must be a nonempty DataFrame')
        if not {'date', 'sampled_event_id'} <= set(frame):
            raise ValueError('Sampled paths require date and sampled_event_id columns')
        frame = frame.copy().reset_index(drop=True)
        frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
        if frame[['date', 'sampled_event_id']].isna().any().any():
            raise ValueError('Sampled path card identifiers must not be missing')
        cards = frame[['date', 'sampled_event_id']].drop_duplicates()
        if cards.date.duplicated().any() or cards.sampled_event_id.duplicated().any():
            raise ValueError('Each sampled_event_id must map to exactly one distinct date within a path')
        order = ['date'] + (['bout_position'] if 'bout_position' in frame else [])
        paths.append(frame.sort_values(order, kind='stable').reset_index(drop=True))
    return paths


def predict_intermediate_paths(paths, *, prediction_model, prediction_model_type,
                               prediction_scalers, prediction_encoders,
                               prediction_se_model=None, search_market='close1'):
    """Infer on already sampled prices; never sample, fit or save anything.

    search_market is a downstream Kelly-search label, selected only at evaluation
    time. Intermediate execution prices are also placed in close1 columns for
    the existing simulator. The stored original paths are never modified.
    """
    from .market_policy import load_predictor, predict_sampled_prices
    from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
    from ufc_betting.BettingStrategy.kelly_scaling import decimal_to_american

    if search_market not in ('close1', 'close2'):
        raise ValueError('search_market must be close1 or close2 for intermediate execution')
    predictor = load_predictor(prediction_model, prediction_model_type,
                               prediction_scalers, prediction_encoders, prediction_se_model)
    outputs = []
    for frame in load_sampled_paths(paths):
        columns = ['dec_intermediate_red', 'dec_intermediate_blue']
        if not set(columns) <= set(frame):
            raise ValueError('Paths must contain saved dec_intermediate_red/blue prices')
        prices = frame[columns].to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 1).any():
            raise ValueError('Intermediate decimal prices must be finite and greater than 1')
        for market in ('close1', 'close2'):
            for side in ('red', 'blue'):
                recorded = f'recorded_{market}_{side}'
                if recorded not in frame:
                    raise ValueError(f'Missing original endpoint: {recorded}')
                # Reset previous evaluation overrides before switching models.
                frame[f'dec_{market}_{side}'] = frame[recorded]
        frame = frame.drop(columns=[c for c in (
            'risk_proba_red', 'risk_proba_blue', 'risk_red_proba', 'risk_choice_proba',
            'prediction_se', 'se', 'choice_proba', 'choice_fighter_bool', 'odds_type',
            'correct_pred', 'prob_winner',
        ) if c in frame])
        frame = predict_sampled_prices(frame, prices, predictor, prediction_model_type)
        if search_market != 'close1':
            frame = replace_opening_odds(frame, search_market, prices=prices, features=[])
        for stage in {'close1', search_market}:
            for index, side in enumerate(('red', 'blue')):
                frame[f'{stage}_{side}'] = [decimal_to_american(x) for x in prices[:, index]]
        frame['execution_odds_type'] = 'intermediate'
        frame.attrs['generation'] = dict(frame.attrs.get('generation', {}),
                                         odds_type=search_market, sample_prices=True)
        outputs.append(frame)
    return outputs


def attach_path_predictions(paths, predictions, *, odds_type, n_paths=None, random_state=None):
    """Attach a chosen prediction set to fixed donor fights without resampling.

    Match the original date and oriented fighter names, not the synthetic date
    or CSV row order. Every sampled donor must have exactly one prediction.
    Returns new frames; neither saved paths nor prediction inputs are changed.
    Only recorded-price paths are supported; sampled prices need fresh inference.
    n_paths optionally samples whole paths without replacement; random_state
    makes the selection reproducible across prediction sets. None uses all paths.
    """
    from ufc_betting.Models.Search.prediction_input import read_frozen_predictions

    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('odds_type must be open, close1, or close2')
    predictions = read_frozen_predictions(predictions)
    keys = ['date', 'fighter_red', 'fighter_blue']
    if not set(keys) <= set(predictions):
        raise ValueError(f'Prediction matching requires {keys}')
    predictions['date'] = pd.to_datetime(predictions.date, errors='raise').dt.normalize()
    if predictions[keys].isna().any().any() or predictions.duplicated(keys).any():
        raise ValueError('Prediction fight keys must be nonmissing and unique')
    for col in ('proba_red', 'risk_red_proba', 'prediction_se'):
        values = predictions[col].to_numpy(float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or (col != 'prediction_se' and (values > 1).any())):
            raise ValueError(f'Invalid prediction values: {col}')
    # Full inference exports may contain hundreds of unrelated feature columns.
    # Reindex only prediction fields and the prices/outcomes checked below.
    lookup_columns = keys + ['proba_red', 'risk_red_proba', 'prediction_se']
    lookup_columns += [col for col in (
        'winner', *(f'{prefix}_{odds_type}_{side}'
                    for prefix in ('dec', 'dec_fair') for side in ('red', 'blue'))
    ) if col in predictions]
    lookup = predictions[lookup_columns].set_index(keys)
    outputs = []
    for frame in load_sampled_paths(paths, n_paths=n_paths, random_state=random_state):
        if (frame.attrs.get('generation', {}).get('sample_prices', False)
                or ('execution_odds_type' in frame and frame.execution_odds_type.eq('intermediate').any())):
            raise ValueError('Sampled-price paths require predictions for their sampled prices')
        if not {'source_date', 'fighter_red', 'fighter_blue'} <= set(frame):
            raise ValueError('Paths require source_date and fighter names for prediction matching')
        donor_keys = pd.MultiIndex.from_arrays([
            pd.to_datetime(frame.source_date).dt.normalize(), frame.fighter_red, frame.fighter_blue,
        ], names=keys)
        missing = ~donor_keys.isin(lookup.index)
        if missing.any():
            raise ValueError(f'Missing predictions for sampled fights: {donor_keys[missing].unique().tolist()[:5]}')
        matched = lookup.reindex(donor_keys)
        if 'winner' in matched and not np.array_equal(matched.winner.to_numpy(), frame.winner.to_numpy()):
            raise ValueError('Prediction outcomes do not match sampled donor outcomes')
        for prefix in ('dec', 'dec_fair'):
            for side in ('red', 'blue'):
                col = f'{prefix}_{odds_type}_{side}'
                values = frame[col].to_numpy(float)
                if not np.isfinite(values).all() or (values <= 1).any():
                    raise ValueError(f'Invalid path prices: {col}')
                if col in matched and not np.allclose(values, matched[col].to_numpy(float), rtol=1e-9, atol=1e-10):
                    raise ValueError(f'Prediction prices do not match sampled donor prices: {col}')
        # Remove aliases which otherwise take precedence over the chosen set.
        frame = frame.drop(columns=[col for col in (
            'choice_proba', 'risk_choice_proba', 'choice_fighter_bool', 'se',
            'correct_pred', 'prob_winner',
        ) if col in frame])
        for col in ('proba_red', 'risk_red_proba', 'prediction_se'):
            frame[col] = matched[col].to_numpy()
        frame['proba_blue'] = 1 - frame.proba_red
        frame['risk_proba_red'] = frame.risk_red_proba
        frame['risk_proba_blue'] = 1 - frame.risk_red_proba
        frame['proba_se'] = frame.prediction_se
        frame['pred_winner'] = (frame.proba_red >= .5).astype(int)
        frame['odds_type'] = odds_type
        frame['execution_odds_type'] = odds_type
        frame.attrs['generation'] = dict(frame.attrs.get('generation', {}),
                                         odds_type=odds_type, sample_prices=False)
        outputs.append(frame)
    return outputs


def simulate_sampled_predictions(paths, odds_type, *, controls, policies,
                                 use_decay, initial_bankroll, batch_size,
                                 return_details, progress_every, return_event_exposure=False):
    """Evaluate stored predictions on stored cards; never sample or infer again."""
    from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
    from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
    from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
    from ufc_betting.BettingStrategy.kelly_scaling import decimal_to_american
    from .mc_engine import settle_bankroll_paths
    all_ml, all_parlay, all_bankroll, details = [], [], [], []
    exposures = []
    prepared = []
    fast = policies is not None and not return_details
    for path_id, path in enumerate(paths):
        data = read_frozen_predictions(path)
        for column in ('proba_red', 'risk_red_proba'):
            values = data[column].to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f'Invalid saved probabilities: {column}')
        errors = data.prediction_se.to_numpy(float)
        if not np.isfinite(errors).all() or (errors < 0).any():
            raise ValueError('Saved prediction standard errors must be finite and nonnegative')
        if 'odds_type' in data and not data.odds_type.eq(odds_type).all():
            raise ValueError('Saved prediction odds_type differs from requested odds_type')
        if odds_type == 'intermediate':
            if 'execution_odds_type' not in data or not data.execution_odds_type.eq('intermediate').all():
                raise ValueError('Saved intermediate paths require intermediate prices/predictions '
                                 'and execution_odds_type="intermediate"')
            stage = 'close1'
            # Fresh prediction paths retain the actual sampled prices. Refuse
            # stale execution aliases rather than silently settling at endpoints.
            sampled_columns = ['dec_intermediate_red', 'dec_intermediate_blue']
            if any(column in data for column in sampled_columns):
                if not all(column in data for column in sampled_columns):
                    raise ValueError('Both saved intermediate price columns are required')
                execution = data[['dec_close1_red', 'dec_close1_blue']].to_numpy(float)
                if not np.allclose(execution, data[sampled_columns].to_numpy(float), rtol=1e-12, atol=1e-12):
                    raise ValueError('Execution odds differ from saved intermediate prices; regenerate predictions')
        else:
            stage = odds_type
        data['proba_blue'] = 1 - data.proba_red
        data['pred_winner'] = (data.proba_red >= .5).astype(int)
        data['proba_se'] = data.prediction_se
        data['risk_proba_blue'] = 1 - data.risk_red_proba
        for side in ('red', 'blue'):
            if f'open_{side}' not in data:
                data[f'open_{side}'] = data[f'dec_open_{side}'].map(decimal_to_american)
        if not data.winner.isin([0, 1, 2, 3]).all():
            raise ValueError('Saved prediction paths require settled outcomes (0/1 or 2/3 for void)')
        if policies is not None:
            for market in ('close1', 'close2'):
                for side in ('red', 'blue'):
                    recorded = f'recorded_{market}_{side}'
                    if recorded not in data:
                        if odds_type == 'intermediate':
                            raise ValueError('Intermediate policy selection requires recorded closing prices')
                        data[recorded] = data[f'dec_{market}_{side}']
        for prefix in ('dec', 'dec_fair'):
            for side in ('red', 'blue'):
                values = data[f'{prefix}_{stage}_{side}'].to_numpy(float)
                if not np.isfinite(values).all() or (values <= 1).any():
                    raise ValueError('Saved execution/fair prices must be finite and greater than 1')
                data[f'{prefix}_close1_{side}'] = values
        data['execution_odds_type'] = odds_type
        if fast:
            prepared.append(data)
            continue
        ml, pl, event_rows, kelly_frames, parlay_frames = [], [], [], [], []
        for date, group in data.groupby('date', sort=True):
            if policies is not None:
                kelly, parlay = BettingReturnCalculator.calculate_market_event_returns(
                    group, policies, use_decay=use_decay, calc_parlay=controls['calc_parlay'])
            else:
                kwargs = dict(controls)
                if kwargs.get('decay_rates') is not None:
                    kwargs['opening_data'] = group
                kelly, parlay = simulate_kelly(group,
                    prob_cols=['proba_blue', 'proba_red'],
                    fair_decimal_cols=['dec_fair_close1_blue', 'dec_fair_close1_red'],
                    real_decimal_cols=['dec_close1_blue', 'dec_close1_red'],
                    risk_prob_cols=['risk_proba_blue', 'risk_red_proba'],
                    pred_winner_col='pred_winner', date_col='date',
                    prediction_se_col='proba_se', init_bankroll=1., **kwargs)
            first = kelly.iloc[0]
            ml.append(float(first.event_payout_money_line))
            pl.append(float(first.get('parlay_net', 0.)))
            event_rows.append(dict(date=date, sampled_event_id=group.sampled_event_id.iloc[0],
                                   moneyline_return=ml[-1], parlay_return=pl[-1],
                                   exposure=float(first.event_total_fstar)))
            if return_details:
                kelly_frames.append(kelly)
                parlay_frames.append(parlay)
        families = dict(moneyline=np.asarray([ml]), parlay=np.asarray([pl]))
        settled = settle_bankroll_paths(families['moneyline'] + families['parlay'], initial_bankroll,
                                        family_returns=families, batch_size=batch_size)
        all_ml.append(settled['moneyline_profits'][0])
        all_parlay.append(settled['parlay_profits'][0])
        all_bankroll.append(settled['bankroll'][0])
        if return_event_exposure:
            active = np.repeat(settled['bankroll'][0, :-1] > 0, batch_size)[:len(event_rows)]
            exposures.append(np.where(active, [row['exposure'] for row in event_rows], 0.))
        if return_details:
            details.append(dict(inputs=data, kelly=pd.concat(kelly_frames, ignore_index=True),
                                parlay=pd.concat(parlay_frames, ignore_index=True),
                                event_returns=pd.DataFrame(event_rows)))
        if progress_every and (path_id == 0 or (path_id + 1) % progress_every == 0 or path_id + 1 == len(paths)):
            print(f'Saved fight paths {path_id + 1}/{len(paths)} evaluated', flush=True)
    if fast:
        from .batched_market_paths import simulate_market_paths
        return simulate_market_paths(prepared, policies, use_decay=use_decay,
            calc_parlay=controls['calc_parlay'], initial_bankroll=initial_bankroll,
            batch_size=batch_size, progress_every=progress_every, return_event_exposure=return_event_exposure)
    result = (all_ml, all_parlay, all_bankroll)
    if return_event_exposure:
        metadata = dict(event_exposure=exposures, event_counts=[len(e) for e in exposures], batch_size=batch_size)
        if return_details:
            metadata['details'] = details
        return (*result, metadata)
    return (*result, details) if return_details else result
