from ufc_betting.BettingStrategy.betting_params import BettingParams
import numpy as np 
from ufc_betting.config import config 

ODDS_TYPES = config.odds_types

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


def portable_kelly_input(betting_data, kwargs, best_params):
    """Bind price aliases to the dataset and fixed betting controls to kwargs.

    Copies inputs. Preserves original market columns and best_* provenance;
    disables row parameter overrides so swapping kwargs actually swaps controls.
    """
    data, args = betting_data.copy(), dict(kwargs)
    for argument, prefix in (('fair_decimal_cols', 'bet_fair_decimal'),
                             ('real_decimal_cols', 'bet_real_decimal')):
        names = [f'{prefix}_blue', f'{prefix}_red']
        data[names] = data[args[argument]].to_numpy(copy=True)
        args[argument] = names
    args.update(BettingParams.from_value(best_params).to_simulation_kwargs())
    args.update(use_walk_forward_params=False, adjust_mdd_by_edge=False)
    args.pop('monthly_params', None)
    return data, args