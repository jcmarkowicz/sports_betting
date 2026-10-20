"""Shared market-price substitution for models trained on opening features."""

import numpy as np

from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.odds_features import devig_two_way


def replace_opening_odds(frame, odds_type='close1', *, prices=None, features=None):
    """Copy selected-market inputs into the opening model's feature columns.

    Example: inputs = replace_opening_odds(frame, 'close2')

    prices optionally supplies an (n_fights, 2) decimal array ordered red, blue.
    Decimal prices, when available, are de-vigged using the power method.
    Otherwise existing market feature columns are used. features lists model
    features to replace; by default only proba_fair_open_diff is replaced.
    Returns a copy; does not fit models or produce predictions.
    """
    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('odds_type must be open, close1, or close2')
    result = frame.copy()
    columns = [f'dec_{odds_type}_red', f'dec_{odds_type}_blue']
    if prices is not None or all(name in result for name in columns):
        values = np.asarray(prices if prices is not None else result[columns], dtype=float)
        if values.shape != (len(result), 2) or not np.isfinite(values).all() or (values <= 1).any():
            raise ValueError('Prices must be aligned red/blue decimal odds, finite and greater than 1')
        fair = np.asarray([devig_two_way(red, blue, method='power')[:2]
                           for red, blue in values]).reshape(-1, 2)
        result[columns] = values
        for index, side in enumerate(('red', 'blue')):
            result[f'proba_fair_{odds_type}_{side}'] = fair[:, index]
            result[f'dec_fair_{odds_type}_{side}'] = 1 / fair[:, index]
        result[f'proba_fair_{odds_type}_diff'] = fair[:, 0] - fair[:, 1]
    features = ['proba_fair_open_diff'] if features is None else list(features)
    # Read from a snapshot so assigning opening columns cannot affect later reads.
    source = result.copy()
    for name in features:
        if 'open' in name:
            replacement = name.replace('open', odds_type)
            if replacement not in source:
                raise ValueError(f'Missing {odds_type} replacement for {name}')
            result[name] = source[replacement]
    return result
