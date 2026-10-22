"""Read supplied validation probabilities without loading or fitting models."""

import numpy as np
import pandas as pd


def read_frozen_predictions(data):
    """Accept red probabilities or chosen-side probabilities with a binary side.

    Canonical fields: proba_red, risk_red_proba, prediction_se.
    Alternatives: choice_proba, choice_fighter_bool (or pred_winner),
    optional risk_choice_proba, and se (or proba_se). choice_proba takes
    precedence over proba_red when supplied. Risk probabilities default to the
    supplied betting probabilities; explicit risk columns remain separate.
    """
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(
        data, float_precision='round_trip').copy()
    if 'prediction_se' not in frame:
        for name in ('se', 'proba_se'):
            if name in frame:
                frame['prediction_se'] = frame[name]
                break
    if 'risk_red_proba' not in frame and 'risk_proba_red' in frame:
        frame['risk_red_proba'] = frame['risk_proba_red']
    if 'choice_proba' in frame:
        side = 'choice_fighter_bool' if 'choice_fighter_bool' in frame else 'pred_winner'
        if side not in frame or not frame[side].isin([0, 1]).all():
            raise ValueError('Choice probabilities require choice_fighter_bool or pred_winner (0=blue, 1=red)')
        frame['proba_red'] = np.where(frame[side] == 1, frame.choice_proba, 1 - frame.choice_proba)
        if 'risk_red_proba' not in frame and 'risk_choice_proba' in frame:
            frame['risk_red_proba'] = np.where(
                frame[side] == 1, frame.risk_choice_proba, 1 - frame.risk_choice_proba)
    if 'risk_red_proba' not in frame and 'proba_red' in frame:
        frame['risk_red_proba'] = frame['proba_red']
    missing = {'proba_red', 'risk_red_proba', 'prediction_se'} - set(frame.columns)
    if missing:
        raise ValueError(f'Frozen validation predictions require columns: {sorted(missing)}')
    return frame
