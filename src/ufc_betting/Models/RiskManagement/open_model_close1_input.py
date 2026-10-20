"""Compatibility imports; implementation lives in Convergence."""
from ufc_betting.BettingStrategy.Convergence.model_input import (
    predict_probabilities, MODEL_COLUMN_ALIASES,
)
from ufc_betting.BettingStrategy.model_statistics import align_model_columns


def _align_model_design(scaled, features, model_names):
    """Compatibility adapter for the shared column alignment helper."""
    return align_model_columns(scaled, model_names, fallback=features,
                               aliases=MODEL_COLUMN_ALIASES)


def opening_model_at_close1(frame=None, prices=None, model=None, scaler=None):
    """Legacy adapter; new callers should use predict_probabilities directly."""
    import pandas as pd
    from ufc_betting.config import config, settings
    from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
    if frame is None:
        frame = pd.read_csv(settings.data_dir / 'model_results/test_logit_close1.csv')
    data = replace_opening_odds(frame, 'close1', prices=prices)
    return predict_probabilities(
        data, config.model_open_path if model is None else model,
        scaler_path=config.scaler_open_path if scaler is None else scaler,
    )
