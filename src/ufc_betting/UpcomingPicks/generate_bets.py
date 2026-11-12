"""Upcoming picks from the opening inference model and saved policies."""
from pathlib import Path

import joblib
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline, seperate_bets_dfs


def generate_bets(df, select_odds=None, return_event_ev=False, *,
                  comparison_dir=None, fit_mode='trainval', params_dir=None,
                  betting_params=None, bankroll=None, use_decay=True):
    """Use the opening inference fit at each market's odds, matching Event Scaling.

    Live predictions default to train+validation fits. Set fit_mode='train'
    to use the train-only artifacts used by the validation parameter search.
    Saved policies are loaded on each call; no models or policies are fitted.
    """
    if fit_mode not in ('train', 'trainval'):
        raise ValueError('fit_mode must be train or trainval')
    if select_odds is not None and (isinstance(select_odds, bool) or select_odds not in (0, 1, 2)):
        raise ValueError('select_odds must be None, 0, 1, or 2')
    root = Path(comparison_dir) if comparison_dir is not None else settings.data_dir / 'model_comparisons/lr_open_close'
    stages = ['open', 'close1', 'close2']
    models, scalers, encoders, features = [], [], [], []
    frame = df.copy()
    for stage in stages:
        directory = root / fit_mode / 'open'
        # model_se is the inference probability model, not just an SE companion.
        model = sm.load(directory / 'model_se.pkl')
        scaler = joblib.load(directory / 'scaler.pkl')
        encoder = joblib.load(directory / 'encoder.pkl')
        models.append(model)
        scalers.append(scaler)
        encoders.append(encoder)
        features.append(list(scaler.feature_names_in_) + list(encoder.feature_names_in_))
        for column in encoder.feature_names_in_:
            if column in frame:
                frame[column] = frame[column].astype('category')

    result = betting_pipeline(
        upcoming_df=frame, feats_list=features, model_list=models,
        scaler_list=scalers, encoder_list=encoders, type_list=stages,
        fair_odds_list=[[f'dec_fair_{stage}_blue', f'dec_fair_{stage}_red'] for stage in stages],
        real_odds_list=[[f'dec_{stage}_blue', f'dec_{stage}_red'] for stage in stages],
        bankroll=config.bankroll if bankroll is None else bankroll,
        model_se_list=models, params_dir=params_dir, betting_params=betting_params,
        use_decay=use_decay, return_event_ev=return_event_ev, opening_model=True,
    )
    moneylines, parlays = result[:2]
    if select_odds is not None:
        singles, tickets = seperate_bets_dfs(moneylines, parlays, stages)
        moneylines, parlays = singles[select_odds], tickets[select_odds]
    return (moneylines, parlays, result[2]) if return_event_ev else (moneylines, parlays)
