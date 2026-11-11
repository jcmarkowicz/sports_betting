"""Apply pre-event scaling overlays to saved live return fractions.

Warmup and validation history are supplied by the caller. Live history always
supersedes overlapping historical observations. No files are written.
"""
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.Scaling import (
    ScalingConfig, calculate_brier_scores, calculate_event_history,
    fit_trend_scale, calculate_event_trends, apply_event_filters,
)
from ufc_betting.Results.returns import prepare_return_plot_data


def live_source_features(moneylines, parlays, balances, market, warmup_fights,
                         historical_events, training_end):
    """Use saved live probabilities for this market, not backtest predictions."""
    events, ml, _ = prepare_return_plot_data(moneylines, parlays, balances, market)
    events=events.sort_values('date').set_index('date')
    live=ml.copy()
    live['proba_red']=live[f'choice_proba_{market}'].where(live[f'pred_winner_{market}'].eq(1),1-live[f'choice_proba_{market}'])
    live=live.drop_duplicates(['date','fighter_red','fighter_blue'],keep='last')
    warm=warmup_fights.copy();warm['date']=pd.to_datetime(warm.date).dt.normalize()
    warm=warm.loc[warm.date<events.index.min()]
    scored=calculate_brier_scores(pd.concat([warm,live],ignore_index=True),odds_type=market)
    history=calculate_event_history(scored)
    scale=fit_trend_scale(history,training_end=training_end)
    target=pd.DataFrame({'return_pct':100*events.bankroll_pct_change})
    featured=calculate_event_trends(target,history,trend_scale=scale)
    historical=historical_events.copy()
    if 'date' not in historical:historical['date']=historical.index
    historical=historical.loc[pd.to_datetime(historical.date)<events.index.min()]
    past=pd.concat([historical,featured],sort=False)
    # Stake columns in historical frames are irrelevant to the target overlay.
    applied=apply_event_filters(featured,past)
    return dict(features=featured,scalers=applied,events=events,
                event_history=history,trend_scale=scale,historical_events=past)


def scaled_live_paths(events, opening, closing=None, *, closing_market="close2"):
    """Raw plus requested gates/scalers; all flat gates use opening history."""
    events=events.sort_index()
    dates=events.index
    pd.testing.assert_index_equal(dates,opening['features'].index)
    keep=opening['features'].trend.isin(['Uptrend','Downtrend'])
    paths={'Raw':pd.Series(1.,index=dates),'Flat only | gate open':keep.astype(float)}
    sources=[('open',opening)] + ([(closing_market,closing)] if closing is not None else [])
    for label,source in sources:
        pd.testing.assert_index_equal(dates,source['scalers'].index)
        scalers=source['scalers']
        for strategy,factor in [('Prior-Brier','brier_multiplier'),('Volatility','volatility_multiplier')]:
            paths[f'{strategy} | gate open, scalers {label}']=scalers[factor].where(keep,0.)
        paths[f'Combined | gate open, scalers {label}']=(scalers.brier_multiplier*scalers.volatility_multiplier).where(keep,0.)
    multipliers=pd.DataFrame(paths)
    returns=multipliers.mul(events.bankroll_pct_change,axis=0)
    wealth=(1+returns).cumprod()
    winner=wealth.iloc[-1].idxmax()  # Stable tie: raw, flat, then displayed scaler order.
    return dict(multipliers=multipliers,returns=returns,wealth=wealth,winner=winner,keep=keep)


def reconstruct_scaled_events(events, multiplier, initial):
    """Recompound from initial balance, no deposits; scale ML/parlay equally."""
    out=events.sort_values('date').copy().set_index('date')
    factor=multiplier.reindex(out.index)
    if factor.isna().any() or not factor.between(0,1).all():raise ValueError('Invalid scaling alignment')
    before=out.bankroll_postevent-out.event_payout_money_line-out.parlay_net
    ml=np.divide(out.event_payout_money_line,before,out=np.zeros(len(out)),where=before>0)
    pl=np.divide(out.parlay_net,before,out=np.zeros(len(out)),where=before>0)
    returns=out.bankroll_pct_change*factor
    wealth=(1+returns).cumprod()
    new_before=initial*wealth.shift(1,fill_value=1.)
    out['event_payout_money_line']=new_before*ml*factor
    out['parlay_net']=new_before*pl*factor
    out['bankroll_postevent']=initial*wealth
    out['bankroll_pct_change']=returns
    np.testing.assert_allclose(out.bankroll_postevent-new_before,out.event_payout_money_line+out.parlay_net,atol=1e-8)
    return out.reset_index()
