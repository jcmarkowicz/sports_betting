"""Read-only upcoming wager previews based on settled pre-event histories."""
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.Scaling import calculate_event_trends, apply_event_filters

FILTERS = {
    'Raw': ('raw', 'open'),
    'Flat': ('flat_only', 'open'),
    'Brier': ('prior_brier_only', 'open'),
    'Vol': ('volatility_only', 'open'),
    'Combined': ('combined', 'open'),
    'Brier · C2': ('prior_brier_only', 'close2'),
    'Vol · C2': ('volatility_only', 'close2'),
    'Combined · C2': ('combined', 'close2'),
}


def event_decision(event_date, label, sources):
    strategy,source=FILTERS[label]
    if strategy=='raw':
        return dict(multiplier=1.,trend='Not applied',source_date=None,status='Original wagers')
    event=pd.DataFrame({'date':[pd.Timestamp(event_date)]})
    opening=sources['open']
    gate=calculate_event_trends(event,opening['event_history'],trend_scale=opening['trend_scale']).iloc[0]
    base=dict(trend=gate.trend,source_date=gate.source_date)
    if gate.trend=='Unknown':
        return dict(**base,multiplier=None,status='Insufficient trend history; preview unavailable')
    if gate.trend=='Flat':
        return dict(**base,multiplier=0.,status='Skipped: opening trend is flat')
    if strategy=='flat_only':return dict(**base,multiplier=1.,status='Opening trend retained')
    history=sources[source]
    featured=calculate_event_trends(event,history['event_history'],trend_scale=history['trend_scale'])
    if not featured.trend_ready.iloc[0]:
        return dict(**base,multiplier=None,status='Insufficient scaler-source trend history')
    # Use component multipliers, not that source's gate: every choice uses opening gate.
    scaled=apply_event_filters(featured,history['historical_events']).iloc[0]
    vol=scaled.volatility_multiplier;brier=scaled.brier_multiplier
    multiplier={'volatility_only':vol,'prior_brier_only':brier,'combined':vol*brier}[strategy]
    fallback=[]
    if strategy in ['volatility_only','combined'] and not scaled.volatility_ready:fallback.append('volatility')
    if strategy in ['prior_brier_only','combined'] and not scaled.brier_ready:fallback.append('Brier')
    status='Reduction applied' if multiplier<1 else 'Original stake retained'
    if fallback:status+='; insufficient '+ '/'.join(fallback)+' history: that scaler stays at 1'
    return dict(**base,multiplier=float(multiplier),status=status)


def wager_tables(moneylines, parlays, decisions):
    """Preserve unknown stakes as NaN; each parlay ticket stake is counted once."""
    ml=pd.DataFrame(index=moneylines.index)
    ml['Fight']=moneylines.fighter_red.fillna('?')+' vs '+moneylines.fighter_blue.fillna('?')
    tickets=[];summary=[]
    for market,title in [('open','Opening'),('close1','Close1'),('close2','Close2')]:
        decision=decisions[market];factor=decision['multiplier']
        stakes=pd.to_numeric(moneylines[f'stake_{market}'],errors='coerce')
        if stakes.dropna().lt(0).any():raise ValueError('Published wager amounts must be nonnegative')
        ml[f'{title} pick']=moneylines[f'pred_name_{market}']
        ml[f'{title} wager']=stakes*factor if factor is not None else np.nan
        names=parlays[f'choice_fighter_name_{market}'].dropna().astype(str).tolist()
        values=pd.to_numeric(parlays[f'stake_{market}'],errors='coerce').dropna()
        if len(values) and (values.lt(0).any() or not np.allclose(values,values.iloc[0])):
            raise ValueError(f'{title}: expected repeated stake for one parlay ticket')
        ticket_stake=float(values.iloc[0]) if len(values) else np.nan
        adjusted=ticket_stake*factor if factor is not None else np.nan
        tickets.append(dict(Market=title,Selections=' + '.join(names) if names else None,Wager=adjusted))
        adjusted_ml=ml[f'{title} wager']
        # A total is unknown if any underlying quote/stake is unavailable.
        total=adjusted_ml.sum()+adjusted if adjusted_ml.notna().all() and pd.notna(adjusted) else np.nan
        summary.append(dict(Market=title,Filter=decision['label'],Multiplier=factor,
            Trend=decision['trend'],Total_wager=total,Status=decision['status']))
    return ml,pd.DataFrame(tickets),pd.DataFrame(summary)
