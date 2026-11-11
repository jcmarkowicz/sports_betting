from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/ufc_betting').is_dir())
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from ufc_betting.config import settings
from ufc_betting.Results.test_simulation import simulate_saved_predictions
PRED_ROOT=settings.data_dir/'model_comparisons/lr_open_close/report_predictions'
PARAMS_DIR=settings.data_dir/'kelly_params/inference_market_models'
OUTPUT_DIR=settings.data_dir/'backtest_results/model_comparisons/closing_filters'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
FIGHT_WINDOW=100
FLAT_BAND=.03
EWMA_SPAN=5
SLOPE_EVENTS=5
VOL_HISTORY=10
BRIER_WARMUP=20
BRIER_RIDGE=10.
INITIAL_BANKROLL=500.
KEYS=['date','fighter_red','fighter_blue']

def score_predictions(frame,market):
    frame=frame.copy();frame['date']=pd.to_datetime(frame.date).dt.normalize()
    p=pd.to_numeric(frame.proba_red);r=pd.to_numeric(frame[f'dec_{market}_red']);b=pd.to_numeric(frame[f'dec_{market}_blue'])
    valid=frame.winner.isin([0,1])&p.between(0,1)&np.isfinite(r)&np.isfinite(b)&r.gt(1)&b.gt(1)
    q=(1/r)/(1/r+1/b)
    frame=frame.loc[valid].copy()
    frame['vegas_actual']=((q-frame.winner)**2).loc[valid]
    frame['vegas_expected']=(q*(1-q)).loc[valid]
    frame['model_brier']=((p-frame.winner)**2).loc[valid]
    frame['excess']=frame.vegas_actual-frame.vegas_expected
    frame['advantage']=frame.vegas_actual-frame.model_brier
    return frame.sort_values('date',kind='stable').reset_index(drop=True)

def closing_market_data(market):
    parts=[];curves=[];sources=[]
    original_train=pd.read_csv(PRED_ROOT/f'train/open/inference/train_with_{market}.csv',parse_dates=['date'])
    training_end=original_train.date.max()
    for label,fit,split in [('Validation','train','val'),('Test','trainval','test')]:
        base=PRED_ROOT/fit/'open/inference'
        prediction_path=base/f'{split}_with_{market}.csv'
        # Regenerate with one consistent saved parameter source; never retune.
        ml,pl,*_=simulate_saved_predictions(market,INITIAL_BANKROLL,use_decay=True,split=split,
            model_type='open',predictions_path=prediction_path,params_dir=PARAMS_DIR)
        tag=f'{fit}_{split}_open_with_{market}_inference_decay'
        ml.to_csv(OUTPUT_DIR/f'{tag}_moneylines.csv',index=False)
        pl.to_csv(OUTPUT_DIR/f'{tag}_parlays.csv',index=False)
        prediction=pd.read_csv(prediction_path,parse_dates=['date'])
        ml['date']=pd.to_datetime(ml.date).dt.normalize()
        prediction['date']=prediction.date.dt.normalize()
        check=ml.merge(prediction[KEYS+['proba_red']],on=KEYS,how='outer',validate='one_to_one',indicator=True,suffixes=('','_reference'))
        assert check['_merge'].eq('both').all()
        saved_red=check.proba_red if 'proba_red' in ml else check.choice_proba.where(check.pred_winner.eq(1),1-check.choice_proba)
        reference=check.proba_red_reference if 'proba_red_reference' in check else check.proba_red
        np.testing.assert_allclose(saved_red,reference,atol=1e-10)
        warm=score_predictions(pd.read_csv(base/f'train_with_{market}.csv'),market)
        current=score_predictions(ml[KEYS].merge(prediction,on=KEYS,validate='one_to_one',sort=False),market)
        assert warm.date.max()<current.date.min()
        history=pd.concat([warm,current],ignore_index=True).sort_values('date',kind='stable').reset_index(drop=True)
        for col in ['vegas_actual','vegas_expected','excess','advantage']:
            history['rolling_'+col]=history[col].rolling(FIGHT_WINDOW,min_periods=FIGHT_WINDOW).mean()
        endpoints=history.groupby('date').rolling_advantage.last().dropna()
        smooth=endpoints.ewm(span=EWMA_SPAN,adjust=False).mean()
        t=np.arange(SLOPE_EVENTS,dtype=float);t-=t.mean()
        slope=smooth.rolling(SLOPE_EVENTS).apply(lambda x:np.dot(t,x)/np.dot(t,t),raw=True)
        sd=smooth.loc[smooth.index<=training_end].std(ddof=1)
        features=pd.DataFrame({'prior_advantage':endpoints,'normalized_slope':slope/sd,
            'prior_brier':history.groupby('date').vegas_actual.mean().reindex(endpoints.index)})
        features['source_date']=features.index
        features=features.shift(1)
        cols=['bankroll_postevent','event_payout_money_line','parlay_net','event_total_fstar']
        assert ml.groupby('date')[cols].nunique(dropna=False).le(1).all().all()
        event=ml.groupby('date')[cols].first().sort_index()
        start=event.bankroll_postevent.shift(1);start.iloc[0]=INITIAL_BANKROLL
        assert start.gt(0).all()
        np.testing.assert_allclose(event.bankroll_postevent-start,
            event.event_payout_money_line+event.parlay_net,rtol=1e-7,atol=1e-6)
        event['return_pct']=100*(event.bankroll_postevent/start-1)
        event=event.join(features)
        event['five_year_mean']=[endpoints.loc[(endpoints.index>=d-pd.DateOffset(years=5))&(endpoints.index<d)].mean() for d in event.index]
        assert event[['prior_advantage','normalized_slope','prior_brier','source_date','five_year_mean']].notna().all().all()
        assert (event.source_date<event.index).all()
        event['regime']=np.where(event.prior_advantage>event.five_year_mean,'Above mean','At/below mean')
        event['trend']=np.select([event.normalized_slope>FLAT_BAND,event.normalized_slope < -FLAT_BAND],['Uptrend','Downtrend'],default='Flat')
        event['split']=label;parts.append(event)
        if label=='Validation':
            display=history.copy();display['split']=np.where(display.date<=training_end,'Train','Validation')
        else:
            display=history.loc[history.date>=event.index.min()].copy();display['split']='Test'
        curves.append(display)
        sources.append({'market':market,'split':label,'model_type':'open','probability_model':'inference','fit':fit,'events':len(event),'prediction_path':str(prediction_path)})
    return pd.concat(parts).sort_index(),pd.concat(curves,ignore_index=True),pd.DataFrame(sources)

def closing_filters(events):
    # Full event history supplies volatility, including flat events; performance excludes flats.
    active=events.loc[events.trend.ne('Flat')].copy();rows=[]
    for date,row in active.iterrows():
        previous=events.loc[events.index<date]
        target=previous.loc[previous.regime.eq('Above mean')&previous.trend.ne('Flat')].tail(VOL_HISTORY)
        own=previous.loc[previous.regime.eq(row.regime)].tail(VOL_HISTORY)
        vol=1.
        if len(target)==VOL_HISTORY and len(own)==VOL_HISTORY:
            denom=own.return_pct.std(ddof=1)
            vol=min(1.,target.return_pct.std(ddof=1)/denom) if denom>1e-12 else 1.
        past=active.loc[active.index<date]
        brier=1.;expected=baseline=np.nan
        if len(past)>=BRIER_WARMUP:
            x=past.prior_brier.to_numpy();y=past.return_pct.to_numpy()
            baseline=float(y.mean());expected=baseline;sd=x.std(ddof=0)
            if sd>1e-12:
                z=(x-x.mean())/sd
                beta=max(0.,np.dot(z,y-baseline)/(np.dot(z,z)+BRIER_RIDGE))
                expected=baseline+beta*(row.prior_brier-x.mean())/sd
            brier=float(np.clip(expected/baseline,0,1)) if baseline>1e-12 else (0. if expected<=0 else 1.)
        rows.append({'date':date,'brier_multiplier':brier,'vol_multiplier':vol,
            'combined_multiplier':brier*vol,'expected_return_pct':expected,'baseline_return_pct':baseline})
    active=active.join(pd.DataFrame(rows).set_index('date'))
    for name,factor in [('Prior Brier reductions','brier_multiplier'),('Volatility only','vol_multiplier'),('Combined','combined_multiplier')]:
        assert active[factor].between(0,1).all()
        active[name]=active.return_pct*active[factor]
        assert (active.event_total_fstar*active[factor]<=1+1e-10).all()
    active['Original (flat excluded)']=active.return_pct
    return active

def closing_metrics(returns):
    r=np.asarray(returns,float)/100
    if (r<=-1).any():raise ValueError('Bankruptcy requires explicit handling')
    wealth=np.r_[1.,np.cumprod(1+r)]
    return {'events':len(r),'geometric_growth_pct':100*np.expm1(np.log1p(r).mean()),
        'mean_return_pct':100*r.mean(),'volatility_pp':100*r.std(ddof=1),'worst_event_pct':100*r.min(),
        'max_drawdown_pct':-100*(wealth/np.maximum.accumulate(wealth)-1).min(),'wealth_multiple':wealth[-1]}

closing_results={}
for market in ['close1','close2']:
    events,curve,sources=closing_market_data(market)
    active=closing_filters(events)
    # Changing future returns must not change already-computed scaling decisions.
    probe=events.copy();boundary=active.index[len(active)//2]
    probe.loc[probe.index>=boundary,'return_pct']+=1
    check=closing_filters(probe)
    np.testing.assert_allclose(active.loc[:boundary,['brier_multiplier','vol_multiplier']],check.loc[:boundary,['brier_multiplier','vol_multiplier']],equal_nan=True)
    fig,axes=plt.subplots(3,1,figsize=(15,12),sharex=True,constrained_layout=True)
    for split,part in curve.groupby('split',sort=False):
        axes[0].plot(part.date,part.rolling_vegas_actual,color='tab:blue',label='Actual Vegas Brier' if split=='Train' else None)
        axes[0].plot(part.date,part.rolling_vegas_expected,color='tab:orange',label='Expected Vegas Brier: p(1-p)' if split=='Train' else None)
        axes[1].plot(part.date,part.rolling_excess,color='tab:purple')
        axes[2].plot(part.date,part.rolling_advantage,label=split)
    axes[0].set(title=f'{market}: opening inference model, 100-fight Brier averages',ylabel='Brier score');axes[0].legend()
    axes[1].set(title='Actual minus expected Vegas Brier',ylabel='Excess Brier');axes[1].axhline(0,color='gray',lw=.7)
    axes[2].set(title='Rolling advantage: Vegas Brier minus model Brier',ylabel='100-fight advantage',xlabel='Event date');axes[2].axhline(0,color='gray',lw=.7);axes[2].legend()
    for ax in axes:
        ax.axvline(events.index.min(),color='gray',ls='--')
        ax.axvline(events.loc[events.split.eq('Test')].index.min(),color='black',ls=':');ax.grid(alpha=.2)
    plt.show()
    strategies=['Original (flat excluded)','Prior Brier reductions','Volatility only','Combined']
    records=[]
    for scope,part in [('Combined periods',active),*list(active.groupby('split',sort=False))]:
        for strategy in strategies:records.append({'market':market,'scope':scope,'strategy':strategy,**closing_metrics(part[strategy])})
    metrics=pd.DataFrame(records)
    print(sources.to_string(index=False));print(f'{market}: {len(events)} total events; {len(active)} retained; {len(events)-len(active)} flat excluded.')
    print(metrics.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    fig,axes=plt.subplots(2,1,figsize=(15,9),sharex=True,constrained_layout=True)
    for strategy in strategies:
        axes[0].plot(active.index,(1+active[strategy]/100).cumprod(),label=strategy)
    axes[0].set_yscale('log');axes[0].set(title=f'{market}: filtered linked strategy wealth',ylabel='Wealth from 1 unit (log scale)');axes[0].legend()
    for col,label in [('brier_multiplier','Prior Brier'),('vol_multiplier','Volatility'),('combined_multiplier','Combined')]:axes[1].plot(active.index,active[col],label=label,alpha=.8)
    axes[1].set(xlabel='Event date',ylabel='Stake multiplier',ylim=(0,1.05));axes[1].legend()
    for ax in axes:ax.axvline(events.loc[events.split.eq('Test')].index.min(),color='black',ls=':');ax.grid(alpha=.2)
    plt.show()
    closing_results[market]={'events':events,'curve':curve,'filtered':active,'metrics':metrics,'sources':sources}
    active.to_csv(OUTPUT_DIR/f'{market}_filtered_event_returns.csv')
    metrics.to_csv(OUTPUT_DIR/f'{market}_filter_metrics.csv',index=False)
