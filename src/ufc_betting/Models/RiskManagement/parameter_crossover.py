"""Compare fixed betting policies on identical opening-to-closing prices."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache, event_values
from ufc_betting.Models.RiskManagement.parlay_ev_target_bands import evaluate_parlays


def compound(returns, bankroll=500., floor=100.):
    path = [bankroll]
    for r in returns:
        bankroll *= 1+r if bankroll >= floor else 1
        path.append(bankroll)
    return np.asarray(path)


def winning_ranges(x, difference):
    """Contiguous sampled ranges, without assuming monotonicity or exact roots."""
    ids = np.flatnonzero(np.asarray(difference) > 1e-8)
    return [(float(x[g[0]]),float(x[g[-1]])) for g in
            np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(g)]


def plot_parameter_crossover(opening_params, closing_params, closing_odds='close1', steps=100, output_dir=None):
    if closing_odds not in ('close1','close2') or not isinstance(steps,int) or steps < 1:
        raise ValueError('Select close1/close2 and a positive integer steps')
    policies = {name: BettingParams.from_value(p) for name,p in
                [('opening',opening_params),('closing',closing_params)]}
    root = settings.data_dir/'model_results'
    keys = ['date','fighter_red','fighter_blue']
    op = pd.read_csv(root/'test_logit_open.csv').sort_values(keys).reset_index(drop=True)
    cl = pd.read_csv(root/f'test_logit_{closing_odds}.csv').sort_values(keys).reset_index(drop=True)
    if op.duplicated(keys).any() or cl.duplicated(keys).any() or not op[keys+['winner']].equals(cl[keys+['winner']]):
        raise ValueError('Fight cohorts must match')
    if not cl.winner.isin([0,1]).all():
        raise ValueError('Only settled binary outcomes supported')
    a = op[['dec_open_red','dec_open_blue']].to_numpy(float)
    b = cl[[f'dec_{closing_odds}_red',f'dec_{closing_odds}_blue']].to_numpy(float)
    if not np.isfinite([a,b]).all() or min(a.min(),b.min()) <= 1:
        raise ValueError('Invalid prices')
    model,scaler = sm.load(config.model_open_path),joblib.load(config.scaler_open_path)
    groups = list(cl.groupby('date',sort=True))
    records,totals = [],[]
    for shift in np.linspace(0,1,steps+1):
        data = opening_model_at_close1(cl,(1-shift)*a+shift*b,model,scaler)
        for policy,params in policies.items():
            cache = fight_cache(data,params)
            returns = {s:[] for s in ('moneyline','parlay','combined')}
            for date,group in groups:
                selected = cache[group.index.to_numpy()][None,:,:]
                ml = event_values(selected,group,params,False)
                kw = dict(mdd_parlay=params.mdd_parlay,N_parlay=params.N_parlay,z=params.z,validate=False)
                parlay = evaluate_parlays(selected,group,**kw)
                combined = evaluate_parlays(selected,group,combined=True,**kw)
                for strategy,values in zip(returns,(ml,parlay,combined)):
                    ev,realized,exposure = [float(v[0]) for v in values]
                    returns[strategy].append(realized)
                    records.append(dict(shift_pct=100*shift,policy=policy,strategy=strategy,date=date,
                        expected_return_pct=100*ev,realized_return_pct=100*realized,exposure_pct=100*exposure))
            for strategy,r in returns.items():
                path = compound(r)
                totals.append(dict(shift_pct=100*shift,policy=policy,strategy=strategy,
                    ending_bankroll=path[-1],compounded_return_pct=100*(path[-1]/500-1),
                    max_drawdown_pct=100*np.max(1-path/np.maximum.accumulate(path))))
    totals,events = pd.DataFrame(totals),pd.DataFrame(records)
    ranges = {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import Image,display
    fig,axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for j,strategy in enumerate(('moneyline','parlay','combined')):
        table = totals.loc[totals.strategy.eq(strategy)].pivot(index='shift_pct',columns='policy',values='ending_bankroll')
        difference = table.closing-table.opening
        ranges[strategy] = winning_ranges(table.index.to_numpy(),difference.to_numpy())
        for policy in ('opening','closing'):
            axes[0,j].plot(table.index,table[policy],label=f'{policy.title()} parameters')
        axes[0,j].set(title=strategy.title(),ylabel='Ending bankroll ($; log scale)',yscale='log')
        axes[0,j].legend(fontsize=8)
        axes[1,j].plot(table.index,difference,color='#6b4091')
        axes[1,j].axhline(0,color='black',lw=1)
        axes[1,j].fill_between(table.index,0,difference,where=difference>0,color='green',alpha=.2)
        axes[1,j].set(xlabel=f'Movement from opening toward {closing_odds} (%)',ylabel='Closing minus opening ending bankroll ($)')
        for ax in axes[:,j]:ax.grid(alpha=.2)
    fig.suptitle(f'Fixed parameter comparison | same prices for both policies | {len(groups)} test events | $500 start')
    fig.supxlabel('0% = opening prices; 100% = historical closing prices. Model weights and both parameter sets stay fixed.\nGreen: closing parameters finish ahead. Shared sizing for combined/parlay; singles-only sizing for moneylines. Stop below $100. Historical crossover, not a validated live threshold.',fontsize=9)
    out = Path(output_dir) if output_dir else root/'notebook_odds_sensitivity'/'parameter_crossover'/closing_odds
    out.mkdir(parents=True,exist_ok=True)
    path = out/'parameter_crossover.png'
    fig.savefig(path,dpi=150);plt.close(fig)
    totals.to_csv(out/'compounded_results.csv',index=False)
    events.to_csv(out/'event_returns.csv',index=False)
    metadata = dict(closing_odds=closing_odds,parameters={k:asdict(v) for k,v in policies.items()},
        closing_wins_sampled_ranges=ranges,grid_step_pct=100/steps,
        interpretation='Ranges are grid observations, not exact crossing estimates; no future-performance claim. Fixed opening model; uniform decimal interpolation; no parameter or stake blending.')
    (out/'methodology.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    display(Image(filename=str(path)))
    for strategy,zones in ranges.items():
        print(f'{strategy}: closing parameters ahead on sampled ranges: {zones or "none"}')
    return dict(summary=totals,events=events,closing_wins_ranges=ranges,plot_path=path)
