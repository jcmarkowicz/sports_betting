"""Historical per-fighter odds changes versus changes in allocated Kelly stake."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache


def fighter_stakes(data, params):
    """Final singles stakes, including the existing backtest risk adjustments."""
    cache = fight_cache(data, params)
    stakes = np.zeros((len(data), 2))  # blue, red; unselected fighter has zero stake
    for _, group in data.groupby('date', sort=True):
        ids = group.index.to_numpy()
        weights = cache[ids, 0]
        raw = np.sum(weights * cache[ids, 3])
        conservative = np.sum(weights * cache[ids, 4])
        uncertainty = 1. if params.z is None else (np.clip(conservative/raw, 0, 1) if raw > 0 else 0.)
        final = weights * uncertainty / max(weights.sum()*uncertainty, 1)
        stakes[ids, data.pred_winner.iloc[ids].to_numpy(int)] = final
    return stakes


def plot_line_movement_kelly(opening_params, closing_params, output_dir=None):
    """Plot decimal-odds change against change in bankroll percentage wagered."""
    params = [BettingParams.from_value(p) for p in (opening_params, closing_params)]
    root = settings.data_dir/'model_results'
    keys = ['date','fighter_red','fighter_blue']
    opening = pd.read_csv(root/'test_logit_open.csv').sort_values(keys).reset_index(drop=True)
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    op_prices = opening[['dec_open_red','dec_open_blue']].to_numpy(float)
    op_data = opening_model_at_close1(opening,op_prices,model,scaler)
    op_stakes = fighter_stakes(op_data,params[0])
    rows=[]
    for stage in ('close1','close2'):
        closing = pd.read_csv(root/f'test_logit_{stage}.csv').sort_values(keys).reset_index(drop=True)
        if opening.duplicated(keys).any() or closing.duplicated(keys).any() or not opening[keys].equals(closing[keys]):
            raise ValueError('Historical fight cohorts must match')
        prices = closing[[f'dec_{stage}_red',f'dec_{stage}_blue']].to_numpy(float)
        data = opening_model_at_close1(closing,prices,model,scaler)
        stakes = fighter_stakes(data,params[1])
        for side,code,column in [('red',1,0),('blue',0,1)]:
            part = opening[keys].copy()
            part['fighter'] = part[f'fighter_{side}']
            part['closing_odds'] = stage
            part['movement'] = prices[:,column]-op_prices[:,column]
            part['opening_fraction'] = op_stakes[:,code]
            part['closing_fraction'] = stakes[:,code]
            part['kelly_change_pp'] = 100*(stakes[:,code]-op_stakes[:,code])
            rows.append(part)
    results=pd.concat(rows,ignore_index=True)
    # A fighter who receives no stake in either snapshot adds no allocation information.
    plotted=results.loc[(results.opening_fraction>0)|(results.closing_fraction>0)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import Image,display
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,constrained_layout=True)
    for ax,negative in zip(axes,(True,False)):
        for stage,color in [('close1','#2366af'),('close2','#cf7626')]:
            g=plotted.loc[plotted.closing_odds.eq(stage)&(plotted.movement.lt(0) if negative else plotted.movement.gt(0))]
            ax.scatter(g.movement,g.kelly_change_pp,s=20,alpha=.5,color=color,label=f'{stage}: {len(g)}')
        ax.axhline(0,color='black',lw=1)
        ax.set(title='Odds decreased from opening' if negative else 'Odds increased from opening',
               xlabel='Change in decimal odds (closing − opening)',
               ylabel='Change in percent of bankroll wagered (percentage points)')
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Change in decimal odds versus change in percent wagered')
    fig.supxlabel('Same fighter compared at both snapshots; unselected/filtered stake = 0. Existing singles backtest sizing; no blending.\nOnly fighters staked at either endpoint; unchanged prices omitted. Two snapshots identify increases/decreases, not movement back toward opening.',fontsize=9)
    out=Path(output_dir) if output_dir else root/'notebook_odds_sensitivity'/'line_movements'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'line_movement_kelly.png'
    fig.savefig(path,dpi=150);plt.close(fig)
    results.to_csv(out/'line_movement_kelly.csv',index=False)
    display(Image(filename=str(path)))
    return results
