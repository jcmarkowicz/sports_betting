"""Search two moneyline allocation-decay rates on bootstrapped event paths.

Both fixed policies are evaluated at the SAME historical execution prices and
probabilities. Opening prices only measure movement. No parlay, odds shifting,
parameter fitting occurs. The decay blend precedes z correction and the exposure cap.

Example using full-validation run_all_odds results (no model refit needed)::

    search = MoneylineDecaySearch(
        betting_data=results['close1']['betting_data'],
        opening_data=results['open']['betting_data'],
        opening_params=results['open']['best_params'],
        closing_params=results['close1']['best_params'],
        odds_type='close1', risk_penalty=.4, k=1000,
    )
    result = search.run(n_trials=100, output_dir='Data/model_results/decay_close1')

This is a tuning result on the supplied events, not held-out validation.
Use separate earlier/later cohorts to evaluate generalization.
"""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    bootstrap_event_indices, simulate_bankroll_paths, terminal_return_score,
)


def decay_weights(opening, current, k_down, k_up):
    if any(not np.isfinite(v) or v < 0 for v in (k_down,k_up)):
        raise ValueError('Decay rates must be finite and nonnegative')
    opening,current=np.asarray(opening,float),np.asarray(current,float)
    if opening.shape != current.shape or not np.isfinite([opening,current]).all() or (opening<=1).any() or (current<=1).any():
        raise ValueError('Matched finite decimal prices greater than 1 required')
    movement=np.log(current/opening)
    return np.exp(-np.where(movement<0,k_down,k_up)*np.abs(movement))


def fixed_kwargs(params):
    p=asdict(params) if is_dataclass(params) else dict(params)
    # Accept BettingParams or the best_params returned by run_all_odds.
    lower=p.get('moneyline_min_american',p.get('min_moneyline'))
    return dict(max_drawdown=p['mdd'],N=int(p['N']),z=p['z'],
        moneyline_american_bounds=None if lower is None else (lower,p.get('moneyline_max_american',np.inf)),
        adjust_mdd_by_edge=False,calc_parlay=False,calc_heavy_favorite_parlay=False,
        use_walk_forward_params=False,bankroll_floor=None)


def suggest_decay_rates(trial, names, max_decay):
    """Uniform log1p coordinates resolve small rates and include exact zero."""
    rates={name:float(np.expm1(trial.suggest_float('log1p_'+name,0.,float(np.log1p(max_decay))))) for name in names}
    for name,value in rates.items():trial.set_user_attr(name,value)
    return rates


def decode_decay_rates(params):
    return {name.removeprefix('log1p_'):float(np.expm1(value)) for name,value in params.items()}


class MoneylineDecaySearch:
    def __init__(self,betting_data,opening_data,opening_params,closing_params,
                 odds_type='close1',reference_odds_type='open',*,k=1000,
                 initial_bankroll=500.,risk_penalty=.25,seed=42):
        if odds_type not in ('open','close1','close2') or reference_odds_type not in ('open','close1','close2'):
            raise ValueError('Unknown odds type')
        if not np.isfinite(risk_penalty) or risk_penalty<0 or not np.isfinite(initial_bankroll) or initial_bankroll<=0:
            raise ValueError('Invalid bankroll or risk penalty')
        keys=['date','fighter_red','fighter_blue']
        frame=betting_data.copy();reference=opening_data.copy()
        for data in (frame,reference):
            data['date']=pd.to_datetime(data.date,errors='raise').dt.normalize()
            if data.empty or data.date.isna().any() or data.duplicated(keys).any():
                raise ValueError('Need nonempty dated unique fights')
        frame=frame.sort_values(keys,kind='stable').reset_index(drop=True)
        cols=[f'dec_{reference_odds_type}_{s}' for s in ('blue','red')]
        ref=reference[keys+cols].rename(columns={c:'reference_'+c for c in cols})
        frame=frame.merge(ref,on=keys,how='left',validate='one_to_one')
        if not frame.winner.isin([0,1]).all() or not frame.pred_winner.isin([0,1]).all():
            raise ValueError('Settled binary outcomes and predictions required')
        real=[f'dec_{odds_type}_{s}' for s in ('blue','red')]
        fair=[f'dec_fair_{odds_type}_{s}' for s in ('blue','red')]
        sides=frame.pred_winner.to_numpy(int)
        current=frame[real].to_numpy(float)[np.arange(len(frame)),sides]
        opening=frame[['reference_'+c for c in cols]].to_numpy(float)[np.arange(len(frame)),sides]
        decay_weights(opening,current,0,0)  # validate alignment and prices
        if not np.isfinite(frame[fair].to_numpy(float)).all() or (frame[fair]<=1).any().any():
            raise ValueError('Invalid fair prices')
        if 'proba_blue' not in frame:frame['proba_blue']=1-frame.proba_red
        if 'risk_proba_red' not in frame and 'risk_red_proba' in frame:
            frame['risk_proba_red']=frame.risk_red_proba
            frame['risk_proba_blue']=1-frame.risk_red_proba
        se=next((c for c in ('prediction_se','se','proba_se') if c in frame),None)
        common=dict(prob_cols=['proba_blue','proba_red'],fair_decimal_cols=fair,
            real_decimal_cols=real,pred_winner_col='pred_winner',init_bankroll=initial_bankroll)
        if se:common['prediction_se_col']=se
        if {'risk_proba_blue','risk_proba_red'}<=set(frame):
            common['risk_prob_cols']=['risk_proba_blue','risk_proba_red']
        for s in ('blue','red'):
            if 'open_'+s not in frame:
                d=frame['reference_'+f'dec_{reference_odds_type}_{s}'].to_numpy(float)
                frame['open_'+s]=np.where(d>=2,100*(d-1),-100/(d-1))
        allocations=[]
        for params in (opening_params,closing_params):
            args=fixed_kwargs(params)
            if args['z'] is not None and se is None:raise ValueError('Prediction SE required for z')
            result,_=simulate_kelly(frame,**common,**args)
            if len(result)!=len(frame):raise ValueError('Simulation changed fight cohort')
            allocations.append(result.choice_fstar.to_numpy(float))
        self.frame=frame
        self.opening_prices,self.current_prices=opening,current
        self.opening_stakes,self.closing_stakes=allocations
        self.unit_profit=np.where(sides==frame.winner.to_numpy(),current-1,-1)
        self.dates=np.sort(frame.date.unique())
        self.event_ids=pd.Index(self.dates).get_indexer(frame.date)
        self.path_indices=bootstrap_event_indices(len(self.dates),k,seed)
        self.initial_bankroll,self.risk_penalty,self.seed=initial_bankroll,risk_penalty,seed
        self.metadata=dict(odds_type=odds_type,reference_odds_type=reference_odds_type,
            opening_params=asdict(opening_params) if is_dataclass(opening_params) else dict(opening_params),
            closing_params=asdict(closing_params) if is_dataclass(closing_params) else dict(closing_params),
            movement='abs(log(current decimal / reference decimal)); direction selects rate',
            staking='Moneylines only; fixed policy stakes at current prices; post-blend closing-policy z correction and exposure cap; edge MDD adjustment disabled as in betting search',
            objective='mean terminal return - risk_penalty * sample SD terminal return',
            k=k,seed=seed,risk_penalty=risk_penalty,initial_bankroll=initial_bankroll,
            evaluation='In-sample tuning on supplied events; whole-event iid bootstrap, no top-ups or bankroll floor')

    def event_returns(self,k_down=0.,k_up=0.,policy='decay'):
        from ufc_betting.Models.Search.exponential_decay import evaluate_decay_portfolio
        return evaluate_decay_portfolio(self, dict(k_down=k_down, k_up=k_up), policy=policy)['events']

    def evaluate(self,k_down=0.,k_up=0.,policy='decay'):
        events=self.event_returns(k_down,k_up,policy)
        if (events.event_return < -1).any():
            # Reject infeasible bankroll paths; do not silently cap/rescale stakes.
            return dict(score=-np.inf,events=events,simulation=None)
        sim=simulate_bankroll_paths(events,self.path_indices,self.initial_bankroll)
        return dict(score=terminal_return_score(sim['terminal_returns'],self.risk_penalty),events=events,simulation=sim)

    def simulate_kelly(self, decay_rates):
        from ufc_betting.Models.Search.exponential_decay import decay_backtest
        return decay_backtest(self, decay_rates)

    def run(self,n_trials=100,max_decay=50.,output_dir=None):
        if not isinstance(n_trials,int) or n_trials<1 or not np.isfinite(max_decay) or max_decay<=0:
            raise ValueError('Positive n_trials and max_decay required')
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.enqueue_trial(dict(log1p_k_down=0.,log1p_k_up=0.))
        def objective(trial):
            result=self.evaluate(**suggest_decay_rates(trial,('k_down','k_up'),max_decay))
            trial.set_user_attr('max_event_exposure',float(result['events'].exposure.max()))
            return result['score']
        study.optimize(objective,n_trials=n_trials)
        if not np.isfinite(study.best_value):raise ValueError('No feasible candidate found')
        best_params=decode_decay_rates(study.best_params)
        best=self.evaluate(**best_params)
        baselines={p:self.evaluate(policy=p) for p in ('opening','closing')}
        result=dict(study=study,best_params=best_params,best_value=study.best_value,
                    event_results=best['events'],simulation=best['simulation'],baselines=baselines)
        if output_dir is not None:
            out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
            study.trials_dataframe().to_csv(out/'trials.csv',index=False)
            best['events'].to_csv(out/'event_returns.csv',index=False)
            np.savez_compressed(out/'bootstrap.npz',indices=self.path_indices,**best['simulation'])
            (out/'search_summary.json').write_text(json.dumps(dict(self.metadata,max_decay=max_decay,n_trials=n_trials,
                best_params=best_params,best_value=study.best_value,
                baseline_scores={p:r['score'] for p,r in baselines.items()}),indent=2),encoding='utf-8')
        return result
