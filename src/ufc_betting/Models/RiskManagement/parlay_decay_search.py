"""Joint four-rate moneyline and parlay decay search on one bankroll."""
from pathlib import Path
from copy import copy
import json
import numpy as np
import pandas as pd
import optuna
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.Models.RiskManagement.exponential_decay_search import fixed_kwargs, suggest_decay_rates, decode_decay_rates
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import simulate_bankroll_paths, terminal_return_score


def return_comparison(decay, no_decay):
    """Compare combined compounded returns; improvement is percentage points."""
    rows=[]
    for label,extract in (
        ('Historical chronological path',lambda r:np.prod(1+r['events'].event_return.to_numpy())-1),
        ('Mean bootstrap terminal return',lambda r:r['simulation']['terminal_returns'].mean()),
    ):
        baseline=float(extract(no_decay))*100
        fitted=float(extract(decay))*100
        rows.append(dict(evaluation=label,no_decay_total_return_pct=baseline,
            decay_total_return_pct=fitted,return_improvement_pp=fitted-baseline))
    return pd.DataFrame(rows)


class ParlayDecaySearch:
    """Optimize all four rates; moneyline_search supplies data, not fitted rates.

    No call to moneyline_search.run() is required. Every trial changes singles
    and parlay allocations before scoring their combined event bankroll paths.
    """
    def __init__(self, moneyline_search):
        moneyline_search = copy(moneyline_search)
        self.moneyline_search=moneyline_search
        self.initial_bankroll=moneyline_search.initial_bankroll
        self.path_indices=moneyline_search.path_indices.copy()
        self.risk_penalty=moneyline_search.risk_penalty
        self.seed=moneyline_search.seed
        self.frame=moneyline_search.frame
        stage=moneyline_search.metadata['odds_type']
        common=dict(prob_cols=['proba_blue','proba_red'],
            fair_decimal_cols=[f'dec_fair_{stage}_{s}' for s in ('blue','red')],
            real_decimal_cols=[f'dec_{stage}_{s}' for s in ('blue','red')],
            pred_winner_col='pred_winner',init_bankroll=self.initial_bankroll)
        common['prediction_se_col']=next((c for c in ('prediction_se','se','proba_se') if c in self.frame),'se')
        if {'risk_proba_blue','risk_proba_red'}<=set(self.frame):
            common['risk_prob_cols']=['risk_proba_blue','risk_proba_red']
        tickets=[]
        for label in ('opening','closing'):
            p=moneyline_search.metadata[label+'_params']
            args=fixed_kwargs(p)
            args.update(calc_parlay=True,parlay_mdd=p['mdd_parlay'],
                        N_parlay=int(p['N_parlay']))
            moneylines,ticket=simulate_kelly(self.frame,**common,**args)
            setattr(moneyline_search, label + '_stakes', moneylines.choice_fstar.to_numpy(float))
            tickets.append(ticket)
        records=[]
        self.ticket_rows=[]
        for event_id,date in enumerate(moneyline_search.dates):
            group=self.frame.loc[self.frame.date.eq(date)]
            sides=group.pred_winner.to_numpy(int)
            p=np.where(sides==1,group.proba_red,group.proba_blue)
            odds=group[common['real_decimal_cols']].to_numpy()[np.arange(len(group)),sides]
            order=pd.Series(p*odds-1).sort_values(ascending=False, kind='stable').index.to_numpy()[:2]
            ids=group.index.to_numpy()[order]
            if len(ids)<2:
                records.append((0.,0.,0.,0.,0.));continue
            movements=np.log(moneyline_search.current_prices[ids]/moneyline_search.opening_prices[ids])
            down=np.maximum(-movements,0).sum();up=np.maximum(movements,0).sum()
            stake=[]
            for ticket in tickets:
                t=ticket.loc[pd.to_datetime(ticket.date).eq(date)]
                stake.append(float(t.fstar_parlay.iloc[0]) if len(t) else 0.)
            won=(group.iloc[order].winner.to_numpy()==sides[order]).all()
            unit=float(odds[order].prod()-1) if won else -1.
            records.append((*stake,down,up,unit))
            for idx in ids:self.ticket_rows.append((event_id,int(idx)))
        self.ticket_values=np.asarray(records)

    def event_returns(self,k_down=0.,k_up=0.,k_parlay_down=0.,k_parlay_up=0.,policy='decay'):
        from ufc_betting.Models.Search.exponential_decay import evaluate_decay_portfolio
        rates = dict(k_down=k_down, k_up=k_up, k_parlay_down=k_parlay_down, k_parlay_up=k_parlay_up)
        return evaluate_decay_portfolio(self.moneyline_search, rates,
                                       include_parlay=True, policy=policy)['events']

    def evaluate(self,**kwargs):
        events=self.event_returns(**kwargs)
        if (events.event_return < -1).any():return dict(score=-np.inf,events=events,simulation=None)
        sim=simulate_bankroll_paths(events,self.path_indices,self.initial_bankroll)
        return dict(score=terminal_return_score(sim['terminal_returns'],self.risk_penalty),events=events,simulation=sim)

    def run(self,n_trials=100,max_decay=50.,output_dir=None):
        if n_trials<1 or not np.isfinite(max_decay) or max_decay<=0:raise ValueError('Invalid search bounds')
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=self.seed))
        names=('k_down','k_up','k_parlay_down','k_parlay_up')
        study.enqueue_trial({'log1p_'+name:0. for name in names})
        movement=np.log(self.moneyline_search.current_prices/self.moneyline_search.opening_prices)
        magnitudes=[-movement[movement<0],movement[movement>0],
                    self.ticket_values[:,2],self.ticket_values[:,3]]
        for weight in (.8,.5,.2):
            rates={}
            for name,values in zip(names,magnitudes):
                positive=values[values>0]
                reference=float(np.median(positive)) if len(positive) else 1.
                rates['log1p_'+name]=float(np.log1p(min(-np.log(weight)/reference,max_decay)))
            study.enqueue_trial(rates)
        def objective(trial):
            return self.evaluate(**suggest_decay_rates(trial,names,max_decay))['score']
        study.optimize(objective,n_trials=n_trials)
        if not np.isfinite(study.best_value):raise ValueError('No feasible allocation found; no stakes were rescaled')
        best_params=decode_decay_rates(study.best_params)
        best=self.evaluate(**best_params)
        baselines={p:self.evaluate(policy=p) for p in ('opening','closing')}
        baselines['no_parlays']=self.evaluate(k_down=best_params['k_down'],k_up=best_params['k_up'],policy='none')
        result=dict(study=study,best_params=best_params,best_value=study.best_value,search_type='joint_four_rate',
            event_results=best['events'],simulation=best['simulation'],baselines=baselines)
        # Zero rates select opening raw stakes, followed by closing-policy z.
        result['return_comparison']=return_comparison(best,baselines['closing'])
        w_ml=np.exp(-np.where(movement<0,best_params['k_down'],best_params['k_up'])*np.abs(movement))
        w_parlay=np.exp(-best_params['k_parlay_down']*self.ticket_values[:,2]-best_params['k_parlay_up']*self.ticket_values[:,3])
        result['allocation_weights']=pd.DataFrame([
            dict(group=label,count=len(values),mean_opening_weight=float(values.mean()),
                 median_opening_weight=float(np.median(values)),
                 fraction_near_open=float((values>=.99).mean()),
                 fraction_near_close=float((values<=.01).mean()))
            for label,values in [('moneyline_down',w_ml[movement<0]),('moneyline_up',w_ml[movement>0]),('parlay',w_parlay)] if len(values)
        ])
        if output_dir is not None:
            out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
            result['allocation_weights'].to_csv(out/'allocation_weights.csv',index=False)
            result['return_comparison'].to_csv(out/'return_comparison.csv',index=False)
            study.trials_dataframe().to_csv(out/'trials.csv',index=False)
            best['events'].to_csv(out/'event_returns.csv',index=False)
            np.savez_compressed(out/'bootstrap.npz',indices=self.path_indices,**best['simulation'])
            (out/'summary.json').write_text(json.dumps(dict(best_params=best_params,
                score=study.best_value,
                baseline_scores={p:v['score'] for p,v in baselines.items()},
                method='All four decay rates jointly optimized. Fixed policy allocations; Parlay stakes use shared backtest endpoint sizing. Post-blend closing-policy z correction and joint exposure cap. In-sample event bootstrap.'),indent=2),encoding='utf-8')
        return result

    def simulate_kelly(self, rates):
        from ufc_betting.Models.Search.exponential_decay import decay_backtest
        return decay_backtest(self.moneyline_search, rates, include_parlay=True)
