"""Joint four-rate moneyline and parlay decay search on one bankroll."""
from pathlib import Path
from copy import copy
import json
import numpy as np
import pandas as pd
import optuna
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.Models.Search.exponential_decay import suggest_decay_rates, decode_decay_rates
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths
from ufc_betting.Models.Search.frozen_betting import terminal_return_score


def return_comparison(decay, no_decay):
    """Compare combined compounded returns; improvement is percentage points."""
    rows=[]
    for label,extract in (
        ('Historical chronological path',lambda r:simulate_bankroll_paths(r['events'], np.arange(len(r['events']))[None, :])['terminal_returns'][0]),
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
        # Joint policies size both bet families with one uncertainty/exposure
        # adjustment, matching simulate_kelly(calc_parlay=True).
        moneyline_search = copy(moneyline_search)
        self.moneyline_search=moneyline_search
        self.initial_bankroll=moneyline_search.initial_bankroll
        self.path_indices=moneyline_search.path_indices.copy()
        self.risk_penalty=moneyline_search.risk_penalty
        self.seed=moneyline_search.seed
        self.frame=moneyline_search.frame
        stage=moneyline_search.metadata['odds_type']
        real_columns = [f'dec_{stage}_{side}' for side in ('blue', 'red')]
        tickets=[]
        for label in ('opening', 'closing'):
            params = moneyline_search.metadata[label + '_params']
            controls = BettingParams.from_value(params).to_policy()
            setattr(moneyline_search, label + '_stakes',
                moneyline_search.bets_engine.calculate_moneyline_stakes(
                    self.frame.proba_red.to_numpy(), self.frame.risk_red_proba.to_numpy(),
                    self.frame.prediction_se.to_numpy(), self.frame.winner.to_numpy(),
                    np.arange(len(self.frame)), controls, include_parlay=True,
                ))
            tickets.append(moneyline_search.bets_engine.calculate_parlay_allocations(
                self.frame.proba_red.to_numpy(), self.frame.risk_red_proba.to_numpy(),
                self.frame.prediction_se.to_numpy(), self.frame.winner.to_numpy(),
                np.arange(len(self.frame)), controls,
            ))
        records=[]
        self.ticket_rows=[]
        for event_id,date in enumerate(moneyline_search.dates):
            group=self.frame.loc[self.frame.date.eq(date)]
            sides=group.pred_winner.to_numpy(int)
            p=np.where(sides==1,group.proba_red,group.proba_blue)
            odds=group[real_columns].to_numpy()[np.arange(len(group)),sides]
            order=pd.Series(p*odds-1).sort_values(ascending=False, kind='stable').index.to_numpy()[:2]
            ids=group.index.to_numpy()[order]
            if len(ids)<2:
                records.append((0.,0.,0.,0.,0.));continue
            movements=np.log(moneyline_search.current_prices[ids]/moneyline_search.opening_prices[ids])
            down=np.maximum(-movements,0).sum();up=np.maximum(movements,0).sum()
            stake=[]
            for ticket in tickets:
                t=ticket.loc[pd.to_datetime(ticket.date).eq(date)]
                stake.append(float(t.fstar.iloc[0]) if len(t) else 0.)
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
        sim=simulate_bankroll_paths(events,self.path_indices,self.initial_bankroll)
        return dict(score=terminal_return_score(sim['terminal_returns'],self.risk_penalty),events=events,simulation=sim)

    def evaluate_parlay_only(self, k_parlay_down=0., k_parlay_up=0., *, policy='decay'):
        """Evaluate a standalone parlay bankroll on the same event schedules.

        Remove moneylines BEFORE uncertainty adjustment and exposure capping.
        No decay uses policy='closing'; zero rates instead select opening stakes.
        Fixed closing z applies after blending the raw opening/closing tickets.
        """
        from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator

        if policy not in ('decay', 'closing', 'opening'):
            raise ValueError('Parlay-only policy must be decay, closing, or opening')
        search = self.moneyline_search
        params = [BettingParams.from_value(search.metadata[name + '_params'])
                  for name in ('opening', 'closing')]
        caches = [BettingReturnCalculator.calculate_scenario_cache(
            search.frame, controls, search.metadata['odds_type']) for controls in params]
        for cache in caches:
            cache[:, 0] = 0.  # Keep candidate legs; remove all moneyline allocations.
        rates = dict(k_down=0., k_up=0., k_parlay_down=k_parlay_down, k_parlay_up=k_parlay_up)
        events = BettingReturnCalculator.calculate_decay_portfolio(
            *caches, np.repeat(search.opening_prices[:, None], 2, axis=1),
            search.frame.winner.to_numpy(), search.frame.date, *params, rates,
            include_parlay=True, policy=policy)['events']
        simulation = simulate_bankroll_paths(events, self.path_indices, self.initial_bankroll)
        return dict(score=terminal_return_score(simulation['terminal_returns'], self.risk_penalty),
                    events=events, simulation=simulation)

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
            evaluated = self.evaluate(**suggest_decay_rates(trial,names,max_decay))
            terminal = evaluated['simulation']['terminal_returns']
            trial.set_user_attr('mean_terminal_return', float(terminal.mean()))
            trial.set_user_attr('std_terminal_return', float(terminal.std(ddof=1)))
            trial.set_user_attr('risk_penalty', self.risk_penalty)
            return evaluated['score']
        study.optimize(objective,n_trials=n_trials)
        if not np.isfinite(study.best_value):raise ValueError('No feasible allocation found; no stakes were rescaled')
        best_params=decode_decay_rates(study.best_params)
        best=self.evaluate(**best_params)
        baselines={p:self.evaluate(policy=p) for p in ('opening','closing')}
        baselines['no_parlays']=self.evaluate(k_down=best_params['k_down'],k_up=best_params['k_up'],policy='none')
        result=dict(study=study,best_params=best_params,best_value=study.best_value,search_type='joint_four_rate',
            event_results=best['events'],simulation=best['simulation'],baselines=baselines)
        # No decay means the ordinary closing policy, as in use_decay=False.
        # Zero rates select opening raw stakes, followed by closing-policy z.
        result['return_comparison']=return_comparison(best,baselines['closing'])
        result['comparison_baseline']='Closing Kelly policy without decay (use_decay=False)'
        result['objective_breakdown'] = pd.DataFrame([
            dict(policy=label,
                 mean_terminal_return_pct=float(value['simulation']['terminal_returns'].mean()) * 100,
                 std_terminal_return_pct=float(value['simulation']['terminal_returns'].std(ddof=1)) * 100,
                 risk_penalty=self.risk_penalty,
                 penalty_pp=self.risk_penalty * float(value['simulation']['terminal_returns'].std(ddof=1)) * 100,
                 objective_pct_points=value['score'] * 100,
                 objective_fraction=value['score'])
            for label, value in [('Fitted decay', best),
                                 ('No decay (closing policy)', baselines['closing']),
                                 ('Opening policy (including opening z)', baselines['opening']),
                                 ('Zero decay rates (closing z)', self.evaluate())]
        ])
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
            result['objective_breakdown'].to_csv(out/'objective_breakdown.csv',index=False)
            study.trials_dataframe().to_csv(out/'trials.csv',index=False)
            best['events'].to_csv(out/'event_returns.csv',index=False)
            np.savez_compressed(out/'bootstrap.npz',indices=self.path_indices,**best['simulation'])
            (out/'summary.json').write_text(json.dumps(dict(best_params=best_params,
                score=study.best_value,
                baseline_scores={p:v['score'] for p,v in baselines.items()},
                method='All four decay rates jointly optimized. Fixed policy allocations; Parlay stakes use BettingReturnCalculator. Post-blend closing-policy z correction and joint exposure cap. In-sample event bootstrap.'),indent=2),encoding='utf-8')
        return result

