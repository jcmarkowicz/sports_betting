"""Search two moneyline allocation-decay rates on bootstrapped event paths.

Both fixed policies are evaluated at the SAME historical execution prices and
probabilities. Opening prices only measure movement. No parlay, odds shifting,
parameter fitting occurs. The decay blend precedes z correction and the exposure cap.

Example using full-validation optimize_kelly_params results (no model refit needed)::

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
bootstrap_mode/ bootstrap_dir support the same saved schedules as FrozenBettingSearch.
Paths retain their final balance and stop betting when that balance is <= zero.
This module only optimizes and evaluates decay rates. Prepare separate backtests
with KellySimInput.from_predictions(..., decay_rates=result['best_params'],
opening_data=opening_data, opening_params=opening_params, calc_parlay=False).
"""
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch, terminal_return_score
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths


def decay_weights(opening, current, k_down, k_up):
    return BettingReturnCalculator.decay_weights(opening, current, k_down, k_up)


def suggest_decay_rates(trial, names, max_decay):
    """Uniform log1p coordinates resolve small rates and include exact zero."""
    rates={name:float(np.expm1(trial.suggest_float('log1p_'+name,0.,float(np.log1p(max_decay))))) for name in names}
    for name,value in rates.items():trial.set_user_attr(name,value)
    return rates


def decode_decay_rates(params):
    return {name.removeprefix('log1p_'):float(np.expm1(value)) for name,value in params.items()}


def evaluate_decay_portfolio(search, rates, *, include_parlay=False, policy='decay'):
    """Use cached raw policies and the calculator's post-blend adjustments."""
    params = [BettingParams.from_value(search.metadata[name + '_params'])
              for name in ('opening', 'closing')]
    key = repr([p.to_dict() for p in params])
    if getattr(search, '_raw_policy_key', None) != key:
        search._raw_policy_caches = [BettingReturnCalculator.calculate_scenario_cache(
            search.frame, p, search.metadata['odds_type']) for p in params]
        search._raw_policy_key = key
    return BettingReturnCalculator.calculate_decay_portfolio(
        *search._raw_policy_caches, np.repeat(search.opening_prices[:, None], 2, axis=1),
        search.frame.winner.to_numpy(), search.frame.date, *params, rates,
        include_parlay=include_parlay, policy=policy)


def decay_backtest(search, rates, *, include_parlay=False):
    """Render the shared calculator's final stakes using one chronological bankroll."""
    rates = rates.get('best_params', rates)
    result = evaluate_decay_portfolio(search, rates, include_parlay=include_parlay)
    events, stakes = result['events'], result['stakes']
    from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import settle_bankroll_paths
    balance = settle_bankroll_paths(events.event_return.to_numpy()[None], search.initial_bankroll)['bankroll'][0]
    before = np.maximum(balance[:-1], 0.)
    ids = search.event_ids
    frame = search.frame.copy()
    frame['choice_fstar'] = stakes
    frame['opening_weight'] = decay_weights(search.opening_prices, search.current_prices,
                                           rates['k_down'], rates['k_up'])
    frame['net_odds'] = np.where(stakes > 0, search.unit_profit, 0.)
    frame['fight_payout'] = before[ids] * stakes * search.unit_profit
    frame['bankroll_postevent'] = balance[1:][ids]
    frame['bankroll_pct_change'] = 100 * events.event_return.to_numpy()[ids]
    frame['event_payout_money_line'] = (before * events.moneyline_return.to_numpy())[ids]
    frame['parlay_net'] = (before * events.parlay_return.to_numpy())[ids]
    frame['event_total_fstar'] = events.exposure.to_numpy()[ids]
    # Exposure-weighted mean of individual adjustments, for diagnostics only.
    frame['uncertainty_multiplier'] = events.uncertainty_multiplier.to_numpy()[ids]
    totals = np.bincount(ids, weights=frame.net_odds, minlength=len(events))
    frame['event_ml_net_odds'] = totals[ids]
    frame['parlay_net_odds'] = 0.
    tickets = result['tickets']
    if not tickets.empty:
        rows = frame.iloc[tickets.fight_index.to_numpy()]
        tickets['choice_fighter_name'] = np.where(rows.pred_winner == 1, rows.fighter_red, rows.fighter_blue)
        unit = tickets.groupby('date').parlay_net_odds.first()
        frame['parlay_net_odds'] = frame.date.map(unit).fillna(0.)
    else:
        tickets = pd.DataFrame(columns=['date', 'fstar_parlay', 'parlay_net_odds'])
    return frame, tickets


class MoneylineDecaySearch:
    def __init__(self,betting_data,opening_data,opening_params,closing_params,
                 odds_type='close1',reference_odds_type='open',*,k=1000,
                 initial_bankroll=500.,risk_penalty=.25,seed=42,
                 bootstrap_mode='generate',bootstrap_dir=None,sample_with_replacement=True):
        if odds_type not in ('open','close1','close2') or reference_odds_type not in ('open','close1','close2'):
            raise ValueError('Unknown odds type')
        if not np.isfinite(risk_penalty) or risk_penalty<0 or not np.isfinite(initial_bankroll) or initial_bankroll<=0:
            raise ValueError('Invalid bankroll or risk penalty')
        keys=['date','fighter_red','fighter_blue']
        frame=read_frozen_predictions(betting_data);reference=opening_data.copy()
        if 'pred_winner' not in frame:
            frame['pred_winner']=(frame.proba_red>=.5).astype(int)
        for data in (frame,reference):
            data['date']=pd.to_datetime(data.date,errors='raise').dt.normalize()
            if data.empty or data.date.isna().any() or data.duplicated(keys).any():
                raise ValueError('Need nonempty dated unique fights')
        frame=frame.sort_values('date',kind='stable').reset_index(drop=True)
        cols=[f'dec_{reference_odds_type}_{s}' for s in ('blue','red')]
        ref=reference[keys+cols].rename(columns={c:'reference_'+c for c in cols})
        frame=frame.merge(ref,on=keys,how='left',validate='one_to_one')
        if not frame.winner.isin([0,1,2,3]).all() or not frame.pred_winner.isin([0,1]).all():
            raise ValueError('Settled outcomes (0/1 or 2/3 for void) and binary predictions required')
        real=[f'dec_{odds_type}_{s}' for s in ('blue','red')]
        fair=[f'dec_fair_{odds_type}_{s}' for s in ('blue','red')]
        sides=frame.pred_winner.to_numpy(int)
        current=frame[real].to_numpy(float)[np.arange(len(frame)),sides]
        opening=frame[['reference_'+c for c in cols]].to_numpy(float)[np.arange(len(frame)),sides]
        decay_weights(opening,current,0,0)  # validate alignment and prices
        if not np.isfinite(frame[fair].to_numpy(float)).all() or (frame[fair]<=1).any().any():
            raise ValueError('Invalid fair prices')
        frame['proba_blue'] = 1 - frame.proba_red
        # Shared validation, odds handling, and generate/save/load event schedules.
        frozen = FrozenBettingSearch(
            frame, odds_type, k=k, initial_bankroll=initial_bankroll,
            risk_penalty=risk_penalty, seed=seed, bootstrap_mode=bootstrap_mode,
            bootstrap_dir=bootstrap_dir, sample_with_replacement=sample_with_replacement,
        )
        if not np.array_equal(frame.pred_winner.to_numpy(), (frame.proba_red >= .5).astype(int)):
            raise ValueError('pred_winner must agree with supplied betting probabilities')
        self.bootstrap_dir = frozen.bootstrap_dir
        self.bootstrap_metadata = frozen.bootstrap_metadata
        self.bets_engine = frozen.bets_engine
        allocations=[]
        for params in (opening_params, closing_params):
            controls = BettingParams.from_value(params).to_policy()
            allocations.append(self.bets_engine.calculate_moneyline_stakes(
                frame.proba_red.to_numpy(), frame.risk_red_proba.to_numpy(),
                frame.prediction_se.to_numpy(), frame.winner.to_numpy(),
                np.arange(len(frame)), controls,
            ))
        self.frame=frame
        self.opening_prices,self.current_prices=opening,current
        self.opening_stakes,self.closing_stakes=allocations
        self.unit_profit=BettingReturnCalculator.moneyline_unit_profit(current, sides, frame.winner.to_numpy())
        self.dates=np.sort(frame.date.unique())
        self.event_ids=pd.Index(self.dates).get_indexer(frame.date)
        self.path_indices=frozen.path_indices
        self.initial_bankroll,self.risk_penalty,self.seed=initial_bankroll,risk_penalty,seed
        self.metadata=dict(odds_type=odds_type,reference_odds_type=reference_odds_type,
            opening_params=BettingParams.from_value(opening_params).to_policy(),
            closing_params=BettingParams.from_value(closing_params).to_policy(),
            movement='abs(log(current decimal / reference decimal)); direction selects rate',
            staking='Moneylines only; fixed policy stakes at current prices; post-blend closing-policy z correction and exposure cap; edge MDD adjustment disabled as in betting search',
            objective='mean terminal return - risk_penalty * sample SD terminal return',
            k=k,seed=seed,risk_penalty=risk_penalty,initial_bankroll=initial_bankroll,
            bootstrap_dir=str(self.bootstrap_dir) if self.bootstrap_dir is not None else None,
            sample_with_replacement=sample_with_replacement,
            evaluation='In-sample tuning on supplied events; shared event schedules; stop betting at nonpositive balance')

    def event_returns(self,k_down=0.,k_up=0.,policy='decay'):
        return evaluate_decay_portfolio(self, dict(k_down=k_down, k_up=k_up), policy=policy)['events']

    def evaluate(self,k_down=0.,k_up=0.,policy='decay'):
        events=self.event_returns(k_down,k_up,policy)
        sim=simulate_bankroll_paths(events,self.path_indices,self.initial_bankroll)
        return dict(score=terminal_return_score(sim['terminal_returns'],self.risk_penalty),events=events,simulation=sim)

    def run(self,n_trials=100,max_decay=50.,output_dir=None):
        if isinstance(n_trials,bool) or not isinstance(n_trials,int) or n_trials<1 or not np.isfinite(max_decay) or max_decay<=0:
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
