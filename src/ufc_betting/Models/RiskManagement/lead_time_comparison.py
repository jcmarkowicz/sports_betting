"""Compare archived lead-time returns against saved test-set Kelly returns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .lead_time_backtest import _name


def compare_lead_time_returns(data_dir, run_dir=None):
    """Use the completed run's frozen controls and saved test predictions.

    Return is 100 * (final bankroll / initial bankroll - 1). Uniform scenarios
    report median terminal returns. Matched tests rerun on the identical fights;
    opening and closing benchmarks use only those matched fights.
    """
    from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
    data_dir = Path(data_dir)
    run_dir = Path(run_dir) if run_dir else data_dir / 'model_results' / 'lead_time_backtests' / 'test'
    config = json.loads((run_dir / 'run_config.json').read_text())
    summary = pd.read_csv(run_dir / 'path_summary.csv')
    bankroll = config['initial_bankroll']
    params, rates = config['params'], config['decay_rates']
    reference = pd.read_csv(data_dir / 'model_results' / 'test_set_data_open_with_open.csv')
    benchmarks, results = [], {}
    for stage in ('open', 'close1', 'close2'):
        frame = pd.read_csv(data_dir / 'model_results' / f'test_set_data_open_with_{stage}.csv')
        frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        frame['fight_key'] = ['|'.join([r.date, *sorted((_name(r.fighter_red), _name(r.fighter_blue)))])
                              for r in frame.itertuples()]
        frame = frame.loc[frame.winner.isin([0, 1])].copy()
        for day in config['days_out']:
            cohort = pd.read_csv(run_dir / f'cohort_{day}_days.csv')
            for scope in ('Matched test fights',):
                key = (stage, scope, day if scope == 'Matched test fights' else None)
                if key not in results:
                    inputs = frame.loc[frame.fight_key.isin(cohort.fight_key)].copy() if scope == 'Matched test fights' else frame.copy()
                    if scope == 'Matched test fights' and set(inputs.fight_key) != set(cohort.fight_key):
                        raise ValueError('Test benchmark does not contain every lead-time fight')
                    p = params[stage]
                    kelly, parlay = simulate_kelly(
                        inputs, prob_cols=['proba_blue', 'proba_red'],
                        fair_decimal_cols=[f'dec_fair_{stage}_blue', f'dec_fair_{stage}_red'],
                        real_decimal_cols=[f'dec_{stage}_blue', f'dec_{stage}_red'],
                        pred_winner_col='pred_winner', prediction_se_col='proba_se',
                        init_bankroll=bankroll, bankroll_floor=None,
                        max_drawdown=p['mdd'], N=p['N'], parlay_mdd=p['mdd_parlay'],
                        N_parlay=p['N_parlay'], z=p['z'], calc_parlay=True,
                        moneyline_american_bounds=(p['moneyline_min_american'], p.get('moneyline_max_american', np.inf)),
                        adjust_mdd_by_edge=False, decay_rates=rates,
                        opening_data=reference, opening_params=params['open'])
                    balance = kelly.groupby('date', sort=True).bankroll_postevent.first().iloc[-1]
                    results[key] = dict(return_pct=100*(balance/bankroll-1), events=kelly.date.nunique(), fights=len(kelly))
                    label = f'{stage}_{scope.replace(" ", "_")}_{key[2]}'
                    kelly.to_csv(run_dir / f'benchmark_{label}_kelly.csv', index=False)
                benchmarks.append(dict(odds_type=stage, days_out=day, scope=scope, **results[key]))
    benchmark = pd.DataFrame(benchmarks)
    rows = []
    for (day, stage, scenario), group in summary.groupby(['days_out', 'policy', 'scenario'], sort=False):
        matched = benchmark.loc[(benchmark.days_out == day) & (benchmark.odds_type == stage)
                                & benchmark.scope.eq('Matched test fights')].iloc[0]
        value = group.total_return_pct.median()
        rows.append(dict(days_out=day, odds_type=stage,
                         scenario='Recorded prices' if scenario == 'recorded' else 'Uniform prices (median)',
                         paths=len(group), events=matched.events, fights=matched.fights,
                         lead_time_return_pct=value, matched_test_return_pct=matched.return_pct,
                         difference_pp=value-matched.return_pct))
    table = pd.DataFrame(rows).sort_values(['scenario', 'days_out', 'odds_type'], ascending=[True,False,True])
    table.to_csv(run_dir / 'return_comparison.csv', index=False)
    midpoint_price_returns(data_dir, run_dir)
    return summarize_return_table(run_dir)


def midpoint_price_returns(data_dir, run_dir):
    """Simulate the arithmetic midpoint of both decimal quotes for every fight."""
    import joblib
    import statsmodels.api as sm
    from ufc_betting.config import config as model_config
    from .open_model_close1_input import opening_model_at_close1
    from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'run_config.json').read_text())
    model, scaler = sm.load(model_config.model_open_path), joblib.load(model_config.scaler_open_path)
    rows = []
    for day in config['days_out']:
        frame = pd.read_csv(run_dir / f'cohort_{day}_days.csv')
        prices = (frame[['dec_close1_red','dec_close1_blue']].to_numpy(float)
                  + frame[['dec_close2_red','dec_close2_blue']].to_numpy(float)) / 2
        predicted = opening_model_at_close1(frame, prices, model, scaler)
        for stage in ('close1','close2'):
            p = config['params'][stage]
            kelly, parlay = simulate_kelly(
                predicted, prob_cols=['proba_blue','proba_red'],
                fair_decimal_cols=['dec_fair_close1_blue','dec_fair_close1_red'],
                real_decimal_cols=['dec_close1_blue','dec_close1_red'],
                pred_winner_col='pred_winner', prediction_se_col='proba_se',
                init_bankroll=config['initial_bankroll'], bankroll_floor=None,
                max_drawdown=p['mdd'], N=p['N'], parlay_mdd=p['mdd_parlay'], N_parlay=p['N_parlay'],
                z=p['z'], calc_parlay=True, adjust_mdd_by_edge=False,
                moneyline_american_bounds=(p['moneyline_min_american'],p.get('moneyline_max_american',np.inf)),
                decay_rates=config['decay_rates'], opening_data=frame, opening_params=config['params']['open'])
            events = kelly.groupby('date',sort=True).first()
            rows.append({'Data':f'{day} days out (midpoint prices)', 'Odds':f'Midpoint / {stage} controls',
                         'Total return (%)':100*(events.bankroll_postevent.iloc[-1]/config['initial_bankroll']-1),
                         'Average event return (%)':events.bankroll_pct_change.mean()})
            kelly.to_csv(run_dir / f'midpoint_{day}_days_{stage}_kelly.csv',index=False)
            parlay.to_csv(run_dir / f'midpoint_{day}_days_{stage}_parlay.csv',index=False)
    result = pd.DataFrame(rows)
    result.to_csv(run_dir / 'midpoint_price_returns.csv',index=False)
    return result


def summarize_return_table(run_dir):
    """Total compounded return and arithmetic mean event return only.

    Midpoint rows use actual midpoint-price simulations, never sampled medians.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'run_config.json').read_text())
    rows = []
    for stage in ('open', 'close1', 'close2'):
        for scope, label, day in [('Matched_test_fights', 'Test set (matched)', config['days_out'][0])]:
            frame = pd.read_csv(run_dir / f'benchmark_{stage}_{scope}_{day}_kelly.csv')
            events = frame.groupby('date', sort=True).first()
            rows.append({'Data': label, 'Odds': stage,
                         'Total return (%)': 100*(events.bankroll_postevent.iloc[-1]/config['initial_bankroll']-1),
                         'Average event return (%)': events.bankroll_pct_change.mean()})
    summary = pd.read_csv(run_dir / 'path_summary.csv')
    events = pd.read_csv(run_dir / 'event_returns.csv')
    for (day, stage, scenario), group in summary.groupby(['days_out', 'policy', 'scenario'], sort=False):
        if scenario != 'recorded':
            continue
        subset = events.loc[(events.days_out == day) & (events.policy == stage) & (events.scenario == scenario)]
        average = (subset.groupby('path').event_return.mean().median()
                   if scenario == 'uniform' else subset.event_return.mean())
        label = f'{day} days out' + (' — sampled median' if scenario == 'uniform' else '')
        rows.append({'Data': label, 'Odds': stage,
                     'Total return (%)': group.total_return_pct.median(),
                     'Average event return (%)': 100*average})
    table = pd.concat([pd.DataFrame(rows), pd.read_csv(run_dir / 'midpoint_price_returns.csv')],ignore_index=True)
    table.to_csv(run_dir / 'return_summary.csv', index=False)
    return table
