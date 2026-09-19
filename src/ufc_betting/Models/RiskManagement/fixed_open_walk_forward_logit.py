"""Frozen opening Logit, close1 inputs, joint risk selection every three months.

Run with --help for options. The first complete three-month block is warmup;
each subsequent block uses parameters selected only on preceding blocks.
The final partial block is reported explicitly. No model or scaler is fitted.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.Models.RiskManagement.model_switch_evaluation import audit
from ufc_betting.Models.RiskManagement.fixed_open_bayesian_search import (
    FixedOpenBayesianSearch, growth_metrics,
)


def walk_forward_fixed_open(predictions, *, model_training_end, n_trials=100,
                            seed=42, initial_bankroll=500., history_start=None,
                            first_test_start=None, data_end_exclusive=None,
                            z=.5, heavy_favorite_cutoff=-300, heavy_parlay_max_legs=4):
    """Search expanding history; freeze parameters for each next 3-month span.

    Date boundaries are month starts; end dates are exclusive. Without an
    explicit coverage end, the last observed day + 1 is used conservatively.
    Caller must supply the saved model's verified final training date.
    """
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be finite and positive')
    frame = predictions.copy()
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    cutoff = pd.Timestamp(model_training_end).normalize()
    frame = frame.loc[frame.date > cutoff].sort_values('date').reset_index(drop=True)
    if frame.empty:
        raise ValueError('No events after model training')
    start = (max(frame.date.min().to_period('M').start_time,
                 (cutoff.to_period('M') + 1).start_time)
             if history_start is None else pd.Timestamp(history_start))
    test_start = start + pd.DateOffset(months=3) if first_test_start is None else pd.Timestamp(first_test_start)
    if start.day != 1 or test_start.day != 1 or start <= cutoff:
        raise ValueError('History/test starts must be month boundaries after model training')
    months = (test_start.year - start.year) * 12 + test_start.month - start.month
    if months < 3 or months % 3:
        raise ValueError('First test must follow one or more complete 3-month history blocks')
    end = (frame.date.max() + pd.Timedelta(days=1) if data_end_exclusive is None
           else pd.Timestamp(data_end_exclusive))
    frame = frame.loc[(frame.date >= start) & (frame.date < end)].reset_index(drop=True)
    fixed = dict(z=z, heavy_favorite_cutoff=heavy_favorite_cutoff,
                 heavy_parlay_max_legs=heavy_parlay_max_legs)
    blocks, paths, studies = [], [], {}
    bankroll, previous = initial_bankroll, None
    while test_start < end:
        block_end = test_start + pd.DateOffset(months=3)
        train = frame.loc[frame.date < test_start]
        test = frame.loc[(frame.date >= test_start) & (frame.date < block_end)]
        if train.empty:
            raise ValueError('No historical events for parameter selection')
        search = FixedOpenBayesianSearch(train, **fixed)
        study = search.run(start, n_trials=n_trials, seed=seed + len(blocks), enqueue_params=previous)
        previous = study.best_params.copy()
        key = str(test_start.date())
        studies[key] = study
        events = (FixedOpenBayesianSearch(test, **fixed).event_returns(previous) if len(test)
                  else pd.DataFrame(columns=['date', 'event_return']))
        metrics = growth_metrics(events.event_return, bankroll)
        blocks.append(dict(test_start=test_start, test_end_exclusive=block_end,
                           observed_end_exclusive=min(end, block_end),
                           partial_block=block_end > end,
                           training_events=train.date.nunique(), training_end=train.date.max(),
                           best_mean_3m_log_growth=study.best_value,
                           starting_bankroll=bankroll, **metrics, **previous))
        events['block_start'] = test_start
        events['bankroll'] = bankroll * (1 + events.event_return).cumprod()
        for name, value in previous.items():
            events[name] = value
        paths.append(events)
        bankroll = metrics['ending_bankroll']
        test_start = block_end
    if not blocks:
        raise ValueError('Need data beyond the initial three-month history')
    return dict(block_results=pd.DataFrame(blocks), event_results=pd.concat(paths, ignore_index=True),
                studies=studies, history_start=start, fixed_controls=fixed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=settings.data_dir / 'model_results/test_logit_close1.csv')
    parser.add_argument('--output', type=Path, default=settings.data_dir / 'model_results/fixed_open_walk_forward')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initial-bankroll', type=float, default=500)
    parser.add_argument('--history-start')
    parser.add_argument('--first-test-start')
    parser.add_argument('--data-end-exclusive')
    parser.add_argument('--z', type=float, default=.5)
    parser.add_argument('--heavy-favorite-cutoff', type=int, default=-300)
    parser.add_argument('--heavy-parlay-max-legs', type=int, default=4)
    args = parser.parse_args()
    model = sm.load(config.model_open_path)
    scaler = joblib.load(config.scaler_open_path)
    training = audit(model, scaler, 'open', settings.data_dir / 'model_results')
    frame = pd.read_csv(args.input)
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    if frame.duplicated(['date', 'fighter_red', 'fighter_blue']).any():
        raise ValueError('Duplicate fights in input')
    frame = frame.loc[frame.date > pd.Timestamp(training['last_training_date'])].reset_index(drop=True)
    predictions = opening_model_at_close1(frame, model=model, scaler=scaler)
    result = walk_forward_fixed_open(
        predictions, model_training_end=training['last_training_date'],
        **{k: getattr(args, k) for k in ('n_trials', 'seed', 'initial_bankroll', 'history_start',
            'first_test_start', 'data_end_exclusive', 'z', 'heavy_favorite_cutoff', 'heavy_parlay_max_legs')},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / 'fixed_predictions.csv', index=False)
    result['block_results'].to_csv(args.output / 'three_month_results.csv', index=False)
    result['event_results'].to_csv(args.output / 'event_bankroll.csv', index=False)
    for key, study in result['studies'].items():
        study.trials_dataframe().to_csv(args.output / f'trials_{key}.csv', index=False)
    metadata = dict(training_audit=training, model=str(config.model_open_path),
                    scaler=str(config.scaler_open_path), input=str(args.input),
                    model_refitted=False, prediction_inputs='close1 prices through opening feature/scaler schema',
                    objective='mean log compounded growth across prior 3-month blocks',
                    history='expanding; first full block reserved for selection',
                    history_start=str(result['history_start']), fixed_controls=result['fixed_controls'],
                    parlays='top_ev and heavy_favorite share mdd_parlay and N_parlay',
                    n_trials=args.n_trials, seed=args.seed,
                    summary=growth_metrics(result['event_results'].event_return, args.initial_bankroll))
    (args.output / 'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(result['block_results'].to_string(index=False))
    print(f'Outputs: {args.output}')


if __name__ == '__main__':
    main()
