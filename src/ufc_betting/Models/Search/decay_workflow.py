"""Run joint decay tuning from saved Kelly controls and matching validation data."""

import json
from pathlib import Path
from ufc_betting.BettingStrategy.betting_params import BettingParams

import pandas as pd

from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.Models.Search.exponential_decay import MoneylineDecaySearch
from ufc_betting.Models.Search.parlay_decay import ParlayDecaySearch
from ufc_betting.BettingStrategy.MonteCarlo.bootstrap_paths import dataset_metadata, load_bootstrap_paths
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput


def _load_policy(data, params):
    frame = read_frozen_predictions(data)
    frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    frame = frame.sort_values('date', kind='stable').reset_index(drop=True)
    params = BettingParams.from_value(params, require_complete=True)
    # Exported search data can prove which fixed controls generated it.
    for name, value in params.to_policy().items():
        column = f'best_{name}'
        if column in frame and not frame[column].eq(value).all():
            raise ValueError(f'Kelly parameters do not match validation data: {column}')
    return frame, params


def optimize_decay_params(*, opening_data, closing_data, opening_params, closing_params,
                          bootstrap_dir, odds_type='close1', k=None, n_trials=100,
                          risk_penalty=.4, initial_bankroll=500., seed=42,
                          output_dir=None, bootstrap_mode='load', save_params=True):
    """Load saved schedules and jointly optimize moneyline/parlay decay rates.

    Data inputs accept DataFrames or CSV paths; parameters accept BettingParams,
    dicts or JSON paths. bootstrap_dir is the run folder containing open/
    and close1/ (or close2/). k=None uses all saved closing-market samples.
    bootstrap_mode='auto' loads an existing closing-market folder, otherwise
    generates samples in memory (1000 when k=None). 'load' requires saved paths;
    'generate' always samples in memory. Existing invalid paths raise an error.
    When closing_params is a JSON path, save_params=True updates that file with
    the fitted decay_rates and opening controls after a successful search.
    Returns betting_params (combined Kelly/decay controls), kelly_sim_input,
    and search diagnostics. output_dir also saves betting_params.json.
    """
    if odds_type not in ('close1', 'close2'):
        raise ValueError('odds_type must be close1 or close2')
    if bootstrap_mode not in ('auto', 'load', 'generate'):
        raise ValueError('bootstrap_mode must be auto, load, or generate')
    opening, opening_controls = _load_policy(opening_data, opening_params)
    closing, closing_controls = _load_policy(closing_data, closing_params)
    root = Path(bootstrap_dir)
    mode = bootstrap_mode
    if mode == 'auto':
        mode = 'load' if (root / odds_type).exists() else 'generate'
    selected = None
    if mode == 'load':
        # The joint objective uses closing schedules; validate opening ones too
        # when present. Never replace incompatible or incomplete saved samples.
        for market, frame in [('open', opening), (odds_type, closing)]:
            if market == 'open' and not (root / market).exists():
                continue
            saved = json.loads((root / market / 'metadata.json').read_text(encoding='utf-8'))
            load_bootstrap_paths(root / market,
                                 dataset_metadata(frame, market, saved['replace']),
                                 saved['n_samples'])
            if market == odds_type:
                selected = saved
    sample_count = (selected['n_samples'] if selected else 1000) if k is None else k
    moneylines = MoneylineDecaySearch(
        betting_data=closing, opening_data=opening,
        opening_params=opening_controls, closing_params=closing_controls,
        odds_type=odds_type, k=sample_count,
        risk_penalty=risk_penalty, initial_bankroll=initial_bankroll, seed=seed,
        bootstrap_mode=mode, bootstrap_dir=root / odds_type if mode == 'load' else None,
        sample_with_replacement=selected['replace'] if selected else True,
    )
    result = ParlayDecaySearch(moneylines).run(n_trials=n_trials, output_dir=output_dir)
    # Standard Kelly reports expect opening prices even with closing execution.
    keys = ['date', 'fighter_red', 'fighter_blue']
    missing_prices = [f'dec_open_{side}' for side in ('blue', 'red')
                      if f'dec_open_{side}' not in closing]
    if missing_prices:
        closing = closing.merge(opening[keys + missing_prices], on=keys,
                                how='left', validate='one_to_one', sort=False)
    combined_params = closing_controls.with_decay(result['best_params'], opening_controls)
    result['betting_params'] = combined_params
    result['kelly_sim_input'] = KellySimInput.from_predictions(
        closing, combined_params, odds_type, initial_bankroll,
        calc_parlay=True, opening_data=opening,
    )
    if output_dir is not None:
        combined_params.save(Path(output_dir) / 'betting_params.json')
    if save_params and isinstance(closing_params, (str, Path)):
        combined_params.save(closing_params)
        result['params_path'] = str(closing_params)
    result['bootstrap_mode'] = mode
    result['bootstrap_dir'] = str(root / odds_type) if mode == 'load' else None
    result['bootstrap_metadata'] = selected
    return result
