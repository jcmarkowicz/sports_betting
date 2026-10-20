"""Prepare Logit predictions before running the frozen betting optimizer."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ufc_betting.config import config, settings
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.utils import discover_alpha_history, load_validation_predictions


def optimize_kelly_params_with_logit(
    *, input_paths=None, alpha_history=None, training_input=None, search_dir=None,
    use_monthly_models=True, model_refit_months=1, parallel=True, **optimization_options,
):
    """Build saved/monthly Logit predictions, then optimize fixed betting inputs.

    Model fitting and historical alpha discovery happen only here. Defaults use
    the standard validation exports for each market. All optimization_options
    are forwarded to optimize_kelly_params (trials, bootstrap paths, risk, etc.).
    With use_monthly_models=False, prediction preparation uses saved models only.
    Betting optimization remains full-validation, not walk-forward betting tuning.
    """
    if 'validation_data' in optimization_options:
        raise ValueError('Use optimize_kelly_params directly for supplied validation_data')
    input_paths = {} if input_paths is None else dict(input_paths)
    unknown = set(input_paths) - set(config.odds_types)
    if unknown:
        raise ValueError(f'Unknown input odds types: {sorted(unknown)}')
    if use_monthly_models:
        if alpha_history is None:
            alpha_history = discover_alpha_history(search_dir)
        elif not isinstance(alpha_history, pd.DataFrame):
            source = Path(alpha_history)
            alpha_history = pd.read_csv(source, float_precision='round_trip')
            alpha_history['alpha_source'] = str(source)

    def prepare_market(odds_type):
        filename = ('test_logit_open.csv' if odds_type == 'open'
                    else f'val_set_data_open_with_{odds_type}.csv')
        path = input_paths.get(odds_type, settings.data_dir / 'model_results' / filename)
        return load_validation_predictions(
            path, odds_type, alpha_history=alpha_history, training_input=training_input,
            search_dir=search_dir, use_monthly_models=use_monthly_models,
            model_refit_months=model_refit_months,
        )

    if parallel:
        with ThreadPoolExecutor(max_workers=len(config.odds_types)) as executor:
            validation_data = dict(zip(config.odds_types, executor.map(prepare_market, config.odds_types)))
    else:
        validation_data = {market: prepare_market(market) for market in config.odds_types}
    return optimize_kelly_params(validation_data=validation_data, parallel=parallel,
                                 **optimization_options)

