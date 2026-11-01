"""Inference-only Kelly search: open/open, close1/close1, close2/close2.

Uses held-out validation predictions from train-only fits. Runs opening first
so both closing searches use this run's opening parameters. Importing is safe.
"""
from pathlib import Path
import sys
import faulthandler

if __name__ == '__main__':
    faulthandler.enable(all_threads=True)

# Direct file execution puts only this script's directory on sys.path.
# Resolve local imports from the file location, independent of the working directory.
if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / 'src'))

import pandas as pd
from ufc_betting.config import settings, VAL_START_DATE, TEST_START_DATE
from Notebooks.Search.run_search import run_search


def run_inference_search(*, n_trials=500, n_paths=None, seed=42,
                         paths_path=None, closing_paths_path=None,
                         comparison_dir=None, params_dir=None,
                         markets=('open', 'close1', 'close2'), score_kwargs=None):

    score_settings = score_kwargs

    markets = tuple(markets)
    if not markets or len(set(markets)) != len(markets) or not set(markets) <= {'open', 'close1', 'close2'}:
        raise ValueError('markets must contain distinct open/close1/close2 entries')
    root = Path(comparison_dir) if comparison_dir else settings.data_dir / 'model_comparisons/lr_open_close'
    destination = Path(params_dir) if params_dir else settings.data_dir / 'kelly_params/inference_market_models'
    paths_path = Path(paths_path) if paths_path else settings.data_dir / 'bootstrapped_paths/sampled_fights_recorded_prices_val'
    closing_paths_path = Path(closing_paths_path) if closing_paths_path else paths_path
    if n_trials < 1 or (n_paths is not None and n_paths < 1):
        raise ValueError('Trials and selected path count must be positive')
    # Preflight all markets before starting an expensive search.
    sources = {}
    for market in markets:
        artifacts = root / 'train' / market
        prediction = artifacts / 'inference' / f'val_with_{market}.csv'
        frame = pd.read_csv(prediction)
        if frame.empty or not frame.probability_model.eq('inference').all():
            raise ValueError(f'{prediction} must contain only inference probabilities')
        dates = pd.to_datetime(frame.date, errors='raise')
        if not (dates.ge(pd.Timestamp(VAL_START_DATE)) & dates.lt(pd.Timestamp(TEST_START_DATE))).all():
            raise ValueError(f'{prediction} contains rows outside held-out validation')
        for name in ('model_se.pkl', 'scaler.pkl', 'encoder.pkl'):
            if not (artifacts/name).is_file():
                raise FileNotFoundError(artifacts/name)
        sources[market] = (prediction, artifacts)
    for path in (paths_path, closing_paths_path):
        if not path.exists():
            raise FileNotFoundError(path)
    results = {}
    for market in markets:
        prediction, artifacts = sources[market]
        results[market] = run_search(
            search_market=market, prediction_model_type=market,
            probability_model='inference', prediction_artifacts_dir=artifacts,
            predictions_path=prediction,
            paths_path=paths_path if market == 'open' else closing_paths_path,
            n_trials=n_trials, n_paths=n_paths, seed=seed, initial_bankroll=500,
            params_dir=destination, opening_params_path=destination/'open.json',
            **score_settings,
        )
    return results


if __name__ == '__main__':
    # Edit these settings directly, just as in run_search.py.
    # Keep open first; closing searches use its saved opening policy.
    # To rerun closing only, use ('close1', 'close2') with an existing open.json.
    markets = ('open', 'close1', 'close2')
    n_trials = 250
    n_paths = None  # None uses all saved paths; set an integer for a subset.
    seed = 42
    score_kwargs = {
        'score_type': 'event_log_growth',
        'drawdown_penalty': 0.01,
        'ulcer_penalty': 0.02,
        'drawdown_tail_fraction': 0.4,  # Average the worst 40% of path drawdowns.
        'wealth_floor': 0.001,  # Scoring floor as a fraction of initial bankroll.
    }
    # score_kwargs_close1 = {
    #     'score_type': 'event_log_growth',
    #     'drawdown_penalty': 0.02,
    #     'ulcer_penalty': 0.01,
    #     'drawdown_tail_fraction': 0.4,  # Average the worst 40% of path drawdowns.
    #     'wealth_floor': 0.001,  # Scoring floor as a fraction of initial bankroll.
    # }
    # score_kwargs_close2 = {
    #     'score_type': 'event_log_growth',
    #     'drawdown_penalty': 0.02,
    #     'ulcer_penalty': 0.01,
    #     'drawdown_tail_fraction': 0.4,  # Average the worst 40% of path drawdowns.
    #     'wealth_floor': 0.001,  # Scoring floor as a fraction of initial bankroll.
    # }
    # kwargs_dict ={
    #     'open':score_kwargs,
    #     'close1':score_kwargs_close1,
    #     'close2':score_kwargs_close2
    # }

    paths_path = settings.data_dir / 'bootstrapped_paths/sampled_fights_recorded_prices_val'
    closing_paths_path = paths_path  # Can instead point to saved sampled-price paths.
    comparison_dir = settings.data_dir / 'model_comparisons/lr_open_close'
    params_dir = settings.data_dir / 'kelly_params/inference_market_models'

    results = {}
    for market in markets:
        # kwargs = kwargs_dict[market]
        kwargs = score_kwargs

        results.update(run_inference_search(
            markets=(market,),
            n_trials=n_trials,
            n_paths=n_paths,
            seed=seed,
            score_kwargs=kwargs,
            paths_path=paths_path,
            closing_paths_path=closing_paths_path,
            comparison_dir=comparison_dir,
            params_dir=params_dir,
        ))
    print(f'Search completed. Results saved to {params_dir / "search_results"}')
