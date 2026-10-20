"""Evaluate saved intermediate-price paths with a selected model and betting policy."""
import pandas as pd
from pathlib import Path
from time import perf_counter

from ufc_betting.config import config, settings
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import (
    load_sampled_paths,
    predict_intermediate_paths,
)


def prepare_prediction_paths(
    sampled_paths_file, *, prediction_model_type="open", search_market="close1",
    n_paths=2000, seed=42,
):
    """Select saved paths and infer on their fixed prices without resampling."""
    shared_paths = load_sampled_paths(
        sampled_paths_file, n_paths=n_paths, random_state=seed,
    )
    model_type = prediction_model_type
    paths = predict_intermediate_paths(
        shared_paths,
        prediction_model=getattr(config, f"model_{model_type}_path"),
        prediction_se_model=getattr(config, f"model_{model_type}_se_path", None),
        prediction_model_type=model_type,
        prediction_scalers={model_type: getattr(config, f"scaler_{model_type}_path")},
        prediction_encoders={model_type: getattr(
            config, f"encoder_{model_type}_path",
            settings.saved_models_dir / f"encoder_{model_type}.pkl",
        )},
        search_market=search_market,
    )
    event_counts = {path.date.nunique() for path in paths}
    if len(event_counts) != 1:
        raise ValueError("Summary plots require the same number of events in every path")
    print(f"Prepared {len(paths)} paths of {event_counts.pop()} events using saved intermediate prices")
    return paths


def load_policy_options(params_dir, *, search_market="close1", use_nearest_closing_params=True):
    """Load both distance-selected closing policies or one fixed market policy."""
    params_dir = Path(params_dir)
    if use_nearest_closing_params:
        closing_params = {
            market: BettingParams.load(params_dir / f"{market}.json")
            for market in ("close1", "close2")
        }
        return {"closing_params": closing_params, "kelly_kwargs": {"calc_parlay": True}}

    betting_params = BettingParams.load(params_dir / f"{search_market}.json")
    return {"kelly_kwargs": {"betting_params": betting_params, "calc_parlay": True}}


def run_simulation(
    sampled_paths_file, *, params_dir=None, prediction_model_type="open",
    search_market="close1", use_nearest_closing_params=True,
    n_paths=2000, initial_bankroll=500, len_temp=1, seed=42, progress_every=10,
):
    """Prepare inputs and return profits, bankroll paths and compact event-exposure metadata.

    n_paths=None uses all saved paths. search_market selects the downstream
    policy label, not new price draws. Nothing is trained, resampled, or saved.
    Nearest-closing evaluation uses shared fight caches and batched cards.
    All saved sizing and decay settings are always applied. Missing opening
    controls or decay rates fail before expensive prediction preparation.
    """
    policy_options = load_policy_options(
        params_dir if params_dir is not None else settings.data_dir / "kelly_params",
        search_market=search_market,
        use_nearest_closing_params=use_nearest_closing_params,
    )
    policies = (policy_options['closing_params'] if 'closing_params' in policy_options
                else {search_market: policy_options['kelly_kwargs']['betting_params']})
    required_rates = {'k_down', 'k_up', 'k_parlay_down', 'k_parlay_up'}
    for market, params in policies.items():
        if params.opening_params is None or params.decay_rates is None or not required_rates <= params.decay_rates.keys():
            raise ValueError(f'{market}: full Kelly policy requires saved opening_params and all four decay rates')
    policy_summary = {market: params.to_dict() for market, params in policies.items()}
    started = perf_counter()
    print('Loading paths and generating predictions...', flush=True)
    paths = prepare_prediction_paths(
        sampled_paths_file, prediction_model_type=prediction_model_type,
        search_market=search_market, n_paths=n_paths, seed=seed,
    )
    print(f'Prediction preparation: {perf_counter() - started:.1f}s', flush=True)
    print('Risk scaling: per-bet conservative/raw edge ratio, then joint bankroll cap', flush=True)
    print('Closing-policy z: ' + ', '.join(f'{market}={params["z"]}' for market, params in policy_summary.items()), flush=True)
    started = perf_counter()
    simulation = simulate(
        sampled_paths=paths, odds_type="intermediate",
        init_bankroll=initial_bankroll, use_decay=True,
        len_temp=len_temp, progress_every=progress_every, return_event_exposure=True, **policy_options,
    )
    simulation[3]['generation'] = dict(
        risk_scaling='per_bet_edge_ratio_v1', execution_prices='saved_intermediate',
        fair_probability_method='power_devig_of_sampled_prices',
        prediction_model_type=prediction_model_type,
        prediction_model_path=str(getattr(config, f'model_{prediction_model_type}_path')),
        policy_selection='nearest_closing_price' if use_nearest_closing_params else search_market,
        policies=policy_summary, use_decay=True, seed=seed,
        n_paths=len(paths), sampled_paths_file=str(sampled_paths_file),
    )
    print(f'Betting-path evaluation: {perf_counter() - started:.1f}s', flush=True)
    return {"intermediate": simulation}


if __name__ == "__main__":
    sampled_paths_file = settings.data_dir / "bootstrapped_paths" / "sampled_events_prices_765_val"
    # The existing saved directory has a leading space in its name.
    if not sampled_paths_file.is_dir():
        sampled_paths_file = sampled_paths_file.with_name(" sampled_events_prices_765_val")
    save_path = settings.data_dir / "montecarlo_results" / "intermediate_results.pkl"
    
    results = run_simulation(
            sampled_paths_file=sampled_paths_file,
            params_dir=settings.data_dir / "kelly_params",
            prediction_model_type="open",
            use_nearest_closing_params=True,
            n_paths=5000,
            initial_bankroll=500,
            len_temp=1,
            seed=42,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(results, save_path)

