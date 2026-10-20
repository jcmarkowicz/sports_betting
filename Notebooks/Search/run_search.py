
import faulthandler

# Native crashes bypass normal exception tracebacks. Print the active Python
# stacks if the interpreter exits on an access violation during a script run.
if __name__ == "__main__":
    faulthandler.enable(all_threads=True)

from pathlib import Path

import pandas as pd
from ufc_betting.config import config, settings
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.Models.Search.frozen_betting import validate_score_settings
from ufc_betting.BettingStrategy.MonteCarlo.compact_paths import CompactPaths
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import (
    attach_path_predictions, load_sampled_paths, predict_intermediate_paths,
)

import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

def load_search_paths(paths_path, predictions_path, *, search_market, n_paths=100,
                      seed=42, prediction_model_type="open"):
    """Select saved paths first, then attach/infer predictions for their prices.

    Compact inputs are the DIRECTORY containing metadata.json, source.pkl and
    .npy arrays. Legacy trusted .pkl files are also supported, but load in full.
    n_paths=None selects all paths. No fights or prices are resampled here.
    """
    paths_path = Path(paths_path)
    if not paths_path.exists():
        raise FileNotFoundError(
            f"Saved paths not found: {paths_path}. Set paths_path to the output_path "
            "used in src/ufc_betting/BettingStrategy/MonteCarlo/generate_paths.py.")
    print(f"{search_market}: reading saved paths from {paths_path}", flush=True)
    if paths_path.is_dir():
        if not (paths_path / "metadata.json").is_file():
            raise ValueError(f"Incomplete compact path directory: {paths_path}; metadata.json is missing")
        with CompactPaths(paths_path) as saved:
            available = len(saved)
            paths = load_sampled_paths(saved, n_paths=n_paths, random_state=seed)
    else:
        paths = load_sampled_paths(paths_path, n_paths=n_paths, random_state=seed)
        available = "legacy pickle"
    print(f"{search_market}: selected {len(paths)} paths from {available} (seed={seed})", flush=True)
    sampled_prices = [bool(path.attrs.get('generation', {}).get('sample_prices', False))
                      or ('execution_odds_type' in path and path.execution_odds_type.eq('intermediate').any())
                      for path in paths]
    if any(sampled_prices):
        if not all(sampled_prices):
            raise ValueError('Cannot mix recorded-price and sampled-price paths in one search')
        if search_market not in ('close1', 'close2'):
            raise ValueError('Sampled-price paths require search_market="close1" or "close2"; '
                             'fit opening parameters separately using recorded-price paths')
        if prediction_model_type not in ('open', 'close1', 'close2'):
            raise ValueError('prediction_model_type must be open, close1, or close2')
        model = prediction_model_type
        print(f"{search_market}: inferring predictions for saved sampled prices using {model} model", flush=True)
        return predict_intermediate_paths(
            paths, search_market=search_market, prediction_model_type=model,
            prediction_model=getattr(config, f"model_{model}_path"),
            prediction_se_model=getattr(config, f"model_{model}_se_path", None),
            prediction_scalers={model: getattr(config, f"scaler_{model}_path")},
            prediction_encoders={model: getattr(config, f"encoder_{model}_path",
                                                settings.saved_models_dir / f"encoder_{model}.pkl")},
        )
    print(f"{search_market}: attaching recorded-price predictions from {predictions_path}", flush=True)
    return attach_path_predictions(paths, predictions_path, odds_type=search_market)


def run_search(search_market, risk_penalty=0.4, n_trials=100, n_paths=1000,
               initial_bankroll=500, paths_path=None, *, seed=42,
               predictions_path=None, prediction_model_type="open", params_dir=None,
               opening_params_path=None, score_type="mean_std", wealth_floor=.001, negative_mean_penalty=1000., drawdown_penalty=0., drawdown_tail_fraction=.1, ulcer_penalty=0.):
    """Search selected saved paths; detects recorded or sampled prices from the data.

    Recorded-price CSV defaults use the opening model at the requested market.
    For another model, also supply its matching reference predictions_path.
    Closing searches require a previously fitted opening_params_path.
    score_type="event_log_growth" averages log bankroll growth per scheduled
    event, then across paths, minus drawdown_penalty * mean of the worst
    drawdown_tail_fraction of path maximum drawdowns (default 10%, rounded up).
    Drawdown is a fraction (0.4 = 40%) computed from actual bankroll peaks.
    ulcer_penalty subtracts its weight times the mean per-path Ulcer Index
    (RMS fractional drawdown over all post-event balances, including no-bet and
    post-ruin events). It defaults to zero and only applies to event_log_growth.
    drawdown_penalty defaults to zero and only applies to event_log_growth.
    Scoring floors bankroll / initial bankroll at
    wealth_floor; balances are unchanged. No-bet events contribute zero.
    score_type="log_growth" averages log(max(ending / starting bankroll, wealth_floor)).
    wealth_floor is a ratio (0.001 = 0.1% of initial bankroll), not dollars.
    score_type="mean_std" preserves the old terminal-return objective.
    score_type="event_mean_std" averages total event bankroll returns within
    each path, then maximizes mean(path averages) - risk_penalty * their sample SD.
    All events count equally, including no-bet/post-ruin events at zero return.
    risk_penalty applies to mean_std and event_mean_std; no cap is used for either.
    score_type="log_mean_roi" takes log(mean(path-average event ROIs)).
    Negative means score -negative_mean_penalty (default 1000); zero scores -inf.
    score_type="path_mean_sharpe" divides mean(path-average event ROIs) by
    their sample SD across paths. No cap, risk penalty, or annualization applies.
    Event ROI = total net profit / total stakes (parlay ticket counted once).
    No-stake events are excluded; a path with no bets causes trial pruning.
    All-zero ROIs score zero; nonzero-mean, zero-SD trials are pruned.
    """
    validate_score_settings(score_type, wealth_floor, negative_mean_penalty=negative_mean_penalty, drawdown_penalty=drawdown_penalty, drawdown_tail_fraction=drawdown_tail_fraction, ulcer_penalty=ulcer_penalty)
    if search_market not in ('open', 'close1', 'close2'):
        raise ValueError('search_market must be open, close1, or close2')
    paths_path = (Path(paths_path) if paths_path is not None else
                  settings.data_dir / "bootstrapped_paths" / "sampled_fights_recorded_prices_val")
    predictions_path = (Path(predictions_path) if predictions_path is not None else
                        settings.model_predictions_folder / f"val_logit_open_with_{search_market}.csv")
    use_decay = search_market != "open"
    params_dir = Path(params_dir) if params_dir is not None else settings.data_dir / "kelly_params"
    opening_params_path = (Path(opening_params_path) if opening_params_path is not None
                           else params_dir / "open.json")
    if use_decay and not opening_params_path.is_file():
        raise FileNotFoundError(f"Opening parameters not found: {opening_params_path}. Run the open search first.")
    paths = load_search_paths(
        paths_path, predictions_path, search_market=search_market, n_paths=n_paths,
        seed=seed, prediction_model_type=prediction_model_type,
    )
    save_path = params_dir / "search_results" / f"{search_market}_search_results.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{search_market}: JSON -> {params_dir / (search_market + '.json')}", flush=True)
    print(f"{search_market}: full results -> {save_path}", flush=True)

    print(f"{search_market}: starting {n_trials} trials over {len(paths)} paths", flush=True)
    search_results = optimize_kelly_params(
        validation_data={search_market: predictions_path},
        sampled_paths={search_market: paths},  # Use the paths with your chosen predictions attached
        n_trials=n_trials, 
        risk_penalty=risk_penalty,
        score_type=score_type, wealth_floor=wealth_floor, negative_mean_penalty=negative_mean_penalty, drawdown_penalty=drawdown_penalty, drawdown_tail_fraction=drawdown_tail_fraction, ulcer_penalty=ulcer_penalty, 
        initial_bankroll=initial_bankroll,
        use_decay=use_decay,
        opening_params=opening_params_path if use_decay else None,
        params_dir=params_dir,
        parallel=False,
        seed=seed,
    )

    search_results["path_selection"] = dict(
        paths_path=str(paths_path.resolve()), n_paths=len(paths), seed=seed,
        sample_prices=bool(paths[0].attrs.get("generation", {}).get("sample_prices", False)),
        prediction_model_type=prediction_model_type,
    )
    pd.to_pickle(search_results, save_path)
    print(f"{search_market}: saved full search results to {save_path}", flush=True)
    return search_results

def plot_search_results(results, *, show=True):
    """Return {market: {plot_name: figure}}; optionally display the figures."""
    figures = {}

    entries = [(results["odds_type"], results)] if "odds_type" in results else results.items()
    for market, result in entries:
        figures[market] = {}
        study = result["studies"]["full_validation"]
        score_type = result.get("score_type", study.user_attrs.get("score_type", "mean_std"))
        scale = 100.0 if score_type in ("mean_std", "event_mean_std") else 1.0
        target_name = {
            "event_log_growth": "Mean log bankroll growth per event minus drawdown and ulcer penalties",
            "log_growth": "Mean log wealth growth (natural log)",
            "mean_std": "Mean final return minus SD penalty (percentage points)",
            "event_mean_std": "Mean path-average event return minus SD penalty (percentage points)",
            "log_mean_roi": "Log of mean path-average event ROI",
            "path_mean_sharpe": "Sharpe ratio of path-average event ROIs",
        }[score_type]
        for title, plotter in (
            ("Optimization history", plot_optimization_history),
            ("Parameter importance", plot_param_importances),
        ):
            if plotter is plot_param_importances and len([
                trial for trial in study.trials if trial.value is not None
            ]) < 2:
                continue
            axes = plotter(                                    
                study, target=lambda trial: trial.value * scale,
                target_name=target_name,
            )
            figure = (axes.flat[0] if hasattr(axes, "flat") else axes).figure
            for ax in figure.axes:
                legend = ax.get_legend()
                if title == "Optimization history":
                    ax.set_ylabel("Objective")
                    if legend is not None:
                        legend.remove()
                elif legend is not None:
                    for text in legend.get_texts():
                        text.set_text(text.get_text().replace(target_name, "Objective"))
            figure.suptitle(f"{market} - {title}")
            figure.set_size_inches(max(12, figure.get_figwidth()), 5)
            figure.tight_layout()
            figures[market][title.lower().replace(" ", "_")] = figure
            if show:
                plt.show()
    return figures

if __name__ == "__main__":
    # This must match output_path in src/ufc_betting/BettingStrategy/MonteCarlo/generate_paths.py.
    # Select the DIRECTORY itself, not metadata.json, source.pkl, or a .npy file.
    paths_path = settings.data_dir / "bootstrapped_paths" / "sampled_fights_recorded_prices_val"

    # Other saved runs work by changing only paths_path:
    # paths_path = settings.data_dir / "bootstrapped_paths" / "intact_cards_recorded_prices_val"
    # paths_path = settings.data_dir / "bootstrapped_paths" / "sampled_fights_sampled_prices_val"
    # paths_path = settings.data_dir / "bootstrapped_paths" / "intact_cards_sampled_prices_val"
    # Folder names are examples; use the exact name you saved. Price mode is detected.

    # Recorded prices: open first, then closing searches use its saved open.json.
    markets = ("open", "close1", "close2")
    # Sampled prices: use markets = ("close1", "close2") and an existing open.json.
    risk_penalty = {"open": 0.15, "close1": 0.25, "close2": 0.25}
    # Average log bankroll growth over all scheduled events, then across paths.
    # risk_penalty is used only for mean_std and event_mean_std.
    score_kwargs = {
        "score_type": "event_log_growth",
        "drawdown_penalty": 0.067,  # 40% tail-average drawdown subtracts 0.0268.
        "ulcer_penalty": 0.02,  # Starting weight: mean Ulcer Index of 20% subtracts 0.002.
        "drawdown_tail_fraction": 0.33,  # Average the worst 40% of path drawdowns.
        "wealth_floor": 0.001,  # Scoring floor: 0.1% of initial bankroll.
    }
    # Other objectives: "mean_std" (terminal returns), "log_growth", "event_mean_std".
    results = {}
    for market in markets:
        results[market] = run_search(
            search_market=market,
            **score_kwargs,
            paths_path=paths_path,
            n_paths=None,                    # Select 100 saved paths; None uses all.
            seed=42,                        # Same subset for every market.
            n_trials=300,
            risk_penalty=risk_penalty[market],
            initial_bankroll=500,
            prediction_model_type="open",  # Inference model for sampled prices.
            predictions_path=settings.model_predictions_folder / f"val_logit_open_with_{market}.csv",
            params_dir=settings.data_dir / "kelly_params",
            opening_params_path=settings.data_dir / "kelly_params" / "open.json",
        )
    print("Search completed. Results saved to kelly_params/search_results/*.pkl")
