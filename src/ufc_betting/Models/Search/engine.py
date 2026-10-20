import numpy as np
import pandas as pd

from pathlib import Path
from typing import Mapping

from ufc_betting.config import config
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.Models.Search.frozen_betting import (
    FrozenBettingSearch, JointDecayBettingSearch, validate_score_settings, path_mean_event_returns, path_mean_log_growth, path_max_drawdowns, path_ulcer_indices,
)
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.BettingStrategy.MonteCarlo.bootstrap_paths import new_bootstrap_directory
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths

from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput

ODDS_TYPES = config.odds_types

def optimize_kelly_params(*, validation_data, n_trials=100, k=1000,
                 risk_penalty=.25, initial_bankroll=500., seed=42,
                 parallel=True,
                 bootstrap_mode='generate', bootstrap_dir=None,
                 sample_with_replacement=True, use_decay=False,
                 opening_params=None, opening_data=None, max_decay=50., params_dir=None,
                 sampled_paths=None, score_type="mean_std", wealth_floor=.001, negative_mean_penalty=1000., drawdown_penalty=0., drawdown_tail_fraction=.1, ulcer_penalty=0.):
    """Optimize Kelly controls, optionally jointly with four decay rates.

    "event_mean_std" averages total event bankroll returns within each path,
    then scores mean(path averages) - risk_penalty * sample SD(path averages).
    All scheduled events count equally, with zero returns after ruin/no bets.
    No Sharpe cap, log transform, or division by sqrt(number of paths) is used.

    validation_data must contain exactly one market: open, close1, or close2.
    Returns that market's result directly, including odds_type and best_trial_number.
    "event_log_growth" averages log bankroll changes per scheduled event, then
    across paths, minus drawdown_penalty * mean of the worst drawdown_tail_fraction
    of path maximum drawdowns (default 10%; ceil(fraction * paths) paths).
    ulcer_penalty subtracts its weight times the mean per-path Ulcer Index
    (RMS fractional drawdown over all post-event balances, including no-bet and
    post-ruin events). It defaults to zero and only applies to event_log_growth.
    drawdown_penalty defaults to zero; drawdown uses actual balances including
    initial wealth. Log wealth ratios use wealth_floor for scoring only. No SD
    penalty applies; no-bet events remain in the denominator.
    "log_mean_roi" takes log(mean(path-average event ROIs)); negative means
    score -negative_mean_penalty (default 1000), zero means score -infinity.
    "path_mean_sharpe" scores mean(path-average event ROIs) / sample SD of those averages.
    Event ROI = total net profit / total stakes, including each parlay ticket once.
    No-stake events are excluded; a path with no bets causes the trial to be pruned.
    No cap or risk penalty applies. Zero-return paths score zero; trials with a
    nonzero mean and zero path SD are pruned because their ratio is undefined.
    "mean_std" uses terminal returns; "log_growth" uses mean log wealth ratio.
    wealth_floor is a positive wealth ratio below 1, used only in log scoring;
    risk_penalty applies to mean_std and event_mean_std. Actual bankrolls are never floored.
    parallel is retained for call compatibility; a single study runs directly.
    First call with {'open': ...}, use_decay=False. Then call separately for closing
    markets with use_decay=True and the fixed opening_params (BettingParams,
    dictionary, or JSON path). Each closing market gets an independent study.
    opening_data optionally supplies original opening prices matched by fight;
    otherwise closing predictions must already contain dec_open_red/blue.
    Each joint trial varies closing Kelly controls and four log1p decay rates.
    The calculator applies decay, then closing z, then the joint exposure cap.
    params_dir saves portable <market>.json files, including decoded decay
    rates and fixed opening controls. Search diagnostics are returned in memory.
    sampled_paths accepts a list of prediction DataFrames, its trusted pickle
    path, or a market -> list/pickle mapping. All supplied paths are used (k is
    ignored), without event resampling or inference. Saved rows must carry each
    market's predictions and, for decay, original opening odds. validation_data
    remains the reference dataset for the ordinary backtest diagnostics.

    Call:
        results = optimize_kelly_params(
            validation_data={"open": open_df},
            n_trials=100, k=1000,
            risk_penalty={"open": 0.25, "close1": 0.4, "close2": 0.4},
            bootstrap_mode="save",  # "load" reuses saved paths; "generate" skips saving
            bootstrap_dir="Data/bootstrapped_events/validation_v1",
        )

    Returns:
        A flat result dictionary containing:
        - odds_type, best_trial_number: market and zero-based best Optuna trial
        - betting_data, validation_predictions: fight DataFrames
        - event_results: date, moneyline_return, parlay_return, event_return, bankroll
        - best_params, best_value: optimized parameters and objective score
        - studies: {"full_validation": Optuna study}
        - kelly_sim_input: KellySimInput with .data and .kwargs
        - kelly_sim_kwargs: simulation arguments
        - bootstrap_dir, bootstrap_metadata, risk_penalty, mode, evaluation

    Required validation columns (each market accepts a DataFrame or CSV path):
        - date, fighter_red, fighter_blue, winner (0=blue, 1=red)
        - dec_<market>_blue, dec_<market>_red
        - dec_fair_<market>_blue, dec_fair_<market>_red
        - proba_red OR choice_proba plus choice_fighter_bool/pred_winner (0/1)
        - prediction_se OR se OR proba_se
        - open_blue, open_red OR dec_open_blue, dec_open_red
        - event_name or event_type when generating schedules without replacement
        Optional: risk_red_proba or risk_choice_proba for separate risk probabilities.
    """

    if (not isinstance(validation_data, Mapping) or len(validation_data) != 1
            or not set(validation_data) <= set(ODDS_TYPES)):
        raise ValueError('validation_data must contain exactly one market: open, close1, or close2')
    if bootstrap_mode not in ('generate', 'save', 'load'):
        raise ValueError('bootstrap_mode must be generate, save, or load')
    if bootstrap_mode == 'load' and bootstrap_dir is None:
        raise ValueError('bootstrap_dir is required to load saved paths')
    if bootstrap_mode == 'save' and bootstrap_dir is None:
        bootstrap_dir = new_bootstrap_directory()
    bootstrap_root = Path(bootstrap_dir) if bootstrap_dir is not None else None
    if sampled_paths is not None and (bootstrap_mode != 'generate' or bootstrap_dir is not None):
        raise ValueError('sampled_paths replaces bootstrap_mode/bootstrap_dir; omit those arguments')
    validate_score_settings(score_type, wealth_floor, negative_mean_penalty=negative_mean_penalty, drawdown_penalty=drawdown_penalty, drawdown_tail_fraction=drawdown_tail_fraction, ulcer_penalty=ulcer_penalty)
    markets = [market for market in ODDS_TYPES if market in validation_data]
    if isinstance(sampled_paths, Mapping) and set(markets) != set(sampled_paths):
        raise ValueError('sampled_paths must contain exactly the requested market')
    if use_decay:
        if 'open' in markets or opening_params is None:
            raise ValueError('use_decay=True requires closing markets and fixed opening_params')
        opening_params = BettingParams.from_value(opening_params, require_complete=True)
    elif opening_params is not None or opening_data is not None:
        raise ValueError('opening_params/opening_data require use_decay=True')

    if isinstance(risk_penalty, Mapping):
        if not set(markets) <= set(risk_penalty):
            raise ValueError('risk_penalty must contain each requested market')
        penalties = {market: risk_penalty[market] for market in markets}
    else:
        penalties = dict.fromkeys(markets, risk_penalty)
    for market, value in penalties.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'risk_penalty for {market} must be finite and nonnegative') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'risk_penalty for {market} must be finite and nonnegative')
        penalties[market] = value
    
    def run_market(odds_type):
        predictions = read_frozen_predictions(validation_data[odds_type])
        predictions['proba_blue'] = 1 - predictions.proba_red
        predictions['pred_winner'] = (predictions.proba_red >= .5).astype(int)
        
        search_type = JointDecayBettingSearch if use_decay else FrozenBettingSearch
        extra = dict(opening_params=opening_params, opening_data=opening_data,
                     max_decay=max_decay) if use_decay else {}
        search = search_type(
            predictions, odds_type, k=k, initial_bankroll=initial_bankroll,
            seed=seed, risk_penalty=penalties[odds_type], score_type=score_type, wealth_floor=wealth_floor, negative_mean_penalty=negative_mean_penalty, drawdown_penalty=drawdown_penalty, drawdown_tail_fraction=drawdown_tail_fraction, ulcer_penalty=ulcer_penalty,
            sample_with_replacement=sample_with_replacement, bootstrap_mode=bootstrap_mode,
            bootstrap_dir=bootstrap_root / odds_type if bootstrap_root is not None else None,
            sampled_paths=sampled_paths[odds_type] if isinstance(sampled_paths, Mapping) else sampled_paths, **extra)
        study = search.run(n_trials=n_trials, seed=seed)
        controls = search.betting_params(study.best_params)
        params_path = None
        if params_dir is not None:
            params_path = Path(params_dir) / f'{odds_type}.json'
            params_path.parent.mkdir(parents=True, exist_ok=True)
            controls.save(params_path)
            print(f'{odds_type}: saved best trial {study.best_trial.number} parameters to {params_path}', flush=True)
        predictions = search.frame.copy()
        # Keep the saved original prices available to the shared decay backtest.
        simulation_extra = dict(opening_data=predictions[
            ['date', 'fighter_red', 'fighter_blue', 'dec_open_red', 'dec_open_blue']].copy()) if use_decay else {}
        simulation_input = KellySimInput.from_predictions(
            predictions, controls, odds_type, initial_bankroll, calc_parlay=True, **simulation_extra)
        betting_data, kwargs = simulation_input.to_simulation_args()

        events = search.event_returns(study.best_params)
        events['bankroll'] = simulate_bankroll_paths(
            events, np.arange(len(events))[None, :], initial_bankroll)['bankroll'][0, 1:]
        result = dict(
            odds_type=odds_type,
            betting_data=betting_data, kelly_sim_kwargs=kwargs,
            kelly_sim_input=simulation_input,
            event_results=events, studies={'full_validation': study},
            best_params=controls.to_dict() if use_decay else study.best_params.copy(), best_value=study.best_value,
            best_trial_number=study.best_trial.number, trial_params=study.best_params.copy(),
            betting_params=controls, use_decay=use_decay,
            evaluation='Full validation tuning results; not held-out performance')
        result['validation_predictions'] = predictions
        result['risk_penalty'] = penalties[odds_type]
        result['score_type'] = score_type
        result['wealth_floor'] = float(wealth_floor)
        if score_type == 'log_mean_roi':
            result['negative_mean_penalty'] = float(negative_mean_penalty)
        result['mode'] = 'full_validation'
        result['bootstrap_dir'] = str(search.bootstrap_dir) if search.bootstrap_dir is not None else None
        result['bootstrap_metadata'] = search.bootstrap_metadata
        if search.sampled_paths is not None:
            sampled_result = search.simulate(study.best_params)
            result['sampled_bankroll_paths'] = sampled_result['bankroll']
            result['sampled_path_results'] = pd.DataFrame(dict(
                path_id=np.arange(len(search.sampled_paths)),
                terminal_return=sampled_result['terminal_returns'],
                ending_bankroll=sampled_result['bankroll'][:, -1]))
            result['evaluation'] = 'Tuning on saved synthetic cards; event_results is the reference-data backtest'
        if score_type in ('path_mean_sharpe', 'log_mean_roi'):
            scored = sampled_result if search.sampled_paths is not None else search.simulate(study.best_params)
            result['path_mean_sharpe_basis'] = 'equal_weight_event_roi'
            for name in ('event_rois', 'path_mean_event_rois', 'betting_event_counts', 'betting_event_coverage'):
                result[name] = scored[name]
            if search.sampled_paths is not None:
                result['sampled_path_results']['mean_event_roi'] = scored['path_mean_event_rois']
                result['sampled_path_results']['betting_event_coverage'] = scored['betting_event_coverage']
        if score_type == 'event_mean_std':
            scored = sampled_result if search.sampled_paths is not None else search.simulate(study.best_params)
            result['path_mean_event_returns'] = path_mean_event_returns(scored['bankroll'])
            if search.sampled_paths is not None:
                result['sampled_path_results']['mean_event_return'] = result['path_mean_event_returns']
        if score_type == 'event_log_growth':
            scored = sampled_result if search.sampled_paths is not None else search.simulate(study.best_params)
            means = path_mean_log_growth(scored['bankroll'], wealth_floor)
            result['path_mean_event_log_growth'] = means
            result['geometric_growth_per_event'] = float(np.expm1(means.mean()))
            result['ulcer_penalty'] = float(ulcer_penalty)
            result['path_ulcer_indices'] = path_ulcer_indices(scored['bankroll'])
            result['drawdown_penalty'] = float(drawdown_penalty)
            result['drawdown_tail_fraction'] = float(drawdown_tail_fraction)
            result['path_max_drawdowns'] = path_max_drawdowns(scored['bankroll'])
            if search.sampled_paths is not None:
                result['sampled_path_results']['ulcer_index'] = result['path_ulcer_indices']
                result['sampled_path_results']['max_drawdown'] = result['path_max_drawdowns']
                result['sampled_path_results']['mean_event_log_growth'] = means
                result['sampled_path_results']['geometric_growth_per_event'] = np.expm1(means)
        result['objective_breakdown'] = pd.DataFrame([dict(
            mean_terminal_return_pct=100 * study.best_trial.user_attrs['mean_terminal_return'],
            std_terminal_return_pct=100 * study.best_trial.user_attrs['std_terminal_return'],
            risk_penalty=penalties[odds_type], score_type=score_type, objective_score=study.best_value,
            wealth_floor=float(wealth_floor),
            floored_path_fraction=study.best_trial.user_attrs['floored_path_fraction'],
            loss_probability=study.best_trial.user_attrs['loss_probability'])])
        if score_type == 'mean_std':
            result['objective_breakdown']['objective_pct'] = 100 * study.best_value
        elif score_type == 'event_log_growth':
            for name in ('mean_event_log_growth', 'mean_max_drawdown', 'worst_tail_max_drawdown', 'drawdown_tail_path_count', 'drawdown_penalty_amount', 'mean_ulcer_index', 'ulcer_penalty_amount'):
                result['objective_breakdown'][name] = study.best_trial.user_attrs[name]
            result['objective_breakdown']['ulcer_penalty'] = float(ulcer_penalty)
            result['objective_breakdown']['drawdown_penalty'] = float(drawdown_penalty)
            result['objective_breakdown']['drawdown_tail_fraction'] = float(drawdown_tail_fraction)
            result['objective_breakdown']['geometric_growth_per_event_pct'] = 100 * result['geometric_growth_per_event']
        elif score_type == 'log_growth':
            result['objective_breakdown']['mean_log_growth'] = study.best_value
        elif score_type in ('event_mean_std', 'path_mean_sharpe', 'log_mean_roi'):
            if score_type == 'event_mean_std':
                result['objective_breakdown']['objective_pct'] = 100 * study.best_value
            else:
                result['objective_breakdown'][score_type] = study.best_value
                if score_type == 'log_mean_roi':
                    result['objective_breakdown']['negative_mean_penalty'] = float(negative_mean_penalty)
            metric = 'event_roi' if score_type in ('path_mean_sharpe', 'log_mean_roi') else 'event_return'
            for name in ('mean_path_mean_' + metric, 'std_path_mean_' + metric):
                result['objective_breakdown'][name + '_pct'] = 100 * study.best_trial.user_attrs[name]
        if params_path is not None:
            result['params_path'] = str(params_path)
        return result

    return run_market(markets[0])
