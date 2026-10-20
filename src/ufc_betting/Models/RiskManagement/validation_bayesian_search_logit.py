"""Tune betting controls once on the entire notebook validation set.

Loads the opening Logit, post-selection inference Logit, scaler, and encoder
saved by lr_open_close.ipynb as a fallback. Available monthly walk-forward
alphas are used to fit fresh opening models on strictly prior training data.
Walk-forward mode retunes betting controls using expanding validation history.
The objective and parameter ranges come directly from FrozenBettingSearch:
mean bootstrap terminal return - risk_penalty * sample standard deviation.
Reported returns are validation tuning results, not held-out performance.

Example (from the repository root):
    python src/ufc_betting/Models/RiskManagement/validation_bayesian_search_logit.py --n-trials 100
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.delayed_bayesian_search import (
    FrozenBettingSearch, ODDS_TYPES,
)
from ufc_betting.Models.RiskManagement.delayed_walk_forward_logit import (
    kelly_simulation_input, validate_alpha_history, fit_opening_bundle,
    predict_odds, prepare_history,
)


def discover_alpha_history(search_dir=None):
    """Combine prior runs' alpha histories; conflicting selections need an explicit file."""
    root = Path(search_dir) if search_dir is not None else settings.data_dir / 'search_results'
    histories = []
    for path in sorted(root.rglob('alpha_history.csv')):
        history = validate_alpha_history(pd.read_csv(path, float_precision='round_trip'))
        history['alpha_source'] = str(path)
        histories.append(history)
    if not histories:
        return pd.DataFrame(columns=['effective_date', 'training_cutoff', 'alpha'])
    combined = pd.concat(histories, ignore_index=True)
    for date, group in combined.groupby('effective_date'):
        columns = ['alpha', 'training_cutoff']
        if 'training_start' in combined:
            columns.append('training_start')
        if len(group[columns].drop_duplicates()) > 1:
            raise ValueError(f'Conflicting saved alphas for {date.date()}; supply --alpha-history')
    return validate_alpha_history(combined.drop_duplicates('effective_date'))


def apply_monthly_logit_models(predictions, training_data, alpha_history, odds_type='close1',
                               *, model_refit_months=1):
    """Refit in every Nth calendar month, anchored to the first validation month.

    Fits each eligible selection once, before betting trials, using the saved
    training interval. Hold that model through the remaining months of its
    interval. Use the first saved selection in each scheduled month, only
    from its effective date onward. Missing selections (and events before
    availability) use the opening fallback. Fit failures are surfaced.
    """
    if (isinstance(model_refit_months, bool)
            or not isinstance(model_refit_months, (int, np.integer))
            or model_refit_months < 1):
        raise ValueError('model_refit_months must be a positive integer')
    data = predictions.copy()
    data['date'] = pd.to_datetime(data.date).dt.normalize()
    data['prediction_source'] = 'saved_open_logit'
    data['model_alpha'] = np.nan
    data['alpha_effective_date'] = pd.NaT
    data['model_training_cutoff'] = pd.NaT
    data['alpha_source'] = ''
    if alpha_history.empty:
        return data.sort_values('date', kind='stable').reset_index(drop=True)
    history = validate_alpha_history(alpha_history)
    assignments = np.full(len(data), -1, dtype=int)
    origin = data.date.min().to_period('M')
    selected_months = set()
    for index, row in history.iterrows():
        month = row.effective_date.to_period('M')
        offset = month.ordinal - origin.ordinal
        if offset < 0 or offset % model_refit_months or month in selected_months:
            continue
        selected_months.add(month)
        end = (month + int(model_refit_months)).start_time
        eligible = (data.date >= row.effective_date) & (data.date < end)
        assignments[eligible.to_numpy()] = index
    parts = [data.loc[assignments == -1].copy()]
    train = training_data.copy()
    train['date'] = pd.to_datetime(train.date).dt.normalize()
    for index in np.unique(assignments[assignments >= 0]):
        row = history.iloc[index]
        mask = train.date < row.training_cutoff
        if 'training_start' in history:
            mask &= train.date >= row.training_start
        print(f'{row.effective_date.date()}: refitting opening Logit with saved alpha '
              f'{row.alpha:.6g} on {int(mask.sum())} prior fights', flush=True)
        bundle = fit_opening_bundle(train.loc[mask], row.training_cutoff, row.alpha)
        part = predict_odds(data.loc[assignments == index], bundle, odds_type)
        part['prediction_source'] = 'refitted_monthly_open_logit'
        part['alpha_effective_date'] = row.effective_date
        part['model_training_cutoff'] = row.training_cutoff
        part['alpha_source'] = row.get('alpha_source', 'supplied_alpha_history')
        parts.append(part)
    return pd.concat([part for part in parts if not part.empty], ignore_index=True).sort_values(
        'date', kind='stable').reset_index(drop=True)


def load_validation_predictions(input_path, odds_type='close1', *, alpha_history=None,
                                training_input=None, search_dir=None,
                                use_monthly_models=True, model_refit_months=1):
    """Recompute predictions using the notebook's saved preprocessing artifacts.

    The input is an already selected validation cohort; every row is retained.
    By default discover other runs' monthly alphas and refit when available.
    alpha_history accepts a DataFrame or CSV path; an empty DataFrame means
    no monthly selections. Set use_monthly_models=False for saved-model only.
    Close prices replace only the opening market-probability feature, as in
    the notebook. Inference probabilities and SE use the separate saved model.
    """
    if odds_type not in ODDS_TYPES:
        raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
    frame = pd.read_csv(input_path)
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
    frame = frame.sort_values('date', kind='stable').reset_index(drop=True)
    keys = ['date', 'fighter_red', 'fighter_blue']
    if frame.empty or frame.date.isna().any() or frame.duplicated(keys).any():
        raise ValueError('Validation requires nonempty, dated, unique fights')

    # Fail rather than silently removing rows from the requested full cohort.
    training = pd.read_csv(settings.data_dir / 'model_results/train_logit_open.csv')
    training['date'] = pd.to_datetime(training.date).dt.normalize()
    if not frame.merge(training[keys], on=keys, how='inner').empty:
        raise ValueError('Validation contains opening-model training fights')

    model = sm.load(config.model_open_path)
    inference = sm.load(config.model_open_se_path)
    scaler = joblib.load(config.scaler_open_path)
    encoder = joblib.load(config.encoder_open_path)
    features = frame[list(config.open_feats)].copy()
    features['proba_fair_open_diff'] = frame[f'proba_fair_{odds_type}_diff']
    numeric = list(scaler.feature_names_in_)
    categorical = list(encoder.feature_names_in_)
    scaled = pd.DataFrame(scaler.transform(features[numeric]),
                          columns=numeric, index=frame.index)
    encoded = pd.DataFrame(encoder.transform(features[categorical]),
                           columns=encoder.get_feature_names_out(categorical),
                           index=frame.index)
    design = sm.add_constant(pd.concat([scaled, encoded], axis=1),
                             has_constant='add')
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError('Validation model features must be finite')
    frame['proba_red'] = np.asarray(model.predict(design[model.model.exog_names]))
    prediction = inference.get_prediction(design[inference.model.exog_names]).summary_frame()
    frame['risk_red_proba'] = prediction['predicted'].to_numpy()
    frame['prediction_se'] = prediction['se'].to_numpy()
    frame['proba_blue'] = 1 - frame.proba_red
    frame['pred_winner'] = (frame.proba_red >= .5).astype(int)
    frame['odds_type'] = odds_type
    frame['prediction_source'] = 'saved_open_logit'
    if use_monthly_models:
        if alpha_history is None:
            alpha_history = discover_alpha_history(search_dir)
        elif not isinstance(alpha_history, pd.DataFrame):
            path = Path(alpha_history)
            alpha_history = pd.read_csv(path, float_precision='round_trip')
            alpha_history['alpha_source'] = str(path)
        training = pd.DataFrame()
        if not alpha_history.empty:
            training_path = (Path(training_input) if training_input is not None else
                             settings.data_dir / 'training_data/entire_odds_stats_2026-03-07.csv')
            training, _ = prepare_history(pd.read_csv(training_path))
        frame = apply_monthly_logit_models(frame, training, alpha_history, odds_type,
                                           model_refit_months=model_refit_months)
    return frame


def walk_forward_validation(predictions, odds_type='close1', *, burn_in_months=6,
                            refit_months=3, n_trials=100, k=1000,
                            risk_penalty=.25, initial_bankroll=500., seed=42):
    """Tune on ALL prior validation events, then bet until the next refit.

    Burn-in starts at the first validation event's date. Boundaries are that
    date plus calendar-month offsets; history is strictly before each boundary
    and evaluation is [boundary, next_boundary). Burn-in events are used only
    for tuning. Later refits retain them and add every newly observed event.
    `refit_months` controls betting-parameter searches. Monthly model selection
    occurs in load_validation_predictions; historical predictions stay frozen.
    Empty evaluation intervals are skipped. The last interval may be partial.
    Returns event_results (one continuous bankroll), refit_results, studies,
    betting_data (post-burn-in fights with best_* controls), and kelly_sim_kwargs.
    Pass the latter two directly to simulate_kelly:
        simulate_kelly(result['betting_data'], **result['kelly_sim_kwargs'])
    """
    for name, value in [('burn_in_months', burn_in_months), ('refit_months', refit_months)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    # Validate the entire cohort before any searches, including burn-in rows.
    validated = FrozenBettingSearch(predictions, odds_type=odds_type, k=k,
                                    initial_bankroll=initial_bankroll, seed=seed,
                                    risk_penalty=risk_penalty)
    frame = validated.frame
    origin = frame.date.min()
    first_start = origin + pd.DateOffset(months=int(burn_in_months))
    if first_start > frame.date.max():
        raise ValueError('No validation events remain after the burn-in period')
    paths, refits, studies = [], [], {}
    betting_frames = []
    offset = int(burn_in_months)
    while origin + pd.DateOffset(months=offset) <= frame.date.max():
        start = origin + pd.DateOffset(months=offset)
        end = origin + pd.DateOffset(months=offset + int(refit_months))
        history = frame.loc[frame.date < start]
        evaluation = frame.loc[(frame.date >= start) & (frame.date < end)]
        offset += int(refit_months)
        if evaluation.empty:
            continue
        refit_seed = seed + len(refits)
        search = FrozenBettingSearch(history, odds_type=odds_type, k=k,
                                     initial_bankroll=initial_bankroll, seed=refit_seed,
                                     risk_penalty=risk_penalty)
        print(f'{start.date()}: tuning on all {history.date.nunique()} prior events; '
              f'evaluating {evaluation.date.nunique()} events', flush=True)
        study = search.run(n_trials=n_trials, seed=refit_seed)
        evaluator = FrozenBettingSearch(evaluation, odds_type=odds_type, k=2,
                                        initial_bankroll=initial_bankroll,
                                        seed=refit_seed, risk_penalty=risk_penalty)
        events = evaluator.event_returns(study.best_params)
        events['refit_start'] = start
        betting_frame = evaluation.copy()
        betting_frame['refit_start'] = start
        betting_frame['pred_winner'] = (betting_frame.proba_red >= .5).astype(int)
        betting_frame['proba_blue'] = 1 - betting_frame.proba_red
        for name, value in study.best_params.items():
            events[name] = value
            betting_frame[f'best_{name}'] = value
        betting_frames.append(betting_frame)
        paths.append(events)
        studies[str(start.date())] = study
        refits.append(dict(refit_start=start, next_refit=end,
                           history_start=history.date.min(), history_end=history.date.max(),
                           history_fights=len(history), history_events=history.date.nunique(),
                           evaluation_end=evaluation.date.max(),
                           evaluation_events=evaluation.date.nunique(),
                           best_value=study.best_value, **study.best_params))
    events = pd.concat(paths, ignore_index=True).sort_values('date').reset_index(drop=True)
    events['bankroll'] = initial_bankroll * (1 + events.event_return).cumprod()
    betting_data, kelly_kwargs = kelly_simulation_input(
        pd.concat(betting_frames, ignore_index=True), odds_type, initial_bankroll)
    return dict(event_results=events, refit_results=pd.DataFrame(refits), studies=studies,
                betting_data=betting_data, kelly_sim_kwargs=kelly_kwargs,
                burn_in_start=origin, first_evaluation_start=first_start)


def portable_kelly_input(betting_data, kwargs, best_params):
    """Bind price aliases to the dataset and fixed betting controls to kwargs.

    Copies inputs. Preserves original market columns and best_* provenance;
    disables row parameter overrides so swapping kwargs actually swaps controls.
    """
    data, args = betting_data.copy(), dict(kwargs)
    for argument, prefix in (('fair_decimal_cols', 'bet_fair_decimal'),
                             ('real_decimal_cols', 'bet_real_decimal')):
        names = [f'{prefix}_blue', f'{prefix}_red']
        data[names] = data[args[argument]].to_numpy(copy=True)
        args[argument] = names
    args.update(
        max_drawdown=float(best_params['mdd']), N=int(best_params['N']),
        parlay_mdd=float(best_params['mdd_parlay']), N_parlay=int(best_params['N_parlay']),
        z=float(best_params['z']),
        moneyline_american_bounds=(float(best_params['moneyline_min_american']),
                                  float(best_params.get('moneyline_max_american', np.inf))),
        use_walk_forward_params=False, adjust_mdd_by_edge=False,
    )
    args.pop('monthly_params', None)
    return data, args


def run_all_odds(*, mode='walk_forward', burn_in_months=6, refit_months=3, n_trials=100, k=1000,
                 risk_penalty=.25, initial_bankroll=500., seed=42,
                 input_paths=None, alpha_history=None, training_input=None,
                 search_dir=None, use_monthly_models=True, model_refit_months=1, parallel=True,
                 run_kelly=True, output_dir=None, bet_probability_source='regularized'):
    """Notebook entry point: run open, close1, and close2 in one call.

    Each market has an independent search and bankroll. By default the three
    runs execute concurrently; parallel=False runs them sequentially instead.
    input_paths optionally maps market names to validation CSVs. Omitted paths
    use the same notebook exports as the CLI. Each cohort keeps its own dates.
    output_dir optionally saves results in open/, close1/, and close2/ folders.
    risk_penalty accepts one nonnegative finite number for all markets, or a
    mapping containing exactly open, close1, and close2 with separate penalties.
    mode='walk_forward' tunes on expanding prior history after burn-in.
    mode='full_validation' tunes once on all validation fights, ignoring
    burn_in_months/refit_months. Its Kelly results are in-sample tuning results.
    Monthly model selection is independent of this betting-search setting.
    model_refit_months controls Logit updates (default 1), anchored to the first
    validation calendar month. The selected model is held between updates.
    bet_probability_source='regularized' (default) or 'inference' selects the
    probabilities for winner selection, EV, Kelly sizing, and parlay selection
    in both tuning and simulation. The z adjustment always uses inference
    probabilities and inference SE, independently of this setting.

    Example in a notebook:
        results = run_all_odds(burn_in_months=6, refit_months=3)
        open_betting_data = results['open']['betting_data']
        close1_kelly = results['close1']['kelly_results']
        close2_parlays = results['close2']['parlay_results']

    Set run_kelly=False to return inputs without running simulate_kelly.

    In full_validation mode, kwargs carry fixed betting controls and all three
    datasets expose common price aliases. To apply close1 controls to opening
    prices/predictions (without retuning):
        simulate_kelly(results['open']['betting_data'],
                       **results['close1']['kelly_sim_kwargs'])
    Any of the nine data/parameter combinations is supported in this mode.
    Walk-forward mode retains date-specific best_* controls on its own data;
    swapping kwargs in that mode does not transfer a parameter schedule.
    """
    if mode not in ('walk_forward', 'full_validation'):
        raise ValueError("mode must be 'walk_forward' or 'full_validation'")
    if bet_probability_source not in ('regularized', 'inference'):
        raise ValueError("bet_probability_source must be 'regularized' or 'inference'")
    input_paths = {} if input_paths is None else dict(input_paths)
    unknown = set(input_paths) - set(ODDS_TYPES)
    if unknown:
        raise ValueError(f'Unknown input odds types: {sorted(unknown)}')
    if isinstance(risk_penalty, Mapping):
        if set(risk_penalty) != set(ODDS_TYPES):
            raise ValueError('risk_penalty must contain exactly open, close1, and close2')
        penalties = dict(risk_penalty)
    else:
        penalties = dict.fromkeys(ODDS_TYPES, risk_penalty)
    for market, value in penalties.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'risk_penalty for {market} must be finite and nonnegative') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'risk_penalty for {market} must be finite and nonnegative')
        penalties[market] = value
    # Resolve one consistent snapshot of the saved alpha selections for all runs.
    if use_monthly_models:
        if alpha_history is None:
            alpha_history = discover_alpha_history(search_dir)
        elif not isinstance(alpha_history, pd.DataFrame):
            source = Path(alpha_history)
            alpha_history = pd.read_csv(source, float_precision='round_trip')
            alpha_history['alpha_source'] = str(source)

    def run_market(odds_type):
        filename = ('test_logit_open.csv' if odds_type == 'open'
                    else f'val_set_data_open_with_{odds_type}.csv')
        path = input_paths.get(odds_type, settings.data_dir / 'model_results' / filename)
        predictions = load_validation_predictions(
            path, odds_type, alpha_history=alpha_history, training_input=training_input,
            search_dir=search_dir, use_monthly_models=use_monthly_models,
            model_refit_months=model_refit_months)
        if bet_probability_source == 'inference':
            predictions['proba_red'] = predictions['risk_red_proba']
        predictions['proba_blue'] = 1 - predictions.proba_red
        predictions['pred_winner'] = (predictions.proba_red >= .5).astype(int)
        predictions['bet_probability_source'] = bet_probability_source
        if mode == 'walk_forward':
            result = walk_forward_validation(
                predictions, odds_type, burn_in_months=burn_in_months,
                refit_months=refit_months, n_trials=n_trials, k=k,
                risk_penalty=penalties[odds_type], initial_bankroll=initial_bankroll, seed=seed)
        else:
            search = FrozenBettingSearch(
                predictions, odds_type, k=k, initial_bankroll=initial_bankroll,
                seed=seed, risk_penalty=penalties[odds_type])
            study = search.run(n_trials=n_trials, seed=seed)
            betting_predictions = predictions.copy()
            betting_predictions['proba_blue'] = 1 - betting_predictions.proba_red
            betting_predictions['pred_winner'] = (betting_predictions.proba_red >= .5).astype(int)
            for name, value in study.best_params.items():
                betting_predictions[f'best_{name}'] = value
            betting_data, kwargs = kelly_simulation_input(
                betting_predictions, odds_type, initial_bankroll)
            betting_data, kwargs = portable_kelly_input(betting_data, kwargs, study.best_params)
            events = search.event_returns(study.best_params)
            events['bankroll'] = initial_bankroll * (1 + events.event_return).cumprod()
            result = dict(
                betting_data=betting_data, kelly_sim_kwargs=kwargs,
                event_results=events, studies={'full_validation': study},
                best_params=study.best_params.copy(), best_value=study.best_value,
                evaluation='Full validation tuning results; not held-out performance')
        result['validation_predictions'] = predictions
        result['risk_penalty'] = penalties[odds_type]
        result['mode'] = mode
        result['model_refit_months'] = model_refit_months
        result['bet_probability_source'] = bet_probability_source
        if run_kelly:
            from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
            result['kelly_results'], result['parlay_results'] = simulate_kelly(
                result['betting_data'], **result['kelly_sim_kwargs'])
        if output_dir is not None:
            directory = Path(output_dir) / odds_type
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in result.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(directory / f'{name}.csv', index=False)
            for date, study in result['studies'].items():
                study.trials_dataframe().to_csv(directory / f'trials_{date}.csv', index=False)
            (directory / 'kelly_sim_kwargs.json').write_text(
                json.dumps(result['kelly_sim_kwargs'], indent=2), encoding='utf-8')
            (directory / 'run_settings.json').write_text(json.dumps(dict(
                mode=mode, odds_type=odds_type, risk_penalty=penalties[odds_type],
                n_trials=n_trials, k=k, seed=seed, initial_bankroll=initial_bankroll,
                burn_in_months=burn_in_months if mode == 'walk_forward' else None,
                refit_months=refit_months if mode == 'walk_forward' else None,
                use_monthly_models=use_monthly_models,
                model_refit_months=model_refit_months,
                bet_probability_source=bet_probability_source,
            ), indent=2), encoding='utf-8')
        return result

    if parallel:
        with ThreadPoolExecutor(max_workers=len(ODDS_TYPES)) as executor:
            return dict(zip(ODDS_TYPES, executor.map(run_market, ODDS_TYPES)))
    return {odds_type: run_market(odds_type) for odds_type in ODDS_TYPES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Already selected validation CSV')
    parser.add_argument('--odds-type', choices=ODDS_TYPES, default='close1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--k', type=int, default=1000, help='Bootstrap paths per trial')
    parser.add_argument('--risk-penalty', type=float, default=.25)
    parser.add_argument('--initial-bankroll', type=float, default=500.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--walk-forward', action='store_true', help='Use expanding validation history')
    parser.add_argument('--burn-in-months', type=int, default=6)
    parser.add_argument('--refit-months', type=int, default=3, help='Months between betting-parameter searches')
    parser.add_argument('--alpha-history', type=Path, help='Explicit prior-run alpha_history.csv')
    parser.add_argument('--search-dir', type=Path, help='Directory to discover prior alpha histories')
    parser.add_argument('--training-input', type=Path, help='Full historical training CSV for monthly fits')
    parser.add_argument('--saved-model-only', action='store_true', help='Disable monthly alpha lookup')
    parser.add_argument('--model-refit-months', type=int, default=1,
                        help='Calendar months between Logit updates, independent of betting refits')
    args = parser.parse_args()
    filename = ('test_logit_open.csv' if args.odds_type == 'open'
                else f'val_set_data_open_with_{args.odds_type}.csv')
    input_path = args.input or settings.data_dir / 'model_results' / filename
    output = args.output or settings.data_dir / 'search_results' / f'validation_open_{args.odds_type}'
    predictions = load_validation_predictions(
        input_path, args.odds_type, alpha_history=args.alpha_history,
        training_input=args.training_input, search_dir=args.search_dir,
        use_monthly_models=not args.saved_model_only, model_refit_months=args.model_refit_months)
    model_refitted = predictions.prediction_source.eq('refitted_monthly_open_logit').any()
    if args.walk_forward:
        if args.output is None:
            output = output / 'walk_forward'
        result = walk_forward_validation(
            predictions, odds_type=args.odds_type, burn_in_months=args.burn_in_months,
            refit_months=args.refit_months, n_trials=args.n_trials, k=args.k,
            risk_penalty=args.risk_penalty, initial_bankroll=args.initial_bankroll, seed=args.seed)
        output.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output / 'validation_predictions.csv', index=False)
        result['event_results'].to_csv(output / 'event_bankroll.csv', index=False)
        result['refit_results'].to_csv(output / 'refit_results.csv', index=False)
        result['betting_data'].to_csv(output / 'betting_data.csv', index=False)
        (output / 'kelly_sim_kwargs.json').write_text(
            json.dumps(result['kelly_sim_kwargs'], indent=2), encoding='utf-8')
        for date, study in result['studies'].items():
            study.trials_dataframe().to_csv(output / f'trials_{date}.csv', index=False)
        metadata = dict(input=str(input_path), odds_type=args.odds_type,
                        burn_in_months=args.burn_in_months, refit_months=args.refit_months,
                        n_trials=args.n_trials, k=args.k, seed=args.seed,
                        risk_penalty=args.risk_penalty, initial_bankroll=args.initial_bankroll,
                        objective='mean_terminal_return - risk_penalty * std_terminal_return',
                        history='All validation events strictly before each refit',
                        model_refitted=bool(model_refitted),
                        model_refit_months=args.model_refit_months,
                        model_policy='Scheduled-month alpha held for interval; otherwise saved opening Logit',
                        first_evaluation_start=str(result['first_evaluation_start'].date()))
        (output / 'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(result['refit_results'].to_string(index=False))
        print(f'Outputs: {output}')
        return
    search = FrozenBettingSearch(predictions, odds_type=args.odds_type, k=args.k,
                                initial_bankroll=args.initial_bankroll,
                                seed=args.seed, risk_penalty=args.risk_penalty)
    print(f'Searching all {len(predictions)} validation fights / '
          f'{predictions.date.nunique()} events', flush=True)
    study = search.run(n_trials=args.n_trials, seed=args.seed)
    events = search.event_returns(study.best_params)
    events['bankroll'] = args.initial_bankroll * (1 + events.event_return).cumprod()
    simulation = search.simulate(study.best_params)
    terminal = simulation['terminal_returns']
    summary = dict(
        objective=study.user_attrs['objective'], best_value=study.best_value,
        best_params=study.best_params, risk_penalty=args.risk_penalty,
        mean_terminal_return=float(terminal.mean()),
        std_terminal_return=float(terminal.std(ddof=1)),
        loss_probability=float((terminal < 0).mean()),
        validation_fights=len(predictions), validation_events=predictions.date.nunique(),
        validation_start=str(predictions.date.min().date()),
        validation_end=str(predictions.date.max().date()),
        input=str(input_path), model=str(config.model_open_path),
        inference_model=str(config.model_open_se_path),
        scaler=str(config.scaler_open_path), encoder=str(config.encoder_open_path),
        odds_type=args.odds_type, n_trials=args.n_trials, k=args.k, seed=args.seed,
        initial_bankroll=args.initial_bankroll, model_refitted=bool(model_refitted),
        model_refit_months=args.model_refit_months,
        evaluation='Full validation tuning results; not held-out performance',
    )
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'validation_predictions.csv', index=False)
    betting_predictions = predictions.copy()
    for name, value in study.best_params.items():
        betting_predictions[f'best_{name}'] = value
    betting_data, kelly_kwargs = kelly_simulation_input(
        betting_predictions, args.odds_type, args.initial_bankroll)
    betting_data.to_csv(output / 'betting_data.csv', index=False)
    (output / 'kelly_sim_kwargs.json').write_text(
        json.dumps(kelly_kwargs, indent=2), encoding='utf-8')
    events.to_csv(output / 'event_bankroll.csv', index=False)
    study.trials_dataframe().to_csv(output / 'trials.csv', index=False)
    pd.DataFrame({'terminal_return': terminal,
                  'ending_bankroll': simulation['bankroll'][:, -1]}).to_csv(
                      output / 'bootstrap_terminal_returns.csv', index=False)
    (output / 'best_params.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {output}')


if __name__ == '__main__':
    main()
