import numpy as np 
import pandas as pd 
from pathlib import Path 

from ufc_betting.config import settings 


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
