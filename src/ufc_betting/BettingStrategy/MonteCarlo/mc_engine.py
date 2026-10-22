import numpy as np 
import pandas as pd 
import json
from pathlib import Path
from time import perf_counter
from .schedule_sampling import sample_schedule
from .event_types import event_types


def advance_bankroll(bankroll, returns, *, floor=0.):
    """Settle one event, freezing nonpositive balances or balances below floor."""
    bankroll, returns = np.broadcast_arrays(np.asarray(bankroll, float), np.asarray(returns, float))
    if not np.isfinite(bankroll).all() or not np.isfinite(returns).all():
        raise ValueError('Bankroll and returns must be finite')
    active = (bankroll > 0) & (bankroll >= floor)
    return np.where(active, bankroll * (1 + returns), bankroll)


def settle_bankroll_paths(event_returns, initial_bankroll=500.0, *,
                          family_returns=None, batch_size=1):
    """Settle a (paths, events) array, freezing balances once they reach <= 0.

    Returns are fractions of the balance available before each settlement.
    Events in each batch share that starting balance, including a final partial
    batch. Optional named family arrays must sum to event_returns. Outputs have
    one column per batch, plus the initial column for bankroll only.
    A cumulative return <= -1 freezes at its exact crossing value, including
    losses below -100%; all remaining scheduled steps retain that balance.
    """
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be positive and finite')
    if (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError('batch_size must be a positive integer')
    returns = np.asarray(event_returns, dtype=float)
    if returns.ndim != 2 or not all(returns.shape):
        raise ValueError('event_returns must be a nonempty (paths, events) array')
    if not np.isfinite(returns).all():
        raise ValueError('Event bankroll returns must be finite')
    families = {name: np.asarray(values, dtype=float)
                for name, values in (family_returns or {}).items()}
    for values in families.values():
        if values.shape != returns.shape or not np.isfinite(values).all():
            raise ValueError('Family returns must be finite and match event_returns shape')
    if families and not np.allclose(sum(families.values()), returns):
        raise ValueError('Bet-family returns must sum to the portfolio return')

    starts = np.arange(0, returns.shape[1], batch_size)
    batch_returns = np.add.reduceat(returns, starts, axis=1)
    bankroll = np.empty((returns.shape[0], len(starts) + 1), dtype=float)
    bankroll[:, 0] = initial_bankroll
    for step in range(len(starts)):
        previous = bankroll[:, step]
        bankroll[:, step + 1] = advance_bankroll(previous, batch_returns[:, step])
    if not np.isfinite(bankroll).all():
        raise ValueError('Nonfinite simulated bankroll')
    result = dict(bankroll=bankroll, terminal_returns=bankroll[:, -1] / initial_bankroll - 1)
    betting_bankroll = np.where(bankroll[:, :-1] > 0, bankroll[:, :-1], 0.)
    for family, values in families.items():
        profit = betting_bankroll * np.add.reduceat(values, starts, axis=1)
        result[f'{family}_profits'] = profit
        result[f'{family}_cumulative_returns'] = np.cumsum(profit, axis=1) / initial_bankroll
    return result


def simulate_bankroll_paths(events, indices, initial_bankroll=500.0):
    """Select fixed event returns by schedule, then settle each sampled path.

    Repeated event indices remain separate steps. Moneyline and parlay returns,
    when supplied together, are accounted for against the same bankroll.
    Paths that cross -100% retain that crossing loss through the final event;
    they are neither truncated nor clamped to -100%.
    """
    returns = events.event_return.to_numpy(float)
    if not np.isfinite(returns).all():
        raise ValueError('Event bankroll returns must be finite')
    indices = np.asarray(indices)
    if (indices.ndim != 2 or not all(indices.shape)
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(events)).any()):
        raise ValueError('Invalid sampled event indices')
    families = {}
    if {'moneyline_return', 'parlay_return'} <= set(events.columns):
        if not np.allclose(events.moneyline_return + events.parlay_return, returns):
            raise ValueError('Bet-family returns must sum to the portfolio return')
        families = {family: events[f'{family}_return'].to_numpy(float)[indices]
                    for family in ('moneyline', 'parlay')}
    result = settle_bankroll_paths(returns[indices], initial_bankroll,
                                  family_returns=families)
    # Preserve the frozen-search result interface.
    for family in families:
        result.pop(f'{family}_profits')
    return result


def _event_paths(source, n_sims, title_col, rng, mode, path, replace):
    """Persist date assignments only; always use the caller's current fight data."""
    if mode not in ('generate', 'save', 'load'):
        raise ValueError('event_paths_mode must be generate, save, or load')
    cards = {date.strftime('%Y-%m-%d'): card for date, card in source.groupby('date', sort=True)}
    dates = list(cards)
    cohort = sorted([
        [row.date.strftime('%Y-%m-%d'), str(row.fighter_red), str(row.fighter_blue), int(getattr(row, title_col))]
        for row in source.itertuples()
    ])
    scheduling = None
    if not replace:
        typed = source[['date']].assign(event_type=event_types(source)).drop_duplicates()
        if typed.date.duplicated().any():
            raise ValueError('Each date must identify one event type')
        scheduling = dict(rule='balanced_numbered_v1', events=sorted(
            [row.date.strftime('%Y-%m-%d'), row.event_type] for row in typed.itertuples()))
    if mode == 'load':
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('version') != 1 or saved.get('cohort') != cohort:
            raise ValueError('Saved event paths do not match the input fights/dates/title metadata')
        if saved.get('replace', False) != replace:
            raise ValueError('Saved sampling mode differs; generate and save new event paths')
        if not replace and saved.get('scheduling') != scheduling:
            raise ValueError('Saved paths use different event types or an older title-card rule; save new paths')
        paths = saved['paths']
        if len(paths) < n_sims:
            raise ValueError(f'Saved file has {len(paths)} paths; requested {n_sims}')
        paths = paths[:n_sims]
        if any(len(order) != len(dates) or not set(order).issubset(dates)
               or (not replace and sorted(order) != dates) for order in paths):
            raise ValueError('Invalid saved event paths for the selected sampling mode')
    else:
        paths = []
        for _ in range(n_sims):
            scheduled = sample_schedule(source, title_col=title_col, replace=replace,
                                        random_state=int(rng.integers(0, 2**32-1)))
            paths.append([d.strftime('%Y-%m-%d') for d in
                          scheduled.groupby('date', sort=True).source_date.first()])
        if mode == 'save':
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(version=1, cohort=cohort, replace=replace, paths=paths,
                                            scheduling=scheduling), indent=2), encoding='utf-8')
    return cards, dates, paths


def simulate(fights=None, odds_type='close1', *, kelly_kwargs=None,
             init_bankroll=500, n_sims=1000, len_temp=1, random_state=None,
             days_out=5, density_models_dir=None,
             data_dir=None, model=None, scaler=None, density_models=None,
             prediction_model=None, prediction_model_type='open',
             prediction_scalers=None, prediction_encoders=None, prediction_se_model=None,
             closing_params=None, use_decay=True,
             return_details=False, return_event_exposure=False, title_col='title_fight',
             event_paths_mode='generate', event_paths_file=None,
             sample_with_replacement=False, progress_every=0, sampled_paths=None):
    """Schedule intact cards, select prices, predict, and size bets afresh.

    return_event_exposure=True with sampled_paths appends compact metadata
    containing per-path event stake fractions, event counts and batch size.
    If return_details is also true, detailed frames are inside that metadata.

    sampled_paths instead accepts a list of prediction DataFrames or its trusted
    pickle file. Uses every saved path (n_sims ignored), keeping attached odds,
    predictions and card assignments. No inference or resampling is performed.
    Omit fights, model/preprocessing inputs, density_models and event-path file
    options in this mode. Saved intermediate predictions must already contain
    sampled execution prices and execution_odds_type='intermediate'.

    Uses recorded open/close1/close2 prices, or odds_type='intermediate' to
    sample from a saved conditional KDE given each fighter's opening price and
    closing endpoint. Intermediate mode always draws a uniform endpoint
    between close1 and close2 and selects the closer market's KDE (ties close1).
    Both sides share the endpoint interpolation fraction; conditional price
    draws are independent. days_out selects a fitted day; a sequence chooses
    one day uniformly per event. Day zero also requires a fitted day-zero KDE.
    density_models requires both close1/close2: a stage -> fit mapping for one
    day, or a day -> stage -> fit mapping. Fits may be joblib paths. If omitted,
    both markets load from density_models_dir (default: Data/saved_models).
    The folder must contain one conditional_price_density_{stage}_{day}_days*.joblib
    file per requested day/market; no split selector is required.

    kelly_kwargs contains betting controls (and optional decay_rates and
    opening_params). All market-dependent predictions, fair odds, and stakes
    are recomputed with the saved opening model by default. prediction_model
    accepts a fitted model or path; prediction_model_type selects open/close1/
    close2 feature names and matching entries in prediction_scalers and
    prediction_encoders (market -> fitted transformer/path, or explicit None).
    prediction_se_model optionally supplies the corresponding uncertainty fit.
    Historical outcomes stay fixed.
    closing_params accepts close1/close2 -> BettingParams/dict/JSON path. When
    supplied, singles select the nearer recorded closing price for their chosen
    side; top-two-EV parlays compare products of those same legs' prices.
    Absolute decimal distance is used, with ties selecting close1. Saved decay
    is applied only when use_decay=True, which requires both closing policies
    to include four decay rates and their fixed opening_params. Raw stakes are
    blended first, then the selected closing z corrections and one joint
    moneyline/parlay exposure cap are applied. kelly_kwargs may then contain only
    calc_parlay. return_details exposes selected policies and ticket prices.
    calc_parlay defaults to True: simulate_kelly uses its top-EV two-leg
    parlay strategy. Its parlay_net is included alongside moneyline profit in
    every bankroll path, including when joint decay rates are supplied.
    The schedule is supplied entirely by sample_schedule, including its
    event-type constraints. len_temp>1 settles that many events against a shared
    starting bankroll; the last partial batch is included. No floor/top-ups.
    A zero or negative balance is retained and stops all subsequent betting.

    sample_with_replacement=True draws whole events uniformly with replacement,
    preserving the number of date slots but allowing repeated/omitted cards.
    Monthly numbered-card balancing applies only when sampling without replacement.
    That mode requires event_name or event_type; title flags do not classify cards.

    event_paths_mode: 'generate' (default, no disk writes), 'save' (generate
    and overwrite event_paths_file), or 'load' (reuse its first n_sims paths).
    Default file: Data/monte_carlo_paths/event_paths.json. Only event ordering
    is saved; prices, predictions and stakes are recalculated on every call.
    Loaded paths require the same fights, original dates and title metadata;
    input row order and betting/price settings may differ.
    progress_every=0 is silent; a positive integer prints the first path and
    every N completed paths, with stage timings and estimated remaining time.

    Returns the existing (moneyline profits, parlay profits, bankroll paths)
    tuple. return_details=True also returns per-run fresh Kelly/parlay frames,
    input prices, and scheduled event returns. Raw fight features are required;
    the former arrays of precomputed backtests cannot recalculate predictions.
    """
    import joblib
    import statsmodels.api as sm
    from ufc_betting.config import config, settings
    from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
    from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
    from .market_policy import load_predictor, predict_sampled_prices
    from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
    from ufc_betting.BettingStrategy.betting_params import BettingParams
    started = perf_counter()
    if isinstance(progress_every, bool) or not isinstance(progress_every, (int, np.integer)) or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer')
    if progress_every and sampled_paths is None:
        print(f'Simulation: initializing {n_sims} paths (event schedules: {event_paths_mode})', flush=True)
    if sampled_paths is None and (not isinstance(fights,pd.DataFrame) or fights.empty):
        raise ValueError('fights must be a nonempty raw fight-feature DataFrame')
    if odds_type not in ('open','close1','close2','intermediate'):
        raise ValueError('odds_type must be open, close1, close2, or intermediate')
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in (n_sims,len_temp)):
        raise ValueError('n_sims and len_temp must be positive integers')
    if not np.isfinite(init_bankroll) or init_bankroll<=0:raise ValueError('Invalid initial bankroll')
    controls=dict(kelly_kwargs or {})
    policies = None
    if closing_params is not None:
        if set(closing_params) != {'close1', 'close2'}:
            raise ValueError('closing_params must supply close1 and close2')
        if controls.keys() - {'calc_parlay'}:
            raise ValueError('With closing_params, kelly_kwargs supports only calc_parlay')
        policies = {stage: BettingParams.from_value(value) for stage, value in closing_params.items()}
        if use_decay:
            for stage, params in policies.items():
                if params.decay_rates is None or params.opening_params is None:
                    raise ValueError(f'{stage}: use_decay=True requires saved decay_rates and opening_params')
                if controls.get('calc_parlay', True) and not {'k_parlay_down', 'k_parlay_up'} <= params.decay_rates.keys():
                    raise ValueError(f'{stage}: parlay simulation requires all four saved decay rates')
    elif not use_decay:
        controls.pop('decay_rates', None)
        controls.pop('opening_params', None)
    if sampled_paths is not None:
        from .sampled_paths import load_sampled_paths, simulate_sampled_predictions
        if policies is None and 'betting_params' in controls:
            supplied = BettingParams.from_value(controls.pop('betting_params'))
            controls.update(supplied.to_simulation_kwargs())
            if not use_decay:
                controls.pop('decay_rates', None)
                controls.pop('opening_params', None)
        if fights is not None or event_paths_mode != 'generate' or event_paths_file is not None:
            raise ValueError('sampled_paths replaces fights and event_paths_mode/event_paths_file')
        if any(value is not None for value in (prediction_model, model, scaler, prediction_scalers,
                prediction_encoders, prediction_se_model, density_models, density_models_dir)):
            raise ValueError('sampled_paths uses saved predictions; omit models, preprocessing and density models')
        owned = {'prob_cols', 'fair_decimal_cols', 'real_decimal_cols', 'pred_winner_col',
                 'date_col', 'winner_col', 'prediction_se_col', 'init_bankroll', 'opening_data', 'risk_prob_cols'}
        if owned & controls.keys():
            raise ValueError(f'Saved-path engine supplies these arguments: {sorted(owned & controls.keys())}')
        if controls.get('bankroll_floor') is not None or controls.get('test_bankroll_scaler', False):
            raise ValueError('Monte Carlo settlement requires no bankroll floor or bankroll scaler')
        controls['bankroll_floor'] = None
        controls.setdefault('calc_parlay', True)
        saved_paths = load_sampled_paths(sampled_paths)
        if progress_every:
            print(f'Simulation: evaluating {len(saved_paths)} saved paths (reusing saved event schedules and prices)', flush=True)
        return simulate_sampled_predictions(saved_paths, odds_type,
            controls=controls, policies=policies, use_decay=use_decay,
            initial_bankroll=init_bankroll, batch_size=len_temp,
            return_details=return_details, progress_every=progress_every,
            return_event_exposure=return_event_exposure)
    if return_event_exposure:
        raise ValueError('return_event_exposure currently requires sampled_paths')
    predictor = None
    if prediction_model is not None:
        if model is not None or scaler is not None:
            raise ValueError('Use prediction_model and its preprocessing instead of legacy model/scaler')
        predictor = load_predictor(prediction_model, prediction_model_type,
                                   prediction_scalers, prediction_encoders, prediction_se_model)
    elif prediction_model_type != 'open' or any(value is not None for value in
            (prediction_scalers, prediction_encoders, prediction_se_model)):
        raise ValueError('Explicit prediction preprocessing requires prediction_model')
    owned={'prob_cols','fair_decimal_cols','real_decimal_cols','pred_winner_col','date_col',
           'winner_col','prediction_se_col','init_bankroll','opening_data','risk_prob_cols'}
    if owned & controls.keys():raise ValueError(f'Engine supplies these arguments: {sorted(owned & controls.keys())}')
    if controls.get('bankroll_floor') is not None or controls.get('test_bankroll_scaler',False):
        raise ValueError('Monte Carlo settlement requires no bankroll floor or bankroll scaler')
    controls['bankroll_floor']=None
    controls.setdefault('calc_parlay',True)
    source=fights.copy().reset_index(drop=True)
    source['date']=pd.to_datetime(source.date,errors='raise').dt.normalize()
    if source.duplicated(['date','fighter_red','fighter_blue']).any():raise ValueError('Duplicate fight rows')
    fitted_densities = None
    if odds_type == 'intermediate':
        from .price_sampling import load_density_models, ConditionalPriceSampler
        fitted_densities = load_density_models(
            density_models_dir if density_models_dir is not None else
            Path(data_dir or settings.data_dir) / 'saved_models', days_out, density_models)
    required_stages = ('open', 'close1', 'close2') if fitted_densities is not None else (odds_type,)
    if controls.get('decay_rates') is not None:required_stages=tuple(set(required_stages)|{'open'})
    if policies is not None:required_stages=('open', 'close1', 'close2')
    for stage in required_stages:
        prices=source[[f'dec_{stage}_red',f'dec_{stage}_blue']].to_numpy(float)
        if not np.isfinite(prices).all() or (prices<=1).any():raise ValueError(f'Invalid {stage} prices')
    if not source.winner.isin([0,1,2,3]).all():raise ValueError('Settled outcomes required (0/1 or 2/3 for void)')
    if policies is not None:
        for stage in ('close1', 'close2'):
            for side in ('red', 'blue'):
                source[f'recorded_{stage}_{side}'] = source[f'dec_{stage}_{side}']
    if title_col not in source:
        source[title_col] = 0  # Optional fight metadata, never an event classifier.
    if source[title_col].isna().any() or not source[title_col].isin([0,1,False,True]).all():
        raise ValueError(f'{title_col} must be 0/1')
    # Retain the schedule seed sequence used by existing recorded-price runs.
    schedule_seed, price_seed = np.random.SeedSequence(random_state).spawn(2)
    price_sampler = (ConditionalPriceSampler(fitted_densities, random_state=price_seed)
                     if fitted_densities is not None else None)
    path = Path(event_paths_file) if event_paths_file is not None else Path(data_dir or settings.data_dir)/'monte_carlo_paths'/'event_paths.json'
    cards, dates, event_paths = _event_paths(source, n_sims, title_col,
        np.random.default_rng(schedule_seed), event_paths_mode, path, sample_with_replacement)
    if predictor is None:
        model=sm.load(config.model_open_path) if model is None else model
        scaler=joblib.load(config.scaler_open_path) if scaler is None else scaler
    if progress_every:
        print(f'Ready: {len(source)} fights, {len(dates)} events/path; setup {perf_counter() - started:.1f}s', flush=True)
    run_started = perf_counter()
    all_ml,all_parlay,all_total,details=[],[],[],[]
    for simulation in range(n_sims):
        path_started = perf_counter()
        scheduled_cards = []
        for event_id, (date, source_date) in enumerate(zip(dates, event_paths[simulation])):
            card = cards[source_date].copy()
            card['source_date'] = card.date
            card['source_row_position'] = card.index
            card['sampled_event_id'] = event_id
            card['date'] = pd.Timestamp(date)
            scheduled_cards.append(card)
        scheduled = pd.concat(scheduled_cards, ignore_index=True)
        sample_started = perf_counter()
        prices = (price_sampler.sample(scheduled) if price_sampler is not None else
                  scheduled[[f'dec_{odds_type}_red',f'dec_{odds_type}_blue']].to_numpy(float))
        prediction_started = perf_counter()
        predicted = (predict_sampled_prices(scheduled, prices, predictor, prediction_model_type)
                     if predictor is not None else opening_model_at_close1(scheduled,prices,model,scaler))
        # Do not let stale risk/SE columns override the new predictions.
        predicted=predicted.drop(columns=[c for c in ('risk_proba_red','risk_proba_blue','risk_red_proba','prediction_se','se') if c in predicted])
        predicted['execution_odds_type']=odds_type
        betting_started = perf_counter()
        ml_return,parlay_return=[],[]
        kelly_frames,parlay_frames=[],[]
        # One-event calls give unit-bankroll returns for delayed settlement too.
        for date,group in predicted.groupby('date',sort=True):
            kwargs=dict(controls)
            if kwargs.get('decay_rates') is not None:kwargs['opening_data']=group
            if policies is not None:
                kelly, parlay = BettingReturnCalculator.calculate_market_event_returns(
                    group, policies, use_decay=use_decay,
                                                    calc_parlay=controls['calc_parlay'])
            else:
                kelly,parlay=simulate_kelly(group,prob_cols=['proba_blue','proba_red'],
                    fair_decimal_cols=['dec_fair_close1_blue','dec_fair_close1_red'],
                    real_decimal_cols=['dec_close1_blue','dec_close1_red'],pred_winner_col='pred_winner',
                    date_col='date',prediction_se_col='proba_se',init_bankroll=1.,**kwargs)
            first=kelly.iloc[0]
            ml=float(first.event_payout_money_line);pl=float(first.get('parlay_net',0.))
            ml_return.append(ml);parlay_return.append(pl)
            if return_details:kelly_frames.append(kelly);parlay_frames.append(parlay)
        families = {'moneyline': np.asarray([ml_return]),
                    'parlay': np.asarray([parlay_return])}
        settled = settle_bankroll_paths(
            families['moneyline'] + families['parlay'], init_bankroll,
            family_returns=families, batch_size=len_temp,
        )
        all_ml.append(settled['moneyline_profits'][0])
        all_parlay.append(settled['parlay_profits'][0])
        all_total.append(settled['bankroll'][0])
        if return_details:
            details.append(dict(inputs=predicted,kelly=pd.concat(kelly_frames,ignore_index=True),
                parlay=pd.concat(parlay_frames,ignore_index=True),
                event_returns=pd.DataFrame(dict(date=sorted(predicted.date.unique()),moneyline_return=ml_return,parlay_return=parlay_return))))
        completed = simulation + 1
        if progress_every and (completed == 1 or completed % progress_every == 0 or completed == n_sims):
            now = perf_counter()
            remaining = (now - run_started) / completed * (n_sims - completed)
            print(f'Paths {completed}/{n_sims} | elapsed {(now - started) / 60:.1f} min | '
                  f'ETA {remaining / 60:.1f} min | last path {now - path_started:.2f}s '
                  f'(prices {prediction_started - sample_started:.2f}s, '
                  f'inference {betting_started - prediction_started:.2f}s, '
                  f'betting/settlement {now - betting_started:.2f}s)', flush=True)
    result=(all_ml,all_parlay,all_total)
    return (*result,details) if return_details else result

def blend_dataframes(
    dfs,
    probs,
    kind="kelly",
    date_col="date",
    random_state=None
):
    
    # random number generator 
    rng = np.random.default_rng(random_state)

    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()

    # stack each odds type column wise 
    if kind == "kelly":
        valid = np.column_stack([
            ((df["choice_ev"] > 0) | (df["choice_fstar"] > 0)).to_numpy()
            for df in dfs
        ])

    elif kind == "parlay":
        # parlay result stored in each row
        dfs = [
            df.groupby(date_col, as_index=False).first()
            for df in dfs
        ]

        valid = np.column_stack([
            (df["parlay_ev"] > 0).to_numpy()
            for df in dfs
        ])

    else:
        raise ValueError("kind must be 'kelly' or 'parlay'")

    # get boolean where there is a valid bet 
    keep_rows = valid.any(axis=1)

    # get proba weights for each row 
    weights = valid[keep_rows] * probs
    weights = weights / weights.sum(axis=1, keepdims=True)

    # cumsum over the rows 
    cum_weights = np.cumsum(weights, axis=1)

    # get array of random values between 0 and 1, len weights.shape[0]
    random_vals = rng.random(weights.shape[0])

    # random_vals[:, None] adds column dimension, (weights.shape[0], 1) 
    # now can do vectorized boolean comparison over the column dimension 
    # because of the cumsum, this sum gives the index matched to the proba sampled and given proba for each df 
    chosen_df_idx = (random_vals[:, None] > cum_weights).sum(axis=1)

    # np.where returns integer indices for both rows and columns 
    kept_positions = np.where(keep_rows)[0]
    selected_rows = [
        dfs[df_idx].iloc[row_idx]
        for row_idx, df_idx in zip(kept_positions, chosen_df_idx)
    ]

    return pd.DataFrame(selected_rows).reset_index(drop=True)


