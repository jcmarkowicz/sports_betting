"""Resample fights within event families and original bout positions."""

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.MonteCarlo.event_types import event_types
from ufc_betting.BettingStrategy.MonteCarlo.schedule_sampling import sample_schedule


def _retry_file_operation(operation):
    """Retry transient Windows sharing violations, not other I/O failures."""
    import time

    for attempt in range(5):
        try:
            return operation()
        except OSError as error:
            if getattr(error, 'winerror', None) not in (32, 33) or attempt == 4:
                raise
            time.sleep(.1 * (2 ** attempt))


def _save_paths(paths, output_path):
    """Close the writer before replacement and never hide the original failure."""
    import os
    import tempfile
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.pkl', delete=False) as file:
            temporary = Path(file.name)
            pd.to_pickle(paths, file)
    except BaseException as error:
        # The context manager closes our handle even if serialization fails.
        if temporary is not None:
            try:
                _retry_file_operation(lambda: temporary.unlink(missing_ok=True))
            except OSError as cleanup_error:
                error.add_note(f'Could not remove incomplete temporary file {temporary}: {cleanup_error}')
        raise
    try:
        _retry_file_operation(lambda: os.replace(temporary, destination))
    except OSError as error:
        # Serialization succeeded: retain the complete file for recovery.
        error.add_note(f'Generated paths were saved to {temporary}. Could not replace {destination}; '
                       'load or rename the temporary file once the lock is released. No regeneration is needed.')
        raise


def sample_ordered_fights(fights, *, date_col='date', position_col='bout_position',
                         event_name_col='event_name', event_type_col='event_type',
                         random_state=None):
    """Shuffle event templates, then bootstrap each slot by type and position.

    Every input event template appears once, using sample_schedule's balanced
    numbered-card placement. Each slot draws uniformly with replacement from
    all input fights of the same event type and original bout position (1 main
    event, 2 co-main, etc.). Source fights may recur, including on their original
    card. Fight outcomes, features, odds and title flags travel with the donor.
    Attached predictions are preserved. Stakes, parlays and event returns must
    be recalculated downstream for the synthetic cards.

    Positions must be recorded on the COMPLETE original cards before filtering;
    gaps are allowed and never renumbered. Dates must identify unique events.
    Sampling is independent across slots, so repeated fighters from different
    historical bouts are possible; no additional matchup constraint is imposed.

    Output date/name/type identify the synthetic destination card. source_date,
    source_event_name and source_row_position identify the donor; template_date,
    template_row_position identify the scheduled original card/slot. All row
    positions reference the input's positional order, not its DataFrame index.
    The caller's frame is not modified; output is chronological, then by slot.
    """
    if not isinstance(fights, pd.DataFrame) or fights.empty:
        raise ValueError('fights must be a nonempty DataFrame')
    missing = {date_col, position_col} - set(fights)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    metadata = {'source_date', 'source_event_name', 'source_row_position',
                'template_date', 'template_row_position', 'sampled_event_id'}
    if metadata & set(fights):
        raise ValueError('Supply original fight rows, not an already sampled schedule')
    data = fights.copy().reset_index(drop=True)
    data[date_col] = pd.to_datetime(data[date_col], errors='raise')
    if data[date_col].isna().any():
        raise ValueError('Event dates must be nonmissing')
    positions = pd.to_numeric(data[position_col], errors='raise')
    if (positions.isna().any() or not np.isfinite(positions).all()
            or (positions < 1).any() or (positions % 1 != 0).any()
            or pd.api.types.is_bool_dtype(positions)):
        raise ValueError('bout positions must be positive integers recorded on original cards')
    data[position_col] = positions.astype(int)
    if data.duplicated([date_col, position_col]).any():
        raise ValueError('Each original event must have unique bout positions')
    data[event_type_col] = event_types(data, event_name_col=event_name_col,
                                     event_type_col=event_type_col)
    rng = np.random.default_rng(random_state)
    schedule = sample_schedule(data, date_col=date_col, event_name_col=event_name_col,
                               event_type_col=event_type_col, random_state=rng, replace=False)
    schedule = schedule.sort_values([date_col, position_col], kind='stable')
    pools = data.groupby([event_type_col, position_col], sort=False).indices
    donors = np.array([rng.choice(pools[(kind, position)]) for kind, position in
                       zip(schedule[event_type_col], schedule[position_col])])
    sampled = data.iloc[donors].copy().reset_index(drop=True)
    sampled['source_date'] = sampled[date_col]
    sampled['source_row_position'] = donors
    if event_name_col in data:
        sampled['source_event_name'] = sampled[event_name_col]
        sampled[event_name_col] = schedule[event_name_col].to_numpy()
    sampled['template_date'] = schedule.source_date.to_numpy()
    sampled['template_row_position'] = schedule.source_row_position.to_numpy()
    sampled['sampled_event_id'] = schedule.sampled_event_id.to_numpy()
    sampled[date_col] = schedule[date_col].to_numpy()
    return sampled


def generate_ordered_paths(
    fights, *, n_paths=100, output_path=None, odds_type='close1',
    sample_prices=False, days_out=(7, 6, 5), density_models_dir=None,
    density_models=None, prediction_model=None, prediction_model_type='open',
    prediction_scalers=None, prediction_encoders=None, prediction_se_model=None,
    random_state=42, progress_every=10,
    sample_fights=True, storage="pickle", strict_numbered_months=False,
):
    """Generate reusable fight/prediction paths for search and Monte Carlo.

    Without sample_prices, retain supplied predictions and market prices exactly.
    With sample_prices and no prediction_model, save dec_intermediate_red/blue
    without inference or market selection. Choose a model later using
    predict_intermediate_paths. With a prediction_model, retain the legacy
    combined generation/inference behavior. odds_type selects the search key;
    in that combined mode sampled prices are
    stored in that market's columns and close1 execution columns used by simulate.
    Original closing endpoints remain in recorded_close{1,2}_{red,blue}.

    storage='compact' streams to a NEW directory and returns a lazy CompactPaths
    sequence. Integer donor arrays and per-occurrence overrides are memory mapped;
    the source is saved once. Use load_sampled_paths(..., n_paths=...) to load only
    selected paths. storage='pickle' preserves the legacy list/file interface.
    sample_fights=False shuffles intact cards without replacement, keeping within-card
    order. strict_numbered_months=True rejects sources unable to provide exactly one
    numbered card per occupied month; otherwise the scheduler balances all cards.

    With legacy storage, return a list of DataFrames, optionally saved as one pandas pickle accepted
    by both consumers' sampled_paths argument. No stakes/returns are frozen:
    consumers recalculate those for each candidate policy. Only load trusted
    pickle files. Source rows must already include original card metadata.
    odds_type=None creates market-independent paths from validation rows, without
    requiring prices or predictions. Attach predictions later with
    attach_path_predictions before evaluating these paths.
    """
    from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
    from .sampled_paths import load_sampled_paths

    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError('n_paths must be a positive integer')
    if odds_type not in (None, 'open', 'close1', 'close2'):
        raise ValueError('odds_type must be open, close1, or close2')
    if sample_prices and prediction_model is not None and odds_type in (None, 'open'):
        raise ValueError('Density price sampling requires a closing search market')
    if not isinstance(fights, pd.DataFrame) or fights.empty:
        raise ValueError('fights must be a nonempty DataFrame')
    if 'winner' not in fights or not fights.winner.isin([0, 1, 2, 3]).all():
        raise ValueError('Supply settled outcomes (0/1 or 2/3 for void) for search-compatible paths')
    if storage not in ('pickle', 'compact'):
        raise ValueError('storage must be pickle or compact')
    if storage == 'compact' and output_path is None:
        raise ValueError('Compact storage requires an output directory')
    source = fights.copy()
    if strict_numbered_months:
        types = event_types(source)
        dated = pd.to_datetime(source.date)
        numbered = pd.DataFrame({'date': dated, 'type': types}).drop_duplicates()
        if numbered.groupby('date')['type'].nunique().gt(1).any():
            raise ValueError('Each card must have one event type')
        if numbered['type'].eq('numbered').sum() != dated.dt.to_period('M').nunique():
            raise ValueError('Exactly one numbered card per month requires the number of numbered cards to equal occupied months')
    stages = ('open', 'close1', 'close2') if sample_prices else (() if odds_type is None else (odds_type,))
    for stage in stages:
        cols = [f'dec_{stage}_{side}' for side in ('red', 'blue')]
        values = source[cols].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 1).any():
            raise ValueError(f'Invalid {stage} decimal prices')
    schedule_seed, price_seed = np.random.SeedSequence(random_state).spawn(2)
    schedule_rng = np.random.default_rng(schedule_seed)
    sampler = predictor = None
    if sample_prices:
        from ufc_betting.config import settings
        from .price_sampling import ConditionalPriceSampler, load_density_models
        if prediction_model is not None:
            from .market_policy import load_predictor, predict_sampled_prices
            from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
            from ufc_betting.BettingStrategy.kelly_scaling import decimal_to_american
            predictor = load_predictor(prediction_model, prediction_model_type,
                                       prediction_scalers, prediction_encoders, prediction_se_model)
        models = load_density_models(density_models_dir or settings.saved_models_dir,
                                     days_out, density_models)
        sampler = ConditionalPriceSampler(models, random_state=price_seed)
        for market in ('close1', 'close2'):
            for side in ('red', 'blue'):
                source[f'recorded_{market}_{side}'] = source[f'dec_{market}_{side}']
    elif odds_type is not None:
        frozen = read_frozen_predictions(source)
        for column in ('proba_red', 'risk_red_proba'):
            values = frozen[column].to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f'Invalid supplied probabilities: {column}')
        errors = frozen.prediction_se.to_numpy(float)
        if not np.isfinite(errors).all() or (errors < 0).any():
            raise ValueError('Prediction standard errors must be finite and nonnegative')
        fair = source[[f'dec_fair_{odds_type}_{side}' for side in ('red', 'blue')]].to_numpy(float)
        if not np.isfinite(fair).all() or (fair <= 1).any():
            raise ValueError('Fair decimal prices must be finite and greater than 1')
    writer = None
    if storage == 'compact':
        from .compact_paths import CompactPathWriter
        writer = CompactPathWriter(output_path, source, n_paths)
    try:
        paths = []
        for number in range(n_paths):
            frame = (sample_ordered_fights(source, random_state=schedule_rng) if sample_fights
                     else sample_schedule(source, random_state=schedule_rng, replace=False))
            if sampler is not None:
                prices = sampler.sample(frame)
                for index, side in enumerate(('red', 'blue')):
                    frame[f'dec_intermediate_{side}'] = prices[:, index]
                if predictor is not None:
                    frame = predict_sampled_prices(frame, prices, predictor, prediction_model_type)
                    if odds_type != 'close1':
                        frame = replace_opening_odds(frame, odds_type, prices=prices, features=[])
                # Saved risk/SE aliases must never override newly inferred values.
                frame = frame.drop(columns=[c for c in (
                    'risk_proba_red', 'risk_proba_blue', 'risk_red_proba', 'risk_choice_proba',
                    'prediction_se', 'se', 'choice_proba', 'odds_type',
                ) if c in frame])
                if predictor is None:
                    frame = frame.drop(columns=[c for c in (
                        'proba_red', 'proba_blue', 'proba_se', 'pred_winner', 'correct_pred',
                        'prob_winner', 'prediction_model_type', 'choice_fighter_bool',
                    ) if c in frame])
                else:
                    for stage in {'close1', odds_type}:
                        for index, side in enumerate(('red', 'blue')):
                            frame[f'{stage}_{side}'] = [decimal_to_american(x) for x in prices[:, index]]
                frame['execution_odds_type'] = 'intermediate'
            elif odds_type is not None:
                frame['execution_odds_type'] = odds_type
            frame.attrs['generation'] = dict(path_id=number,
                                            odds_type=None if sampler is not None and predictor is None else odds_type,
                                            sample_prices=bool(sample_prices), sample_fights=bool(sample_fights), random_state=random_state)
            if writer is not None:
                writer.append(frame)
            else:
                paths.append(frame)
            if progress_every and ((number + 1) % progress_every == 0 or number + 1 == n_paths):
                print(f'Generated {number + 1}/{n_paths} ordered fight paths', flush=True)
        if writer is not None:
            return writer.finish()
    finally:
        if writer is not None:
            writer.close()
    paths = load_sampled_paths(paths)
    if output_path is not None:
        _save_paths(paths, output_path)
    return paths
