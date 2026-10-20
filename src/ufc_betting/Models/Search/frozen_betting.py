import numpy as np 
import pandas as pd 
from pathlib import Path
from ufc_betting.BettingStrategy.betting_params import BettingParams

from ufc_betting.config import config
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
from ufc_betting.BettingStrategy.MonteCarlo.schedule_sampling import sample_schedule
from ufc_betting.BettingStrategy.MonteCarlo.event_types import event_types
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate_bankroll_paths
from ufc_betting.BettingStrategy.MonteCarlo.bootstrap_paths import (
    dataset_metadata, load_bootstrap_paths, save_bootstrap_paths, new_bootstrap_directory,
)
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions

ODDS_TYPES = config.odds_types

import optuna


def log_growth(returns):
    returns = np.asarray(returns, dtype=float)
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError('Invalid event bankroll returns')
    if (returns == -1).any():
        return -np.inf
    return float(np.log1p(returns).sum())


def validate_score_settings(score_type, wealth_floor=.001, risk_penalty=.25, negative_mean_penalty=1000., drawdown_penalty=0., drawdown_tail_fraction=.1, ulcer_penalty=0.):
    if score_type not in ("mean_std", "log_growth", "path_mean_sharpe", "event_mean_std", "log_mean_roi", "event_log_growth"):
        raise ValueError('score_type must be "mean_std", "log_growth", "path_mean_sharpe", "event_mean_std", "log_mean_roi", or "event_log_growth"')
    if isinstance(wealth_floor, (bool, np.bool_)) or not np.isfinite(wealth_floor) or not 0 < wealth_floor < 1:
        raise ValueError('wealth_floor must be finite and strictly between 0 and 1')
    if not np.isfinite(risk_penalty) or risk_penalty < 0:
        raise ValueError('risk_penalty must be finite and nonnegative')
    if (isinstance(negative_mean_penalty, (bool, np.bool_))
            or not np.isfinite(negative_mean_penalty) or negative_mean_penalty <= 0):
        raise ValueError('negative_mean_penalty must be positive and finite')
    if (isinstance(drawdown_penalty, (bool, np.bool_))
            or not np.isfinite(drawdown_penalty) or drawdown_penalty < 0):
        raise ValueError('drawdown_penalty must be finite and nonnegative')
    if (isinstance(drawdown_tail_fraction, (bool, np.bool_))
            or not np.isfinite(drawdown_tail_fraction) or not 0 < drawdown_tail_fraction <= 1):
        raise ValueError('drawdown_tail_fraction must be finite and in (0, 1]')
    if (isinstance(ulcer_penalty, (bool, np.bool_))
            or not np.isfinite(ulcer_penalty) or ulcer_penalty < 0):
        raise ValueError('ulcer_penalty must be finite and nonnegative')
    if ulcer_penalty and score_type != 'event_log_growth':
        raise ValueError('ulcer_penalty applies only to event_log_growth')
    if drawdown_penalty and score_type != 'event_log_growth':
        raise ValueError('drawdown_penalty applies only to event_log_growth')


def path_mean_event_returns(bankroll):
    """Average total event returns within each path, equally weighting all events.

    Each event return is its bankroll change / pre-event bankroll. No-bet events
    and scheduled events after ruin contribute zero; the ruin event retains its
    actual loss. Input includes the initial balance followed by event balances.
    """
    bankroll = np.asarray(bankroll, dtype=float)
    if (bankroll.ndim != 2 or not bankroll.shape[0] or bankroll.shape[1] < 2
            or not np.isfinite(bankroll).all() or (bankroll[:, 0] <= 0).any()):
        raise ValueError('Need finite (paths, events + 1) bankrolls with positive initial balances')
    previous = bankroll[:, :-1]
    returns = np.divide(np.diff(bankroll, axis=1), previous,
                        out=np.zeros_like(previous), where=previous > 0)
    return returns.mean(axis=1)


def path_max_drawdowns(bankroll):
    """Maximum peak-to-trough fractional loss per path, including initial wealth.

    Uses actual balances, not the log scoring floor. Negative balances can
    produce drawdowns above 100%; these are deliberately not clipped.
    """
    bankroll = np.asarray(bankroll, dtype=float)
    if (bankroll.ndim != 2 or not bankroll.shape[0] or bankroll.shape[1] < 2
            or not np.isfinite(bankroll).all() or (bankroll[:, 0] <= 0).any()):
        raise ValueError('Need finite bankroll paths with positive initial balances')
    peaks = np.maximum.accumulate(bankroll, axis=1)
    return ((peaks - bankroll) / peaks).max(axis=1)


def path_ulcer_indices(bankroll):
    """RMS fractional drawdown per path over all post-event balances.

    Initial wealth establishes the peak but is excluded from the average.
    Includes no-bet, unrecovered and post-ruin events; no floor or clipping.
    """
    bankroll = np.asarray(bankroll, dtype=float)
    if (bankroll.ndim != 2 or not bankroll.shape[0] or bankroll.shape[1] < 2
            or not np.isfinite(bankroll).all() or (bankroll[:, 0] <= 0).any()):
        raise ValueError('Need finite bankroll paths with positive initial balances')
    peaks = np.maximum.accumulate(bankroll, axis=1)
    drawdowns = ((peaks - bankroll) / peaks)[:, 1:]
    return np.sqrt(np.mean(np.square(drawdowns), axis=1))


def worst_tail_drawdown(drawdowns, fraction=.1):
    """Mean of the largest ceil(fraction * path_count) maximum drawdowns.

    At least one path is included. fraction=1 recovers the overall mean.
    """
    validate_score_settings('event_log_growth', drawdown_tail_fraction=fraction)
    drawdowns = np.asarray(drawdowns, dtype=float)
    if drawdowns.ndim != 1 or not len(drawdowns) or not np.isfinite(drawdowns).all() or (drawdowns < 0).any():
        raise ValueError('Need finite nonnegative path drawdowns')
    count = max(1, int(np.ceil(fraction * len(drawdowns))))
    return float(np.partition(drawdowns, len(drawdowns) - count)[-count:].mean())


def path_mean_log_growth(bankroll, wealth_floor=.001):
    """Mean log bankroll growth per scheduled event for each path.

    Log wealth is measured relative to initial bankroll and floored only for
    scoring. Differences of these log wealth levels telescope to the final log
    wealth ratio. No-bet/post-ruin events count with zero growth; actual bankrolls
    are unchanged. The floor also applies to positive wealth below the floor.
    """
    validate_score_settings('event_log_growth', wealth_floor)
    bankroll = np.asarray(bankroll, dtype=float)
    if (bankroll.ndim != 2 or not bankroll.shape[0] or bankroll.shape[1] < 2
            or not np.isfinite(bankroll).all() or (bankroll[:, 0] <= 0).any()):
        raise ValueError('Need finite bankroll paths with positive initial balances and at least one event')
    log_wealth = np.log(np.maximum(bankroll / bankroll[:, :1], wealth_floor))
    return np.diff(log_wealth, axis=1).mean(axis=1)


def event_roi_statistics(returns, stakes, bankroll):
    """Equal-weight mean ROI over actually wagered events in each path.

    Returns and stakes are fractions of pre-event bankroll. A parlay ticket is
    one stake. No-stake/post-ruin events have NaN ROI and are excluded. Paths
    with no wagered events have NaN mean ROI so the objective rejects them.
    """
    returns, stakes, bankroll = map(lambda x: np.asarray(x, float), (returns, stakes, bankroll))
    if (returns.ndim != 2 or stakes.shape != returns.shape
            or bankroll.shape != (returns.shape[0], returns.shape[1] + 1)
            or not np.isfinite(returns).all() or not np.isfinite(stakes).all()
            or not np.isfinite(bankroll).all() or (stakes < 0).any()):
        raise ValueError('Need matching finite event returns, nonnegative stakes, and bankroll paths')
    active = (stakes > 0) & (bankroll[:, :-1] > 0)
    roi = np.divide(returns, stakes, out=np.full_like(returns, np.nan), where=active)
    count = active.sum(axis=1)
    means = np.divide(np.nansum(roi, axis=1), count,
                      out=np.full(len(count), np.nan), where=count > 0)
    return dict(event_rois=roi, path_mean_event_rois=means,
                betting_event_counts=count, betting_event_coverage=active.mean(axis=1))


def log_mean_roi_score(means, negative_mean_penalty=1000.):
    """Log of the mean of path-average event ROIs, NOT mean log ROI.

    Negative means score -negative_mean_penalty; zero means score -infinity.
    Positive means use the natural log directly, without a floor or SD penalty.
    """
    validate_score_settings('log_mean_roi', negative_mean_penalty=negative_mean_penalty)
    means = np.asarray(means, dtype=float)
    if means.ndim != 1 or not len(means) or not np.isfinite(means).all():
        raise ValueError('Need finite path-average event ROIs')
    mean = float(means.mean())
    if mean < 0:
        return -float(negative_mean_penalty)
    if mean == 0:
        return -np.inf
    return float(np.log(mean))


def path_mean_sharpe_score(means):
    """Mean of path-average event ROIs / sample SD across paths.

    No cap, annualization, risk penalty, or sqrt(path count) scaling. All-zero
    paths score zero. A nonzero mean with zero SD is undefined and rejected.
    """
    means = np.asarray(means, dtype=float)
    if means.ndim != 1 or len(means) < 2 or not np.isfinite(means).all():
        raise ValueError('Need at least two finite path-average event ROIs')
    mean, std = float(means.mean()), float(means.std(ddof=1))
    if std == 0:
        if mean == 0:
            return 0.
        raise ValueError('Path-mean Sharpe is undefined: nonzero mean with zero path SD')
    score = mean / std
    if not np.isfinite(score):
        raise ValueError('Path-mean Sharpe is nonfinite')
    return score


def terminal_return_score(returns, risk_penalty=.25, *, score_type="mean_std", wealth_floor=.001):
    """Score terminal fractional returns; the floor affects log scoring only.

    log_growth = mean(log(max(1 + return, wealth_floor))). No SD penalty.
    mean_std retains the original mean return minus sample-SD penalty.
    """
    validate_score_settings(score_type, wealth_floor, risk_penalty)
    if score_type in ('path_mean_sharpe', 'log_mean_roi'):
        raise ValueError('path_mean_sharpe requires event returns and stakes; use event_roi_statistics')
    if score_type == 'event_log_growth':
        raise ValueError('event_log_growth requires event bankroll paths; use path_mean_log_growth')
    if score_type == 'event_mean_std':
        raise ValueError('event_mean_std requires event bankroll paths; use path_mean_event_returns')
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError('Need at least two finite terminal returns')
    if score_type == "log_growth":
        return float(np.log(np.maximum(1.0 + returns, wealth_floor)).mean())
    return float(returns.mean() - risk_penalty * returns.std(ddof=1))


class FrozenBettingSearch:
    """Tune fixed predictions using schedules shared across betting trials.

    Sampling with replacement preserves bootstrap terminal-return scoring.
    Without replacement, event-type metadata is required and numbered-card
    placement follows sample_schedule. Paths stop betting when their bankroll
    reaches zero or below, retaining the balance after the final loss.
    """

    def __init__(self, predictions, odds_type='open', *, k=1000,
                 initial_bankroll=500.0, seed=42, risk_penalty=.25,
                 sample_with_replacement=True, bootstrap_mode='generate', bootstrap_dir=None,
                 sampled_paths=None, _fixed_path_indices=None, score_type="mean_std", wealth_floor=.001, negative_mean_penalty=1000., drawdown_penalty=0., drawdown_tail_fraction=.1, ulcer_penalty=0.):
        validate_score_settings(score_type, wealth_floor, risk_penalty, negative_mean_penalty, drawdown_penalty, drawdown_tail_fraction, ulcer_penalty)
        self.drawdown_tail_fraction = float(drawdown_tail_fraction)
        self.drawdown_penalty = float(drawdown_penalty)
        self.ulcer_penalty = float(ulcer_penalty)
        self.negative_mean_penalty = float(negative_mean_penalty)
        self.score_type, self.wealth_floor = score_type, float(wealth_floor)
        self.sampled_paths = None
        self._path_searches = None
        if sampled_paths is not None:
            from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import load_sampled_paths
            if bootstrap_mode != 'generate' or bootstrap_dir is not None:
                raise ValueError('sampled_paths replaces bootstrap_mode/bootstrap_dir')
            self.sampled_paths = load_sampled_paths(sampled_paths)
            k = len(self.sampled_paths)
            if len({path.date.nunique() for path in self.sampled_paths}) != 1:
                raise ValueError('Search sampled paths must have the same number of cards')
        if bootstrap_mode not in ('generate', 'save', 'load'):
            raise ValueError('bootstrap_mode must be generate, save, or load')
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:
            raise ValueError('k must be an integer >= 2')
        if not isinstance(sample_with_replacement, (bool, np.bool_)):
            raise ValueError('sample_with_replacement must be a boolean')
        if not np.isfinite(risk_penalty) or risk_penalty < 0:
            raise ValueError('risk_penalty must be finite and nonnegative')
        self.risk_penalty = float(risk_penalty)
        if odds_type not in ODDS_TYPES:
            raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
        self.frame = read_frozen_predictions(predictions)
        self.frame['date'] = pd.to_datetime(self.frame.date, errors='raise').dt.normalize()
        self.frame = self.frame.sort_values('date', kind='stable').reset_index(drop=True)
        if self.frame.empty or self.frame.date.isna().any():
            raise ValueError('Need nonempty, dated betting predictions')
        for col in ('proba_red', 'risk_red_proba'):
            values = self.frame[col].to_numpy(float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f'Invalid {col}')
        se = self.frame.prediction_se.to_numpy(float)
        if not np.isfinite(se).all() or (se < 0).any():
            raise ValueError('Invalid inference probability SE')
        if not self.frame.winner.isin([0, 1]).all():
            raise ValueError('Betting evaluation requires binary settled outcomes')
        if 'odds_type' in self.frame and not self.frame.odds_type.eq(odds_type).all():
            raise ValueError('Prediction odds_type differs from execution odds_type')
        
        self.bets_engine = BettingReturnCalculator(
            real_odds=self.frame[[f'dec_{odds_type}_blue', f'dec_{odds_type}_red']],
            fair_odds=self.frame[[f'dec_fair_{odds_type}_blue', f'dec_fair_{odds_type}_red']],
            dates=self.frame.date,
        )

        for odds in (self.bets_engine.real_odds, self.bets_engine.fair_odds):
            if not np.isfinite(odds.to_numpy()).all() or (odds <= 1).any().any():
                raise ValueError('Decimal odds must be finite and exceed 1')
            
        if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
            raise ValueError('initial_bankroll must be positive and finite')
        
        self.odds_type, self.initial_bankroll = odds_type, float(initial_bankroll)
        self.sample_with_replacement = sample_with_replacement
        self.bootstrap_dir = Path(bootstrap_dir) if bootstrap_dir is not None else None
        self.bootstrap_metadata = None
        if self.sampled_paths is not None:
            import hashlib
            digest = hashlib.sha256()
            for path in self.sampled_paths:
                payload = path.to_json(orient='split', date_format='iso', index=False).encode()
                digest.update(len(payload).to_bytes(8, 'big'))
                digest.update(payload)
            self.path_indices = None
            self.bootstrap_metadata = dict(format='sampled_fights', n_samples=k,
                                           dataset_sha256=digest.hexdigest())
        elif _fixed_path_indices is not None:
            self.path_indices = _fixed_path_indices
        elif bootstrap_mode == 'load':
            if self.bootstrap_dir is None:
                raise ValueError('bootstrap_dir is required to load saved paths')
            expected = dataset_metadata(self.frame, odds_type, sample_with_replacement)
            self.path_indices, self.bootstrap_metadata = load_bootstrap_paths(
                self.bootstrap_dir, expected, k)
        else:
            metadata = (dataset_metadata(self.frame, odds_type, sample_with_replacement)
                        if bootstrap_mode == 'save' else None)
            self.path_indices = self._sample_event_indices(k, seed)
            if bootstrap_mode == 'save':
                if self.bootstrap_dir is None:
                    self.bootstrap_dir = new_bootstrap_directory() / odds_type
                self.bootstrap_metadata = save_bootstrap_paths(
                    self.bootstrap_dir, self.path_indices, metadata, seed)

    def _sample_event_indices(self, k, seed):
        """Map sampled source dates to the calculator's chronological event rows."""
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 2:
            raise ValueError('k must be an integer >= 2')
        if self.sample_with_replacement:
            schedule = self.frame[['date']].drop_duplicates()
        else:
            schedule = self.frame[['date']].assign(event_type=event_types(self.frame)).drop_duplicates()
            if schedule.date.duplicated().any():
                raise ValueError('Each date must identify one event type')
        schedule = schedule.sort_values('date', kind='stable')
        dates = pd.DatetimeIndex(schedule['date'])
        rng = np.random.default_rng(seed)
        paths = []
        for _ in range(k):
            sampled = sample_schedule(
                schedule, replace=self.sample_with_replacement,
                random_state=int(rng.integers(0, 2**32 - 1)),
            )
            paths.append(dates.get_indexer(sampled['source_date']))
        return np.asarray(paths, dtype=int)

    @staticmethod
    def suggest(trial):
        return dict(
            mdd=trial.suggest_float('mdd', .02, .99, step=.01),
            N=trial.suggest_int('N', 10, 2000, log=True),
            mdd_parlay=trial.suggest_float('mdd_parlay', .01, .99, step=.01),
            N_parlay=trial.suggest_int('N_parlay', 10, 2000, log=True),
            z=trial.suggest_float('z', 0.01, 3.00, step=.01),
            moneyline_min_american=trial.suggest_int('moneyline_min_american', -600, -200, step=25),
        )

    def event_returns(self, params):
        events = self.bets_engine.calculate_fight_returns(
            red_proba=self.frame.proba_red.to_numpy(),
            risk_red_proba=self.frame.risk_red_proba.to_numpy(),
            se=self.frame.prediction_se.to_numpy(),
            winner_fold=self.frame.winner.to_numpy(), val_idx=np.arange(len(self.frame)),
            param=BettingParams.from_value(params).to_policy(),
        )
        if not np.isfinite(events.event_return).all():
            raise ValueError('Event bankroll returns must be finite')
        return events

    def simulate(self, params):
        if self.sampled_paths is not None or self.score_type in ('path_mean_sharpe', 'log_mean_roi'):
            return self._simulate_shared_fights(params)
        return simulate_bankroll_paths(self.event_returns(params), self.path_indices,
                                       self.initial_bankroll)

    def _prepare_shared_fights(self):
        """Deduplicate actual numerical inputs, including occurrence-specific prices."""
        columns = ['proba_red', 'risk_red_proba', 'prediction_se', 'winner']
        columns += [f'{prefix}_{self.odds_type}_{side}'
                    for prefix in ('dec', 'dec_fair') for side in ('blue', 'red')]
        if isinstance(self, JointDecayBettingSearch):
            columns += ['dec_open_red', 'dec_open_blue']
        inputs, cards, offset = [], [], 0
        for path_id, path in enumerate(self.sampled_paths if self.sampled_paths is not None else [self.frame]):
            frame = read_frozen_predictions(path)
            if 'odds_type' in frame and not frame.odds_type.eq(self.odds_type).all():
                raise ValueError('Prediction odds_type differs from execution odds_type')
            values = frame[columns].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError('Saved fight inputs must be finite')
            if (not frame.winner.isin([0, 1]).all()
                    or not frame.proba_red.between(0, 1).all()
                    or not frame.risk_red_proba.between(0, 1).all()
                    or (frame.prediction_se < 0).any()
                    or (frame[[c for c in columns if c.startswith('dec_')]] <= 1).any().any()):
                raise ValueError('Invalid saved fight outcomes, probabilities, errors, or prices')
            inputs.append(frame[columns])
            for event_id, ids in enumerate(frame.groupby('date', sort=True).indices.values()):
                cards.append((path_id, event_id, offset + ids))
            offset += len(frame)
        combined = pd.concat(inputs, ignore_index=True)
        codes, unique = pd.factorize(pd.MultiIndex.from_frame(combined), sort=False)
        self._unique_fights = unique.to_frame(index=False)
        self._unique_fights.columns = columns
        self._unique_fights['proba_blue'] = 1 - self._unique_fights.proba_red
        self._unique_fights['pred_winner'] = (self._unique_fights.proba_red >= .5).astype(int)
        self._card_batches = {}
        for path_id, event_id, ids in cards:
            self._card_batches.setdefault(len(ids), []).append((path_id, event_id, codes[ids]))
        self._shared_opening_cache = (BettingReturnCalculator.calculate_scenario_cache(
            self._unique_fights, self.opening_params, self.odds_type)
            if isinstance(self, JointDecayBettingSearch) else None)

    def _simulate_shared_fights(self, params):
        from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import settle_bankroll_paths
        if not hasattr(self, '_unique_fights'):
            self._prepare_shared_fights()
        controls = self.betting_params(params)
        cache = BettingReturnCalculator.calculate_scenario_cache(
            self._unique_fights, controls, self.odds_type)
        winners = self._unique_fights.winner.to_numpy()
        shape = ((len(self.sampled_paths), self.sampled_paths[0].date.nunique())
                 if self.sampled_paths is not None else (1, self.frame.date.nunique()))
        ml, pl = np.zeros(shape), np.zeros(shape)
        stakes = np.zeros(shape)
        for cards in self._card_batches.values():
            for start in range(0, len(cards), 1024):
                batch = cards[start:start + 1024]
                paths = np.array([card[0] for card in batch])
                events = np.array([card[1] for card in batch])
                ids = np.stack([card[2] for card in batch])
                selected = cache[ids]
                if self._shared_opening_cache is None:
                    alloc = BettingReturnCalculator.scenario_allocations(
                        selected, winners[ids], controls, include_parlay=True)
                else:
                    prices = self._unique_fights[['dec_open_red', 'dec_open_blue']].to_numpy()
                    alloc = BettingReturnCalculator.decay_scenario_allocations(
                        self._shared_opening_cache[ids], selected, prices[ids], winners[ids],
                        self.opening_params, controls, controls.decay_rates, include_parlay=True)
                ml[paths, events] = (alloc['moneyline_stakes'] * selected[:, :, 2]).sum(axis=1)
                pl[paths, events] = (alloc['parlay_stakes'] * alloc['multiplier'] * alloc['parlay_profit'])
                stakes[paths, events] = (alloc['moneyline_stakes'].sum(axis=1)
                                        + alloc['parlay_stakes'] * alloc['multiplier'])
        if self.sampled_paths is None:
            ml, pl, stakes = (values[0, self.path_indices] for values in (ml, pl, stakes))
        result = settle_bankroll_paths(ml + pl, self.initial_bankroll,
                                      family_returns=dict(moneyline=ml, parlay=pl))
        if self.score_type in ('path_mean_sharpe', 'log_mean_roi'):
            result.update(event_roi_statistics(ml + pl, stakes, result['bankroll']))
        for family in ('moneyline', 'parlay'):
            result.pop(f'{family}_profits')
        return result

    def betting_params(self, trial_params):
        """Decode a trial into portable strategy controls."""
        return BettingParams.from_value(trial_params)

    def run(self, n_trials=100, seed=42):
        if n_trials < 1:
            raise ValueError('n_trials must be positive')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        study.set_user_attr('score_type', self.score_type)
        study.set_user_attr('risk_scaling', 'per_bet_edge_ratio_then_joint_exposure_cap')
        study.set_user_attr('wealth_floor', self.wealth_floor)
        if self.score_type == 'event_log_growth':
            study.set_user_attr('ulcer_penalty', self.ulcer_penalty)
            study.set_user_attr('ulcer_aggregation', 'mean of per-path RMS fractional drawdown over post-event balances')
            study.set_user_attr('drawdown_penalty', self.drawdown_penalty)
            study.set_user_attr('drawdown_tail_fraction', self.drawdown_tail_fraction)
            study.set_user_attr('drawdown_aggregation', 'mean of largest ceil(fraction * path_count) drawdowns')
        if self.score_type == 'log_mean_roi':
            study.set_user_attr('negative_mean_penalty', self.negative_mean_penalty)
        if self.score_type in ('path_mean_sharpe', 'log_mean_roi'):
            study.set_user_attr('path_mean_sharpe_basis', 'equal_weight_event_roi')
            study.set_user_attr('no_bet_events', 'excluded; trials with an empty path are pruned')
        study.set_user_attr('objective',
            'mean(log(max(1 + terminal_return, wealth_floor)))' if self.score_type == 'log_growth'
            else 'mean_paths(mean_events(diff(log(max(bankroll / initial_bankroll, wealth_floor))))) - drawdown_penalty * worst_tail_mean(path_max_drawdown, drawdown_tail_fraction) - ulcer_penalty * mean(path_ulcer_indices)' if self.score_type == 'event_log_growth'
            else 'mean(path_mean_event_returns) - risk_penalty * std(path_mean_event_returns, ddof=1)' if self.score_type == 'event_mean_std'
            else 'log(mean(path_mean_event_rois)) if mean > 0; -negative_mean_penalty if mean < 0; -inf if zero' if self.score_type == 'log_mean_roi'
            else 'mean(path_mean_event_rois) / std(path_mean_event_rois, ddof=1)' if self.score_type in ('path_mean_sharpe', 'log_mean_roi')
            else 'mean_terminal_return - risk_penalty * std_terminal_return')
        study.set_user_attr('risk_penalty', self.risk_penalty)
        study.set_user_attr('k', len(self.sampled_paths) if self.sampled_paths is not None else len(self.path_indices))
        study.set_user_attr('sampling_unit', 'saved_fight_paths' if self.sampled_paths is not None else 'events')
        study.set_user_attr('odds_type', self.odds_type)
        study.set_user_attr('sample_with_replacement', self.sample_with_replacement)
        if self.bootstrap_metadata is not None:
            study.set_user_attr('bootstrap_dir', str(self.bootstrap_dir))
            study.set_user_attr('dataset_sha256', self.bootstrap_metadata['dataset_sha256'])

        def objective(trial):
            simulation = self.simulate(self.suggest(trial))
            terminal = simulation['terminal_returns']
            trial.set_user_attr('mean_terminal_return', float(terminal.mean()))
            trial.set_user_attr('std_terminal_return', float(terminal.std(ddof=1)))
            trial.set_user_attr('loss_probability', float((terminal < 0).mean()))
            trial.set_user_attr('floored_path_fraction', float((1.0 + terminal < self.wealth_floor).mean()))
            if self.score_type == 'event_log_growth':
                means = path_mean_log_growth(simulation['bankroll'], self.wealth_floor)
                score = float(means.mean())
                trial.set_user_attr('mean_event_log_growth', score)
                trial.set_user_attr('geometric_growth_per_event', float(np.expm1(score)))
                drawdowns = path_max_drawdowns(simulation['bankroll'])
                mean_drawdown = float(drawdowns.mean())
                tail_drawdown = worst_tail_drawdown(drawdowns, self.drawdown_tail_fraction)
                penalty = self.drawdown_penalty * tail_drawdown
                trial.set_user_attr('worst_tail_max_drawdown', tail_drawdown)
                trial.set_user_attr('drawdown_tail_path_count', max(1, int(np.ceil(self.drawdown_tail_fraction * len(drawdowns)))))
                trial.set_user_attr('mean_max_drawdown', mean_drawdown)
                trial.set_user_attr('drawdown_penalty_amount', penalty)
                mean_ulcer = float(path_ulcer_indices(simulation['bankroll']).mean())
                ulcer_amount = self.ulcer_penalty * mean_ulcer
                trial.set_user_attr('mean_ulcer_index', mean_ulcer)
                trial.set_user_attr('ulcer_penalty_amount', ulcer_amount)
                return score - penalty - ulcer_amount
            if self.score_type in ('path_mean_sharpe', 'log_mean_roi'):
                means = simulation['path_mean_event_rois']
                if not np.isfinite(means).all():
                    raise optuna.TrialPruned('At least one path has no betting events or nonfinite ROI')
                trial.set_user_attr('mean_path_mean_event_roi', float(means.mean()))
                trial.set_user_attr('std_path_mean_event_roi', float(means.std(ddof=1)))
                trial.set_user_attr('mean_betting_event_coverage', float(simulation['betting_event_coverage'].mean()))
                if self.score_type == 'log_mean_roi':
                    return log_mean_roi_score(means, self.negative_mean_penalty)
                try:
                    return path_mean_sharpe_score(means)
                except ValueError as exc:
                    raise optuna.TrialPruned(str(exc)) from exc
            if self.score_type == 'event_mean_std':
                means = path_mean_event_returns(simulation['bankroll'])
                trial.set_user_attr('mean_path_mean_event_return', float(means.mean()))
                trial.set_user_attr('std_path_mean_event_return', float(means.std(ddof=1)))
                return float(means.mean() - self.risk_penalty * means.std(ddof=1))
            return terminal_return_score(terminal, self.risk_penalty,
                                         score_type=self.score_type, wealth_floor=self.wealth_floor)

        def report_progress(study, trial):
            print(f'{self.odds_type}: trial {trial.number + 1}/{n_trials} '
                  f'{trial.state.name}, objective={trial.value}', flush=True)
            if self.score_type == 'event_log_growth' and 'mean_max_drawdown' in trial.user_attrs:
                print(f"  Mean event log growth={trial.user_attrs['mean_event_log_growth']:.6g}; "
                      f"worst {self.drawdown_tail_fraction:.0%} mean max drawdown={trial.user_attrs['worst_tail_max_drawdown']:.2%}; "
                      f"drawdown penalty={trial.user_attrs['drawdown_penalty_amount']:.6g}; "
                      f"mean Ulcer Index={trial.user_attrs['mean_ulcer_index']:.2%}; "
                      f"ulcer penalty={trial.user_attrs['ulcer_penalty_amount']:.6g}", flush=True)
            if self.score_type in ('path_mean_sharpe', 'log_mean_roi') and 'mean_path_mean_event_roi' in trial.user_attrs:
                print(f"  Path-average event ROI: mean={trial.user_attrs['mean_path_mean_event_roi']:.6%}, "
                      f"SD={trial.user_attrs['std_path_mean_event_roi']:.6%}, "
                      f"betting-event coverage={trial.user_attrs['mean_betting_event_coverage']:.2%}", flush=True)
            if self.score_type == 'event_mean_std' and 'mean_path_mean_event_return' in trial.user_attrs:
                print(f"  Path-average event return: mean={trial.user_attrs['mean_path_mean_event_return']:.6%}, "
                      f"SD={trial.user_attrs['std_path_mean_event_return']:.6%}, "
                      f"risk_penalty={self.risk_penalty:g}", flush=True)
            try:
                best = study.best_trial
            except ValueError:
                print('  No completed trials yet.', flush=True)
                return
            print(f'  Best so far: trial {best.number + 1}, objective={best.value}', flush=True)
            print(f'  Best parameters (Optuna): {best.params}', flush=True)
            if isinstance(self, JointDecayBettingSearch):
                rates = {name: float(np.expm1(best.params['log1p_' + name]))
                         for name in self.rate_names}
                print(f'  Best decay rates (decoded): {rates}', flush=True)

        study.optimize(objective, n_trials=n_trials, callbacks=[report_progress])
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise ValueError('No completed trials; path-mean Sharpe requires nonzero variation across paths')
        if not np.isfinite(study.best_value):
            raise ValueError('No trial has a finite objective score')
        return study


class JointDecayBettingSearch(FrozenBettingSearch):
    """Tune one closing policy and its decay rates with a fixed opening policy."""

    rate_names = ('k_down', 'k_up', 'k_parlay_down', 'k_parlay_up')

    def __init__(self, predictions, odds_type, *, opening_params, opening_data=None,
                 max_decay=50., **kwargs):
        if odds_type not in ('close1', 'close2'):
            raise ValueError('Joint decay search requires close1 or close2')
        self.opening_params = BettingParams.from_value(opening_params, require_complete=True)
        if self.opening_params.decay_rates is not None or self.opening_params.opening_params is not None:
            raise ValueError('opening_params must be optimized without decay')
        if not np.isfinite(max_decay) or max_decay <= 0:
            raise ValueError('max_decay must be positive and finite')
        self.max_decay = float(max_decay)
        frame = read_frozen_predictions(predictions)
        frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
        keys = ['date', 'fighter_red', 'fighter_blue']
        prices = ['dec_open_red', 'dec_open_blue']
        if frame.duplicated(keys).any():
            raise ValueError('Closing predictions must contain unique fights')
        if opening_data is not None:
            reference = (opening_data.copy() if isinstance(opening_data, pd.DataFrame)
                         else pd.read_csv(opening_data, float_precision='round_trip'))
            reference['date'] = pd.to_datetime(reference.date, errors='raise').dt.normalize()
            if reference.duplicated(keys).any():
                raise ValueError('Opening data must contain unique fights')
            aligned = frame[keys + ['winner']].merge(
                reference[keys + prices + ['winner']], on=keys, how='outer',
                validate='one_to_one', indicator=True, suffixes=('', '_opening'))
            if not aligned['_merge'].eq('both').all() or not aligned.winner.eq(aligned.winner_opening).all():
                raise ValueError('Opening and closing data must contain the same fights and outcomes')
            reference = frame[keys].merge(reference[keys + prices], on=keys, validate='one_to_one', sort=False)
            for name in prices:
                if name in frame and not np.allclose(frame[name], reference[name], rtol=1e-10, atol=1e-10):
                    raise ValueError(f'Opening reference differs from closing data: {name}')
                frame[name] = reference[name].to_numpy()
        values = frame[prices].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 1).any():
            raise ValueError('Joint decay search requires valid original opening decimal odds')
        frame['proba_blue'] = 1 - frame.proba_red
        frame['pred_winner'] = (frame.proba_red >= .5).astype(int)
        super().__init__(frame, odds_type, **kwargs)
        # Opening controls stay fixed, but their raw stakes use closing prices
        # and predictions, exactly as in the shared decay backtest.
        self.opening_cache = BettingReturnCalculator.calculate_scenario_cache(
            self.frame, self.opening_params, odds_type)

    def suggest(self, trial):
        params = super().suggest(trial)
        for name in self.rate_names:
            coordinate = trial.suggest_float('log1p_' + name, 0., float(np.log1p(self.max_decay)))
            params['log1p_' + name] = coordinate
            trial.set_user_attr(name, float(np.expm1(coordinate)))
        return params

    def betting_params(self, trial_params):
        values = dict(trial_params)
        rates = {name: float(np.expm1(values.pop('log1p_' + name))) for name in self.rate_names}
        return BettingParams.from_value(values).with_decay(rates, self.opening_params)

    def event_returns(self, params):
        controls = self.betting_params(params)
        closing = BettingReturnCalculator.calculate_scenario_cache(self.frame, controls, self.odds_type)
        result = BettingReturnCalculator.calculate_decay_portfolio(
            self.opening_cache, closing,
            self.frame[['dec_open_red', 'dec_open_blue']].to_numpy(),
            self.frame.winner.to_numpy(), self.frame.date,
            self.opening_params, controls, controls.decay_rates, include_parlay=True)
        return result['events']

    def run(self, n_trials=100, seed=42):
        study = super().run(n_trials, seed)
        study.set_user_attr('use_decay', True)
        study.set_user_attr('opening_params', self.opening_params.to_dict())
        study.set_user_attr('adjustment_order', 'raw stakes -> decay -> closing z -> joint exposure cap')
        return study
