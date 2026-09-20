"""Betting-only search on frozen historical model predictions.

Score = mean terminal cumulative return / sample standard deviation across
equal-bankroll bootstrap paths. Whole events are sampled with replacement and
shuffled. Moneylines and within-event parlays settle together after each event.
"""
import numpy as np
import optuna
import pandas as pd

from ufc_betting.Models.RiskManagement.bayesian_search import BayesianSearchLR

ODDS_TYPES = ('open', 'close1', 'close2')


def log_growth(returns):
    returns = np.asarray(returns, dtype=float)
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError('Invalid event bankroll returns')
    if (returns == -1).any():
        return -np.inf
    return float(np.log1p(returns).sum())


def terminal_return_score(returns):
    """Cash scores zero; nonzero constant returns have an undefined ratio."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError('Need at least two finite terminal returns')
    mean, std = float(returns.mean()), float(returns.std(ddof=1))
    if std <= 1e-12:
        return 0.0 if np.all(np.abs(returns) <= 1e-12) else -np.inf
    return mean / std


def bootstrap_event_indices(n_events, k=1000, seed=42):
    if n_events < 1 or k < 2:
        raise ValueError('Need at least one event and k >= 2 paths')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_events, size=(k, n_events))
    for row in indices:
        rng.shuffle(row)
    return indices


def simulate_bankroll_paths(events, indices, initial_bankroll=500.0):
    """Repeated dates remain separate steps; families share ONE bankroll."""
    if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be positive and finite')
    returns = events.event_return.to_numpy(float)
    log_growth(returns)
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] < 1
            or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(events)).any()):
        raise ValueError('Invalid bootstrap event indices')
    sampled = returns[indices]
    bankroll = np.column_stack((np.full(len(indices), initial_bankroll),
                               initial_bankroll * np.cumprod(1 + sampled, axis=1)))
    if not np.isfinite(bankroll).all():
        raise ValueError('Nonfinite simulated bankroll')
    result = dict(bankroll=bankroll, terminal_returns=bankroll[:, -1] / initial_bankroll - 1)
    if {'moneyline_return', 'parlay_return'} <= set(events.columns):
        if not np.allclose(events.moneyline_return + events.parlay_return, returns):
            raise ValueError('Bet-family returns must sum to the portfolio return')
        for family in ('moneyline', 'parlay'):
            profit = bankroll[:, :-1] * events[f'{family}_return'].to_numpy(float)[indices]
            result[f'{family}_cumulative_returns'] = np.cumsum(profit, axis=1) / initial_bankroll
    return result


class FrozenBettingSearch:
    """No alpha selection or model fitting occurs inside betting trials."""

    def __init__(self, predictions, odds_type='open', *, k=1000,
                 initial_bankroll=500.0, seed=42):
        if odds_type not in ODDS_TYPES:
            raise ValueError(f'odds_type must be one of {ODDS_TYPES}')
        self.frame = predictions.copy()
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
        self.engine = BayesianSearchLR(
            X_train=pd.DataFrame(index=self.frame.index), y_train=self.frame.winner,
            real_odds=self.frame[[f'dec_{odds_type}_blue', f'dec_{odds_type}_red']],
            fair_odds=self.frame[[f'dec_fair_{odds_type}_blue', f'dec_fair_{odds_type}_red']],
            dates=self.frame.date, scale_encode=None,
        )
        for odds in (self.engine.real_odds, self.engine.fair_odds):
            if not np.isfinite(odds.to_numpy()).all() or (odds <= 1).any().any():
                raise ValueError('Decimal odds must be finite and exceed 1')
        if not np.isfinite(initial_bankroll) or initial_bankroll <= 0:
            raise ValueError('initial_bankroll must be positive and finite')
        self.odds_type, self.initial_bankroll = odds_type, float(initial_bankroll)
        self.path_indices = bootstrap_event_indices(self.frame.date.nunique(), k, seed)

    @staticmethod
    def suggest(trial):
        return dict(
            mdd=trial.suggest_float('mdd', .15, .70, step=.01),
            N=trial.suggest_int('N', 200, 1500, log=True),
            mdd_parlay=trial.suggest_float('mdd_parlay', .15, .70, step=.01),
            N_parlay=trial.suggest_int('N_parlay', 200, 2500, log=True),
            z=trial.suggest_float('z', 0.0, 2.5, step=.1),
            moneyline_min_american=trial.suggest_int('moneyline_min_american', -600, -100, step=25),
            moneyline_max_american=trial.suggest_int('moneyline_max_american', 100, 500, step=25),
        )

    def event_returns(self, params):
        events = self.engine.calculate_fight_returns(
            red_proba=self.frame.proba_red.to_numpy(),
            risk_red_proba=self.frame.risk_red_proba.to_numpy(),
            se=self.frame.prediction_se.to_numpy(),
            winner_fold=self.frame.winner.to_numpy(), val_idx=np.arange(len(self.frame)),
            param=params, include_parlays=True,
            include_heavy_favorite_parlays=False, return_components=True,
        )
        log_growth(events.event_return)
        return events

    def simulate(self, params):
        return simulate_bankroll_paths(self.event_returns(params), self.path_indices,
                                       self.initial_bankroll)

    def run(self, n_trials=100, seed=42):
        if n_trials < 1:
            raise ValueError('n_trials must be positive')
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        study.set_user_attr('objective', 'mean_terminal_cumulative_return / sample_std')
        study.set_user_attr('k', len(self.path_indices))
        study.set_user_attr('odds_type', self.odds_type)

        def objective(trial):
            terminal = self.simulate(self.suggest(trial))['terminal_returns']
            trial.set_user_attr('mean_terminal_return', float(terminal.mean()))
            trial.set_user_attr('std_terminal_return', float(terminal.std(ddof=1)))
            trial.set_user_attr('loss_probability', float((terminal < 0).mean()))
            return terminal_return_score(terminal)

        study.optimize(objective, n_trials=n_trials)
        if not np.isfinite(study.best_value):
            raise ValueError('No trial has a defined terminal-return ratio; increase window or k')
        return study
