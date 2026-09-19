"""Joint risk search on cached, fixed-model predictions at close1 prices.

No fitting or prediction occurs here. Scores are mean log bankroll growth
across completed three-month blocks (including zero-return no-bet events).
Both existing parlay strategies share the searched parlay MDD and N.
"""
import numpy as np
import pandas as pd
import optuna

from ufc_betting.Models.RiskManagement.bayesian_search import BayesianSearchLR


SEARCH_NAMES = {
    'mdd', 'N', 'mdd_parlay', 'N_parlay',
    'moneyline_min_american', 'moneyline_max_american',
}


def growth_metrics(returns, initial_bankroll=1.0):
    returns = np.asarray(returns, dtype=float)
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError('Event returns must be finite and at least -1')
    wealth = np.r_[1., np.cumprod(1 + returns)]
    return {
        'compounded_return': float(wealth[-1] - 1),
        'ending_bankroll': float(initial_bankroll * wealth[-1]),
        'max_drawdown': float(np.max(1 - wealth / np.maximum.accumulate(wealth))),
        'profitable_events': int((returns > 0).sum()),
        'events': len(returns),
    }


class FixedOpenBayesianSearch:
    def __init__(self, predictions, *, z=0.5, heavy_favorite_cutoff=-300,
                 heavy_parlay_max_legs=4):
        self.frame = predictions.sort_values('date', kind='stable').reset_index(drop=True).copy()
        self.frame['date'] = pd.to_datetime(self.frame.date).dt.normalize()
        if self.frame.empty:
            raise ValueError('Search requires events')
        required = ['proba_red', 'proba_se', 'winner', 'dec_close1_blue',
                    'dec_close1_red', 'dec_fair_close1_blue', 'dec_fair_close1_red']
        values = self.frame[required].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Predictions, prices, and outcomes must be finite')
        if not self.frame.winner.isin([0, 1]).all():
            raise ValueError('All fights must have settled binary outcomes')
        if not self.frame.proba_red.between(0, 1).all() or (self.frame.proba_se < 0).any():
            raise ValueError('Invalid probabilities or standard errors')
        if (self.frame[required[3:]] <= 1).any().any():
            raise ValueError('Decimal odds must exceed 1')
        if not np.isfinite(z) or z < 0:
            raise ValueError('z must be finite and nonnegative')
        self.fixed = dict(z=z, heavy_favorite_cutoff=heavy_favorite_cutoff,
                          heavy_parlay_max_legs=heavy_parlay_max_legs)
        # Reuse settlement only; the legacy search/model fitting methods are never called.
        self.evaluator = BayesianSearchLR(
            X_train=pd.DataFrame(index=self.frame.index), y_train=self.frame.winner,
            real_odds=self.frame[['dec_close1_blue', 'dec_close1_red']],
            fair_odds=self.frame[['dec_fair_close1_blue', 'dec_fair_close1_red']],
            dates=self.frame.date, scale_encode=None,
        )

    def event_returns(self, params):
        if set(params) != SEARCH_NAMES:
            raise ValueError(f'Parameters must be exactly {sorted(SEARCH_NAMES)}')
        if params['moneyline_min_american'] > params['moneyline_max_american']:
            raise ValueError('Reversed moneyline bounds')
        p = self.frame.proba_red.to_numpy()
        returns = self.evaluator.calculate_fight_returns(
            red_proba=p, risk_red_proba=p, se=self.frame.proba_se.to_numpy(),
            winner_fold=self.frame.winner.to_numpy(), val_idx=np.arange(len(self.frame)),
            param={**params, **self.fixed}, include_parlays=True,
        )
        result = pd.DataFrame({'date': self.frame.date.drop_duplicates().to_numpy(),
                               'event_return': returns})
        # Legacy settlement uses NaN exclusively for events with no stakes.
        result['event_return'] = result.event_return.fillna(0.)
        growth_metrics(result.event_return)
        return result

    def score(self, params, block_start):
        events = self.event_returns(params)
        start = pd.Timestamp(block_start)
        if start.day != 1 or events.date.min() < start:
            raise ValueError('block_start must be a month boundary before all events')
        month = (events.date.dt.year - start.year) * 12 + events.date.dt.month - start.month
        if (events.event_return <= -1).any():
            return -np.inf
        logs = pd.Series(np.log1p(events.event_return.to_numpy()))
        block_logs = logs.groupby((month // 3).to_numpy()).sum()
        # Empty calendar blocks contribute zero growth.
        return float(block_logs.reindex(range(int(month.max() // 3) + 1), fill_value=0).mean())

    def run(self, block_start, n_trials=100, seed=42, enqueue_params=None):
        if n_trials < 1:
            raise ValueError('n_trials must be positive')
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        if enqueue_params is not None:
            study.enqueue_trial(enqueue_params)

        def objective(trial):
            params = {
                'mdd': trial.suggest_float('mdd', .15, .7, step=.01),
                'N': trial.suggest_int('N', 200, 1500, log=True),
                'mdd_parlay': trial.suggest_float('mdd_parlay', .15, .7, step=.01),
                'N_parlay': trial.suggest_int('N_parlay', 200, 2500, log=True),
                'moneyline_min_american': trial.suggest_int('moneyline_min_american', -600, -275, step=25),
                'moneyline_max_american': trial.suggest_int('moneyline_max_american', 150, 500, step=25),
            }
            return self.score(params, block_start)

        study.optimize(objective, n_trials=n_trials)
        if not np.isfinite(study.best_value):
            raise ValueError('No non-bankrupt candidate found')
        return study
