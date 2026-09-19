"""Joint betting search on cached, frozen opening-model predictions.

No model fitting occurs here. Inputs use close1 prices and one prediction per
fight. The existing event settlement rules, top-two-EV parlays, and 100% event
exposure cap are reused. No-bet events contribute zero growth.
"""
import numpy as np
import optuna
import pandas as pd

from ufc_betting.Models.RiskManagement.bayesian_search import BayesianSearchLR


def log_growth(returns):
    returns = np.asarray(returns, dtype=float)
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("Invalid event bankroll returns")
    if (returns == -1).any():
        return -np.inf
    return float(np.log1p(returns).sum())


class FrozenBettingSearch:
    """Search only MDD/N for each bet family and moneyline American bounds."""

    def __init__(self, predictions):
        self.frame = predictions.sort_values('date', kind='stable').reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("The betting window has no events")
        self.frame['date'] = pd.to_datetime(self.frame.date).dt.normalize()
        p = self.frame.proba_red.to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid frozen probabilities")
        if not self.frame.winner.isin([0, 1]).all():
            raise ValueError("Betting evaluation requires binary settled outcomes")
        self.engine = BayesianSearchLR(
            X_train=pd.DataFrame(index=self.frame.index), y_train=self.frame.winner,
            real_odds=self.frame[['dec_close1_blue', 'dec_close1_red']],
            fair_odds=self.frame[['dec_fair_close1_blue', 'dec_fair_close1_red']],
            dates=self.frame.date, scale_encode=None,
        )
        for odds in (self.engine.real_odds, self.engine.fair_odds):
            if not np.isfinite(odds.to_numpy()).all() or (odds <= 1).any().any():
                raise ValueError("Decimal odds must be finite and exceed 1")
        self.fixed = dict(z=0.0)

    @staticmethod
    def suggest(trial):
        return dict(
            mdd=trial.suggest_float('mdd', .15, .70, step=.01),
            N=trial.suggest_int('N', 200, 1500, log=True),
            mdd_parlay=trial.suggest_float('mdd_parlay', .15, .70, step=.01),
            N_parlay=trial.suggest_int('N_parlay', 200, 2500, log=True),
            moneyline_min_american=trial.suggest_int('moneyline_min_american', -600, -100, step=25),
            moneyline_max_american=trial.suggest_int('moneyline_max_american', 100, 500, step=25),
        )

    def event_returns(self, params):
        params = {**params, **self.fixed}
        p = self.frame.proba_red.to_numpy()
        values = self.engine.calculate_fight_returns(
            red_proba=p, risk_red_proba=p, se=np.zeros(len(p)),
            winner_fold=self.frame.winner.to_numpy(), val_idx=np.arange(len(p)),
            param=params, include_parlays=True,
            include_heavy_favorite_parlays=False,
        )
        values = np.asarray(values, float)
        values[np.isnan(values)] = 0.0  # existing engine's no-bet sentinel
        log_growth(values)
        return pd.DataFrame({'date': self.frame.date.drop_duplicates().to_numpy(),
                             'event_return': values})

    def run(self, n_trials=100, seed=42):
        if n_trials < 1:
            raise ValueError("n_trials must be positive")
        study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=seed))
        def objective(trial):
            events = self.event_returns(self.suggest(trial))
            score = log_growth(events.event_return)
            trial.set_user_attr('compounded_return', float(np.expm1(score)))
            trial.set_user_attr('profitable_events', int((events.event_return > 0).sum()))
            return score
        study.optimize(objective, n_trials=n_trials)
        return study
