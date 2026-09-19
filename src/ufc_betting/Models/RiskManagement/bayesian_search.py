import numpy as np 
import pandas as pd 
import warnings
from dataclasses import dataclass

import optuna 
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from xgboost import XGBClassifier

from ufc_betting.BettingStrategy.kelly_worker import run_per_bet_scaling, parlay_top_ev
from ufc_betting.BettingStrategy.kelly_scaling import kelly_edge, expected_value

from ufc_betting.config import config 
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder


@dataclass
class TwoStageSearchResult:
    """Combined result from the moneyline and parlay Optuna studies."""
    moneyline_study: object
    parlay_study: object
    best_params: dict
    best_value: float
    best_trial: object
    baseline_score: float
    improvement: float
    parlay_win_rate: float



class BayesianSearchLR: 

    def __init__(
            self, 
            X_train, 
            y_train, 
            real_odds, 
            fair_odds, 
            dates,
            scale_encode,
            model_type='logit',
            stacking_feature_sets=None,
            stacking_inner_splits=5,
            n_splits=15,
            score_type='average_return',
            calibration_size=0.20,
            calibration_bins=10,
    ):
        self.X_train = X_train.copy()
        self.y_train = y_train

        # american to decimal here if needed 
        self.real_odds = real_odds
        self.fair_odds = fair_odds

        self.n_splits = n_splits
        self.dates = dates

        self.scale_encode = scale_encode
        self.score_type = score_type
        valid_score_types = {
            'sharpe',
            'sortino',
            'total_return',
            'average_return',
            'profitable_event_rate',
        }
        if self.score_type not in valid_score_types:
            raise ValueError(
                "score_type must be 'sharpe', 'sortino', 'total_return', "
                "'average_return', or 'profitable_event_rate'"
            )
        self.model_type = model_type.lower()
        if self.model_type not in {'logit', 'xgboost'}:
            raise ValueError("model_type must be 'logit' or 'xgboost'")
        if not 0 < calibration_size < 1:
            raise ValueError("calibration_size must be between 0 and 1")
        if calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")
        self.calibration_size = calibration_size
        self.calibration_bins = calibration_bins
        self.stacking_feature_sets = stacking_feature_sets
        self.stacking_inner_splits = stacking_inner_splits
        if self.model_type == 'xgboost':
            if not stacking_feature_sets:
                raise ValueError(
                    "stacking_feature_sets is required for model_type='xgboost'"
                )
            expected_length = len(self.X_train)
            if any(
                len(feature_set) != expected_length
                for feature_set in stacking_feature_sets.values()
            ):
                raise ValueError(
                    "Every stacking feature set must align with X_train"
                )
            if stacking_inner_splits < 2:
                raise ValueError("stacking_inner_splits must be at least 2")

    def run(
        self, n_trials=50, enqueue_params=None, study=None,
        parlay_study=None, two_stage=True,
    ):
        if two_stage and self.model_type == 'logit':
            return self.run_two_stage(
                n_trials=n_trials,
                enqueue_params=enqueue_params,
                moneyline_study=study,
                parlay_study=parlay_study,
            )
        return self.study(
            n_trials,
            enqueue_params=enqueue_params,
            study=study,
        )

    @staticmethod
    def _moneyline_param_names():
        return {
            'alpha', 'mdd', 'N', 'z',
            'moneyline_min_american', 'moneyline_max_american',
        }

    @staticmethod
    def _parlay_param_names():
        return {
            'mdd_parlay', 'heavy_parlay_mdd', 'N_parlay',
            'heavy_parlay_max_legs', 'heavy_favorite_cutoff',
        }

    def suggest_moneyline_params(self, trial):
        return {
            'mdd': trial.suggest_float('mdd', 0.15, 0.7, step=0.01),
            'N': trial.suggest_int('N', 200, 1500, log=True),
            'z': trial.suggest_float('z', 0.0, 2.5, step=0.1),
            'moneyline_min_american': trial.suggest_int(
                'moneyline_min_american', -600, -275, step=25
            ),
            'moneyline_max_american': trial.suggest_int(
                'moneyline_max_american', 150, 500, step=25
            ),
            'alpha': trial.suggest_float('alpha', 0.2, 7.0, log=True),
        }

    def suggest_parlay_params(self, trial):
        return {
            'heavy_parlay_mdd': trial.suggest_float(
                'heavy_parlay_mdd', 0.15, 0.7, step=0.01
            ),
            'mdd_parlay': trial.suggest_float(
                'mdd_parlay', 0.15, 0.7, step=0.01
            ),
            'N_parlay': trial.suggest_int(
                'N_parlay', 200, 2500, log=True
            ),
            'heavy_favorite_cutoff': trial.suggest_int(
                'heavy_favorite_cutoff', -600, -250, step=25
            ),
            'heavy_parlay_max_legs': trial.suggest_int(
                'heavy_parlay_max_legs', 2, 4
            ),
        }

    def run_two_stage(
        self, n_trials, enqueue_params=None, moneyline_study=None,
        parlay_study=None,
    ):
        """Optimize moneylines first, then parlays without refitting Logit."""
        if self.model_type != 'logit':
            raise ValueError('Two-stage search currently supports Logit only')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if moneyline_study is None:
            moneyline_study = optuna.create_study(direction='maximize')
        moneyline_objective_name = f'moneyline_{self.score_type}_v2'
        if (
            any(trial.value is not None for trial in moneyline_study.trials)
            and moneyline_study.user_attrs.get('objective') != moneyline_objective_name
        ):
            raise ValueError('Start a new moneyline study for the updated objective.')
        moneyline_study.set_user_attr('objective', moneyline_objective_name)
        if parlay_study is None:
            parlay_study = optuna.create_study(direction='maximize')
        objective_name = f'combined_improvement_{self.score_type}_independent_cutoff_v2'
        if (
            any(trial.value is not None for trial in parlay_study.trials)
            and parlay_study.user_attrs.get('objective') != objective_name
        ):
            raise ValueError(
                'Start a new parlay study: existing trials use a different objective.'
            )
        parlay_study.set_user_attr('objective', objective_name)

        if enqueue_params:
            queued = {
                key: value for key, value in enqueue_params.items()
                if key in self._moneyline_param_names()
            }
            if queued:
                moneyline_study.enqueue_trial(queued)

        def moneyline_objective(trial):
            return self.cross_validate(
                self.suggest_moneyline_params(trial),
                include_parlays=False,
            )

        print(f'Created Stage 1 moneyline study with n_trials={n_trials}')
        moneyline_study.optimize(moneyline_objective, n_trials=n_trials)
        moneyline_params = moneyline_study.best_params.copy()

        # Stage 2 reuses this cache; it never refits the Logit.
        oof_cache = self.build_oof_prediction_cache(
            alpha=moneyline_params['alpha']
        )
        baseline_score = self.score_cached_predictions(
            oof_cache, moneyline_params, include_parlays=False
        )

        if enqueue_params:
            queued = {
                key: value for key, value in enqueue_params.items()
                if key in self._parlay_param_names()
            }
            if queued:
                parlay_study.enqueue_trial(queued)

        def parlay_objective(trial):
            params = {
                **moneyline_params,
                **self.suggest_parlay_params(trial),
            }
            return self.score_cached_predictions(
                oof_cache, params, include_parlays=True
            ) - baseline_score

        print(f'Created Stage 2 parlay study with n_trials={n_trials}')
        parlay_study.optimize(parlay_objective, n_trials=n_trials)
        best_params = {**moneyline_params, **parlay_study.best_params}
        combined_score = self.score_cached_predictions(
            oof_cache, best_params, include_parlays=True
        )
        improvement = combined_score - baseline_score
        return TwoStageSearchResult(
            moneyline_study=moneyline_study,
            parlay_study=parlay_study,
            best_params={**moneyline_params, **parlay_study.best_params},
            best_value=baseline_score + improvement,
            best_trial=parlay_study.best_trial,
            baseline_score=baseline_score,
            improvement=improvement,
            parlay_win_rate=self.score_cached_parlay_win_rate(oof_cache, best_params),
        )

    def objective(
            self, trial
    ):
        param = {
            'mdd':trial.suggest_float('mdd', 0.15, 0.7, step=0.01, log=False),
            'N':trial.suggest_int('N', 200, 2500, log=True),
            'mdd_parlay':trial.suggest_float('mdd_parlay', 0.15, 0.7, step=0.01, log=False),
            'heavy_parlay_mdd': trial.suggest_float(
                'heavy_parlay_mdd', 0.15, 0.7, step=0.01
            ),
            'N_parlay':trial.suggest_int('N_parlay', 200, 2500, log=True),
            "z": trial.suggest_float("z", 0.0, 2.5, step=0.1),
            'moneyline_min_american': trial.suggest_int(
                'moneyline_min_american', -600, -275, step=25
            ),
            'moneyline_max_american': trial.suggest_int(
                'moneyline_max_american', 150, 500, step=25
            ),
            'heavy_favorite_cutoff': trial.suggest_int(
                'heavy_favorite_cutoff', -600, -100, step=25
            ),
            'heavy_parlay_max_legs': trial.suggest_int(
                'heavy_parlay_max_legs', 2, 4
            ),
        }
        if self.model_type == 'logit':
            param['alpha'] = trial.suggest_float(
                'alpha', 0.2, 7.0, log=True
            )
        else:
            param.update({
                **{
                    f'alpha_{name}': trial.suggest_float(
                        f'alpha_{name}', 0.2, 7.0, log=True
                    )
                    for name in self.stacking_feature_sets
                },
                'n_estimators': trial.suggest_int(
                    'n_estimators', 150, 250, step=25
                ),
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.005, 0.05, step=0.001
                ),
                'subsample': trial.suggest_float(
                    'subsample', 0.75, 1, step=0.01
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree', 0.75, 1, step=0.01
                ),
                'min_child_weight': trial.suggest_int(
                    'min_child_weight', 2, 6
                ),
                'reg_lambda': trial.suggest_float(
                    'reg_lambda', 1.0, 10.0, log=True
                ),
            })
        score = self.cross_validate(params=param)
        self.score = score
        return score

    def study(self, n_trials, enqueue_params=None, study=None):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if study is None:
            study = optuna.create_study(direction='maximize')
        if enqueue_params is not None:
            study.enqueue_trial(enqueue_params)
        print(f'Created study with n_trials={n_trials}')
        study.optimize(self.objective, n_trials=n_trials)
        return study

    def cross_validate(self, params, include_parlays=True):
        fold_ids = self.generate_folds(
            X=self.X_train,
            y=self.y_train,
            dates=self.dates,
            n_splits=self.n_splits,
        )
        oof_results = []
        cat_cols = self.X_train.select_dtypes('category').columns
        num_cols = self.X_train.select_dtypes('number').columns

        for fold_id in range(self.n_splits):
            val_idx = np.flatnonzero(fold_ids == fold_id)
            train_idx = np.flatnonzero(fold_ids != fold_id)

            n_events = pd.to_datetime(
                pd.Series(self.dates).iloc[val_idx]
            ).dt.normalize().nunique()

            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            if self.model_type == 'logit':
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                scale_pkt = self.scale_encode(
                    X_train=X_fold_train,
                    X_test=X_fold_val,
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                )
                red_proba, risk_red_proba, se = self.fit_logistic_model(
                    X_train=scale_pkt["X_train"],
                    y_train=y_fold_train,
                    X_test=scale_pkt["X_test"],
                    alpha=params['alpha'],
                )
            else:
                red_proba, risk_red_proba, se = (
                    self.fit_stacked_xgboost_outer_fold(
                    train_idx=train_idx,
                    val_idx=val_idx,
                    params=params,
                    )
                )

            # Apply the trial's MDD and calculate one total return per event.
            event_returns_fold = self.calculate_fight_returns(
                red_proba,
                risk_red_proba=risk_red_proba,
                se=se, 
                val_idx=val_idx, 
                winner_fold=y_fold_val, 
                param=params,
                include_parlays=include_parlays,
            )
            oof_results.extend(event_returns_fold)

        return self.pct_returns_score(np.asarray(oof_results, dtype=float))

    def build_oof_prediction_cache(self, alpha):
        """Fit each fold once and retain its validation predictions."""
        fold_ids = self.generate_folds(
            X=self.X_train, y=self.y_train, dates=self.dates,
            n_splits=self.n_splits,
        )
        cat_cols = self.X_train.select_dtypes('category').columns
        num_cols = self.X_train.select_dtypes('number').columns
        cache = []
        for fold_id in range(self.n_splits):
            val_idx = np.flatnonzero(fold_ids == fold_id)
            train_idx = np.flatnonzero(fold_ids != fold_id)
            scaled = self.scale_encode(
                X_train=self.X_train.iloc[train_idx],
                X_test=self.X_train.iloc[val_idx],
                cat_cols=cat_cols,
                num_cols=num_cols,
            )
            red_proba, risk_red_proba, se = self.fit_logistic_model(
                X_train=scaled['X_train'],
                y_train=self.y_train.iloc[train_idx],
                X_test=scaled['X_test'],
                alpha=alpha,
            )
            cache.append({
                'red_proba': red_proba,
                'risk_red_proba': risk_red_proba,
                'se': se,
                'winner_fold': self.y_train.iloc[val_idx],
                'val_idx': val_idx,
            })
        return cache

    def score_cached_predictions(self, cache, params, include_parlays):
        event_returns = []
        for fold in cache:
            event_returns.extend(self.calculate_fight_returns(
                red_proba=fold['red_proba'],
                risk_red_proba=fold['risk_red_proba'],
                se=fold['se'],
                winner_fold=fold['winner_fold'],
                val_idx=fold['val_idx'],
                param=params,
                include_parlays=include_parlays,
            ))
        return self.pct_returns_score(np.asarray(event_returns, dtype=float))

    def score_cached_parlay_win_rate(self, cache, params):
        """Pool ticket outcomes across folds, excluding unplaced parlays."""
        outcomes = []
        for fold in cache:
            outcomes.extend(self.calculate_fight_returns(
                red_proba=fold['red_proba'],
                risk_red_proba=fold['risk_red_proba'],
                se=fold['se'],
                winner_fold=fold['winner_fold'],
                val_idx=fold['val_idx'],
                param=params,
                include_parlays=True,
                return_parlay_outcomes=True,
            ))
        return float(np.mean(outcomes)) if outcomes else 0.0

    def fit_logistic_model(
        self, X_train, y_train, X_test, alpha, return_models=False
    ):

        X_fold_train = sm.add_constant(
            X_train,
            has_constant="add",
        )
        X_fold_val = sm.add_constant(
            X_test,
            has_constant="add",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = sm.Logit(
                endog=y_train, exog=X_fold_train
            ).fit_regularized(
                method='l1', alpha=alpha, disp=False
            )

        model_results = model.predict(X_fold_val)
        red_proba = np.asarray(model_results)

        trimmed = np.asarray(model.mle_retvals["trimmed"])
        selected = model.params.index[~trimmed]
        inference_model = sm.Logit(
            y_train,
            X_fold_train[selected],
        ).fit(disp=False)

        prediction = inference_model.get_prediction(
            X_fold_val[selected]
        ).summary_frame()
        predictions = (
            red_proba,
            prediction['predicted'].to_numpy(),
            prediction['se'].to_numpy(),
        )
        if return_models:
            return predictions + (model, inference_model)
        return predictions

    @staticmethod
    def fit_logistic_predictions(X_train, y_train, X_test, alpha):
        X_train = sm.add_constant(X_train, has_constant='add')
        X_test = sm.add_constant(X_test, has_constant='add')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model = sm.Logit(y_train, X_train).fit_regularized(
                method='l1', alpha=alpha, disp=False
            )
        return np.asarray(model.predict(X_test))

    def fit_stacked_xgboost_outer_fold(
            self, train_idx, val_idx, params
        ):
        outer_dates = (
            pd.to_datetime(pd.Series(self.dates).iloc[train_idx])
            .dt.normalize()
            .reset_index(drop=True)
        )
        outer_y = self.y_train.iloc[train_idx]
        n_inner_splits = min(
            self.stacking_inner_splits,
            outer_dates.nunique(),
        )
        if n_inner_splits < 2:
            raise ValueError(
                "At least two outer-training event dates are required for stacking"
            )

        reference_X = next(iter(self.stacking_feature_sets.values())).iloc[
            train_idx
        ]
        inner_fold_ids = self.generate_folds(
            X=reference_X,
            y=outer_y,
            dates=outer_dates,
            n_splits=n_inner_splits,
        )

        model_names = list(self.stacking_feature_sets)
        meta_train = np.empty((len(train_idx), len(model_names)), dtype=float)
        meta_validation = np.empty((len(val_idx), len(model_names)), dtype=float)

        for column_idx, model_name in enumerate(model_names):
            feature_set = self.stacking_feature_sets[model_name]
            alpha = params[f'alpha_{model_name}']

            for inner_fold_id in range(n_inner_splits):
                inner_val_pos = np.flatnonzero(
                    inner_fold_ids == inner_fold_id
                )
                inner_train_pos = np.flatnonzero(
                    inner_fold_ids != inner_fold_id
                )
                inner_train_idx = train_idx[inner_train_pos]
                inner_val_idx = train_idx[inner_val_pos]

                X_inner_train = feature_set.iloc[inner_train_idx]
                X_inner_val = feature_set.iloc[inner_val_idx]
                cat_cols = X_inner_train.select_dtypes('category').columns
                num_cols = X_inner_train.select_dtypes('number').columns
                scale_pkt = self.scale_encode(
                    X_train=X_inner_train,
                    X_test=X_inner_val,
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                )
                meta_train[inner_val_pos, column_idx] = (
                    self.fit_logistic_predictions(
                        X_train=scale_pkt['X_train'],
                        y_train=self.y_train.iloc[inner_train_idx],
                        X_test=scale_pkt['X_test'],
                        alpha=alpha,
                    )
                )

            X_outer_train = feature_set.iloc[train_idx]
            X_outer_val = feature_set.iloc[val_idx]
            cat_cols = X_outer_train.select_dtypes('category').columns
            num_cols = X_outer_train.select_dtypes('number').columns
            scale_pkt = self.scale_encode(
                X_train=X_outer_train,
                X_test=X_outer_val,
                cat_cols=cat_cols,
                num_cols=num_cols,
            )
            meta_validation[:, column_idx] = self.fit_logistic_predictions(
                X_train=scale_pkt['X_train'],
                y_train=outer_y,
                X_test=scale_pkt['X_test'],
                alpha=alpha,
            )

        meta_columns = [f'proba_red_{name}' for name in model_names]
        meta_train = pd.DataFrame(meta_train, columns=meta_columns)
        meta_validation = pd.DataFrame(
            meta_validation, columns=meta_columns
        )
        return self.fit_xgboost_model(
            X_train=meta_train,
            y_train=outer_y.reset_index(drop=True),
            X_test=meta_validation,
            training_dates=outer_dates.to_numpy(),
            params=params,
        )

    def fit_xgboost_model(
            self, X_train, y_train, X_test, training_dates, params
        ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.calibration_size,
            random_state=42,
        )
        fit_idx, calibration_idx = next(
            splitter.split(X_train, y_train, groups=training_dates)
        )

        X_fit = X_train.iloc[fit_idx]
        y_fit = y_train.iloc[fit_idx]
        X_calibration = X_train.iloc[calibration_idx]
        y_calibration = y_train.iloc[calibration_idx]

        if y_fit.nunique() < 2 or y_calibration.nunique() < 2:
            raise ValueError(
                "XGBoost fit and calibration partitions must contain both classes"
            )

        model = XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            reg_lambda=params['reg_lambda'],
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_fit, y_fit)

        calibration_proba = model.predict_proba(X_calibration)[:, 1]
        red_proba = model.predict_proba(X_test)[:, 1]
        calibrated_proba, calibration_se = self.calibration_bin_stats(
            calibration_proba=calibration_proba,
            calibration_y=np.asarray(y_calibration),
            prediction_proba=red_proba,
            n_bins=self.calibration_bins,
        )
        return red_proba, calibrated_proba, calibration_se

    @staticmethod
    def calibration_bin_stats(
            calibration_proba, calibration_y, prediction_proba, n_bins
        ):
        calibration_proba = np.asarray(calibration_proba, dtype=float)
        calibration_y = np.asarray(calibration_y, dtype=int)
        prediction_proba = np.asarray(prediction_proba, dtype=float)

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(calibration_proba, quantiles))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

        calibration_bins = np.digitize(
            calibration_proba, edges[1:-1], right=True
        )
        prediction_bins = np.digitize(
            prediction_proba, edges[1:-1], right=True
        )

        calibrated = np.empty(len(prediction_proba), dtype=float)
        standard_error = np.empty(len(prediction_proba), dtype=float)
        global_wins = calibration_y.sum()
        global_n = len(calibration_y)
        for bin_id in np.unique(prediction_bins):
            in_calibration_bin = calibration_bins == bin_id
            n = in_calibration_bin.sum()
            wins = calibration_y[in_calibration_bin].sum()
            if n == 0:
                n = global_n
                wins = global_wins

            # Beta(1, 1) smoothing avoids probabilities of exactly 0 or 1
            # and a misleading zero uncertainty in small calibration bins.
            alpha = wins + 1
            beta = n - wins + 1
            probability = alpha / (alpha + beta)
            se = np.sqrt(
                alpha * beta
                / ((alpha + beta) ** 2 * (alpha + beta + 1))
            )

            output_rows = prediction_bins == bin_id
            calibrated[output_rows] = probability
            standard_error[output_rows] = se

        return calibrated, standard_error

    def generate_folds(self, X, y, dates, n_splits=5, random_state=42):
    # Ensure every timestamp from the same calendar day becomes one group
        event_dates = (
            pd.to_datetime(dates)
            .dt.normalize()
            .reset_index(drop=True)
        )

        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        n_dates = event_dates.nunique()

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if n_splits > n_dates:
            raise ValueError(
                f"n_splits={n_splits} exceeds the number "
                f"of unique event dates ({n_dates})"
            )

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        fold_ids = np.full(len(X), -1, dtype=int)

        for fold_id, (_, val_idx) in enumerate(
            splitter.split(
                X=X,
                y=y,
                groups=event_dates,
            )
        ):
            fold_ids[val_idx] = fold_id

        return fold_ids

    def calculate_fight_returns(
            self, red_proba, risk_red_proba, se,
            winner_fold, val_idx, param, include_parlays=True,
            return_parlay_outcomes=False,
            include_heavy_favorite_parlays=True,
        ):
        pred_class = (red_proba >= 0.5).astype(int)
        winner_fold = np.asarray(winner_fold)
        pred_win = pred_class == winner_fold

        pred_proba = np.where(
            pred_class == 1,
            red_proba,
            1 - red_proba
        )
        risk_pred_proba = np.where(
            pred_class == 1,
            risk_red_proba,
            1 - risk_red_proba,
        )

        fair_odds_fold = self.fair_odds.iloc[val_idx].to_numpy()
        real_odds_fold = self.real_odds.iloc[val_idx].to_numpy()

        pred_class = np.asarray(pred_class, dtype=int)
        row_idx = np.arange(len(pred_class))

        pred_fair_odds = fair_odds_fold[row_idx, pred_class]
        pred_real_odds = real_odds_fold[row_idx, pred_class]
        pred_american_odds = np.array([
            round(self.decimal_to_american(odds))
            for odds in pred_real_odds
        ])
        moneyline_allowed = (
            (pred_american_odds >= param['moneyline_min_american'])
            & (pred_american_odds <= param['moneyline_max_american'])
        )

        pred_ev = expected_value(
            p = pred_proba, o=pred_real_odds
        )

        kelly = np.array([
            kelly_edge(p, odds)
            for p, odds in zip(pred_proba, pred_fair_odds)
        ])
        kelly = np.where(moneyline_allowed, kelly, 0.0)

        bets_df = pd.DataFrame({
            'f_star_unscaled':kelly, 
            'p':pred_proba, 
            'fair_odds':pred_fair_odds,
            'real_odds':pred_real_odds,
            'ev':pred_ev
        })
        winner_id = pd.DataFrame({
            'pred_win':pred_win,
            'id':np.arange(0,len(pred_proba))
        })

        results_df = run_per_bet_scaling(
            bets_df=bets_df, 
            max_drawdown=param['mdd'], 
            bankroll=500, 
            N=param['N']
        )

        kelly_scaled = results_df['fstar_scaled'].to_numpy()
        pct_returns = np.where(
            kelly_scaled > 0,
            np.where(
                pred_win,
                kelly_scaled * (pred_real_odds - 1),
                -kelly_scaled,
            ),
            np.nan,
        )
        validation_dates = (
            pd.to_datetime(
                pd.Series(self.dates).iloc[val_idx]
            )
            .dt.normalize()
            .to_numpy()
        )
        fold_results = pd.DataFrame({
            "date": validation_dates,
            "pct_return": pct_returns,
            "fstar": kelly_scaled,
            'fair_odds':pred_fair_odds,
            'se':se,
            'choice_proba':pred_proba,
            'risk_proba':risk_pred_proba,
        })
        dat = pd.DataFrame({
            'choice_fighter_name':np.zeros(len(pred_proba)),  
            'choice_fighter_bool':pred_class,
            'choice_proba':pred_proba,
            'choice_real_odds':pred_real_odds,
            'choice_ev':pred_ev, 
            'choice_sharpe':results_df['sharpe'].to_numpy(),
            'choice_american_odds':pred_american_odds,
            'moneyline_allowed':moneyline_allowed,
            'id':np.arange(0,len(pred_proba)),
            'date':validation_dates
        })
        if include_parlays:
            parlay_scores = self.parlay_returns(
                dat, winner_id, param, strategy='top_ev'
            )
            heavy_parlay_scores = (
                self.parlay_returns(dat, winner_id, param, strategy='heavy_favorite')
                if include_heavy_favorite_parlays else pd.DataFrame(
                    columns=['date', 'pct_return', 'fstar', 'selected_ids']
                )
            )
        else:
            empty_columns = ['date', 'pct_return', 'fstar', 'selected_ids']
            parlay_scores = pd.DataFrame(columns=empty_columns)
            heavy_parlay_scores = pd.DataFrame(columns=empty_columns)

        event_returns = []
        parlay_outcomes = []
        for date, group in fold_results.groupby("date", sort=False):
            placed_moneylines = group["pct_return"].notna()
            moneyline_returns = group.loc[
                placed_moneylines, "pct_return"
            ].to_numpy()
            moneyline_fstars = group.loc[
                placed_moneylines, "fstar"
            ].to_numpy()

            parlay_result = parlay_scores.loc[
                parlay_scores["date"].eq(date)
            ]
            parlay_return = np.nan
            parlay_fstar = 0.0
            if not parlay_result.empty:
                parlay_return = parlay_result["pct_return"].iloc[0]
                parlay_fstar = parlay_result["fstar"].iloc[0]

            heavy_parlay_result = heavy_parlay_scores.loc[
                heavy_parlay_scores['date'].eq(date)
            ]
            heavy_parlay_return = np.nan
            heavy_parlay_fstar = 0.0
            if not heavy_parlay_result.empty:
                heavy_parlay_return = heavy_parlay_result[
                    'pct_return'
                ].iloc[0]
                heavy_parlay_fstar = heavy_parlay_result['fstar'].iloc[0]

            total_fstar = (
                moneyline_fstars.sum()
                + parlay_fstar
                + heavy_parlay_fstar
            )
            if total_fstar <= 0:
                event_returns.append(np.nan)
                continue

            p_vegas = 1 / group["fair_odds"].to_numpy()
            p_model = group['risk_proba'].to_numpy()
            pred_se = group['se'].to_numpy()

            raw_edges = np.maximum(
                p_model - p_vegas,
                0,
            )
            conservative_edges = np.maximum(
                p_model - param['z'] * pred_se - p_vegas,
                0,
            )

            uncertainty_weights = group['fstar'].to_numpy(
                dtype=float, copy=True
            )
            group_ids = group.index.to_numpy()
            for result in (parlay_result, heavy_parlay_result):
                if result.empty or result['fstar'].iloc[0] <= 0:
                    continue
                selected_ids = result['selected_ids'].iloc[0]
                if not selected_ids:
                    continue
                leg_weight = result['fstar'].iloc[0] / len(selected_ids)
                uncertainty_weights += (
                    np.isin(group_ids, selected_ids) * leg_weight
                )

            raw_score = np.sum(uncertainty_weights * raw_edges)
            conservative_score = np.sum(
                uncertainty_weights * conservative_edges
            )
                
            uncertainty_multiplier = (
                np.clip(conservative_score / raw_score, 0, 1)
                if raw_score > 0
                else 0.0
            )
            event_return = moneyline_returns.sum()
            if pd.notna(parlay_return):
                event_return += parlay_return
            if pd.notna(heavy_parlay_return):
                event_return += heavy_parlay_return

            exposure_after_uncertainty = (
                total_fstar * uncertainty_multiplier
            )
            exposure_multiplier = (
                min(1.0, 1.0 / exposure_after_uncertainty)
                if exposure_after_uncertainty > 0
                else 1.0
            )
            if uncertainty_multiplier * exposure_multiplier > 0:
                for ticket_return in (parlay_return, heavy_parlay_return):
                    if pd.notna(ticket_return):
                        parlay_outcomes.append(bool(ticket_return > 0))
            event_returns.append(
                event_return
                * uncertainty_multiplier
                * exposure_multiplier
            )

        return parlay_outcomes if return_parlay_outcomes else event_returns

    @staticmethod
    def decimal_to_american(decimal_odds):
        decimal_odds = float(decimal_odds)
        if not np.isfinite(decimal_odds) or decimal_odds <= 1:
            raise ValueError(
                "All real decimal odds must be finite and greater than 1"
            )
        if decimal_odds >= 2:
            return 100 * (decimal_odds - 1)
        return -100 / (decimal_odds - 1)

    def parlay_returns(self, dat, winner_id, param, strategy='top_ev'):

        pct_returns = []
        for date, group in dat.groupby('date'):
            sort_column = 'choice_ev'
            if strategy == 'top_ev':
                candidates = group
                selected = candidates.sort_values(
                    sort_column, ascending=False
                ).head(2)
            elif strategy == 'heavy_favorite':
                candidates = group.loc[
                    (group['choice_ev'] > 0)
                    & (group['choice_american_odds'] <= param['heavy_favorite_cutoff'])
                ].sort_values(sort_column, ascending=False)
                selected = candidates.head(
                    param['heavy_parlay_max_legs']
                )
                if len(selected) < 2:
                    selected = candidates.iloc[0:0]
            else:
                raise ValueError(
                    "strategy must be 'top_ev' or 'heavy_favorite'"
                )

            selected_ids = selected['id'].tolist()
            if len(selected) < 2:
                pct_returns.append({
                    'date': date,
                    'pct_return': np.nan,
                    'fstar': 0.0,
                    'selected_ids': [],
                    'strategy': strategy,
                })
                continue

            parlay_df = parlay_top_ev(
                data=selected,
                bankroll=500, 
                type='open', 
                top_n=list(range(len(selected))),
                parlay_mdd=(
                    param.get('heavy_parlay_mdd', param['mdd_parlay'])
                    if strategy == 'heavy_favorite' else param['mdd_parlay']
                ),
                N=param['N_parlay']
            )
            if parlay_df.isna().all().all():
                pct_returns.append({
                    'date': date,
                    'pct_return': np.nan,
                    'fstar': 0.0,
                    'selected_ids': [],
                    'strategy': strategy,
                })
                continue

            parlay_df = pd.merge(
                parlay_df, winner_id, how='left', on='id'
            )
            parlay_fstar = parlay_df['parlay_fstar_open'].iloc[0]
            parlay_net_odds = parlay_df['parlay_odds_open'].iloc[0]
            parlay_wins = (
                parlay_df["pred_win"].notna().all()
                and parlay_df["pred_win"].all()
            )

            if not np.isfinite(parlay_fstar) or parlay_fstar <= 0:
                pct_return_parlay = np.nan
            elif parlay_wins:
                pct_return_parlay = parlay_fstar * parlay_net_odds
            else: 
                pct_return_parlay = -parlay_fstar
            pct_returns.append({
                'date': date,
                'pct_return': pct_return_parlay,
                'fstar': parlay_fstar if pd.notna(pct_return_parlay) else 0.0,
                'selected_ids': selected_ids,
                'strategy': strategy,
            })
            
        return pd.DataFrame(pct_returns)

    def pct_returns_score(
            self, pct_returns
    ):
        """ 
        Score event bankroll returns. For average_return, no-bet events count
        as zero; other metrics omit them. Returns are fractions (0.01 = 1%).
        """
        pct_returns = np.asarray(pct_returns, dtype=float)
        if self.score_type == 'average_return':
            # Every validation event counts; abstaining contributes zero.
            return float(np.nan_to_num(pct_returns, nan=0.0).mean()) if len(pct_returns) else 0.0
        returns = pct_returns[~np.isnan(pct_returns)]
        if self.score_type == 'profitable_event_rate':
            if len(returns) == 0:
                return 0.0
            return float(np.mean(returns > 0))

        if self.score_type == 'total_return':
            return returns.sum()
        
        if len(returns) < 2:
            return np.nan
        
        if self.score_type == 'sortino':
            downside = np.minimum(returns, 0.0)
            downside_deviation = np.sqrt(np.mean(downside ** 2))
            if downside_deviation == 0:
                print('Sortino Ratio score resulted in all events with positive returns. Returning np.inf.')
                return np.inf if returns.mean() > 0 else 0.0
            return returns.mean() / downside_deviation

        if self.score_type != 'sharpe':
            raise ValueError(
                "score_type must be 'sharpe', 'sortino', 'total_return', "
                "'average_return', or 'profitable_event_rate'"
            )

        volatility = returns.std(ddof=1)
        if not np.isfinite(volatility) or volatility == 0:
            return 0.0

        return returns.mean() / volatility



if __name__ == "__main__":


    fp = r'C:\Users\jcmar\my_files\SportsBetting\data\training_data\entire_odds_stats_2026-03-07.csv'
    df_model = pd.read_csv(fp)
    odds_type = 'close1'
    stacking_features = {
        'open': config.open_feats,
        'close1': config.close1_feats,
        'close2': config.close2_feats,
    }
    feats = list(dict.fromkeys(
        feature
        for feature_list in stacking_features.values()
        for feature in feature_list
    ))
    date_col = 'date'
    winner_col = 'winner'

    builder = TrainTestBuilder(
        df=df_model,
        feats=feats,
        target_col=winner_col,
        date_col=date_col,
        odds_type=odds_type,
        year=2010, 
        month=2, 
        day=26, 
    )
    pkt = builder.prepare_train_test(
        train_size=0.85, scale=False
    )
    filtered_df = pkt['filtered_df']
    dates = filtered_df[date_col]

    # col order must be 0, 1 (blue, red) 
    real_odds = filtered_df[[f'dec_{odds_type}_blue', f'dec_{odds_type}_red']]
    fair_odds = filtered_df[[f'dec_fair_{odds_type}_blue', f'dec_fair_{odds_type}_red']]
    base_feature_sets = {
        name: pkt['X_train'][feature_list].copy()
        for name, feature_list in stacking_features.items()
    }

    search = BayesianSearchLR(
        X_train=pkt['X_train'],
        y_train=pkt['y_train'],
        real_odds=real_odds,
        fair_odds=fair_odds,
        dates=pkt['dates_train'],
        scale_encode=builder.scale_encode,
        model_type='logit',
        stacking_feature_sets=base_feature_sets,
        stacking_inner_splits=5,
        n_splits=15,
        score_type='average_return'
    )
    study = search.run(n_trials=30)

    print("Best trial number:", study.best_trial.number)
    print("Best parameters:", study.best_params)
    print("Best average score:", study.best_value)
