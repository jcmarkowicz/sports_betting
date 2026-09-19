import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from ufc_betting.Models.RiskManagement.bayesian_search import BayesianSearchLR
from ufc_betting.Models.LogisticRegression.train_test_builder import (
    TrainTestBuilder,
)
from ufc_betting.config import config
from ufc_betting.Models.RiskManagement.sql_manager import OptunaSQLiteManager

def transform_with_fitted_preprocessor(
    X, scaler, encoder, cat_cols, num_cols
):
    """Transform a future month without refitting preprocessing objects."""
    index = X.index
    numeric = pd.DataFrame(
        scaler.transform(X[num_cols]), columns=num_cols, index=index
    )
    encoded_columns = encoder.get_feature_names_out(cat_cols)
    categorical = pd.DataFrame(
        encoder.transform(X[cat_cols]),
        columns=encoded_columns,
        index=index,
    )
    return pd.concat([numeric, categorical], axis=1)

def walk_forward_logit(
    builder,
    n_trials=50,
    n_splits=15,
    score_type='average_return',
    seed=42,
    refit_frequency_months=1,
    sqlite_path=None,
    overwrite_existing_study=False,
):
    """Run an expanding-window monthly Logit search and backtest.

    The builder's ``year``, ``month``, and ``day`` define the first holdout
    date. For each holdout month:

    1. Search Logit/Kelly parameters using only events before that month.
    2. Refit the Logit on all data before that month.
    3. Predict and evaluate bets for that month only.

    Returns the original holdout fight rows augmented with the columns needed
    by ``simulate_kelly``, plus monthly parameters and Optuna studies.
    """
    df = builder.all_processed_df.copy().reset_index(drop=True)
    odds_type = builder.odds_type
    X = df[builder.selected_feats].copy()
    y = df[builder.target_col].copy()
    dates = pd.to_datetime(df[builder.date_col], errors='raise').dt.normalize()
    real_odds = df[[
        f'dec_{odds_type}_blue', f'dec_{odds_type}_red'
    ]].copy()
    fair_odds = df[[
        f'dec_fair_{odds_type}_blue', f'dec_fair_{odds_type}_red'
    ]].copy()

    lengths = {
        len(X), len(y), len(real_odds), len(fair_odds), len(dates)
    }
    if len(lengths) != 1:
        raise ValueError(
            "X, y, real_odds, fair_odds, and dates must have equal lengths"
        )

    holdout_start = builder.start_date.normalize()
    holdout_months = sorted(
        dates.loc[dates >= holdout_start].dt.to_period('M').unique()
    )
    if not holdout_months:
        raise ValueError("No observations exist on or after holdout_start")
    if (
        not isinstance(refit_frequency_months, int)
        or refit_frequency_months < 1
    ):
        raise ValueError("refit_frequency_months must be a positive integer")

    if sqlite_path is None:
        sqlite_path = Path(__file__).resolve().parent / 'optuna_studies.db'
    study_manager = OptunaSQLiteManager(sqlite_path, seed=seed)

    cat_cols = X.select_dtypes(include='category').columns
    num_cols = X.select_dtypes(include='number').columns
    predictions = []
    monthly_results = []
    studies = {}
    previous_best_params = None
    fitted_model = None
    inference_model = None
    fitted_scaler = None
    fitted_encoder = None
    active_study = None
    active_train_rows = None
    active_train_events = None

    for month_number, month in enumerate(holdout_months):
        month_start = month.to_timestamp()
        next_month = month_start + pd.offsets.MonthBegin(1)
        evaluation_start = max(month_start, holdout_start)
        train_mask = dates < evaluation_start
        test_mask = (dates >= evaluation_start) & (dates < next_month)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        training_dates = dates.iloc[train_idx].reset_index(drop=True)
        unique_training_events = training_dates.nunique()
        month_splits = min(n_splits, unique_training_events)
        if month_splits < 2:
            raise ValueError(
                f"Month {month} has only {unique_training_events} prior "
                "event dates; at least two are required"
            )

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_month = X.iloc[test_idx].reset_index(drop=True)
        real_odds_train = real_odds.iloc[train_idx].reset_index(drop=True)
        fair_odds_train = fair_odds.iloc[train_idx].reset_index(drop=True)

        refit_performed = month_number % refit_frequency_months == 0
        print('HERE')
        if refit_performed:
            print(
                f"{month}: searching on {unique_training_events} prior "
                f"events; predicting {dates.iloc[test_idx].nunique()} events"
            )
            search = BayesianSearchLR(
                X_train=X_train,
                y_train=y_train,
                real_odds=real_odds_train,
                fair_odds=fair_odds_train,
                dates=training_dates,
                scale_encode=builder.scale_encode,
                model_type='logit',
                n_splits=month_splits,
                score_type=score_type,
            )
            study_name = f'walk_forward_{odds_type}_{score_type}_{month}'
            moneyline_study = study_manager.create_study(
                study_name=f'{study_name}_moneyline',
                overwrite=overwrite_existing_study,
            )
            parlay_study = study_manager.create_study(
                study_name=f'{study_name}_parlay',
                overwrite=overwrite_existing_study,
            )
            study = search.run(
                n_trials=n_trials,
                enqueue_params=previous_best_params,
                study=moneyline_study,
                parlay_study=parlay_study,
            )
            best_params = study.best_params.copy()
            previous_best_params = best_params.copy()
            studies[str(month)] = study
            active_study = study
            active_train_rows = len(train_idx)
            active_train_events = unique_training_events

            scaled = builder.scale_encode(
                X_train=X_train,
                X_test=X_month,
                cat_cols=cat_cols,
                num_cols=num_cols,
            )
            (
                red_proba,
                risk_red_proba,
                prediction_se,
                fitted_model,
                inference_model,
            ) = search.fit_logistic_model(
                X_train=scaled['X_train'],
                y_train=y_train,
                X_test=scaled['X_test'],
                alpha=best_params['alpha'],
                return_models=True,
            )
            fitted_scaler = scaled['scaler']
            fitted_encoder = scaled['encoder']
        else:
            print(
                f"{month}: reusing prior fitted model; predicting "
                f"{dates.iloc[test_idx].nunique()} events"
            )
            best_params = previous_best_params.copy()
            X_month_scaled = transform_with_fitted_preprocessor(
                X_month,
                fitted_scaler,
                fitted_encoder,
                cat_cols,
                num_cols,
            )
            X_month_sm = sm.add_constant(
                X_month_scaled, has_constant='add'
            )
            red_proba = np.asarray(fitted_model.predict(X_month_sm))
            selected = inference_model.model.exog_names
            inference = inference_model.get_prediction(
                X_month_sm[selected]
            ).summary_frame()
            risk_red_proba = inference['predicted'].to_numpy()
            prediction_se = inference['se'].to_numpy()

        month_dates = dates.iloc[test_idx].reset_index(drop=True)
        month_predictions = df.iloc[test_idx].copy().reset_index(drop=True)
        month_predictions['source_index'] = test_idx
        month_predictions['walk_forward_month'] = str(month)
        month_predictions['proba_red'] = red_proba
        month_predictions['proba_blue'] = 1 - red_proba
        month_predictions['risk_proba_red'] = risk_red_proba
        month_predictions['risk_proba_blue'] = 1 - risk_red_proba
        month_predictions['proba_se'] = prediction_se
        month_predictions['pred_winner'] = (
            red_proba >= 0.5
        ).astype(int)
        for name, value in best_params.items():
            month_predictions[f'best_{name}'] = value
        predictions.append(month_predictions)

        monthly_results.append({
            'month': str(month),
            'refit_performed': refit_performed,
            'train_rows': active_train_rows,
            'train_events': active_train_events,
            'test_rows': len(test_idx),
            'test_events': month_dates.nunique(),
            'best_cv_score': active_study.best_value,
            'moneyline_baseline_score': active_study.baseline_score,
            'parlay_improvement': active_study.improvement,
            'parlay_win_rate': active_study.parlay_win_rate,
            'best_moneyline_trial_number': (
                active_study.moneyline_study.best_trial.number
            ),
            'best_parlay_trial_number': (
                active_study.parlay_study.best_trial.number
            ),
            'best_trial_number': active_study.best_trial.number,
            **best_params,
        })

    betting_data = pd.concat(predictions, ignore_index=True)
    return {
        'betting_data': betting_data,
        'predictions': betting_data,
        'monthly_results': pd.DataFrame(monthly_results),
        'studies': studies,
    }

if __name__ == '__main__':
    df_model = pd.read_csv(
        r'C:\Users\jcmar\my_files\SportsBetting\data\training_data'
        r'\entire_odds_stats_2026-03-09.csv'
    )
    builder = TrainTestBuilder(
        df=df_model,
        feats=config.close1_feats,
        target_col='winner',
        date_col='date',
        odds_type='close1',
        year=2023,
        month=9,
        day=9,
    )
    results = walk_forward_logit(
        builder=builder,
        n_trials=75,
        n_splits=75,
        score_type='sharpe',
        seed=42,
        refit_frequency_months=3,
        sqlite_path=(
            Path(__file__).resolve().parent / 'close1_optuna_studies.db'
        ),
        overwrite_existing_study=True,
    )
    output_dir = Path(__file__).resolve().parent
    results['betting_data'].to_csv(
        output_dir / 'close1_walk_forward_betting_data.csv',
        index=False,
    )
    results['monthly_results'].to_csv(
        output_dir / 'close1_walk_forward_monthly_results.csv',
        index=False,
    )
    print(results['monthly_results'])
