"""Build non-merged fight features and score the opening model at close prices.

This module deliberately uses the project's normal feature-engineering pipeline.
Historical stats/odds provide prior-fight context; non-merged stats/odds are the
target fights. Output is restricted to event dates present in both non-merged
source files.

Run:
    python -m ufc_betting.Models.RiskManagement.build_non_merged_open_predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.DataPipeline.dataframes.odds import OddsRepository
from ufc_betting.DataPipeline.dataframes.stats import StatsRepository
from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import (
    FeatureEngineering,
)
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import (
    fair_probabilities,
)
from ufc_betting.Models.RiskManagement.open_model_close1_input import (
    _align_model_design,
)


OUTPUT_FEATURES = "all_merged_stats_odds.csv"
OUTPUT_PREDICTIONS = "open_model_close1_close2_results.csv"
PREDICTION_STAGES = ("close1", "close2")


def _normalized_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="raise").dt.normalize()


def _target_dates(
    non_merged_stats: pd.DataFrame,
    non_merged_odds: pd.DataFrame,
) -> pd.DatetimeIndex:
    stats_dates = pd.Index(_normalized_dates(non_merged_stats, "event_date").unique())
    odds_dates = pd.Index(_normalized_dates(non_merged_odds, "event_date").unique())
    dates = stats_dates.intersection(odds_dates).sort_values()
    if dates.empty:
        raise ValueError("Non-merged stats and odds have no event dates in common")
    return pd.DatetimeIndex(dates)


def build_feature_matrix() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    stats_repo = StatsRepository(
        stats_history_file=settings.stats_history_file,
        non_merged_stats_file=settings.non_merged_stats_file,
    )
    odds_repo = OddsRepository(
        odds_history_file=settings.odds_history_file,
        non_merged_odds_file=settings.non_merged_odds_file,
    )

    non_merged_stats = stats_repo.load_non_merged()
    non_merged_odds = odds_repo.load_non_merged()
    dates = _target_dates(non_merged_stats, non_merged_odds)
    combined_stats = stats_repo.load_combined()
    combined_odds = odds_repo.load_combined()

    feature_engineering = FeatureEngineering()
    all_features, _ = feature_engineering.build_all_stats(
        combined_stats.copy(),
        pd.DataFrame(),
        combined_odds.copy(),
        pd.DataFrame(),
        ignore_upcoming=True,
    )
    target_features = all_features.copy()
    target_features["date"] = _normalized_dates(target_features, "date")
    target_features = target_features.loc[
        target_features["date"].isin(dates)
        & pd.to_numeric(target_features["winner"], errors="coerce").lt(2)
    ].copy()
    target_features["winner"] = target_features["winner"].astype(int)

    output_dates = pd.DatetimeIndex(target_features["date"].unique())
    if not output_dates.difference(dates).empty:
        raise ValueError("Feature output contains dates outside the non-merged sources")
    return target_features.sort_values(["date", "fighter_red", "fighter_blue"]), dates


def _prediction_se(model, design: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    active = np.asarray(model.params) != 0
    covariance = np.asarray(model.cov_params())[np.ix_(active, active)]
    if not np.isfinite(covariance).all():
        raise ValueError("Opening model has nonfinite active coefficient covariance")
    active_design = design.to_numpy()[:, active]
    variance = np.einsum("ij,jk,ik->i", active_design, covariance, active_design)
    if (variance < -1e-8).any():
        raise ValueError("Opening model produced invalid negative prediction variance")
    return probability * (1 - probability) * np.sqrt(np.maximum(variance, 0))


def score_opening_model(
    features: pd.DataFrame,
    stage: str,
    model,
    scaler,
) -> pd.DataFrame:
    if stage not in PREDICTION_STAGES:
        raise ValueError(f"Unsupported odds stage: {stage!r}")
    price_cols = [f"dec_{stage}_red", f"dec_{stage}_blue"]
    missing = sorted(set(price_cols) - set(features.columns))
    if missing:
        raise KeyError(f"Feature matrix is missing price columns: {missing}")

    result = features.dropna(subset=price_cols).copy().reset_index(drop=True)
    prices = result[price_cols].to_numpy(dtype=float)
    fair = fair_probabilities(prices)
    # The opening model keeps its training schema; only its market feature is
    # replaced with the selected close-stage fair-probability difference.
    result["proba_fair_open_diff"] = fair[:, 0] - fair[:, 1]

    numeric = list(scaler.feature_names_in_)
    missing = sorted(set(numeric) - set(result.columns))
    if missing:
        raise KeyError(f"Generated feature matrix is missing model inputs: {missing}")
    complete = result[numeric].notna().all(axis=1)
    result = result.loc[complete].copy()
    if result.empty:
        raise ValueError(f"No complete opening-model rows remain for {stage}")

    scaled = pd.DataFrame(
        scaler.transform(result[numeric]),
        columns=numeric,
        index=result.index,
    )
    names = list(model.model.exog_names)
    design = _align_model_design(scaled, result, names)
    probability = np.asarray(model.predict(design))
    if not np.isfinite(probability).all():
        raise ValueError(f"Opening model produced nonfinite {stage} probabilities")

    result["proba_red"] = probability
    result["proba_blue"] = 1 - probability
    result["pred_winner"] = (probability >= 0.5).astype(int)
    result["proba_se"] = _prediction_se(model, design, probability)
    result["correct_pred"] = result["pred_winner"].eq(result["winner"]).astype(int)
    result["prob_winner"] = np.maximum(probability, 1 - probability)
    result[f"dec_fair_{stage}_red"] = 1 / fair[result.index, 0]
    result[f"dec_fair_{stage}_blue"] = 1 / fair[result.index, 1]
    result["odds_type"] = stage
    result["prediction_source"] = f"saved_open_model_at_{stage}_prices"
    return result.reset_index(drop=True)


def run(output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = settings.non_merged_features_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features, target_dates = build_feature_matrix()
    model = sm.load(config.model_open_path)
    scaler = joblib.load(config.scaler_open_path)
    predictions = pd.concat(
        [score_opening_model(features, stage, model, scaler) for stage in PREDICTION_STAGES],
        ignore_index=True,
    )
    prediction_dates = pd.DatetimeIndex(predictions["date"].unique())
    if not prediction_dates.difference(target_dates).empty:
        raise ValueError("Prediction output contains dates outside the non-merged sources")

    features_path = output_dir / OUTPUT_FEATURES
    predictions_path = output_dir / OUTPUT_PREDICTIONS
    features.to_csv(features_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    print(f"Wrote {len(features):,} feature rows to {features_path}")
    print(f"Wrote {len(predictions):,} prediction rows to {predictions_path}")
    return features_path, predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.non_merged_features_dir,
        help="Directory for the merged feature and prediction CSV files",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
