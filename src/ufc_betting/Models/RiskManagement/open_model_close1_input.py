"""Prepare close1-priced test inputs using the fixed opening Logit model."""
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import fair_probabilities


def _align_model_design(scaled, features, model_names):
    """Align scaler output with the fitted Statsmodels design schema."""
    design = scaled.copy()
    aliases = {
        # Preprocessing retains the raw source name, while the fitted model
        # retains the one-hot encoded feature name.
        "elo_pred_1": "elo_pred",
        "elo_pred1": "elo_pred",
    }
    for name in model_names:
        if name in design:
            continue
        if name == "const":
            design[name] = 1.0
        elif aliases.get(name) in design:
            design[name] = design[aliases[name]]
        elif aliases.get(name) in features:
            # TrainTestBuilder leaves categorical binary columns out of the
            # numeric scaler. Recreate the fitted one-hot column from the raw
            # 0/1 source value in that case.
            design[name] = pd.to_numeric(
                features[aliases[name]], errors="raise"
            ).to_numpy()
        elif name in features:
            design[name] = features[name].to_numpy()
        else:
            raise KeyError(
                f"Model feature {name!r} is absent from both scaler output "
                "and the input frame"
            )
    return design[list(model_names)]


def opening_model_at_close1(frame=None, prices=None, model=None, scaler=None):
    """Optionally evaluate supplied interpolated prices without reloading weights."""
    frame = (pd.read_csv(settings.data_dir / "model_results" / "test_logit_close1.csv")
             if frame is None else frame.copy().reset_index(drop=True))
    model = sm.load(config.model_open_path) if model is None else model
    scaler = joblib.load(config.scaler_open_path) if scaler is None else scaler
    prices = frame[["dec_close1_red", "dec_close1_blue"]].to_numpy() if prices is None else np.asarray(prices)
    frame[["dec_close1_red", "dec_close1_blue"]] = prices
    fair = fair_probabilities(prices)
    features = frame.copy()
    # The opening model's current-market feature must now reflect close1 prices.
    features["proba_fair_open_diff"] = fair[:, 0] - fair[:, 1]
    numeric = list(scaler.feature_names_in_)
    design = pd.DataFrame(
        scaler.transform(features[numeric]),
        columns=numeric,
        index=features.index,
    )
    names = list(model.model.exog_names)
    design = _align_model_design(design, features, names)
    probability = np.asarray(model.predict(design))
    # Delta-method SE for the fixed fitted model's probability prediction.
    # Regularization may mark trimmed coefficients' covariance as NaN; omit
    # those exact-zero coefficients, never silently fill active covariance.
    active = np.asarray(model.params) != 0
    covariance = np.asarray(model.cov_params())[np.ix_(active, active)]
    if not np.isfinite(covariance).all():
        raise ValueError("Opening model has nonfinite active coefficient covariance")
    x = design.to_numpy()[:, active]
    variance = np.einsum("ij,jk,ik->i", x, covariance, x)
    if (variance < -1e-8).any():
        raise ValueError("Invalid negative prediction variance")
    se = probability * (1 - probability) * np.sqrt(np.maximum(variance, 0))
    frame["proba_red"] = probability
    frame["proba_blue"] = 1 - probability
    frame["pred_winner"] = (probability >= .5).astype(int)
    frame["proba_se"] = se
    frame["correct_pred"] = frame.pred_winner.eq(frame.winner).astype(int)
    frame["prob_winner"] = np.maximum(probability, 1 - probability)
    frame["dec_fair_close1_red"] = 1 / fair[:, 0]
    frame["dec_fair_close1_blue"] = 1 / fair[:, 1]
    frame["prediction_source"] = "saved_open_model_at_close1_prices"
    return frame
