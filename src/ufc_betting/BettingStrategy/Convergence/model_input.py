"""Predict probabilities and standard errors from supplied model inputs."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ufc_betting.BettingStrategy.model_statistics import (
    align_model_columns, model_covariance_and_variance, probability_standard_error,
)

MODEL_COLUMN_ALIASES = {'elo_pred_1': 'elo_pred', 'elo_pred1': 'elo_pred'}


def predict_probabilities(frame, model_path, *, scaler_path=None):
    """Return a copy with probabilities, predicted side, and prediction SE.

    predictions = predict_probabilities(data, 'model.pkl', scaler_path='scaler.pkl')

    Supply the model's feature columns. Omit scaler_path for already prepared
    features. Paths or loaded Statsmodels model/scaler objects are accepted so
    repeated scenario predictions can reuse the same fitted objects.
    """
    frame = frame.copy()
    model = sm.load(model_path) if isinstance(model_path, (str, Path)) else model_path
    scaler = joblib.load(scaler_path) if isinstance(scaler_path, (str, Path)) else scaler_path
    design = frame.copy()
    if scaler is not None:
        numeric = list(scaler.feature_names_in_)
        design = pd.DataFrame(scaler.transform(frame[numeric]), columns=numeric, index=frame.index)
    names = list(model.model.exog_names)
    design = align_model_columns(design, names, fallback=frame, aliases=MODEL_COLUMN_ALIASES)
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError('Model features must be finite')
    probability = np.asarray(model.predict(design))
    if probability.shape != (len(frame),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Model must return one finite probability per row')
    _, variance = model_covariance_and_variance(model, design)
    se = probability_standard_error(probability, variance)
    frame["proba_red"] = probability
    frame["proba_blue"] = 1 - probability
    frame["pred_winner"] = (probability >= .5).astype(int)
    frame["proba_se"] = se
    # Keep shared-calculator inputs aligned with these freshly evaluated prices.
    frame["prediction_se"] = se
    frame["risk_red_proba"] = probability
    if 'winner' in frame:
        frame["correct_pred"] = frame.pred_winner.eq(frame.winner).astype(int)
    frame["prob_winner"] = np.maximum(probability, 1 - probability)
    frame["prediction_source"] = "supplied_model"
    return frame
