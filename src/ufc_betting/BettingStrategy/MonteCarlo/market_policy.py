"""Fresh model inference and price-based selection of closing Kelly controls."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.BettingStrategy.model_statistics import (
    align_model_columns, model_covariance_and_variance, probability_standard_error,
)
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.inference import predict_logit
from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds


def load_predictor(model, model_type, scalers, encoders, model_se=None):
    """Load the selected model with its own fitted preprocessing, once per run."""
    if model_type not in ('open', 'close1', 'close2'):
        raise ValueError('prediction_model_type must be open, close1, or close2')
    if scalers is None or encoders is None or model_type not in scalers or model_type not in encoders:
        raise ValueError('Supply prediction_scalers and prediction_encoders for the selected model type')
    def load(value, loader):
        return loader(value) if isinstance(value, (str, Path)) else value
    return (load(model, sm.load), load(scalers[model_type], joblib.load),
            load(encoders[model_type], joblib.load), load(model_se, sm.load))


def predict_sampled_prices(frame, prices, predictor, model_type):
    """Update the selected model's market inputs while preserving original prices."""
    model, scaler, encoder, model_se = predictor
    data = replace_opening_odds(frame, model_type, prices=prices, features=[])
    inputs = TrainTestBuilder.prepare_inference_data(data, scaler, encoder)
    inputs = align_model_columns(inputs, model.model.exog_names, fallback=data,
                                 aliases={'elo_pred_1': 'elo_pred', 'elo_pred1': 'elo_pred'})
    predictions = predict_logit(inputs, model, model_se)
    if model_se is None:
        _, variance = model_covariance_and_variance(model, inputs)
        predictions['proba_se'] = probability_standard_error(predictions.proba_red, variance)
    # Common execution-price names used by the existing settlement calculator.
    result = replace_opening_odds(frame, 'close1', prices=prices, features=[])
    for column in predictions:
        result[column] = predictions[column].to_numpy()
    result['proba_blue'] = 1 - result.proba_red
    result['prediction_model_type'] = model_type
    return result


# Compatibility names; all policy sizing, decay, and settlement live in the calculator.
nearest_market = Calculator.nearest_market
settle_market_event = Calculator.calculate_market_event_returns
