"""Predictions and standard errors for prepared Statsmodels model inputs."""

import numpy as np
import pandas as pd


def predict_logit(features, model, model_se=None):
    """Return probabilities, classes, and optional probability standard errors.

    features must already be renamed, scaled, and encoded by TrainTestBuilder.
    Each fitted model selects its own design columns; the input index is retained.
    """
    design = features.copy()
    design['const'] = 1.0
    design = design.loc[:, model.model.exog_names]
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError('Model features must be finite')
    probability = pd.Series(np.asarray(model.predict(design)), index=features.index)
    result = pd.DataFrame({'proba_red': probability,
                           'pred_winner': (probability >= .5).astype(int)})
    if model_se is not None:
        summary = model_se.get_prediction(
            design.loc[:, model_se.model.exog_names]
        ).summary_frame()
        result['proba_se'] = np.asarray(summary['se'])
    return result
