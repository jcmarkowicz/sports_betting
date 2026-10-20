"""Regularized probability models used by the Brier walk-forward search."""

import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder


class BuilderPreprocessor:
    """Use TrainTestBuilder's fitted transforms in walk-forward folds."""

    def fit_transform(self, X):
        categorical = list(X.select_dtypes(include=['category', 'object', 'bool', 'string']).columns)
        numeric = [name for name in X if name not in categorical]
        prepared = TrainTestBuilder.scale_encode(X, X, categorical, numeric)
        self.scaler = prepared['scaler']
        self.encoder = prepared['encoder']
        return prepared['X_train']

    def transform(self, X):
        return TrainTestBuilder.prepare_inference_data(X, self.scaler, self.encoder)


def fit_regularized_probability(X, y, *, model_type='logit', alpha=0.1,
                                L1_wt=0.5, maxiter=1000):
    """Fit penalized coefficients directly, including the intercept penalty.

    Logit uses a summed likelihood; GLM elastic net uses mean likelihood.
    Their alpha values therefore should not be compared on the same scale.
    """
    if model_type not in ('logit', 'glm'):
        raise ValueError("model_type must be 'logit' or 'glm'")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite')
    if not np.isfinite(L1_wt) or not 0 <= L1_wt <= 1:
        raise ValueError('L1_wt must lie between zero and one')
    design = sm.add_constant(np.asarray(X, dtype=float), has_constant='add')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if model_type == 'logit':
            result = sm.Logit(np.asarray(y), design).fit_regularized(
                method='l1', alpha=alpha, maxiter=maxiter, disp=False)
        else:
            result = sm.GLM(np.asarray(y), design, family=sm.families.Binomial()).fit_regularized(
                method='elastic_net', alpha=alpha, L1_wt=L1_wt, maxiter=maxiter)
    if (not getattr(result, 'mle_retvals', {}).get('converged', True)
            or not getattr(result, 'converged', True)
            or not np.isfinite(result.params).all()):
        raise ValueError('Regularized model did not converge to finite coefficients')
    return result


def predict_regularized_probability(model, X):
    probabilities = np.asarray(model.predict(
        sm.add_constant(np.asarray(X, dtype=float), has_constant='add')))
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Model returned invalid probabilities')
    return probabilities
