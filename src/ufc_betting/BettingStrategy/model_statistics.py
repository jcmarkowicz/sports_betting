"""Shared model-column alignment and Logit prediction uncertainty."""
import numpy as np
import pandas as pd


def align_model_columns(data, columns, *, fallback=None, aliases=None):
    """Copy/reorder model features, resolving explicit aliases and an intercept.

    Scaled data takes priority over fallback raw features. Rows must align.
    aliases maps model column names to their input column names.
    """
    columns = list(columns)
    if fallback is not None and fallback.empty:
        fallback = None
    if data.columns.has_duplicates or len(set(columns)) != len(columns):
        raise ValueError('Model and input column names must be unique')
    if fallback is not None and (not data.index.equals(fallback.index) or fallback.columns.has_duplicates):
        raise ValueError('Fallback features must have matching rows and unique columns')
    aliases = aliases or {}
    design = data.copy()
    for name in columns:
        source = aliases.get(name, name)
        if name in design:
            continue
        if name == 'const':
            design[name] = 1.
        elif source in design:
            design[name] = design[source]
        elif fallback is not None and source in fallback:
            design[name] = pd.to_numeric(fallback[source], errors='raise')
        elif fallback is not None and name in fallback:
            design[name] = pd.to_numeric(fallback[name], errors='raise')
        else:
            raise KeyError(f'Model feature {name!r} is absent from the supplied data')
    return design.loc[:, columns].copy()


def model_covariance_and_variance(model, design):
    """Return coefficient covariance and per-row linear-predictor variance.

    design contains the fitted model's features. Variance is x @ cov @ x.T
    per row, before conversion to probability-scale standard errors.
    Missing covariance for regularization-trimmed zero coefficients is omitted;
    finite covariance for ordinary zero-valued coefficients is retained.
    """
    names = list(model.model.exog_names)
    if isinstance(design, pd.DataFrame):
        design = align_model_columns(design, names)
    x = np.asarray(design, dtype=float)
    params = model.params
    if isinstance(params, pd.Series):
        params = params.loc[names]
    params = np.asarray(params, dtype=float)
    covariance = model.cov_params()
    if isinstance(covariance, pd.DataFrame):
        covariance = covariance.loc[names, names]
    covariance = np.asarray(covariance, dtype=float).copy()
    if (params.shape != (len(names),) or covariance.shape != (len(names), len(names))
            or x.ndim != 2 or x.shape[1] != len(names)):
        raise ValueError('Model coefficients, covariance, and design dimensions must match')
    if not np.isfinite(x).all() or not np.isfinite(params).all():
        raise ValueError('Model features and coefficients must be finite')
    # Some regularized Statsmodels fits leave NaN covariance for zeroed terms.
    omitted = (params == 0) & ~np.isfinite(np.diag(covariance))
    trimmed = (getattr(model, 'mle_retvals', None) or {}).get('trimmed')
    if trimmed is not None:
        trimmed = np.asarray(trimmed, dtype=bool)
        if trimmed.shape != params.shape:
            raise ValueError('Trimmed coefficient mask must match model coefficients')
        omitted |= trimmed & (params == 0)
    covariance[omitted, :] = 0.
    covariance[:, omitted] = 0.
    if not np.isfinite(covariance).all():
        raise ValueError('Model has nonfinite active coefficient covariance')
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12):
        raise ValueError('Model covariance must be symmetric')
    variance = np.einsum('ij,jk,ik->i', x, covariance, x)
    if not np.isfinite(variance).all() or (variance < -1e-8).any():
        raise ValueError('Invalid prediction variance')
    return covariance, np.maximum(variance, 0.)


def probability_standard_error(probability, variance):
    """Logit delta-method SE: p * (1 - p) * sqrt(linear-predictor variance)."""
    probability = np.asarray(probability, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if probability.shape != variance.shape:
        raise ValueError('Probabilities and variances must have matching shapes')
    if (not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()
            or not np.isfinite(variance).all() or (variance < 0).any()):
        raise ValueError('Need probabilities in [0, 1] and finite nonnegative variances')
    return probability * (1 - probability) * np.sqrt(variance)
