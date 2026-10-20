
import numpy as np 
import pandas as pd 
import statsmodels.api as sm
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.inference import predict_logit
from ufc_betting.BettingStrategy.model_statistics import (
    align_model_columns, model_covariance_and_variance, probability_standard_error,
)

def logit_predict(
        model, 
        df, 
        y_hat, 
        feats:list[str], 
        num_feats:list[str], 
        cat_feats:list[str], 
        valid_mask, 
        scaler,
        cat_encoder, 
        required_df_idx,
        rename_odds=None,
        model_se=None,
        return_se=False,
):

    if rename_odds:
        feats_split = [f.split('_') for f in feats]
        assert any(
            'close1' in parts or 'close2' in parts
            for parts in feats_split
        ), "rename_odds should be used for close1 or close2 features only"

        # The opening column already exists; renaming creates duplicates.
        df = df.copy()
        df['proba_fair_open_diff'] = df[f'proba_fair_{rename_odds}_diff']
        feats = [
            'proba_fair_open_diff' if f == f'proba_fair_{rename_odds}_diff' else f
            for f in feats
        ]
        num_feats = [
            'proba_fair_open_diff' if f == f'proba_fair_{rename_odds}_diff' else f
            for f in num_feats
        ]

    # scale numeric features
    # use transform for scaler to avoid data leakage 
    scaled_num = pd.DataFrame(
        scaler.transform(df.loc[valid_mask, num_feats]), # ~nan rows, numerical columns 
        columns=num_feats,
        index=required_df_idx[valid_mask]
    ) # len valid mask = len X_valid

    # handle categorical features that come from TrainTestBuilder
    encoded = pd.DataFrame(
        cat_encoder.transform(df.loc[valid_mask, cat_feats]),
        columns=cat_encoder.get_feature_names_out(cat_feats),
        index=required_df_idx[valid_mask],
    )
    # combine them
    scaled_valid = pd.concat([scaled_num, encoded], axis=1)

    # alwaysadd constant , check below that its not duplicated, other cat cols might trigger as constant 
    X_valid = sm.add_constant(scaled_valid, has_constant='add')

    # constant_like_cols = [
    #     col for col in X_valid.columns
    #     if X_valid[col].nunique(dropna=False) == 1
    # ]
    # print("Constant-like columns:", constant_like_cols)

    # ensure no duplicate 'const' column
    assert X_valid.columns.duplicated().sum() == 0, "Duplicate columns found in X_valid"
    train_cols = model.model.exog_names
    if return_se:
        X_valid = align_model_columns(
            X_valid, train_cols, fallback=df.loc[valid_mask],
            aliases={'elo_pred_1': 'elo_pred', 'elo_pred1': 'elo_pred'},
        )
        predictions = predict_logit(X_valid, model, model_se)
        if model_se is None:
            _, variance = model_covariance_and_variance(model, X_valid)
            predictions['proba_se'] = probability_standard_error(predictions.proba_red, variance)
        predictions = predictions.reindex(required_df_idx)
        return (predictions.pred_winner, predictions.proba_red,
                1 - predictions.proba_red, predictions.proba_se)
    

    # test if missing or extra columns 
    missing = set(train_cols) - set(X_valid.columns)
    extra = set(X_valid.columns) - set(train_cols)
    if missing or extra:
        raise ValueError(f"Column mismatch — missing: {missing}, extra: {extra}")

    # test for nans 
    X_valid = X_valid[train_cols]        
    if X_valid.isna().any().any():
        raise ValueError("NaNs present after alignment")

    # predict
    proba_red = pd.Series(np.nan, index=required_df_idx)
    proba_red.loc[valid_mask] = model.predict(X_valid)
    proba_blue = 1 - proba_red

    y_hat.loc[valid_mask] = (proba_red >= 0.5).astype(int) 

    # nan values for missing fights 
    y_hat.loc[~valid_mask] = np.nan

    return y_hat, proba_red, proba_blue


def xgboost_predict(
        model, 
        X_stacked, 
        required_df_idx
):

    valid_mask = ~X_stacked.isna().any(axis=1)

    X_valid = X_stacked.loc[valid_mask]
    X_valid = X_valid.reindex(columns=model.get_booster().feature_names)

    y_hat = pd.Series(np.nan, index=required_df_idx)
    proba_red = pd.Series(np.nan, index=required_df_idx)
    proba_blue = pd.Series(np.nan, index=required_df_idx)
        
    if len(X_valid) > 0: 
        y_hat.loc[valid_mask] = model.predict(X_valid)
        model_proba = model.predict_proba(X_valid)
        proba_blue.loc[valid_mask] = model_proba[:, 0]
        proba_red.loc[valid_mask] = model_proba[:, 1]

    result = {
        'y_hat': y_hat,
        'proba_red': proba_red,
        'proba_blue': proba_blue
    }

    return result
