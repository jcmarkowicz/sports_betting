
import numpy as np 
import pandas as pd 
import statsmodels.api as sm


def logit_predict(
        model, 
        df, 
        y_hat, 
        feats, 
        num_feats, 
        cat_feats, 
        valid_mask, 
        scaler, 
        required_df_idx
):

    # scale numeric features
    # use transform for scaler to avoid data leakage 
    scaled_num = pd.DataFrame(
        scaler.transform(df.loc[valid_mask, num_feats]), # ~nan rows, numerical columns 
        columns=num_feats,
        index=required_df_idx[valid_mask]
    ) # len valid mask = len X_valid

    # keep categorical features unchanged
    cat_data = df.loc[valid_mask, cat_feats]

    # combine them
    scaled_valid = pd.concat([scaled_num, cat_data], axis=1)

    # keep original column order
    X_valid = scaled_valid[feats]
    X_valid = sm.add_constant(X_valid, has_constant='add')

    constant_like_cols = [
        col for col in X_valid.columns
        if X_valid[col].nunique(dropna=False) == 1
    ]
    print("Constant-like columns:", constant_like_cols)


    # ensure no duplicate 'const' column
    assert X_valid.columns.duplicated().sum() == 0, "Duplicate columns found in X_valid"
    train_cols = model.model.exog_names            

    # test if missing or extra columns 
    missing = set(train_cols) - set(X_valid.columns)
    extra = set(X_valid.columns) - set(train_cols)
    if missing or extra:
        raise ValueError(f"Column mismatch — missing: {missing}, extra: {extra}")

    # test for nans 
    X_valid = X_valid.reindex(columns=train_cols)
    if X_valid.isna().any().any():
        raise ValueError("NaNs present after alignment")

    # predict
    proba_red = model.predict(X_valid)
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
