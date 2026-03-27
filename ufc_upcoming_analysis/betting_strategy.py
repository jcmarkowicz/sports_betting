
import numpy as np
import pandas as pd
import statsmodels.api as sm

import os 
import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from betting_algos import parlay_top_ev, run_per_bet_scaling, expected_value, kelly_edge
from set_column_names import set_ml_bets_cols, set_parlay_cols, get_ml_bet_cols, get_parlay_cols

PARLAY_SIZE = 2


def model_predict(model, df, y_hat, feats, num_feats, cat_feats, valid_mask, scaler, required_df_idx):

    # scale numeric features
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
    
    # check for single-row case
    if len(X_valid) == 1:
        X_valid = sm.add_constant(X_valid, has_constant='add')
    else:
        X_valid = sm.add_constant(X_valid)

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
    y_hat.loc[valid_mask] = model.predict(X_valid)

    # nan values for missing fights 
    y_hat.loc[~valid_mask] = np.nan

    return y_hat


def df_bets_tests(df_bets, df_bets_combined, valid_mask, choice_ev, fstar_list):

    non_nan_mask = df_bets.notna().all(axis=1)
    n_non_nan = non_nan_mask.sum()
    n_valid = valid_mask.sum()
    assert n_non_nan == n_valid, (
        f"Mismatch: non-NaN rows ({n_non_nan}) != valid rows ({n_valid})"
    )

    assert np.array_equal(~np.isnan(fstar_list), ~np.isnan(choice_ev)), f'error with fstar and choice_ev, {print(choice_ev)}, {print(fstar_list)}'

    assert df_bets.shape[0] == df_bets_combined.shape[0], 'mismatch shapes df bets and df bets combined '

def df_parlay_tests(df_parlay, choice_ev):
    
    if np.count_nonzero(~np.isnan(choice_ev)) >= 2:
        assert not df_parlay.isna().any().any(), 'Parlay Bets Error'

    assert df_parlay.shape[0] == PARLAY_SIZE, 'Parlay df size error '

def merge_bets_types(df_bets, df_bets_combined):
    df_bets_combined = df_bets_combined.merge(
                    df_bets,
                    left_index=True,
                    right_index=True,
                    how="left"  
                )
    return df_bets_combined

def merge_parlay_types(df_parlay, df_parlay_combined):
    if df_parlay_combined.empty:
        df_parlay_combined = df_parlay.copy()
    else:
        df_parlay_combined = df_parlay_combined.merge(
            df_parlay,
            left_index=True,
            right_index=True,
            how="left"
        )
    return df_parlay_combined

def betting_pipeline(upcoming_df, feats_list, model_list, scaler_list, type_list, fair_odds_list, real_odds_list, bankroll, max_drawdown=0.15, N=1000):

    other_cols = ['date', 'fighter_red', 'fighter_blue', 'open_red', 'open_blue', 'close1_red', 'close2_red', 'close1_blue', 'close2_blue']
    df = upcoming_df.copy().reset_index(drop=True)
    required_df_idx = df.index

    df_bets_combined = pd.DataFrame(df[other_cols].values, columns=other_cols, index=required_df_idx)
    df_parlay_combined = pd.DataFrame()

    fighter_red = df["fighter_red"].values
    fighter_blue = df["fighter_blue"].values
    dates = df["date"].values

    for model, scaler, feats, type, fair_odds, real_odds in zip(model_list, scaler_list, feats_list, type_list, real_odds_list, fair_odds_list):

        valid_mask = ~df[feats].isna().any(axis=1)
        y_hat = pd.Series(0, index=required_df_idx, dtype=float)

        # split features by dtype
        num_feats = df[feats].select_dtypes(exclude='category').columns
        cat_feats = df[feats].select_dtypes(include='category').columns
        df_valid_num = df.loc[valid_mask, num_feats]

        if df_valid_num.shape[0] == 0:

            df_bets = set_ml_bets_cols(type, {}, required_df_idx, all_na=True)
            df_bets_combined = merge_bets_types(df_bets, df_bets_combined)

            df_parlay = set_parlay_cols(type, {}, np.arange(2), all_na=True)
            df_parlay_combined = merge_parlay_types(df_parlay, df_parlay_combined)
            continue

        y_hat = model_predict(model, df, y_hat, feats, num_feats, cat_feats, valid_mask, scaler, required_df_idx)

        proba_red = y_hat
        proba_blue = 1 - y_hat

        pred_winner_bool = (proba_red >= 0.5).astype('Int64') 

        pred_winner_names = pd.Series(np.where(pred_winner_bool == 1, fighter_red,
                                            np.where(pred_winner_bool == 0, fighter_blue, None)),
                                    index=y_hat.index)

        choice_proba = pd.Series(np.where(pred_winner_bool == 1, proba_red,
                                        np.where(pred_winner_bool == 0, proba_blue, np.nan)),
                                index=y_hat.index)

        
        assert real_odds[1].split('_')[-1] == 'red', 'red/blue odds column order error'

        choice_real_odds = pd.Series(
            np.where(pred_winner_bool == 1, df[real_odds[1]],
                    np.where(pred_winner_bool == 0, df[real_odds[0]], np.nan)),
            index=y_hat.index
        )

        choice_fair_odds = pd.Series(
            np.where(pred_winner_bool == 1, df[fair_odds[1]],
                    np.where(pred_winner_bool == 0, df[fair_odds[0]], np.nan)),
            index=y_hat.index
        )

        choice_edge = choice_proba - (1 / choice_fair_odds)

        # choice_ev as Series
        choice_ev = pd.Series(
            np.where(
                choice_proba.isna() | choice_real_odds.isna(),
                np.nan,
                expected_value(choice_proba, choice_real_odds)
            ),
            index=y_hat.index
        )

        # unweighted_fstar as Series
        unweighted_fstar = pd.Series(
            [
                kelly_edge(p, fo) if not (pd.isna(p) or pd.isna(fo)) else np.nan
                for p, fo in zip(choice_proba, choice_fair_odds)
            ],
            index=y_hat.index
        )

        # prepare data for mdd scaling 
        bets_input_df = pd.DataFrame({
                "p": choice_proba,
                "fair_odds": choice_fair_odds,
                "real_odds": choice_real_odds,
                "ev": choice_ev,
                "f_star_unscaled": unweighted_fstar
            }, index=y_hat.index)

        df_per_bet = run_per_bet_scaling(bets_input_df, max_drawdown, bankroll, N)
        fstar_list = df_per_bet['fstar_scaled'].values
        stake_list = df_per_bet['stake'].values

        bets_pkt = {f'pred_name_col': pred_winner_names, 
                    'pred_winner_col': pred_winner_bool, 
                    'choice_proba_col':choice_proba, 
                    'choice_fstar_col':fstar_list, 
                    'choice_stake_col':stake_list,
                    'edge_col':choice_edge, 
                    'ev_col':choice_ev}
        
        df_bets = set_ml_bets_cols(type, bets_pkt, required_idx=y_hat.index, all_na=False) 
        df_bets_tests(df_bets, df_bets_combined, valid_mask, choice_ev, fstar_list)
        df_bets_combined = merge_bets_types(df_bets, df_bets_combined)

        # no nans in parlay input df 
        parlay_input_df = pd.DataFrame({
            "choice_ev": choice_ev,
            "choice_proba": choice_proba,
            "choice_real_odds": choice_real_odds,
            "choice_fstar": fstar_list,
            'choice_fighter_name':pred_winner_names,
            "fighter_red": fighter_red,
            "fighter_blue": fighter_blue,
            "date": dates,
        }).dropna().reset_index(drop=True)

        df_parlay = parlay_top_ev(parlay_input_df, bankroll, type, top_n=[0,1])
        parlay_pkt = {'choice_fighter_name_col': df_parlay[f'choice_fighter_name_{type}'].values,
                    'parlay_fstar_col': df_parlay[f'parlay_fstar_{type}'].values,
                    'parlay_odds_col': df_parlay[f'parlay_odds_{type}'].values, 
                    'stake_col': df_parlay[f'stake_{type}'].values, 
                    'parlay_ev_col': df_parlay[f'parlay_ev_{type}'].values, 
                    'parlay_prob_col': df_parlay[f'parlay_prob_{type}'].values}
        
        df_parlay_final = set_parlay_cols(type, parlay_pkt, required_idx=np.arange(PARLAY_SIZE), all_na=False)
        df_parlay_tests(df_parlay_final, choice_ev)
        df_parlay_combined = merge_parlay_types(df_parlay_final, df_parlay_combined)


    return df_bets_combined, df_parlay_combined


def seperate_bets_dfs(df_bets, df_parlay, types):
    dfs = []
    dfs_parlay = []

    for type in types: 

        columns = get_ml_bet_cols(type).values()
        dfs.append(df_bets[columns])

        c_parlay = get_parlay_cols(type).values()
        dfs_parlay.append(df_parlay[c_parlay])

    return dfs, dfs_parlay
