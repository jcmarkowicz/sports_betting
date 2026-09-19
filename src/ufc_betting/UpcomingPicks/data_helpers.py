import numpy as np 
import pandas as pd 

from ufc_betting.BettingStrategy.kelly_scaling import expected_value, kelly_edge
from ufc_betting.UpcomingPicks.set_column_names import set_ml_bets_cols, set_parlay_cols, get_ml_bet_cols, get_parlay_cols

def merge_bets_types(df_bets, df_bets_combined):
    """ matching rows by their index and adding the columns  """
    df_bets_combined = df_bets_combined.merge(
                    df_bets,
                    left_index=True,
                    right_index=True,
                    how="left"  
                )
    return df_bets_combined

def merge_parlay_types(df_parlay, df_parlay_combined, odds_type):
    
    # Store the original index as a column
    df_parlay[f"fight_index_{odds_type}"] = df_parlay.index.to_numpy()
    col = f"fight_index_{odds_type}"

    # reset for merging, index doesnt matter now 
    df_parlay = df_parlay.copy().reset_index(drop=True)

    if df_parlay_combined.empty:
        df_parlay_combined = df_parlay
    else:
        df_parlay_combined = df_parlay_combined.merge(
            df_parlay,
            left_index=True,
            right_index=True,
            how="left",
        )
    return df_parlay_combined

def get_bets_input(
        df, 
        y_hat,
        proba_red, 
        proba_blue, 
        real_odds, 
        fair_odds
    ):
    """ 
    take model results and generate a dataframe thats ready for kelly scaling,
    aligns by yhat index which comes from required_df_idx, which is the same as the df index.
    """
        
    fighter_red = df["fighter_red"].to_numpy()
    fighter_blue = df["fighter_blue"].to_numpy()

    choice_fair_odds = pd.Series(
        np.where(y_hat == 1, df[fair_odds[1]],
                np.where(y_hat == 0, df[fair_odds[0]], np.nan)),
        index=y_hat.index
    )
    
    choice_real_odds = pd.Series(
        np.where(y_hat == 1, df[real_odds[1]],
                np.where(y_hat == 0, df[real_odds[0]], np.nan)),
        index=y_hat.index
    )

    choice_proba = pd.Series(
            np.where(y_hat == 1, proba_red, proba_blue),
            index=y_hat.index,
            name='choice_proba'
        )
    
    assert real_odds[1].split('_')[-1] == 'red', 'red/blue odds column order error'

    unweighted_fstar = pd.Series(
        [
            kelly_edge(p, fo) if not (pd.isna(p) or pd.isna(fo)) else np.nan
            for p, fo in zip(choice_proba, choice_fair_odds)
        ],
        index=y_hat.index
    )

    choice_ev = pd.Series(
        np.where(
            choice_proba.isna() | choice_real_odds.isna(),
            np.nan,
            expected_value(choice_proba, choice_real_odds)
        ),
        index=y_hat.index
    )

    choice_edge = choice_proba - (1 / choice_fair_odds)

    pred_winner_names = pd.Series(
        np.where(y_hat == 1, fighter_red,
            np.where(
                y_hat == 0, 
                fighter_blue, 
                None
            )),
    index=y_hat.index)

    bets_input_df = pd.DataFrame({
        "p": choice_proba,
        'choice_proba':choice_proba, 
        "fair_odds": choice_fair_odds,
        "real_odds": choice_real_odds,
        "choice_real_odds":choice_real_odds,
        "ev": choice_ev,
        "choice_ev": choice_ev,
        "f_star_unscaled": unweighted_fstar,
        'pred_winner_names':pred_winner_names,
        'choice_edge': choice_edge,
        'pred_winner_bool':y_hat, 
    }, index=y_hat.index) # y_hat index is the same as required_idx 

    return bets_input_df

def get_parlay_input(bets_input_df, fighter_red, fighter_blue, dates, required_idx):
    """ all cols should be aligned by required_idx which comes from df """

    parlay_input_df = pd.DataFrame({
        "choice_ev": bets_input_df['choice_ev'],
        "choice_proba": bets_input_df['choice_proba'],
        "choice_real_odds": bets_input_df['choice_real_odds'],
        'choice_fighter_name':bets_input_df['pred_winner_names'],
        'choice_fighter_bool':bets_input_df['pred_winner_bool'],
        "fighter_red": fighter_red,
        "fighter_blue": fighter_blue,
        "date": dates,
    }, index=required_idx)

    return parlay_input_df.dropna()

def get_parlay_pkt(df_parlay, type, all_na=False):
    parlay_pkt = {
        'choice_fighter_name_col': df_parlay[f'choice_fighter_name_{type}'],
        'choice_fighter_bool_col': df_parlay[f'choice_fighter_bool_{type}'],
        'parlay_fstar_col': df_parlay[f'parlay_fstar_{type}'],
        'parlay_odds_col': df_parlay[f'parlay_odds_{type}'], 
        'stake_col': df_parlay[f'stake_{type}'], 
        'parlay_ev_col': df_parlay[f'parlay_ev_{type}'], 
        'parlay_prob_col': df_parlay[f'parlay_prob_{type}']
    }

    df = set_parlay_cols(
        type_=type,
        pkt=parlay_pkt,
        required_idx=df_parlay.index,
        all_na=all_na
    )

    return df

def get_bets_pkt(bets_input_df, df_per_bet, type_, required_idx, all_na) -> dict:
    """ take columns from bets_input and df_per_bet and rename them """
    bets_pkt = {
            'pred_name_col': bets_input_df['pred_winner_names'], 
            'pred_winner_col': bets_input_df['pred_winner_bool'], 
            'choice_proba_col': bets_input_df['choice_proba'], 
            'choice_fstar_col': df_per_bet['fstar_scaled'], 
            'choice_stake_col': df_per_bet['stake'],
            'edge_col': bets_input_df['choice_edge'], 
            'ev_col': bets_input_df['choice_ev'],
        }

    df = set_ml_bets_cols(
        type_=type_, 
        pkt=bets_pkt, 
        required_idx=required_idx, 
        all_na=all_na
    )
    return df

def get_X_stacked(df, df_proba, df_bets_combined, required_df_idx):

    X_stacked = pd.DataFrame({
        'proba_diff_0': df_proba['proba_red_open'] - df_proba['proba_blue_open'],
        'proba_diff_1': df_proba['proba_red_close1'] - df_proba['proba_blue_close1'],
        'proba_diff_2': df_proba['proba_red_close2'] - df_proba['proba_blue_close2'],
    }, index=required_df_idx)
    X_stacked['total_fights_pred_open'] = np.where(df_bets_combined['pred_winner_open'] == 1, df['obs_red'], df['obs_blue']) 

    return X_stacked