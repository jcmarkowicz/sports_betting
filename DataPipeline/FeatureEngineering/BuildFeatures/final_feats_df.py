import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd 

from RatingAlgos.elo import elo_rating
from RatingAlgos.glicko import glicko_rating
from RatingAlgos.glicko2 import glicko2_run
from RatingAlgos.elo_scope import run_elo_on_matches

from DataPipeline.FeatureEngineering.BuildFeatures.feature_functions import total_bonus,\
      womens_fight, mma_math, win_lose_streak, months_since_last, method_wins, method_win_pct,\
      avg_fight_time, title_fights_stats_columns, method_losses, max_rating_won_against, opponent_avg_features, expected_value_stats

def compute_differences(df, feature, type):
    if type is not None: 
        df[f'{feature}_{type}_diff'] = df[f'{feature}_{type}_red'] - df[f'{feature}_{type}_blue']
    else: 
        df[f'{feature}_diff'] = df[f'{feature}_red'] - df[f'{feature}_blue']
    return df

def non_rolling_stats(df_):
    """
    Take in df with rolling features, compute non rolling features that dont depend on time series.
    See feature_functions.py for helper functions, Rating_Algos folder.
    Compute differences between red and blue fighter for relevant features.

    Args:
        df_ (pd.DataFrame): DataFrame with rolling features computed
    Returns:
        df (pd.DataFrame): DataFrame with non rolling features computed
    """

    df = df_.copy()
    feats = {}

    striking_features = ['kd', 'sig_str_landed', 'sig_str_absorbed', 'sig_str_attempted' , 'leg_str', 'head_str', 'body_str', 'clinch_str']
    grapling_features = ['td_landed', 'td_attempted', 'control', 'sub_att', 'reverse']
    striking_grapling_types = ['total', 'pm']
    fighter_attr = ['age', 'height', 'reach']

    for feature in striking_features + grapling_features: 
        for type in striking_grapling_types:
            df = compute_differences(df, feature, type)

    for attr in fighter_attr: 
        df = compute_differences(df, attr, None)

    striking_features = ['sig_str_defense', 'sig_str_accuracy']
    grapling_features = ['td_accuracy', 'td_defense']
    striking_grapling_types = ['pct']
    for feature in striking_features + grapling_features: 
        for type in striking_grapling_types:
            df = compute_differences(df, feature, type)

    for col in ['reach_red', 'reach_blue', 'age_red', 'age_blue', 'height_red', 'height_blue']:
        df[col] = df[col].fillna(
            df.groupby('weight_class')[col].transform('mean')
        )

    # opponents history
    red_opp_age, blue_opp_age = opponent_avg_features(df, 'age').T
    red_opp_reach, blue_opp_reach = opponent_avg_features(df, 'reach').T
    feats['opp_age_red'] = red_opp_age
    feats['opp_age_blue'] = blue_opp_age
    feats['opp_reach_red'] = red_opp_reach
    feats['opp_reach_blue'] = blue_opp_reach

    # efficiency, does fighter outperform average wc stats 
    # red_expected_td, blue_expected_td = expected_value_stats(df, 'td_attempted_pm', 'td_landed_pm').T
    # red_expected_str, blue_expected_str = expected_value_stats(df, 'sig_str_attempted_pm', 'sig_str_landed_pm').T

    # feats['td_pm_efficiency_red'] = red_expected_td
    # feats['td_pm_efficiency_blue'] = blue_expected_td

    # feats['sig_str_efficiency_red'] = red_expected_str
    # feats['sig_str_efficiency_blue'] = blue_expected_str

    # avg fight time in minutes
    avg_red, avg_blue = avg_fight_time(df).T
    feats['avg_fight_min_red'] = avg_red
    feats['avg_fight_min_blue'] = avg_blue
    feats['avg_fight_min_diff'] = avg_red - avg_blue

    # total bonus earned by fighter
    bon_red, bon_blue = total_bonus(df).T
    feats['total_bonus_red'] = bon_red
    feats['total_bonus_blue'] = bon_blue
    feats['total_bonus_diff'] = bon_red - bon_blue

    # Elo (choose ONE source of elo_red/elo_blue)
    params = {'base_k':55,'mov_mode':'linear', 'cutoff_rating':1800, 'cutoff_k_scale':1.0,
              'w90':None, 'regress_to_mean':.05, 'regress_every_n_matches':1000}

    df_elo = run_elo_on_matches(df, **params)
    feats['elo_red'] = df_elo['elo_pre_red'].to_numpy()
    feats['elo_blue'] = df_elo['elo_pre_blue'].to_numpy()
    feats['elo_red_proba'] = df_elo['elo_red_proba'].to_numpy()
    feats['elo_blue_proba'] = df_elo['elo_blue_proba'].to_numpy()
    feats['elo_diff'] = feats['elo_red'] - feats['elo_blue']
    feats['elo_pred'] = (feats['elo_red'] >= feats['elo_blue']).astype(int)

    # Glicko2
    df_glicko, _ = glicko2_run(df, initial_rating=1500.0, initial_rd=250.0,
                              initial_sigma=0.06, tau=.5, inactivity_scaling_days=365.0*2)

    feats['glicko_red'] = df_glicko['rating_red_pre'].to_numpy()
    feats['glicko_blue'] = df_glicko['rating_blue_pre'].to_numpy()
    feats['glicko_rd_red'] = df_glicko['sigma_red_pre'].to_numpy()
    feats['glicko_rd_blue'] = df_glicko['sigma_blue_pre'].to_numpy()
    feats['glicko_proba_red'] = df_glicko['p_red_pred'].to_numpy()
    feats['glicko_proba_blue'] = 1 - feats['glicko_proba_red']
    feats['glicko_diff'] = feats['glicko_red'] - feats['glicko_blue']
    feats['glicko_pred'] = (feats['glicko_red'] >= feats['glicko_blue']).astype(int)
    feats['rating_agree'] = (feats['glicko_pred'] == feats['elo_pred']).astype(int)

    # MMA math
    math_red, math_blue = mma_math(df).T
    feats['math_red'] = math_red
    feats['math_blue'] = math_blue

    # months since last fight
    m_red, m_blue = months_since_last(df).T
    feats['months_since_red'] = m_red
    feats['months_since_blue'] = m_blue
    feats['months_since_diff'] = m_red - m_blue

    # win/lose streaks etc (returns many arrays)
    (ws_r, ls_r, ws_b, ls_b,
     wp_r, wp_b,
     nf_r, nf_b,
     nw_r, nw_b,
     nl_r, nl_b) = win_lose_streak(df).T

    feats['win_streak_red'] = ws_r
    feats['lose_streak_red'] = ls_r
    feats['win_streak_blue'] = ws_b
    feats['lose_streak_blue'] = ls_b
    feats['win_pct_red'] = wp_r
    feats['win_pct_blue'] = wp_b
    feats['num_fights_red'] = nf_r
    feats['num_fights_blue'] = nf_b
    feats['num_wins_red'] = nw_r
    feats['num_wins_blue'] = nw_b
    feats['num_losses_red'] = nl_r
    feats['num_losses_blue'] = nl_b

    feats['num_fights_diff'] = nf_r - nf_b
    feats['win_streak_diff'] = ws_r - ws_b
    feats['lose_streak_diff'] = ls_r - ls_b
    feats['wins_diff'] = nw_r - nw_b
    feats['losses_diff'] = nl_r - nl_b
    feats['win_pct_diff'] = wp_r - wp_b

    # method wins
    (dec_wr, ko_wr, sub_wr,
     dec_wb, ko_wb, sub_wb) = method_wins(df).T

    feats['decision_wins_red'] = dec_wr
    feats['ko_wins_red'] = ko_wr
    feats['sub_wins_red'] = sub_wr
    feats['decision_wins_blue'] = dec_wb
    feats['ko_wins_blue'] = ko_wb
    feats['sub_wins_blue'] = sub_wb

    feats['decision_wins_diff'] = dec_wr - dec_wb
    feats['ko_wins_diff'] = ko_wr - ko_wb
    feats['sub_wins_diff'] = sub_wr - sub_wb

    # method losses
    (dec_lr, ko_lr, sub_lr,
     dec_lb, ko_lb, sub_lb) = method_losses(df).T

    feats['decision_losses_red'] = dec_lr
    feats['ko_losses_red'] = ko_lr
    feats['sub_losses_red'] = sub_lr
    feats['decision_losses_blue'] = dec_lb
    feats['ko_losses_blue'] = ko_lb
    feats['sub_losses_blue'] = sub_lb

    feats['decision_losses_diff'] = dec_lr - dec_lb
    feats['ko_losses_diff'] = ko_lr - ko_lb
    feats['sub_losses_diff'] = sub_lr - sub_lb

    # max rating won against
    # max_elo_r, max_elo_b = max_rating_won_against(df, 'elo').T
    # feats['max_elo_win_red'] = max_elo_r
    # feats['max_elo_win_blue'] = max_elo_b

    # max_g_r, max_g_b = max_rating_won_against(df, 'glicko').T
    # feats['max_glicko_win_red'] = max_g_r
    # feats['max_glicko_win_blue'] = max_g_b

    # method win pct
    (ko_pr, dec_pr, sub_pr,
     ko_pb, dec_pb, sub_pb) = method_win_pct(df).T

    feats['ko_pct_red'] = ko_pr
    feats['dec_pct_red'] = dec_pr
    feats['sub_pct_red'] = sub_pr
    feats['ko_pct_blue'] = ko_pb
    feats['dec_pct_blue'] = dec_pb
    feats['sub_pct_blue'] = sub_pb

    feats['ko_pct_diff'] = ko_pr - ko_pb
    feats['dec_pct_diff'] = dec_pr - dec_pb
    feats['sub_pct_diff'] = sub_pr - sub_pb

    # womens fight flag
    feats['womens_fight'] = womens_fight(df).astype(int)

    # title fight stats (returns array-like rows)
    cols_to_add = title_fights_stats_columns(df)  # shape (n, 8)
    cols_to_add = np.asarray(cols_to_add)

    feats['red_title_fights'] = cols_to_add[:, 0]
    feats['blue_title_fights'] = cols_to_add[:, 1]
    feats['red_title_wins'] = cols_to_add[:, 2]
    feats['blue_title_wins'] = cols_to_add[:, 3]
    feats['red_title_losses'] = cols_to_add[:, 4]
    feats['blue_title_losses'] = cols_to_add[:, 5]
    feats['red_title_win_pct'] = cols_to_add[:, 6]
    feats['blue_title_win_pct'] = cols_to_add[:, 7]

    # event age in days
    feats['event_age'] = (pd.Timestamp.today() - df['date']).dt.days.to_numpy()

    # ---- concat ONCE ----
    new_cols = pd.DataFrame(feats, index=df.index)
    df = pd.concat([df, new_cols], axis=1)

    return df


