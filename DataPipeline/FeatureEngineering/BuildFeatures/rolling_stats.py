import numpy as np
import pandas as pd

from collections import defaultdict

from DataPipeline.FeatureEngineering.BuildFeatures.stats_ratios import control_pr_ratio, td_ratio, sig_strikes_ratio
from DataPipeline.FeatureEngineering.BuildFeatures.adjusted_performance import compute_adjusted_performance

import time

def time_decay_average(arr, decay_lambda=0.13):
    arr = np.asarray(arr)
    
    # 0 = most recent, larger = older
    age = np.arange(len(arr)-1, -1, -1)
    
    weights = np.exp(-decay_lambda * age)
    
    return np.sum(weights * arr) / np.sum(weights)


def compute_defense_feats(fighter_name, row, feat, color, df_dict, stats_dict, time_col, df, weight_class, date, shrinkage_k=3):

    """ 
    computes the following feat types: total_attempted_against, total_landed_against, pct per fight
    pct feats: 1 - landed_against / attempted against,
    use bayesian shringage for pct feats

    returns None for first instance of a fighter    

    creates features for total attempted/landed AGAINST red/blue fighter
    """

    opp = 'red' if color == 'blue' else 'red'

    if len(stats_dict[fighter_name][time_col]) == 0:
        pct_feat = None
        sum_attempted_against = None
        sum_landed_against = None

    else: 
        attempted_against = stats_dict[fighter_name][f'{feat}_attempted_against'] # should not be None saved here
        landed_against = stats_dict[fighter_name][f'{feat}_landed_against']

        # mostly for takedowns 
        # get average pct feat
        # get weight class stats pre fight date
        df_weight_class = df[(df['weight_class'] == weight_class) & (df['date'] < date)]
        
        # weight class mean PER FIGHT 
        wc_mean_attempted_against = df_weight_class[[f'{feat}_attempted_red', f'{feat}_attempted_blue']].mean().mean()
        wc_mean_landed_against = df_weight_class[[f'{feat}_landed_red', f'{feat}_landed_blue']].mean().mean()

        # bayesian weighting 
        n = len(stats_dict[fighter_name][f'{feat}_landed_against'])
        weight = n / (n + shrinkage_k)

        # weighted pcts per fight 
        landed_against_weighted = (weight * time_decay_average(landed_against) + (1-weight) * wc_mean_landed_against)
        attempted_against_weighted = (weight * time_decay_average(attempted_against) + (1-weight) * wc_mean_attempted_against)
            
        pct_feat = 1 - (landed_against_weighted / attempted_against_weighted)

        # sums for total counts, not weighted
        sum_attempted_against = np.sum(attempted_against)
        sum_landed_against = np.sum(landed_against)

    # pct PER FIGHT
    df_dict[f'{feat}_defense_pct_{color}'].append(pct_feat)

    # save total counts ALL TIME 
    df_dict[f'{feat}_total_attempted_against_{color}'].append(sum_attempted_against) 
    df_dict[f'{feat}_total_landed_against_{color}'].append(sum_landed_against)

    # per fight counts appended to stats dict, OPP fighter, against feature here 
    stats_dict[fighter_name][f'{feat}_attempted_against'].append(row[f'{feat}_attempted_{opp}'])
    stats_dict[fighter_name][f'{feat}_landed_against'].append(row[f'{feat}_landed_{opp}'])

    return stats_dict, df_dict


def compute_accuracy_feats(fighter_name, feat, color, df_dict, stats_dict, time_col, df, weight_class, date, shrinkage_k=3):
    """
    computes the following feat types: attempted, landed, pct
    pct feats: landed / attempted
    use bayesian shringage for pct feats

    returns None for first instance of a fighter  
    """

    if len(stats_dict[fighter_name][time_col]) == 0:
        acc_pct = None
    
    else: 
        total_att = stats_dict[fighter_name][f'{feat}_attempted']
        total_land = stats_dict[fighter_name][f'{feat}_landed']

        # mostly for takedowns 
        # get average pct feat
        # get weight class stats pre fight date
        df_weight_class = df[(df['weight_class'] == weight_class) & (df['date'] < date)]

        n = len(stats_dict[fighter_name][f'{feat}_attempted'])
        weight = n / (n + shrinkage_k)

        wc_mean_attempted = df_weight_class[[f'{feat}_attempted_red', f'{feat}_attempted_blue']].mean().mean()
        wc_mean_landed = df_weight_class[[f'{feat}_landed_red', f'{feat}_landed_blue']].mean().mean()

        total_att_weighted = weight * time_decay_average(total_att) + (1-weight) * wc_mean_attempted
        total_land_weighted = weight * time_decay_average(total_land) + (1-weight) * wc_mean_landed

        acc_pct = total_land_weighted / total_att_weighted

    # attempted feats appended to stats dict in prefight stats function
    df_dict[f'{feat}_accuracy_pct_{color}'].append(acc_pct)
    return df_dict


def prefight_stats(stats_dict, df_dict, fighter_name, feature, row, time_col, color, time_decay=False, lm=.13):
    """
    Computes the following feat types: total, pm rates

    Appends current fight stats to stats dict 

    """
    
    if len(stats_dict[fighter_name][time_col]) == 0:
        time_feature = None
        total_feature = None
        pm_feature = None

    else: 
        if time_decay:
            features = np.array(stats_dict[fighter_name][feature])
            minutes  = np.array(stats_dict[fighter_name][time_col])
            n = len(features) # None not included in stats dict for these features

            # fight index decay (0 = most recent)
            age = np.arange(n-1, -1, -1)
            weights = np.exp(-lm * age)
            pm_feature = np.sum(features * weights) / np.sum(minutes * weights)
        
        else:
            total_feature = np.sum(stats_dict[fighter_name][feature])
            time_feature = np.sum(stats_dict[fighter_name][time_col])
            pm_feature = total_feature / time_feature
        
        total_feature = np.sum(stats_dict[fighter_name][feature])

    df_dict[f'{feature}_pm_{color}'].append(pm_feature)
    df_dict[f'{feature}_total_{color}'].append(total_feature)

    stats_dict[fighter_name][feature].append(row[f'{feature}_{color}'])

    return stats_dict, df_dict 


def apply_rolling_stats(ufc_features): 
    """
    Apply rolling stats in ascending order of date
    """  

    # stats dict holds the history of stats for fighter PRE FIGHT
    stats_dict = defaultdict(lambda: defaultdict(list))

    # df dict holds the rolling stats PRE FIGHT 
    df_dict = defaultdict(list)

    per_fight_features = ufc_features.copy()
    per_fight_features['date'] = pd.to_datetime(per_fight_features['date'])
    per_fight_features = per_fight_features.sort_values(by='date', ascending=True).reset_index(drop=True)

    # per minute rates, total counts
    striking_features = ['kd', 'sig_str_landed', 'sig_str_absorbed', 'sig_str_attempted', 'leg_str', 'head_str', 'body_str', 'clinch_str']
    grapling_features = ['td_landed', 'td_attempted', 'td_defended', 'control', 'sub_att', 'reverse']
    
    # compute total counts for and against, percentage total landed/attempted for/against
    defense_features = ['td', 'sig_str']
    accuracy_features = ['td', 'sig_str']

    fighter_attr = ['age', 'height', 'reach']
    general_features = ['date', 'event_location', 'weight_class', 'title_fight']
    win_features = ["performance_bonus_winner", "fight_otn_bonus", 'method', 'winner', 'fighter_red', 'fighter_blue']
    colors = ['red', 'blue']
    time_col = 'fight_minutes'

    for _, row in per_fight_features.iterrows(): 
    
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        
        # compute PRE fight rolling feats for ACC and DEFENSE
        for feat in defense_features: 
            stats_dict, df_dict = compute_defense_feats(red_fighter, row, feat, 'red', df_dict, stats_dict, time_col, per_fight_features, row['weight_class'], row['date'])
            stats_dict, df_dict = compute_defense_feats(blue_fighter, row, feat, 'blue', df_dict, stats_dict, time_col, per_fight_features, row['weight_class'], row['date'])

        for feat in accuracy_features: 
            df_dict = compute_accuracy_feats(red_fighter, feat,'red', df_dict, stats_dict, time_col, per_fight_features, row['weight_class'], row['date'])
            df_dict = compute_accuracy_feats(blue_fighter, feat,'blue', df_dict, stats_dict, time_col, per_fight_features, row['weight_class'], row['date'])

        # Compute PM rates, append result to df dict
        for feature in striking_features + grapling_features:
            stats_dict, df_dict = prefight_stats(stats_dict, df_dict, red_fighter, feature, row, time_col, 'red', time_decay=True, lm=.13) # time updated in stats dict here
            stats_dict, df_dict = prefight_stats(stats_dict, df_dict, blue_fighter, feature, row, time_col, 'blue', time_decay=True, lm=.13) # time updated in stats dict here 

        # append post fight time 
        for color in colors:
            fighter = red_fighter if color=='red' else blue_fighter
            time_feature = np.sum(stats_dict[fighter][time_col])

            df_dict[f'total_fight_time_{color}'].append(time_feature)
            stats_dict[fighter][time_col].append(row[time_col])

        # non rolling stats: age, height, reach, title fight, winner
        for i, attr in enumerate(fighter_attr): 
            df_dict[f'{attr}_red'].append(row[f'{attr}_red'])
            df_dict[f'{attr}_blue'].append(row[f'{attr}_blue'])

        # add date, ect. 
        for feat in general_features: 
            df_dict[feat].append(row[feat])

        for win_feat in win_features:
            if win_feat == 'winner': 
                if row['winner'] == red_fighter: 
                    color = 1.0
                elif row['winner'] == blue_fighter:
                    color = 0.0
                elif row['winner'] == 'NC' or row['winner'] == 'DRAW':
                    color = 2.0
                elif pd.isna(row['winner']):
                    color = 3.0

                df_dict['winner'].append(color)
                df_dict['winner_name'].append(row['winner'])

            else: 
                df_dict[win_feat].append(row[win_feat])

    final_df = pd.DataFrame(df_dict)

    # ratios for total stats
    # Compute all new columns with per fight features
    # normalize ratios with PM feats, note PM feats already are already pre fight stats
    # print(f'REACHED RATIOS')
    # time.sleep(5)
    
    td_cols = td_ratio(final_df)
    control_cols = control_pr_ratio(final_df)
    sigstrike_cols = sig_strikes_ratio(final_df)

    # print(f'REACHED ADJ perf')
    # time.sleep(5)

    adjusted_sig_str_cols= compute_adjusted_performance(per_fight_features, performance_col='sig_str_landed', 
                                                        opponent_performance_col='sig_str_absorbed', time_decay=True)
    
    adjusted_td_cols = compute_adjusted_performance(per_fight_features, performance_col='td_landed', 
                                                    opponent_performance_col='td_defended', time_decay=True)
    
    
    # Build one DataFrame containing all new columns
    new_cols = pd.DataFrame({
        'ratio_td_red': td_cols[:,0],
        'ratio_td_blue': td_cols[:,1],
        'ratio_control_red': control_cols[:,0],
        'ratio_control_blue': control_cols[:,1],
        'ratio_sig_str_red': sigstrike_cols[:,0],
        'ratio_sig_str_blue': sigstrike_cols[:,1],
        'adjusted_sig_str_red': adjusted_sig_str_cols[:,0],
        'adjusted_sig_str_blue':adjusted_sig_str_cols[:,1],
        'adjusted_td_red': adjusted_td_cols[:,0],
        'adjusted_td_blue': adjusted_td_cols[:,1]
    })
    final_df = pd.concat([final_df, new_cols], axis=1)

    final_df['ratio_td_diff'] = final_df['ratio_td_red'] - final_df['ratio_td_blue']
    final_df['ratio_control_diff'] = final_df['ratio_control_red'] - final_df['ratio_control_blue']
    final_df['ratio_sigstrike_diff'] = final_df['ratio_sig_str_red'] - final_df['ratio_sig_str_blue']

    return final_df 