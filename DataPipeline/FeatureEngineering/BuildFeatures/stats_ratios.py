import numpy as np 
import pandas as pd 

from collections import defaultdict

def bayesian_smoothing(df, hero_total, villain_total, feature_hero, feature_villain, weight_class, date, k=3):

    # filter Nones because pre fight should have None 
    hero_total = [x for x in hero_total if pd.isna(x) is False]
    villain_total = [x for x in villain_total if pd.isna(x) is False] 
    
    if len(hero_total) == 0:
        return None
    if len(villain_total) == 0:
        return None

    # PRE FIGHT STATS INCLUDE CURR DATE 
    df_weight_class = df[(df['weight_class'] == weight_class) & (df['date'] <= date)].dropna()# [[feature_villian, feature hero, date]] ISSUE WITH DIVIDE BY ZERO WIT
   
    # number of prev obs
    n_hero = len(hero_total)
    n_villain = len(villain_total)

    weight_hero = n_hero / (n_hero + k)
    weight_villain = n_villain / (n_villain + k)

    # average per fight, need red blue averages 
    if df_weight_class.shape[0] == 0:
        wc_total_mean = 0
        weight_hero = 1
        weight_villain = 1
    
    else:
        wc_total_mean = df_weight_class[[feature_hero, feature_villain]].mean().mean()

    # if this data is already time delayed dont do it again 
    # numerator = weight_hero * time_decay_average(hero_total) + (1-weight_hero) * wc_total_mean
    # denominator = weight_villian * time_decay_average(villain_total) + (1-weight_villian) * wc_total_mean

    numerator = weight_hero * np.mean(hero_total) + (1-weight_hero) * wc_total_mean
    denominator = weight_villain * np.mean(villain_total) + (1-weight_villain) * wc_total_mean

    ratio_hero = numerator / denominator if denominator != 0 else numerator
    return ratio_hero

def time_decay_average(arr, decay_lambda=0.13):
    arr = np.asarray(arr)
    
    # 0 = most recent, larger = older
    age = np.arange(len(arr)-1, -1, -1)
    
    weights = np.exp(-decay_lambda * age)
    
    return np.sum(weights * arr) / np.sum(weights)

def td_ratio(df):
    """ 
    Ratio of td_landed per fight to opponett td_landed per fight, with bayesian smoothing and time decay.

    per minute feats already pre fight, currently time smoothed
    """

    ratio_dic = defaultdict(lambda: defaultdict(lambda: []))
    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        # append current fight stats, assumes this is already pre fight 
        ratio_dic[red_fighter]['fighter_td'].append(row['td_landed_pm_red']) # dont use total, get per fight 
        ratio_dic[red_fighter]['opponent_td'].append(row['td_landed_pm_blue'])

        ratio_dic[blue_fighter]['fighter_td'].append(row['td_landed_pm_blue'])
        ratio_dic[blue_fighter]['opponent_td'].append(row['td_landed_pm_red'])
        
        red_ratio_curr = bayesian_smoothing(df, ratio_dic[red_fighter]['fighter_td'], ratio_dic[red_fighter]['opponent_td'], 
                                            feature_hero='td_landed_pm_red', feature_villain='td_landed_pm_blue', 
                                            weight_class=row['weight_class'], date=row['date'])
        
        blue_ratio_curr = bayesian_smoothing(df, ratio_dic[blue_fighter]['fighter_td'], ratio_dic[blue_fighter]['opponent_td'],
                                            feature_hero='td_landed_pm_blue', feature_villain='td_landed_pm_red',
                                              weight_class=row['weight_class'], date=row['date'])
    
        red_ratio.append(red_ratio_curr)
        blue_ratio.append(blue_ratio_curr)


    return np.column_stack([red_ratio, blue_ratio])


def sig_strikes_ratio(df):
    """
    Sig strike landed hero history / sig strikes landed villain history,  with bayesian smoothing and time decay

    per minute feats already pre fight
    """

    ratio_dic = defaultdict(lambda: defaultdict(lambda: []))

    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        ratio_dic[red_fighter]['fighter_ss'].append(row['sig_str_landed_pm_red'])
        ratio_dic[red_fighter]['opponent_ss'].append(row['sig_str_landed_pm_blue'])

        ratio_dic[blue_fighter]['fighter_ss'].append(row['sig_str_landed_pm_blue'])
        ratio_dic[blue_fighter]['opponent_ss'].append(row['sig_str_landed_pm_red'])

        red_ratio_curr = bayesian_smoothing(df, ratio_dic[red_fighter]['fighter_ss'], ratio_dic[red_fighter]['opponent_ss'],
                                            feature_hero='sig_str_landed_pm_red', feature_villain='sig_str_landed_pm_blue', 
                                            weight_class=row['weight_class'], date=row['date'])
        
        blue_ratio_curr = bayesian_smoothing(df, ratio_dic[blue_fighter]['fighter_ss'], ratio_dic[blue_fighter]['opponent_ss'],
                                                feature_hero='sig_str_landed_pm_blue', feature_villain='sig_str_landed_pm_red',
                                                  weight_class=row['weight_class'], date=row['date'])
        
        red_ratio.append(red_ratio_curr)
        blue_ratio.append(blue_ratio_curr)

    return np.column_stack([red_ratio, blue_ratio])


def control_pr_ratio(df):
    """
    control per minute hero history / control per minute villain history,  with bayesian smoothing and time decay

    Asumme df already contains PRE FIGHT Stats 
    """

    control_stats = defaultdict(lambda: defaultdict(lambda: []))

    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']


        control_stats[red_fighter]['fighter_control'].append(row['control_pm_red'])
        control_stats[red_fighter]['opponent_control'].append(row['control_pm_blue'])

        control_stats[blue_fighter]['fighter_control'].append(row['control_pm_blue'])
        control_stats[blue_fighter]['opponent_control'].append(row['control_pm_red'])

        red_ratio_curr = bayesian_smoothing(df, control_stats[red_fighter]['fighter_control'], control_stats[red_fighter]['opponent_control'],
                                            feature_hero='control_pm_red', feature_villain='control_pm_blue', 
                                            weight_class=row['weight_class'], date=row['date'])
        
        blue_ratio_curr = bayesian_smoothing(df, control_stats[blue_fighter]['fighter_control'], control_stats[blue_fighter]['opponent_control'],
                                            feature_hero='control_pm_blue', feature_villain='control_pm_red',
                                              weight_class=row['weight_class'], date=row['date'])

        # assumes df already has pre fight stats, see rolling_stats.py pm_feats
        red_ratio.append(red_ratio_curr)
        blue_ratio.append(blue_ratio_curr)

    return np.column_stack([red_ratio, blue_ratio])