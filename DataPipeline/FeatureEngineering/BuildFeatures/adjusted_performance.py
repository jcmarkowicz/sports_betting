import numpy as np 
import pandas as pd 

from collections import defaultdict

def time_decay_average(arr, decay_lambda=0.13):
    arr = np.asarray(arr)
    
    # 0 = most recent, larger = older
    age = np.arange(len(arr)-1, -1, -1)
    
    weights = np.exp(-decay_lambda * age)
    
    return np.sum(weights * arr) / np.sum(weights)

def mean_absolute_deviation(series):
    return np.mean(np.abs(series - np.mean(series)))

def adjusted_performance(df, row, hero, villain, weight_class, fight_date, performance_col, opponent_performance_col, hero_color, k=3):
    """
    Usage: performance col=sig strikes landed, opponent performance col=sig strikes absorbed

    Computes performance of hero fighter for SINGLE FIGHT 
    """

    # must include fight date stats, this stat is for next fight 
    df_history = df[(df['weight_class'] == weight_class) & (df['date'] < fight_date)].dropna()
    df_villain = df_history[(df_history['fighter_red'] == villain) | (df_history['fighter_blue'] == villain)]

    if df_history.shape[0] == 0 or df_villain.shape[0] == 0: 
         return None

    # bayesian shrinkage
    n = df_villain.shape[0]
    villain_allowed = np.where(df_villain['fighter_red'] == villain, df_villain[opponent_performance_col + '_red'], df_villain[opponent_performance_col + '_blue'])
    villain_allowed_mean = time_decay_average(villain_allowed)
    if n == 0:
         print('wrong df villain ')

    # total weight class average 
    weight_class_allowed_mean = df_history[[opponent_performance_col + '_red', opponent_performance_col + '_blue']].mean().mean()

    wc_total_allowed = pd.concat([
        df_history[opponent_performance_col + '_red'],
        df_history[opponent_performance_col + '_blue']
    ])

    # Z score formulation
  
    # average allowed by villain, wc weighted
    w_bayes = n / (n+k)
    mu_shrunk = w_bayes * villain_allowed_mean + (1-w_bayes) * weight_class_allowed_mean

    # get sigma, wc weighted 
    MAD_villian = mean_absolute_deviation(villain_allowed)
    MAD_wc = mean_absolute_deviation(wc_total_allowed)

    mad_shrunk = w_bayes * MAD_villian + (1-w_bayes) * MAD_wc
    mad_shrunk = max(mad_shrunk, 1e-3) # avoid division by zero
    
    # get performance of current fight 
    hero_performance = row[performance_col + f'_{hero_color}']

    if hero_performance is np.isnan(hero_performance):
         print(f'Hero performance nan')

    # adjusted performance for current fight 
    adjusted_perf = (hero_performance - mu_shrunk) / mad_shrunk 
    adjusted_perf = np.clip(adjusted_perf, -7, 7) # cap extreme values

    return adjusted_perf 


def compute_adjusted_performance(df_, performance_col, opponent_performance_col, time_decay=False):
        """
        assumes data frame already sorted by date ascending 
        assumes CURRENT FIGHT STATS
        """
        
        df = df_.copy()

        red_arr = []
        blue_arr = []
        fighter_dict = defaultdict(lambda: [None])

        for _, row in df.iterrows():
             
            red_fighter = row['fighter_red']
            blue_fighter = row['fighter_blue']

            if time_decay:

                red_adj_hist = [x for x in fighter_dict[red_fighter] if x is not None]
                blue_adj_hist = [x for x in fighter_dict[blue_fighter] if x is not None]

                red_adj_dec = time_decay_average(red_adj_hist) if len(red_adj_hist) != 0 else None 
                blue_adj_dec = time_decay_average(blue_adj_hist) if len(blue_adj_hist) != 0 else None 

                red_arr.append(red_adj_dec)
                blue_arr.append(blue_adj_dec)
                
            else:         
                red_arr.append(fighter_dict[red_fighter][-1])
                blue_arr.append(fighter_dict[blue_fighter][-1])

            hero_perf_col = performance_col #+ '_red'
            villain_perf_col = opponent_performance_col #+ '_blue'
            red_adj_perf = adjusted_performance(df, row, hero=red_fighter, villain=blue_fighter, weight_class=row['weight_class'], fight_date=row['date'], 
                                                performance_col=hero_perf_col, opponent_performance_col=villain_perf_col, hero_color='red')
            
            hero_perf_col = performance_col #+ '_blue'
            villain_perf_col = opponent_performance_col #+ '_red'
            blue_adj_perf = adjusted_performance(df, row, hero=blue_fighter, villain=red_fighter, weight_class=row['weight_class'], fight_date=row['date'],
                                                  performance_col=hero_perf_col, opponent_performance_col=villain_perf_col, hero_color='blue')
            
            fighter_dict[red_fighter].append(red_adj_perf)
            fighter_dict[blue_fighter].append(blue_adj_perf)
  
        return np.column_stack([red_arr, blue_arr])
