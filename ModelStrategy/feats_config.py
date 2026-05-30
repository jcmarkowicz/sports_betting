
import numpy as np
from dataclasses import dataclass

@dataclass
class FeatsConfig:
    non_feats_open = [
        'date','event_date','event_location','fighter_blue','fighter_red',
        'method','og_blue_name','og_red_fighter', 'red_fighter_stats', 'blue_fighter_stats',
        'pimp_close1_blue','pimp_close1_red','pimp_close2_blue','pimp_close2_red',
        'juice_close1_blue','juice_close1_red','juice_close2_blue','juice_close2_red',
        'line_movement_close1_blue','line_movement_close1_red','line_movement_close2_blue','line_movement_close2_red',
        'winner_name',
        
        'red_fighter_odds','blue_fighter_odds',
        'dec_close1_blue','dec_close1_red','dec_close2_blue','dec_close2_red',
        'dec_fair_close1_blue','dec_fair_close1_red','dec_fair_close2_blue','dec_fair_close2_red',
        'red_ud_to_fav_close1','red_ud_to_fav_close2','blue_ud_to_fav_close1','blue_ud_to_fav_close2',
        'red_stayed_fav_close1','red_stayed_fav_close2','blue_stayed_fav_close1','blue_stayed_fav_close2',
        'red_fav_to_ud_close1','red_fav_to_ud_close2','blue_fav_to_ud_close1','blue_fav_to_ud_close2',
        'red_stayed_dog_close1','red_stayed_dog_close2','blue_stayed_dog_close1','blue_stayed_dog_close2',
        'proba_fair_close1_red','proba_fair_close1_blue','proba_fair_close2_red','proba_fair_close2_blue',
        'performance_bonus_winner', 'fight_otn_bonus', 'close1_blue','close1_red', 'close2_blue', 'close2_red'
    ] 

    selected_feats_open = [
                    'proba_fair_open_diff', 'reach_diff', 
                    
                    'sub_att_pm_red', 'sub_att_pm_blue',
                    'ratio_control_diff',

                    'td_landed_pm_diff',  
                    'ratio_td_diff', 
                    'adjusted_td_red', 'adjusted_td_blue',

                    'sig_str_absorbed_total_diff', 
                    'sig_str_accuracy_pct_diff',
                    'sig_str_defense_pct_diff',
                    'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
                    
                    'win_pct_red', 'win_pct_blue',
                    'win_streak_diff', 'lose_streak_diff',
                    'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue',
                    ]


    non_feats_close1 = [
        'date','event_date','event_location','fighter_blue','fighter_red',
        'method','og_blue_name','og_red_fighter', 'red_fighter_stats', 'blue_fighter_stats',
        'pimp_close1_blue','pimp_close1_red','pimp_close2_blue','pimp_close2_red',
        'juice_close1_blue','juice_close1_red','juice_close2_blue','juice_close2_red',
        'winner_name',
        
        'red_fighter_odds','blue_fighter_odds',
        'dec_close1_blue','dec_close1_red','dec_close2_blue','dec_close2_red',
        'dec_fair_close1_blue','dec_fair_close1_red','dec_fair_close2_blue','dec_fair_close2_red',
        'red_ud_to_fav_close1','red_ud_to_fav_close2','blue_ud_to_fav_close1','blue_ud_to_fav_close2',
        'red_stayed_fav_close1','red_stayed_fav_close2','blue_stayed_fav_close1','blue_stayed_fav_close2',
        'red_fav_to_ud_close1','red_fav_to_ud_close2','blue_fav_to_ud_close1','blue_fav_to_ud_close2',
        'red_stayed_dog_close1','red_stayed_dog_close2','blue_stayed_dog_close1','blue_stayed_dog_close2',
        'performance_bonus_winner', 'fight_otn_bonus'
    ] 


    selected_feats_close1 = [
                    'proba_fair_close1_diff', 'proba_fair_open_diff', 'reach_diff', 
                    
                    'sub_att_pm_red', 'sub_att_pm_blue',
                    'ratio_control_diff',

                    'td_landed_pm_diff',  
                    'ratio_td_diff', 
                    'adjusted_td_red', 'adjusted_td_blue',

                    'sig_str_absorbed_total_diff', 
                    'sig_str_accuracy_pct_diff',
                    'sig_str_defense_pct_diff',
                    'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
                    
                    'win_pct_red', 'win_pct_blue',
                    'win_streak_diff', 'lose_streak_diff',
                    'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue',
                    ]

    non_feats_close2 = [
        'date','event_date','event_location','fighter_blue','fighter_red',
        'method','og_blue_name','og_red_fighter', 'red_fighter_stats', 'blue_fighter_stats',
        'pimp_close1_blue','pimp_close1_red','pimp_close2_blue','pimp_close2_red',
        'juice_close1_blue','juice_close1_red','juice_close2_blue','juice_close2_red',
        'winner_name',
        
        'red_fighter_odds','blue_fighter_odds',
        'dec_close1_blue','dec_close1_red','dec_close2_blue','dec_close2_red',
        'dec_fair_close1_blue','dec_fair_close1_red','dec_fair_close2_blue','dec_fair_close2_red',
        'red_ud_to_fav_close1','red_ud_to_fav_close2','blue_ud_to_fav_close1','blue_ud_to_fav_close2',
        'red_stayed_fav_close1','red_stayed_fav_close2','blue_stayed_fav_close1','blue_stayed_fav_close2',
        'red_fav_to_ud_close1','red_fav_to_ud_close2','blue_fav_to_ud_close1','blue_fav_to_ud_close2',
        'red_stayed_dog_close1','red_stayed_dog_close2','blue_stayed_dog_close1','blue_stayed_dog_close2',
        'performance_bonus_winner', 'fight_otn_bonus'
    ] 

    selected_feats_close2 = [
                    'proba_fair_close2_diff', 'proba_fair_open_diff', 'reach_diff', 
                    
                    'sub_att_pm_red', 'sub_att_pm_blue',
                    'ratio_control_diff',

                    'td_landed_pm_diff',  
                    'ratio_td_diff', 
                    'adjusted_td_red', 'adjusted_td_blue',

                    'sig_str_absorbed_total_diff', 
                    'sig_str_accuracy_pct_diff',
                    'sig_str_defense_pct_diff',
                    'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
                    
                    'win_pct_red', 'win_pct_blue',
                    'win_streak_diff', 'lose_streak_diff',
                    'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue',
                    ]
