# Recovered from local IPython execution history; original notebook not modified.
# Session 1522, input 12: last recovered cell defining all three lists together.

open_feats = [
        'proba_fair_open_diff', 'reach_diff', 
        
        'td_accuracy_pct_diff', 'td_defense_pct_diff','td_attempted_pm_diff',
        'adjusted_td_red', 'adjusted_td_blue',

        'sig_str_absorbed_total_diff', 
        'sig_str_accuracy_pct_diff',
        'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
        'sig_str_absorbed_pm_diff',
        
        'win_pct_diff',
        'win_streak_diff', 'lose_streak_diff',
        'age_diff', 'kd_total_diff', 'adjusted_control_red', 'adjusted_control_blue',
        'opp_age_red','opp_age_blue',  'head_str_total_diff','ko_losses_red', 'ko_losses_blue',
        
    ]

open_close2_feats = [
        'proba_fair_close2_diff', 'reach_diff', 
        
        'td_accuracy_pct_diff', 'td_defense_pct_diff','td_attempted_pm_diff',
        'adjusted_td_red', 'adjusted_td_blue',

        'sig_str_absorbed_total_diff', 
        'sig_str_accuracy_pct_diff',
        'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
        'sig_str_absorbed_pm_diff',
        
        'win_pct_diff',
        'win_streak_diff', 'lose_streak_diff',
        'age_diff', 'kd_total_diff', 'adjusted_control_red', 'adjusted_control_blue',
        'opp_age_red','opp_age_blue','head_str_total_diff','ko_losses_red', 'ko_losses_blue',
    ]

open_close1_feats = [
        'proba_fair_close1_diff', 'reach_diff', 
        
        'td_accuracy_pct_diff', 'td_defense_pct_diff','td_attempted_pm_diff',
        'adjusted_td_red', 'adjusted_td_blue',

        'sig_str_absorbed_total_diff', 
        'sig_str_accuracy_pct_diff',
        'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
        'sig_str_absorbed_pm_diff',
        
        'win_pct_diff',
        'win_streak_diff', 'lose_streak_diff',
        'age_diff', 'kd_total_diff', 'adjusted_control_red', 'adjusted_control_blue',
        'opp_age_red','opp_age_blue',  'head_str_total_diff','ko_losses_red', 'ko_losses_blue',
    ]

# Session 1522, input 22: later open-only definition (matches current notebook).
later_open_feats = [
        'proba_fair_open_diff', 'reach_diff', 
        
        'td_landed_pm_diff',
        'td_accuracy_pct_diff', 'td_defense_pct_diff',
        'ratio_td_diff', 
        'adjusted_td_red', 'adjusted_td_blue',

        'sig_str_absorbed_total_diff', 
        'sig_str_accuracy_pct_diff',
        'adjusted_sig_str_blue', 'adjusted_sig_str_red', 
        'td_attempted_pm_diff', 'sig_str_absorbed_pm_diff',
        
        'win_pct_red', 'win_pct_blue',
        'win_streak_diff', 'lose_streak_diff',
        'age_diff', 'kd_total_diff', 'adjusted_control_red', 'adjusted_control_blue',
        'opp_age_red','opp_age_blue'
    ]
