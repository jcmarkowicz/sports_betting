from dataclasses import dataclass
from pathlib import Path
from DataPipeline.FeatureEngineering.features_pipeline import BASE_DIR



@dataclass
class config:
    
    # --- Bet Params ---
    bankroll:int = 800
    N_paths:int = 250
    max_drawdown_array = [0.3, 0.45, 0.45]
    parlay_top_ev = 2  

    # --- Paths --- 
    base_dir = Path(__file__).resolve().parents[0]
    db_scrape_date = r'2026-02-26'

    stats_history_file_string = base_dir / "Data" / "scraped_data_main" / f"stats_history_{db_scrape_date}.csv"
    odds_history_file_string = base_dir / "Data" / "scraped_data_main" / f"odds_history_{db_scrape_date}.csv"

    missing_stats_history_fp = base_dir / "Data" / "non_merged_features" / f"non_merged_stats.csv"
    missing_odds_history_fp = base_dir / "Data" / "non_merged_features" / f"non_merged_odds.csv"

    upcoming_scraped_stats_string = base_dir / "Data" / "upcoming_events" / "scraped_data" / "upcoming_stats"
    upcoming_scraped_odds_string = base_dir / "Data" /"upcoming_events" /"scraped_data" / "upcoming_odds"

    # folder for stats/odds FEATURES per event 
    upcoming_events_folder =  base_dir / "Data" / "upcoming_events" / "event_features" 

    # folder for 
    ml_bets_folder = base_dir / "Data" / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = base_dir / "Data" / "upcoming_events" / "parlays"

    model_open_path = base_dir / "Data" / "saved_models" / "logit_model_open.pkl"
    model_close1_path = base_dir / "Data" / "saved_models" / "logit_model_close1.pkl"
    model_close2_path = base_dir / "Data" / "saved_models" / "logit_model_close2.pkl"

    scaler_open_path = base_dir / "Data" / "saved_models" / "scaler_open.pkl"
    scaler_close1_path = base_dir / "Data" / "saved_models" / "scaler_close1.pkl"
    scaler_close2_path = base_dir / "Data" / "saved_models" / "scaler_close2.pkl"


    ml_folder = base_dir / "Data" / "upcoming_events" / "straight_bets"
    parlay_folder = base_dir / "Data" / "upcoming_events" / "parlays" 

    # --- Results Tracking Params ---
    ml_history_fp = base_dir / 'Data' / 'betting_results' / 'moneyline_results.csv'
    parlay_history_fp = base_dir / 'Data' / 'betting_results' / 'parlay_results.csv'
    event_features_folder = base_dir / "Data" / "upcoming_events" / "event_features" / "upcoming_odds_stats"

    non_merged_stats_fp = base_dir / "Data" / "non_merged_features" / "non_merged_stats.csv"
    non_merged_odds_fp = base_dir / "Data" / "non_merged_features" / "non_merged_odds.csv"

    ml_pct_returns_fp = base_dir / 'Data' / 'betting_results' / 'ml_pct_returns.csv'
    parlay_pct_returns_fp = base_dir / 'Data' / 'betting_results' / 'parlay_pct_returns.csv'

    # email params
    sender_email = "jcmarkufc@gmail.com"
    reciever_email = ['jcmarkowicz@outlook.com']
    columns_to_email = ['fighter_red', 'fighter_blue','pred_name_open',
                    'open_red','open_blue','pred_winner_open',
                    'fstar_open','stake_open'
                    ]  

    parlay_columns = [
        'choice_fighter_name_open',
        'parlay_fstar_open',
        'parlay_odds_open',
        'stake_open'
        ]
    

    # --- Model Params ---
    open_feats = [
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
            'elo_red', 'elo_blue', 'elo_pred', 'age_red', 'age_blue'
            ]
    
    close1_feats = [
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
    
    close2_feats = [
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
    

    # --- Display Params ---
    ml_column_order = ['fighter_red', 'fighter_blue', 'open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue',
                       'stake_open', 'stake_close1', 'stake_close2',  
                       'pred_name_open', 'pred_name_close1', 'pred_name_close2',
                       'fstar_open', 'fstar_close1', 'fstar_close2', 
                        'choice_proba_open','choice_proba_close1', 'choice_proba_close2',
                        'edge_open', 'edge_close1', 'edge_close2',
                        'ev_open', 'ev_close1', 'ev_close2']
    
    
    parlay_column_order = ['choice_fighter_name_open', 'parlay_odds_open', 'parlay_ev_open', 'stake_open', 
                        'choice_fighter_name_close1', 'parlay_odds_close1', 'parlay_ev_close1', 'stake_close1',
                        'choice_fighter_name_close2', 'parlay_odds_close2', 'parlay_ev_close2', 'stake_close2',
                        'parlay_prob_open', 'parlay_prob_close1', 'parlay_prob_close2']