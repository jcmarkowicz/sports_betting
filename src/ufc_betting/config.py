import os 
from pathlib import Path
from dataclasses import dataclass, field

def find_project_root() -> Path:
    """Find the repository directory containing pyproject.toml."""
    start = Path(__file__).resolve().parent

    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise RuntimeError(
        "Could not find the project root containing pyproject.toml"
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "Data"

@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SPORTS_BETTING_DATA_DIR",
                str(PROJECT_ROOT / "Data"),
            )
        ).expanduser().resolve()
    )
    db_scrape_date: str = "2026-02-26"

    @property
    def stats_history_file(self) -> Path:
        return (
            self.data_dir
            / "scraped_data_main"
            / f"stats_history_{self.db_scrape_date}.csv"
        )

    @property
    def odds_history_file(self) -> Path:
        return (
            self.data_dir
            / "scraped_data_main"
            / f"odds_history_{self.db_scrape_date}.csv"
        )

    @property
    def betting_results_dir(self) -> Path:
        return self.data_dir / "betting_results"

    @property
    def ml_history_file(self) -> Path:
        return self.betting_results_dir / "moneyline_results.csv"

    @property
    def parlay_history_file(self) -> Path:
        return self.betting_results_dir / "parlay_results.csv"

    @property
    def ml_returns_file(self) -> Path:
        return self.betting_results_dir / "ml_returns.csv"

    @property
    def parlay_returns_file(self) -> Path:
        return self.betting_results_dir / "parlay_returns.csv"

    @property
    def bankroll_returns_file(self) -> Path:
        return self.betting_results_dir / "bankroll_returns.csv"

    @property
    def non_merged_features_dir(self) -> Path:
        return self.data_dir / "non_merged_features"

    @property
    def non_merged_stats_file(self) -> Path:
        return self.non_merged_features_dir / "non_merged_stats.csv"

    @property
    def non_merged_odds_file(self) -> Path:
        return self.non_merged_features_dir / "non_merged_odds.csv"

    @property
    def upcoming_events_dir(self) -> Path:
        return self.data_dir / "upcoming_events"

    @property
    def ml_bets_dir(self) -> Path:
        return self.upcoming_events_dir / "straight_bets"

    @property
    def parlay_bets_dir(self) -> Path:
        return self.upcoming_events_dir / "parlays"

    @property
    def event_features_dir(self) -> Path:
        return self.upcoming_events_dir / "event_features"

    @property
    def upcoming_events_dir(self) -> Path:
        return self.data_dir / "upcoming_events"

    @property
    def ml_bets_dir(self) -> Path:
        """Directory containing event-specific moneyline bet files."""
        return self.upcoming_events_dir / "straight_bets"

    @property
    def parlay_bets_dir(self) -> Path:
        """Directory containing event-specific parlay bet files."""
        return self.upcoming_events_dir / "parlays"

    @property
    def upcoming_event_features_dir(self) -> Path:
        """Directory containing feature files for upcoming events."""
        return self.upcoming_events_dir / "event_features"

    def ml_bets_file(self, event_date: str) -> Path:
        """Return the moneyline bets file for a YYYY-MM-DD date."""
        return self.ml_bets_dir / f"ml_all_{event_date}.csv"

    def parlay_bets_file(self, event_date: str) -> Path:
        """Return the parlay bets file for a YYYY-MM-DD date."""
        return self.parlay_bets_dir / f"parlay_all_{event_date}.csv"

    def event_file_path(self, date_string: str) -> Path:
        return (
            self.upcoming_event_features_dir
            / f"upcoming_odds_stats_{date_string}.csv"
        )
    
    def test_features_file(self, event_date: object) -> Path:
        return self.data_dir / f"test_features_{event_date}"


settings = Settings()


@dataclass
class config:
    
    # --- Bet Params ---
    bankroll:int = 3500

    # for opening and closing 
    mdd_ml = [.3, .4, .4] 
    mdd_parlay = [.5, .5, .5]

    N_parlay = [500, 250, 250]
    N_ml = [250, 250, 250]

    # just for closing odds 
    mdd_ml_stack = [.4, .4]
    mdd_parlay_stack = [.5, .5]

    N_ml_stack = [250, 250]
    N_parlay_stack = [1000, 1000]

    parlay_top_ev = 2  

    # --- Paths --- 
    
    db_scrape_date = r'2026-02-26'

    stats_history_file_string = settings.data_dir / "scraped_data_main" / f"stats_history_{db_scrape_date}.csv"
    odds_history_file_string = settings.data_dir/ "scraped_data_main" / f"odds_history_{db_scrape_date}.csv"

    upcoming_scraped_stats_string = settings.data_dir/ "upcoming_events" / "scraped_data" / "upcoming_stats"
    upcoming_scraped_odds_string = settings.data_dir /"upcoming_events" /"scraped_data" / "upcoming_odds"

    # folder for stats/odds FEATURES per event 
    upcoming_events_folder =  settings.data_dir/ "upcoming_events" / "event_features" 

    # folder for model 
    ml_bets_folder = settings.data_dir / "upcoming_events" / "straight_bets" 
    parlay_bets_folder = settings.data_dir/ "upcoming_events" / "parlays"

    model_open_path = settings.data_dir/ "saved_models" / "logit_model_open.pkl"
    model_close1_path = settings.data_dir/ "saved_models" / "logit_model_close1.pkl"
    model_close2_path = settings.data_dir / "saved_models" / "logit_model_close2.pkl"
    xgb_stack_path = settings.data_dir/ "saved_models" / "xgboost_stacked.pkl"

    scaler_open_path = settings.data_dir/ "saved_models" / "scaler_open.pkl"
    scaler_close1_path = settings.data_dir/ "saved_models" / "scaler_close1.pkl"
    scaler_close2_path = settings.data_dir/ "saved_models" / "scaler_close2.pkl"

    ml_folder = settings.data_dir/ "upcoming_events" / "straight_bets"
    parlay_folder = settings.data_dir/ "upcoming_events" / "parlays" 

    # --- Results Tracking Params ---
    ml_history_fp =settings.data_dir/ 'betting_results' / 'moneyline_results.csv'
    parlay_history_fp =settings.data_dir/ 'betting_results' / 'parlay_results.csv'
    event_features_folder = settings.data_dir/ "upcoming_events" / "event_features" 

    non_merged_stats_fp = settings.data_dir/ "non_merged_features" / "non_merged_stats.csv"
    non_merged_odds_fp = settings.data_dir/ "non_merged_features" / "non_merged_odds.csv"

    ml_returns_fp =settings.data_dir/ 'betting_results' / 'ml_returns.csv'
    parlay_returns_fp =settings.data_dir/ 'betting_results' / 'parlay_returns.csv'
    bankroll_returns_fp =settings.data_dir/ 'betting_results' / 'bankroll_returns.csv'

    # email params
    sender_email = "jcmarkufc@gmail.com"
    reciever_email = ['jcmarkowicz@outlook.com']
    columns_to_email = [
        'fighter_red', 'fighter_blue','pred_name_open',
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
    ml_column_order = ['fighter_red', 'fighter_blue', 
                       'open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue',
                       'stake_open', 'stake_close1_stack', 'stake_close2_stack',  
                       'pred_name_open', 'pred_name_close1_stack', 'pred_name_close2_stack',
                       'fstar_open', 'fstar_close1_stack', 'fstar_close2_stack', 
                       'choice_proba_open','choice_proba_close1_stack', 'choice_proba_close2_stack',
                       'edge_open', 'edge_close1_stack', 'edge_close2_stack',
                       'ev_open', 'ev_close1_stack', 'ev_close2_stack']
    
    
    parlay_column_order = ['choice_fighter_name_open', 'parlay_odds_open', 'parlay_ev_open', 'stake_open', 
                        'choice_fighter_name_close1_stack', 'parlay_odds_close1_stack', 'parlay_ev_close1_stack', 'stake_close1_stack',
                        'choice_fighter_name_close2_stack', 'parlay_odds_close2_stack', 'parlay_ev_close2_stack', 'stake_close2_stack',
                        'parlay_prob_open', 'parlay_prob_close1_stack', 'parlay_prob_close2_stack']
