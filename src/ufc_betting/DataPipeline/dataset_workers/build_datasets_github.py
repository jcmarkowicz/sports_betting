import os
import pandas as pd 
from datetime import datetime 

from ufc_betting.DataPipeline.dataframes.odds import OddsRepository
from ufc_betting.DataPipeline.dataframes.stats import StatsRepository
from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import (
    FeatureEngineering,
)
from ufc_betting.DataPipeline.utils.bets_utils import generate_bets
from ufc_betting.DataPipeline.utils.email_utils import email_bets
from ufc_betting.DataPipeline.utils.github_utils import commit_if_changed
from ufc_betting.DataPipeline.webscrapers.scraping_pipeline import (
    UFC_Webscraper,
)
from ufc_betting.config import config, settings

history_dir = settings.data_dir / "scraped" / "historic"
history_dir.mkdir(parents=True, exist_ok=True)

# find the most recent stats and odds history with most recent timestamp 
# recent date from last merger with missing stats
# from Data/HISTORY folder 
date_today = datetime.now().strftime("%Y-%m-%d") 

def env_bool(name, default):
    """Read a boolean workflow flag without treating the string 'false' as true."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false")


get_all_stats = env_bool("GET_ALL_STATS", False)
get_all_odds = env_bool("GET_ALL_ODDS", False)
generate_historical_feats = env_bool("GENERATE_HISTORICAL_FEATS", False)
get_next_fight_stats_odds = env_bool("GET_NEXT_FIGHT_STATS_ODDS", True)
generate_model_df = env_bool("GENERATE_MODEL_DF", True)


stats_repo = StatsRepository(
    stats_history_file=settings.stats_history_file,
    non_merged_stats_file=settings.non_merged_stats_file,
)

odds_repo = OddsRepository(
    odds_history_file=settings.odds_history_file,
    non_merged_odds_file=settings.non_merged_odds_file,
)

stats_history = None if get_all_stats else stats_repo.load_combined()
odds_history = None if get_all_odds else odds_repo.load_combined()


if __name__ == "__main__":
    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    # scrape all stats
    if get_all_stats:
        stats_history = scraper.scrape_until(date=None)
        stats_history.to_csv(
            history_dir / f"stats_history_{date_today}.csv",
            index=False,
        )

    if get_all_odds:
        odds_history = scraper.get_fighter_odds(stats_history)
        odds_history.to_csv(
            history_dir / f"odds_history_{date_today}.csv",
            index=False,
        )

    if generate_historical_feats: 
        all_feats, _ = features.build_all_stats(
            stats_df=stats_history,
            upcoming_stats=None,
            odds_df=odds_history,
            upcoming_odds=None,
            ignore_upcoming=True,
        )

        output_dir = settings.data_dir / "train_val_test" / "all_feats"
        output_dir.mkdir(parents=True, exist_ok=True)

        all_feats.to_csv(
            output_dir / f"entire_odds_stats_{date_today}.csv",
            index=False,
        )


    # get upcoming fight stats
    # can run withoug getting missing stats, returns dataframes in upcoming folder 
    if get_next_fight_stats_odds is True: 

        # scrape upcoming events, MULTIPLE DATES handled in generate_model_df  
        next_fight_stats = scraper.scrape_upcoming_card()
        next_fight_date = next_fight_stats['event_date'][0] # unique date returned here, only one fightcard is scraped 
        
        # save next stats 
        commit_if_changed(
            next_fight_stats, 
            f'{config.upcoming_scraped_stats_string}.csv', 
            f'Updating upcoming scraped stats starting at: {next_fight_date}'
        )

        # save next_odds 
        next_odds_df = scraper.get_fighter_odds(next_fight_stats)
        commit_if_changed(
            next_odds_df, 
            f'{config.upcoming_scraped_odds_string}.csv' , 
            f'Updating upcoming scraped odds starting at: {next_fight_date}'
        )

    # Build features for history and upcoming 
    if generate_model_df is True: 

        # get scraped data for ALL future events 
        next_stats_df = pd.read_csv(f'{config.upcoming_scraped_stats_string}.csv')
        next_odds_df = pd.read_csv(f'{config.upcoming_scraped_odds_string}.csv')

        # FEATURES DATAFRAMES 
        odds_stats_df, upcoming_df = features.build_all_stats(stats_history, next_stats_df, odds_history, next_odds_df)

        upcoming_groups = upcoming_df.groupby('date')
        for date, group in upcoming_groups:

            # group includes all event fights 
            group = group.reset_index(drop=True)

            date_str = date.strftime("%Y-%m-%d")   # or "%Y%m%d"
            event_file_path = settings.event_file_path(date_str)

            # no odds available
            if group[['open_red', 'open_blue']].isna().all().all():
                pass

            # new event found 
            elif not event_file_path.exists():
                email_bets(group, date_str)
            
            # else check if different from existing
            else:
                # existing open will be nan if not available 
                existing_df = pd.read_csv(event_file_path)

                aligned = group.merge(
                    existing_df[["fighter_red", "fighter_blue", "open_red", "open_blue"]],
                    on=["fighter_red", "fighter_blue"],
                    how="left"
                )
                mask_na_update = (
                    (aligned['open_red_y'].isna() & aligned['open_red_x'].notna()) |
                    (aligned['open_blue_y'].isna() & aligned['open_blue_x'].notna())
                )
                sub_group = group[mask_na_update]
                if not sub_group.empty:
                    email_bets(sub_group, date_str)
            
            # commit event csv data every time 
            commit_if_changed(group, event_file_path, f'Updating Features for {date_str}')

            # generate betting picks with latest data 
            df_bets_all, df_parlay_all = generate_bets(group, select_odds=None)

            # commit csv 
            commit_if_changed(
                df_bets_all, settings.ml_bets_file(date_str), f'Update Bets for {date_str}'
            )
            commit_if_changed(
                df_parlay_all, settings.parlay_bets_file(date_str), f'Update Parlay for {date_str}'
            )
