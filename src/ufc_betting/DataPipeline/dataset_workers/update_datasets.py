
import numpy as np
import pandas as pd 

from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from ufc_betting.DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper
from ufc_betting.DataPipeline.utils.github_utils import commit_if_changed
from ufc_betting.DataPipeline.dataframes.stats import StatsRepository
from ufc_betting.DataPipeline.dataframes.odds import OddsRepository

from ufc_betting.config import settings


def get_missing_stats(prev_fight_date): 
    """ previous fight date from file string """

    scraper = UFC_Webscraper()
    features = FeatureEngineering()
    stats_repo = StatsRepository(
        stats_history_file=settings.stats_history_file, non_merged_stats_file=settings.non_merged_stats_file
    )
    odds_repo = OddsRepository(
        odds_history_file=settings.odds_history_file, non_merged_odds_file=settings.non_merged_odds_file
    )

    missing_stats = scraper.scrape_until(prev_fight_date)
    missing_odds = scraper.get_fighter_odds(missing_stats) 

    stats_missing_all = stats_repo.merge_missing(missing_stats)
    commit_if_changed(
        stats_missing_all, 
        settings.non_merged_stats_file, 
        f'Updating Non Merged Stats for fight date: {prev_fight_date}'
    )

    odds_missing_all = odds_repo.merge_missing(missing_odds)
    commit_if_changed(
        odds_missing_all, 
        settings.non_merged_odds_file, 
        f'Updating Non Merged Odds for fight date: {prev_fight_date}'
    )

    all_odds = odds_repo.combine_with_history(odds_missing_all)
    all_stats = stats_repo.combine_with_history(stats_missing_all)

    odds_stats_df, upcoming_df = features.build_all_stats(all_stats, missing_stats.iloc[:5], all_odds, missing_odds.iloc[:5], ignore_upcoming=True)

    return missing_stats, missing_odds, odds_stats_df[odds_stats_df['date'] >= prev_fight_date]