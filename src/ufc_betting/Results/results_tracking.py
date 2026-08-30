
import pandas as pd 
from ufc_betting.DataPipeline.utils.github_utils import commit_if_changed, delete_and_commit
from ufc_betting.DataPipeline.utils.bets_utils import generate_bets
from ufc_betting.DataPipeline.dataset_workers.update_datasets import get_missing_stats

from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame
from ufc_betting.DataPipeline.dataframes.moneylines import MoneylineDataFrame

from ufc_betting.Results.returns import (
    accuracy_analysis,
    plot_returns,
    returns_by_date,
)

from ufc_betting.config import settings

#start_date='2026-02-23'
def archive_results(start_date='2026-08-28', delete_old=True):
    """
    Maintain moneyline and parlay history betting results. 

    Give a start date, scrape all stats of odds from now until / including that date. 
    For each event the following is calculated: 
        - Money line results
        - Parlay results
        - Missing odds stats that are not in the original history 
    
    Create and maintain the following 3 dataframes: 
        - df_ml_history: money line results history 
        - df_parlay_history: parlay results history 
        - df_missing_stats: missing stats and odds history 

    How to handle existing ml and parlay files: 
        - delete 
    """

    if settings.ml_history_file.exists():
        df_ml_history = pd.read_csv(settings.ml_history_file)
    else: 
         df_ml_history = pd.DataFrame()

    if settings.parlay_history_file.exists(): 
        df_parlay_history = pd.read_csv(settings.parlay_history_file)
    else: 
        df_parlay_history = pd.DataFrame()

    scraped_stats, scraped_odds, df_upcoming = get_missing_stats(start_date) 
    for date, event_group in df_upcoming.groupby('date'):

        event_group = event_group.copy().reset_index(drop=True)
        date_str = date.strftime("%Y-%m-%d")
        d_ts = pd.to_datetime(date_str)  

        # check if event date is in the past 
        if pd.Timestamp.now().normalize() > d_ts:

            df_ml, df_parlay = generate_bets(event_group, select_odds=None)

            generated_parlay = ParlayDataFrame.from_generated(
                frame=df_parlay,
                event_date=date_str,
            )
            settled_parlay = generated_parlay.with_results(event_group)
            df_parlay_results = settled_parlay.frame
            df_parlay_history = pd.concat([
                df_parlay_history, 
                df_parlay_results
            ], axis=0)

            generated_moneyline = MoneylineDataFrame.from_generated(
                frame = df_ml
            )
            settled_moneyline = generated_moneyline.with_results(event_group)
            df_moneyline_results = settled_moneyline.frame
            df_ml_history = pd.concat([
                df_ml_history, 
                df_moneyline_results
            ], axis=0)

            # all of these deleted files are regenerated in the above functions
            if delete_old: 
                delete_old_files(date_str)

    commit_if_changed(df_parlay_history, settings.parlay_history_file, f'Updating parlay results')
    commit_if_changed(df_ml_history, settings.ml_history_file, f'updating money line results')


def delete_old_files(date_str):
    """
    Delete ml bets, parlay bets, and event features for a given date.
    """
    ml_fp = settings.ml_bets_dir / f'ml_all_{date_str}.csv'
    delete_and_commit(ml_fp, f'Deleting Money Line bets for event {date_str}')

    parlay_fp = settings.parlay_bets_dir / f'parlay_all_{date_str}.csv'
    delete_and_commit(parlay_fp, f'Deleting parlay bets for event {date_str}')

    features_fp = settings.event_features_dir / f"upcoming_odds_stats_{date_str}.csv"
    delete_and_commit(features_fp, f'Deleting event features for event {date_str}')



if __name__ == "__main__":
    archive_results()
    bankroll_results = returns_by_date()

    ml_results = MoneylineDataFrame(
        pd.read_csv(settings.ml_history_file)
    ).frame

    parlay_results = ParlayDataFrame(
        pd.read_csv(settings.parlay_history_file)
    ).frame

    plot_returns(
        ml_results=ml_results,
        parlay_results=parlay_results,
        bankroll_results=bankroll_results,
    )

    accuracy_results, bet_type_results = accuracy_analysis(
        ml_results=ml_results,
        parlay_results=parlay_results,
    )

    commit_if_changed(
        accuracy_results, 
        settings.data_dir / 'betting_results' / 'prediction_accuracies.csv', 
        f'Updating Bets Accuracies'
    )

    commit_if_changed(
        bet_type_results, 
        settings.data_dir / 'betting_results' / 'bet_types.csv',
        f'Updating Bet Type Stats'
    )

