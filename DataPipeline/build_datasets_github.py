import numpy as np 
import pandas as pd 
import statsmodels.api as sm

import subprocess

import os
import sys 
import joblib
from pathlib import Path
from datetime import datetime 

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from DataPipeline.webscrapers.scraping_pipeline import UFC_Webscraper
from ufc_upcoming_analysis.betting_strategy import betting_pipeline, seperate_bets_dfs

BASE_DIR = Path(__file__).resolve().parents[1]

get_all_stats = False
get_all_odds = False

get_next_fight_stats_odds = True

get_missing_stats = False
generate_model_df = True 

recent_date = r'2026-02-26'
stats_history_file_string = BASE_DIR / "Data" / "history" / f"stats_history_{recent_date}.csv"
odds_history_file_string = BASE_DIR / "Data" / "history" / f"odds_history_{recent_date}.csv"

upcoming_stats_string = BASE_DIR / "Data/upcoming_events/upcoming_stats"
upcoming_odds_string = BASE_DIR / "Data/upcoming_events/upcoming_odds"

# find the most recent stats and odds history by most recent date 
# recent date from last merger with missing stats
# from /HISTORY folder 
date_today = datetime.now().strftime("%Y-%m-%d") #use this to mark odds 

next_fight_date = r'2026-03-14'

stats_history = pd.read_csv(stats_history_file_string) # frames BEFORE any feature engineering 
stats_history = stats_history.drop(columns=[col for col in stats_history.columns if "Unnamed" in col])

num_duplicate = stats_history.duplicated(
    subset=['fighter_red', 'fighter_blue', 'event_date']
).sum()


stats_history = stats_history[~stats_history.duplicated(
    subset=['fighter_red', 'fighter_blue', 'event_date'],
    keep='first'
)]

odds_history = pd.read_csv(odds_history_file_string)
odds_history = odds_history.drop(columns=[col for col in odds_history.columns if "Unnamed" in col])
next_event_folder = BASE_DIR / "Data/upcoming_events"

# Columns you want to send in email
columns_to_email = ['fighter_red', 'fighter_blue','pred_name_open','open_red','open_blue','pred_winner_open',
                    'fstar_open','stake_open'
                    ]  # adjust to your DataFrame

model_open = sm.load(BASE_DIR / "Data" / "saved_models" / "logit_model_open.pkl")
model_close1 = sm.load(BASE_DIR / "Data" / "saved_models" / "logit_model_close1.pkl")
model_close2 = sm.load( BASE_DIR / "Data" / "saved_models" / "logit_model_close2.pkl")

scaler_open = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_open.pkl")
scaler_close1 = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_close1.pkl")
scaler_close2 = joblib.load(BASE_DIR / "Data" / "saved_models" / "scaler_close2.pkl")

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

feats_list = [open_feats, close1_feats, close2_feats]
model_list = [model_open, model_close1, model_close2]
scaler_list = [scaler_open, scaler_close1, scaler_close2]

type_list = ['open', 'close1', 'close2']
fair_odds_list = [['dec_fair_open_blue', 'dec_fair_open_red'], ['dec_fair_close1_blue', 'dec_fair_close1_red'], ['dec_fair_close2_blue', 'dec_fair_close2_red']]
real_odds_list = [['dec_open_blue', 'dec_open_red'], ['dec_close1_blue', 'dec_close1_red'], ['dec_close2_blue', 'dec_close2_red'] ]


# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_FROM = "jcmarkufc@gmail.com"  # Gmail sender

EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_TO = ["jcmarkowicz@outlook.com"] #, 'jimmymarkowicz28@gmail.com','jasonszat@gmail.com']


def email_bets(df_, date):
    df = df_.copy()

    df['math_red'] = df['math_red'].astype('category')
    df['math_blue'] = df['math_blue'].astype('category')
    df['elo_pred'] = df['elo_pred'].astype('category')

    df_bets_all, df_parlay_all = betting_pipeline(df, 
                                                feats_list=feats_list, model_list=model_list, scaler_list=scaler_list, type_list=type_list,
                                                fair_odds_list=fair_odds_list, real_odds_list=real_odds_list, 
                                                bankroll=500, max_drawdown=0.25, N=2000)
    
    df_bets_all[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']] = df[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']]
    df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
    df_bets = df_bets_arr[0]
    df_parlay = df_parlay_arr[0]

    msg = MIMEMultipart()
    msg["Subject"] = f"Betting Report {date}"
    msg["From"] = EMAIL_FROM
    # Join the list into a comma-separated string for the header
    msg["To"] = ", ".join(EMAIL_TO)


    # ---- Body with HTML link ----
    html_body = """
    <p>See attached newly announced fights.</p>
    <p>Click here for all upcoming picks: <a href="https://sportsbetting-cn2kwvhykyrxdw2gxmuifl.streamlit.app/upcoming_picks">Dashboard</a></p>
    """
    msg.attach(MIMEText(html_body, "html"))


    # ---- Straight Bets CSV ----
    bets_csv = df_bets[columns_to_email].to_csv(index=False)

    part = MIMEBase("application", "octet-stream")
    part.set_payload(bets_csv.encode())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        "attachment; filename=straight_bets.csv",
    )
    msg.attach(part)

    # ---- Parlay CSV ----
    parlay_columns = [
        'choice_fighter_name_open',
        'parlay_fstar_open',
        'parlay_odds_open',
        'stake_open'
    ]

    parlay_csv = df_parlay[parlay_columns].to_csv(index=False)

    part2 = MIMEBase("application", "octet-stream")
    part2.set_payload(parlay_csv.encode())
    encoders.encode_base64(part2)
    part2.add_header(
        "Content-Disposition",
        "attachment; filename=parlay_bets.csv",
    )
    msg.attach(part2)

    # ---- Send email ----
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        # Pass the list of recipients to send_message
        server.send_message(msg, from_addr=EMAIL_FROM, to_addrs=EMAIL_TO)

# Function to commit if changed
def commit_if_changed(file_path, msg, branch="main"):

    repo = os.environ["GITHUB_REPOSITORY"]  # e.g., 'username/repo'
    token = os.environ["GITHUB_TOKEN"]      # Provided automatically in Actions

    # Stage file
    subprocess.run(["git", "add", str(file_path)], check=True)

    # Check if staged changes exist
    diff_check = subprocess.run(
        ["git", "diff", "--cached", "--quiet", str(file_path)]
    )

    if diff_check.returncode != 0:
        # Configure git
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", msg], check=True)

        # Push using token authentication
        push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
        subprocess.run(["git", "push", push_url, f"HEAD:{branch}"], check=True)



if __name__ == "__main__":
    scraper = UFC_Webscraper()
    features = FeatureEngineering()

    # scrape all stats
    if get_all_stats is True: 
        df_entire_stats_history = scraper.scrape_until(date=None)
        df_entire_stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv')
        stats_history = df_entire_stats_history
    
    # scrape all odds based on stats history 
    if get_all_odds is True: 
        df_entire_odds_history = scraper.get_fighter_odds(stats_history)
        df_entire_odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')
        odds_history = df_entire_odds_history

    # get upcoming fight stats
    # can run withoug getting missing stats, returns dataframes in upcoming folder 
    if get_next_fight_stats_odds is True: 

        # scrape upcoming events, MULTIPLE DATES handled in generate_model_df  
        next_fight_stats = scraper.scrape_upcoming_card()
        next_fight_date = next_fight_stats['event_date'][0] # unique date returned here, only one fightcard is scraped 
        
        # save next stats 
        next_stats_path = f'{upcoming_stats_string}_{next_fight_date}.csv'
        next_fight_stats.to_csv(next_stats_path, index=False)
        next_fight_stats = pd.read_csv(next_stats_path) # read back in to ensure same formatting for odds function

        # save next_odds 
        next_odds_path = f'{upcoming_odds_string}_{next_fight_date}.csv' # upcoming odds for the next fight card
        next_odds_df = scraper.get_fighter_odds(next_fight_stats)
        next_odds_df.to_csv(next_odds_path, index=False)

    # merge stats history scrape_until, merge odds_history with missing odds by fighter/date 
    if get_missing_stats:

        # scrape stats from current date to prev_fight_date (last date in stats_history) 
        prev_fight_date = stats_history.sort_values(by='event_date', ascending=False, inplace=False).reset_index(drop=True)['event_date'][0] # get index 0, largest value in non ascending order  
        print(f"Previous fight date in stats history: {prev_fight_date}")

        missing_stats = scraper.scrape_until(prev_fight_date)
        missing_odds = scraper.get_fighter_odds(missing_stats) # finds odds based on fighters/date
    
        # merge with stats history, save by current date
        stats_history = pd.concat([stats_history, missing_stats], axis=0)
        stats_history['event_date'] = pd.to_datetime(stats_history['event_date'], errors='coerce')
        
        stats_history = stats_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        stats_history.to_csv(f'{stats_history_file_string}_{date_today}.csv')

        odds_history = pd.concat([odds_history, missing_odds], axis=0)
        odds_history['event_date'] = pd.to_datetime(odds_history['event_date'], errors='coerce')

        odds_history = odds_history.sort_values(by='event_date', ascending=True).reset_index(drop=True)
        odds_history.to_csv(f'{odds_history_file_string}_{date_today}.csv')

    # Take stats_history, odds_history, next_fight_stats, next_fight_odds
    # Build dfs with all features used for model training 
    if generate_model_df is True: 

        next_stats_df = pd.read_csv(f'{upcoming_stats_string}_{next_fight_date}.csv')
        next_odds_df = pd.read_csv(f'{upcoming_odds_string}_{next_fight_date}.csv')

        # odds stats is for events that already happened, upcoming df for future events
        odds_stats_df, upcoming_df = features.build_all_stats(stats_history, next_stats_df, odds_history, next_odds_df)
        odds_stats_df.to_csv(BASE_DIR / f'Data/training_data/entire_odds_stats_{date_today}.csv', index=False)
      
        upcoming_groups = upcoming_df.groupby('date')
        for date, group in upcoming_groups:

            date_str = date.strftime("%Y-%m-%d")   # or "%Y%m%d"
            file_path = next_event_folder / f"event_dfs/upcoming_odds_stats_{date_str}.csv"
            print(date_str)

            # group includes all event fights 
            group = group.reset_index(drop=True)

            if (group[['open_red', 'open_blue']].isna().sum() == group.shape[0]).all():
                pass

            elif not file_path.exists():
                email_bets(group, date_str)
            
            else:

                # check if new data found  
                existing_df = pd.read_csv(file_path)
                
                mask_na_update = (
                    existing_df[['open_red','open_blue']].isna() & group[['open_red','open_blue']].notna()
                ).any(axis=1)

                sub_group = group[mask_na_update]

                if not sub_group.empty:
                    email_bets(sub_group, date_str)
            
            # save after above else
            group.to_csv(file_path, index=False)
            commit_if_changed(file_path, f'Updating Features for {date_str}')

            group['math_red'] = group['math_red'].astype('category')
            group['math_blue'] = group['math_blue'].astype('category')
            group['elo_pred'] = group['elo_pred'].astype('category')

            df_bets_all, df_parlay_all = betting_pipeline(group, 
                                                        feats_list=feats_list, model_list=model_list, scaler_list=scaler_list, type_list=type_list,
                                                        fair_odds_list=fair_odds_list, real_odds_list=real_odds_list, 
                                                        bankroll=500, max_drawdown=0.25, N=2000)
            
            df_bets_all[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']] = group[['open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']]
            df_bets_arr, df_parlay_arr = seperate_bets_dfs(df_bets_all, df_parlay_all, type_list)
  
            # bets_fp = BASE_DIR / f"/Data/upcoming_events/straight_bets/open_odds_{date_str}"
            # palray_fp = BASE_DIR / f"Data/upcoming_events/parlays/open_odds_{date_str}"

            straight_path = BASE_DIR / f'Data/upcoming_events/straight_bets/ml_all_{date_str}.csv'
            parlay_path = BASE_DIR / f'Data/upcoming_events/parlays/parlay_all_{date_str}.csv'

            df_bets_all.to_csv(straight_path, index=False)
            df_parlay_all.to_csv(parlay_path, index=False)
            
            commit_if_changed(straight_path, f'Update Bets Open for {date_str}')
            commit_if_changed(parlay_path, f'Update Parlay Open for {date_str}')
