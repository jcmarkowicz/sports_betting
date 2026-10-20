import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import pandas as pd 
import streamlit as st

import os 
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from utils import display_paginated_df

from ufc_betting.config import config, settings
from ufc_betting.UpcomingPicks.betting_pipeline import build_event_ev_table
###

def extract_date(filename):
    date_str = filename.split('_')[-1].replace('.csv', '')
    return datetime.strptime(date_str, "%Y-%m-%d")


def available_picks_dir(configured_path, folder):
    configured_path = Path(configured_path)
    fallback = settings.data_dir / "bets" / folder
    return configured_path if configured_path.is_dir() else fallback


def files_by_event(folder, pattern):
    files = {}
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        try:
            event_date = extract_date(path.name).date()
        except ValueError:
            continue
        files[event_date] = path
    return files


ml_dir = available_picks_dir(settings.ml_bets_dir, "straight_bets")
parlay_dir = available_picks_dir(settings.parlay_bets_dir, "parlays")
for folder in (ml_dir, parlay_dir):
    if not folder.is_dir():
        st.info(f"Upcoming picks folder is unavailable: {folder}. Publish the generated picks CSVs to this deployment.")
        st.stop()

ml_files = files_by_event(ml_dir, "ml_all_*.csv")
parlay_files = files_by_event(parlay_dir, "parlay_all_*.csv")
today = datetime.now(ZoneInfo("America/New_York")).date()
upcoming_dates = {date for date in ml_files.keys() | parlay_files.keys() if date >= today}
if not upcoming_dates:
    st.info("No upcoming picks have been published yet.")
    st.stop()

for event_date in sorted(upcoming_dates):
    date_str = event_date.isoformat()
    if event_date not in ml_files or event_date not in parlay_files:
        st.info(f"Picks for {date_str} are incomplete: both moneyline and parlay CSVs are required.")
        continue
    ml_file = ml_files[event_date].name
    parlay_file = parlay_files[event_date].name

    # ----- ML TABLE -----
    table_name = f"Money Line {date_str}"

    path = ml_files[event_date]
    moneyline_table = pd.read_csv(path)
    table_select = moneyline_table[config.ml_column_order]

    if not table_select.empty:
        display_paginated_df(table_select, title=table_name, key_prefix=f"ml_{ml_file}")


    # ----- PARLAY TABLE -----
    table_name = f"Parlay {date_str}"

    path = parlay_files[event_date]
    parlay_table = pd.read_csv(path)
    table_select = parlay_table[config.parlay_column_order]

    if not table_select.empty:
        display_paginated_df(table_select, title=table_name, key_prefix=f"parlay_{parlay_file}")

    # ----- EVENT EV TABLE -----
    event_ev = build_event_ev_table(
        moneyline_bets=moneyline_table,
        parlay_bets=parlay_table,
    ).drop(columns="date")
    event_ev = event_ev.rename(columns={
        "bet_type": "Bet Type",
        "open_ev_pct": "Open EV (% bankroll)",
        "close1_ev_pct": "Close1 EV (% bankroll)",
        "close2_ev_pct": "Close2 EV (% bankroll)",
        "close1_pct_opening_ev": "Close1 EV (% of open)",
        "close2_pct_opening_ev": "Close2 EV (% of open)",
    }).round(2)
    st.subheader(f"Event EV {date_str}")
    st.dataframe(event_ev, use_container_width=True, hide_index=True)
