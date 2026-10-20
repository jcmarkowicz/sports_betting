import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import pandas as pd 
import streamlit as st

import os 
from datetime import datetime
from zoneinfo import ZoneInfo

from utils import display_paginated_df

from ufc_betting.config import config, settings
from ufc_betting.UpcomingPicks.betting_pipeline import build_event_ev_table
###

def extract_date(filename):
    date_str = filename.split('_')[-1].replace('.csv', '')
    return datetime.strptime(date_str, "%Y-%m-%d")


ml_files = sorted(os.listdir(settings.ml_bets_dir), key=extract_date)
parlay_files = sorted(os.listdir(settings.parlay_bets_dir), key=extract_date)

assert len(ml_files) == len(parlay_files), f"Mismatch in number of ML and Parlay files: {len(ml_files)} vs {len(parlay_files)}"

for ml_file, parlay_file in zip(ml_files, parlay_files):
    today = datetime.now(ZoneInfo("America/New_York")).date()
    date_str = ml_file.split('_')[-1].replace('.csv', '')
    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if today > event_date:
        continue

    # ----- ML TABLE -----
    table_name = f"Money Line {date_str}"

    path = os.path.join(settings.ml_bets_dir, ml_file)
    moneyline_table = pd.read_csv(path)
    table_select = moneyline_table[config.ml_column_order]

    display_paginated_df(table_select, title=table_name, key_prefix=f"ml_{ml_file}")


    # ----- PARLAY TABLE -----
    table_name = f"Parlay {date_str}"

    path = os.path.join(settings.parlay_bets_dir, parlay_file)
    parlay_table = pd.read_csv(path)
    table_select = parlay_table[config.parlay_column_order]

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
