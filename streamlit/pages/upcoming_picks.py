
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import os 
import sys 
from datetime import datetime

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils import display_paginated_df

###

ml_folder = BASE_DIR / "Data" / "upcoming_events" / "straight_bets"
parlay_folder = BASE_DIR / "Data" / "upcoming_events" / "parlays" 

ml_column_order = ['fighter_red', 'fighter_blue', 'open_red', 'open_blue', 'pred_name_open', 'fstar_open', 'stake_open',
                   'close1_red', 'close1_blue', 'pred_name_close1', 'fstar_close1', 'stake_close1', 
                   'close2_red', 'close2_blue', 'pred_name_close2', 'fstar_close2', 'stake_close2']

parlay_column_order = ['choice_fighter_name_open', 'parlay_odds_open', 'parlay_ev_open', 'stake_open', 
                       'choice_fighter_name_close1', 'parlay_odds_close1', 'parlay_ev_close1', 'stake_close1',
                       'choice_fighter_name_close2', 'parlay_odds_close2', 'parlay_ev_close2', 'stake_close2']

def extract_date(filename):
    date_str = filename.split('_')[-1].replace('.csv', '')
    return datetime.strptime(date_str, "%Y-%m-%d")

ml_files = sorted(os.listdir(ml_folder), key=extract_date)
parlay_files = sorted(os.listdir(parlay_folder), key=extract_date)

assert len(ml_files) == len(parlay_files), "Mismatch in number of ML and Parlay files"

for ml_file, parlay_file in zip(ml_files, parlay_files):
    today = datetime.today().date()
    date_str = ml_file.split('_')[-1].replace('.csv','')
    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if today > event_date:
        continue

    # ----- ML TABLE -----
    table_name = f"Money Line {date_str}"

    path = os.path.join(ml_folder, ml_file)
    table = pd.read_csv(path)
    table_select = table[ml_column_order]

    display_paginated_df(table_select, title=table_name, key_prefix=f"ml_{ml_file}")


    # ----- PARLAY TABLE -----
    table_name = f"Parlay {date_str}"

    path = os.path.join(parlay_folder, parlay_file)
    table = pd.read_csv(path)
    table_select = table[parlay_column_order]

    display_paginated_df(table_select, title=table_name, key_prefix=f"parlay_{parlay_file}")
