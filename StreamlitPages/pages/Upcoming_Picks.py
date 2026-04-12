
import pandas as pd 

import os 
from datetime import datetime

from StreamlitPages.utils import display_paginated_df

from config import config
###

def extract_date(filename):
    date_str = filename.split('_')[-1].replace('.csv', '')
    return datetime.strptime(date_str, "%Y-%m-%d")


ml_files = sorted(os.listdir(config.ml_folder), key=extract_date)
parlay_files = sorted(os.listdir(config.parlay_folder), key=extract_date)

assert len(ml_files) == len(parlay_files), "Mismatch in number of ML and Parlay files"

for ml_file, parlay_file in zip(ml_files, parlay_files):
    today = datetime.today().date()
    date_str = ml_file.split('_')[-1].replace('.csv','')
    event_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if today > event_date:
        continue

    # ----- ML TABLE -----
    table_name = f"Money Line {date_str}"

    path = os.path.join(config.ml_folder, ml_file)
    table = pd.read_csv(path)
    table_select = table[config.ml_column_order]

    display_paginated_df(table_select, title=table_name, key_prefix=f"ml_{ml_file}")


    # ----- PARLAY TABLE -----
    table_name = f"Parlay {date_str}"

    path = os.path.join(config.parlay_folder, parlay_file)
    table = pd.read_csv(path)
    table_select = table[config.parlay_column_order]

    display_paginated_df(table_select, title=table_name, key_prefix=f"parlay_{parlay_file}")
