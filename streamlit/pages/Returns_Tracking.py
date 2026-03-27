

import os
import sys 
from pathlib import Path 

import numpy as np 
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
BASE_DIR = Path(__file__).resolve().parents[1]


ml_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'ml_pct_returns.csv'
parlay_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'parlay_pct_returns.csv'

from Streamlit.utils import display_paginated_df, show_image

df_ml_pct = pd.read_csv(ml_pct_returns_fp)
df_parlay_pct = pd.read_csv(parlay_pct_returns_fp)


display_paginated_df(df_ml_pct, title='Money Line Percent Returns')
display_paginated_df(df_parlay_pct, title='Parlay Percent Returns')

path = BASE_DIR / 'Data' / 'plot_pngs' / 'pct_returns.png'
show_image(path, title='Percent Returns')
