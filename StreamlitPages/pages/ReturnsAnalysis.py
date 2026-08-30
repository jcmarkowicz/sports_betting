import streamlit as st

import numpy as np 
import pandas as pd 
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from StreamlitPages.utils import display_paginated_df, show_image
from ufc_betting.config import config

BASE_DIR = config.base_dir  



# show_image(path, title='Percent Returns')
st.title(" Prediction and Betting Results since 2026-02-21")


show_image(path, title='Bankroll Returns: Moneyline + Parlay')

show_image(path, title='Odds Returns: Moneyline and Parlay')

ml_results = pd.read_csv(config.ml_returns_fp)
parlay_results = pd.read_csv(config.parlay_returns_fp)
bankroll_results = pd.read_csv(config.bankroll_returns_fp)

plot_returns(ml_results, parlay_results, bankroll_results)

df_accuracy, df_bet_types = accuracy_analysis(ml_results, parlay_results)
display_paginated_df(df_accuracy, title='Returns Accuracy', key_prefix='returns')
display_paginated_df(df_bet_types, title='Bet Type Stats', key_prefix='')
