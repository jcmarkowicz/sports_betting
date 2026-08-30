import streamlit as st
import pandas as pd 

from utils import display_paginated_df, show_image
from ufc_betting.config import config, settings


# show_image(path, title='Percent Returns')
st.title(" Prediction and Betting Results since 2026-02-21")

show_image(
    settings.data_dir / 'plot_pngs' / 'bankroll_returns_live.png'
    , title='Bankroll Returns: Moneyline + Parlay'
)
show_image(
    settings.data_dir / 'plot_pngs' / 'avg_returns_live.png',
    title='Odds Returns: Moneyline and Parlay'
)

df_accuracy = pd.read_csv(settings.data_dir / 'betting_results' / 'prediction_accuracies.csv')
df_bet_types = pd.read_csv(settings.data_dir / 'betting_results' / 'bet_types.csv')

display_paginated_df(df_accuracy, title='Returns Accuracy', key_prefix='returns')
display_paginated_df(df_bet_types, title='Bet Type Stats', key_prefix='')
