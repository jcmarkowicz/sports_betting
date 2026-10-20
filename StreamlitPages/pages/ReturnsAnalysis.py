import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

from utils import display_paginated_df, show_image
from ufc_betting.config import settings
from ufc_betting.Results.returns import SETTLED_TYPES, prepare_return_plot_data
from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
    plot_bankroll_distributions,
    plot_event_accuracy_distributions,
)


# show_image(path, title='Percent Returns')
st.title(" Prediction and Betting Results since 2026-02-21")

plot_path = settings.bankroll_returns_plot_file
if plot_path.is_file():
    show_image(plot_path, title='Bankroll Returns: Moneyline + Parlay')
else:
    st.info('The bankroll plot has not been generated yet. Run the Results Tracking workflow.')
required_tables = [settings.ml_history_file, settings.parlay_history_file,
                   settings.bankroll_returns_file,
                   settings.betting_results_dir / 'prediction_accuracies.csv',
                   settings.betting_results_dir / 'bet_types.csv']
missing = [path.name for path in required_tables if not path.is_file()]
if missing:
    st.info('Results Tracking has not generated these tables yet: ' + ', '.join(missing))
    st.stop()

odds_type = st.selectbox('Odds type', SETTLED_TYPES)
events, moneylines, parlays = prepare_return_plot_data(
    pd.read_csv(settings.ml_history_file),
    pd.read_csv(settings.parlay_history_file),
    pd.read_csv(settings.bankroll_returns_file),
    odds_type,
)
if not events.empty:
    fig, _ = plot_bankroll_distributions(events)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info('No settled events to plot.')
if not moneylines.empty:
    fig, _, _ = plot_event_accuracy_distributions(moneylines, parlays, odds_type=odds_type)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info('No fights with valid market odds for accuracy distributions.')

df_accuracy = pd.read_csv(settings.data_dir / 'betting_results' / 'prediction_accuracies.csv')
df_bet_types = pd.read_csv(settings.data_dir / 'betting_results' / 'bet_types.csv')

if not df_accuracy.empty:
    display_paginated_df(df_accuracy, title='Returns Accuracy', key_prefix='returns')
if not df_bet_types.empty:
    display_paginated_df(df_bet_types, title='Bet Type Stats', key_prefix='bet_types')
