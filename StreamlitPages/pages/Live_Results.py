"""Test simulations using the same inputs and plots as test_results.ipynb."""
import streamlit as st
import pandas as pd

from utils import display_paginated_df, show_image
from ufc_betting.config import settings
st.title("Live Results")
st.caption("Test dataset simulations, matching test_results.ipynb: opening model, $500 initial bankroll, decay enabled for close1 and close2 only.")
market = st.selectbox("Odds type", ("open", "close1", "close2"))
plot_dir = settings.data_dir / "images" / "plot_pngs"
for suffix, title in (
    ("backtest", "Bankroll and cumulative results"),
    ("bankroll_distributions", "Bankroll return distributions"),
    ("event_accuracy", "Event accuracy distributions"),
):
    path = plot_dir / f"test_{market}_{suffix}.png"
    if path.is_file():
        show_image(path, title=title)
    else:
        st.info(f"Saved plot is unavailable: {path.name}. Run Results Tracking to generate and publish it.")

st.caption("Prediction accuracies below are the saved Results Tracking summary; its coverage may differ from the frozen test simulation above.")
accuracy_paths = (
    settings.data_dir / "betting_results" / "prediction_accuracies.csv",
    settings.data_dir / "bets" / "betting_results" / "prediction_accuracies.csv",
)
accuracy_path = next((path for path in accuracy_paths if path.is_file()), None)
if accuracy_path is None:
    st.info("Prediction accuracies have not been published. Run Results Tracking to generate prediction_accuracies.csv.")
else:
    try:
        accuracy = pd.read_csv(accuracy_path)
        if accuracy.empty:
            st.info("The saved prediction accuracies table is empty.")
        else:
            display_paginated_df(accuracy, title="Prediction Accuracies", key_prefix="live_prediction_accuracies")
    except (OSError, ValueError) as error:
        st.error(f"Could not read prediction accuracies: {error}")
