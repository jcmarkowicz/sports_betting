import pandas as pd
import streamlit as st
from ufc_betting.config import settings
from utils import display_paginated_df, show_image

st.title("AI Input Features")
st.markdown(
    "Odds data: [BestFightOdds](https://www.bestfightodds.com/). "
    "Fight statistics: [UFCStats](http://www.ufcstats.com/statistics/events/completed)."
)
st.markdown(
    "Lasso fits logistic models using progressively stronger L1 regularization "
    "on the training split. Features whose coefficients persist longer receive "
    "higher ranks; ties share a rank. These ranks describe coefficient persistence, "
    "not causal effects, and depend on the penalty grid and correlated inputs."
)
output_dir = settings.feature_selection_dir
for filename, title in (
    ("lasso_paths.png", "Lasso Coefficient Paths"),
    ("lasso_importance.png", "Lasso Feature Importance"),
):
    path = output_dir / filename
    if path.is_file():
        show_image(path, title=title)
    else:
        st.info(f"Run lasso.ipynb to export {path}")

ranked_path = output_dir / "ranked_features.csv"
if ranked_path.is_file():
    ranked_df = pd.read_csv(ranked_path)
    if not ranked_df.empty:
        display_paginated_df(ranked_df, title="Ranked Input Features", key_prefix="lasso_ranked")
else:
    st.info(f"Run lasso.ipynb to export {ranked_path}")


st.subheader("Training Input Distributions")
input_plot = settings.plot_pngs_dir / "input_plots.png"
if input_plot.is_file():
    show_image(input_plot, title="Opening Model Input Features — Training Split")
else:
    st.info(f"Run input_plots.ipynb to export {input_plot}")

