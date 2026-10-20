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
# Support deployments that still have an older Settings module loaded.
output_dir = getattr(settings, "feature_selection_dir", None)
if output_dir is None:
    output_dir = settings.data_dir / "feature_selection"
path = output_dir / "lasso_paths.png"
if path.is_file():
    show_image(path, title="Lasso Coefficient Paths")
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
plot_dir = getattr(settings, "plot_pngs_dir", None)
if plot_dir is None:
    images_dir = getattr(settings, "images_dir", settings.data_dir / "images")
    plot_dir = images_dir / "plot_pngs"
input_plot = plot_dir / "input_plots.png"
if input_plot.is_file():
    show_image(input_plot, title="Opening Model Input Features — Training Split")
else:
    st.info(f"Run input_plots.ipynb to export {input_plot}")

