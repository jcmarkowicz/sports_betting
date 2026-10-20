"""Display the validation reports exported by vegas_accuracy.ipynb."""
import pandas as pd
import streamlit as st

from ufc_betting.config import settings
from utils import display_paginated_df, show_image


st.title("Vegas vs Model Predictions")
st.caption(
    "Saved validation results from vegas_accuracy.ipynb. "
    "The opening model is evaluated using opening, close1 and close2 odds."
)
plot_dir = getattr(settings, "plot_pngs_dir", None)
if plot_dir is None:
    images_dir = getattr(settings, "images_dir", settings.data_dir / "images")
    plot_dir = images_dir / "plot_pngs"

for tab, market in zip(st.tabs(["Opening odds", "Close1 odds", "Close2 odds"]),
                       ("open", "close1", "close2")):
    with tab:
        for filename, title in (
            (f"val_accuracy_{market}_vegas.png", "Event Accuracy, F1 and Brier Scores"),
            (f"val_accuracy_regression_{market}_vegas.png", "Model Accuracy vs Vegas Accuracy Regression"),
        ):
            path = plot_dir / filename
            if path.is_file():
                show_image(path, title=title)
            else:
                st.info(f"Saved plot is unavailable: {path}. Export it from vegas_accuracy.ipynb.")

summary_path = settings.data_dir / "backtest_results" / "regression_coefficients_open_vegas.csv"
if summary_path.is_file():
    try:
        summary = pd.read_csv(summary_path, keep_default_na=False)
        if summary.empty:
            st.info("The saved regression summary table is empty.")
        else:
            for column in summary.select_dtypes(include="number").columns:
                summary[column] = summary[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
            st.caption("Coefficient significance: *** p < 0.001; ** p < 0.01; * p < 0.05.")
            display_paginated_df(summary, title="Regression Summary", key_prefix="vegas_regression_summary")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not read the saved regression summary: {error}")
else:
    st.info(f"Saved regression summary is unavailable: {summary_path}. Export it from vegas_accuracy.ipynb.")
