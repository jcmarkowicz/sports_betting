"""Test-split Vegas/model event accuracy and regression reports."""
import pandas as pd
import streamlit as st

from ufc_betting.config import settings
from ufc_betting.Results.returns import SETTLED_TYPES, generate_test_accuracy_results


st.title("Vegas vs Model Predictions — Test Results")
st.caption("Test data only. Each point represents one event. Opening model evaluated at open, close1 and close2 odds.")
output_dir = settings.betting_results_dir / "vegas_accuracy_test"

if st.button("Generate / refresh test results", type="primary"):
    try:
        with st.spinner("Generating test accuracy and regression reports..."):
            generate_test_accuracy_results(output_dir=output_dir)
        st.success("Test plots and regression tables saved.")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not generate test results: {error}")
        st.stop()

required = [output_dir / f"{name}.csv" for name in ("accuracy_summary", "fit_summary", "coverage", "coefficients")]
for market in SETTLED_TYPES:
    required.extend(output_dir / f"test_open_{market}_{suffix}" for suffix in
                    ("event_accuracy.png", "accuracy_regression.png", "regression_coefficients.csv"))
if not all(path.is_file() for path in required):
    st.info("Click Generate / refresh test results to create the saved test reports.")
    st.stop()

try:
    summary = pd.read_csv(output_dir / "accuracy_summary.csv")
    fit_summary = pd.read_csv(output_dir / "fit_summary.csv")
    coverage = pd.read_csv(output_dir / "coverage.csv")
    st.subheader("Overall test accuracy")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    with st.expander("Data coverage and prediction files"):
        st.dataframe(coverage, hide_index=True, use_container_width=True)
        st.caption("Use Generate / refresh after updating prediction files.")
    for tab, market in zip(st.tabs(["Opening odds", "Close1 odds", "Close2 odds"]), SETTLED_TYPES):
        with tab:
            prefix = f"test_open_{market}"
            st.subheader("Event accuracy and correlation")
            st.image(str(output_dir / f"{prefix}_event_accuracy.png"), use_container_width=True)
            st.subheader("Model accuracy regressed on Vegas accuracy")
            st.image(str(output_dir / f"{prefix}_accuracy_regression.png"), use_container_width=True)
            st.caption("OLS with an intercept and Vegas accuracy as the only predictor; one equally weighted observation per event.")
            st.dataframe(fit_summary.loc[fit_summary.market == market], hide_index=True, use_container_width=True)
            coefficients_path = output_dir / f"{prefix}_regression_coefficients.csv"
            coefficients = pd.read_csv(coefficients_path).fillna({"significance": ""})
            st.subheader("Coefficient tests")
            st.caption("Two-sided t-tests of coefficient = 0, with 95% confidence intervals and ordinary OLS standard errors.")
            st.text("Significance: *** p < 0.001; ** p < 0.01; * p < 0.05; blank otherwise.")
            numeric_columns = coefficients.select_dtypes(include="number").columns
            st.dataframe(
                coefficients.round(4), hide_index=True, use_container_width=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.4f")
                    for column in numeric_columns
                },
            )
            st.download_button("Download regression coefficients", coefficients_path.read_bytes(),
                               file_name=coefficients_path.name, mime="text/csv", key=f"coefficients_{market}")
    st.download_button("Download all regression results", (output_dir / "coefficients.csv").read_bytes(),
                       file_name="test_regression_coefficients.csv", mime="text/csv")
    st.download_button("Download regression fit summary", (output_dir / "fit_summary.csv").read_bytes(),
                       file_name="test_regression_fit_summary.csv", mime="text/csv")
except (OSError, ValueError, KeyError) as error:
    st.error(f"Saved report could not be read. Generate / refresh to rebuild it. Details: {error}")
