import pandas as pd
import streamlit as st
from utils import display_paginated_df, show_image
from ufc_betting.config import settings, VAL_START_DATE, TEST_START_DATE

st.title("AI Training and Testing")
st.markdown(
    "The opening model is logistic regression with L1 regularization. Its inputs "
    "contain information available before the fight, and its output estimates "
    "the probability that the red fighter wins."
)
st.markdown(
    f"The data is split chronologically: training precedes {VAL_START_DATE}, "
    f"validation runs from {VAL_START_DATE} up to {TEST_START_DATE}, and test "
    f"starts on {TEST_START_DATE}. The Betting Simulation page shows validation backtests."
)
st.markdown(
    "Each metrics figure shows calibration, a confusion matrix, ROC discrimination, "
    "and F1 across probability thresholds. Calibration compares predicted probabilities "
    "with observed win frequencies; classification accuracy alone does not measure calibration."
)

st.subheader("Model Results and Coefficients")
st.markdown(
    "Validation metrics use the model fitted on training data. Test metrics and "
    "the coefficient tables use the model fitted on training plus validation data. "
    "The three variants share fitted coefficients and use opening, close1, or "
    "close2 odds inputs; input_feature identifies the input used by each variant."
)
st.markdown("Coefficient significance: *** p < 0.001; ** p < 0.01; * p < 0.05. Missing statistics are shown as —.")
metrics_dir = settings.plot_pngs_dir
coefficients_dir = settings.data_dir / "model_results" / "coefficients"
for market in ("open", "close1", "close2"):
    model_type = f"open_with_{market}"
    st.subheader(model_type)
    for split, split_label in (("val", "Validation"), ("test", "Test")):
        # Preserve the opening validation filename used by the training workflow.
        filename = ("val_open_model_metrics.png" if split == "val" and market == "open"
                    else f"{split}_{model_type}_model_metrics.png")
        path = metrics_dir / filename
        if path.is_file():
            show_image(path, title=f"{split_label} Split | Model: {model_type}")
        else:
            st.info(f"Run results_visuals.ipynb to export {path}")

    # Explicit filenames exclude older fits and inference-model exports.
    table_path = coefficients_dir / f"{model_type}_trainval_coefficients.csv"
    if table_path.is_file():
        coefficients = pd.read_csv(table_path, keep_default_na=False)
        if coefficients.empty:
            st.info(f"No coefficients in {table_path.name}")
        else:
            for column in ("coefficient", "std_error", "z", "p_value", "ci_lower", "ci_upper"):
                values = pd.to_numeric(coefficients[column], errors="coerce")
                coefficients[column] = values.map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
            display_paginated_df(
                coefficients,
                title=f"{model_type} Coefficients — Train + Validation Fit",
                key_prefix=f"coefficients_{model_type}_trainval",
            )
    else:
        st.info(f"Run results_visuals.ipynb to export {table_path}")
