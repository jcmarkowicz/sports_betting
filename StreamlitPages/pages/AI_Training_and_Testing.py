"""Inference-only notebook reports, rendered directly without PNG exports."""
import matplotlib.pyplot as plt
import streamlit as st

from ufc_betting.config import settings
from ufc_betting.Results.inference_reports import (
    REPORTS, report_paths, load_report, calibration_figure, inference_coefficients,
)


@st.cache_data(show_spinner=False)
def cached_report(root, mode, split, file_versions):
    return load_report(root, mode, split)


@st.cache_data(show_spinner=False)
def cached_coefficients(path, modified):
    return inference_coefficients(path)


st.title("AI Training and Testing")
st.write(
    "Inference logistic models and reports from lr_open_close_manual.ipynb. "
    "Each curve identifies its model and odds input; open → close1 and open → close2 "
    "are opening-model comparisons, while close1 → close1 and close2 → close2 use "
    "their own fitted models."
)
st.caption(
    "Training results are in-sample. Validation uses the train-only fit. The combined-fit "
    "training report covers training plus validation and is in-sample. Held-out test "
    "results are excluded. Calibration uses red-win probabilities and ten equal-width bins."
)
root = settings.data_dir / 'model_comparisons/lr_open_close'
for tab, (label, (mode, split)) in zip(st.tabs(list(REPORTS)), REPORTS.items()):
    with tab:
        try:
            metrics_path = report_paths(root, mode, split)
            prediction_paths = sorted((root / 'report_predictions' / mode).glob(f'*/inference/{split}_with_*.csv'))
            versions = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                             for p in [metrics_path, *prediction_paths])
            metrics, curves = cached_report(str(root), mode, split, versions)
            fig = calibration_figure(curves, label)
            try:
                st.pyplot(fig)
            finally:
                plt.close(fig)
            st.subheader('Accuracy and probability metrics')
            st.dataframe(metrics.style.format(precision=4, na_rep='—'), use_container_width=True)
            st.caption(
                "Parlay accuracy is a percentage of winning settled, positive-stake tickets; "
                "other metrics are fractions. F1 uses red as the positive class. Brier is "
                "mean squared probability error. Expected calibration error weights bin gaps "
                "by counts; mean calibration error averages nonempty-bin gaps equally. "
                "Metrics are the saved notebook reports, including their saved betting-policy evaluation."
            )
        except (OSError, ValueError, KeyError) as error:
            st.info(f'Inference report unavailable: {error}. Generate the comparison reports first.')

        st.subheader('Inference model coefficients')
        st.caption('*** p < 0.001; ** p < 0.01; * p < 0.05. Coefficients use the fitted model feature scales.')
        for market in ('open', 'close1', 'close2'):
            with st.expander(f'{market} model — {mode} fit'):
                path = root / mode / market / 'model_se.pkl'
                try:
                    table = cached_coefficients(str(path), path.stat().st_mtime_ns)
                    st.dataframe(table.style.format({c: '{:.4f}' for c in
                        ('coefficient', 'std_error', 'z', 'p_value', 'ci_lower', 'ci_upper')},
                        na_rep='—'), use_container_width=True)
                except (OSError, ValueError, AttributeError) as error:
                    st.info(f'Inference coefficients unavailable: {error}')
