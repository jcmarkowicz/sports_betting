"""Interactive cached charts from the saved Results Tracking data."""
import streamlit as st
import pandas as pd

from utils import display_paginated_df
from ufc_betting.config import settings
from ufc_betting.Results.live_plotly import scaled_live_figures
from ufc_betting.Results.live_scaling import live_source_features
from event_scaling_report import load_market, PREDICTIONS, source_signature

st.set_page_config(page_title="Live Results", layout="wide")

@st.cache_data(show_spinner=False)
def cached_csv(path, version):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_sources(versions, history_signature):
    frames = [cached_csv(path, version) for path, version in versions]
    train_end = pd.to_datetime(pd.read_csv(PREDICTIONS / 'train/open/inference/train_with_open.csv').date).max()
    sources = {}
    for market in ['open', 'close1', 'close2']:
        historical = load_market(market)['events']
        warm = pd.read_csv(PREDICTIONS / f'trainval/open/inference/train_with_{market}.csv')
        sources[market] = live_source_features(*frames, market, warm, historical, train_end)
    return sources


@st.cache_data(show_spinner=False, max_entries=12)
def cached_figures(market, versions, history_signature, style_version=7):
    frames = [cached_csv(path, version) for path, version in versions]
    sources = cached_sources(versions, history_signature)
    return scaled_live_figures(*frames, market, sources['open'],
                               sources[market] if market in ('close1', 'close2') else None)


def file_version(path):
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)

st.title("Live Results")
st.caption("Settled results recalculated with the train+validation opening inference model, market-specific odds and Kelly policies, matching Event Scaling's test setup.")
sources = (settings.ml_history_file, settings.parlay_history_file, settings.bankroll_returns_file)
versions = None
try:
    versions = tuple((str(path), file_version(path)) for path in sources)
except OSError:
    st.info("Live results data is unavailable. Run Results Tracking to publish the result CSVs.")
for tab, market in zip(st.tabs(["Opening", "Close1", "Close2"]),
                       ("open", "close1", "close2")):
    with tab:
        st.subheader("Prediction calibration")
        st.caption(
            "All predictions includes "
            "zero-stake picks. Model and Vegas use the same settled fights in each panel; "
            "draws, no-contests, and missing probabilities or odds are excluded. "
            "Red-win probabilities, ten equal-width bins, and no-vig Vegas probabilities "
            "are used. These are recalculated strategy results, not a record of historically placed wagers."
        )
        if versions is not None:
            try:
                with st.spinner("Loading live chartsâ€¦"):
                    result = cached_figures(market, versions, source_signature(), style_version=7)
                    figures = result['figures']
                for name, title in (
                    ('calibration', None),
                    ('backtest', 'Bankroll and cumulative results'),
                    ('bankroll_distributions', 'Bankroll return distributions'),
                    ('event_accuracy', 'Event accuracy distributions'),
                ):
                    if title:
                        st.subheader(title)
                    if name == 'backtest':
                        st.caption('EWMA-5, ±0.02. Every filtered path uses the opening flat-event gate. Returns compound from the live initial bankroll without replenishments; historical results seed scalers before live tracking.')
                    if name == 'bankroll_distributions':
                        st.info(f"Diagnostics use the highest-ending-return path: {result['winner']} — {result['ending_return_pct']:,.2f}% cumulative return; {result['selected_events']}/{result['total_events']} events retained. This is a retrospective comparison.")
                    figure = figures[name]
                    # Apply presentation rules outside the cache, including to
                    # figures produced by an older imported plotting module.
                    figure.update_layout(template='none', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color=None))
                    for annotation in figure.layout.annotations:
                        annotation.update(bgcolor='rgba(0,0,0,0)', font=dict(color=None))
                    for axis in (*figure.select_xaxes(), *figure.select_yaxes()):
                        axis.update(showgrid=False, gridcolor='rgba(0,0,0,0)',
                                    zeroline=False, showline=False,
                                    minor=dict(showgrid=False, gridcolor='rgba(0,0,0,0)'))
                    st.plotly_chart(figure, width="stretch", theme="streamlit",
                                    key=f'live_{market}_{name}',
                                    config={'displaylogo': False, 'scrollZoom': False})
            except (OSError, ValueError, KeyError, IndexError) as error:
                st.error(f"Could not build live charts: {error}")

st.caption("Prediction accuracies generated by Results Tracking.")
accuracy_path = settings.betting_results_dir / "prediction_accuracies.csv"
if not accuracy_path.is_file():
    st.info("Prediction accuracies have not been published. Run Results Tracking to generate prediction_accuracies.csv.")
else:
    try:
        accuracy = cached_csv(str(accuracy_path), file_version(accuracy_path))
        if accuracy.empty:
            st.info("The saved prediction accuracies table is empty.")
        else:
            display_paginated_df(accuracy, title="Prediction Accuracies", key_prefix="live_prediction_accuracies")
    except (OSError, ValueError) as error:
        st.error(f"Could not read prediction accuracies: {error}")
