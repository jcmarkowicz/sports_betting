"""Saved Monte Carlo reports for the sampled-fight, conditional-price strategy."""
import json

import pandas as pd
import streamlit as st

from utils import display_paginated_df, show_image
from ufc_betting.config import settings


REPORT_DIR = settings.data_dir / "montecarlo_results" / "intermediate_results_metrics"
DENSITY_EXAMPLE = settings.data_dir / "montecarlo_results" / "conditional_price_density_example.png"


def show_report_plot(filename, title):
    path = REPORT_DIR / filename
    if not path.is_file():
        st.info(f"Plot unavailable: {filename}. Export it from Notebooks/MonteCarlo/results.ipynb.")
        return
    try:
        show_image(path, title=title)
    except (OSError, ValueError) as error:
        st.error(f"Could not display {filename}: {error}")


def show_report_table(filename, title, key, *, transpose=False):
    path = REPORT_DIR / filename
    if not path.is_file():
        st.info(f"Table unavailable: {filename}. Export it from Notebooks/MonteCarlo/results.ipynb.")
        return
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            st.info(f"No rows available for {title}.")
            return
        if transpose:
            frame = frame.set_index("Prices").T.rename_axis("Metric").reset_index()
        display_paginated_df(frame.round(4), title=title, key_prefix=key)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not read {filename}: {error}")


st.title("Monte Carlo Simulation")
st.write("These reports explore how sampled fight cards and execution odds affect the combined moneyline and parlay strategy. "
         "The page displays saved simulation outputs; opening it does not retrain models or run another simulation.")

st.header("How the paths are sampled")
st.markdown("""
1. **Build event paths.** Event templates are rearranged while preserving their card structure; in fight-resampling mode, each slot draws a historical fight with replacement from the same event type and bout position. The donor fight carries its features and recorded outcome, and may appear more than once.
2. **Choose the price scenario.** Each event draws a lead time from 7, 6 or 5 days; each fight draws a uniform closing endpoint between its recorded close1 and close2 odds, using the same interpolation fraction for both fighters.
3. **Sample execution odds.** For each fighter, the nearer endpoint's fitted conditional density is conditioned on opening odds and the interpolated closing endpoint, then used to draw intermediate odds. Execution odds are sampled from this density, not uniformly between the endpoints.
4. **Recalculate bets.** The selected opening prediction model and its standard errors are evaluated at the sampled prices, then the strategy recalculates stakes and parlays on each synthetic card.
""")
st.latex(r"u\sim U(0,1),\qquad c=(1-u)c_1+uc_2,\qquad o_{execution}\sim\widehat f_d(o\mid o_{open},c)")
st.write("The density example below uses the saved close1 model for seven days before an event, conditioned on opening odds of 1.50 and a closing endpoint of 3.00. "
         "The curve shows probability density over possible intermediate decimal odds; it is an illustrative scenario, not a specific fight prediction.")
if DENSITY_EXAMPLE.is_file():
    try:
        show_image(DENSITY_EXAMPLE, title="Conditional execution-odds sampling example")
    except (OSError, ValueError) as error:
        st.error(f"Could not display the density example: {error}")
else:
    st.info("Run the density-example export cell in Notebooks/Odds/odds_movement.ipynb to create this illustration.")

st.header("How the betting strategy is applied")
st.write("In nearest-closing mode, each moneyline uses the close1 or close2 policy nearest its sampled offered odds; the two-leg parlay selects its policy by comparing product odds. "
         "The parlay legs are the two highest-EV predicted sides on the sampled card.")
st.write("Kelly stakes are reduced using each policy's drawdown budget and horizon, then opening-policy and closing-policy stakes are blended according to price movement. "
         "The selected closing policy's z adjustment reduces stakes for probability uncertainty, followed by a shared cap on total moneyline and parlay exposure.")
st.write("Recorded fight outcomes settle the wagers, and the combined bankroll compounds at the configured settlement interval. "
         "After ruin, the path stops placing bets and retains its terminal balance, including any loss beyond zero.")
st.caption("These are historical resampling scenarios based on the validation workflow and fitted price densities, not an independent forecast of future profitability. "
           "Original outcomes are reused; fight outcomes are not newly drawn from model probabilities.")

settings_path = REPORT_DIR / "analysis_settings.json"
if settings_path.is_file():
    try:
        analysis = json.loads(settings_path.read_text(encoding="utf-8"))
        st.caption(f"Saved report: {analysis.get('event_count', 'unknown')} scheduled events per path; "
                   f"settlement interval: {analysis.get('len_temp', 'unknown')} event(s).")
    except (OSError, ValueError) as error:
        st.info(f"Report settings could not be read: {error}")

st.header("Simulation summary")
show_report_table("summary.csv", "Summary statistics", "mc_summary", transpose=True)

st.header("Bankroll paths and terminal returns")
show_report_plot("bankroll_paths.png", "Simulated Bankroll Paths")
show_report_plot("terminal_return_distribution.png", "Terminal return distribution")
st.caption("Terminal return is the percentage change from each path's starting bankroll; negative returns and ruin remain in the distribution. "
           "P05 and P75 are the 5th and 75th percentiles, alongside the median.")

st.header("Drawdown risk")
show_report_plot("max_drawdown_histogram.png", "Worst peak-to-trough drawdown per path")
show_report_plot("drawdown_count_histogram.png", "Drawdown episodes reaching 50% or more")
st.caption("Maximum drawdown measures the largest fall from a previous bankroll peak. "
           "A new 50% drawdown episode is counted only after recovery to the running peak resets the previous episode.")

st.header("First reach of bankroll targets")
show_report_plot("events_to_bankroll_targets.png", "Events required to first reach each bankroll target")
show_report_table("events_to_targets_summary.csv", "Target reach statistics", "mc_targets")
st.caption("First-reach statistics count when a target is first crossed, even if the path later falls below it. "
           "Unreached paths are reported separately; event-count percentiles use reached paths only.")

st.header("Geometric growth and bankroll exposure")
show_report_plot("geometric_growth_distribution.png", "Final geometric growth rates")
st.caption("Each path contributes one compounded growth rate per scheduled event. "
           "Negative final balances have no defined geometric growth rate and are excluded; zero final balances contribute −100%.")
show_report_plot("mean_path_wager_distribution.png", "Average bankroll wagered per event")
st.caption("Each observation is one path's average total bankroll fraction wagered per scheduled event, counting its parlay ticket once. "
           "The title shows the mean and sample standard deviation across these path averages.")

with st.expander("Detailed simulation tables"):
    show_report_table("path_statistics.csv", "Per-path performance and drawdowns", "mc_paths")
    show_report_table("events_to_targets_by_path.csv", "Target reach by path", "mc_target_paths")
    show_report_table("geometric_growth_by_path.csv", "Geometric growth by path", "mc_growth")
    show_report_table("mean_bankroll_wagered_by_path.csv", "Average event exposure by path", "mc_exposure")
