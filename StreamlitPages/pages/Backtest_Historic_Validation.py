import streamlit as st
from utils import show_image
from ufc_betting.config import settings

st.title("Betting Simulation")
st.subheader("Bet sizing")
st.markdown(
    "The Kelly criterion gives the bankroll fraction that maximizes expected "
    "logarithmic growth for a binary bet under the supplied win probability and payout. "
    "For decimal odds d, net odds b = d − 1, and estimated win probability p:"
)
st.latex(r"f_{\mathrm{Kelly}} = \frac{bp-(1-p)}{b} = \frac{dp-1}{d-1}, \qquad f_{\mathrm{bet}}=\max(0,f_{\mathrm{Kelly}})")
st.markdown(
    "A positive Kelly stake requires p > 1/d, the break-even probability at the "
    "offered price. The simulation applies saved moneyline and parlay risk controls "
    "and probability-uncertainty adjustments to these base stakes, with a shared "
    "event exposure limit. Closing-market runs also use the saved decay controls "
    "that blend opening and closing stake allocations."
)
st.markdown(
    "These are historical validation backtests starting with a $500 bankroll. "
    "The opening model supplies predictions with opening, close1, or close2 inputs. "
    "The saved betting controls were selected using validation data, so these "
    "results describe that sample rather than a separate held-out test."
)

st.subheader("Reading the plots")
st.markdown(
    "**Bankroll and profit:** the three panels show total bankroll over time, "
    "cumulative parlay profit, and cumulative moneyline profit. The component "
    "profits are dollar contributions to the shared bankroll."
)
st.markdown(
    "**Bankroll distributions:** the top row shows event returns as percentages "
    "of the bankroll available before each event, for moneylines, parlays, and "
    "their combined result. The bottom row shows each component's cumulative "
    "profit relative to initial bankroll, followed by combined compounded return "
    "and geometric average growth per event. These are observed event returns, "
    "not distributions of simulated terminal bankrolls."
)
st.markdown(
    "**Event accuracy and exposure:** compare the accuracy of wagered model picks "
    "with Vegas favorites across all available fights and across wagered fights. "
    "Each fight counts once, including fights selected through parlays; void outcomes "
    "are excluded, and tied prices are excluded from favorite accuracy. The fourth "
    "panel shows the percentage of starting bankroll allocated to singles and "
    "parlays at each event. Accuracy alone does not determine profitability: prices "
    "and stake sizes also matter."
)

plot_dir = getattr(settings, "plot_pngs_dir", None)
if plot_dir is None:
    images_dir = getattr(settings, "images_dir", settings.data_dir / "images")
    plot_dir = images_dir / "plot_pngs"
for market, label, backtest_file in (
    ("open", "Opening Odds", "open_backtest.png"),
    ("close1", "Close1 Odds", "close1.png"),
    ("close2", "Close2 Odds", "close2_backtest.png"),
):
    st.subheader(f"{label} — Validation Results")
    for filename, caption in (
        (backtest_file, "Bankroll and cumulative profit"),
        (f"{market}_bankroll_distributions.png", "Event bankroll returns and growth"),
        (f"{market}_event_accuracy.png", "Event accuracy and bankroll exposure"),
    ):
        path = plot_dir / filename
        if path.is_file():
            show_image(path, title=f"{label}: {caption}")
        else:
            st.info(f"Run val_results.ipynb to export {path}")
