"""Opening and closing event-scaling diagnostics."""
import streamlit as st
from event_scaling_report import opening_report, closing_report, source_signature


@st.cache_data(show_spinner=False, max_entries=6)
def report(market, signature):
    # signature makes file updates invalidate otherwise identical reports.
    return opening_report() if market == "open" else closing_report(market)


st.title("Event Scaling")
st.caption("Opening inference model • Validation and test • Historical event filters and reduction-only stake scaling")
opening, close1, close2 = st.tabs(["Opening odds", "Close1 odds", "Close2 odds"])
try:
    signature = source_signature()
except OSError as error:
    st.error(f"Could not inspect saved event data: {error}")
    st.stop()

for tab, market in [(opening, "open"), (close1, "close1"), (close2, "close2")]:
    with tab:
        if market == "open":
            st.write("Smoothing comparisons use EWMA and rolling means over 5/10 events with ±0.02/±0.03 bands. Detailed plots use EWMA-5, ±0.02.")
        else:
            st.write("The crossed-source bars compare opening and closing gates/scalers. Bankroll and rolling-advantage plots below use this closing market’s own history.")
        st.caption("Volatility target: last 10 above-mean non-flat events; comparison: last 10 current-mean-group events including flats. All filtered strategies skip flat events. Separate-period bankrolls reset to 500; prior-event histories carry into test.")
        try:
            with st.spinner(f"Calculating {market} event scaling…"):
                result = report(market, signature)
        except (OSError, ValueError, KeyError, AssertionError) as error:
            st.error(f"Could not build {market} results: {error}")
            continue
        for figure in result["figures"]:
            st.image(figure, width="stretch")
        with st.expander("Results tables"):
            for name, table in result.items():
                if name != "figures":
                    st.subheader(name.replace("_", " ").title())
                    st.dataframe(table, hide_index=True, width="stretch")
