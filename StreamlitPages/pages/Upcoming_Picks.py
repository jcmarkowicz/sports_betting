import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
import pandas as pd 
import streamlit as st

import os 
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from utils import display_paginated_df

from ufc_betting.config import config, settings
from ufc_betting.UpcomingPicks.betting_pipeline import build_event_ev_table
from ufc_betting.Results.live_scaling import live_source_features
from event_scaling_report import load_market, PREDICTIONS, source_signature
from upcoming_scaling import FILTERS, event_decision, wager_tables


@st.cache_data(show_spinner=False, max_entries=4)
def scaling_sources(versions, historical_signature):
    frames = [pd.read_csv(path) for path, _, _ in versions]
    train_end = pd.to_datetime(pd.read_csv(PREDICTIONS / 'train/open/inference/train_with_open.csv').date).max()
    result = {}
    for market in ['open', 'close2']:
        historical = load_market(market)['events']
        warm = pd.read_csv(PREDICTIONS / f'trainval/open/inference/train_with_{market}.csv')
        result[market] = live_source_features(*frames, market, warm, historical, train_end)
    return result


st.title('Upcoming Picks')
st.subheader('Wager filters')
st.markdown("""
| Label | What it does |
|---|---|
| **Raw** | Original published wager, without filtering or scaling. |
| **Flat** | Skip an event when the opening trend is flat. |
| **Brier** | Flat filter plus a prior-Brier stake reduction. |
| **Vol** | Flat filter plus a volatility stake reduction. |
| **Combined** | Flat filter plus both stake reductions. |
| **· C2** | Use Close2 scalers, keeping the opening flat-event gate. |
""")
st.caption('EWMA-5, ±0.02. All filtered options use the opening trend gate. Options without · C2 use opening scalers. Changes preview wagers only; published picks and saved bets are not modified.')
columns = st.columns(3)
selected_filters = {}
for column, market, label in zip(columns, ['open','close1','close2'], ['Opening filter','Close1 filter','Close2 filter']):
    with column:
        options = list(FILTERS) if market == 'close2' else list(FILTERS)[:5]
        selected_filters[market] = st.selectbox(label, options, key=f'upcoming_filter_{market}')
sources = {}
scaling_error = None
if any(label != 'Raw' for label in selected_filters.values()):
    try:
        paths = [settings.ml_history_file, settings.parlay_history_file, settings.bankroll_returns_file]
        versions = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        with st.spinner('Calculating filters from completed events…'):
            sources = scaling_sources(versions, source_signature())
    except (OSError, ValueError, KeyError, AssertionError) as error:
        scaling_error = str(error)
        st.warning(f'Filtered wager previews are unavailable: {error}. Raw wagers remain available.')

###

def extract_date(filename):
    date_str = filename.split('_')[-1].replace('.csv', '')
    return datetime.strptime(date_str, "%Y-%m-%d")


def available_picks_dir(configured_path, folder):
    configured_path = Path(configured_path)
    fallback = settings.data_dir / "bets" / folder
    return configured_path if configured_path.is_dir() else fallback


def files_by_event(folder, pattern):
    files = {}
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        try:
            event_date = extract_date(path.name).date()
        except ValueError:
            continue
        files[event_date] = path
    return files


ml_dir = available_picks_dir(settings.ml_bets_dir, "straight_bets")
parlay_dir = available_picks_dir(settings.parlay_bets_dir, "parlays")
for folder in (ml_dir, parlay_dir):
    if not folder.is_dir():
        st.info(f"Upcoming picks folder is unavailable: {folder}. Publish the generated picks CSVs to this deployment.")
        st.stop()

ml_files = files_by_event(ml_dir, "ml_all_*.csv")
parlay_files = files_by_event(parlay_dir, "parlay_all_*.csv")
today = datetime.now(ZoneInfo("America/New_York")).date()
upcoming_dates = {date for date in ml_files.keys() | parlay_files.keys() if date >= today}
if not upcoming_dates:
    st.info("No upcoming picks have been published yet.")
    st.stop()

for event_date in sorted(upcoming_dates):
    date_str = event_date.isoformat()
    if event_date not in ml_files or event_date not in parlay_files:
        st.info(f"Picks for {date_str} are incomplete: both moneyline and parlay CSVs are required.")
        continue
    ml_file = ml_files[event_date].name
    parlay_file = parlay_files[event_date].name

    moneyline_table = pd.read_csv(ml_files[event_date])
    parlay_table = pd.read_csv(parlay_files[event_date])
    st.header(date_str)
    decisions = {}
    for market, label in selected_filters.items():
        if scaling_error and label != 'Raw':
            decision = dict(multiplier=None, trend='Unavailable', source_date=None,
                            status='Scaling history unavailable')
        else:
            decision = event_decision(event_date, label, sources)
        decisions[market] = dict(decision, label=label)
    try:
        wagers, parlay_wagers, summary = wager_tables(moneyline_table, parlay_table, decisions)
        st.subheader('Selected filters and total wagers')
        st.dataframe(summary.rename(columns={'Total_wager':'Total wager'}), hide_index=True, width='stretch',
            column_config={'Multiplier':st.column_config.NumberColumn(format='%.3f'),
                           'Total wager':st.column_config.NumberColumn(format='$%.2f')})
        prior_dates = [d['source_date'] for d in decisions.values() if pd.notna(d.get('source_date'))]
        if prior_dates:
            st.caption(f'Latest completed event used for the opening gate: {max(prior_dates):%Y-%m-%d}. Future scheduled events are evaluated from currently available history.')
        st.subheader('Moneyline wagers')
        st.dataframe(wagers, hide_index=True, width='stretch',
            column_config={f'{name} wager':st.column_config.NumberColumn(format='$%.2f')
                           for name in ['Opening','Close1','Close2']})
        st.subheader('Parlay wagers — one amount per ticket')
        st.dataframe(parlay_wagers, hide_index=True, width='stretch',
            column_config={'Wager':st.column_config.NumberColumn(format='$%.2f')})
        st.caption('A blank wager means the published amount or scaling inputs are unavailable. Zero means no wager. Totals include each parlay ticket once.')
    except (ValueError, KeyError) as error:
        st.error(f'Could not build wager preview: {error}')

    # Prediction tables retain odds/probabilities; allocations are displayed only above.
    ml_columns = [c for c in config.ml_column_order if not c.startswith(('stake_', 'fstar_'))]
    if not moneyline_table.empty:
        display_paginated_df(moneyline_table.reindex(columns=ml_columns),
            title=f'Moneyline predictions {date_str}', key_prefix=f'ml_{ml_file}')
    parlay_columns = [c for c in config.parlay_column_order if not c.startswith(('stake_', 'parlay_fstar_'))]
    if not parlay_table.empty:
        display_paginated_df(parlay_table.reindex(columns=parlay_columns),
            title=f'Parlay predictions {date_str}', key_prefix=f'parlay_{parlay_file}')

    # ----- EVENT EV TABLE -----
    event_ev = build_event_ev_table(
        moneyline_bets=moneyline_table,
        parlay_bets=parlay_table,
    )[["bet_type", "close1_pct_opening_ev", "close2_pct_opening_ev"]]
    event_ev = event_ev.rename(columns={
        "bet_type": "Bet Type",
        "close1_pct_opening_ev": "Close1 EV (% of open)",
        "close2_pct_opening_ev": "Close2 EV (% of open)",
    }).round(2)
    st.subheader(f"Published strategy EV {date_str}")
    st.caption("EV comparison uses the original published picks, before the selected wager reductions.")
    st.dataframe(event_ev, use_container_width=True, hide_index=True)
