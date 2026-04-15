import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import streamlit as st
from config import config

from StreamlitPages.utils import display_paginated_df, show_image

BASE_DIR = config.base_dir

st.title("Data Page")

st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Odds data was scraped from the following website: 
<a href="https://www.bestfightodds.com/" target="_blank">BestFightOdds</a>.
</p>
""", unsafe_allow_html=True)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
UFC stats data was scraped from the following website: 
<a href="http://www.ufcstats.com/statistics/events/completed" target="_blank">BestFightOdds</a>.
</p>
""", unsafe_allow_html=True)


fp = BASE_DIR / "Data" / "plot_pngs" / "input_plots.png" 
show_image(fp, title='Model Input Features (no odds)')



