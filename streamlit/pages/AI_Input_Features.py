import streamlit as st

import numpy as np 
import pandas as pd 
st.title("Data Page")

import matplotlib.pyplot as plt
from pathlib import Path

import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils import display_paginated_df, show_image
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

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



