import streamlit as st

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats

import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from utils import show_image
st.title("Betting Simulation")

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

# Section header
st.markdown('<h2>Profit Data Results</h2>', unsafe_allow_html=True)

# First paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Three types of odds were scraped from the following website: 
<a href="https://www.bestfightodds.com/" target="_blank">BestFightOdds</a>.
</p>
""", unsafe_allow_html=True)

# Second paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Betting strategy is a risk-adjusted formulation of Kelly Criteria. 
Kelly Formula calculates the optimal percentage of bankroll to bet on a given fight. 
The key property of this formula is that it only produces a value &gt; 0 when the betting edge is positive. 
For the derivation of the basic formula, see this 
<a href="https://en.wikipedia.org/wiki/Kelly_criterion" target="_blank">Wikipedia article</a>.
</p>
""", unsafe_allow_html=True)

# Third paragraph
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Data included in plots below: Choice fighter betting Juice is defined as the difference between fair odds and real odds, or the bookmaker built-in edge. 
I use a devig power algorithm to estimate the fair odds from the real odds (see articles on Google explaining devigging). 
Tests showed using the fair odds in betting strategy greatly increased profit. 
Additionally, probability derived from fair odds was heavily weighted in the AI model (see the AI model page).
</p>
""", unsafe_allow_html=True)

# Fourth paragraph
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Open Odds: best results were obtained with opening odds. 
These odds are unaffected by line movement and market betting behavior.
</p>
""", unsafe_allow_html=True)

path =  BASE_DIR / "Data" / "plot_pngs" / "open_kelly_sim.png"
show_image(path, title='Open Odds Simulation')

path=r'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\returns_distributions_open.png'
show_image(path, title='Open Returns Distributions')

st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Below is the saved data displayed in the plots. The key variable is choice fstar. This is the optimal percentage of bankroll to wager. 
</p>
""", unsafe_allow_html=True)



st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)


st.markdown("""
## Close1 Odds Analysis

These odds produced the worst results. See net odds and number of bets placed. 
The bets are filtered by expected value (probability win * net odds) greater than zero.

The data shows market behavior greatly diminishes the edge from AI predictions. 
Notice the dip in starting bankroll at the start; tests revealed that replenishing bankroll back to the initial 
greatly improved profits when the strategy started to succeed. 

Plots displaying juice and line movement are for choice fighters with positive EV.
""")

path =  BASE_DIR / "Data" / "plot_pngs" / "close1_kelly_sim.png"
show_image(path, title='Close1 Odds Simulation')

path=r'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\returns_distributions_close1.png'
show_image(path, title='Close1 Returns Distributions')


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)


st.markdown('## Close2 Odds: Better results from close1 but worse results from Open ')

path =  BASE_DIR / "Data" / "plot_pngs" / "close2_kelly_sim.png"
show_image(path, title='Close2 Simulation')


path=r'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\returns_distributions_close2.png'
show_image(path, title='Close2 Returns Distributions')

st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)

