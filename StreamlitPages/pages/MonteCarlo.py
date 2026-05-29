import sys
import os


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import streamlit as st
from StreamlitPages.utils import display_paginated_df, show_image
from config import config

import pandas as pd 

BASE_DIR = config.base_dir


st.title("Monte Carlo Simulation")

# Section header
st.markdown('<h2>Quantifying Uncertainty in Betting Strategies</h2>', unsafe_allow_html=True)

# Second paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Once fight predictions have been made and bankroll wagers computed, we can walk forward in time and propogage profits and losses for a single path
or rather permutation of events. However, implementing these algorithms and expecting the same test results is a mistake. While it is true using a hold out set 
provides an estimate of online(real time) performance, the final bankroll in the betting simulation is highly dependent on the order of events, thus it could vary greatly based
on the sequence due to how much we can wager at each point in time. Enter Monte Carlo simulations, a method that quantifies uncertainty of path dependence by randomly 
shuffling the order of events and propogating bankroll over time. Each simulation produces a different permutation, or order of events such that a final distribution emerges
which can provide the min, max, average, and standard devatiation of the final bankroll, all very important for determing the utility of the underlying betting algorithms. 
            
</p>
""", unsafe_allow_html=True)

st.markdown('<h2>Other Important Considerations</h2>', unsafe_allow_html=True)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
In addition to sampling random permutations, I also consider the probability of selecting each odds type. The probabilities are chosen based on my beliefs and experience with 
placing the bets manually. Betting lines often shift within hours of opening so getting the openining odds is somewhat difficult. However, they do take some time to stabalize so this is where the real money can come in. 
Thinking realistically, I chose 15% for opening odds, 35% for close1 and .50% for close2. This is also based on close 1 being the odds resulting in the worst results, so Im assuming those are the final stabalized odds and close2 is somewhat in between. 
If I were to place bets roughly a week in advance, I think its fair to be getting somewhat between open and close 2 85% of the time. Also, since I am placing bets weeks in advance, I 
am adding a 3-event delay to updating my total bankroll since its unrealistic to expect to have bankroll updated in real time. All results are based on the final xgboost stacked model for close1 and close2 odds, 
while the opening odds model remains the same. Lastly, if a simulation path results in a bankroll < 0, I 
do not update that path anymore and keep it as zero in order to calcualte the probability of going broke.  
</p>
""", unsafe_allow_html=True)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
The results of the parlay strategy indicate I may need to reduce risk tolerance. Money line and finaly distributions have 95% of suceeding. 
</p>
""", unsafe_allow_html=True)



path =  BASE_DIR / "Data" / "plot_pngs" / "benchmark_results.csv"

table = pd.read_csv(path)

display_paginated_df(table, title='Probability of Final Bankrolls', key_prefix=f"mc_1")


path =  BASE_DIR / "Data" / "plot_pngs" / "mc_paths_delay.png"
show_image(path, title='Bankroll Paths for 10,000 Simulations')

path = BASE_DIR / "Data" / "plot_pngs" / "mc_hists_delay.png"
show_image(path, title='Final Distributions of Each Bankroll Type')

path = BASE_DIR / "Data" / "plot_pngs" / "mc_below_zero_delay.png"
show_image(path, title='Fraction of Simulations with Negative Returns')