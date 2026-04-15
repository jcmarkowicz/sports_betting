import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import streamlit as st
from StreamlitPages.utils import show_image
from config import config

BASE_DIR = config.base_dir

st.title("Betting Simulation")

# Section header
st.markdown('<h2>Bets Sizing</h2>', unsafe_allow_html=True)

# Second paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Once fight predictions have been computed along with their probability of occurance, an optimal betting strategy can be derived. In short, 
it can be proven that the Kelly Criterion is the optimal fraction of ones bankroll that one should bet which will maximize wealth in the log run. The formula's simplicty
has made it effective in finance and its deriviation is shown here: <a href="https://en.wikipedia.org/wiki/Kelly_criterion" target="_blank">Wikipedia article</a>.
The key property of this formula is that it only produces a value greater than 0 when the betting edge is positive, meaning the computed AI probability is greater than the probability derived from vegas money lines.
</p>
""", unsafe_allow_html=True)

# Third paragraph
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
There are some limiations and risk when imposing the kelly strategy on real world data. For one, the formula assumes the probabilites are known, and the AI's probability is only an estimate. Therefore, the more calibrated (see AI training and testing page) the
AI is, the more reliable its probabilites will be and wealth over the long run will grow at faster rates. Another noted limiation the formula faces is that the variance of returns it produces is often high, and so it is common
to use a fraction of the kelly criterion in order to limit large drawdowns or swings. In my strategy, I implemented a binary search that finds the largest fraction of Kelly that keeps expected maximum drawdown less than pre selected amount.
The expected maximum draw is derived using the variance of wealth growth under kelly and a heuristic which estimates max drawdown under Gaussian returns from that variance. In simpler terms, if we were to place a bet on the same fight a given number of times,
the algorithm estimates the maximum possible difference in consecutive returns from high to low, and limits our strategy from crossing that threshold. As with all the following statistical models below, each comes with its own set of assumptions about the data that if broken
can severly hinder the interpretation of results. 
</p>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Some refernces for the deriviation of the maximum drawdown heuristic. Saving these for myself to explore more: <a href="https://en.wikipedia.org/wiki/Extreme_value_theory" target="_blank">Extreme Value Theory Wiki</a>.
            
</p><a href="https://en.wikipedia.org/wiki/Fisher%E2%80%93Tippett%E2%80%93Gnedenko_theorem?utm_source=chatgpt.com" target="_blank">extreme value theorem wiki</a>
""", unsafe_allow_html=True)

st.markdown('<h2>Results with Opening Odds</h2>', unsafe_allow_html=True)
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
With opening odds, the results are phenomenal with profit at is peak reaching 80 million. However, there are a couple of caveats with this strategy. Bankroll is updated
after every event meaning current sizing is reflected by the previous event's profit/losses. This will rarely be the case in the real world because opening odds require placing bets potentially weeks in advance in order to 
avoid shifting lines. Parlays are also effected from the staggered timeline because not all event fights are announced at once. Still, given the accuracy of both money line and parlay bets, I 
expect these concerns to not have much of an effect, its likely wealth will grow slower then displayed at first. Achieving a fraction of the displayed results is convincing enough to attempt the strategy. 
</p>
""", unsafe_allow_html=True)

path =  BASE_DIR / "Data" / "plot_pngs" / "open_kelly_sim.png"
show_image(path, title='Open Odds Simulation')

path = BASE_DIR / "Data" / "plot_pngs" / "returns_distributions_open.png"
show_image(path, title='Open Returns Distributions')


st.markdown("""
## Close1 Odds Results

This data shows market behavior greatly diminishes the edge/equity from AI predictions. All the more reason to lock in opening odds bets ahead of time 

""")

path =  BASE_DIR / "Data" / "plot_pngs" / "close1_kelly_sim.png"
show_image(path, title='Close1 Odds Simulation')

path = BASE_DIR / "Data" / "plot_pngs" / "returns_distributions_close1.png"
show_image(path, title='Close1 Returns Distributions')


st.markdown("""
## Close2 Odds Results:
            
Better results from close1 but still much worse results from Open 

""")

path =  BASE_DIR / "Data" / "plot_pngs" / "close2_kelly_sim.png"
show_image(path, title='Close2 Simulation')


path = BASE_DIR / "Data" / "plot_pngs" / "returns_distributions_close2.png"
show_image(path, title='Close2 Returns Distributions')


