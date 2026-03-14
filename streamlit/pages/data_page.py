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


fp = r'C:\Users\jcmar\my_files\SportsBetting\Data\plot_pngs\input_plots.png'

show_image(fp, title='Model Input Features (no odds)')



