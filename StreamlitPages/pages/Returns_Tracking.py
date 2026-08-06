import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import os
import sys 
from pathlib import Path 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from utils import display_paginated_df, show_image
from config import config

BASE_DIR = config.base_dir  

ml_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'ml_pct_returns.csv'
parlay_pct_returns_fp = BASE_DIR / 'Data' / 'betting_results' / 'parlay_pct_returns.csv'


df_ml_pct = pd.read_csv( config.ml_pct_returns_fp)
df_parlay_pct = pd.read_csv(config.parlay_pct_returns_fp)


display_paginated_df(df_ml_pct, title='Money Line Percent Returns', key_prefix='ml')
display_paginated_df(df_parlay_pct, title='Parlay Percent Returns', key_prefix='parlay')

# show_image(path, title='Percent Returns')

def plot_returns():

    types = ['open', 'close1', 'close2']

    fig, axes = plt.subplots(2, len(types), figsize=(15,6))
    for i, type_ in enumerate(types):
        # ML percent returns
        sns.histplot(df_ml_pct[type_], ax=axes[0, i], kde=False, color='skyblue')
        avg_ml = np.mean(df_ml_pct[type_])
        total_ml = np.sum(df_ml_pct[type_])
        axes[0, i].set_title(f"ML {type_} — Avg: {avg_ml:.2f}, Total: {total_ml:.2f}")
        
        # Parlay percent returns
        sns.histplot(df_parlay_pct[type_], ax=axes[1, i], kde=False, color='salmon')
        avg_parlay = np.mean(df_parlay_pct[type_])
        total_parlay = np.sum(df_parlay_pct[type_])
        axes[1, i].set_title(f"Parlay {type_} — Avg: {avg_parlay:.2f}, Total: {total_parlay:.2f}")

    plt.tight_layout()

    path = BASE_DIR / 'Data' / 'plot_pngs' / 'pct_returns.png'
    fig.savefig(path,
            dpi=300,
            bbox_inches="tight")

    show_image(path, title='Percent Returns')  


plot_returns()