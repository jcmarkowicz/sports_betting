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
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from utils import display_paginated_df, show_image
from config import config

BASE_DIR = config.base_dir  



df_ml = pd.read_csv( config.ml_returns_fp)
df_parlay = pd.read_csv(config.parlay_returns_fp)
df_bankroll = pd.read_csv(config.bankroll_returns_fp)


# show_image(path, title='Percent Returns')

def plot_returns(ml_results, parlay_results, bankroll_results):

    types = ['open', 'close1', 'close2']
    fig, axes = plt.subplots(2, len(types), figsize=(15,6))
    
    for i, type_ in enumerate(types):
        # ML percent returns
        net_odds = ml_results[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[0, i], kde=False, color='skyblue')
        avg_ml = np.mean(net_odds)
        total_ml = np.sum(net_odds)
        axes[0, i].set_title(f"ML {type_} — Avg: {avg_ml:.2f}, Total: {total_ml:.2f}")
        
        # Parlay percent returns
        net_odds = parlay_results[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[1, i], kde=False, color='salmon')
        avg_parlay = np.mean(net_odds)
        total_parlay = np.sum(net_odds)
        axes[1, i].set_title(f"Parlay {type_} — Avg: {avg_parlay:.2f}, Total: {total_parlay:.2f}")

    plt.tight_layout()
    path = BASE_DIR / 'Data' / 'plot_pngs' / 'avg_returns_live.png'
    
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    show_image(path, title='Odds Returns: Moneyline and Parlay')

    fig, axes = plt.subplots(nrows=3, figsize=(15,6))
    date = bankroll_results['date']
    for i, type_ in enumerate(types): 
        bankroll = bankroll_results[f'bankroll_{type_}']
        axes[i].plot(date, bankroll)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Bankroll')
        axes[i].set_title(f'Bankroll for odds {type_}, total={np.sum(bankroll):.2f}')
    plt.tight_layout()

    path = BASE_DIR / 'Data' / 'plot_pngs' / 'bankroll_returns_live.png'
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    show_image(path, title='Bankroll Returns: Moneyline + Parlay')



def accuracy_analysis(ml_results, parlay_results):

    accuracies = defaultdict(list)
    types = ['open', 'close1', 'close2']

    for type_ in types: 
        accuracy_all = (ml_results[f'pred_winner_{type_}'] == ml_results['winner_bool']).mean()
        accuracies[f'accuracy_all_{type_}'].append(accuracy_all)

        bets_only = ml_results[ml_results[f'fstar_{type_}'] > 0]
        accuracy_bets = (bets_only[f'pred_winner_{type_}'] == bets_only['winner_bool']).mean()
        accuracies[f'accuracy_bets_{type_}'].append(accuracy_bets)

        parlay_accuracy = (parlay_results[f'net_odds_{type_}'] > 0).mean()
        accuracies[f'accuracy_parlays_{type_}'].append(parlay_accuracy)

    return pd.DataFrame(accuracies)

plot_returns()

df_accuracy = accuracy_analysis(df_ml, df_parlay)
display_paginated_df(df_accuracy, title='Returns Accuracy', key_prefix='all returns')
