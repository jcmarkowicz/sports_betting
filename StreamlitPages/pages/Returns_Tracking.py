import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import os
import sys 
from pathlib import Path 
import streamlit as st

import numpy as np 
import pandas as pd 
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from utils import display_paginated_df, show_image
from config import config

BASE_DIR = config.base_dir  



# show_image(path, title='Percent Returns')
st.title(" Prediction and Betting Results since 2026-02-21")


def plot_returns(ml_results, parlay_results, bankroll_results):

    types = ['open', 'close1', 'close2']
    fig, axes = plt.subplots(2, len(types), figsize=(15,6))
    
    for i, type_ in enumerate(types):
        # ML percent returns
        no_draws = ml_results[ml_results[f'pred_winner_{type_}'] < 2]
        all_preds = no_draws.dropna(subset=[f'pred_winner_{type_}'])
        bets_only = all_preds[all_preds[f'net_stake_{type_}'] != 0]

        net_odds = bets_only[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[0, i], kde=True, color='skyblue', bins=50)
        avg_ml = np.mean(net_odds)
        total_ml = np.sum(net_odds)
        axes[0, i].set_title(f"ML {type_} — Avg: {avg_ml:.2f}, Total: {total_ml:.2f}")
        
        # Parlay percent returns
        net_odds = parlay_results[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[1, i], kde=True, color='salmon', bins=50)
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
        axes[i].scatter(date, bankroll)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Bankroll')
        axes[i].set_title(f'Bankroll for Odds Type {type_}, Current Total: {bankroll.iloc[-1]:.2f}')

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
    bet_types = defaultdict(list)

    types = ['open', 'close1', 'close2']

    for type_ in types: 

        no_draws = ml_results[ml_results[f'pred_winner_{type_}'] < 2]
        all_preds = no_draws.dropna(subset=[f'pred_winner_{type_}'])
        bets_only = all_preds[all_preds[f'net_stake_{type_}'] != 0]

        accuracy_all = (all_preds[f'pred_winner_{type_}'] == all_preds['winner_bool']).mean()
        accuracies[f'preds_all_{type_}'].append(accuracy_all)

        accuracy_bets = (bets_only[f'pred_winner_{type_}'] == bets_only['winner_bool']).mean()
        accuracies[f'bets_{type_}'].append(accuracy_bets)

        parlay_accuracy = (parlay_results[f'net_odds_{type_}'] > 0).mean()
        accuracies[f'parlays_{type_}'].append(parlay_accuracy)

        avail_vegas = no_draws.dropna(subset=[f'{type_}_red', f'{type_}_blue'])
        odds_vegas = avail_vegas[[f'{type_}_blue', f'{type_}_red']].to_numpy()
        winners = avail_vegas['winner_bool']

        vegas_preds = np.where(
            odds_vegas[:, 0] == odds_vegas[:, 1],
            1,
            np.argmin(odds_vegas, axis=1)
        )
        accuracies[f'vegas_{type_}'] = (vegas_preds == winners).mean()

        choice_odds = np.where(bets_only[f'pred_winner_{type_}'] == 1, bets_only[f'{type_}_red'], bets_only[f'{type_}_blue'])
        dog_bets = bets_only[choice_odds > 0]
        total_dog = dog_bets.shape[0]
        win_dog = dog_bets[f'pred_winner_{type_}'] == dog_bets[f'winner_bool']
        n_win_dog = win_dog.sum()

        fav_bets = bets_only[choice_odds < 0]
        total_fav = fav_bets.shape[0]
        win_dog = fav_bets[f'pred_winner_{type_}'] == fav_bets[f'winner_bool']
        n_win_fav = win_dog.sum()

        bet_types[f'total_fav_{type_}'].append(total_fav)
        bet_types[f'total_dog_{type_}'].append(total_dog)
        bet_types[f'n_win_fav_{type_}'].append(n_win_fav)
        bet_types[f'n_win_dog_{type_}'].append(n_win_dog)

        win_pct_fav = n_win_fav / total_fav if total_fav != 0 else np.nan
        bet_types[f'accuracy_fav_{type_}'].append(win_pct_fav)

        win_pct_dog = n_win_dog / total_dog if total_dog != 0 else np.nan
        bet_types[f'accuracy_dog_{type_}'].append(win_pct_dog)

    df_accuracies = pd.DataFrame(accuracies)
    df_accuracies = df_accuracies.T.set_axis(['Accuracies'], axis=1).round(2)

    df_bet_types = pd.DataFrame(bet_types).T.set_axis(['Bet Type Stats'], axis=1).round(2)
    return df_accuracies, df_bet_types

ml_results = pd.read_csv(config.ml_returns_fp)
parlay_results = pd.read_csv(config.parlay_returns_fp)
bankroll_results = pd.read_csv(config.bankroll_returns_fp)

plot_returns(ml_results, parlay_results, bankroll_results)

df_accuracy, df_bet_types = accuracy_analysis(ml_results, parlay_results)
display_paginated_df(df_accuracy, title='Returns Accuracy', key_prefix='returns')
display_paginated_df(df_bet_types, title='Bet Type Stats', key_prefix='')
