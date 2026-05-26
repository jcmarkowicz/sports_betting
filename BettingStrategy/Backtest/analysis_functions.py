
import numpy as np
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt



def compare_parlay(parlay_open, parlay_close): 

    n_same = 0
    n_diff = 0

    net_odds_same = 0
    net_odds_diff = 0

    wins_same = 0
    wins_diff = 0

    for date, group_open in parlay_open.groupby('date'):
        group_close = parlay_close[parlay_close['date'] == date]

        net_odds = group_open['parlay_net_odds'].iloc[0] - group_close['parlay_net_odds'].iloc[0]

        if (group_close['choice_fighter_name'] == group_open['choice_fighter_name']).all():
            n_same += 1
            net_odds_same += net_odds
            if group_open['parlay_net_odds'].iloc[0] > 0:
                wins_same += 1
        else:
            n_diff += 1
            net_odds_diff += net_odds
            if group_open['parlay_net_odds'].iloc[0] > 0:
                wins_diff += 1

    print(f"Same choice: {n_same}, Diff choice: {n_diff}, net_odds_diff: {net_odds_diff}, net_odds_same: {net_odds_same} wins_same: {wins_same}, wins_diff: {wins_diff}")
            


def pick_accuracy(df_):
    df= df_.copy()

    df_choice = df[df['choice_fstar'] > 0]

    event_accuracy = (
    (df_choice['pred_winner'] == df_choice['winner'])
    .groupby(df_choice['date'])
    .mean()
    )
    n_picks = df_choice.groupby('date').size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Event accuracy ---
    sns.histplot(event_accuracy, bins=20, kde=True, ax=axes[0])
    axes[0].set_title(f'Event-Level Accuracy\nmean={event_accuracy.mean():.2f}')
    axes[0].set_xlabel('Event-Level Accuracy')
    axes[0].set_ylabel('Frequency')

    # --- Picks per day ---
    n_picks = df_choice.groupby('date').size()

    sns.histplot(n_picks, bins=20, kde=True, ax=axes[1])
    axes[1].set_title(f'Number of Picks per Day\nmean={n_picks.mean():.2f}')
    axes[1].set_xlabel('Number of Picks per Day')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def bin_parlay_prob(df_parlay_):
    df = df_parlay_.copy()
    df['parlay_win'] = 

    # --- bin probabilities ---
    df['prob_bin'] = pd.cut(
        df['parlay_prob'],
        bins=np.linspace(0, 1, 11),
        include_lowest=True
    )

    # --- ensure win column exists ---
    # if not already present, uncomment/adapt:
    # df['parlay_win'] = (df['winner'] == df['pred_winner']).astype(int)

    # --- aggregations ---
    agg = (
        df.groupby('prob_bin')
        .agg(
            expected_net_odds=('parlay_net_odds', 'mean'),
            accuracy=('parlay_win', 'mean'),
            count=('parlay_win', 'size')
        )
        .reset_index()
    )

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # expected net odds
    sns.barplot(data=agg, x='prob_bin', y='expected_net_odds', ax=axes[0])
    axes[0].set_title('Expected Net Odds by Parlay Probability Bin')
    axes[0].set_xlabel('Probability Bin')
    axes[0].set_ylabel('Expected Net Odds')
    axes[0].tick_params(axis='x', rotation=45)

    # accuracy
    sns.barplot(data=agg, x='prob_bin', y='accuracy', ax=axes[1])
    axes[1].set_title('Accuracy by Parlay Probability Bin')
    axes[1].set_xlabel('Probability Bin')
    axes[1].set_ylabel('Win Rate')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return agg
