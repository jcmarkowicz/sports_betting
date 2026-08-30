import numpy as np
import pandas as pd 

from collections import defaultdict 

def fighter_prediction_stats(df, pred_col='pred_winner', red_col='fighter_red', blue_col='fighter_blue', winner_col='winner'):
    
    stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    for _, row in df.iterrows():
        red_fighter = row[red_col]
        blue_fighter = row[blue_col]
        winner = row[winner_col]
        pred = row[pred_col]
        if pred == winner :
            stats[red_fighter]['correct'] += 1
            stats[blue_fighter]['correct'] += 1
        else:
            stats[red_fighter]['incorrect'] += 1
            stats[blue_fighter]['incorrect'] += 1
    return stats

def fighter_accuracy_check(preds_results_dict, red_name, blue_name, total_cutoff=6, min_accuracy=60):
    # Get stats for red fighter
    red_stats = preds_results_dict.get(red_name, {'correct': 0, 'incorrect': 0})
    red_correct = red_stats['correct']
    red_incorrect = red_stats['incorrect']
    red_total = red_correct + red_incorrect
    red_accuracy = (red_correct / red_total * 100) if red_total > 0 else 0

    # Get stats for blue fighter
    blue_stats = preds_results_dict.get(blue_name, {'correct': 0, 'incorrect': 0})
    blue_correct = blue_stats['correct']
    blue_incorrect = blue_stats['incorrect']
    blue_total = blue_correct + blue_incorrect
    blue_accuracy = (blue_correct / blue_total * 100) if blue_total > 0 else 0

    info = {'red_accuracy':red_accuracy, 'blue_accuracy': blue_accuracy, 'red_total':red_total, 
     'blue_total':blue_total, 'red_correct':red_correct, 'blue_correct':blue_correct, 'red_incorrect':red_incorrect, 'blue_incorrect':blue_incorrect}

    df_accuracy_history = pd.DataFrame([info])
    return df_accuracy_history

def calculate_results_history(results_dict, row, bet_idx, p, fair_odds):
    correct_red = results_dict[row['fighter_red']]['correct']
    incorrect_red = results_dict[row['fighter_red']]['incorrect']
    correct_blue = results_dict[row['fighter_blue']]['correct']
    incorrect_blue = results_dict[row['fighter_blue']]['incorrect']
    pred = bet_idx
    if pred == 1:
        kelly_frac, posterior = bayesian_edge(p, fair_odds, correct_red, incorrect_red)
    else:
        kelly_frac, posterior = bayesian_edge(p, fair_odds, correct_blue, incorrect_blue)

    winner = row['winner']
    red_fighter = row['fighter_red']
    blue_fighter = row['fighter_blue']
    if pred == winner :
        if pred == 1: 
            results_dict[red_fighter]['correct'] += 1
        else: 
            results_dict[blue_fighter]['correct'] += 1
    else:
        if pred == 1: 
            results_dict[red_fighter]['incorrect'] += 1
        else:
            results_dict[blue_fighter]['incorrect'] += 1
    return results_dict, kelly_frac, posterior

def summarize_line_movement(df):
    return (
        df.groupby('choice_close1')
        .agg(
            count=('net_odds', 'size'),
            avg_net_odds=('net_odds', 'mean'),
            avg_line_movement=('choice_close1', lambda x: (x - df.loc[x.index, 'choice_open']).mean())
        )
        .reset_index()
        .sort_values('choice_close1')
    )

# 1️⃣ Underdog → Favorite
df_ud_to_fav = df_kelly1[(df_kelly1['choice_close1'] > 0) & (df_kelly1['choice_open'] < 0)]
df_ud_to_fav['line_movement'] = df_ud_to_fav['choice_open'] - df_ud_to_fav['choice_close1'] 
table_ud_to_fav = summarize_line_movement(df_ud_to_fav)
print("📈 Line moved from underdog → favorite")
print(table_ud_to_fav, "\n")

# 2️⃣ Stayed Favorite
df_stayed_fav = df_kelly[(df_kelly['choice_open'] > 0) & (df_kelly['choice_close1'] > 0)]
df_stayed_fav['line_movement'] = df_stayed_fav['choice_open'] - df_stayed_fav['choice_close1'] 

table_stayed_fav = summarize_line_movement(df_stayed_fav)
print("🏆 Line stayed positive (favorite all along)")
print(table_stayed_fav, "\n")

# 3️⃣ Favorite → Underdog
df_fav_to_ud = df_kelly1[(df_kelly1['choice_open'] > 0) & (df_kelly1['choice_close1'] < 0)]

df_fav_to_ud['line_movement'] = df_fav_to_ud['choice_open'] - df_fav_to_ud['choice_close1'] 

table_fav_to_ud = summarize_line_movement(df_fav_to_ud)
print("📉 Line moved from favorite → underdog")
print(table_fav_to_ud)