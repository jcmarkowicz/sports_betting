import numpy as np 
import pandas as pd 

from sklearn.metrics import accuracy_score

def save_results(df_model, probs_2d, fp):
    """
    df_model : full dataframe BEFORE splitting
    probs_2d : blue model probs, red model probs
    fp : output filepath
    """
    df = df_model.copy()
    df[['proba_blue', 'proba_red']] = probs_2d

    df["pred_winner"] = np.argmax(probs_2d, axis=1)
    df["correct_pred"] = (df["pred_winner"] == df["winner"]).astype(int)
    df["prob_winner"] = df[["proba_blue", "proba_red"]].max(axis=1)

    df['Date'] = df['event_date']
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.to_csv(fp, index=False)
    return df


def print_gender_accuracy(df, weight_col='weight_class', pred_col='pred_winner', true_col='winner'):
    """
    Prints the accuracy of predictions separately for men and women weight classes.

    Parameters:
    - df: pd.DataFrame containing the columns for weight class, predicted winner, and true winner
    - weight_col: name of the column with weight class strings
    - pred_col: name of the column with predicted winner
    - true_col: name of the column with true winner
    """
    
    # Identify Men / Women
    df = df.copy()
    df['gender_group'] = df[weight_col].apply(lambda x: 'Women' if 'Women' in str(x) else 'Men')

    # Compute accuracy for men
    men_df = df[df['gender_group'] == 'Men']
    men_acc = accuracy_score(men_df[true_col], men_df[pred_col])

    # Compute accuracy for women
    women_df = df[df['gender_group'] == 'Women']
    women_acc = accuracy_score(women_df[true_col], women_df[pred_col])

    print(f"Accuracy for Men weight classes:   {men_acc:.3f}")
    print(f"Accuracy for Women weight classes: {women_acc:.3f}")


def calc_net_odds(df_test_results, odds_type):
    # Choose the probabilities and decimal odds according to prediction
    df_test_results['proba_choice'] = np.where(df_test_results['pred_winner']==1,
                                               df_test_results['proba_red'], 
                                               df_test_results['proba_blue'])
    df_test_results['dec_choice'] = np.where(df_test_results['pred_winner']==1,
                                             df_test_results[f'dec_{odds_type}_red'],
                                             df_test_results[f'dec_{odds_type}_blue'])
    # EV
    df_test_results['ev'] = df_test_results['proba_choice'] * (df_test_results['dec_choice']-1) - (1-df_test_results['proba_choice'])
    
    # Keep only positive EV bets
    df_ev = df_test_results[df_test_results['ev'] > 0].copy()

    print(df_ev.shape, df_test_results.shape)
    
    # Calculate adjusted profit per bet
    df_ev['adj_profit'] = np.where(df_ev['pred_winner'] == df_ev['winner'],
                                   df_ev['dec_choice']-1, -1)
    
    # Separate fav vs dog
    df_ev['is_fav'] = np.where(
        ((df_ev['pred_winner']==1) & (df_ev[f'{odds_type}_red'] <= df_ev[f'{odds_type}_blue'])) |
        ((df_ev['pred_winner']==0) & (df_ev[f'{odds_type}_blue'] < df_ev[f'{odds_type}_red'])),
        1, 0
    )
    
    fav_sum = df_ev.loc[df_ev['is_fav']==1, 'adj_profit'].tolist()
    dog_sum = df_ev.loc[df_ev['is_fav']==0, 'adj_profit'].tolist()
    
    # Compute stats
    def summarize(profits):
        n_total = len(profits)
        n_correct = sum(1 for x in profits if x != -1)
        n_incorrect = n_total - n_correct
        pct_correct = n_correct/n_total*100 if n_total>0 else 0
        return n_total, n_correct, n_incorrect, pct_correct

    fav_stats = summarize(fav_sum)
    dog_stats = summarize(dog_sum)
    
    print("Sum of decimal odds for favorite predictions (adjusted):", sum(fav_sum))
    print("Sum of decimal odds for underdog predictions (adjusted):", sum(dog_sum))
    print(f"Fav bets: {fav_stats[0]} total, {fav_stats[1]} correct, {fav_stats[2]} incorrect, {fav_stats[3]:.2f}% correct")
    print(f"Dog bets: {dog_stats[0]} total, {dog_stats[1]} correct, {dog_stats[2]} incorrect, {dog_stats[3]:.2f}% correct")


def stack_close_features(df,
                         other_cols=['open_pred']):
    close1_cols = [
        'proba_fair_close1_diff',
        'line_movement_close1_blue',
        'line_movement_close1_red'
    ]

    close2_cols = [
        'proba_fair_close2_diff',
        'line_movement_close2_blue',
        'line_movement_close2_red'
    ]

    rename_map_close1 = {
        'proba_fair_close1_diff': 'proba_fair_diff',
        'line_movement_close1_blue': 'line_movement_blue',
        'line_movement_close1_red': 'line_movement_red'
    }

    rename_map_close2 = {
        'proba_fair_close2_diff': 'proba_fair_diff',
        'line_movement_close2_blue': 'line_movement_blue',
        'line_movement_close2_red': 'line_movement_red'
    }

    df_close1 = df[close1_cols + other_cols].copy()
    df_close1 = df_close1.rename(columns=rename_map_close1)
    df_close2 = df[close2_cols + other_cols].copy()
    df_close2 = df_close2.rename(columns=rename_map_close2)
    return pd.concat([df_close1, df_close2], ignore_index=True)


def add_gaussian_noise(df, columns, std=0.01, random_state=None):
    """
    Add Gaussian noise N(0, std^2) to the specified columns in a copy of df.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list
        Columns to apply noise to.
    std : float
        Standard deviation of the Gaussian noise.
    random_state : int or None
        Reproducibility.
        
    Returns
    -------
    df_noisy : pd.DataFrame
        DataFrame with noise added.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()

    for col in columns:
        noise = rng.normal(loc=0.0, scale=std, size=len(df))
        df_noisy[col] = df_noisy[col] + noise

    return df_noisy
    
