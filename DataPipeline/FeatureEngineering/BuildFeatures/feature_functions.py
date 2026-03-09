import numpy as np 
import pandas as pd 
from collections import defaultdict
from math import isfinite

def months_since_last(ufc_df):
    
    fighter_last_event = defaultdict(lambda: [None])# keep last event date per fighter

    months_since_red = []
    months_since_blue = []

    for _, row in ufc_df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        event_date = pd.to_datetime(row['date'])

        # For red fighter
        last_date_red = fighter_last_event.get(red_fighter)
        if last_date_red is not None:
            months_diff = (event_date.year - last_date_red.year) * 12 + (event_date.month - last_date_red.month)
            months_since_red.append(months_diff)
        else:
            months_since_red.append(np.nan)

        # For blue fighter
        last_date_blue = fighter_last_event.get(blue_fighter)
        if last_date_blue is not None:
            months_diff = (event_date.year - last_date_blue.year) * 12 + (event_date.month - last_date_blue.month)
            months_since_blue.append(months_diff)
        else:
            months_since_blue.append(np.nan)

        # Update last seen dates
        fighter_last_event[red_fighter] = event_date
        fighter_last_event[blue_fighter] = event_date

    # Add columns back to dataframe if you want
    return np.column_stack([months_since_red, months_since_blue])

def time_decay_average(arr, decay_lambda=0.13):
    arr = np.asarray(arr)
    
    # 0 = most recent, larger = older
    age = np.arange(len(arr)-1, -1, -1)
    
    weights = np.exp(-decay_lambda * age)
    
    return np.sum(weights * arr) / np.sum(weights)

def opponent_avg_features(df, feat):

    """
    
    """
    red_col = []
    blue_col = []

    fighter_history = defaultdict(list)
    fighter_adjusted_history = defaultdict(lambda:[np.nan])

    for _, row in df.iterrows(): 

        fighter_red = row['fighter_red']
        fighter_blue = row['fighter_blue']

        red_col.append(fighter_adjusted_history[fighter_red][-1])
        blue_col.append(fighter_adjusted_history[fighter_blue][-1])

        red_feat = row[f'{feat}_red']
        blue_feat = row[f'{feat}_blue']

        if not pd.isna(blue_feat):
            fighter_history[fighter_red].append(blue_feat)

        if not pd.isna(red_feat):
            fighter_history[fighter_blue].append(red_feat)

        red_opp_hist = time_decay_average(fighter_history[fighter_red]) if len(fighter_history[fighter_red]) != 0 else np.nan
        blue_opp_hist = time_decay_average(fighter_history[fighter_blue]) if len(fighter_history[fighter_blue]) != 0 else np.nan

        fighter_adjusted_history[fighter_red].append(red_opp_hist)
        fighter_adjusted_history[fighter_blue].append(blue_opp_hist)

    return np.column_stack([red_col, blue_col])
        

def class_acc(df, weight_class, attempted_col, landed_col, curr_date):
    """ get red and blue attempted and landed """

    df_weight_class = df[(df['weight_class'] == weight_class) & (df['date'] < curr_date)].dropna()
    
    class_acc_red = df_weight_class[landed_col + '_red'].sum() / df_weight_class[attempted_col + '_red'].sum()
    class_acc_blue = df_weight_class[landed_col + '_blue'].sum() / df_weight_class[attempted_col + '_blue'].sum()

    class_acc_mean = (class_acc_red + class_acc_blue) / 2
    
    return class_acc_mean

def expected_value_stats(df, attempt_col, land_col):

    """" pre fight stats, per minute """

    red_col = []
    blue_col = []
    efficiency = defaultdict(list)
    fighter_history = defaultdict(lambda: defaultdict(list))

    def compute_fighter_performance(row, fighter_name, wc_acc, color):

        att_hist = [x for x in fighter_history[fighter_name]['attempted'] if pd.notna(x)]
        land_hist = [x for x in fighter_history[fighter_name]['landed'] if pd.notna(x)]

        if len(land_hist)!=0:

            # get past accuracy of fighter, NOT INCLUDIG CURRENT
            acc = np.sum(land_hist) / np.sum(att_hist)

            # take most recent fight landed
            curr_fight_attempted = row[attempt_col + f'_{color}']

            # get expected strikes class
            expected_class = wc_acc * curr_fight_attempted

            # get expected fighter
            expected_fighter = acc * curr_fight_attempted

            # get performance
            performance = expected_fighter - expected_class
        else: 
            performance = None
        
        fighter_history[fighter_name]['attempted'].append(row[f'{attempt_col}_{color}'])
        fighter_history[fighter_name]['landed'].append(row[f'{land_col}_{color}'])

        return performance

    for _, row in df.iterrows(): 

        wc = row['weight_class']
        date = row['date']

        fighter_red = row['fighter_red']
        fighter_blue = row['fighter_blue']

        # get weighted average of weight class HISTORY
        wc_acc = class_acc(df, wc, attempt_col, land_col, date)

        # current fight performance 
        red_performance = compute_fighter_performance(row, fighter_red, wc_acc, 'red') 
        blue_performance = compute_fighter_performance(row, fighter_blue, wc_acc, 'blue') 

        # print(red_performance, blue_performance)

        # red_efficiency = time_decay_average(efficiency[fighter_red] ) if len(efficiency[fighter_red])!= 0 else None
        # blue_efficiency = time_decay_average(efficiency[fighter_blue]) if len(efficiency[fighter_blue])!= 0 else None

        # get performance average 
        red_efficiency = np.mean([x for x in efficiency[fighter_red] if x is not pd.isna(x)]) if len(efficiency[fighter_red])!= 0 else None
        blue_efficiency = np.mean([x for x in efficiency[fighter_blue] if x is not pd.isna(x)]) if len(efficiency[fighter_blue])!= 0 else None

        red_col.append(red_efficiency)
        blue_col.append(blue_efficiency)

        if red_performance is not None: 
            efficiency[fighter_red].append(red_performance)

        if blue_performance is not None:
            efficiency[fighter_blue].append(blue_performance)

    return np.column_stack([red_col, blue_col])

def mma_math(df):
    fighter_dic_wins = defaultdict(set)  # Use sets for faster lookup
    fighter_dic_losses = defaultdict(set)

    mma_math_red = []
    mma_math_blue = []

    for idx, row in df.iterrows(): 
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        # Slice the dataframe to get all previous rows before the current row
        previous_fights = df.iloc[:idx]

        # Get the most recent fight where red_fighter or blue_fighter appeared as red
        red_history = previous_fights[
            (previous_fights['fighter_red'] == red_fighter) | 
            (previous_fights['fighter_blue'] == red_fighter)
        ].tail(1)  # Ensures a DataFrame is returned

        # Get the most recent fight where blue_fighter appeared as either red or blue
        blue_history = previous_fights[
            (previous_fights['fighter_red'] == blue_fighter) | 
            (previous_fights['fighter_blue'] == blue_fighter)
        ].tail(1)

        if not red_history.empty:
            red_history = red_history.iloc[0]  # Convert to Series
            if red_history['fighter_red'] == red_fighter and red_history['winner'] == 1:
                fighter_dic_wins[red_fighter].add(red_history['fighter_blue'])

            if red_history['fighter_blue'] == red_fighter and red_history['winner'] == 0:
                fighter_dic_wins[red_fighter].add(red_history['fighter_red'])

            # Red fighter's opponents lost to
            if red_history['fighter_red'] == red_fighter and red_history['winner'] == 0:
                fighter_dic_losses[red_fighter].add(red_history['fighter_blue'])
            
            if red_history['fighter_blue'] == red_fighter and red_history['winner'] == 1:
                fighter_dic_losses[red_fighter].add(red_history['fighter_red'])

        if not blue_history.empty:
            blue_history = blue_history.iloc[0]
            # Get blue fighter's defeated opponents
            if blue_history['fighter_red'] == blue_fighter and blue_history['winner'] == 1:               
                fighter_dic_wins[blue_fighter].add(blue_history['fighter_blue'])

            if blue_history['fighter_blue'] == blue_fighter and blue_history['winner'] == 0:
                fighter_dic_wins[blue_fighter].add(blue_history['fighter_red'])

            # Blue fighter's opponents lost to
            if blue_history['fighter_red'] == blue_fighter and blue_history['winner'] == 0:
                fighter_dic_losses[blue_fighter].add(blue_history['fighter_blue'])
            
            if blue_history['fighter_blue'] == blue_fighter and blue_history['winner'] == 1:
                fighter_dic_losses[blue_fighter].add(blue_history['fighter_red'])

        # Assign 1 if the red fighter has previously defeated the blue fighter, else 0
        common_opponents_red = fighter_dic_wins[red_fighter] & fighter_dic_losses[blue_fighter]
        mma_math_red.append(1 if common_opponents_red else 0)

        # Check if there exists a fighter that blue beat and red lost to
        common_opponents_blue = fighter_dic_wins[blue_fighter] & fighter_dic_losses[red_fighter]
        mma_math_blue.append(1 if common_opponents_blue else 0)
    return np.column_stack([mma_math_red, mma_math_blue])

def rolling_avg(values): # tune the window for rolling stats 
    value_window = values
    non_none_values = [x for x in values if x is not None and not pd.isna(x)]
    if len(non_none_values)==0:
        return None #return none for debuts 
    else:
        total = sum(non_none_values) / len(non_none_values) 
        return total
    
def count_fav_dog(df_):
    df = df_.copy()
    fighter_counts = defaultdict(lambda:defaultdict(int))
    red_fav_counts = []
    blue_fav_counts = []

    red_dog_counts = []
    blue_dog_counts = []

    for i, row in df.iterrows(): 
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        red_open = row['open_red']
        blue_open = row['open_blue']

        if red_open <= blue_open: 
            fighter_counts[red_fighter]['fav_counts'] += 1
            fighter_counts[blue_fighter]['dog_counts'] += 1
        else: 
            fighter_counts[blue_fighter]['fav_counts'] += 1
            fighter_counts[red_fighter]['dog_counts'] += 1

        red_fav_counts.append(fighter_counts[red_fighter]['fav_counts'])
        red_dog_counts.append(fighter_counts[red_fighter]['dog_counts'])

        blue_fav_counts.append(fighter_counts[blue_fighter]['fav_counts'])
        blue_dog_counts.append(fighter_counts[blue_fighter]['dog_counts'])

    print("Shapes of fav counts:")
    print(np.array(red_fav_counts).shape, np.array(blue_fav_counts).shape, np.array(red_dog_counts).shape, np.array(blue_dog_counts).shape )

    assert len(red_fav_counts) == df.shape[0], 'bad fav counts'
    assert len(blue_fav_counts) == len(red_fav_counts) == len(red_dog_counts) == len(blue_dog_counts), 'mismatched counts lengths'  
    r1 = np.array(red_fav_counts, dtype=np.int64)
    r2 = np.array(red_dog_counts, dtype=np.int64)
    r3 = np.array(blue_fav_counts, dtype=np.int64)
    r4 = np.array(blue_dog_counts, dtype=np.int64)

    return np.stack([r1, r2, r3, r4], axis=1)    


def total_knockdowns(df):
    fighter_dic = defaultdict(list)
    red_kd = []
    blue_kd = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        if len(fighter_dic[red_fighter])==0:
            red_kd.append(np.nan)
        else:
            red_kd.append(sum(fighter_dic[red_fighter]))

        if len(fighter_dic[blue_fighter])==0:
            blue_kd.append(np.nan)
        else:
            blue_kd.append(sum(fighter_dic[blue_fighter])) 

        #update after fight
        fighter_dic[red_fighter].append(row['red_kd'])
        fighter_dic[blue_fighter].append(row['blue_kd'])
    
    return np.column_stack([red_kd,blue_kd])

def total_bonus(df):

    red_bonus = []
    blue_bonus = []

    fighter_dic = defaultdict(list)

    for i, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        
        # update bonus history 
        if len(fighter_dic[red_fighter]) == 0:
            red_bonus.append(np.nan)
        else: 
            red_bonus.append(np.sum([x for x in fighter_dic[red_fighter] if x is not None]))
        
        if len(fighter_dic[blue_fighter]) == 0:
            blue_bonus.append(np.nan)
        else:
            blue_bonus.append(np.sum([x for x in fighter_dic[blue_fighter] if x is not None]))

        # current fight update
        if row['performance_bonus_winner'] == 1:
            fighter_dic[red_fighter].append(1)

        else:
            fighter_dic[red_fighter].append(0)

        if row['fight_otn_bonus'] == 1:
            fighter_dic[red_fighter].append(1)
            fighter_dic[blue_fighter].append(1)
        
        else:
            fighter_dic[blue_fighter].append(0)

    return np.column_stack([red_bonus, blue_bonus])

def win_lose_streak(df):
    """ 
    returns 0 for first time fighters 
    """
    # Initialize with default lists
    fighter_dic = defaultdict(lambda: defaultdict(lambda: [0]))
    wins = defaultdict(lambda: [None])  # store past wins (1/0)
    
    # Lists to collect features
    red_win_streak, blue_win_streak = [], []
    red_lose_streak, blue_lose_streak = [], []

    wins_pct_red, wins_pct_blue = [], []
    num_fights_red, num_fights_blue = [], []
    num_wins_red, num_wins_blue = [], []
    num_losses_red, num_losses_blue = [], []

    for _, row in df.iterrows():
        red = row['fighter_red']
        blue = row['fighter_blue']

        # Read streaks before this fight
        red_win_streak.append(fighter_dic[red]['win_streak'][-1])
        red_lose_streak.append(fighter_dic[red]['lose_streak'][-1])
        blue_win_streak.append(fighter_dic[blue]['win_streak'][-1])
        blue_lose_streak.append(fighter_dic[blue]['lose_streak'][-1])

        # Collect past fight results, excluding initial None
        red_results = [v for v in wins[red] if v is not None]
        blue_results = [v for v in wins[blue] if v is not None]

        # Num of fights so far
        num_fights_red.append(len(red_results))
        num_fights_blue.append(len(blue_results))

        # Num of wins
        num_wins_red.append(sum(red_results) if red_results else 0)
        num_wins_blue.append(sum(blue_results) if blue_results else 0)

        # Num of losses = total fights - wins
        num_losses_red.append(len(red_results) - sum(red_results) if red_results else 0)
        num_losses_blue.append(len(blue_results) - sum(blue_results) if blue_results else 0)

        # Win percentage (avoid div by zero)
        wins_pct_red.append(
            sum(red_results) / len(red_results) if red_results else 0
        )
        wins_pct_blue.append(
            sum(blue_results) / len(blue_results) if blue_results else 0
        )

        # Update fighter_dic & wins after this fight
        if row['winner'] == 1:
            # red won
            fighter_dic[red]['win_streak'].append(fighter_dic[red]['win_streak'][-1] + 1)
            fighter_dic[red]['lose_streak'].append(0)

            fighter_dic[blue]['win_streak'].append(0)
            fighter_dic[blue]['lose_streak'].append(fighter_dic[blue]['lose_streak'][-1] + 1)

            wins[red].append(1)
            wins[blue].append(0)

        elif row['winner'] == 0:
            # blue won
            fighter_dic[blue]['win_streak'].append(fighter_dic[blue]['win_streak'][-1] + 1)
            fighter_dic[blue]['lose_streak'].append(0)

            fighter_dic[red]['win_streak'].append(0)
            fighter_dic[red]['lose_streak'].append(fighter_dic[red]['lose_streak'][-1] + 1)

            wins[red].append(0)
            wins[blue].append(1)

        else:
            # Optional: handle draw/no contest etc.
            fighter_dic[red]['win_streak'].append(0)
            fighter_dic[red]['lose_streak'].append(0)
            fighter_dic[blue]['win_streak'].append(0)
            fighter_dic[blue]['lose_streak'].append(0)
            wins[red].append(0)
            wins[blue].append(0)

    return np.column_stack([
        red_win_streak,
        red_lose_streak,
        blue_win_streak,
        blue_lose_streak,
        wins_pct_red,
        wins_pct_blue,
        num_fights_red,
        num_fights_blue,
        num_wins_red,
        num_wins_blue,
        num_losses_red,
        num_losses_blue
    ])

def womens_fight(df):
    weight_classes = []

    for _, row in df.iterrows():
        if row['weight_class'] is not np.nan:
            if 'women' in row['weight_class'].lower():
                weight_classes.append(1)
            else:
                weight_classes.append(0)
        else:
            weight_classes.append(None)

    return np.array(weight_classes) 

def method_wins(df):
    fighter_dic = defaultdict(lambda: defaultdict(int))  # Set the default type to int for counting wins
    
    # Lists to store the results
    decision_wins_red = []
    ko_wins_red = []
    sub_wins_red = []
    total_red = []

    decision_wins_blue = []
    ko_wins_blue = []
    sub_wins_blue = []
    total_blue = []

    for _, row in df.iterrows():
        red = row['fighter_red']
        blue = row['fighter_blue']
        
        # Ensure each fighter has an initialized dictionary for win methods
        if red not in fighter_dic:
            fighter_dic[red] = defaultdict(int)
        if blue not in fighter_dic:
            fighter_dic[blue] = defaultdict(int)

        # Append the current method wins before the fight
        decision_wins_red.append(fighter_dic[red]['dec_wins'])
        ko_wins_red.append(fighter_dic[red]['ko_wins'])
        sub_wins_red.append(fighter_dic[red]['sub_wins'])
        total_red.append(fighter_dic[red]['total_wins'])

        decision_wins_blue.append(fighter_dic[blue]['dec_wins'])
        ko_wins_blue.append(fighter_dic[blue]['ko_wins'])
        sub_wins_blue.append(fighter_dic[blue]['sub_wins'])
        total_blue.append(fighter_dic[blue]['total_wins'])

        if pd.isna(row['method']):
            continue

        # Update the win counts based on the current fight
        if 'DEC' in row['method'] and row['winner'] == 1:
            fighter_dic[red]['dec_wins'] += 1
        if 'KO' in row['method'] and row['winner'] == 1:
            fighter_dic[red]['ko_wins'] += 1
        if 'SUB' in row['method'] and row['winner'] == 1:
            fighter_dic[red]['sub_wins'] += 1
        if row['winner'] == 1:
            fighter_dic[red]['total_wins'] += 1

        if 'DEC' in row['method'] and row['winner'] == 0:
            fighter_dic[blue]['dec_wins'] += 1
        if 'KO' in row['method'] and row['winner'] == 0:
            fighter_dic[blue]['ko_wins'] += 1
        if 'SUB' in row['method'] and row['winner'] == 0:
            fighter_dic[blue]['sub_wins'] += 1
        if row['winner'] == 0:
            fighter_dic[blue]['total_wins'] += 1

    # Stack the results into an array for easy assignment to the DataFrame
    return np.column_stack([
        decision_wins_red,
        ko_wins_red,
        sub_wins_red,
        decision_wins_blue,
        ko_wins_blue,
        sub_wins_blue
    ])

def method_win_pct(df):
    """
    Compute per-fight historical win percentages by method 
    (KO, DEC, SUB) for each fighter before each fight.
    """
    # running history per fighter
    fighter_history = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0}))

    # results
    red_ko_pct, red_dec_pct, red_sub_pct = [], [], []
    blue_ko_pct, blue_dec_pct, blue_sub_pct = [], [], []

    for _, row in df.iterrows():
        red, blue = row['fighter_red'], row['fighter_blue']
        method = row.get('method', None)

        # initialize for both fighters
        for fighter in [red, blue]:
            for m in ['KO', 'DEC', 'SUB']:
                if m not in fighter_history[fighter]:
                    fighter_history[fighter][m] = {'wins': 0, 'total': 0}

        # record pre-fight win percentage
        def get_pct(fighter, method_type):
            stats = fighter_history[fighter][method_type]
            return stats['wins'] / stats['total'] if stats['total'] > 0 else 0.0

        red_ko_pct.append(get_pct(red, 'KO'))
        red_dec_pct.append(get_pct(red, 'DEC'))
        red_sub_pct.append(get_pct(red, 'SUB'))
        blue_ko_pct.append(get_pct(blue, 'KO'))
        blue_dec_pct.append(get_pct(blue, 'DEC'))
        blue_sub_pct.append(get_pct(blue, 'SUB'))

        if pd.isna(method):
            continue

        # determine method type
        method_type = None
        for m in ['KO', 'DEC', 'SUB']:
            if m in method:
                method_type = m
                break
        if method_type is None:
            continue

        # update total appearances
        fighter_history[red][method_type]['total'] += 1
        fighter_history[blue][method_type]['total'] += 1

        # update wins
        if row['winner'] == 1:
            fighter_history[red][method_type]['wins'] += 1
        elif row['winner'] == 0:
            fighter_history[blue][method_type]['wins'] += 1

    return np.column_stack([
        red_ko_pct, red_dec_pct, red_sub_pct,
        blue_ko_pct, blue_dec_pct, blue_sub_pct
    ])

def method_losses(df):
    fighter_dic = defaultdict(lambda: defaultdict(int))  # Initialize counts
    
    # Lists to store the results
    decision_losses_red = []
    ko_losses_red = []
    sub_losses_red = []
    total_red = []

    decision_losses_blue = []
    ko_losses_blue = []
    sub_losses_blue = []
    total_blue = []

    for _, row in df.iterrows():
        red = row['fighter_red']
        blue = row['fighter_blue']

        # Initialize dictionary for each fighter
        if red not in fighter_dic:
            fighter_dic[red] = defaultdict(int)
        if blue not in fighter_dic:
            fighter_dic[blue] = defaultdict(int)

        # Append current losses before the fight
        decision_losses_red.append(fighter_dic[red]['dec_losses'])
        ko_losses_red.append(fighter_dic[red]['ko_losses'])
        sub_losses_red.append(fighter_dic[red]['sub_losses'])
        total_red.append(fighter_dic[red]['total_losses'])

        decision_losses_blue.append(fighter_dic[blue]['dec_losses'])
        ko_losses_blue.append(fighter_dic[blue]['ko_losses'])
        sub_losses_blue.append(fighter_dic[blue]['sub_losses'])
        total_blue.append(fighter_dic[blue]['total_losses'])

        if pd.isna(row['method']):
            continue

        # Update the loss counts based on the current fight
        # If red lost
        if row['winner'] == 0:
            if 'DEC' in row['method']:
                fighter_dic[red]['dec_losses'] += 1
            if 'KO' in row['method']:
                fighter_dic[red]['ko_losses'] += 1
            if 'SUB' in row['method']:
                fighter_dic[red]['sub_losses'] += 1
            fighter_dic[red]['total_losses'] += 1

        # If blue lost
        if row['winner'] == 1:
            if 'DEC' in row['method']:
                fighter_dic[blue]['dec_losses'] += 1
            if 'KO' in row['method']:
                fighter_dic[blue]['ko_losses'] += 1
            if 'SUB' in row['method']:
                fighter_dic[blue]['sub_losses'] += 1
            fighter_dic[blue]['total_losses'] += 1

    # Stack the results into an array for assignment
    return np.column_stack([
        decision_losses_red,
        ko_losses_red,
        sub_losses_red,
        decision_losses_blue,
        ko_losses_blue,
        sub_losses_blue
    ])


def max_rating_won_against(df: pd.DataFrame, rating_type: str = 'elo'):
    """
    Tracks the highest opponent rating each fighter has defeated.

    Args:
        df (pd.DataFrame): Must contain 'fighter_red', 'fighter_blue', 'winner', and rating columns like
                           'elo_red', 'elo_blue' or 'glicko_red', 'glicko_blue'.
        rating_type (str): 'elo' or 'glicko' to select which rating to track.

    Returns:
        np.ndarray: Two columns: max opponent rating red has beaten, max opponent rating blue has beaten
    """
    if rating_type not in ['elo', 'glicko']:
        raise ValueError("rating_type must be 'elo' or 'glicko'")

    red_col = f"{rating_type}_red"
    blue_col = f"{rating_type}_blue"

    fighter_max_rating = defaultdict(lambda: 0)  # Default max rating = 0

    max_rating_red = []
    max_rating_blue = []

    for _, row in df.iterrows():
        red = row['fighter_red']
        blue = row['fighter_blue']
        rating_red = row[red_col]
        rating_blue = row[blue_col]
        winner = row['winner']  # 1 = red, 0 = blue

        # Append current max opponent rating before this fight
        max_rating_red.append(fighter_max_rating[red])
        max_rating_blue.append(fighter_max_rating[blue])

        # Update max opponent rating if fighter won
        if winner == 1:  # Red won
            try: 
                fighter_max_rating[red] = max(fighter_max_rating[red], rating_blue)
            except: 
                fighter_max_rating[red] = None
        elif winner == 0:  # Blue won
            try:
                fighter_max_rating[blue] = max(fighter_max_rating[blue], rating_red)
            except:
                fighter_max_rating[blue] = None
    return np.column_stack([max_rating_red, max_rating_blue])


def avg_fight_time(df_):
    """
    assumes total fight time is current fight
    """
    df = df_.copy()

    fight_time_dict = defaultdict(list)

    avg_min_red = []
    avg_min_blue = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        avg_min_red.append(
                            np.mean([t for t in fight_time_dict[red_fighter] if t is not None])
                            if len(fight_time_dict[red_fighter]) > 0 else np.nan
                        )
        
        avg_min_blue.append(
                            np.mean([t for t in fight_time_dict[blue_fighter] if t is not None])
                            if len(fight_time_dict[blue_fighter]) > 0 else np.nan
                        )

        fight_time_dict[red_fighter].append(row['total_fight_time_red'])
        fight_time_dict[blue_fighter].append(row['total_fight_time_blue'])

    return np.column_stack([avg_min_red, avg_min_blue])

def title_fights_stats_columns(df_):
    """
    returns 0 for first time fights, 0 for fighers with no title fights 
    """
    df = df_.copy()
    
    # Ensure title_fight is numeric
    df['title_fight'] = df['title_fight'].astype(int)
    
    # Encode wins for each fighter directly on the main df
    df['red_win'] = (df['winner'] == 1).astype(int)  # Red won
    df['blue_win'] = (df['winner'] == 0).astype(int)  # Blue won
    
    # Total title fights per fighter (cumulative, excluding current fight)
    df['red_title_fights'] = df.groupby('fighter_red')['title_fight'].cumsum().shift(fill_value=0)
    df['blue_title_fights'] = df.groupby('fighter_blue')['title_fight'].cumsum().shift(fill_value=0)
    
    # Total title wins per fighter (cumulative, excluding current fight)
    df['red_title_wins'] = df.groupby('fighter_red')['red_win'].cumsum().shift(fill_value=0)
    df['blue_title_wins'] = df.groupby('fighter_blue')['blue_win'].cumsum().shift(fill_value=0)
    
    # Calculate losses
    df['red_title_losses'] = df['red_title_fights'] - df['red_title_wins']
    df['blue_title_losses'] = df['blue_title_fights'] - df['blue_title_wins']
    
    # Calculate win percentage
    df['red_title_win_pct'] = df['red_title_wins'] / df['red_title_fights'].replace(0, np.nan)
    df['blue_title_win_pct'] = df['blue_title_wins'] / df['blue_title_fights'].replace(0, np.nan)
    
    # Fill NaN win pct with 0 for fighters with no prior title fights
    df['red_title_win_pct'] = df['red_title_win_pct'].fillna(0)
    df['blue_title_win_pct'] = df['blue_title_win_pct'].fillna(0)
    
    # Return as numpy column stack
    return df[['red_title_fights','blue_title_fights',
               'red_title_wins','blue_title_wins',
               'red_title_losses','blue_title_losses',
               'red_title_win_pct','blue_title_win_pct']].to_numpy()