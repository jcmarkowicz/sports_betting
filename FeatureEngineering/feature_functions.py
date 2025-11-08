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
            months_since_red.append(None)

        # For blue fighter
        last_date_blue = fighter_last_event.get(blue_fighter)
        if last_date_blue is not None:
            months_diff = (event_date.year - last_date_blue.year) * 12 + (event_date.month - last_date_blue.month)
            months_since_blue.append(months_diff)
        else:
            months_since_blue.append(None)

        # Update last seen dates
        fighter_last_event[red_fighter] = event_date
        fighter_last_event[blue_fighter] = event_date

    # Add columns back to dataframe if you want

    return np.column_stack([months_since_red, months_since_blue])

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
    
def count_fav_dog(df):
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

    return np.column_stack([red_fav_counts, red_dog_counts, blue_fav_counts, blue_dog_counts])
                     
def td_ratio(df):

    ratio_dic = defaultdict(lambda: defaultdict(lambda: [None]))
    td_dic = defaultdict(lambda: [None])

    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        red_ratio.append(td_dic[red_fighter][-1])
        blue_ratio.append(td_dic[blue_fighter][-1])

        ratio_dic[red_fighter]['fighter_td'].append(row['td_landed_total_red'])
        ratio_dic[red_fighter]['opponent_td'].append(row['td_landed_total_blue'])

        ratio_dic[blue_fighter]['fighter_td'].append(row['td_landed_total_blue'])
        ratio_dic[blue_fighter]['opponent_td'].append(row['td_landed_total_red'])

        red_td = np.sum([v for v in ratio_dic[red_fighter]['fighter_td'] if v is not None])
        red_opponent_td = np.sum([v for v in ratio_dic[red_fighter]['opponent_td'] if v is not None])

        blue_td = np.sum([v for v in ratio_dic[blue_fighter]['fighter_td'] if v is not None])
        blue_opponent_td = np.sum([v for v in ratio_dic[blue_fighter]['opponent_td'] if v is not None])

        red_ratio_curr = red_td / red_opponent_td if red_opponent_td != 0 else red_td 
        blue_ratio_curr = blue_td / blue_opponent_td if blue_opponent_td !=0 else blue_td

        td_dic[red_fighter].append(red_ratio_curr)
        td_dic[blue_fighter].append(blue_ratio_curr)

    return np.column_stack([red_ratio, blue_ratio])


def sig_strikes_ratio(df):

    ratio_dic = defaultdict(lambda: defaultdict(lambda: [None]))
    stats_dic = defaultdict(lambda: [None])

    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        red_ratio.append(stats_dic[red_fighter][-1])
        blue_ratio.append(stats_dic[blue_fighter][-1])

        ratio_dic[red_fighter]['fighter_control'].append(row['sig_str_landed_total_red'])
        ratio_dic[red_fighter]['opponent_control'].append(row['sig_str_landed_total_blue'])

        ratio_dic[blue_fighter]['fighter_control'].append(row['sig_str_landed_total_blue'])
        ratio_dic[blue_fighter]['opponent_control'].append(row['sig_str_landed_total_red'])

        red_control = np.sum([v for v in ratio_dic[red_fighter]['fighter_control'] if v is not None])
        red_opponent_control = np.sum([v for v in ratio_dic[red_fighter]['opponent_control'] if v is not None])

        blue_control = np.sum([v for v in ratio_dic[blue_fighter]['fighter_control'] if v is not None])
        blue_opponent_control = np.sum([v for v in ratio_dic[blue_fighter]['opponent_control'] if v is not None])

        red_ratio_curr = red_control / red_opponent_control if red_opponent_control != 0 else red_control 
        blue_ratio_curr = blue_control / blue_opponent_control if blue_opponent_control !=0 else blue_control

        stats_dic[red_fighter].append(red_ratio_curr)
        stats_dic[blue_fighter].append(blue_ratio_curr)

    return np.column_stack([red_ratio, blue_ratio])

def control_pr_ratio(df):

    control_stats = defaultdict(lambda: defaultdict(lambda: []))
    ratio_dic = defaultdict(lambda: [None])

    red_ratio = []
    blue_ratio = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        red_ratio.append(ratio_dic[red_fighter][-1])
        blue_ratio.append(ratio_dic[blue_fighter][-1])

        control_stats[red_fighter]['fighter_control'].append(row['control_pm_red'])
        control_stats[red_fighter]['opponent_control'].append(row['control_pm_blue'])

        control_stats[blue_fighter]['fighter_control'].append(row['control_pm_blue'])
        control_stats[blue_fighter]['opponent_control'].append(row['control_pm_red'])


        red_control = np.sum([v for v in control_stats[red_fighter]['fighter_control'] if v is not None and not pd.isna(v)])
        red_opponent_control = np.sum([v for v in control_stats[red_fighter]['opponent_control'] if v is not None and not pd.isna(v)])

        blue_control = np.sum([v for v in control_stats[blue_fighter]['fighter_control'] if v is not None and not pd.isna(v)])
        blue_opponent_control = np.sum([v for v in control_stats[blue_fighter]['opponent_control'] if v is not None and not pd.isna(v)])


        red_ratio_curr = red_control / red_opponent_control if red_opponent_control != 0 else red_control 
        blue_ratio_curr = blue_control / blue_opponent_control if blue_opponent_control !=0 else blue_control

        ratio_dic[red_fighter].append(red_ratio_curr)
        ratio_dic[blue_fighter].append(blue_ratio_curr)

    return np.column_stack([red_ratio, blue_ratio])

def total_knockdowns(df):
    fighter_dic = defaultdict(list)
    red_kd = []
    blue_kd = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        if len(fighter_dic[red_fighter])==0:
            red_kd.append(None)
        else:
            red_kd.append(sum(fighter_dic[red_fighter]))

        if len(fighter_dic[blue_fighter])==0:
            blue_kd.append(None)
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
            red_bonus.append(None)
        else: 
            red_bonus.append(np.sum([x for x in fighter_dic[red_fighter] if x is not None]))
        
        if len(fighter_dic[blue_fighter]) == 0:
            blue_bonus.append(None)
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

    return weight_classes 

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

def avg_fight_time(df_):
    df = df_.copy()

    fight_time_dict = defaultdict(list)

    avg_min_red = []
    avg_min_blue = []

    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        avg_min_red.append(
                            np.mean([t for t in fight_time_dict[red_fighter] if t is not None])
                            if len(fight_time_dict[red_fighter]) > 0 else None
                        )
        avg_min_blue.append(
                            np.mean([t for t in fight_time_dict[blue_fighter] if t is not None])
                            if len(fight_time_dict[blue_fighter]) > 0 else None
                        )

        fight_time_dict[red_fighter].append(row['total_fight_time_red'])
        fight_time_dict[blue_fighter].append(row['total_fight_time_blue'])

    return np.column_stack([avg_min_red, avg_min_blue])