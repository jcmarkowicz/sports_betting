import sys
import os
from pandas.api.types import is_string_dtype

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np 
import pandas as pd 

import re
from datetime import datetime, date 
from collections import defaultdict

from RatingAlgos.elo import elo_rating
from RatingAlgos.glicko import glicko_rating
from RatingAlgos.glicko2 import glicko2_run
from RatingAlgos.elo_scope import run_elo_on_matches

from FeatureEngineering.feature_functions import total_bonus, sig_strikes_ratio, td_ratio,control_pr_ratio,\
      womens_fight, mma_math, win_lose_streak, months_since_last, method_wins, method_win_pct,\
      avg_fight_time, title_fights_stats_columns, method_losses, max_rating_won_against

# winners_df = pd.read_csv(r'C:\Users\jcmar\my_files\SportsBetting\winners.csv')
# ufc_df = pd.read_csv(r'C:\Users\jcmar\my_files\SportsBetting\scraped_data_ufc.csv')

def get_event_country(dat):
    "take event location and extract the country"
    parts = dat.split(',')
    country = parts[-1]
    return country 

def is_valid_april_format(date_str):
    datetime.strptime(date_str, "%B %d, %Y")
    return True

def get_wins_losses(dat):
    parts = dat.split('-')
    wins = int(parts[0])
    losses = int(parts[1])
    return wins, losses 

def current_age(dob, current_date):
    if dob == '--':
        return None
    dob = pd.to_datetime(dob)
    current_date = pd.to_datetime(current_date)
    
    age = current_date.year - dob.year
    return age 

def height_inches(height):
    height = height.replace("HEIGHT: ","")
    match = re.match(r"(\d+)' (\d+)", height)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        total_inches = feet * 12 + inches
        return total_inches

def reach_inches(reach):
    if reach == '--':
        return None
    clean = reach.replace('"', '').strip()
    return int(clean)

def SL_pct(dat):
    "significant strikes landed in decimal, this is the same as sig strike accuracy"
    decimal_value = float(dat.strip("%")) / 100
    return decimal_value 

def SLpM(sl, min):
    """significant strikes landed per minute, round must be when the fight finished, not scheduled rounds, all rounds are 5 min just round
    sl => 68 of 121
    """
    sl = int(sl.split()[0])
    slpm = sl / min #ignoring partial rounds when fight finished during, need to scrape this under TIME: (next to where round is)
    return slpm 

def SApM(opponent_sl, min):
    "Significant stikes absorbed per minute, get opponents SL "
    stikes_absorbed_pm = SLpM(opponent_sl, min)
    return stikes_absorbed_pm

def sig_strikes_landed(sl):
    """strikes landed in a single fight, format: (n_landed of n_attempted)"""
    landed = int(sl.split()[0])
    return landed 

def sig_strikes_attempted(sl):
    """strikes atttempted in a single fight, format: (n_landed of n_attempted)"""
    attempted = int(sl.split()[2])
    return attempted 

def str_defense(opponent_sl):
    opponent = SL_pct(opponent_sl)
    defense = 1 - opponent
    return defense 

def knock_downs_pm(kd, min):
    kd_pm = kd / min
    return kd_pm

def kd_landed(dat):
    """format: single val"""
    return int(dat)

def leg_strikes(dat):
    """format: n_landed of n_attempted"""
    leg_strikes = dat.split(' of ')
    return int(leg_strikes[0])

def head_strikes(dat):
    """format: n_landed of n_attempted"""
    head_strikes = dat.split(' of ')
    return int(head_strikes[0]) 

def body_strikes(dat):
    """format: n_landed of n_attempted"""
    body_strikes = dat.split(' of ')
    return int(body_strikes[0])

def clinch_strikes(dat):
    """format: n_landed of n_attempted"""
    clinch_strikes = dat.split(' of ')
    return int(clinch_strikes[0])

def td_percent(dat):
    if dat == '---':
        return None 
    else: 
        decimal_value = float(dat.strip("%")) / 100
        return decimal_value 

def td_landed(dat):
    tdl = int(dat.split()[0])
    return tdl 

def td_attempted(dat):
    tda = int(dat.split()[2])
    return tda 

def td_defense(opponents_td):
    if opponents_td == '---':
        return .55 #prior if no prio attempts on fighter 
    else:
        opponent_percent = td_percent(opponents_td) #decimal value 
        return 1 - opponent_percent
    
def sub_attemtped(subatt):
    "submissions attempted per round"
    sub = int(subatt) 
    return sub

def reverse(reverse):
    """how often a fighter reverses position per round, indicator of good ground game"""
    rev = int(reverse)
    return rev

def minsecs_to_float(time_str):
    """
    Convert a 'min:secs' string to total minutes as a float.
    
    Example:
        '3:30' -> 3.5
    """
    minutes, seconds = map(int, time_str.split(":"))
    return minutes + seconds / 60

def get_years_past(date_str, ref_year):
    try:
        # Explicitly try "April 22, 2004" style first
        try:
            date_obj = datetime.strptime(str(date_str).strip(), "%B %d, %Y")
        except ValueError:
            date_obj = pd.to_datetime(date_str, errors="raise")

        return ref_year - date_obj.year

    except Exception as e:
        print(f"Error parsing YEARS PAST '{date_str}': {e}")
        return None

def parse_date(date_str):
    "pass in event date and get how many years since that event"
    try:
        # If already datetime.datetime or pandas Timestamp, return as is
        if isinstance(date_str, (datetime, pd.Timestamp, date)):
            return date_str
        
        # If it's a string, parse it
        elif isinstance(date_str, str):
            # Try your custom format
            if date_name_format(date_str):  # your custom checker
                date_new = datetime.strptime(date_str, "%B %d, %Y")
                return date
            else:
                # fallback to pandas
                date_new = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(date_new):
                    raise ValueError("Invalid date format or missing value")
                return date_new
        else:
            raise TypeError(f"Unsupported type: {type(date_str)}")
        
    except Exception as e:
        print(f"Error parsing date EVENT DATE '{date_str}': {e}")
        return None
    
def date_name_format(date_str):
    try:
        datetime.strptime(date_str, "%B %d, %Y")
        return True
    except ValueError:
        return False
#event

def upcoming_event_features(ufc_df):
    current_year = float(datetime.now().year)
    ufc_df['date'] = ufc_df['event_date'].apply(parse_date)
    ufc_df["event_location"] = ufc_df["event_location"].apply(get_event_country)

    ufc_df['height_red'] = ufc_df['height_red'].apply(height_inches)
    ufc_df['height_blue'] = ufc_df['height_blue'].apply(height_inches)
    ufc_df['reach_red'] = ufc_df['reach_red'].apply(reach_inches)
    ufc_df['reach_blue'] = ufc_df['reach_blue'].apply(reach_inches)
    ufc_df['red_age'] = ufc_df['dob_red'].apply(lambda x: current_age(x, current_year))
    ufc_df['blue_age'] = ufc_df['dob_blue'].apply(lambda x: current_age(x, current_year))
    ufc_df['title_fight'] = ufc_df['title_fight'].astype(float)
    return ufc_df

def single_event_features(webscrape_df):
    """Pass in webscrape df, calculate features per single event"""
    current_year = float(datetime.now().year)
    ufc_df = webscrape_df.copy()

    # --- EVENT FEATURES ---
    ufc_df["event_location"] = ufc_df["event_location"].apply(get_event_country)
    ufc_df["event_age"] = ufc_df["event_date"].apply(lambda x: get_years_past(x, current_year))
    ufc_df["date"] = ufc_df["event_date"].apply(parse_date)
    ufc_df["fight_minutes"] = ufc_df['fight_time'].apply(minsecs_to_float)
    ufc_df['title_fight'] = ufc_df['title_fight'].astype(float)

    # --- BASIC FIGHTER ATTRIBUTES ---
    for color in ["red", "blue"]:
        ufc_df[f"height_{color}"] = ufc_df[f"height_{color}"].apply(height_inches)
        ufc_df[f"reach_{color}"] = ufc_df[f"reach_{color}"].apply(reach_inches)
        ufc_df[[f"wins_{color}", f"losses_{color}"]] = (ufc_df[f"record_{color}"].apply(lambda x: pd.Series(get_wins_losses(x))))
        ufc_df[f"age_{color}"] = ufc_df.apply(lambda row: current_age(row[f"dob_{color}"], row['date']), axis=1)

    # --- PERFORMANCE BONUSES ---
    for col in ["performance_bonus_winner", "fight_otn_bonus"]:
        ufc_df[col] = ufc_df[col].astype(float)

    time_col = 'fight_minutes'
    colors = ["red", "blue"]
    strike_features = {}
    grappling_features = {}

    for color in colors:
        opp_color = "blue" if color == "red" else "red"

        strike_features.update({
            f"sig_str_landed_{color}": lambda row, c=color: sig_strikes_landed(row[f"sig_str_{c}"]),
            f"sig_str_attempted_{color}": lambda row, c=color: sig_strikes_attempted(row[f"sig_str_{c}"]),
            f"sig_str_absorbed_{color}": lambda row, c=color, o=opp_color: sig_strikes_landed(row[f"sig_str_{o}"]),

            # these are landed 
            f"kd_{color}": lambda row, c=color: kd_landed(row[f"kd_{c}"]),
            f"leg_str_{color}": lambda row, c=color: leg_strikes(row[f"leg_{c}"]),
            f"head_str_{color}": lambda row, c=color: head_strikes(row[f"head_{c}"]),
            f"body_str_{color}": lambda row, c=color: body_strikes(row[f"body_{c}"]),
            f"clinch_str_{color}": lambda row, c=color: clinch_strikes(row[f"clinch_{c}"])
        })

        grappling_features.update({
            f"td_landed_{color}": lambda row, c=color: td_landed(row[f"td_{c}"]),
            f"td_attempted_{color}": lambda row, c=color: td_attempted(row[f"td_{c}"]),
            f"td_defended_{color}": lambda row, c=color, o=opp_color: td_defense(row[f"td_pct_{o}"]),

            f"control_{color}": lambda row, c=color: minsecs_to_float(row[f"ctrl_{c}"]),
            f"sub_att_{color}": lambda row, c=color: sub_attemtped(row[f"sub_att_{c}"]),
            f"reverse_{color}": lambda row, c=color: reverse(row[f"rev_{c}"]),
        })

    # Apply all grappling features to the DataFrame
    for col_name, func in grappling_features.items():
        ufc_df[col_name] = ufc_df.apply(func, axis=1)
    
    for col_name, func in strike_features.items():
        ufc_df[col_name] = ufc_df.apply(func, axis=1)    
    
    return ufc_df

def compute_defense_features(fighter_name, row, feat, color, df_dict, stats_dict, time_col):
    opp = 'red' if color == 'blue' else 'red'

    if len(stats_dict[fighter_name][time_col]) == 0:
        pct_feat = None
        attemtped_against = None
        landed_against = None
    else: 
        attemtped_against = np.sum(stats_dict[fighter_name][f'{feat}_total_attempted_against'])
        landed_against =  np.sum(stats_dict[fighter_name][f'{feat}_total_landed_against'])
        if attemtped_against == 0:
            pct_feat = 1
        else: 
            pct_feat = 1 - (landed_against / attemtped_against)

    df_dict[f'{feat}_defense_pct_{color}'].append(pct_feat)
    df_dict[f'{feat}_total_attempted_against_{color}'].append(attemtped_against) 
    df_dict[f'{feat}_total_landed_against_{color}'].append(landed_against)

    stats_dict[fighter_name][f'{feat}_total_attempted_against'].append(row[f'{feat}_attempted_{opp}'])
    stats_dict[fighter_name][f'{feat}_total_landed_against'].append(row[f'{feat}_landed_{opp}'])

    return stats_dict, df_dict

def prefight_stats(stats_dict, df_dict, fighter_name, feature, row, time_col, color):
    
    if len(stats_dict[fighter_name][time_col]) == 0:
        time_feature = None
        total_feature = None
        pm_feature = None

    else: 
        time_feature = np.sum(stats_dict[fighter_name][time_col])
        total_feature = np.sum(stats_dict[fighter_name][feature])
        pm_feature = total_feature / time_feature

    df_dict[f'{feature}_pm_{color}'].append(pm_feature)
    df_dict[f'{feature}_total_{color}'].append(total_feature)

    stats_dict[fighter_name][feature].append(row[f'{feature}_{color}'])

    return stats_dict, df_dict 

def compute_accuracy_stats(fighter_name, row, feat, color, df_dict, stats_dict, time_col):
    if len(stats_dict[fighter_name][time_col]) == 0:
        acc_pct = None
    else: 
        total_att = np.sum(stats_dict[fighter_name][f'{feat}_attempted'])
        total_land = np.sum(stats_dict[fighter_name][f'{feat}_landed'])

        if total_att == 0:
            acc_pct = 0
        else:
            acc_pct = total_land / total_att

    df_dict[f'{feat}_accuracy_pct_{color}'].append(acc_pct)
    return df_dict
    
def apply_rolling_stats(ufc_features): 
    """
    Take in df of precomputed features that reflect current fight stats, and apply rolling average to get pre fight stats
    
    Args: 
        ufc_features (pd.DataFrame): DataFrame with single event features computed
    Returns:
        final_df (pd.DataFrame): DataFrame with rolling features computed
    """  
    
    print('UFC FEATURES WINNER NA:', ufc_features['winner'].isna().sum())

    stats_dict = defaultdict(lambda: defaultdict(list))
    df_dict = defaultdict(list)

    per_fight_features = ufc_features.copy()
    per_fight_features['date'] = pd.to_datetime(per_fight_features['date'])
    per_fight_features = per_fight_features.sort_values(by='date', ascending=True).reset_index(drop=True)

    striking_features = ['kd', 'sig_str_landed', 'sig_str_absorbed', 'sig_str_attempted', 'leg_str', 'head_str', 'body_str', 'clinch_str']
    grapling_features = ['td_landed', 'td_attempted', 'td_defended', 'control', 'sub_att', 'reverse']
    defense_features = ['td', 'sig_str']
    accuracy_features = ['td', 'sig_str']

    fighter_attr = ['age', 'height', 'reach']
    general_features = ['date', 'event_location', 'weight_class', 'title_fight']
    win_features = ["performance_bonus_winner", "fight_otn_bonus", 'method', 'winner', 'fighter_red', 'fighter_blue']
    colors = ['red', 'blue']
    time_col = 'fight_minutes'

    for _, row in per_fight_features.iterrows(): 
    
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        
        for feat in defense_features: 
            stats_dict, df_dict = compute_defense_features(red_fighter, row, feat,'red', df_dict, stats_dict, time_col)
            stats_dict, df_dict = compute_defense_features(blue_fighter, row, feat,'blue', df_dict, stats_dict, time_col)

        # this needs to go before prefight stats, prefight stats updates a column this function references, 
        for feat in accuracy_features: 
            df_dict = compute_accuracy_stats(red_fighter, row, feat,'red', df_dict, stats_dict, time_col)
            df_dict = compute_accuracy_stats(blue_fighter, row, feat,'blue', df_dict, stats_dict, time_col)

        for feature in striking_features + grapling_features:
            stats_dict, df_dict = prefight_stats(stats_dict, df_dict, red_fighter, feature, row, time_col, 'red')
            stats_dict, df_dict = prefight_stats(stats_dict, df_dict, blue_fighter, feature, row, time_col, 'blue') # time updated in stats dict here 

        for color in colors:
            fighter = red_fighter if color=='red' else blue_fighter
            time_feature = np.sum(stats_dict[fighter][time_col])

            df_dict[f'total_fight_time_{color}'].append(time_feature)
            stats_dict[fighter][time_col].append(row[time_col])

        for i, attr in enumerate(fighter_attr): 
            df_dict[f'{attr}_red'].append(row[f'{attr}_red'])
            df_dict[f'{attr}_blue'].append(row[f'{attr}_blue'])

        for feat in general_features: 
            df_dict[feat].append(row[feat])

        for win_feat in win_features:
            if win_feat == 'winner': 
                if row['winner'] == red_fighter: 
                    color = 1.0
                elif row['winner'] == blue_fighter:
                    color = 0.0
                elif row['winner'] == 'NC' or row['winner'] == 'DRAW':
                    color = 2.0
                elif pd.isna(row['winner']):
                    color = 3.0

                df_dict['winner'].append(color)
                df_dict['winner_name'].append(row['winner'])

            else: 
                df_dict[win_feat].append(row[win_feat])

    test = [(key, len(value)) for key, value in df_dict.items() if len(value)!=8007]
    print(test)
    final_df = pd.DataFrame(df_dict)
    print('WINNER FINAL DF', final_df['winner'])
    return final_df 

def compute_differences(df, feature, type):
    if type is not None: 
        df[f'{feature}_{type}_diff'] = df[f'{feature}_{type}_red'] - df[f'{feature}_{type}_blue']
    else: 
        df[f'{feature}_diff'] = df[f'{feature}_red'] - df[f'{feature}_blue']
    return df

def non_rolling_stats(df_):
    """
    Take in df with rolling features, compute non rolling features that dont depend on time series.
    See feature_functions.py for helper functions, Rating_Algos folder.
    Compute differences between red and blue fighter for relevant features.

    Args:
        df_ (pd.DataFrame): DataFrame with rolling features computed
    Returns:
        df (pd.DataFrame): DataFrame with non rolling features computed
    """

    df = df_.copy()

    striking_features = ['kd', 'sig_str_landed', 'sig_str_absorbed', 'sig_str_attempted' , 'leg_str', 'head_str', 'body_str', 'clinch_str']
    grapling_features = ['td_landed', 'td_attempted', 'control', 'sub_att', 'reverse']
    striking_grapling_types = ['total', 'pm']
    fighter_attr = ['age', 'height', 'reach']

    for feature in striking_features + grapling_features: 
        for type in striking_grapling_types:
            df = compute_differences(df, feature, type)

    for attr in fighter_attr: 
        df = compute_differences(df, attr, None)

    striking_features = ['sig_str_defense', 'sig_str_accuracy']
    grapling_features = ['td_accuracy', 'td_defense']
    striking_grapling_types = ['pct']
    for feature in striking_features + grapling_features: 
        for type in striking_grapling_types:
            df = compute_differences(df, feature, type)

    # avg fight time in minutes 
    df[['avg_fight_min_red', 'avg_fight_min_blue']] = avg_fight_time(df)
    df['avg_fight_min_diff'] = df['avg_fight_min_red'] - df['avg_fight_min_blue']

    # total bonus earned by fighter 
    df[['total_bonus_red', 'total_bonus_blue']] = total_bonus(df)
    df['total_bonus_diff'] = df['total_bonus_red'] - df['total_bonus_blue']

    # rating algorithms 
    # df[['elo_red', 'elo_blue','elo_proba_red','elo_proba_blue']] = elo_rating(df, 130, 320)

    params = {'base_k':55,'mov_mode':'linear', 'cutoff_rating':1800, 'cutoff_k_scale':1.0, 'w90':None, 'regress_to_mean':.05, 'regress_every_n_matches':1000}
    df_elo = run_elo_on_matches(df,**params)
    df[['elo_red', 'elo_blue']] = df_elo[['elo_pre_red', 'elo_pre_blue']].values


    df[['elo_red', 'elo_blue']] = elo_rating(df, k=100, w90=250)
    df[['elo_red_proba', 'elo_blue_proba']] = df_elo[['elo_red_proba', 'elo_blue_proba']].values

    df['elo_diff'] = df['elo_red'] - df['elo_blue']
    df['elo_pred'] = (df['elo_red'] >= df['elo_blue']).astype(int)

    df_glicko, _ = glicko2_run(df, initial_rating=1500.0,
                                    initial_rd=250.0,
                                    initial_sigma=0.06,
                                    tau=.5,
                                    inactivity_scaling_days=365.0*2)
    
    df[['glicko_red', 'glicko_blue', 
        'glicko_rd_red', 'glicko_rd_blue',
        'glicko_proba_red']] = df_glicko[['rating_red_pre', 'rating_blue_pre',
                                        'sigma_red_pre','sigma_blue_pre', 'p_red_pred']]
                                       
    # Compute blue win probability
    df['glicko_proba_blue'] = 1 - df['glicko_proba_red']
    df['glicko_diff'] = df['glicko_red'] - df['glicko_blue']

    # Boolean column: 1 if red is higher rated, else 0
    df['glicko_pred'] = (df['glicko_red'] >= df['glicko_blue']).astype(int)
    df['rating_agree'] = (df['glicko_pred'] == df['elo_pred']).astype(int)

    # MMA math 
    df[['math_red','math_blue']] = mma_math(df)

    # months since last fight 
    df[['months_since_red', 'months_since_blue']] = months_since_last(df)
    df['months_since_diff'] = df['months_since_red'] - df['months_since_blue']

    # win lose streaks, pct, num fights, num wins/losses, only in ufc  
    df[['win_streak_red','lose_streak_red',
        'win_streak_blue','lose_streak_blue',
        'win_pct_red','win_pct_blue',
        'num_fights_red','num_fights_blue',
        'num_wins_red','num_wins_blue',
        'num_losses_red','num_losses_blue'
    ]] = win_lose_streak(df)

    # win lose diffs 
    df['num_fights_diff'] = df['num_fights_red'] - df['num_fights_blue']
    df['win_streak_diff'] = df['win_streak_red'] - df['win_streak_blue']
    df['lose_streak_diff'] = df['lose_streak_red'] - df['lose_streak_blue']
    df['wins_diff'] = df['num_wins_red'] - df['num_wins_blue']
    df['losses_diff'] = df['num_losses_red'] - df['num_losses_blue']
    df['win_pct_diff'] = df['win_pct_red'] - df['win_pct_blue']

    # method wins 
    df[['decision_wins_red', 'ko_wins_red', 'sub_wins_red', 'decision_wins_blue', 'ko_wins_blue', 'sub_wins_blue']] = method_wins(df)
    win_types = ['decision_wins', 'ko_wins', 'sub_wins']
    for win in win_types: 
        df = compute_differences(df, win, None)

    # method losses
    df[['decision_losses_red',
        'ko_losses_red',
        'sub_losses_red',
        'decision_losses_blue',
        'ko_losses_blue',
        'sub_losses_blue']] = method_losses(df)
    
    df['decision_losses_diff'] = df['decision_losses_red'] - df['decision_losses_blue']
    df['ko_losses_diff'] = df['ko_losses_red'] - df['ko_losses_blue']
    df['sub_losses_diff'] = df['sub_losses_red'] - df['sub_losses_blue']

    # max elo/glicko rating won against
    df[['max_elo_win_red', 'max_elo_win_blue']] = max_rating_won_against(df, 'elo')
    df[['max_glicko_win_red', 'max_glicko_win_blue']] = max_rating_won_against(df, 'glicko')

    # method win pct
    df[['ko_pct_red', 'dec_pct_red', 'sub_pct_red',
        'ko_pct_blue', 'dec_pct_blue', 'sub_pct_blue']] = method_win_pct(df) 
    
    df['ko_pct_diff'] = df['ko_pct_red'] - df['ko_pct_blue']
    df['dec_pct_diff'] = df['dec_pct_red'] - df['dec_pct_blue']
    df['sub_pct_diff'] = df['sub_pct_red'] - df['sub_pct_blue']

    # womens fight flag 
    df['womens_fight'] = womens_fight(df)

    # ratios for total stats
    # Compute all new columns
    td_cols = td_ratio(df)
    control_cols = control_pr_ratio(df)
    sigstrike_cols = sig_strikes_ratio(df)

    # Build one DataFrame containing all new columns
    new_cols = pd.DataFrame({
        'ratio_td_red': td_cols[:,0],
        'ratio_td_blue': td_cols[:,1],
        'ratio_control_red': control_cols[:,0],
        'ratio_control_blue': control_cols[:,1],
        'ratio_sigstrike_red': sigstrike_cols[:,0],
        'ratio_sigstrike_blue': sigstrike_cols[:,1],
    })

    # Concatenate once to avoid fragmentation
    print(td_cols.shape, control_cols.shape, sigstrike_cols.shape, df.shape)
    df = pd.concat([df.reset_index(drop=True), new_cols.reset_index(drop=True)], axis=1)

    # Optional: defragment
    df = df.copy()

    df['ratio_td_diff'] = df['ratio_td_red'] - df['ratio_td_blue']
    df['ratio_control_diff'] = df['ratio_control_red'] - df['ratio_control_blue']
    df['ratio_sigstrike_diff'] = df['ratio_sigstrike_red'] - df['ratio_sigstrike_blue']

    # title fight stats
    cols_to_add = title_fights_stats_columns(df)
    new_cols = pd.DataFrame(
        cols_to_add,
        columns=[
            'red_title_fights','blue_title_fights',
            'red_title_wins','blue_title_wins',
            'red_title_losses','blue_title_losses',
            'red_title_win_pct','blue_title_win_pct'
        ]
    )

    # Concatenate in one shot -> NO fragmentation warning
    df = pd.concat([df, new_cols], axis=1)
    # Defragment
    df = df.copy()

    # get event age in days
    df['event_age'] = (pd.Timestamp.today() - df['date']).dt.days

    return df


