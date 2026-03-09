from datetime import datetime, date 
import numpy as np
import pandas as pd

import re 

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
    # subtract 1 if birthday hasn't happened yet this year
    if (current_date.month, current_date.day) < (dob.month, dob.day):
        age -= 1

    return age 

def height_inches(height):

    if height == '--':
        return None

    if pd.isna(height) is True: 
        return pd.NA

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

    ufc_df['age_red'] = ufc_df.apply(lambda row: current_age(row['dob_red'], row['date']), axis=1)
    ufc_df['age_blue'] = ufc_df.apply(lambda row: current_age(row['dob_blue'], row['date']), axis=1)

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
