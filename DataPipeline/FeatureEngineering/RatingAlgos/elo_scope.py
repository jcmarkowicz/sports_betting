from math import log10

import numpy as np 
import pandas as pd 

def map_method_to_score(method):
    method = str(method).lower()
    if "ko" in method or "tko" in method:
        return 1.5
    elif "sub" in method:
        return 1.25
    elif "decision" in method:
        return 1.0
    else:
        return 1.0  # default for unknown / draw

# ----------------------------
# Helper: expected probability
# ----------------------------
def expected_prob(r_a, r_b, scale=400.0):
    # p(A wins) = 1 / (1 + 10^((Rb - Ra)/scale))
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))

# ----------------------------
# MOV scaling functions
# ----------------------------
def mov_linear(w):
    return max(w - 1.0, 1.0)

def mov_log(w):
    # avoid log(0)
    return np.log(150.0 * max(w - 1.0, 0.0) + 1.0)

def mov_sqrt(w):
    return np.sqrt(100.0 * max(w, 0.0))

def mov_exp(w):
    return 3.0 ** max(w, 0.0)

MOV_MAP = {
    "linear": mov_linear,
    "log": mov_log,
    "sqrt": mov_sqrt,
    "exp": mov_exp
}

# ----------------------------
# Single Elo run function
# ----------------------------
def mov_log(margin):
    """Margin of Victory multiplier (log version)."""
    return np.log2(margin + 2)

def run_elo_on_matches(df, base_k=20.0,
                       mov_mode="log",
                       cutoff_rating=None,
                       cutoff_k_scale=0.5,
                       w90=None,
                       regress_to_mean=0.0,
                       regress_every_n_matches=None,
                       verbose=False,
                       predict_on=None,
                       scale_override=None):

    df = df.copy().reset_index(drop=True)
    use_method = True  # optional argumen
    if use_method and 'method' in df.columns:
        df['score_a'] = df.apply(
        lambda row: map_method_to_score(row['method']) if row['winner'] == 1.0 else 0.0,
        axis=1
        )
        df['score_b'] = df.apply(
            lambda row: map_method_to_score(row['method']) if row['winner'] == 0.0 else 0.0,
            axis=1
        )
    else:
        print("ELO SCOPE ERROR")
        df['score_a'] = df['winner']
        df['score_b'] = 1 - df['winner']

    # compute scale parameter
    if scale_override is not None:
        scale = float(scale_override)
    elif w90 is not None:
        p = 0.9
        denom = log10(p / (1 - p))  # ~= log10(9) ~= 0.9542
        scale = float(w90) / denom
    else:
        scale = 400.0

    mov_func = MOV_MAP.get(mov_mode)
    if mov_func is None:
        raise ValueError("mov_mode must be one of " + ", ".join(MOV_MAP.keys()))

    # ratings store
    ratings = {}
    default_rating = 1500.0
    # optional K per player (kept simple here as constant base_k, but could be per-player)
    # processed count for regression schedule
    processed = 0

    # helper to get rating
    def get_rating(player):
        return ratings.get(player, default_rating)

    # predictions storage if asked
    preds = []
    pred_ids = []

    pre_ratings_red = []
    pre_ratings_blue = []
    fighter_red = []
    fighter_blue = []
    dates = []
    expected_probs = []
    # iterate chronologically
    for idx, row in df.iterrows():
        dates.append(row['date'])
        a = row['fighter_red']
        b = row['fighter_blue']
        sa = row.get('winner', None)
        if sa is None:
            print('winner is none')
            sa = 1 if row['score_a'] > row['score_b'] else 0
        sb = 1 - sa

        ra = get_rating(a)
        rb = get_rating(b)

        fighter_red.append(a)
        fighter_blue.append(b)
        
        pre_ratings_red.append(ra)
        pre_ratings_blue.append(rb)
        
        # expected prob using scale
        pa = expected_prob(ra, rb, scale=scale)
        expected_probs.append(pa)

        # margin of victory
        w = abs(row['score_a'] - row['score_b'])
        # ensure w>0 for some functions
        if w <= 0:
            w = 1.0

        k_mult = mov_func(w)

        # apply cutoff scaling if rating above threshold (apply if either or both > cutoff)
        effective_k = base_k * k_mult
        if cutoff_rating is not None:
            # if both above cutoff, scale down (you can change this rule)
            if ra >= cutoff_rating and rb >= cutoff_rating:
                effective_k *= cutoff_k_scale

        # update ratings
        delta = effective_k * (sa - pa)
        ratings[a] = ra + delta
        ratings[b] = rb - delta

        processed += 1
        # optional regression to mean periodically
        if regress_to_mean and regress_every_n_matches and (processed % regress_every_n_matches == 0):
            # regress all ratings toward mean rating
            if len(ratings) > 0:
                mean_rating = np.mean(list(ratings.values()))
                for k in list(ratings.keys()):
                    ratings[k] = ratings[k] + regress_to_mean * (default_rating - ratings[k])


    # If prediction required on a separate DataFrame (e.g., validation set), compute probs using final ratings
    if predict_on is not None:
        preds = []
        for _, row in predict_on.reset_index(drop=True).iterrows():
            a = row['fighter_red']; b = row['fighter_blue']
            ra = ratings.get(a, default_rating)
            rb = ratings.get(b, default_rating)
            p = expected_prob(ra, rb, scale=scale)
            preds.append(p)
        return np.array(preds), ratings


    ratings_over_time = {}
    # # Save history
    # for f in [red, blue]:
    #     if f not in ratings_over_time:
    #         ratings_over_time[f] = []
    #     ratings_over_time[f].append((idx, ratings[f]))
    red_probs = expected_probs
    blue_probs = 1 - np.array(red_probs)
    df_rating = pd.DataFrame({'elo_pre_red':pre_ratings_red, 'elo_pre_blue':pre_ratings_blue, 'fighter_red':fighter_red, 'fighter_blue':fighter_blue, 'date':dates,
                              'elo_red_proba':red_probs, 'elo_blue_proba':blue_probs})
    
    fighters_red = df_rating[['fighter_red', 'elo_pre_red','date']].rename(columns={'fighter_red': 'fighter', 'elo_pre_red': 'elo'})
    fighters_blue = df_rating[['fighter_blue', 'elo_pre_blue', 'date']].rename(columns={'fighter_blue': 'fighter', 'elo_pre_blue': 'elo'})
    all_fighters = pd.concat([fighters_red, fighters_blue], ignore_index=True)
    all_fighters = all_fighters.sort_values('date')   # or whatever your chronological column is

    top_fighters = all_fighters.groupby('fighter')['elo'].last().sort_values(ascending=False)
    top_10_fighters = top_fighters.head(50)
    # print(top_10_fighters)

    return df_rating
    # # attach to dataframe
    # df["elo_pre_red"] = elo_pre_red
    # df["elo_pre_blue"] = elo_pre_blue
    # return df