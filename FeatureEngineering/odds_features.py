import numpy as np 
import pandas as pd 
from math import isfinite

def implied_probs_from_decimal(odds):
    """Vectorized implied probabilities from decimal odds."""
    odds = np.asarray(odds, dtype=float)
    return 1.0 / odds

def devig_normalize(probs):
    """
    Simple normalization (a.k.a. proportional scaling).
    Works for 2+ outcomes. Sum of probs becomes 1.
    """
    probs = np.asarray(probs, dtype=float)
    s = probs.sum()
    if s <= 0 or not isfinite(s):
        raise ValueError("Invalid probabilities for normalization.")
    return probs / s

def devig_power(p_imp, tol=1e-12, max_iter=200):
    """
    'Power' or 'Harville' style reweighting: find exponent alpha so that
    sum(p_i^alpha) = 1. Returns p_fair_i ∝ p_i^alpha.
    Works for 2+ outcomes. More flexible than simple normalization
    when bookmakers’ overround is not proportional.
    """
    probs = np.asarray(p_imp, dtype=float)
    if np.any(probs <= 0):
        raise ValueError("All probs must be > 0 for power devig.")
    # Binary search on alpha
    lo, hi = 0.0, 5.0  # broad range; increase hi if needed
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = np.sum(probs ** mid)
        if abs(s - 1.0) < tol:
            alpha = mid
            break
        if s > 1.0:
            # Need larger alpha to shrink sum
            alpha = None
            lo = mid
        else:
            hi = mid
    alpha = mid if alpha is None else alpha
    fair = probs ** alpha
    return fair / fair.sum()

def devig_two_way(dec_a, dec_b, method="normalize"):
    """
    Devig a two-outcome market.
    Returns fair probabilities (pA, pB) and fair decimal odds (oddsA, oddsB).
    """
    imp = implied_probs_from_decimal([dec_a, dec_b])
    if method == "normalize":
        pf = devig_normalize(imp)
    elif method == "power":
        pf = devig_power(imp)
    # elif method == "shin":
    #     pf = devig_shin(imp)
    else:
        raise ValueError("method must be one of: normalize, power, shin")
    fair_odds = 1.0 / pf
    return pf[0], pf[1], fair_odds[0], fair_odds[1], imp

def american_to_decimal(odds):
    return np.where(odds > 0, odds / (100 + 1), 100 /( np.abs(odds) + 1))

def run_devig(df, red_odds_c, blue_odds_c, dec_new_red, dec_new_blue, p_new_red, p_new_blue, p_imp_red, p_imp_blue, method='power'):
    pfair_red = []
    pfair_blue = []
    dfair_red = []
    dfair_blue = []
    pimp_red = []
    pimp_blue = []

    for i, row in df.iterrows(): 
        red_odds = row[red_odds_c]
        blue_odds = row[blue_odds_c]

        pf_red, pf_blue, df_red, df_blue, imp_red_blue = devig_two_way(red_odds, blue_odds, method)
        pfair_red.append(pf_red)
        pfair_blue.append(pf_blue)
        dfair_red.append(df_red)
        dfair_blue.append(df_blue)
        pimp_red.append(imp_red_blue[0])
        pimp_blue.append(imp_red_blue[1])

    df[dec_new_red] = dfair_red
    df[dec_new_blue] = dfair_blue
    df[p_new_red] = pfair_red
    df[p_new_blue] = pfair_blue
    df[p_imp_red] = pimp_red
    df[p_imp_blue] = pimp_blue
    
    return df
def american_to_decimal(odds):
    return np.where(odds > 0, odds / 100 + 1, 100 / np.abs(odds) + 1)


def build_odds_features(df_):
    df = df_.copy()

    n_close = 2
    colors = ['red', 'blue']

    for color in colors: 
        df[f'dec_open_{color}'] = american_to_decimal(df[f'open_{color}'])
        for i in range(1, n_close+1):
            df[f'dec_close{i}_{color}'] = american_to_decimal(df[f'close{i}_{color}'])
    
    df = run_devig(df, 'dec_open_red', 'dec_open_blue', 'dec_fair_open_red', 'dec_fair_open_blue', \
                'proba_fair_open_red', 'proba_fair_open_blue', 'pimp_open_red', 'pimp_open_blue')

    df = run_devig(df, 'dec_close1_red', 'dec_close1_blue', 'dec_fair_close1_red', 'dec_fair_close1_blue', \
                'proba_fair_close1_red', 'proba_fair_close1_blue', 'pimp_close1_red', 'pimp_close1_blue')

    df = run_devig(df, 'dec_close2_red', 'dec_close2_blue', 'dec_fair_close2_red', 'dec_fair_close2_blue', \
                'proba_fair_close2_red', 'proba_fair_close2_blue', 'pimp_close2_red', 'pimp_close2_blue')
    
    # calculate juice
    for color in colors:
        df[f'juice_open_{color}'] = df[f'pimp_open_{color}'] - df[f'proba_fair_open_{color}']
        for i in range(1, n_close+1):
            df[f'juice_close{i}_{color}'] = df[f'pimp_close{i}_{color}'] - df[f'proba_fair_close{i}_{color}']

    
    df['line_movement_close1_red'] = df['open_red'] - df['close1_red']
    df['line_movement_close1_blue'] = df['open_blue'] - df['close1_blue']

    df['line_movement_close2_red'] = df['open_red'] - df['close2_red']
    df['line_movement_close2_blue'] = df['open_blue'] - df['close2_blue']

    df['red_ud_to_fav_close1'] = (df['open_red'] > 0) & (df['close1_red'] < 0).astype(int)
    df['red_ud_to_fav_close2'] = (df['open_red'] > 0) & (df['close2_red'] < 0).astype(int)

    df['red_stayed_fav_close1'] = (df['open_red'] < 0) & (df['close1_red'] < 0).astype(int)
    df['red_stayed_fav_close2'] = (df['open_red'] < 0) & (df['close2_red'] < 0).astype(int)

    df['red_fav_to_ud_close1'] = (df['open_red'] < 0) & (df['close1_red'] > 0).astype(int)
    df['red_fav_to_ud_close2'] = (df['open_red'] < 0) & (df['close2_red'] > 0).astype(int)

    df['red_stayed_dog_close1'] = (df['open_red'] > 0) & (df['close1_red'] > 0).astype(int)
    df['red_stayed_dog_close2'] = (df['open_red'] > 0) & (df['close2_red'] > 0).astype(int)

    df['blue_ud_to_fav_close1'] = ((df['open_blue'] > 0) & (df['close1_blue'] < 0)).astype(int)
    df['blue_ud_to_fav_close2'] = ((df['open_blue'] > 0) & (df['close2_blue'] < 0)).astype(int)

    df['blue_stayed_fav_close1'] = ((df['open_blue'] < 0) & (df['close1_blue'] < 0)).astype(int)
    df['blue_stayed_fav_close2'] = ((df['open_blue'] < 0) & (df['close2_blue'] < 0)).astype(int)

    df['blue_fav_to_ud_close1'] = ((df['open_blue'] < 0) & (df['close1_blue'] > 0)).astype(int)
    df['blue_fav_to_ud_close2'] = ((df['open_blue'] < 0) & (df['close2_blue'] > 0)).astype(int)

    df['blue_stayed_dog_close1'] = ((df['open_blue'] > 0) & (df['close1_blue'] > 0)).astype(int)
    df['blue_stayed_dog_close2'] = ((df['open_blue'] > 0) & (df['close2_blue'] > 0)).astype(int)

    return df

    