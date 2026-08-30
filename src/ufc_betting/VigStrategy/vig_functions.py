
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt


from math import isfinite
from scipy.optimize import brentq


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
    elif method == "shin":
        pf = devig_shin(imp)
    else:
        raise ValueError("method must be one of: normalize, power, shin")
    fair_odds = 1.0 / pf
    return pf[0], pf[1], fair_odds[0], fair_odds[1]


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


def devig_shin(p_imp, tol=1e-8, debug=False):
    """
    Convert implied probabilities to fair probs using Shin (1993).
    Works for 2+ outcomes.
    """
    p_imp = np.asarray(p_imp, dtype=float)
    s = p_imp.sum()

    if s <= 1 + tol:
        raise ValueError(f"Implied probs sum to {s:.3f}, must be > 1 for Shin method.")

    def f(z):
        return np.sum(p_imp / (1.0 - z * (1.0 - p_imp / s))) - 1.0

    if debug:
        print(f"Testing f(z) at 0: {f(0):.6f}, at 1-tol: {f(1-tol):.6f}")

    try:
        z = brentq(f, 0, 1 - tol)
    except ValueError as e:
        raise RuntimeError(f"Root not bracketed: {e}. Check inputs {p_imp}.") from e

    fair = p_imp / (1.0 - z * (1.0 - p_imp / s))
    return fair / fair.sum()


def fair_odds_curve_df(df,
                       red_col="dec_open_red",
                       blue_col="dec_open_blue",
                       method="shin",   # "normalize" | "power" | "shin"
                       n_bins=25):
    """
    For each row, devig the two-way market, compute fair probs & fair odds.
    Then bin by fair probability of 'red' and aggregate to a smooth curve.
    Returns (per_row, curve) dataframes.
    """
    rows = []
    for _, r in df.iterrows():
        da = float(r[red_col])
        db = float(r[blue_col])
        pA_imp, pB_imp = 1/da, 1/db
        pA_fair, pB_fair, oA_fair, oB_fair = devig_two_way(da, db, method=method)
        rows.append({
            "dec_open_red": da,
            "dec_open_blue": db,
            "p_imp_red_raw": pA_imp,       # <-- raw, not normalized
            "p_imp_blue_raw": pB_imp,
            "p_imp_red": pA_imp / (pA_imp + pB_imp),  # normalized implied red
            "p_imp_blue": pB_imp / (pA_imp + pB_imp),
            "p_fair_red": pA_fair,
            "p_fair_blue": pB_fair,
            "odds_fair_red": oA_fair,
            "odds_fair_blue": oB_fair
        })
    per_row = pd.DataFrame(rows)

    # Bin by fair probability of red and summarize the corresponding fair odds
    per_row["p_bin"] = pd.cut(per_row["p_fair_red"], bins=n_bins, include_lowest=True)
    curve = per_row.groupby("p_bin").agg(
        p_fair_red_mean=("p_fair_red", "mean"),
        odds_fair_red_median=("odds_fair_red", "median"),
        p_imp_red_mean=("p_imp_red", "mean"),              # for comparison
        n=("p_fair_red", "size")
    ).reset_index(drop=True)

    # Add the theoretical curve (1/p) for the averaged p_fair in each bin
    curve["odds_theoretical"] = 1.0 / curve["p_fair_red_mean"]
    return per_row, curve


