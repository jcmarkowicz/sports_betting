
import numpy as np 
import pandas as pd 

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from set_column_names import get_parlay_cols


def parlay_top_ev(data, bankroll, type, top_n=[0,1]):

    data = data[data['choice_ev'] > 0]

    if data.shape[0] < 2:
        df_parlay = get_parlay_cols(type, {}, np.arange(len(top_n)), all_na=True)
        return df_parlay
    
    df_top_n = data.sort_values(by='choice_ev', ascending=False).iloc[top_n].copy()

    parlay_prob = np.prod(df_top_n['choice_proba'])
    parlay_odds = np.prod(df_top_n['choice_real_odds'])
    net_odds = np.prod(df_top_n['choice_real_odds'])-1
    parlay_ev = parlay_prob * parlay_odds - 1

    b = parlay_odds - 1
    parlay_kelly = max((b * parlay_prob - (1 - parlay_prob)) / b, 0)
    stake = bankroll * parlay_kelly

    df_top_n[f'parlay_prob_{type}'] = parlay_prob
    df_top_n[f'parlay_ev_{type}'] = parlay_ev

    df_top_n[f'choice_fighter_name_{type}'] = data['choice_fighter_name'].values

    df_top_n[f'parlay_fstar_{type}'] = parlay_kelly
    df_top_n[f'stake_{type}'] = stake
    df_top_n[f'parlay_odds_{type}'] = net_odds

    df_top_n = df_top_n[[f'choice_fighter_name_{type}', 
                         f'parlay_fstar_{type}',
                         f'parlay_odds_{type}', 
                         f'stake_{type}', 
                         f'parlay_ev_{type}', 
                         f'parlay_prob_{type}']]
    return df_top_n


def run_per_bet_scaling(bets_df, max_drawdown, bankroll, N):

    idx = bets_df.index

    # initialize outputs aligned to original df
    fstar = pd.Series(np.nan, index=idx)
    stake = pd.Series(np.nan, index=idx)
    potential_profit = pd.Series(np.nan, index=idx)

    # define rows where computation is valid (no NaNs in required cols)
    required_cols = ["f_star_unscaled", "p", "fair_odds", "real_odds", "ev"]
    valid_mask = bets_df[required_cols].notna().all(axis=1)

    for i in bets_df[valid_mask].index:
        row = bets_df.loc[i]

        f_star = row["f_star_unscaled"]
        p = row["p"]
        fair_odds = row["fair_odds"]
        real_odds = row["real_odds"]
        ev = row["ev"]

        if f_star <= 0 or ev <= 0:
            f_final = f_star
            s = 0

        else:
            edge = p - (1/(fair_odds))
            adj_mdd = scale_mdd(edge, max_drawdown)
            
            f_final = scale_kelly_for_mdd(p, fair_odds, f_star, N=N, max_drawdown=adj_mdd)
            s = bankroll * f_final
            
        fstar.loc[i] = f_final
        stake.loc[i] = s
        potential_profit.loc[i] = s * (real_odds - 1)

    return pd.DataFrame({
        "fstar_scaled": fstar,
        "stake": stake,
        "potential_profit": potential_profit
    })

def scale_mdd(edge, mdd):
    adj_mdd = mdd
    if 0.1 <= edge <= 0.15:
        adj_mdd += .02
    elif edge > .15 and edge < .2:
        adj_mdd += .05
    elif edge >= .2 and edge < .25:
        adj_mdd += .1
    elif edge >= .25:
        adj_mdd += .15

    return adj_mdd

def kelly_edge(p, fair_decimal):

    if p <= 0 or p >= 1:
        return 0.0  # invalid probability

    b = fair_decimal - 1  # net odds
    q = 1 - p

    f = (b * p - q) / b
    return max(0, f)


def expected_value(p, o):
    EV = p * (o - 1) - (1 - p) * 1
    return EV 

def log_return_volatility(f, b, p):
    """
    Compute per-bet log-return volatility (sigma) and expected log return (mu).
    
    f : fraction of bankroll bet
    b : net odds (decimal odds - 1)
    p : probability of winning
    """
    r_win = np.log(1 + f * b)
    r_lose = np.log(1 - f)
    mu = p * r_win + (1 - p) * r_lose
    sigma2 = p * (r_win - mu)**2 + (1 - p) * (r_lose - mu)**2
    sigma = np.sqrt(sigma2)
    return sigma, mu


def expected_max_drawdown(sigma, N):
    """
    Heuristic for expected maximum drawdown over N bets
    """
    emdd = sigma * (
        np.sqrt(
            2*np.log(N) -
            (np.log(np.log(N)) + np.log(4*np.pi)) /
            (2*np.sqrt(2*np.log(N)))
        )
    )
    return emdd


def scale_kelly_for_mdd(p, odds, f_full, N, max_drawdown, tol=1e-4):
    """
    Find the largest fraction of full Kelly that keeps expected MDD <= max_drawdown
    
    p : probability of winning
    odds : decimal odds
    f_full : full Kelly fraction (fraction of bankroll)
    N : number of bets
    max_drawdown : tolerable drawdown fraction (0 < max_drawdown < 1)
    tol : numerical tolerance for convergence
    """
    b = odds - 1
    # binary search between 0 and 1 (fraction of full Kelly)
    low, high = 0.0, 1.0
    best_fraction = 0.0
    
    while high - low > tol:
        k = (low + high) / 2
        f_trial = k * f_full
        sigma, mu = log_return_volatility(f_trial, b, p)
        mdd_est = expected_max_drawdown(sigma, N)
        
        if mdd_est <= max_drawdown:
            best_fraction = k  # this fraction is safe, try higher
            low = k
        else:
            high = k  # too aggressive, try lower
            
    return best_fraction * f_full

def check_neighbors(df_history, df_upcoming, feature_cols, n_neighbors=5):
    X_hist = df_history[feature_cols].values
    X_up = df_upcoming.values

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(X_hist)                 
    distances, indices = nn.kneighbors(X_up)  

    rows = []
    for i, up_idx in enumerate(df_upcoming.index):
        for k in range(n_neighbors):
            hist_row = df_history.iloc[indices[i, k]]
            rows.append({
                "upcoming_index": up_idx,
                "neighbor_rank": k + 1,
                "history_index": hist_row.name,
                "distance": distances[i, k],
                **hist_row.to_dict()   # append all history columns
            })

    neighbors_df = pd.DataFrame(rows)
    return neighbors_df
