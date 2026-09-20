
import numpy as np 
import pandas as pd 
from ufc_betting.BettingStrategy.kelly_scaling import scale_kelly_for_mdd

def parlay_top_ev(data, bankroll, type, top_n=[0,1], parlay_mdd=0.5, N=250):

    if len(data) < len(top_n):
        cols = [
            f'choice_fighter_name_{type}', 
            f'parlay_fstar_{type}',
            f'parlay_odds_{type}', 
            f'stake_{type}', 
            f'parlay_ev_{type}', 
            f'parlay_prob_{type}',
            f'choice_fighter_bool_{type}',
        ]
        df_nans = pd.DataFrame({col: pd.NA for col in cols}, index=np.arange(2))        
        return df_nans
    
    df_top_n = data.sort_values(by='choice_ev', ascending=False).iloc[top_n].copy()

    parlay_prob = np.prod(df_top_n['choice_proba'].to_numpy())
    parlay_odds = np.prod(df_top_n['choice_real_odds'].to_numpy())
    net_odds = np.prod(df_top_n['choice_real_odds'].to_numpy()) - 1
    parlay_ev = parlay_prob * parlay_odds - 1

    b = parlay_odds - 1
    kelly_full = max((b * parlay_prob - (1 - parlay_prob)) / b, 0)

    parlay_kelly = kelly_full
    if parlay_mdd is not None:
        parlay_kelly = scale_kelly_for_mdd(parlay_prob, parlay_odds, kelly_full, N, parlay_mdd)

    stake = bankroll * parlay_kelly
    df_top_n[f'parlay_prob_{type}'] = parlay_prob
    df_top_n[f'parlay_ev_{type}'] = parlay_ev

    df_top_n[f'parlay_fstar_{type}'] = parlay_kelly
    df_top_n[f'stake_{type}'] = stake
    df_top_n[f'parlay_odds_{type}'] = net_odds

    df_top_n = df_top_n.rename(
        columns={"choice_fighter_bool": f"choice_fighter_bool_{type}",
                 'choice_fighter_name': f'choice_fighter_name_{type}'}
    )

    df_top_n = df_top_n[[
        f'choice_fighter_name_{type}', 
        f'parlay_fstar_{type}',
        f'parlay_odds_{type}', 
        f'stake_{type}', 
        f'parlay_ev_{type}', 
        f'parlay_prob_{type}',
        f'choice_fighter_bool_{type}',
    ]]
    return df_top_n


def run_per_bet_scaling(
        bets_df, 
        max_drawdown, 
        bankroll, 
        N
    ):

    idx = bets_df.index

    # initialize outputs aligned to original df
    fstar = pd.Series(np.nan, index=idx)
    stake = pd.Series(np.nan, index=idx)
    potential_profit = pd.Series(np.nan, index=idx)
    sigma = pd.Series(np.nan, index=idx)
    mu = pd.Series(np.nan, index=idx)
    sharpe = pd.Series(np.nan, index=idx)

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
            f_final = 0
            s = 0
            bet_sigma = 0
            bet_mu = 0

        else:
            f_final, bet_sigma, bet_mu = scale_kelly_for_mdd(
                p,
                fair_odds,
                f_star,
                N=N,
                max_drawdown=max_drawdown,
                return_sigma=True,
            )
            s = bankroll * f_final
            
        fstar.loc[i] = f_final
        stake.loc[i] = s
        potential_profit.loc[i] = s * (real_odds - 1)
        sigma.loc[i] = bet_sigma
        mu.loc[i] = bet_mu
        sharpe.loc[i] = bet_mu / bet_sigma if bet_sigma > 0 else 0.0

    return pd.DataFrame({
        "fstar_scaled": fstar,
        "stake": stake,
        "potential_profit": potential_profit,
        "sigma": sigma,
        "mu": mu,
        "sharpe": sharpe,
    })



def cap_event_exposure(moneylines, parlays, bet_type):
    """Cap one event/odds strategy at full bankroll before settlement.

    The parlay's fraction is repeated on each leg; count that ticket once.
    Apply one proportional multiplier to both fractions and dollar stakes.
    """
    moneylines = moneylines.copy()
    parlays = parlays.copy()
    ml_fraction = f'fstar_{bet_type}'
    parlay_fraction = f'parlay_fstar_{bet_type}'
    ticket_fractions = parlays[parlay_fraction].dropna()
    ticket_fraction = float(ticket_fractions.iloc[0]) if len(ticket_fractions) else 0.0
    total_fraction = float(moneylines[ml_fraction].sum()) + ticket_fraction
    multiplier = min(1.0, 1.0 / total_fraction) if total_fraction > 0 else 1.0
    moneylines[[ml_fraction, f'stake_{bet_type}']] *= multiplier
    parlays[[parlay_fraction, f'stake_{bet_type}']] *= multiplier
    return moneylines, parlays


# def check_neighbors(df_history, df_upcoming, feature_cols, n_neighbors=5):
#     X_hist = df_history[feature_cols].values
#     X_up = df_upcoming.values

#     nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
#     nn.fit(X_hist)                 
#     distances, indices = nn.kneighbors(X_up)  

#     rows = []
#     for i, up_idx in enumerate(df_upcoming.index):
#         for k in range(n_neighbors):
#             hist_row = df_history.iloc[indices[i, k]]
#             rows.append({
#                 "upcoming_index": up_idx,
#                 "neighbor_rank": k + 1,
#                 "history_index": hist_row.name,
#                 "distance": distances[i, k],
#                 **hist_row.to_dict()   # append all history columns
#             })

#     neighbors_df = pd.DataFrame(rows)
#     return neighbors_df
