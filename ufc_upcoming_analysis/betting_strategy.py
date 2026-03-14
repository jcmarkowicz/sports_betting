import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_per_bet_scaling(bets_df, max_drawdown, bankroll, N, max_k=0, sort_max_wins=False):

    rows = []
    fstar_list = []
    stake_list = []

    if sort_max_wins is True:
        bets_df = bets_df.sort_values(by=['ev'], ascending=False).reset_index(drop=True)

    for _, row in bets_df.iterrows():

        f_star = row["f_star_unscaled"]
        p = row["p"]
        fair_odds = row["fair_odds"]
        real_odds = row["real_odds"]
        ev = row["ev"]

        if f_star <= 0 or ev < 0 or (sort_max_wins is True and row.name >= max_k):
            f_final = f_star
            stake = 0
        else:
            
            edge = p - (1/(fair_odds))
            if edge >= .1 and edge <=.15:
                max_drawdown += .02
            
            elif edge > .15 and edge <.2:
                max_drawdown += .05

            elif edge >= .2 and edge <.25:
                max_drawdown += .1
            
            elif edge >= .25:
                max_drawdown += .15

            f_final = scale_kelly_for_mdd(p, fair_odds, f_star, N=N, max_drawdown=max_drawdown)
            stake = bankroll * f_final
            
        fstar_list.append(f_final)
        stake_list.append(stake)

        profit_if_win = stake * real_odds-1
        rows.append({ "p": p, "fair_odds": fair_odds, "real_odds": real_odds, "ev": ev,"f_star_unscaled": f_star,"f_star_scaled": f_final,
            "stake": stake, 'profit_if_win':profit_if_win
        })

    bets_out_df = pd.DataFrame(rows)
    return fstar_list, stake_list 


def kelly_edge(p, fair_decimal):

    if p <= 0 or p >= 1:
        return 0.0  # invalid probability

    b = fair_decimal - 1  # net odds
    q = 1 - p

    f = (b * p - q) / b

    # Never return a negative fraction (means no value bet)
    return max(0, f)


def parlay_top_ev(data, bankroll, type, top_n=[0,1]):

    if data.shape[0] < 2:
        df_parlay = pd.DataFrame({
            f'choice_fighter_name_{type}': [pd.NA],
            f'parlay_fstar_{type}': [pd.NA],
            f'parlay_odds_{type}': [pd.NA],
            f'stake_{type}': [pd.NA],
            f'parlay_ev_{type}': [pd.NA],
            f'parlay_prob_{type}': [pd.NA]
        })
        return df_parlay
    
    df_top_n = data.sort_values(by='choice_ev', ascending=False).iloc[top_n].copy()

    parlay_prob = np.prod(df_top_n['choice_proba'])
    parlay_odds = np.prod(df_top_n['choice_real_odds'])
    parlay_ev = parlay_prob * parlay_odds - 1

    b = parlay_odds - 1
    parlay_kelly = max((b * parlay_prob - (1 - parlay_prob)) / b, 0)
    net_odds = np.prod(df_top_n['choice_real_odds'])-1

    stake = bankroll * parlay_kelly
    parlay_avg_edge = np.mean(df_top_n['choice_proba'] - (1/df_top_n['choice_real_odds']))

    df_top_n[f'parlay_prob_{type}'] = np.ones(len(df_top_n)) * parlay_prob
    df_top_n[f'parlay_ev_{type}'] = np.ones(len(df_top_n)) * parlay_ev
    
    df_top_n[f'choice_fighter_name_{type}'] = np.where(df_top_n['pred_winner']==1, df_top_n['fighter_red'], df_top_n['fighter_blue'])

    df_top_n[f'parlay_fstar_{type}'] = np.ones(len(df_top_n)) * parlay_kelly
    df_top_n[f'stake_{type}'] = np.ones(len(df_top_n)) * stake 
    
    df_top_n[f'parlay_odds_{type}'] = np.ones(len(df_top_n)) * net_odds

    df_top_n = df_top_n[[f'choice_fighter_name_{type}', f'parlay_fstar_{type}', f'parlay_odds_{type}', f'stake_{type}', f'parlay_ev_{type}', f'parlay_prob_{type}']]
           
    return df_top_n


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
    return sigma * np.sqrt(2 * np.log(N))


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


def betting_pipeline(upcoming_df, feats_list, model_list, scaler_list, type_list, fair_odds_list, real_odds_list, bankroll, max_drawdown=0.15, N=1000):

    other_cols = ['fighter_red', 'fighter_blue', 'date', 'close1_red', 'close2_red', 'close1_blue', 'close2_blue']
    all_feats_list = list(set(feat for outer in feats_list for feat in outer))
    df = upcoming_df.copy()
    column_names = all_feats_list+other_cols

    df_bets_combined = pd.DataFrame(df[all_feats_list+other_cols].values, columns=column_names)
    df_parlay_combined = pd.DataFrame()

    for model, scaler, feats, type, fair_odds, real_odds in zip(model_list, scaler_list, feats_list, type_list, real_odds_list, fair_odds_list):

        valid_mask = ~df[feats].isna().any(axis=1)
        y_hat = pd.Series(0, index=df.index, dtype=float)

        # split features by dtype
        num_feats = df[feats].select_dtypes(exclude='category').columns
        cat_feats = df[feats].select_dtypes(include='category').columns
        df_valid_num = df.loc[valid_mask, num_feats]

        if df_valid_num.shape[0] == 0:

            choice_proba_col = f'choice_proba_{type}'
            choice_fstar_col = f'fstar_{type}'
            choice_stake_col = f'stake_{type}'
            proba_red_col = f'proba_red_{type}'
            proba_blue_col = f'proba_blue_{type}'
            pred_winner_col = f'pred_winner_{type}'

            df_bets = pd.DataFrame({f'pred_name_{type}': [pd.NA], pred_winner_col: [pd.NA], 
                                    choice_proba_col:[pd.NA], choice_fstar_col:[pd.NA], choice_stake_col:[pd.NA],
                                    f'edge_{type}':[pd.NA], f'ev_{type}':[pd.NA]})
            
            df_bets_combined = pd.concat([df_bets_combined, df_bets.reset_index(drop=True)], axis=1)

            df_parlay = pd.DataFrame({f'choice_fighter_name_{type}':[pd.NA], f'parlay_fstar_{type}':[pd.NA], f'parlay_odds_{type}':[pd.NA], f'stake_{type}':[pd.NA], f'parlay_ev_{type}':[ pd.NA], f'parlay_prob_{type}':[pd.NA]})
            df_parlay_combined = pd.concat([df_parlay_combined, df_parlay.reset_index(drop=True)], axis=1)

            continue

        # scale numeric features
        scaled_num = pd.DataFrame(
            scaler.transform(df.loc[valid_mask, num_feats]),
            columns=num_feats,
            index=df.loc[valid_mask].index
        )

        # keep categorical features unchanged
        cat_data = df.loc[valid_mask, cat_feats]

        # combine them
        scaled_valid = pd.concat([scaled_num, cat_data], axis=1)

        # optional: keep original column order
        scaled_valid = scaled_valid[feats]

        X_valid = scaled_valid
        X_valid = sm.add_constant(X_valid)

        train_cols = model.model.exog_names              # includes 'const' if you used it
        X_valid = X_valid.reindex(columns=train_cols, fill_value=0)

        # Now predict
        y_hat.loc[valid_mask] = model.predict(X_valid)
        y_hat.loc[~valid_mask] = None

        proba_red_col = f'proba_red_{type}'
        proba_blue_col = f'proba_blue_{type}'
        pred_winner_col = f'pred_winner_{type}'

        df[proba_red_col] = y_hat
        df[proba_blue_col] = 1 - y_hat
        df[pred_winner_col] = np.where(df[proba_red_col] >= 0.5, 1, 0)
        pred_winner_names = np.where(df[proba_red_col] >= 0.5, df['fighter_red'], df['fighter_blue'])

        choice_proba = np.where(df[pred_winner_col] == 1, df[proba_red_col], df[proba_blue_col])
        choice_fair_odds = np.where(df[pred_winner_col] == 1, df[fair_odds[1]], df[fair_odds[0]])
        choice_edge = choice_proba - (1/choice_fair_odds)

        choice_real_odds = np.where(df[pred_winner_col] == 1, df[real_odds[1]], df[real_odds[0]])
        choice_ev = expected_value(choice_proba, choice_real_odds)
        unweighted_fstar = np.array([kelly_edge(p, fair_odds) for p, fair_odds in zip(choice_proba, choice_fair_odds)])
        choice_idx = df[pred_winner_col].values

        bets_input_df = pd.DataFrame({
                "p": choice_proba,
                "fair_odds": choice_fair_odds,
                "real_odds": choice_real_odds,
                "ev": choice_ev,
                "f_star_unscaled": unweighted_fstar
            })
    
        fstar_list, stake_list = run_per_bet_scaling(bets_input_df, max_drawdown, bankroll, N)
        choice_proba_col = f'choice_proba_{type}'
        choice_fstar_col = f'fstar_{type}'
        choice_stake_col = f'stake_{type}'

        print(len(pred_winner_names), len(df[pred_winner_col].values), len(choice_proba), len(fstar_list), len(stake_list), len(choice_edge), len(choice_ev))

        df_bets = pd.DataFrame({f'pred_name_{type}': pred_winner_names, pred_winner_col: df[pred_winner_col].values, 
                                choice_proba_col:choice_proba, choice_fstar_col:fstar_list, choice_stake_col:stake_list,
                                f'edge_{type}':choice_edge, f'ev_{type}':choice_ev})

        df_bets_combined = pd.concat([df_bets_combined, df_bets.reset_index(drop=True)], axis=1)
        
        parlay_input_df = pd.DataFrame({'pred_winner':df[pred_winner_col], 'choice_ev':choice_ev,
                                    'choice_real_odds':choice_real_odds, 'choice_fstar':df_bets[choice_fstar_col],
                                    'fighter_red': df['fighter_red'], 
                                    'fighter_blue': df['fighter_blue'], 
                                    'date':df['date'], 'choice_proba':choice_proba})

        df_parlay = parlay_top_ev(parlay_input_df, bankroll, type, top_n=[0,1])
        df_parlay_combined = pd.concat([df_parlay_combined, df_parlay.reset_index(drop=True)], axis=1)

    return df_bets_combined, df_parlay_combined


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


def seperate_bets_dfs(df_bets, df_parlay, types):
    dfs = []
    dfs_parlay = []
    for type in types: 
        columns = [f'fighter_red', f'fighter_blue', f'pred_name_{type}', f'pred_winner_{type}', 
                   f'choice_proba_{type}', f'{type}_red', f'{type}_blue', f'fstar_{type}', f'stake_{type}', 
                   f'ev_{type}', f'edge_{type}']
        
        df_type = df_bets[columns]
        dfs.append(df_type)

        c_parlay = [f'choice_fighter_name_{type}', f'parlay_fstar_{type}', f'parlay_odds_{type}', 
                    f'stake_{type}', f'parlay_ev_{type}', f'parlay_prob_{type}']
        
        dfs_parlay.append(df_parlay[c_parlay])
    return dfs, dfs_parlay
