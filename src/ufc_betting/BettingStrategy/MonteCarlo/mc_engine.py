import numpy as np 
import pandas as pd 

def blend_dataframes(
    dfs,
    probs,
    kind="kelly",
    date_col="date",
    random_state=None
):
    
    # random number generator 
    rng = np.random.default_rng(random_state)

    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()

    # stack each odds type column wise 
    if kind == "kelly":
        valid = np.column_stack([
            ((df["choice_ev"] > 0) | (df["choice_fstar"] > 0)).to_numpy()
            for df in dfs
        ])

    elif kind == "parlay":
        # parlay result stored in each row
        dfs = [
            df.groupby(date_col, as_index=False).first()
            for df in dfs
        ]

        valid = np.column_stack([
            (df["parlay_ev"] > 0).to_numpy()
            for df in dfs
        ])

    else:
        raise ValueError("kind must be 'kelly' or 'parlay'")

    # get boolean where there is a valid bet 
    keep_rows = valid.any(axis=1)

    # get proba weights for each row 
    weights = valid[keep_rows] * probs
    weights = weights / weights.sum(axis=1, keepdims=True)

    # cumsum over the rows 
    cum_weights = np.cumsum(weights, axis=1)

    # get array of random values between 0 and 1, len weights.shape[0]
    random_vals = rng.random(weights.shape[0])

    # random_vals[:, None] adds column dimension, (weights.shape[0], 1) 
    # now can do vectorized boolean comparison over the column dimension 
    # because of the cumsum, this sum gives the index matched to the proba sampled and given proba for each df 
    chosen_df_idx = (random_vals[:, None] > cum_weights).sum(axis=1)

    # np.where returns integer indices for both rows and columns 
    kept_positions = np.where(keep_rows)[0]
    selected_rows = [
        dfs[df_idx].iloc[row_idx]
        for row_idx, df_idx in zip(kept_positions, chosen_df_idx)
    ]

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def simulate(
        kelly_arr, 
        parlay_arr, 
        probs, 
        init_bankroll=500, 
        n_sims=1000,
        len_temp=3
):

    all_parlay = []
    all_ml = []
    all_total = []

    for _ in range(n_sims):

        random_state = np.random.randint(0, 1_000_000)

        df_kelly = blend_dataframes(kelly_arr, probs, kind='kelly', random_state=random_state)
        df_parlay = blend_dataframes(parlay_arr, probs, kind='parlay', random_state=random_state)

        shuffled_dates = np.random.permutation(df_kelly['date'].unique())
        df_randomized = pd.concat(
                [df_kelly[df_kelly['date'] == d] for d in shuffled_dates],
                ignore_index=True
            )
        parlay_groups = dict(tuple(df_parlay.groupby('date', sort=False)))

        parlay_results = [] 
        ml_results = []
        cum_bankroll = [init_bankroll]

        temp_cum = []
        temp_ml = []
        temp_parlay = []
        
        for date, group in df_randomized.groupby('date', sort=False):
            
            parlay_date = parlay_groups.get(date)
            curr_bankroll = cum_bankroll[-1] 
            
            if curr_bankroll <= 0:
                curr_bankroll = 0

            ml_fstar_net = group.copy()
            ml_decimal_odds = np.where(
                            ml_fstar_net['pred_winner'] == ml_fstar_net['winner'],
                            ml_fstar_net['choice_decimal_odds'],
                            2
                        )
            ml_pl = (ml_fstar_net['fstar_net'] * (ml_decimal_odds - 1) * curr_bankroll).sum()

            parlay_fstar_net = parlay_date['fstar_net'].iloc[-1]
            parlay_odds = parlay_date['parlay_net_odds'].iloc[-1] if parlay_fstar_net > 0 else 1 
            parlay_pl = parlay_fstar_net * (parlay_odds) * curr_bankroll

            total_pl = ml_pl + parlay_pl
            
            temp_cum.append(total_pl)
            temp_ml.append(ml_pl)
            temp_parlay.append(parlay_pl)

            if len(temp_cum) == len_temp:
                cum_bankroll.append(cum_bankroll[-1] + np.sum(temp_cum))
                ml_results.append(np.sum(temp_ml))
                parlay_results.append(np.sum(temp_parlay))

                temp_cum = []
                temp_ml = []
                temp_parlay = []


        all_ml.append(np.array(ml_results))
        all_parlay.append(np.array(parlay_results))
        all_total.append(np.array(cum_bankroll))

    return all_ml, all_parlay, all_total 



