def xgboost_stacking(
        xgb, 
        df,
        required_idx,
        feats,
        X_stacked,
        fair_odds_arr, 
        real_odds_arr,
        types_arr,  
        mdd_ml_arr,
        mdd_parlay_arr, 
        N_ml_arr, 
        N_parlay_arr,
        bankroll
):
    result = xgboost_predict(xgb, X_stacked, required_idx)
    
    proba_red = result['proba_red']
    proba_blue = result['proba_blue']
    y_hat = result['y_hat']

    dat_list = []
    for i in range(len(fair_odds_arr)):
        dat = {
            'proba_red':proba_red,
            'proba_blue':proba_blue,
            'pred_winner':y_hat,
            'fair_odds':fair_odds_arr[i],
            'real_odds':real_odds_arr[i],
            'type':types_arr[i],
            'N_ml':N_ml_arr[i],
            'N_parlay':N_parlay_arr[i],
            'mdd_ml':mdd_ml_arr[i],
            'mdd_parlay':mdd_parlay_arr[i]
        }
        dat_list.append(dat)

    stacked_ml = pd.DataFrame(index=required_idx)
    stacked_parlay = pd.DataFrame()

    fighter_red = df["fighter_red"].values
    fighter_blue = df["fighter_blue"].values

    for dat in dat_list:

        valid_mask = ~df[feats].isna().any(axis=1)

        real_odds = dat['real_odds']
        fair_odds = dat['fair_odds']
        N_ml = dat['N_ml']
        N_parlay = dat['N_parlay']
        mdd_ml = dat['mdd_ml']
        mdd_parlay = dat['mdd_parlay']
        type = dat['type']
        
        bets_input_df = get_bets_input(
            df=df, 
            y_hat=y_hat, 
            proba_red=proba_red,
            proba_blue=proba_blue, 
            real_odds=real_odds, 
            fair_odds=fair_odds
        )

        df_per_bet = run_per_bet_scaling(
            bets_df=bets_input_df, 
            max_drawdown=mdd_ml, 
            bankroll=bankroll, 
            N=N_ml
        )
        
        bets_pkt = get_bets_pkt(
            bets_input_df=bets_input_df, 
            df_per_bet=df_per_bet
            )

        df_bets = set_ml_bets_cols(
            type_=f'{type}_stack', 
            pkt=bets_pkt, 
            required_idx=y_hat.index, 
            all_na=False
        ) 

        df_bets_tests(
            df_bets=df_bets, 
            df_bets_combined=stacked_ml, 
            valid_mask=valid_mask, 
            choice_ev=bets_input_df['choice_ev'], 
            fstar_list=df_per_bet['fstar_scaled']
        )

        stacked_ml = merge_bets_types(
            df_bets=df_bets, 
            df_bets_combined=stacked_ml
        )

        parlay_input_df = get_parlay_input(
            df=df,
            bets_input_df=bets_input_df,
            fighter_red=fighter_red, 
            fighter_blue=fighter_blue,
            required_idx=required_idx
        )

        df_parlay = parlay_top_ev(
            parlay_input_df, 
            bankroll, 
            type, 
            top_n=[0,1],
            parlay_mdd = mdd_parlay,
            N = N_parlay
        )
        # print(df_parlay[f'stake_{type}'])

        parlay_pkt = get_parlay_pkt(
            df_parlay=df_parlay, 
            type=type
        )
        
        df_parlay_final = set_parlay_cols(
            type_=f'{type}_stack', 
            pkt=parlay_pkt, 
            required_idx=df_parlay.index, 
            all_na=False
        )
        
        df_parlay_tests(
            df_parlay=df_parlay_final, 
            choice_ev=bets_input_df['choice_ev']
        )

        # created fight index for stacked bets 
        stacked_parlay = merge_parlay_types(df_parlay_final, stacked_parlay, odds_type=f'{type}_stack')

    return stacked_ml, stacked_parlay 