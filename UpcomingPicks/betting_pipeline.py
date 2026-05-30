
import numpy as np
import pandas as pd

from BettingStrategy.kelly_worker import parlay_top_ev, run_per_bet_scaling
from UpcomingPicks.set_column_names import set_ml_bets_cols, set_parlay_cols, get_ml_bet_cols, get_parlay_cols
from UpcomingPicks.test_helpers import df_bets_tests, df_parlay_tests
from UpcomingPicks.model_helpers import logit_predict, xgboost_predict
from UpcomingPicks.data_helpers import get_X_stacked, merge_bets_types, merge_parlay_types, get_bets_input, get_bets_pkt, get_parlay_input, get_parlay_pkt
from config import config

PARLAY_SIZE = config.parlay_top_ev


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
        print('XGBoost Stacking')
        print(bets_input_df['f_star_unscaled'])
        print(bets_input_df['choice_edge'])
        print(bets_input_df['choice_ev'])
        print(bets_input_df['choice_proba'])
        print(bets_input_df['p'])

        df_per_bet = run_per_bet_scaling(
            bets_df=bets_input_df, 
            max_drawdown=mdd_ml, 
            bankroll=bankroll, 
            N=N_ml
        )
        print(df_per_bet['fstar_scaled'])
        
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
            fighter_blue=fighter_blue
        )

        df_parlay = parlay_top_ev(
            parlay_input_df, 
            bankroll, 
            type, 
            top_n=[0,1],
            parlay_mdd = mdd_parlay,
            N = N_parlay
        )
        print(df_parlay[f'stake_{type}'])

        parlay_pkt = get_parlay_pkt(
            df_parlay=df_parlay, 
            type=type
        )
        
        df_parlay_final = set_parlay_cols(
            type_=f'{type}_stack', 
            pkt=parlay_pkt, 
            required_idx=np.arange(PARLAY_SIZE), 
            all_na=False
        )
        
        df_parlay_tests(
            df_parlay=df_parlay_final, 
            choice_ev=bets_input_df['choice_ev']
        )
        
        stacked_parlay = merge_parlay_types(df_parlay_final, stacked_parlay)

    return stacked_ml, stacked_parlay 


def betting_pipeline(
        upcoming_df, 
        feats_list, 
        model_list, 
        xgb_stack, 
        scaler_list, 
        type_list, 
        fair_odds_list, 
        real_odds_list, 
        bankroll, 
        mdd_ml_arr,
        mdd_parlay_arr, 
        N_ml_arr,
        N_parlay_arr,
        mdd_ml_stack_arr,
        mdd_parlay_stack_arr,
        N_ml_stack_arr,
        N_parlay_stack_arr
):

    other_cols = ['date', 'fighter_red', 'fighter_blue', 'open_red', 'open_blue', 'close1_red', 'close2_red', 'close1_blue', 'close2_blue']
    df = upcoming_df.copy().reset_index(drop=True)
    required_df_idx = df.index

    df_bets_combined = pd.DataFrame(df[other_cols].values, columns=other_cols, index=required_df_idx)
    df_parlay_combined = pd.DataFrame()
    df_proba = pd.DataFrame(columns=['proba_red_open', 'proba_blue_open', 'proba_red_close1', 'proba_blue_close1', 'proba_red_close2','proba_blue_close2'], 
                            index=required_df_idx)

    fighter_red = df["fighter_red"].values
    fighter_blue = df["fighter_blue"].values
    dates = df["date"].values

    data_list = []
    for i in range(len(feats_list)):
        dat = {
            'feats':feats_list[i],
            'model':model_list[i],
            'scaler':scaler_list[i],
            'type':type_list[i],
            'fair_odds':fair_odds_list[i],
            'real_odds':real_odds_list[i],
            'mdd_ml':mdd_ml_arr[i],
            'mdd_parlay':mdd_parlay_arr[i],
            'N_ml':N_ml_arr[i],
            'N_parlay':N_parlay_arr[i]
        }
        data_list.append(dat)
    

    for i, dat in enumerate(data_list):
        
        model = dat['model']
        scaler = dat['scaler']
        feats = dat['feats']
        type = dat['type']
        real_odds = dat['real_odds']
        fair_odds = dat['fair_odds']
        mdd_ml = dat['mdd_ml']
        mdd_parlay = dat['mdd_parlay']
        N_ml = dat['N_ml']
        N_parlay = dat['N_parlay']

        valid_mask = ~df[feats].isna().any(axis=1)
        y_hat = pd.Series(0, index=required_df_idx, dtype=float)

        # split features by dtype
        num_feats = df[feats].select_dtypes(exclude='category').columns
        cat_feats = df[feats].select_dtypes(include='category').columns
        df_valid_num = df.loc[valid_mask, num_feats]

        if df_valid_num.shape[0] == 0:
            proba_red = pd.Series(np.nan, index=required_df_idx)
            proba_blue = pd.Series(np.nan, index=required_df_idx)
            df_proba[f'proba_red_{type}'] = proba_red
            df_proba[f'proba_blue_{type}'] = proba_blue

            df_bets = set_ml_bets_cols(type, {}, required_df_idx, all_na=True)
            df_bets_combined = merge_bets_types(df_bets, df_bets_combined)

            df_parlay = set_parlay_cols(type, {}, np.arange(2), all_na=True)
            df_parlay_combined = merge_parlay_types(df_parlay, df_parlay_combined)
            continue

        y_hat = logit_predict(
            model=model, 
            df=df, 
            y_hat=y_hat, 
            feats=feats, 
            num_feats=num_feats, 
            cat_feats=cat_feats, 
            valid_mask=valid_mask, 
            scaler=scaler, 
            required_df_idx=required_df_idx
        )

        proba_red = y_hat
        proba_blue = 1 - y_hat
        pred_winner_bool = (proba_red >= 0.5).astype('Int64') 

        df_proba[f'proba_red_{type}'] = proba_red
        df_proba[f'proba_blue_{type}'] = proba_blue

        bets_input_df = get_bets_input(
            df=df, 
            y_hat=pred_winner_bool, 
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
            type_=type, 
            pkt=bets_pkt, 
            required_idx=y_hat.index, 
            all_na=False
        )
        df_bets_tests(
            df_bets=df_bets, 
            df_bets_combined=df_bets_combined, 
            valid_mask=valid_mask, 
            choice_ev=bets_input_df['choice_ev'], 
            fstar_list=df_per_bet['fstar_scaled'].values
        )

        df_bets_combined = merge_bets_types(
            df_bets=df_bets, 
            df_bets_combined=df_bets_combined
        )

        parlay_input_df = get_parlay_input(
            df=df, 
            bets_input_df=bets_input_df, 
            fighter_red=fighter_red, 
            fighter_blue=fighter_blue
        )

        df_parlay = parlay_top_ev(
            parlay_input_df, 
            bankroll, 
            type, 
            top_n=[0,1],
            parlay_mdd = mdd_parlay,
            N = N_parlay
        )
        
        parlay_pkt = get_parlay_pkt(
            df_parlay, 
            type
        )

        df_parlay_final = set_parlay_cols(
            type_=type, 
            pkt=parlay_pkt, 
            required_idx=np.arange(PARLAY_SIZE), 
            all_na=False
        )
        df_parlay_tests(
            df_parlay=df_parlay_final, 
            choice_ev=bets_input_df['choice_ev']
        )
        df_parlay_combined = merge_parlay_types(
            df_parlay=df_parlay_final, 
            df_parlay_combined=df_parlay_combined
        )

    X_stacked = get_X_stacked(
        df=df, 
        df_proba=df_proba, 
        df_bets_combined=df_bets_combined, 
        required_df_idx=required_df_idx
    )

    bets_stacking, parlay_stacking = xgboost_stacking(
        xgb=xgb_stack,
        df=df,
        required_idx=required_df_idx,  
        feats=feats,
        X_stacked=X_stacked,
        fair_odds_arr=fair_odds_list[1:],
        real_odds_arr=real_odds_list[1:], 
        types_arr=type_list[1:], 
        mdd_ml_arr=mdd_ml_stack_arr,
        mdd_parlay_arr=mdd_parlay_stack_arr,
        N_ml_arr=N_ml_stack_arr,
        N_parlay_arr=N_parlay_stack_arr, 
        bankroll=bankroll
    )

    df_bets_combined = pd.concat([df_bets_combined, bets_stacking], axis=1)
    df_parlay_combined = pd.concat([df_parlay_combined, parlay_stacking], axis=1)
    return df_bets_combined, df_parlay_combined


def seperate_bets_dfs(df_bets, df_parlay, types):
    dfs = []
    dfs_parlay = []

    for type in types: 

        columns = get_ml_bet_cols(type).values()
        dfs.append(df_bets[columns])

        c_parlay = get_parlay_cols(type).values()
        dfs_parlay.append(df_parlay[c_parlay])

    return dfs, dfs_parlay
