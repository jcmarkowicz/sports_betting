import numpy as np 
import pandas as pd 

def calc_winner_parlay(df_parlay, df_single_event):

    winner_bool = df_single_event['winner']
    winner_name = df_single_event['winner_name']

    choice_index_open = df_parlay['fight_index_open'].astype(int)
    choice_index_close1 = df_parlay['fight_index_close1_stack'].astype(int)
    choice_index_close2 = df_parlay['fight_index_close2_stack'].astype(int)

    choice_fighters_open = df_parlay['choice_fighter_bool_open'].to_numpy(dtype=int)
    choice_fighters_close1 = df_parlay['choice_fighter_bool_close1_stack'].to_numpy(dtype=int)
    choice_fighters_close2 = df_parlay['choice_fighter_bool_close2_stack'].to_numpy(dtype=int)

    winners_bool_open = winner_bool.loc[choice_index_open].to_numpy(dtype=int)
    winner_bool_close1 = winner_bool.loc[choice_index_close1].to_numpy(dtype=int)
    winner_bool_close2 = winner_bool.loc[choice_index_close2].to_numpy(dtype=int)

    open_red = df_single_event.loc[choice_index_open]['open_red'].to_numpy()
    open_blue = df_single_event.loc[choice_index_open]['open_blue'].to_numpy()

    close1_red = df_single_event.loc[choice_index_close1]['close1_red'].to_numpy()
    close1_blue = df_single_event.loc[choice_index_close1]['close1_blue'].to_numpy()

    close2_red = df_single_event.loc[choice_index_close2]['close2_red'].to_numpy()
    close2_blue = df_single_event.loc[choice_index_close2]['close2_blue'].to_numpy()

    open_win = (choice_fighters_open == winners_bool_open).all()
    close1_win = (choice_fighters_close1 == winner_bool_close1).all()
    close2_win = (choice_fighters_close2 == winner_bool_close2).all()

    open_odds = df_parlay['parlay_odds_open'].to_numpy()
    close1_odds = df_parlay['parlay_odds_close1_stack'].to_numpy()
    close2_odds = df_parlay['parlay_odds_close2_stack'].to_numpy()
    
    open_stake = df_parlay['parlay_fstar_open'].to_numpy()
    close1_stake = df_parlay['parlay_fstar_close1_stack'].to_numpy()
    close2_stake = df_parlay['parlay_fstar_close2_stack'].to_numpy()

    profit_open = np.where(open_win, open_odds, -1)
    profit_close1 = np.where(close1_win, close1_odds, -1)
    profit_close2 = np.where(close2_win, close2_odds, -1)

    open_net_fstar = np.where(open_win, open_stake, -open_stake)
    close1_net_fstar = np.where(close1_win, close1_stake, -close1_stake)
    close2_net_fstar = np.where(close2_win, close2_stake, -close2_stake)

    winner_name_open = winner_name.loc[choice_index_open].to_numpy()
    winner_name_close1 = winner_name.loc[choice_index_close1].to_numpy()
    winner_name_close2 = winner_name.loc[choice_index_close2].to_numpy()


    parlay_results = pd.DataFrame({
        'open_net_fstar':open_net_fstar, 'close1_net_fstar':close1_net_fstar, 'close2_net_fstar':close2_net_fstar,

        'open_red':open_red, 'open_blue':open_blue, 
        'close1_red':close1_red,'close1_blue':close1_blue, 
        'close2_red':close2_red, 'close2_blue':close2_blue, 

        'open_net_odds':profit_open, 'close1_net_odds':profit_close1, 'close2_net_odds':profit_close2,
        'open_fstar':open_stake,'close1_fstar':close1_stake, 'close2_fstar':close2_stake, 

        'choice_fighter_bool_open':choice_fighters_open, 
        'choice_fighter_bool_close1':choice_fighters_close1, 
        'choice_fighter_bool_close2':choice_fighters_close2, 

        'winner_bool_open':winners_bool_open,
        'winner_bool_close1':winner_bool_close1,
        'winner_bool_close2':winner_bool_close2,

        'choice_fighter_name_open':df_parlay['choice_fighter_name_open'], 
        'choice_fighter_name_close1':df_parlay['choice_fighter_name_close1_stack'], 
        'choice_fighter_name_close2':df_parlay['choice_fighter_name_close2_stack'],

        'fight_index_open':df_parlay['fight_index_open'],
        'fight_index_close1_stack':df_parlay['fight_index_close1_stack'],
        'fight_index_close2_stack':df_parlay['fight_index_close2_stack'],

        'winner_name':winner_name_open,
        'winner_name_close1': winner_name_close1,
        'winner_name_close2': winner_name_close2, 
    })

    return parlay_results


def calc_winner_ml(df_single_event, df_money_line):

    assert df_single_event.shape[0] == df_money_line.shape[0], 'scraped results df shape mismatch with bets df'

    winner_bool = df_single_event['winner'].to_numpy(dtype=float)

    pred_open = df_money_line['pred_winner_open'].to_numpy(dtype=float)
    pred_close1 = df_money_line['pred_winner_close1_stack'].to_numpy(dtype=float)
    pred_close2 = df_money_line['pred_winner_close2_stack'].to_numpy(dtype=float)

    def calc_color(x):
        result = np.empty(len(x), dtype=object)
        result[x == 1] = "red"
        result[x == 0] = "blue"
        result[np.isnan(x)] = np.nan
        return result

    def odds_from_color(df, color_list, odds_type):
        odds = np.empty(len(color_list), dtype=float)
        for i, color in enumerate(color_list):
            if color == "red":
                odds[i] = df[f"{odds_type}_red"].iloc[i]
            elif color == "blue":
                odds[i] = df[f"{odds_type}_blue"].iloc[i]
            else:
                odds[i] = np.nan  # Handle NaN or unexpected values
        return odds
    
    pred_color_open = calc_color(pred_open)
    pred_color_close1 = calc_color(pred_close1)
    pred_color_close2 = calc_color(pred_close2)
    
    open_odds =  odds_from_color(df_money_line, pred_color_open, "open")
    close1_odds = odds_from_color(df_money_line, pred_color_close1, "close1")
    close2_odds = odds_from_color(df_money_line, pred_color_close2, "close2")

    open_stake = df_money_line['fstar_open'].to_numpy(dtype=float)
    close1_stake = df_money_line['fstar_close1_stack'].to_numpy(dtype=float)
    close2_stake = df_money_line['fstar_close2_stack'].to_numpy(dtype=float)

    winner_name = df_single_event['winner_name'].to_numpy()
    name_open = df_money_line['pred_name_open'].to_list()
    name_close1 = df_money_line['pred_name_close1_stack'].to_list()
    name_close2 = df_money_line['pred_name_close2_stack'].to_list()
    
    df_data = pd.DataFrame({
        'fstar_open':open_stake, 
        'fstar_close1':close1_stake, 
        'fstar_close2':close2_stake,
        'winner_bool': winner_bool,
        'winner_name':winner_name,
        'pred_name_open':name_open, 
        'pred_name_close1':name_close1,
        'pred_name_close2':name_close2,
        'pred_winner_open':pred_open,
        'pred_winner_close1':pred_close1, 
        'pred_winner_close2':pred_close2, 
        'pred_color_open':pred_color_open, 
        'pred_color_close1':pred_color_close1,
        'pred_color_close2':pred_color_close2
    })

    df_data = pd.concat([
        df_data,
        df_money_line[[
            "fighter_red", "fighter_blue",
            "open_red", "open_blue",
            "close1_red", "close1_blue",
            "close2_red", "close2_blue",
        ]].reset_index(drop=True),
    ], axis=1)

    return df_data