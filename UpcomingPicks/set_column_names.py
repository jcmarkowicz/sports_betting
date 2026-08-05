import pandas as pd 

def get_ml_bet_cols(type_):
    """
    Returns a dict mapping standard packet keys -> column names with type suffix.
    """
    return {
        "pred_name_col":    f"pred_name_{type_}",
        "pred_winner_col":  f"pred_bool_{type_}",
        "choice_proba_col": f"choice_proba_{type_}",
        "choice_fstar_col": f"fstar_{type_}",
        "choice_stake_col": f"stake_{type_}",
        "edge_col":         f"edge_{type_}",
        "ev_col":           f"ev_{type_}",
    }

def get_parlay_cols(type_):
    
    return {
        'choice_fighter_name_col' : f"choice_fighter_name_{type_}",
        'choice_fighter_bool_col' : f"choice_fighter_bool_{type_}",
        'parlay_fstar_col'        : f"parlay_fstar_{type_}",
        'parlay_odds_col'         : f"parlay_odds_{type_}",
        'stake_col'               : f"stake_{type_}",
        'parlay_ev_col'           : f"parlay_ev_{type_}",
        'parlay_prob_col'         : f"parlay_prob_{type_}"
    }


def set_ml_bets_cols(type_, pkt, required_idx, all_na=False):

    cols = get_ml_bet_cols(type_)

    if all_na:
        df = pd.DataFrame({v: pd.NA for v in cols.values()}, index=required_idx)
    else: 
        assert set(pkt.keys()) == set(cols), 'money line pkt error'
        df = pd.DataFrame({cols[key]:value for key, value in pkt.items()}, index=required_idx)

    return df
        

def set_parlay_cols(type_, pkt, required_idx, all_na=False):
    cols = get_parlay_cols(type_)

    if all_na:
        df = pd.DataFrame({v: pd.NA for v in cols.values()}, index=required_idx)
        assert df.shape[0] == len(required_idx), 'wrong index parlay df'
    else: 
        assert set(pkt.keys()) == set(cols), 'parlay pkt error'
        df = pd.DataFrame({cols[key]:value for key, value in pkt.items()}, index=required_idx)

    return df