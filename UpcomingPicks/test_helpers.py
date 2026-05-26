
import numpy as np 
import pandas as pd 
from config import config

def df_bets_tests(df_bets, df_bets_combined, valid_mask, choice_ev, fstar_list):

    non_nan_mask = df_bets.notna().all(axis=1)
    n_non_nan = non_nan_mask.sum()
    n_valid = valid_mask.sum()
    assert n_non_nan == n_valid, (
        f"Mismatch: non-NaN rows ({n_non_nan}) != valid rows ({n_valid})"
    )

    assert np.array_equal(~np.isnan(fstar_list), ~np.isnan(choice_ev)), f'error with fstar and choice_ev, {print(choice_ev)}, {print(fstar_list)}'

    assert df_bets.shape[0] == df_bets_combined.shape[0], 'mismatch shapes df bets and df bets combined '

def df_parlay_tests(df_parlay, choice_ev):
    
    if np.count_nonzero(~np.isnan(choice_ev[choice_ev > 0])) >= 2:
        assert not df_parlay.isna().any().any(), 'Parlay Bets Error'

    assert df_parlay.shape[0] == config.parlay_top_ev, 'Parlay df size error'