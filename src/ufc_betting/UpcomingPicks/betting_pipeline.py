import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
from ufc_betting.UpcomingPicks.set_column_names import (
    set_ml_bets_cols, set_parlay_cols, get_ml_bet_cols, get_parlay_cols,
)
from ufc_betting.UpcomingPicks.model_helpers import logit_predict
from ufc_betting.UpcomingPicks.data_helpers import get_bets_input, get_bets_pkt
from ufc_betting.config import settings
from ufc_betting.DataPipeline.dataframes.moneylines import MoneylineDataFrame
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame


def build_event_ev_table(
    moneyline_bets: pd.DataFrame,
    parlay_bets: pd.DataFrame,
) -> pd.DataFrame:
    """Build frontend-ready moneyline, parlay, and combined event EV rows."""
    moneylines = MoneylineDataFrame.from_generated(moneyline_bets)
    event_dates = moneylines.frame["date"].drop_duplicates()
    if len(event_dates) != 1:
        raise ValueError("Event EV table requires exactly one event date")

    parlays = ParlayDataFrame.from_generated(
        parlay_bets,
        event_date=event_dates.iloc[0],
    )
    moneyline_ev = moneylines.event_ev().set_index("date")
    parlay_ev = parlays.event_ev().set_index("date")
    combined_ev = moneyline_ev.add(parlay_ev, fill_value=0.0)

    rows = []
    for label, event_ev in (
        ("Moneyline", moneyline_ev),
        ("Parlay", parlay_ev),
        ("Combined", combined_ev),
    ):
        opening = float(event_ev["event_ev_open"].iloc[0])
        row = {
            "date": event_ev.index[0],
            "bet_type": label,
        }
        for stage in ("open", "close1", "close2"):
            value = float(event_ev[f"event_ev_{stage}"].iloc[0])
            row[f"{stage}_ev_pct"] = 100 * value
            row[f"{stage}_pct_opening_ev"] = (
                100 * value / opening if opening > 1e-10 else np.nan
            )
        rows.append(row)

    result = pd.DataFrame(rows)[[
        "date",
        "bet_type",
        "open_ev_pct",
        "close1_ev_pct",
        "close2_ev_pct",
        "close1_pct_opening_ev",
        "close2_pct_opening_ev",
    ]]
    value_columns = [
        column for column in result.columns
        if column.endswith("_ev_pct") or column.endswith("_pct_opening_ev")
    ]
    result[value_columns] = result[value_columns].mask(
        result[value_columns].abs() < 1e-12,
        0.0,
    )
    return result


def allocate_upcoming_bets(data, params, stage, *, use_decay=True):
    """Reuse simulation allocations without requiring fight outcomes.

    Void placeholders affect only settlement fields in the internal cache,
    never selection or stakes. Live output uses only the allocation fields.
    """
    params = BettingParams.from_value(params)
    data = data.copy()
    data['winner'] = 2
    cache = Calculator.calculate_scenario_cache(data, params, odds_type=stage)[None]
    winners = data.winner.to_numpy()
    if stage != 'open' and use_decay and params.decay_rates is not None:
        if params.opening_params is None:
            raise ValueError(f'{stage} decay requires opening_params')
        opening = Calculator.calculate_scenario_cache(
            data, params.opening_params, odds_type=stage)[None]
        return Calculator.decay_scenario_allocations(
            opening, cache, data[['dec_open_red', 'dec_open_blue']].to_numpy(),
            winners, params.opening_params, params, params.decay_rates, True,
        )
    return Calculator.scenario_allocations(cache, winners, params, True)


def betting_pipeline(
    upcoming_df: pd.DataFrame,
    feats_list: list[list[str]],
    model_list: list,
    scaler_list: list,
    encoder_list: list,
    type_list: list[str],
    fair_odds_list: list[list[str]],
    real_odds_list: list[list[str]],
    bankroll: float,
    mdd_ml_arr=None,
    mdd_parlay_arr=None,
    N_ml_arr=None,
    N_parlay_arr=None,
    return_event_ev: bool = False,
    *,
    params_dir=None,
    betting_params=None,
    model_se_list=None,
    use_decay=True,
):
    """Generate one card with saved Kelly policies and existing output columns.

    Defaults to Data/kelly_params/{open,close1,close2}.json. betting_params
    optionally maps stages to BettingParams, dictionaries, or JSON paths.
    Closing stages blend their embedded opening policy at current predictions
    and prices, then apply closing uncertainty and a joint exposure cap.
    Legacy mdd/N arrays are deprecated and ignored; saved policies take priority.
    model_se_list supplies companion SE models; otherwise use model covariance.
    """
    if any(value is not None for value in (mdd_ml_arr, mdd_parlay_arr, N_ml_arr, N_parlay_arr)):
        warnings.warn('mdd/N arrays are ignored; use betting_params or params_dir',
                      DeprecationWarning, stacklevel=2)
    if not np.isfinite(bankroll) or bankroll < 0:
        raise ValueError('bankroll must be finite and nonnegative')
    count = len(type_list)
    if not count or len(set(type_list)) != count or not set(type_list) <= {'open', 'close1', 'close2'}:
        raise ValueError('type_list must contain unique open/close1/close2 stages')
    if return_event_ev and set(type_list) != {'open', 'close1', 'close2'}:
        raise ValueError('Event EV comparison requires all three price stages')
    lists = (feats_list, model_list, scaler_list, encoder_list, fair_odds_list, real_odds_list)
    if any(len(values) != count for values in lists) or (model_se_list is not None and len(model_se_list) != count):
        raise ValueError('Model, feature, odds, and SE lists must align with type_list')
    df = upcoming_df.copy().reset_index(drop=True)
    if df.empty or df.date.isna().any() or pd.to_datetime(df.date).dt.normalize().nunique() != 1:
        raise ValueError('betting_pipeline requires exactly one nonempty event date')
    params_dir = Path(params_dir) if params_dir is not None else settings.data_dir / 'kelly_params'
    policies = {
        stage: (BettingParams.load(params_dir / f'{stage}.json') if betting_params is None
                else BettingParams.from_value(betting_params[stage]))
        for stage in type_list
    }
    other_cols = ['date', 'fighter_red', 'fighter_blue', 'open_red', 'open_blue',
                  'close1_red', 'close2_red', 'close1_blue', 'close2_blue']
    moneylines = df[other_cols].copy()
    parlays = pd.DataFrame(index=np.arange(2))
    for i, stage in enumerate(type_list):
        params = policies[stage]
        feats = feats_list[i]
        missing = set(feats) - set(df.columns)
        if missing:
            raise ValueError(
                f'Missing {stage} model features: {sorted(missing)}. '
                'Rebuild upcoming event features with the current feature pipeline.'
            )
        real_odds, fair_odds = real_odds_list[i], fair_odds_list[i]
        for columns in (real_odds, fair_odds):
            if len(columns) != 2 or not columns[0].endswith('_blue') or not columns[1].endswith('_red'):
                raise ValueError('Odds columns must be ordered blue, then red')
        price_cols = list(real_odds) + list(fair_odds)
        if stage != 'open' and use_decay and params.decay_rates is not None:
            price_cols += ['dec_open_red', 'dec_open_blue']
        prices = df[price_cols].apply(pd.to_numeric, errors='coerce')
        valid = df[feats].notna().all(axis=1) & np.isfinite(prices).all(axis=1) & prices.gt(1).all(axis=1)
        ml = set_ml_bets_cols(stage, {}, df.index, all_na=True)
        ticket = set_parlay_cols(stage, {}, np.arange(2), all_na=True)
        ticket[f'fight_index_{stage}'] = np.nan
        if valid.any():
            num_feats = df[feats].select_dtypes(exclude='category').columns
            cat_feats = df[feats].select_dtypes(include='category').columns
            sides, red, blue, se = logit_predict(
                model=model_list[i], df=df, y_hat=pd.Series(np.nan, index=df.index),
                feats=feats, num_feats=num_feats, cat_feats=cat_feats, valid_mask=valid,
                scaler=scaler_list[i], cat_encoder=encoder_list[i], required_df_idx=df.index,
                rename_odds=stage if stage != 'open' else None,
                model_se=None if model_se_list is None else model_se_list[i], return_se=True,
            )
            data = df.loc[valid].copy()
            data['pred_winner'] = sides.loc[valid]
            data['proba_red'], data['proba_blue'] = red.loc[valid], blue.loc[valid]
            data['proba_se'] = se.loc[valid]
            for j, side in enumerate(('blue', 'red')):
                data[f'dec_{stage}_{side}'] = df.loc[valid, real_odds[j]]
                data[f'dec_fair_{stage}_{side}'] = df.loc[valid, fair_odds[j]]
            alloc = allocate_upcoming_bets(data, params, stage, use_decay=use_decay)
            inputs = get_bets_input(df, sides, red, blue, real_odds, fair_odds)
            fractions = pd.Series(np.nan, index=df.index)
            fractions.loc[valid] = alloc['moneyline_stakes'][0]
            ml = get_bets_pkt(inputs, pd.DataFrame({'fstar_scaled': fractions,
                'stake': fractions * bankroll}), stage, df.index, False)
            if len(data) >= 2:
                ids = data.index[alloc['legs'][0]]
                fraction = float(alloc['parlay_stakes'][0] * alloc['multiplier'][0])
                packet = {
                    'choice_fighter_name_col': inputs.loc[ids, 'pred_winner_names'],
                    'choice_fighter_bool_col': inputs.loc[ids, 'pred_winner_bool'],
                    'parlay_fstar_col': fraction,
                    'parlay_odds_col': inputs.loc[ids, 'choice_real_odds'].prod(),
                    'stake_col': fraction * bankroll,
                    'parlay_ev_col': float(alloc['parlay_ev'][0]),
                    'parlay_prob_col': inputs.loc[ids, 'choice_proba'].prod(),
                }
                ticket = set_parlay_cols(stage, packet, ids)
                ticket[f'fight_index_{stage}'] = ids
                ticket = ticket.reset_index(drop=True)
        moneylines = moneylines.join(ml)
        parlays = parlays.join(ticket)
    if return_event_ev:
        return moneylines, parlays, build_event_ev_table(moneylines, parlays)
    return moneylines, parlays


def seperate_bets_dfs(df_bets, df_parlay, types):
    dfs = []
    dfs_parlay = []

    for type in types: 

        columns = get_ml_bet_cols(type).values()
        dfs.append(df_bets[columns])

        c_parlay = get_parlay_cols(type).values()
        dfs_parlay.append(df_parlay[c_parlay])

    return dfs, dfs_parlay
