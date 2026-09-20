import numpy as np 
import pandas as pd
from itertools import combinations
from collections import defaultdict

from ufc_betting.BettingStrategy.kelly_scaling import (
    scale_mdd, 
    kelly_edge, 
    kelly_edge, 
    expected_value, 
    log_return_volatility, 
    scale_kelly_for_mdd
)

def run_per_bet_scaling(
    bets_df,
    max_drawdown,
    bankroll,
    N,
    adjust_mdd_by_edge=True,
):

    rows = []
    group_profit = 0

    for _, row in bets_df.iterrows():

        f_star = row["f_star_unscaled"]
        p = row["choice_proba"]
        fair_odds = row["choice_fair_odds"]
        real_odds = row["choice_real_odds"]
        ev = row["choice_ev"]
        bet_idx = row["choice_idx"]
        winner = row["winner"]

        if f_star <= 0 or ev <= 0 :
            f_final = 0
            profit = 0
            net_odds = 0
            win = False
            mu = 0
            sigma = 0

        else:
            edge = p - (1/(fair_odds))
            adjusted_mdd = (
                scale_mdd(edge, max_drawdown)
                if adjust_mdd_by_edge
                else max_drawdown
            )

            f_final, sigma, mu = scale_kelly_for_mdd(
                p,
                fair_odds,
                f_star,
                N=N,
                max_drawdown=adjusted_mdd,
                return_sigma=True,
            )
            stake = bankroll * f_final
            if winner == 2:
                # Refund draws/no-contests without changing pre-fight sizing.
                win = pd.NA
                profit = 0.0
                net_odds = 0.0
            else:
                win = int(winner) == int(bet_idx)
                profit = stake * (real_odds - 1) if win else -stake
                net_odds = (real_odds - 1) if win else -1

        group_profit += profit
        rows.append({ 
            "p": p, "fair_odds": fair_odds, "real_odds": real_odds, 
            "ev": ev, 'sharpe': mu / sigma if sigma > 0 else 0,
            "f_star_unscaled": f_star,"f_star_scaled": f_final,
            "stake": bankroll * f_final,"profit": profit, 
            "net_odds": net_odds, "win": win, "sigma": sigma
        })

    bets_out_df = pd.DataFrame(rows)
    return bets_out_df, group_profit

def parlay_top_ev(
    data,
    bankroll,
    top_n=[0, 1],
    parlay_mdd=.25,
    N=250,
    sort_column='choice_ev',
):
    
    empty_parlay = data.empty or data.shape[0] < len(top_n)
    
    parlay_ev = 0
    parlay_prob = 0
    parlay_kelly = 0

    if not empty_parlay:
        df_top_n = data.sort_values(
            by=sort_column, ascending=False
        ).iloc[top_n].copy()

        parlay_prob = np.prod(df_top_n['choice_proba'])
        parlay_odds = np.prod(df_top_n['choice_real_odds'])
        parlay_ev = parlay_prob * parlay_odds - 1

        # function for kelly full 
        b = parlay_odds - 1
        if b <= 0:
            kelly_full = 0.0
        else:
            kelly_full = max((b * parlay_prob - (1 - parlay_prob)) / b, 0)

        # scale kelly mdd 
        if parlay_mdd is not None:
            parlay_kelly = scale_kelly_for_mdd(parlay_prob, parlay_odds, kelly_full, N=N, max_drawdown=parlay_mdd)
        else:
            parlay_kelly = kelly_full

    if not empty_parlay and parlay_ev > 0: 

        # profit and win 
        stake = bankroll * parlay_kelly
        parlay_win = (df_top_n['winner'] == df_top_n['pred_winner']).all()
        net_odds = parlay_odds - 1

        if df_top_n['winner'].eq(2).any():
            # Match tracking: a draw/no-contest voids the whole ticket.
            # Keep the selected legs and stake; do not reselect using outcomes.
            profit = 0.0
            net_odds = 0.0
        elif parlay_win:
            profit = stake * net_odds
        else:
            profit = -stake
            net_odds = -1

    else:
        # fully zeroed, but schema preserved
        stake = 0.0
        profit = 0.0
        net_odds = 0.0
        df_top_n = data.copy()

    df_top_n['parlay_prob'] = parlay_prob
    df_top_n['parlay_ev'] = parlay_ev
    
    df_top_n['choice_fighter_name'] = np.where(df_top_n['pred_winner']==1, df_top_n['fighter_red'], df_top_n['fighter_blue'])
    df_top_n['fstar_parlay'] = parlay_kelly
    df_top_n['parlay_net_odds'] = net_odds

    df_top_n = df_top_n[[
        'choice_fighter_name', 'fstar_parlay','choice_ev',
        'choice_real_odds', 'parlay_net_odds', 
        'pred_winner', 'choice_proba',
        'parlay_prob', 'parlay_ev',
        'date', 'winner',
    ]]

    return profit, net_odds, df_top_n

def decimal_to_american(decimal_odds):
    """Convert positive decimal odds to American odds."""
    decimal_odds = float(decimal_odds)
    if not np.isfinite(decimal_odds) or decimal_odds <= 1:
        raise ValueError("decimal_odds must be finite and greater than 1")
    if decimal_odds >= 2:
        return 100 * (decimal_odds - 1)
    return -100 / (decimal_odds - 1)

def parlay_heavy_favorites(
    data,
    bankroll,
    max_legs,
    favorite_cutoff=-300,
    parlay_mdd=.25,
    N=250,
    sort_column='choice_ev',
):
    """Parlay positive-EV picks below an independent American-odds cutoff."""
    if not isinstance(max_legs, (int, np.integer)) or not 2 <= max_legs <= 4:
        raise ValueError("max_legs must be an integer between 2 and 4")
    if not np.isfinite(favorite_cutoff) or favorite_cutoff > -100:
        raise ValueError("favorite_cutoff must be finite and at most -100")

    candidates = data.loc[
        (data['choice_ev'] > 0)
        & (data['choice_american_odds'] <= favorite_cutoff)
    ].sort_values(sort_column, ascending=False)

    selected = candidates.head(max_legs).copy()
    if len(selected) < 2:
        selected = candidates.iloc[0:0].copy()

    return parlay_top_ev(
        selected,
        bankroll,
        top_n=list(range(len(selected))),
        parlay_mdd=parlay_mdd,
        N=N,
        sort_column=sort_column,
    )

def event_bankroll_scaled(
    bankroll,
    raw_kelly,
    edge,
    max_event_exposure=0.50,
    edge_scale=10.0
):
    raw_kelly = np.asarray(raw_kelly)
    edge = np.asarray(edge)

    # only positive Kelly bets
    k = np.maximum(raw_kelly, 0)

    # only reward positive edge
    positive_edge = np.maximum(edge, 0)

    # convert edge into confidence-like multiplier
    edge_multiplier = 1 - np.exp(-edge_scale * positive_edge)

    # scale Kelly by edge strength
    k = k * edge_multiplier

    if k.sum() == 0:
        return 0.0

    # total bankroll allocated to this event
    event_exposure = min(max_event_exposure, k.sum())
    return bankroll * event_exposure

def finalize_event_stats(
    group_stats,
    group,
    bets_out_df,
    edge,
    bankroll,
    ml_profit,
    parlay_net_money,
    heavy_parlay_net_money,
    parlay_fstar,
    heavy_parlay_fstar,
    uncertainty_multiplier,
    event_multiplier,
    prob_cols,
    real_decimal_cols,
    winner_col='winner',
    date_col='date',
):
    """Add final per-fight and per-event results and update bankroll."""
    shape = len(group)
    if len(bets_out_df) != shape:
        raise ValueError("bets_out_df and event group must have equal lengths")

    repeat = lambda value: np.full(shape, value)
    choice_ev = np.asarray(group_stats['choice_ev'], dtype=float)
    pred_winner = np.asarray(group_stats['pred_winner'], dtype=int)
    winners = group[winner_col].to_numpy(dtype=int)
    scaled_fstar = bets_out_df['f_star_scaled'].to_numpy(dtype=float)
    positive_ev = choice_ev > 0

    event_profit = ml_profit + parlay_net_money + heavy_parlay_net_money
    new_bankroll = bankroll + event_profit
    bankroll_return = event_profit / bankroll
    total_fstar = (
        scaled_fstar.sum()
        + (parlay_fstar + heavy_parlay_fstar) * event_multiplier
    )
    choice_odds = np.asarray(group_stats['choice_decimal_odds'], dtype=float)

    scalar_values = {
        'uncertainty_multiplier': uncertainty_multiplier,
        'event_multiplier': event_multiplier,
        'event_total_fstar': total_fstar,
        'bankroll_pct_change': bankroll_return,
        'bankroll_postevent': new_bankroll,
        'vegas_acc_choice': (
            (choice_odds[positive_ev] < 2).mean()
            if positive_ev.any() else 0.0
        ),
        'event_payout_money_line': ml_profit,
        'event_ml_net_odds': bets_out_df['net_odds'].sum(),

        'avg_choice_edge': (
            np.asarray(edge)[positive_ev].mean()
            if positive_ev.any() else np.nan
        ),

    }
    group_stats.update({
        name: repeat(value) for name, value in scalar_values.items()
    })

    group_stats.update({
        'fight_payout': bets_out_df['profit'].to_numpy(),
        'net_odds': bets_out_df['net_odds'].to_numpy(),
        'choice_fstar': scaled_fstar,
        'kelly_edge': group_stats['f_star_unscaled'],
        'choice_edge': edge,
        'date': group[date_col].to_list(),
        'fighter_red': group['fighter_red'].to_list(),
        'fighter_blue': group['fighter_blue'].to_list(),
        'winner': group[winner_col].to_list(),
        'red_proba': group[prob_cols[1]].to_list(),
        'blue_proba': group[prob_cols[0]].to_list(),
        'open_red': group['open_red'].to_list(),
        'open_blue': group['open_blue'].to_list(),
        'real_odds_blue': group[real_decimal_cols].iloc[:, 0].to_numpy(),
        'real_odds_red': group[real_decimal_cols].iloc[:, 1].to_numpy(),
    })

    kelly_valid = np.where(positive_ev, scaled_fstar, 0.0)
    group_stats['fstar_net'] = np.where(
        winners == 2, 0.0,
        np.where(pred_winner == winners, kelly_valid, -kelly_valid),
    )
    return group_stats, new_bankroll


def resolve_walk_forward_params(group, defaults, prefix='best_'):
    """Resolve the single walk-forward parameter set assigned to an event."""
    mapping = {
        'max_drawdown': 'mdd',
        'N': 'N',
        'parlay_mdd': 'mdd_parlay',
        'N_parlay': 'N_parlay',
        'z': 'z',
        'moneyline_min_american': 'moneyline_min_american',
        'moneyline_max_american': 'moneyline_max_american',
        'heavy_max_legs': 'heavy_parlay_max_legs',
        'heavy_favorite_cutoff': 'heavy_favorite_cutoff',
    }
    resolved = defaults.copy()
    for output_name, parameter_name in mapping.items():
        column = f'{prefix}{parameter_name}'
        if column not in group.columns:
            raise KeyError(f"Missing walk-forward parameter column '{column}'")
        values = group[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(
                f"Each event must have exactly one value for '{column}'"
            )
        resolved[output_name] = values[0]

    resolved['N'] = int(resolved['N'])
    heavy_mdd_column = f'{prefix}heavy_parlay_mdd'
    if heavy_mdd_column in group:
        values = group[heavy_mdd_column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"Each event must have exactly one value for '{heavy_mdd_column}'")
        resolved['heavy_parlay_mdd'] = values[0]
    else:
        resolved['heavy_parlay_mdd'] = resolved['parlay_mdd']
    resolved['N_parlay'] = int(resolved['N_parlay'])
    if resolved['heavy_max_legs'] not in (2, 3, 4):
        raise ValueError("heavy_parlay_max_legs must be an integer between 2 and 4")
    resolved['heavy_max_legs'] = int(resolved['heavy_max_legs'])
    resolved['moneyline_american_bounds'] = (
        float(resolved['moneyline_min_american']),
        float(resolved['moneyline_max_american']),
    )
    return resolved


def merge_walk_forward_params(
    betting_data,
    monthly_params,
    date_col='date',
    prefix='best_',
):
    """Attach monthly Optuna parameters to fight-level betting data."""
    if isinstance(monthly_params, (str, bytes)) or hasattr(
        monthly_params, '__fspath__'
    ):
        monthly_params = pd.read_csv(monthly_params)
    else:
        monthly_params = monthly_params.copy()

    if 'month' not in monthly_params.columns:
        raise KeyError("monthly_params must contain a 'month' column")
    if monthly_params['month'].duplicated().any():
        raise ValueError("monthly_params must contain one row per month")

    parameter_names = [
        'mdd',
        'N',
        'mdd_parlay',
        'N_parlay',
        'z',
        'moneyline_min_american',
        'moneyline_max_american',
        'heavy_parlay_max_legs',
        'heavy_favorite_cutoff',
    ]
    missing = [
        name for name in parameter_names if name not in monthly_params.columns
    ]
    if missing:
        raise KeyError(f"monthly_params is missing columns: {missing}")

    # Older schedules used the regular-parlay MDD for both strategies.
    if 'heavy_parlay_mdd' not in monthly_params:
        monthly_params['heavy_parlay_mdd'] = monthly_params['mdd_parlay']
    parameter_names.append('heavy_parlay_mdd')
    data = betting_data.copy()
    data['_walk_forward_month'] = pd.to_datetime(
        data[date_col], errors='raise'
    ).dt.to_period('M').astype(str)
    schedule = monthly_params[['month', *parameter_names]].copy()
    schedule['_walk_forward_month'] = pd.PeriodIndex(
        schedule.pop('month'), freq='M'
    ).astype(str)
    schedule = schedule.rename(
        columns={name: f'{prefix}{name}' for name in parameter_names}
    )

    existing_parameter_columns = [
        f'{prefix}{name}' for name in parameter_names
        if f'{prefix}{name}' in data.columns
    ]
    data = data.drop(columns=existing_parameter_columns)
    data = data.merge(
        schedule,
        on='_walk_forward_month',
        how='left',
        validate='many_to_one',
    )
    missing_months = data.loc[
        data[f'{prefix}mdd'].isna(), '_walk_forward_month'
    ].unique()
    if len(missing_months):
        raise ValueError(
            f"No monthly parameters found for months: {missing_months.tolist()}"
        )
    return data.drop(columns='_walk_forward_month')


def simulate_kelly(
        df_final, prob_cols, fair_decimal_cols, real_decimal_cols,
        pred_winner_col, winner_col='winner', date_col='date',
        init_bankroll=1000, bankroll_floor=None,
        max_drawdown=.30, parlay_mdd=.25, N=1000,N_parlay=2000,
        calc_parlay=False,
        test_other_ev = False,
        test_bankroll_scaler = False,
        z=None,
        prediction_se_col='se',
        moneyline_american_bounds=None,
        calc_heavy_favorite_parlay=False,
        heavy_favorite_cutoff=-300,
        heavy_parlay_max_legs=4,
        use_walk_forward_params=False,
        walk_forward_param_prefix='best_',
        monthly_params=None,
        risk_prob_cols=None,
        heavy_parlay_mdd=None,
        adjust_mdd_by_edge=True,
    ):

    """Simulate Kelly betting with optional regular and heavy-favorite parlays.

    ``heavy_parlay_max_legs`` caps each heavy-favorite parlay at 2, 3, or 4
    legs (default 4). Walk-forward parameters override this per event via
    ``best_heavy_parlay_max_legs`` or the configured parameter prefix.
    ``heavy_parlay_mdd`` controls heavy-parlay sizing independently; None
    inherits ``parlay_mdd``. Both strategies share ``N_parlay``.
    Set ``adjust_mdd_by_edge=False`` to size moneylines using the supplied
    drawdown limit without increasing it for larger edges. Walk-forward
    parameter runs continue to disable this adjustment. This flag does not
    disable the separate ``z`` uncertainty adjustment or exposure cap.
    """
    bankroll = init_bankroll

    df_results = pd.DataFrame()
    df_parlay = pd.DataFrame(columns=[
        'choice_fighter_name', 'fstar_parlay', 'choice_ev',
        'choice_real_odds', 'parlay_net_odds', 'pred_winner',
        'choice_proba', 'parlay_prob', 'parlay_ev', 'date', 'winner',
        'fstar_net', 'choice_decimal_odds', 'parlay_strategy',
    ])

    if monthly_params is not None:
        df_final = merge_walk_forward_params(
            betting_data=df_final,
            monthly_params=monthly_params,
            date_col=date_col,
            prefix=walk_forward_param_prefix,
        )
        use_walk_forward_params = True

    if (
        use_walk_forward_params
        and risk_prob_cols is None
        and {'risk_proba_blue', 'risk_proba_red'}.issubset(df_final.columns)
    ):
        risk_prob_cols = ['risk_proba_blue', 'risk_proba_red']

    df = df_final.sort_values(by=date_col)

    for date, group in df.groupby(date_col, sort=True):
        group = group.reset_index(drop=True)

        event_params = {
            'max_drawdown': max_drawdown,
            'N': N,
            'parlay_mdd': parlay_mdd,
            'heavy_parlay_mdd': (
                parlay_mdd if heavy_parlay_mdd is None else heavy_parlay_mdd
            ),
            'N_parlay': N_parlay,
            'z': z,
            'moneyline_american_bounds': moneyline_american_bounds,
            'heavy_favorite_cutoff': heavy_favorite_cutoff,
            'heavy_max_legs': heavy_parlay_max_legs,
        }
        if use_walk_forward_params:
            event_params = resolve_walk_forward_params(
                group, event_params, prefix=walk_forward_param_prefix
            )

        group_stats = defaultdict(list)

        for _, row in group.iterrows():

            bet_idx = int(row[pred_winner_col])
            p = row[prob_cols[bet_idx]]
            risk_p = (
                row[risk_prob_cols[bet_idx]]
                if risk_prob_cols is not None
                else p
            )
            fair_odds = row[fair_decimal_cols[bet_idx]]
            real_odds = row[real_decimal_cols[bet_idx]]
            ev = expected_value(p, real_odds)
            pred_winner = bet_idx

            # append this bet 
            if ev <= 0 and test_other_ev is True: 
                new_idx = 1 if bet_idx == 0 else 0 
                ev_new = expected_value(1-p, row[real_decimal_cols[new_idx]])
                if ev_new > 0: 
                    bet_idx = new_idx
                    real_odds = row[real_decimal_cols[bet_idx]]
                    fair_odds = row[fair_decimal_cols[bet_idx]]
                    p = 1-p
                    risk_p = (
                        row[risk_prob_cols[bet_idx]]
                        if risk_prob_cols is not None
                        else p
                    )
                    ev = ev_new
                    pred_winner = bet_idx

            # Sportsbook lines are discrete; rounding avoids excluding a -300
            # line because its decimal representation was stored as 1.3333.
            american_odds = round(decimal_to_american(real_odds))
            moneyline_allowed = True
            if event_params['moneyline_american_bounds'] is not None:
                lower, upper = event_params['moneyline_american_bounds']
                if lower > upper:
                    raise ValueError(
                        "moneyline_american_bounds must be (lower, upper)"
                    )
                moneyline_allowed = lower <= american_odds <= upper
  
            new_row = {
                'f_star_unscaled': (
                    kelly_edge(p, fair_odds) if moneyline_allowed else 0.0
                ),
                'choice_ev': ev,
                'choice_fair_odds': fair_odds,
                'choice_real_odds': real_odds, 
                'choice_idx': bet_idx,
                'choice_decimal_odds': real_odds,
                'choice_proba': p,
                'choice_risk_proba': risk_p,
                'pred_winner':pred_winner,
                'choice_american_odds': american_odds,
                'moneyline_allowed': moneyline_allowed,
            }

            for k,v in new_row.items(): 
                group_stats[k].append(v)
            
        bets_input_df = pd.DataFrame({
            **group_stats,
            "winner": group[winner_col].values
        })

        edge = np.asarray(group_stats['choice_proba']) - (1/(np.asarray(group_stats['choice_fair_odds'])))

        if test_bankroll_scaler:
            scaled_bankroll = event_bankroll_scaled(
                bankroll, 
                group_stats['f_star_unscaled'], 
                edge,
                max_event_exposure=1,
                edge_scale=5
        )
        else: 
            scaled_bankroll = bankroll 

        bets_out_df, ml_profit = run_per_bet_scaling(
            bets_input_df,
            max_drawdown=event_params['max_drawdown'],
            bankroll=scaled_bankroll,
            N=event_params['N'],
            adjust_mdd_by_edge=adjust_mdd_by_edge and not use_walk_forward_params,
        ) 
            
        parlay_net_money = 0
        parlay_fstar = 0.0
        parlay_net_odds = 0.0
        df_top_n = None
        heavy_parlay_net_money = 0.0
        heavy_parlay_fstar = 0.0
        heavy_parlay_net_odds = 0.0
        heavy_parlay_df = None

        if calc_parlay is True or calc_heavy_favorite_parlay is True:

            df_data = pd.DataFrame({
                'winner': group['winner'].values,
                'pred_winner': group_stats['pred_winner'],
                'choice_ev': group_stats['choice_ev'],
                'choice_real_odds': group_stats['choice_real_odds'],
                'choice_fstar': bets_out_df['f_star_scaled'].values,
                'fighter_red': group['fighter_red'].values,
                'fighter_blue': group['fighter_blue'].values,
                'date': group['date'].values,
                'choice_proba': group_stats['choice_proba'],
                'choice_sharpe': bets_out_df['sharpe'].values,
                'choice_american_odds': group_stats['choice_american_odds'],
                'moneyline_allowed': group_stats['moneyline_allowed'],
            })

        if calc_parlay is True:
            parlay_net_money, parlay_net_odds, df_top_n = parlay_top_ev(
                df_data,
                scaled_bankroll,
                top_n=[0, 1],
                parlay_mdd=event_params['parlay_mdd'],
                N=event_params['N_parlay'],
                sort_column='choice_ev',
            )
            if not df_top_n.empty:
                parlay_fstar = float(df_top_n['fstar_parlay'].iloc[0])
                df_top_n['parlay_strategy'] = 'top_ev'

        if calc_heavy_favorite_parlay is True:
            (
                heavy_parlay_net_money,
                heavy_parlay_net_odds,
                heavy_parlay_df,
            ) = parlay_heavy_favorites(
                df_data,
                scaled_bankroll,
                max_legs=event_params['heavy_max_legs'],
                favorite_cutoff=event_params['heavy_favorite_cutoff'],
                parlay_mdd=event_params['heavy_parlay_mdd'],
                N=event_params['N_parlay'],
                sort_column='choice_ev',
            )
            if not heavy_parlay_df.empty:
                heavy_parlay_fstar = float(
                    heavy_parlay_df['fstar_parlay'].iloc[0]
                )

        uncertainty_multiplier = 1.0
        event_z = event_params['z']
        if event_z is not None:
            if not np.isfinite(event_z) or event_z < 0:
                raise ValueError("z must be a finite non-negative number or None")
            if prediction_se_col not in group.columns:
                raise KeyError(
                    f"z was provided, but prediction standard-error column "
                    f"'{prediction_se_col}' is missing"
                )

            model_probabilities = np.asarray(
                group_stats['choice_risk_proba'], dtype=float
            )
            vegas_probabilities = 1 / np.asarray(
                group_stats['choice_fair_odds'], dtype=float
            )
            prediction_se = group[prediction_se_col].to_numpy(dtype=float)
            if not np.isfinite(prediction_se).all():
                raise ValueError(
                    f"Column '{prediction_se_col}' contains NaN or infinite values"
                )
            uncertainty_weights = bets_out_df[
                'f_star_scaled'
            ].to_numpy(dtype=float, copy=True)
            for parlay_df, parlay_weight in (
                (df_top_n, parlay_fstar),
                (heavy_parlay_df, heavy_parlay_fstar),
            ):
                if (
                    parlay_df is None
                    or parlay_df.empty
                    or parlay_weight <= 0
                ):
                    continue
                leg_weight = parlay_weight / len(parlay_df)
                uncertainty_weights += (
                    np.isin(np.arange(len(group)), parlay_df.index)
                    * leg_weight
                )

            raw_edges = np.maximum(
                model_probabilities - vegas_probabilities,
                0,
            )
            conservative_edges = np.maximum(
                model_probabilities
                - event_z * prediction_se
                - vegas_probabilities,
                0,
            )
            raw_score = np.sum(uncertainty_weights * raw_edges)
            conservative_score = np.sum(
                uncertainty_weights * conservative_edges
            )
            uncertainty_multiplier = (
                np.clip(conservative_score / raw_score, 0.0, 1.0)
                if raw_score > 0
                else 0.0
            )

        total_fstar = (
            bets_out_df['f_star_scaled'].sum()
            + parlay_fstar
            + heavy_parlay_fstar
        ) * uncertainty_multiplier
        exposure_multiplier = (
            min(1.0, 1.0 / total_fstar)
            if total_fstar > 0
            else 1.0
        )
        event_multiplier = uncertainty_multiplier * exposure_multiplier

        bets_out_df['f_star_scaled'] *= event_multiplier
        bets_out_df['stake'] *= event_multiplier
        bets_out_df['profit'] *= event_multiplier
        ml_profit = bets_out_df['profit'].sum()


        if calc_parlay is True:
            parlay_net_money *= event_multiplier
            df_top_n['fstar_parlay'] *= event_multiplier
            df_top_n['fstar_net'] = np.sign(parlay_net_odds) * df_top_n['fstar_parlay']
            df_top_n['choice_decimal_odds'] = df_top_n['choice_real_odds'] 

            group_stats['parlay_net'] = np.full(group.shape[0], parlay_net_money)
            group_stats['parlay_net_odds'] = np.full(group.shape[0], parlay_net_odds)
            group_stats['parlay_ev'] = np.full(
                group.shape[0],
                df_top_n['parlay_ev'].iloc[0] if not df_top_n.empty else 0.0,
            )

        if calc_heavy_favorite_parlay is True:
            heavy_parlay_net_money *= event_multiplier
            if not heavy_parlay_df.empty:
                heavy_parlay_df['fstar_parlay'] *= event_multiplier
                heavy_parlay_df['fstar_net'] = (
                    np.sign(heavy_parlay_net_odds)
                    * heavy_parlay_df['fstar_parlay']
                )
                heavy_parlay_df['choice_decimal_odds'] = (
                    heavy_parlay_df['choice_real_odds']
                )
                heavy_parlay_df['parlay_strategy'] = 'heavy_favorite_target'

        # Keep the event schema stable for every combination of toggles.
        group_stats['parlay_net'] = np.full(
            len(group), parlay_net_money + heavy_parlay_net_money
        )
        group_stats['parlay_net_odds'] = np.full(
            len(group), parlay_net_odds + heavy_parlay_net_odds
        )
        group_stats['parlay_ev'] = np.full(
            len(group), sum(
                frame['parlay_ev'].iloc[0]
                for frame in (df_top_n, heavy_parlay_df)
                if frame is not None and not frame.empty
            )
        )
        group_stats['regular_parlay_net'] = np.full(
            len(group), parlay_net_money
        )
        group_stats['heavy_parlay_net'] = np.full(
            len(group), heavy_parlay_net_money
        )


        group_stats, bankroll = finalize_event_stats(
            group_stats=group_stats,
            group=group,
            bets_out_df=bets_out_df,
            edge=edge,
            bankroll=bankroll,
            ml_profit=ml_profit,
            parlay_net_money=parlay_net_money,
            heavy_parlay_net_money=heavy_parlay_net_money,
            parlay_fstar=parlay_fstar,
            heavy_parlay_fstar=heavy_parlay_fstar,
            uncertainty_multiplier=uncertainty_multiplier,
            event_multiplier=event_multiplier,
            prob_cols=prob_cols,
            real_decimal_cols=real_decimal_cols,
            winner_col=winner_col,
            date_col=date_col,
        )

        # print([(key, len(value)) for key,value in group_stats.items()])

        if bankroll_floor is not None and bankroll < bankroll_floor:
            print(f"Bankroll low on {date}, replenishing funds")
            bankroll = 500

        group_df = pd.DataFrame(group_stats)

        if calc_parlay is True: 
            df_parlay = pd.concat([df_parlay, df_top_n], axis=0)
        if calc_heavy_favorite_parlay is True:
            df_parlay = pd.concat(
                [df_parlay, heavy_parlay_df], axis=0, ignore_index=True
            )
        df_results = pd.concat([df_results, group_df], ignore_index=True)

    return df_results, df_parlay

