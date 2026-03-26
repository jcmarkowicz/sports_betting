import numpy as np 
import pandas as pd


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
    emdd = sigma * (
        np.sqrt(
            2*np.log(N) -
            (np.log(np.log(N)) + np.log(4*np.pi)) /
            (2*np.sqrt(2*np.log(N)))
        )
    )
    return emdd

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


def scale_kelly_portfolio(bets, N, max_drawdown):
    """
    Scale multiple simultaneous Kelly bets to meet portfolio MDD target.
    
    bets : list of dicts with keys ['p', 'odds', 'f_full']
    N : number of rounds
    max_drawdown : tolerable drawdown fraction
    """
    mus = []
    sigmas = []
    sharpe_ratio_bet = []
    f_full_list = []

    for bet in bets:
        f = bet.get('f_full', 0)
        b = bet.get('odds', 0) - 1
        p = bet.get('p', 0)

        # ✅ Skip invalid or non-betting entries
        if f <= 0 or p <= 0 or b <= 0:
            mus.append(0)
            sigmas.append(0)
            sharpe_ratio_bet.append(0)
            f_full_list.append(0)
            continue

        sigma, _ = log_return_volatility(f, b, p)
        mu_ev = expected_value(p, b + 1) * f  # scaled EV by fraction
        mus.append(mu_ev)
        sigmas.append(sigma)
        sharpe_ratio_bet.append(mu_ev / sigma if sigma > 0 else 0)
        f_full_list.append(f)

    mus = np.array(mus)
    sigmas = np.array(sigmas)
    f_full_list = np.array(f_full_list)

    # ✅ If all f_full = 0 → no bets made → return zeros safely
    if np.all(f_full_list == 0):
        return (
            np.zeros_like(f_full_list),
            0,  # sigma_portfolio
            0,  # sigma_portfolio_scaled
            0,  # mu_portfolio
            0,  # sharpe_ratio_portfolio
            np.zeros_like(f_full_list),  # sharpe_ratio_bet
            np.zeros_like(f_full_list),  # sigmas
            0   # k
        )

    # ✅ Compute portfolio stats only for active bets
    sigma_portfolio = np.sqrt(np.sum((f_full_list * sigmas) ** 2))
    mu_portfolio = np.sum(f_full_list * mus)
    sharpe_ratio_portfolio = mu_portfolio / sigma_portfolio if sigma_portfolio > 0 else 0

    # ✅ Compute scaling factor
    k = max_drawdown / (sigma_portfolio * np.sqrt(2 * np.log(N))) if sigma_portfolio > 0 else 0
    k = min(k, 1.0)  # cannot exceed full Kelly

    f_scaled = k * f_full_list
    sigma_portfolio_scaled = k * sigma_portfolio

    return (
        f_scaled,
        sigma_portfolio,
        sigma_portfolio_scaled,
        mu_portfolio,
        sharpe_ratio_portfolio,
        sharpe_ratio_bet,
        sigmas,
        k
    )


def run_portfolio_scaling(choice_ev, choice_proba, unweighted_fstar, choice_fair_odds, max_drawdown, bankroll, choice_real_odds, choice_idx, winner_col, group, group_stats, group_profit):
    bets = []
    for i in range(len(choice_proba)):
        if choice_ev[i] > 0 and unweighted_fstar[i] > 0:
            bets.append({
                'p': choice_proba[i],
                'odds': choice_fair_odds[i],
                'f_full': unweighted_fstar[i]
            })
        else:
            bets.append({
                'p': choice_proba[i],
                'odds': choice_fair_odds[i],
                'f_full': 0
            })

    f_scaled, sigma_portfolio, sigma_portfolio_scaled, mu_portfolio, sharpe_portfolio, sharpe_per_bet, sigma_per_bet, k = \
        scale_kelly_portfolio(bets, N=1000, max_drawdown=max_drawdown)
    
    for i, (kelly_frac, p, real_odds, ev, bet_idx) in enumerate(zip(f_scaled, choice_proba, choice_real_odds, choice_ev, choice_idx)):
        stake = bankroll * kelly_frac
        if kelly_frac < 0 or ev <0 : 
            profit = 0
            net_odds = 0
        else: 
            profit = stake * (real_odds - 1) if int(group.iloc[i][winner_col]) == bet_idx else -stake
            net_odds = (real_odds - 1) if int(group.iloc[i][winner_col]) == bet_idx else -(real_odds - 1)
        group_stats['choice_fstar'].append(kelly_frac)
        group_stats['fight_payout'].append(profit)
        group_stats['net_odds'].append(net_odds)
        group_profit += profit
    
    return group_stats, sigma_portfolio, sigma_portfolio_scaled, mu_portfolio, sharpe_portfolio, sharpe_per_bet, sigma_per_bet, k, group_stats, group_profit


def run_per_bet_scaling(bets_df, max_drawdown, bankroll, N, max_k=0, sort_max_wins=False, edge_harder=False):

    rows = []
    group_profit = 0

    if sort_max_wins is True:
        bets_df = bets_df.sort_values(by=['ev'], ascending=False).reset_index(drop=True)

    for _, row in bets_df.iterrows():

        f_star = row["f_star_unscaled"]
        p = row["p"]
        fair_odds = row["fair_odds"]
        real_odds = row["real_odds"]
        ev = row["ev"]
        bet_idx = row["bet_idx"]
        winner = row["winner"]

        if f_star <= 0 or ev < 0 or (sort_max_wins is True and row.name >= max_k):
            f_final = 0
            profit = 0
            net_odds = 0
            win = False

            if edge_harder: 
                edge = p - (1/(fair_odds))
                if edge >= -.035:
                    f_final = .035
                    stake = bankroll * f_final
                    win = int(winner) == int(bet_idx)
                    profit = stake * (real_odds - 1) if win else -stake
                    net_odds = (real_odds - 1) if win else -1
        else:
            
            edge = p - (1/(fair_odds))

            adj_mdd = max_drawdown

            if 0.1 <= edge <= 0.15:
                adj_mdd += .02
            elif edge > .15 and edge < .2:
                adj_mdd += .05
            elif edge >= .2 and edge < .25:
                adj_mdd += .1
            elif edge >= .25:
                adj_mdd += .15

            f_final = scale_kelly_for_mdd(p, fair_odds, f_star, N=N, max_drawdown=adj_mdd)
            stake = bankroll * f_final
            win = int(winner) == int(bet_idx)
            
            profit = stake * (real_odds - 1) if win else -stake
            net_odds = (real_odds - 1) if win else -1

        group_profit += profit
        rows.append({ "p": p, "fair_odds": fair_odds, "real_odds": real_odds, "ev": ev,
                     "f_star_unscaled": f_star,"f_star_scaled": f_final,
            "stake": bankroll * f_final,"profit": profit, "net_odds": net_odds, "win": win
        })

    bets_out_df = pd.DataFrame(rows)

    # Portfolio-level metrics
    portfolio_input = bets_out_df.rename(
        columns={"real_odds": "odds", "f_star_unscaled": "f_full"}
    )[["p", "odds", "f_full"]].to_dict("records")

    f_scaled, sigma_portfolio, sigma_portfolio_scaled, mu_portfolio, \
    sharpe_portfolio, sharpe_per_bet, sigma_per_bet, k = \
        scale_kelly_portfolio(portfolio_input, N=N, max_drawdown=max_drawdown)

    portfolio_df = pd.DataFrame([{
        "f_scaled": f_scaled,
        "sigma_portfolio": sigma_portfolio,
        "sigma_portfolio_scaled": sigma_portfolio_scaled,
        "mu_portfolio": mu_portfolio,
        "sharpe_portfolio": sharpe_portfolio,
        "sharpe_per_bet": sharpe_per_bet,
        "sigma_per_bet": sigma_per_bet,
        "k": k,
        "group_profit": group_profit
    }])

    return bets_out_df, portfolio_df, group_profit


def kelly_edge(p, fair_decimal):

    if p <= 0 or p >= 1:
        return 0.0  # invalid probability

    b = fair_decimal - 1  # net odds
    q = 1 - p

    f = (b * p - q) / b

    # Never return a negative fraction (means no value bet)
    return max(0, f)


def poisson_binomial_pmf(probs, k):
    """
    probs : list or array of success probabilities p_i
    k     : number of total successes in the sum
    """
    # Start distribution at P(X=0) = 1
    pmf = np.array([1.0])
    
    # Convolution process
    for p in probs:
        pmf = np.convolve(pmf, [1-p, p])
    return pmf[k]

def pmf_num_wins(choice_proba):
    probs = choice_proba
    k_total = len(probs)
    pmf_vals = [poisson_binomial_pmf(probs, k) for k in range(k_total + 1)]
    best_k = np.argmax(pmf_vals)
    return best_k

def parlay_bottom_odds(data, bankroll, parlay_mdd=.25, cutoff=-350):
    df_bottom = data[data['choice_real_odds'] <= cutoff]
    invalid_parlay = df_bottom.empty or df_bottom.shape[0] == 1
    
    if not invalid_parlay: 
        parlay_prob = np.prod(df_bottom['choice_proba'])
        parlay_odds = np.prod(df_bottom['choice_real_odds'])
        parlay_ev = parlay_prob * parlay_odds - 1

        # function for kelly full 
        b = parlay_odds - 1
        if b <= 0:
            kelly_full = 0.0
        else:
            kelly_full = max((b * parlay_prob - (1 - parlay_prob)) / b, 0)

        # scale kelly mdd 
        if parlay_ev < 0:
            parlay_kelly = 0.0
        else:
            if parlay_mdd is not None:
                parlay_kelly = scale_kelly_for_mdd(parlay_prob, parlay_odds, kelly_full, 2000, parlay_mdd)
            else:
                parlay_kelly = kelly_full

        if parlay_ev < 0:
            parlay_kelly = 0.0
        else:
            if parlay_mdd is not None:
                parlay_kelly = scale_kelly_for_mdd(parlay_prob, parlay_odds, kelly_full, 2000, parlay_mdd)
            else:
                parlay_kelly = kelly_full

        # profit and win 
        stake = bankroll * parlay_kelly
        parlay_win = (df_bottom['winner'] == df_bottom['pred_winner']).all()
        net_odds = parlay_odds - 1

        if parlay_win:
            profit = stake * net_odds
        else:
            profit = -stake
            net_odds = -1
        
        print(profit)

    else:
        # fully zeroed, but schema preserved
        parlay_prob = 0.0
        parlay_odds = 0.0
        parlay_ev = 0.0
        parlay_kelly = 0.0
        stake = 0.0
        profit = 0.0
        net_odds = 0.0

    df_bottom['parlay_prob'] = parlay_prob
    df_bottom['parlay_ev'] = parlay_ev
    
    df_bottom['choice_parlay_name'] = np.where(df_bottom['pred_winner']==1, df_bottom['fighter_red'], df_bottom['fighter_blue'])
    df_bottom['choice_fighter_name'] = df_bottom['choice_parlay_name']
    df_bottom['fstar_parlay'] = parlay_kelly
    df_bottom['parlay_net_odds'] =  net_odds

    df_bottom = df_bottom[['choice_fighter_name', 'fstar_parlay','choice_ev',
                        'choice_real_odds', 'parlay_net_odds', 
                         'pred_winner', 'choice_proba',
                         'parlay_prob', 'parlay_ev',
                        'date', 'winner',]]
    
    return profit, net_odds, df_bottom



def parlay_top_ev(data, bankroll, top_n=[0,1], parlay_mdd=.25):
    
    df_top_n = data.sort_values(by='choice_ev', ascending=False).iloc[top_n].copy()

    invalid_parlay = df_top_n.empty or df_top_n.shape[0] < len(top_n)

    if not invalid_parlay:
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
        if parlay_ev < 0:
            parlay_kelly = 0.0
        else:
            if parlay_mdd is not None:
                parlay_kelly = scale_kelly_for_mdd(parlay_prob, parlay_odds, kelly_full, 2000, parlay_mdd)
            else:
                parlay_kelly = kelly_full

        # profit and win 
        stake = bankroll * parlay_kelly
        parlay_win = (df_top_n['winner'] == df_top_n['pred_winner']).all()
        net_odds = parlay_odds - 1

        if parlay_win:
            profit = stake * net_odds
        else:
            profit = -stake
            net_odds = -1

    else:
        # fully zeroed, but schema preserved
        parlay_prob = 0.0
        parlay_odds = 0.0
        parlay_ev = 0.0
        parlay_kelly = 0.0
        stake = 0.0
        profit = 0.0
        net_odds = 0.0

    df_top_n['parlay_prob'] = parlay_prob
    df_top_n['parlay_ev'] = parlay_ev
    
    df_top_n['choice_parlay_name'] = np.where(df_top_n['pred_winner']==1, df_top_n['fighter_red'], df_top_n['fighter_blue'])
    df_top_n['choice_fighter_name'] = df_top_n['choice_parlay_name']
    df_top_n['fstar_parlay'] = parlay_kelly
    df_top_n['parlay_net_odds'] =  net_odds

    df_top_n = df_top_n[['choice_fighter_name', 'fstar_parlay','choice_ev',
                        'choice_real_odds', 'parlay_net_odds', 
                         'pred_winner', 'choice_proba',
                         'parlay_prob', 'parlay_ev',
                        'date', 'winner',]]

    return profit, net_odds, df_top_n


def simulate_kelly(df_final, prob_cols, fair_decimal_cols, real_decimal_cols,
                        pred_winner_col, winner_col='winner', date_col='date',
                        init_bankroll=1000, bankroll_floor=None, portfolio_scaling=False,\
                        adaptive_scaling=False, max_drawdown=.30, parlay_mdd=.25, N=1000,
                        calc_parlay=False,
                        test_other_ev = False
                        ):

    bankroll = init_bankroll

    df_results = pd.DataFrame()
    df_parlay = pd.DataFrame()
    df = df_final.sort_values(by=date_col)

    for date, group in df.groupby(date_col, sort=True):

        group_stats = {'fight_payout':[], 'choice_fstar':[], 'net_odds':[], 'choice_ev':[], 'choice_juice':[], 'choice_decimal_odds' : []}

        unweighted_fstar = []
        choice_proba = []
        choice_fair_odds = []
        choice_ev = []
        choice_real_odds = []
        choice_idx = []
        choice_posterior = []
        group = group.reset_index(drop=True)

        for idx, row in group.iterrows():

            bet_idx = int(row[pred_winner_col])
            p = row[prob_cols[bet_idx]]
            fair_odds = row[fair_decimal_cols[bet_idx]]
            real_odds = row[real_decimal_cols[bet_idx]]
            ev = expected_value(p, real_odds)
            group_stats['choice_decimal_odds'].append(real_odds)

            # append this bet 
            if ev <= 0 and test_other_ev is True: 
                new_bet = np.abs(bet_idx -1)
                ev = expected_value(1-p, row[real_decimal_cols[new_bet]])
                if ev > 0: 
                    bet_idx = new_bet
                    real_odds = row[real_decimal_cols[bet_idx]]
                    fair_odds = row[fair_decimal_cols[bet_idx]]
                    p = 1-p

            kelly_default = kelly_edge(p, fair_odds) 
            unweighted_fstar.append(kelly_default)

            choice_ev.append(ev)
            choice_proba.append(p)
            choice_fair_odds.append(fair_odds)
            choice_real_odds.append(real_odds)
            choice_idx.append(bet_idx)
            group_stats['choice_juice'].append(row['juice_open_red'] if bet_idx ==1 else row['juice_open_blue'])

        best_k = pmf_num_wins(choice_proba)
        bets_input_df = pd.DataFrame({
                    "p": choice_proba,
                    "fair_odds": choice_fair_odds,
                    "real_odds": choice_real_odds,
                    "ev": choice_ev,
                    "f_star_unscaled": unweighted_fstar,
                    "bet_idx": choice_idx,
                    "winner": group[winner_col].values
                })

        if portfolio_scaling is True: 
            group_stats, sigma_portfolio, \
            sigma_portfolio_scaled, mu_portfolio, \
            sharpe_portfolio, sharpe_per_bet,\
            sigma_per_bet, k,\
             group_stats, ml_profit = run_portfolio_scaling(choice_ev, choice_proba, unweighted_fstar, 
                                                            choice_fair_odds, max_drawdown, bankroll,
                                                            choice_real_odds, choice_idx, 
                                                            winner_col, group, group_stats)

        elif adaptive_scaling is True: 
            bets_out_df, portfolio_df, ml_profit = run_per_bet_scaling(bets_input_df,
                                                            max_drawdown=max_drawdown,
                                                            bankroll=bankroll,
                                                            N=N, max_k=best_k,
                                                            sort_max_wins=False,
                                                            edge_harder=False) #edging harder does not work
            
            edge = np.asarray(choice_proba) - (1/(np.asarray(choice_fair_odds)))
            
                                                                                                                   
        parlay_net_money = 0
        if calc_parlay is True:

            df_data = pd.DataFrame({
                'winner': group['winner'].values,
                'pred_winner': group['pred_winner'].values,
                'choice_ev': choice_ev,
                'choice_real_odds': choice_real_odds,
                'choice_fstar': bets_out_df['f_star_scaled'].values,
                'fighter_red': group['fighter_red'].values,
                'fighter_blue': group['fighter_blue'].values,
                'date': group['date'].values,
                'choice_proba': choice_proba
            })
                        
            parlay_net_money, parlay_net_odds, df_top_n = parlay_top_ev(df_data, bankroll, top_n=[0,1], parlay_mdd=parlay_mdd)
                                                       
            df_top_n['fstar_net'] = np.where(parlay_net_odds >= 0, df_top_n['fstar_parlay'], -df_top_n['fstar_parlay'])
            df_top_n['choice_decimal_odds'] = df_top_n['choice_real_odds'] 

            group_stats['parlay_net'] = np.full(group.shape[0], parlay_net_money)
            group_stats['parlay_net_odds'] = np.full(group.shape[0], parlay_net_odds)

            # if df_data.shape[0]>=3:
            #     profit1, _, _ = parlay_top_ev(df_data, bankroll, top_n=[0,2])
            #     # profit2,_,_ = parlay_top_ev(df_data, bankroll, top_n=[1,2])
            #     parlay_net_money += profit1

            # profit1, _, _ = parlay_bottom_odds(df_data, bankroll, parlay_mdd, cutoff=1.2857)
            # parlay_net_money += profit1


        shape = len(choice_fair_odds)
        assert len(bets_out_df) == shape == group.shape[0], 'bets df and shape incorrect'

        choice_idxs = np.array(choice_ev) > 0

        # bankroll updates 
        event_total_profit = ml_profit + parlay_net_money
        new_bankroll = bankroll + event_total_profit

        bankroll_pct_change = (new_bankroll - bankroll) / bankroll
        group_stats['bankroll_pct_change'] = np.ones(shape)*bankroll_pct_change

        bankroll = new_bankroll
        group_stats['bankroll_postevent'] = np.ones(shape)*bankroll

        # vegas acc for choice fighters 
        odds_ = np.array(group_stats['choice_decimal_odds'])
        mask = np.array(choice_ev) > 0

        vegas_acc_choice = (
            (odds_[mask] < 2).mean()
            if mask.any() else 0.0
        )

        group_stats['vegas_acc_choice'] = np.full(shape, vegas_acc_choice)

        group_stats['event_payout_money_line'] = np.full(shape, ml_profit) # Total ML profit
        group_stats['fight_payout'] = bets_out_df['profit'].values # individual payout
        group_stats['net_odds'] = bets_out_df['net_odds'].values

        group_stats['choice_ev'] = choice_ev
        group_stats['choice_fstar'] = bets_out_df['f_star_scaled'].values
        group_stats['event_ml_net_odds'] = np.full(shape, sum(bets_out_df['net_odds']))

        group_stats['kelly_edge'] = unweighted_fstar

        group_stats['choice_bet_sharpe'] = portfolio_df['sharpe_per_bet'].iloc[0]
        group_stats['choice_bet_sigma'] = portfolio_df['sigma_per_bet'].iloc[0]
        group_stats['choice_idx'] = choice_idx

        group_stats['choice_edge'] = edge
        group_stats['avg_choice_edge'] = np.full(shape, np.mean(edge[choice_idxs]))

        group_stats['date'] = group['date'].to_list()
        group_stats['fighter_red'] = group['fighter_red'].to_list()
        group_stats['fighter_blue'] = group['fighter_blue'].to_list()

        group_stats['pred_winner'] = group['pred_winner'].to_list()
        group_stats['winner'] = group['winner'].to_list()
        
        group_stats['red_proba'] = group[prob_cols[1]].to_list()
        group_stats['blue_proba'] = group[prob_cols[0]].to_list()
        group_stats['choice_proba'] = choice_proba

        group_stats['open_red'] = group['open_red'].to_list()
        group_stats['open_blue'] = group['open_blue'].to_list()
        group_stats['real_odds_blue'] = group[real_decimal_cols].iloc[:,0].values
        group_stats['real_odds_red'] = group[real_decimal_cols].iloc[:,1].values

        group_stats['sigma_portfolio_scaled'] = portfolio_df['sigma_portfolio_scaled'].iloc[0] * np.ones(shape)
        group_stats['portfolio_fscaled'] = portfolio_df['f_scaled'].iloc[0] * np.ones(shape)
        group_stats['portfolio_sigma'] = np.ones(shape)*portfolio_df['sigma_portfolio'].iloc[0]
        group_stats['mu_portfolio'] = np.ones(shape)*portfolio_df['mu_portfolio'].iloc[0]
        group_stats['sharpe_portfolio'] = np.ones(shape)*portfolio_df['sharpe_portfolio'].iloc[0]
        group_stats['k'] = np.ones(shape)* portfolio_df['k'].iloc[0]

        kelly_valid = np.where(np.array(group_stats['choice_ev']) > 0, bets_out_df['f_star_scaled'], 0)
        group_stats['fstar_net'] = np.where(group_stats['pred_winner'] == np.array(group_stats['winner']).astype(int), kelly_valid, -kelly_valid)

        # print([(key, len(value)) for key,value in group_stats.items()])

        if bankroll_floor is not None and bankroll < bankroll_floor:
            print(f"Bankroll low on {date}, adding funds")
            bankroll += 1000

        group_df = pd.DataFrame(group_stats)
        group_df[['juice_open_red', 'juice_open_blue', 'juice_close1_red',
                'juice_close1_blue', 'juice_close2_red', 'juice_close2_blue', 'open_red', 'open_blue',
                'close1_red', 'close1_blue', 'close2_red', 'close2_blue']] = group[['juice_open_red', 'juice_open_blue', 'juice_close1_red',
                'juice_close1_blue', 'juice_close2_red', 'juice_close2_blue',
                'open_red', 'open_blue', 'close1_red', 'close1_blue', 'close2_red', 'close2_blue']].copy() 
        
        if calc_parlay is True: 
            df_parlay = pd.concat([df_parlay , df_top_n], axis=0)
        df_results = pd.concat([df_results, group_df], ignore_index=True)

    return df_results, df_parlay


def calc_event_bankroll_pct(df):
    """
    Calculates percentage bankroll gain per event (grouped by 'date').

    Required columns:
    - 'date'
    - 'choice_fstar'
    - 'choice_decimal_odds'
    """

    df = df.copy()

    # Per-bet bankroll change
    df["event_bankroll_pct"] = (
        df["fstar_net"] * (df["choice_decimal_odds"] - 1)
    )

    # Aggregate per event
    df["event_bankroll_pct"] = (
        df["event_bankroll_pct"]
            .where(df["choice_ev"] > 0, 0)
            .groupby(df["date"])
            .transform("sum")
    )

    return df