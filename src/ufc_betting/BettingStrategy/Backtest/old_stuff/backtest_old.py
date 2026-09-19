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


def calculate_positive_ev_parlays(data, bankroll, n_legs):
    parlays = []

    for idxs in combinations(data.index, n_legs):
        df_parlay = data.loc[list(idxs)]

        parlay_prob = np.prod(df_parlay["choice_proba"])
        parlay_odds = np.prod(df_parlay["choice_real_odds"])
        parlay_ev = parlay_prob * parlay_odds - 1

        # only keep positive EV parlays
        if parlay_ev <= 0:
            continue

        b = parlay_odds - 1
        kelly_full = max((b * parlay_prob - (1 - parlay_prob)) / b, 0) if b > 0 else 0.0

        parlay_kelly = kelly_full
        stake = bankroll * parlay_kelly

        parlay_win = (df_parlay["winner"] == df_parlay["pred_winner"]).all()

        net_odds = parlay_odds - 1
        if parlay_win:
            profit = stake * net_odds
        else:
            profit = -stake
            net_odds = -1

        parlays.append({
            "n_legs": n_legs,
            "parlay_indices": idxs,
            "parlay_prob": parlay_prob,
            "parlay_odds": parlay_odds,
            "parlay_ev": parlay_ev,
            "kelly_full": kelly_full,
            "stake": stake,
            "parlay_win": parlay_win,
            "profit": profit,
            "net_odds": net_odds,
        })

    return parlays


