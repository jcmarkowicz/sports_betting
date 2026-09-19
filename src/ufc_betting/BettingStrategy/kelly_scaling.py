import numpy as np 

def scale_mdd(edge, mdd):
    adj_mdd = mdd
    if 0.1 <= edge <= 0.15:
        adj_mdd += .02
    elif edge > .15 and edge < .2:
        adj_mdd += .05
    elif edge >= .2 and edge < .25:
        adj_mdd += .1
    elif edge >= .25:
        adj_mdd += .15

    return adj_mdd

def kelly_edge(p, fair_decimal):

    if p <= 0 or p >= 1:
        return 0.0  # invalid probability

    b = fair_decimal - 1  # net odds
    q = 1 - p

    f = (b * p - q) / b
    return max(0, f)


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
    Extreme-value approximation in log-return units over N bets.
    """
    if N <= 1:
        raise ValueError("N must be greater than 1")

    leading_term = np.sqrt(2 * np.log(N))
    correction = (
        np.log(np.log(N)) + np.log(4 * np.pi)
    ) / (2 * leading_term)
    return sigma * (
        leading_term - correction
    )


def scale_kelly_for_mdd(p, odds, f_full, N, max_drawdown, tol=1e-4, return_sigma=False):
    """
    Find the largest fraction of full Kelly that keeps expected MDD <= max_drawdown
    
    p : probability of winning
    odds : decimal odds
    f_full : full Kelly fraction (fraction of bankroll)
    N : number of bets
    max_drawdown : tolerable drawdown fraction (0 < max_drawdown < 1)
    tol : numerical tolerance for convergence
    """
    if not 0 < max_drawdown < 1:
        raise ValueError("max_drawdown must be between 0 and 1")
    if N <= 1:
        raise ValueError("N must be greater than 1")

    b = odds - 1
    max_log_drawdown = -np.log1p(-max_drawdown)
    # binary search between 0 and 1 (fraction of full Kelly)
    low, high = 0.0, 1.0
    best_fraction = 0.0
    
    while high - low > tol:
        k = (low + high) / 2
        f_trial = k * f_full
        sigma, mu = log_return_volatility(f_trial, b, p)
        mdd_est = expected_max_drawdown(sigma, N)
        
        if mdd_est <= max_log_drawdown:
            best_fraction = k  # this fraction is safe, try higher
            low = k
        else:
            high = k  # too aggressive, try lower
    if return_sigma:
        sigma, mu = log_return_volatility(best_fraction * f_full, b, p)
        return best_fraction * f_full, sigma, mu         
    return best_fraction * f_full


