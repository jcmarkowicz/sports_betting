# scipy minimize optimizer
# https://arxiv.org/pdf/2307.13807 


from scipy.optimize import minimize
import itertools 
import numpy as np 

def multivariate_simultaneous_kelly(probs, odds, max_fraction_per_bet=0.15, epsilon=1e-12):
    """
    Multivariate simultaneous Kelly optimizer.

    probs: array (r x m) of probabilities per event/outcome
    odds: array (r x m) of decimal odds
    max_fraction_per_bet: maximum fraction of bankroll per bet
    epsilon: small number to avoid log(0)
    """
    r, m = probs.shape
    M = r * m  # total number of bets

    # Build all joint outcome combinations
    joint_indices = list(itertools.product(*[range(m) for _ in range(r)]))
    N = len(joint_indices)

    # Build joint probability vector
    p = np.zeros(N)
    for j, combo in enumerate(joint_indices):
        prob = 1.0
        for fight_idx, outcome_idx in enumerate(combo):
            prob *= probs[fight_idx, outcome_idx]
        p[j] = prob
    assert np.isclose(p.sum(), 1.0), f"Probabilities do not sum to 1, got {p.sum()}"

    # Build W matrix: each row = a bet, each column = joint scenario
    W = np.zeros((M, N))
    for bet_idx in range(M):
        fight = bet_idx // m
        outcome = bet_idx % m
        for j, combo in enumerate(joint_indices):
            W[bet_idx, j] = (odds[fight, outcome]) if combo[fight] == outcome else 0

    # Objective: negative expected log growth
    def objective(l, p, W):
        v = 1 + W.T @ l - np.sum(l)
        v = np.maximum(v, epsilon)  # avoid log(0) or negative
        return -np.dot(p, np.log(v))

    # Bounds: each bet fraction between 0 and max_fraction_per_bet
    bounds = [(0, max_fraction_per_bet) for _ in range(M)]

    # Constraint: sum of bets <= 1 (total bankroll)
    constraints = [{'type': 'ineq', 'fun': lambda l: 1 - np.sum(l)}]

    # Initial guess: small fraction to avoid hitting zero scenarios
    initial_l = np.full(M, min(max_fraction_per_bet / 2, 0.05))

    # Optimize
    result = minimize(
        objective,
        initial_l,
        args=(p, W),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )

    if not result.success:
        print("Warning: optimizer did not converge:", result.message)

    return result.x

def clip(x):
    """Clip value to [0,1]."""
    return max(0.0, min(1.0, x))

def multiple_simultaneous_expectation_log_wealth(bets_data, fs):
    """
    Compute expected log-wealth and its gradient for multiple simultaneous Kelly bets.

    Parameters
    ----------
    bets_data : list of (p, odds)
        p = win probability (float between 0 and 1)
        odds = decimal odds
    fs : list or array of float
        bet fractions (same length as bets_data)

    Returns
    -------
    res : float
        expected log-wealth
    grad : list of float
        gradient vector
    """
    n = len(bets_data)
    assert len(fs) == n

    res = 0.0
    grad = np.zeros(n)

    # Iterate over all win/loss combinations
    for outcome in itertools.product([0, 1], repeat=n):
        prob = 1.0
        wealth = 1.0
        local_grad = np.zeros(n)

        for i, j in enumerate(outcome):
            p, odds = bets_data[i]
            if j == 0:
                # win
                prob *= p
                wealth += fs[i] * odds
                local_grad[i] += odds
            else:
                # loss
                prob *= 1.0 - p
                wealth -= fs[i]
                local_grad[i] -= 1.0

        if wealth > 0.0:
            res += prob * np.log(wealth)
            grad += prob * local_grad / wealth

    return res, grad.tolist()

def multiple_simultaneous_kelly(bets_data, alpha=0.01, max_iter=100):
    """
    Gradient ascent optimization for simultaneous Kelly fractions.

    Parameters
    ----------
    bets_data : list of (p, odds)
        p = win probability
        odds = decimal odds
    alpha : float
        learning rate
    max_iter : int
        maximum iterations

    Returns
    -------
    value_var : float
        final expected log-wealth
    fs_var : list of float
        optimal bet fractions
    """
    value_var = 0.0
    fs_var = [0.0] * len(bets_data)

    for _ in range(max_iter):
        value, grad = multiple_simultaneous_expectation_log_wealth(bets_data, fs_var)
        fs_candidate = [clip(f + alpha * g) for f, g in zip(fs_var, grad)]

        # Scale if fractions sum > 1
        total = sum(fs_candidate)
        if total > 1.0:
            fs_candidate = [f / total for f in fs_candidate]

        value_candidate, _ = multiple_simultaneous_expectation_log_wealth(bets_data, fs_candidate)

        if value_candidate <= value:
            break
        else:
            fs_var = fs_candidate
            value_var = value_candidate

    return value_var, fs_var
