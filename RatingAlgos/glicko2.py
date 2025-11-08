import numpy as np 

# g(phi) function
def g(phi):
    return 1 / np.sqrt(1 + 3 * (phi**2) / (np.pi**2))

# E(r, r_j, phi_j)
def E(mu, mu_j, phi_j):
    return 1 / (1 + np.exp(-g(phi_j) * (mu - mu_j)))

# Convert rating to Glicko-2 scale
def scale_down(rating):
    return (rating - 1500) / 173.7178

# Convert rating back to original scale
def scale_up(mu):
    return 173.7178 * mu + 1500

def compute_v(mu, mu_j_list, phi_j_list):
    summation = 0
    for mu_j, phi_j in zip(mu_j_list, phi_j_list):
        e = E(mu, mu_j, phi_j)
        summation += (g(phi_j) ** 2) * e * (1 - e)
    return 1 / summation

def compute_delta(mu, mu_j_list, phi_j_list, outcomes):
    v = compute_v(mu, mu_j_list, phi_j_list)
    summation = 0
    for mu_j, phi_j, s in zip(mu_j_list, phi_j_list, outcomes):
        e = E(mu, mu_j, phi_j)
        summation += g(phi_j) * (s - e)
    return v * summation

def update_sigma(mu, phi, sigma, mu_j_list, phi_j_list, outcomes):
    delta = compute_delta(mu, mu_j_list, phi_j_list, outcomes)
    v = compute_v(mu, mu_j_list, phi_j_list)
    
    a = np.log(sigma**2)
    A = a
    B = None
    if delta**2 > phi**2 + v:
        B = np.log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k*TAU, delta, phi, v, a) < 0:
            k += 1
        B = a - k*TAU

    f_B = lambda x: f(x, delta, phi, v, a)
    
    # Newton-Raphson iteration
    f_A = lambda x: f(x, delta, phi, v, a)
    while abs(B - A) > EPSILON:
        C = A + (A - B) * f_A(A) / (f_B(B) - f_A(A))
        f_C = f(C, delta, phi, v, a)
        if f_C * f_B(B) < 0:
            A = B
            f_A = f_B
        else:
            f_A = lambda x: f_A(x)/2
        B = C
        f_B = f_C
    sigma_prime = np.exp(A/2)
    return sigma_prime

def f(x, delta, phi, v, a):
    exp_x = np.exp(x)
    num = exp_x * (delta**2 - phi**2 - v - exp_x)
    denom = 2 * (phi**2 + v + exp_x)**2
    return num / denom - (x - a) / (TAU**2)

def update_rating(mu, phi, sigma_prime, mu_j_list, phi_j_list, outcomes):
    v = compute_v(mu, mu_j_list, phi_j_list)
    delta = compute_delta(mu, mu_j_list, phi_j_list, outcomes)
    
    phi_star = np.sqrt(phi**2 + sigma_prime**2)
    phi_prime = 1 / np.sqrt(1/phi_star**2 + 1/v)
    
    # Use delta instead of recomputing summation
    mu_prime = mu + (phi_prime**2 / v) * delta
    
    return mu_prime, phi_prime


# Constants for Glicko-2
TAU = 0.3       # System constant, usually 0.3-1.2
EPSILON = 1e-6  # Convergence tolerance

rating = 1500
rd = 320
sigma = 0.06

# Opponents
opponent_ratings = [1400, 1550, 1700]
opponent_rds = [30, 100, 300]
outcomes = [1, 0, 1]  # 1 = win, 0 = loss

# Scale down
mu = scale_down(rating)
phi = rd / 173.7178
mu_j_list = [scale_down(r) for r in opponent_ratings]
phi_j_list = [r/173.7178 for r in opponent_rds]



player_sigma = sigma 
mu = (player_rating - 1500) / 173.7178
phi = player_rd / 173.7178


sigma = player_sigma
for match in matches:
    opp_rating = match['opponent_rating']
    opp_rd = match['opponent_rd']
    outcome = match['outcome']  # 1, 0, or 0.5
    
    mu_j = (opp_rating - 1500) / 173.7178
    phi_j = opp_rd / 173.7178
    
    # Update volatility (sigma)
    sigma = update_sigma(mu, phi, sigma, [mu_j], [phi_j], [outcome])
    
    # Update rating and RD
    mu, phi = update_rating(mu, phi, sigma, [mu_j], [phi_j], [outcome])
    
    # Optional: convert back to original scale for logging
    player_rating = mu * 173.7178 + 1500
    player_rd = phi * 173.7178
    
    print(f"After match vs {opp_rating}, new rating: {player_rating:.1f}, RD: {player_rd:.1f}")