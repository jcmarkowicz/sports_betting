#https://en.wikipedia.org/wiki/Glicko_rating_system
# https://ojs.aaai.org/index.php/AIIDE/article/view/5233
# https://www.glicko.net/glicko/glicko.pdf 


import numpy as np 
import pandas as pd 
from collections import defaultdict

def ratings_deviation(prev_rd, c, t, rd_unrated=350):
    """
    Params:
        prev_rd: previous ratings deviation 
        c: uncertainty of a players skill 
        t: amount of time (rating periods) since last competition 
        rd_unrated: assumed ratings deviation of unrated player
    """
    rd = min(np.sqrt(prev_rd**2 + c**2 * t), rd_unrated) 
    return rd 

def new_rating(prev_r, opponent_rating_history, opponent_rd_history, outcome_history, new_rd):
    # q = np.log(10) / 400
    q = 0.0057565
    d_squared = compute_d_squared(q, prev_r, opponent_rating_history, opponent_rd_history)
    
    summation = []
    for outcome, rd, r in zip(outcome_history, opponent_rd_history, opponent_rating_history):
        g_rd_i = glicko_rd(q, rd)
        x = g_rd_i * (outcome - expected_outcome(g_rd_i, prev_r, r))
        summation.append(x)

    new_r = prev_r + (q / ((1/new_rd**2) + (1/d_squared))) * np.sum(summation)
    return new_r, d_squared

def glicko_rd(q, rd_i):
    g_rd = 1 / np.sqrt(1 + ((3 * q**2 * rd_i**2) / np.pi**2))
    return g_rd

def expected_outcome(g_rd_i, r0, r_i):
    expected_s = 1 / ( 1 + 10**((g_rd_i * (r0 - r_i)) / -400 ))
    return expected_s 

def compute_d_squared(q, r0, rating_history, rd_history):
    denom = []
    for r, rd in zip(rating_history, rd_history):
        g_rd_i = glicko_rd(q, rd)
        e_s = expected_outcome(g_rd_i, r0, r)
        denom.append(g_rd_i**2 * e_s * (1-e_s))
    d_squared = 1 / (q**2 * np.sum(denom))
    return d_squared

def update_rd(rd_new, d_squared):
    rd_prime = np.sqrt(((1/rd_new**2) + (1/d_squared))**-1)
    return rd_prime 

def compute_ratings(outcome_history, time_between, rating_history, rating_deviation_history, opponent_rating_history, opponent_rd_history, c, r0 = 1500 ,rd_unrated=350):
    
    if len(rating_deviation_history) == 1:
        new_rd = rating_deviation_history[-1]
    else:
        new_rd = ratings_deviation(rating_deviation_history[-1], c, time_between)

    rating_prime, d_squared = new_rating(rating_history[-1], [opponent_rating_history[-1]], [opponent_rd_history[-1]], [outcome_history[-1]], new_rd)
    rd_prime = update_rd(new_rd, d_squared)

    confidence_interval = [rating_prime - 1.96*rd_prime, rating_prime + 1.96*rd_prime]
    return rating_prime, rd_prime, confidence_interval 


def glicko_rating(df):

    inital_rd = 350
    fighter_rd = defaultdict(lambda : [inital_rd])
    fighter_r = defaultdict(lambda: [1500])
    fighter_ci = defaultdict(lambda: [[1500 - 1.96*inital_rd, 1500 + 1.96*inital_rd]])

    fighter_outcome = defaultdict(list)

    for _, row in df.iterrows():

        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']
        winner = row['winner']

        if winner == 1:
            fighter_outcome[red_fighter].append(1)
            fighter_outcome[blue_fighter].append(0)
        else:
            fighter_outcome[red_fighter].append(0)
            fighter_outcome[blue_fighter].append(1)

        time_between = 1
        c = 34
        red_rating_prime, red_rd_prime, red_confidence_interval = compute_ratings(fighter_outcome[red_fighter],
                                                                                time_between, fighter_r[red_fighter],
                                                                                fighter_rd[red_fighter], 
                                                                                fighter_r[blue_fighter], fighter_rd[blue_fighter],
                                                                                c)
        blue_rating_prime, blue_rd_prime, blue_confidence_interval = compute_ratings(fighter_outcome[blue_fighter],
                                                                                time_between, fighter_r[blue_fighter],
                                                                                fighter_rd[blue_fighter], 
                                                                                fighter_r[red_fighter], fighter_rd[red_fighter],
                                                                                c)
        fighter_rd[red_fighter].append(red_rd_prime)
        fighter_r[red_fighter].append(red_rating_prime)
        fighter_ci[red_fighter].append(red_confidence_interval)

        fighter_rd[blue_fighter].append(blue_rd_prime)
        fighter_r[blue_fighter].append(blue_rating_prime)
        fighter_ci[blue_fighter].append(blue_confidence_interval)


    fighter_idx = defaultdict(lambda: [0])
    red_glicko = []
    blue_glicko = []
    red_conf = []
    blue_conf = []
    red_glicko_rd = []
    blue_glicko_rd = []
    for _, row in df.iterrows():
        red_fighter = row['fighter_red']
        blue_fighter = row['fighter_blue']

        blue_idx = fighter_idx[blue_fighter][-1]
        red_idx = fighter_idx[red_fighter][-1]

        red_fighter_rating = fighter_r[red_fighter][red_idx]
        blue_fighter_rating = fighter_r[blue_fighter][blue_idx]

        red_rd = fighter_rd[red_fighter][red_idx]
        blue_rd = fighter_rd[blue_fighter][blue_idx]

        red_ci = fighter_ci[red_fighter][red_idx]
        blue_ci = fighter_ci[blue_fighter][blue_idx]

        red_glicko.append(red_fighter_rating)
        blue_glicko.append(blue_fighter_rating)
        red_glicko_rd.append(red_rd)
        blue_glicko_rd.append(blue_rd)
        red_conf.append(red_ci)
        blue_conf.append(blue_ci)

        fighter_idx[blue_fighter][-1] += 1
        fighter_idx[red_fighter][-1] += 1
        
    return np.column_stack([red_glicko, blue_glicko, red_glicko_rd, blue_glicko_rd])