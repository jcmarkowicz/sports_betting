import trueskill
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from particle_filter import predict_particle_probs, update_weights, resample_and_jitter, effective_sample_size, init_particles

env = trueskill.TrueSkill(
    mu=30.0,
    sigma=25.0 / 3.0,
    beta=45.0 / 6.0,
    tau=40.0 / 300.0,   # small default dynamics factor
    draw_probability=0.0
)

@dataclass
class FighterState:
    rating: trueskill.Rating
    last_fight_date: pd.Timestamp | None = None


def inflate_for_inactivity(
    rating: trueskill.Rating,
    days_inactive: int,
    sigma_max: float = 25.0 / 3.0,
    annual_inflation: float = 1.25,
):
    """
    Increase uncertainty as fighter becomes inactive.
    Does not change mu, only sigma.
    """
    years = max(days_inactive, 0) / 365.25

    new_sigma = np.sqrt(rating.sigma**2 + (annual_inflation * years)**2)
    new_sigma = min(new_sigma, sigma_max)

    return env.create_rating(mu=rating.mu, sigma=new_sigma)


def get_state(fighters, name):
    if name not in fighters:
        fighters[name] = FighterState(rating=env.create_rating())
    return fighters[name]


def update_fight(red, blue, winner, fight_date, red_state, blue_state):
    fight_date = pd.Timestamp(fight_date)

    r_red = red_state.rating
    r_blue = blue_state.rating

    # lower rank number means better result in trueskill
    if winner == red:
        new_red, new_blue = env.rate_1vs1(r_red, r_blue)
    elif winner == blue:
        new_blue, new_red = env.rate_1vs1(r_blue, r_red)
    else:
        raise ValueError("winner must be red or blue")

    red_state.rating = new_red
    blue_state.rating = new_blue

    red_state.last_fight_date = fight_date
    blue_state.last_fight_date = fight_date


def expected_score(r_red, r_blue):
    delta_mu = r_red.mu - r_blue.mu
    denom = np.sqrt(
        2 * env.beta**2 + r_red.sigma**2 + r_blue.sigma**2
    )
    return env.cdf(delta_mu / denom)


def run_trueskill(df, inflate=False, particle_filter=False):
    fighters = {}
    rows = []

    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    weight_classes = df["weight_class"].dropna().unique()

    particle_filters = {}
    n_particles = 5000
    for wc in weight_classes:
        p, w = init_particles(n_particles)
        particle_filters[wc] = {
            "particles": p,
            "weights": w
        }

    for idx, row in df.iterrows():

        red = row["fighter_red"]
        blue = row["fighter_blue"]
        winner = row["winner_name"]
        fight_date = row["date"]

        red_state = get_state(fighters, red)
        blue_state = get_state(fighters, blue)

        # inactivity inflation BEFORE prediction
        if inflate:
            if red_state.last_fight_date is not None:
                days = (fight_date - red_state.last_fight_date).days
                red_state.rating = inflate_for_inactivity(red_state.rating, days)

            if blue_state.last_fight_date is not None:
                days = (fight_date - blue_state.last_fight_date).days
                blue_state.rating = inflate_for_inactivity(blue_state.rating, days)

        red_rating = red_state.rating
        blue_rating = blue_state.rating

        # make row contain CURRENT pre-fight TrueSkill ratings
        pred_row = row.copy()
        pred_row["red_mu"] = red_rating.mu
        pred_row["red_sigma"] = red_rating.sigma
        pred_row["blue_mu"] = blue_rating.mu
        pred_row["blue_sigma"] = blue_rating.sigma

        # age-adjusted particle prediction
        if particle_filter: 
            wc = row["weight_class"]
            particles = particle_filters[wc]["particles"]
            weights = particle_filters[wc]["weights"]

            p_red_samples = predict_particle_probs(pred_row, particles, env)
            p_red = np.average(p_red_samples, weights=weights)

            peak_age_mean = np.average(particles["peak_age"], weights=weights),
            pre_peak_gain_mean = np.average(particles["pre_peak_gain"], weights=weights),
            post_peak_decline_mean = np.average(particles["post_peak_decline"], weights=weights)
        
        else: 
            peak_age_mean = None
            pre_peak_gain_mean = None
            post_peak_decline_mean = None
            p_red = expected_score(red_rating, blue_rating)

        y_true = int(winner == red)

        rows.append({
            "index": idx,
            "date": fight_date,
            "fighter_red": red,
            "fighter_blue": blue,
            "winner": winner,
            "y_true": y_true,
            "red_mu": red_rating.mu,
            "red_sigma": red_rating.sigma,
            "blue_mu": blue_rating.mu,
            "blue_sigma": blue_rating.sigma,
            "p_red": p_red,
            "pred": int(p_red >= 0.5),
            "peak_age_mean": peak_age_mean,
            "pre_peak_gain_mean": pre_peak_gain_mean,
            "post_peak_decline_mean": post_peak_decline_mean,
        })

        # update age model from fight result
        if particle_filter: 
            weights = update_weights(weights, p_red_samples, y_true)

            if effective_sample_size(weights) < n_particles / 2:
                particles, weights = resample_and_jitter(particles, weights)

            particle_filters[wc]["particles"] = particles
            particle_filters[wc]["weights"] = weights

        # update TrueSkill from fight result
        update_fight(
            red=red,
            blue=blue,
            winner=winner,
            fight_date=fight_date,
            red_state=red_state,
            blue_state=blue_state
        )

    ratings_history = pd.DataFrame(rows)
    return ratings_history, particle_filters 