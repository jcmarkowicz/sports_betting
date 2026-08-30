import numpy as np
import pandas as pd 

def init_particles(n_particles=5000):
    particles = pd.DataFrame({
        "peak_age": np.random.normal(32, 3, n_particles),
        "pre_peak_gain": np.random.normal(0.15, 0.05, n_particles),
        "post_peak_decline": np.random.normal(0.5, 0.15, n_particles),
    })

    # reflect peak_age
    particles.loc[particles["peak_age"] < 18, "peak_age"] = (
        18 + (18 - particles.loc[particles["peak_age"] < 18, "peak_age"])
    )

    particles.loc[particles["peak_age"] > 45, "peak_age"] = (
        45 - (
            particles.loc[particles["peak_age"] > 45, "peak_age"] - 45
        )
    )

    # reflect pre_peak_gain
    particles.loc[particles["pre_peak_gain"] < 0, "pre_peak_gain"] = (
        -particles.loc[particles["pre_peak_gain"] < 0, "pre_peak_gain"]
    )

    particles.loc[particles["pre_peak_gain"] > 1, "pre_peak_gain"] = (
        1 - (
            particles.loc[particles["pre_peak_gain"] > 1, "pre_peak_gain"] - 1
        )
    )

    # reflect post_peak_decline
    particles.loc[particles["post_peak_decline"] < 0, "post_peak_decline"] = (
        -particles.loc[particles["post_peak_decline"] < 0, "post_peak_decline"]
    )

    particles.loc[particles["post_peak_decline"] > 2, "post_peak_decline"] = (
        2 - (
            particles.loc[particles["post_peak_decline"] > 2, "post_peak_decline"] - 2
        )
    )

    weights = np.ones(n_particles) / n_particles

    return particles, weights


def particle_age_adjustment(age, particles):
    if pd.isna(age):
        return np.zeros(len(particles))

    age = float(age)

    return np.where(
        age <= particles["peak_age"].values,
        particles["pre_peak_gain"].values * (particles["peak_age"].values - age),
        -particles["post_peak_decline"].values * (age - particles["peak_age"].values)
    )


def predict_particle_probs(row, particles, env):
    red_adj = particle_age_adjustment(row["age_red"], particles)
    blue_adj = particle_age_adjustment(row["age_blue"], particles)

    delta_mu = (
        row["red_mu"]
        + red_adj
        - row["blue_mu"]
        - blue_adj
    )

    denom = np.sqrt(
        2 * env.beta**2
        + row["red_sigma"]**2
        + row["blue_sigma"]**2
    )
    
    # delta_mu = row["red_mu"] - row["blue_mu"]

    # red_sigma_eff = np.sqrt(row["red_sigma"]**2 + red_adj**2)
    # blue_sigma_eff = np.sqrt(row["blue_sigma"]**2 + blue_adj**2)

    # denom = np.sqrt(
    #     2 * env.beta**2
    #     + red_sigma_eff**2
    #     + blue_sigma_eff**2
    # )

    z = delta_mu / denom
    return np.array([env.cdf(v) for v in z])


def update_weights(weights, p_red_samples, y_true):
    
    likelihood = (
        p_red_samples
        if y_true == 1
        else 1 - p_red_samples
    )

    likelihood = np.clip(likelihood, 1e-8, 1.0)

    # give more weight to the particles with correct outcomes
    weights = weights * likelihood
    weights = weights / weights.sum()

    return weights

def effective_sample_size(weights):
    return 1.0 / np.sum(weights**2)


def resample_and_jitter(particles, weights):
    n = len(weights)

    idx = np.random.choice(
        np.arange(n),
        size=n,
        replace=True,
        p=weights
    )

    particles = particles.iloc[idx].reset_index(drop=True).copy()

    particles["peak_age"] += np.random.normal(0, 0.20, n)
    particles["pre_peak_gain"] += np.random.normal(0, 0.01, n)
    particles["post_peak_decline"] += np.random.normal(0, 0.025, n)

    # reflect peak_age
    particles.loc[particles["peak_age"] < 18, "peak_age"] = (
        18 + (18 - particles.loc[particles["peak_age"] < 18, "peak_age"])
    )

    particles.loc[particles["peak_age"] > 45, "peak_age"] = (
        45 - (
            particles.loc[particles["peak_age"] > 45, "peak_age"] - 45
        )
    )

    # reflect pre_peak_gain
    particles.loc[particles["pre_peak_gain"] < 0, "pre_peak_gain"] = (
        -particles.loc[particles["pre_peak_gain"] < 0, "pre_peak_gain"]
    )

    particles.loc[particles["pre_peak_gain"] > 1, "pre_peak_gain"] = (
        1 - (
            particles.loc[particles["pre_peak_gain"] > 1, "pre_peak_gain"] - 1
        )
    )

    # reflect post_peak_decline
    particles.loc[particles["post_peak_decline"] < 0, "post_peak_decline"] = (
        -particles.loc[particles["post_peak_decline"] < 0, "post_peak_decline"]
    )

    particles.loc[particles["post_peak_decline"] > 2, "post_peak_decline"] = (
        2 - (
            particles.loc[particles["post_peak_decline"] > 2, "post_peak_decline"] - 2
        )
    )

    weights = np.ones(n) / n

    return particles, weights