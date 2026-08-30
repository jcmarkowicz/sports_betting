import matplotlib.pyplot as plt
import numpy as np


def plot_fair_odds_curve(curve_df, title="Fair Odds Curve (Red side)"):
    plt.figure(figsize=(7,5))
    # Empirical (binned) fair odds from your market
    plt.scatter(curve_df["p_fair_red_mean"], curve_df["odds_fair_red_median"],
                label="Empirical fair price (median)", s=25)
    # Theoretical 1/p line
    plt.plot(curve_df["p_fair_red_mean"], curve_df["odds_theoretical"],
             label="Theoretical fair odds = 1/p", linewidth=2)
    plt.xlabel("Fair probability (Red)")
    plt.ylabel("Decimal odds")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_prob_margin(df, fair_col, initial_col, title="Margin Between Market & Fair Probabilities"):
    """
    Plot initial (vigged) probabilities vs their margin over fair (de-vigged).
    """
    df = df.copy()
    df["margin"] = df[initial_col] - df[fair_col]

    plt.figure(figsize=(8,6))
    plt.scatter(df[initial_col], df["margin"], alpha=0.7, c="blue")
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Initial (Market) Probability")
    plt.ylabel("Margin (Market - Fair)")
    plt.title(title)
    # plt.grid(True)
    plt.show()

def plot_fair_vs_market_probs(df, fair_col, initial_col, title="Market vs Fair Probabilities"):
    plt.figure(figsize=(8,6))
    plt.scatter(df[fair_col], df[initial_col], alpha=0.6, label="Market (vigged)")
    plt.scatter(df[fair_col], df[fair_col], alpha=0.6, label="Fair (de-vigged)")
    plt.plot([0,1],[0,1], "k--", label="y = x")

    plt.xlabel("Fair Probability")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_margin_hist(df, fair_col, initial_col, title="Distribution of Market Margin"):
    margin = df[initial_col] - df[fair_col]
    plt.figure(figsize=(8,6))
    plt.hist(margin, bins=30, alpha=0.7, color="purple")
    plt.axvline(0, color="red", linestyle="--")

    plt.xlabel("Margin (Market - Fair)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.show()