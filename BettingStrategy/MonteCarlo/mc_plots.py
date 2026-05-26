
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

def plot_final_distributions(mc_ml_open, mc_parlay_open, mc_total_open, path):

    # cumulative P/L paths
    ml_final = np.cumsum(np.asarray(mc_ml_open), axis=1)[:, -1]
    parlay_final = np.cumsum(np.asarray(mc_parlay_open), axis=1)[:, -1]
    total_final = np.array(mc_total_open)[:, -1]

    # probabilities above zero
    ml_prob_positive = (ml_final > 0).mean()
    parlay_prob_positive = (parlay_final > 0).mean()
    total_prob_positive = (total_final > 0).mean()

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Money Line
    sns.histplot(
        ml_final,
        bins=40,
        kde=True,
        ax=axes[0]
    )

    axes[0].axvline(0, linestyle='--')

    axes[0].set_title(
        f"Money Line Final Distribution\n"
        f"P(Final > 0) = {ml_prob_positive:.2%}, Avg Final = {ml_final.mean():.2f}, Min Final = {ml_final.min():.2f}, Max Final = {ml_final.max():.2f}"
    )

    axes[0].set_xlabel("Final Cumulative Result")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)

    # Parlay
    sns.histplot(
        parlay_final,
        bins=40,
        kde=True,
        ax=axes[1]
    )

    axes[1].axvline(0, linestyle='--')

    axes[1].set_title(
        f"Parlay Final Distribution\n"
        f"P(Final > 0) = {parlay_prob_positive:.2%}, Avg Final = {parlay_final.mean():.2f}, Min Final = {parlay_final.min():.2f}, Max Final = {parlay_final.max():.2f}"
    )

    axes[1].set_xlabel("Final Cumulative Result")
    axes[1].set_ylabel("Count")
    axes[1].grid(True)

    sns.histplot(
        total_final,
        bins=40,
        kde=True,
        ax=axes[2]
    )

    axes[2].axvline(0, linestyle='--')

    axes[2].set_title(
        f"Total Bankroll Final Distribution\n"
        f"P(Final > 0) = {total_prob_positive:.2%}, Avg Final = {total_final.mean():.2f}, Min Final = {total_final.min():.2f}, Max Final = {total_final.max():.2f}"
    )

    axes[2].set_xlabel("Final Cumulative Result")
    axes[2].set_ylabel("Count")
    axes[2].grid(True)

    plt.savefig(path)

    plt.tight_layout()
    plt.show()

def plot_negative_fraction_histograms(
    mc_ml_open,
    mc_parlay_open,
    mc_total_open,
    path
):

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    datasets = [
        ("Money Line", np.cumsum(np.asarray(mc_ml_open), axis=1)),
        ("Parlay", np.cumsum(np.asarray(mc_parlay_open), axis=1)),
        ("Total Bankroll", np.asarray(mc_total_open))
    ]

    for ax, (title, data) in zip(axes, datasets):

        # fraction of observations below zero per path
        pct_below_zero = (data < 0).mean(axis=1)

        avg_fraction = pct_below_zero.mean()

        sns.histplot(
            pct_below_zero,
            bins=30,
            kde=True,
            ax=ax
        )

        ax.axvline(
            avg_fraction,
            linestyle='--',
            linewidth=2,
            label=f"Mean = {avg_fraction:.3f}"
        )

        ax.set_title(
            f"{title} - Fraction Below Zero Per Path\n"
            f"Average Fraction Below Zero = {avg_fraction:.3f}"
        )

        ax.set_xlabel("Fraction Below Zero")
        ax.set_ylabel("Count")
        ax.grid(True)
        ax.legend()

    plt.savefig(path)

    plt.tight_layout()
    plt.show()


def mc_analysis(mc_ml_open, mc_parlay_open, mc_total_open, path):

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    datasets = [
        ("Money Line", np.cumsum(np.array(mc_ml_open), axis=1)),
        ("Parlay", np.cumsum(np.array(mc_parlay_open), axis=1)),
        ("Total Bankroll", np.array(mc_total_open))
    ]

    for ax, (title, data) in zip(axes, datasets):

        data = np.asarray(data)

        mean_path = data.mean(axis=0)

        lower = np.percentile(data, 2.5, axis=0)
        upper = np.percentile(data, 97.5, axis=0)

        x = np.arange(data.shape[1])

        # individual MC paths
        ax.plot(data.T, alpha=0.15)

        # mean
        ax.plot(x, mean_path, linewidth=3, label='Mean')

        # confidence interval
        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.175,
            label='95% CI'
        )

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.savefig(path)
    plt.tight_layout()
    plt.show()

    