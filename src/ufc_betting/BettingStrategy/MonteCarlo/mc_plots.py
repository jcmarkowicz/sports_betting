
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_bankroll_paths(results, *, len_temp=1, max_paths=100, scale='auto', linthresh=1):
    """Plot {label: simulate_result}; return (figure, axes) without saving files."""
    if not results:
        raise ValueError('results must contain at least one simulation result')
    fig, axes = plt.subplots(1, len(results), figsize=(12 * len(results), 6),
                             layout='constrained', squeeze=False)
    axes = axes[0]
    for ax, (label, result) in zip(axes, results.items()):
        paths = np.asarray(result[2], dtype=float)
        value_scale = ('log' if (paths > 0).all() else 'symlog') if scale == 'auto' else scale
        ax.set_yscale(value_scale, **({'linthresh': linthresh} if value_scale == 'symlog' else {}))
        x = np.arange(paths.shape[1])
        ax.plot(x, paths[:max_paths].T, color='#3579ad', alpha=.10, linewidth=.7)
        ax.plot(x, np.median(paths, axis=0), color='#c44e52', linewidth=2,
                label='Pointwise median')
        ax.axhline(paths[0, 0], color='gray', linestyle=':')
        ax.set(title='Simulated Bankroll Paths' if len(results) == 1 else f'Simulated Bankroll Paths — {label}', xlabel='Settled event batches' if len_temp > 1 else 'Completed events',
               ylabel=f'Bankroll ($, {value_scale} scale)')
        ax.legend()
        ax.grid(alpha=.2)
    return fig, axes

def _distribution_title(title, groups, *, mean_std=False):
    """Summarize each displayed group using unrounded observations."""
    lines = [title]
    for label, values in groups:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            lines.append(f"{label}: no defined values")
        elif mean_std:
            std = f"{np.std(values, ddof=1):.2f}%" if len(values) > 1 else "N/A"
            lines.append(f"{label}: Mean {np.mean(values):.2f}% | SD {std}")
        else:
            p05, median, p75 = np.quantile(values, [.05, .5, .75])
            lines.append(f"{label}: Median {median:.2f}% | P05 {p05:.2f}% | P75 {p75:.2f}%")
    return "\n".join(lines)


def plot_terminal_return_distributions(results, *, bins=30, cdf_scale='symlog', linthresh=100):
    """One terminal-return histogram, without a cumulative panel.

    Returns (figure, axes) with axes shaped (1, 1). cdf_scale retains its old
    keyword for callers but now controls only the histogram x-axis. All groups
    share bins; each title summary uses that group's path returns, including ruin.
    """
    if not results:
        raise ValueError('results must contain at least one simulation result')
    if cdf_scale not in ('linear', 'symlog'):
        raise ValueError('cdf_scale must be linear or symlog')
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError('linthresh must be positive and finite')
    returns = {}
    for label, result in results.items():
        paths = np.asarray(result[2], dtype=float)
        if (paths.ndim != 2 or 0 in paths.shape or not np.isfinite(paths).all()
                or (paths[:, 0] <= 0).any()):
            raise ValueError('Bankroll paths must be nonempty, finite, and start positive')
        returns[label] = 100 * (paths[:, -1] / paths[:, 0] - 1)
    fig, axes = plt.subplots(1, 1, figsize=(14, 7), layout='constrained', squeeze=False)
    ax = axes[0, 0]
    if cdf_scale == 'symlog':
        ax.set_xscale('symlog', linthresh=linthresh)
    values = np.concatenate(list(returns.values()))
    edges = np.histogram_bin_edges(values, bins=bins)
    if cdf_scale == 'symlog' and np.isscalar(bins) and not isinstance(bins, str) and np.ptp(values) > 0:
        transform = ax.xaxis.get_transform()
        edges = transform.inverted().transform(np.linspace(
            *transform.transform([values.min(), values.max()]), int(bins) + 1))
        edges[0], edges[-1] = values.min(), values.max()
    for label, values in returns.items():
        ax.hist(values, bins=edges, alpha=.65, label=label)
    ax.set(title=_distribution_title('Terminal return distribution', returns.items()),
           xlabel='Total return (%)', ylabel='Number of paths')
    if len(returns) > 1:
        ax.legend()
    ax.axvline(-100, color='gray', linestyle=':')
    ax.grid(alpha=.2)
    return fig, axes


def _plot_path_histogram(frame, column, title, xlabel, *, mean_std=False, nonnegative=False, zero_reference=True):
    """Shared histogram layout for per-path statistics, excluding undefined values."""
    valid = frame.loc[np.isfinite(frame[column])].copy()
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
    groups = [(label, group[column].to_numpy()) for label, group in valid.groupby('Prices', sort=False)]
    if groups:
        # Avoid degenerate KDE fits for constant groups or a single observation.
        kde = all(len(values) > 1 and np.ptp(values) > 0 for _, values in groups)
        sns.histplot(data=valid, x=column, hue='Prices', bins='auto', kde=kde,
                     kde_kws={'cut': 0, **({'clip': (0, None)} if nonnegative else {})},
                     element='bars', edgecolor='black', linewidth=.6, alpha=.4, ax=ax)
    else:
        ax.text(.5, .5, 'No defined values', ha='center', transform=ax.transAxes)
    ax.set(title=_distribution_title(title, groups, mean_std=mean_std),
           xlabel=xlabel, ylabel='Number of paths')
    if nonnegative:
        ax.set_xlim(left=0)
    if zero_reference:
        ax.axvline(0, color='gray', linestyle=':')
    else:
        ax.autoscale(enable=True, axis='x')
    return fig, ax


def plot_max_drawdown_histogram(path_statistics):
    """Plot worst peak-to-trough drawdowns, with median/P05/P75 in the title."""
    return _plot_path_histogram(path_statistics, 'Max drawdown (%)',
        'Worst peak-to-trough drawdown per path', 'Maximum drawdown (%)', nonnegative=True)


def plot_geometric_growth_distribution(growth_by_path):
    """Plot defined per-path growth rates, with median/P05/P75 in the title."""
    return _plot_path_histogram(growth_by_path, 'Geometric growth per event (%)',
        'Final geometric growth rate across simulation paths', 'Compounded growth per scheduled event (%)')


def plot_mean_bankroll_wagered(path_wager_averages):
    """Plot mean event exposure per path, showing the mean and sample SD."""
    return _plot_path_histogram(path_wager_averages, 'Mean bankroll wagered (%)',
        'Average bankroll wagered per event across simulation paths',
        'Average total bankroll wagered per event within a path (%)', mean_std=True, zero_reference=False)


def plot_terminal_return_violin(results, *, scale='symlog', linthresh=100):
    """Plot terminal percentage returns with median and interquartile range.

    Accepts {label: simulate_result}, including saved paths. Each path's own
    starting bankroll is the denominator; all returns, including ruin, remain.
    Returns (figure, ax) without saving or showing it.
    """
    if not results:
        raise ValueError('results must contain at least one simulation result')
    returns = []
    for result in results.values():
        paths = np.asarray(result[2], dtype=float)
        if (paths.ndim != 2 or 0 in paths.shape or not np.isfinite(paths).all()
                or (paths[:, 0] <= 0).any()):
            raise ValueError('Bankroll paths must be nonempty, finite, and start positive')
        returns.append(100 * (paths[:, -1] / paths[:, 0] - 1))

    return _plot_percentage_violins(list(results), returns,
        title='Terminal return distribution', ylabel='Total return (%)',
        scale=scale, linthresh=linthresh, ruin_reference=True)


def _plot_percentage_violins(labels, values_by_label, *, title, ylabel, scale, linthresh,
                            ruin_reference=False):
    fig, ax = plt.subplots(figsize=(max(7, 2 * len(labels)), 5), layout='constrained')
    ax.set_yscale(scale, **({'linthresh': linthresh} if scale == 'symlog' else {}))
    transform = ax.yaxis.get_transform()
    for position, values in enumerate(values_by_label, start=1):
        # Estimate density in display space, then restore real percentage coordinates.
        if np.ptp(values) > 0:
            violin = ax.violinplot(transform.transform(values), positions=[position],
                                   widths=.75, showextrema=False)
            for body in violin['bodies']:
                for path in body.get_paths():
                    path.vertices[:, 1] = transform.inverted().transform(path.vertices[:, 1])
                body.set_facecolor('tab:blue')
                body.set_alpha(.45)
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        ax.vlines(position, q25, q75, color='black', linewidth=5,
                  label='25th-75th percentile' if position == 1 else None)
        ax.scatter(position, median, color='white', edgecolor='black', zorder=3,
                   label='Median' if position == 1 else None)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set(title=title, ylabel=ylabel, xlim=(.5, len(labels) + .5))
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    if ruin_reference:
        ax.axhline(-100, color='gray', linestyle=':', linewidth=1)
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    # Violin artists were initially constructed in transformed space.
    values = np.concatenate(values_by_label)
    ax.update_datalim(np.column_stack([np.ones(len(values)), values]))
    ax.autoscale_view()
    return fig, ax


def plot_max_drawdown_violin(results, *, scale='linear', linthresh=1):
    """Distribution of each path's worst peak-to-trough drawdown, in percent."""
    from .mc_analysis import summarize_simulations
    from matplotlib.ticker import PercentFormatter
    _, details = summarize_simulations(results, return_path_statistics=True)
    values = [details.loc[details.Prices == label, 'Max drawdown (%)'].to_numpy()
              for label in results]
    fig, ax = _plot_percentage_violins(list(results), values,
        title='Worst peak-to-trough drawdown per path', ylabel='Maximum drawdown (%)',
        scale=scale, linthresh=linthresh)
    minimum = details['Max drawdown (%)'].min()
    ax.set_ylim(max(0, minimum * .8), max(1, details['Max drawdown (%)'].max() * 1.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    return fig, ax


def plot_drawdown_count_histogram(results):
    """Count distinct >=50% drawdown episodes; full peak recovery resets a count.

    Includes unrecovered episodes and paths with zero qualifying episodes.
    """
    from .mc_analysis import summarize_simulations
    from matplotlib.ticker import MaxNLocator
    _, details = summarize_simulations(results, return_path_statistics=True)
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             layout='constrained', squeeze=False)
    for ax, label in zip(axes[0], results):
        counts = details.loc[details.Prices == label, 'Drawdown episodes >= 50%'].to_numpy()
        ax.hist(counts, bins=np.arange(-.5, counts.max() + 1.5),
                color='tab:blue', edgecolor='white', rwidth=.85)
        ax.set(title=f'{label}: drawdowns reaching 50% or more',
               xlabel='Distinct drawdown episodes per path', ylabel='Number of paths',
               xlim=(-.5, counts.max() + .5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=.2)
    return fig, axes[0]


def plot_bankroll_targets(reach_by_path, event_count):
    """Plot analyze_bankroll_targets' per-path DataFrame; return (figure, axes).

    Rows are price sources, columns are targets. No files are saved.
    """
    if reach_by_path.empty:
        raise ValueError('reach_by_path must contain target results')
    labels = reach_by_path['Prices'].unique()
    targets = reach_by_path['Target ($)'].unique()
    fig, axes = plt.subplots(len(labels), len(targets),
                             figsize=(5.3 * len(targets), 6 * len(labels)),
                             layout='constrained', sharex=True, squeeze=False)
    for row, label in enumerate(labels):
        for col, target in enumerate(targets):
            data = reach_by_path.loc[reach_by_path['Prices'].eq(label)
                                     & reach_by_path['Target ($)'].eq(target)]
            hit = data['Reached'].to_numpy(bool)
            observed = data.loc[data['Reached'], 'First event count']
            ax = axes[row, col]
            color = '#3579ad' if label == 'close2' else '#df8332'
            ax.hist(observed, bins=np.arange(-.5, event_count + 1.5, 1),
                    color=color, edgecolor='white', linewidth=.3)
            ax.set(title=f'{label} - ${target:,.0f}', xlabel='Events completed at first reach',
                   ylabel='Number of paths', xlim=(-.5, event_count + .5))
            ax.text(.03, .97,
                    f'Reached: {hit.sum()}/{len(hit)} ({100 * hit.mean():.1f}%)\n'
                    f'Not reached: {(~hit).sum()} ({100 * (~hit).mean():.1f}%)',
                    transform=ax.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('First reach of bankroll targets - combined moneyline and parlay paths')
    return fig, axes


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

    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

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
    plt.tight_layout()

    if path is not None:
        plt.savefig(path)
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

    plt.tight_layout()

    if path is not None:
        plt.savefig(path)

    plt.show()


