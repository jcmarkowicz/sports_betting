import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns 

from scipy import stats
from scipy.stats import ttest_1samp, sem, t

def correlate_with_vegas(df_results):

    vegas_acc = df_results.groupby('date')['vegas_acc_choice'].first()
    pct_returns = df_results.groupby('date')['bankroll_pct_change'].first()

    r = np.corrcoef(vegas_acc, pct_returns)[0,1]
    sns.regplot(x=vegas_acc, y=pct_returns)
    plt.title(f'Correlation: {r:.2f}')
    plt.xlabel('vegas acc choice fighter')
    plt.ylabel('pct return')
    plt.show()

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

def pmf_plot(df_kelly):
    bets_df = df_kelly[df_kelly['choice_ev'] > 0]
    best_k = bets_df.groupby('date')['choice_proba'].apply(pmf_num_wins)
    event_returns = bets_df.groupby('date').apply(lambda g: g['bankroll_pct_change'].iloc[0])

    fig, axes = plt.subplots(2, figsize=(10,6))
    sns.histplot(best_k, ax=axes[0])

    r = np.corrcoef(best_k, event_returns)[1,0]
    sns.regplot(x=best_k, y=event_returns)
    axes[1].set_title(f'r={r:.2f}')

    plt.tight_layout()
    plt.show()


def summary_stats(df_kelly):

    bets_df = df_kelly[df_kelly['choice_ev'] > 0]

    # Group by date
    bets_group = bets_df.groupby('date')

    # --- Event accuracy per event ---
    # 1 if pred_winner == winner else 0, then mean per event
    event_acc = bets_group.apply(
        lambda g: (g['pred_winner'] == g['winner']).mean()
    )

    # --- Event total sizing per event ---
    event_sizing = bets_group['choice_fstar'].sum()
    event_returns = bets_group.apply(lambda g: g['bankroll_pct_change'].iloc[0])


    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    sns.histplot(event_acc, ax=axes[0,0], bins=20, kde=False)
    axes[0,0].set_xlabel('Event Accuracy')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Distribution of Event Accuracy')

    r = np.corrcoef(event_acc, event_returns)[0,1]
    sns.regplot(x=event_acc, y=event_returns, ax=axes[0,1])
    axes[0,1].set_xlabel('Event Accuracy')
    axes[0,1].set_ylabel('Bankroll Pct Return')
    axes[0,1].set_title(f'r={r:.2f}')

    sns.histplot(event_sizing, ax=axes[1,0], bins=20, kde=False)
    axes[1,0].set_xlabel('Total Percentage Wagered per Event')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Distribution of Event Sizing')

    r = np.corrcoef(event_acc, event_returns)[1,1]
    sns.regplot(x=event_sizing, y=event_returns, ax=axes[1,1])
    axes[1,1].set_xlabel('Event totla Fstar')
    axes[1,1].set_ylabel('Bankroll Pct Return')
    axes[1,1].set_title(f'r={r:.2f}') 

    plt.tight_layout()
    plt.show()


def event_analysis(df):

    df['date'] = pd.to_datetime(df['date'])

    avg_ev_per = df[df['choice_ev']>0].groupby('date')['choice_ev'].mean()
    bets_per = df[df['choice_ev'] > 0].groupby('date')['choice_ev'].count()

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Histogram 1: avg EV per day
    sns.histplot(avg_ev_per, kde=True, bins=20, ax=axes[0,0])
    axes[0, 0].set_title(f"Avg EV Over Fights per Event (mean={avg_ev_per.mean():.3f})")
    axes[0,0].set_xlabel("Avg Choice EV")
    axes[0,0].set_ylabel("Frequency")

    # Histogram 2: bets per day
    sns.histplot(bets_per, kde=True, bins=20, ax=axes[0,1])
    axes[0,1].set_title(f" Number of Plus EV bets Per Event (mean={bets_per.mean():.3f})")
    axes[0,1].set_xlabel("number bets per event")
    axes[0,1].set_ylabel("Frequency")

    event_net_fstar = df.groupby('date').first()['bankroll_pct_change']
    r = np.corrcoef(avg_ev_per, event_net_fstar)[0, 1]
    sns.regplot(x=avg_ev_per, y=event_net_fstar, ax=axes[1,0])
    axes[1,0].set_title(f'Correlation {r:.3f}')
    axes[1,0].set_xlabel('average event choice ev')
    axes[1,0].set_ylabel('Percent Returns Total')


    r = np.corrcoef(bets_per, event_net_fstar)[0, 1]
    sns.regplot(x=bets_per, y=event_net_fstar, ax=axes[1,1])
    axes[1,1].set_title(f'Correlation {r:.3f}')
    axes[1,1].set_xlabel('number bets per event ')
    axes[1,1].set_ylabel('Percent Returns Total')

    plt.tight_layout()
    plt.show()



def plot_mean_feature_histograms(df, mean_cols, date_col='date', win_mask_col='event_ml_net_odds'):
    """
    Plots histograms of daily mean columns for winners and losers.
    
    df            : DataFrame
    mean_cols     : list of mean columns to plot
    date_col      : date column
    win_mask_col  : column where >0 = winner, <=0 = loser
    """

    # Split winners/losers
    df_winner = df[df[win_mask_col] > 0]
    df_loser  = df[df[win_mask_col] < 0]

    # For each, keep ONE value per date
    df_winner_unique = df_winner.groupby(date_col)[mean_cols].first()
    df_loser_unique  = df_loser.groupby(date_col)[mean_cols].first()

    # Setup subplots
    n = len(mean_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))

    if n == 1:  # handle single subplot case
        axes = [axes]

    for ax, col in zip(axes, mean_cols):
        ax.hist(df_winner_unique[col], alpha=0.6, label="Winner", bins=20)
        ax.hist(df_loser_unique[col], alpha=0.6, label="Loser", bins=20)

        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()


def plot_event_odds_stats(df_results):


    df_results = df_results.copy()
    df_results["date"] = pd.to_datetime(df_results["date"])

    variables = ["event_ml_net_odds", "bankroll_pct_change"]
    titles = ['Money Line Net Odds', 'Parlay+ML Percent Returns']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for col_idx, var in enumerate(variables):

        # --- Aggregate per event ---
        df_event = df_results.groupby("date")[var].mean().reset_index()
        dates = df_event["date"]
        values = df_event[var].values
        x_positions = np.arange(len(dates))

        # --- Stats ---
        t_stat, p_two_sided = ttest_1samp(values, 0, nan_policy='omit')
        p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0

        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        ci_range = sem(values, nan_policy='omit') * t.ppf(0.975, n-1)
        ci_lower = mean_val - ci_range
        ci_upper = mean_val + ci_range

        # Format month-year labels
        month_labels = dates.dt.strftime("%b-%Y")

        # === TOP ROW: Time Series ===
        axs[0, col_idx].plot(x_positions, values, marker="o")
        axs[0, col_idx].axhline(0, linestyle='--')

        axs[0, col_idx].set_xticks(
            x_positions[::5],
            month_labels.iloc[::5],
            rotation=45
        )

        if col_idx == 1:
            sharpe = mean_val / std_val
            axs[0, col_idx].set_title(f"{titles[col_idx]} Over Time | Mean={mean_val:.3f} | Sharpe: {sharpe:.2f}")
        else:
            axs[0, col_idx].set_title(f"{titles[col_idx]} Over Time | Mean={mean_val:.3f}")
        axs[0, col_idx].set_ylabel("Average Value")

        # === BOTTOM ROW: Histogram ===
        axs[1, col_idx].hist(values, bins=20, alpha=0.7, edgecolor='black')
        axs[1, col_idx].axvline(mean_val, linestyle='--', label=f"Mean={mean_val:.3f}")
        axs[1, col_idx].axvline(0, linestyle='--')

        axs[1, col_idx].set_title(
            f"{titles[col_idx]}\n"
            f"p={p_one_sided:.4f}, SD={std_val:.3f}, "
            f"95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]"
        )
        axs[1, col_idx].set_xlabel("Average Value")
        axs[1, col_idx].set_ylabel("Frequency")
        axs[1, col_idx].legend()

    plt.tight_layout()
    plt.show()

    # return {
    #     "mean_avg_net_odds": mean_val,
    #     "std_dev": std_val,
    #     "ci_95": (ci_lower, ci_upper),
    #     "t_stat": t_stat,
    #     "p_value_one_sided": p_one_sided
    # }


def plot_backtest(df_results, init_bankroll):
    import numpy as np
    import matplotlib.pyplot as plt

    # GROUP BANKROLL
    df_group = df_results.groupby('date').first().reset_index()
    dates = df_group['date'].tolist()
    x_positions = np.arange(len(dates))

    bankroll_history = df_group['bankroll_postevent'].values

    proportion_wins_per_bet = (
        df_group.groupby('date')['event_payout_money_line']
        .first()
        .gt(0)
        .mean()
    )

    # proportion of events where bankroll increased
    proportion_wins_total = (
        df_group.groupby('date')['bankroll_postevent']
        .first()
        .diff()
        .dropna()
        .gt(0)
        .mean()
    )

    # proportion of events where parlay won
    proportion_wins_parlay = (
        df_group.groupby('date')['parlay_net']
        .first()
        .gt(0)
        .mean()
    )


    # MONEY LINE NET PROFIT
    df_group_profit = df_results.groupby('date')['event_payout_money_line'].first().reset_index()
    per_fight_net = np.cumsum(df_group_profit['event_payout_money_line'].values)

    # MONEY LINE NET ODDS 
    df_event_odds = df_results.groupby('date')['event_ml_net_odds'].first().reset_index()
    event_net_vals = df_event_odds['event_ml_net_odds'].values
    event_odds_cumsum = np.cumsum(event_net_vals)

    # PARLAY NET ODDS 
    df_parlay_odds = df_results.groupby('date')['parlay_net_odds'].first().reset_index()
    parlay_net_vals = df_parlay_odds['parlay_net_odds'].values
    parlay_odds_cumsum = np.cumsum(parlay_net_vals)

    # COMPUTE CUMULATIVE PARLAY
    df_parlay_daily = df_results.groupby('date')['parlay_net'].first().reset_index()
    parlay_vals = df_parlay_daily['parlay_net'].values
    parlay_cumsum = np.cumsum(parlay_vals)
    x_parlay = np.arange(len(df_parlay_daily))


    # CREATE FIGURE WITH 5 SUBPLOTS
    fig, axs = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    label_every = 5  # show every 5th date

    axs[0].plot(x_positions, bankroll_history, marker='o', label='Bankroll')
    axs[0].axhline(init_bankroll, color='gray', linestyle='--', label='Initial Bankroll')
    axs[0].set_ylabel("Bankroll")
    axs[0].set_title(f"TOTAL Bankroll Over Time | Event Win Rate: {proportion_wins_total:.2%}")
    axs[0].legend(loc='upper left')

    # ⭐ RESTORE MILESTONE BOX ⭐
    milestones = [10_000, 100_000, 1_000_000]
    milestone_texts = []
    for milestone in milestones:
        idx = np.argmax(bankroll_history >= milestone)
        if bankroll_history[idx] >= milestone:
            milestone_texts.append(f"{milestone:,}$ reached at event {idx+1}")
        else:
            milestone_texts.append(f"{milestone:,}$ not reached")

    axs[0].text(
        1.02, 0.5, "\n".join(milestone_texts),
        transform=axs[0].transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )

    axs[1].plot(x_parlay, parlay_cumsum, marker='o', color='orange', label='Cumulative Parlay Net')
    axs[1].set_title(f"Cumulative Parlay Net Profit | Parlay Win Rate: {proportion_wins_parlay:.2%}")
    axs[1].set_ylabel("Cumulative Profit")
    axs[1].legend()

    axs[2].plot(per_fight_net, marker='o', label='Money Line Bet Profit')
    axs[2].set_title(f"Per Fight Bet Profit | Win Rate: {proportion_wins_per_bet:.2%}")
    axs[2].set_ylabel("Cumulative Profit")
    axs[2].legend()

    axs[3].plot(event_odds_cumsum, marker='o', color='green', label='Cumulative Event Net Odds')
    axs[3].set_title("Cumulative Event Net Odds")
    axs[3].set_ylabel("Cumulative Odds")
    axs[3].legend()

    axs[4].plot(parlay_odds_cumsum, marker='o', color='purple', label='Cumulative Parlay Net Odds')
    axs[4].set_title("Cumulative Parlay Net Odds")
    axs[4].set_ylabel("Cumulative Odds")
    axs[4].legend()

    # Correct shared date labeling for ALL subplots
    label_every = 5

    tick_positions = x_positions[::label_every]
    tick_labels = [str(d) for i, d in enumerate(dates) if i % label_every == 0]

    # Apply ticks ONLY to the bottom subplot
    axs[-1].set_xticks(tick_positions)
    axs[-1].set_xticklabels(tick_labels)

    # Rotate and format like the working example
    fig.autofmt_xdate(rotation=45, ha='right')

    # Remove all x tick labels from the upper subplots
    for ax in axs[:-1]:
        ax.label_outer()


def plot_bankroll_distributions(group_stats_, df_parlay):
    group_stats = group_stats_.copy()
    group_stats = group_stats.sort_values('date')
    group_stats['date'] = pd.to_datetime(group_stats['date'])

    numeric_cols = [
        'pct_bankroll_gain_per',
        'pct_bankroll_gain_parlay',
        'pct_bankroll_gain_total'
    ]

    daily = (
        group_stats.groupby('date')[numeric_cols]
        .mean()
        .reset_index()
    )

    daily['cum_per'] = daily['pct_bankroll_gain_per'].cumsum()
    daily['cum_parlay'] = daily['pct_bankroll_gain_parlay'].cumsum()
    daily['cum_total'] = daily['pct_bankroll_gain_total'].cumsum()

    base_cols = ['pct_bankroll_gain_per', 'pct_bankroll_gain_parlay', 'pct_bankroll_gain_total']
    cum_cols  = ['cum_per',              'cum_parlay',              'cum_total']

    titles     = ['Per Bet Gain',        'Parlay Gain',             'Total Gain']
    cum_titles = ['Cumulative Per Bet',  'Cumulative Parlay',       'Cumulative Total']

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # --- top row: histograms ---
    for ax, col, title in zip(axes[0], base_cols, titles):
        sns.histplot(daily[col], kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # --- bottom row: cumulative time series ---
    for ax, col, title in zip(axes[1], cum_cols, cum_titles):
        sns.lineplot(x='date', y=col, data=daily, ax=ax)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # --- format x axis ---
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    filtered = group_stats[group_stats['choice_edge'] > 0].copy()
    filtered['edge_bins'] = pd.qcut(filtered['choice_edge'], q=4)
    filtered['avg_choice_edge_bins'] = pd.qcut(filtered['avg_choice_edge'], q=4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=filtered, x='edge_bins', y='net_odds', estimator='mean', ax=axes[0])
    axes[0].set_title('Per Fight Avg Net Odds by Choice Edge Quantile')
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=filtered, x='avg_choice_edge_bins', y='net_odds', estimator='mean', ax=axes[1])
    axes[1].set_title('Per Fight Avg Net Odds by Avg Choice Edge Quantile')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


    filtered_parlay = df_parlay[df_parlay['parlay_net_odds'] != 0].copy()
    filtered_parlay['parlay_ev_bins'] = pd.qcut(filtered_parlay['parlay_ev'], q=4)
    filtered_parlay['parlay_proba_bins'] = pd.qcut(filtered_parlay['parlay_prob'], q=4)
    filtered_parlay['parlay_avg_edge_bins'] = pd.qcut(filtered_parlay['parlay_avg_edge'], q=4)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # ---- Fight-level edge bins ----
    sns.barplot(data=filtered, x='edge_bins', y='net_odds', estimator='mean', ax=axes[0])
    axes[0].set_title('Net Odds by Choice Edge Quantile')
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=filtered, x='avg_choice_edge_bins', y='net_odds', estimator='mean', ax=axes[1])
    axes[1].set_title('Net Odds by Avg Choice Edge Quantile')
    axes[1].tick_params(axis='x', rotation=45)

    # ---- Parlay-level bins ----
    sns.barplot(data=filtered_parlay, x='parlay_ev_bins', y='parlay_net_odds', estimator='mean', ax=axes[2])
    axes[2].set_title('Parlay Net Odds by EV Quantile')
    axes[2].tick_params(axis='x', rotation=45)

    sns.barplot(data=filtered_parlay, x='parlay_proba_bins', y='parlay_net_odds', estimator='mean', ax=axes[3])
    axes[3].set_title('Parlay Net Odds by Probability Quantile')
    axes[3].tick_params(axis='x', rotation=45)

    sns.barplot(data=filtered_parlay, x='parlay_avg_edge_bins', y='parlay_net_odds', estimator='mean', ax=axes[4])
    axes[4].set_title('Parlay Net Odds by Avg Edge Quantile')
    axes[4].tick_params(axis='x', rotation=45)

    # Remove empty last subplot
    fig.delaxes(axes[5])

    # ---- Format tick labels to 2 decimals ----
    def format_bins(ax):
        labels = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = []
        for lab in labels:
            # lab looks like "(0.12, 0.37]"
            a, b = lab.strip("()[]").split(",")
            new_labels.append(f"({float(a):.2f}, {float(b):.2f}]")
        ax.set_xticklabels(new_labels)

    for ax in axes[:5]:
        format_bins(ax)

    plt.tight_layout()
    plt.show()


def kelly_analysis(df_kelly, x_vars, y_var, bins=8):

    # Standardize y_var
    fig, axes = plt.subplots(2, len(x_vars), figsize=(15, 12))

    # --- Row 1: Regression plots ---
    for i, x in enumerate(x_vars):
        r, p = stats.pearsonr(df_kelly[x], df_kelly[y_var])
        sns.regplot(
            data=df_kelly, x=x, y=y_var, ax=axes[0, i],
            scatter_kws={'s': 30}, line_kws={'color': 'red'}
        )
        axes[0, i].set_title(f"{x.replace('_', ' ').title()} (r={r:.2f}, p={p:.3f})", fontsize=11)
        axes[0, i].set_xlabel(x.replace('_', ' ').title())
        axes[0, i].set_ylabel('Standardized Event Payout')

    # --- Row 2: Bar plots with confidence intervals and one-sided t-tests ---
    for i, x in enumerate(x_vars):
        df_kelly[f'{x}_bin'] = pd.cut(df_kelly[x], bins=bins)
        
        agg_list = []
        for b in df_kelly[f'{x}_bin'].cat.categories:
            vals = df_kelly.loc[df_kelly[f'{x}_bin'] == b, y_var]
            n = len(vals)
            mean = vals.mean() if n > 0 else np.nan
            std = vals.std() if n > 1 else np.nan
            
            if n > 1:
                se = std / np.sqrt(n)
                ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
                if mean >= 0:
                    p_val = stats.ttest_1samp(vals, 0, alternative='greater').pvalue
                else:
                    p_val = stats.ttest_1samp(vals, 0, alternative='less').pvalue
            else:
                ci = (np.nan, np.nan)
                p_val = np.nan
            
            sig = f"n={n}, p={p_val:.3f}" if n > 0 else "n/a"
            agg_list.append({'bin': b, 'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1], 'sig': sig})
        
        agg = pd.DataFrame(agg_list)
        sns.barplot(
            x='bin', y='mean', data=agg, ax=axes[1, i],
            palette='magma', edgecolor='black', linewidth=0.5
        )
        
        axes[1, i].errorbar(
            x=np.arange(len(agg)), 
            y=agg['mean'], 
            yerr=[agg['mean'] - agg['ci_lower'], agg['ci_upper'] - agg['mean']], 
            fmt='none', c='black', capsize=5
        )
        
        for j, row in agg.iterrows():
            y_pos = row['ci_upper'] + 0.04 * np.nanmax(np.abs(agg['mean']))
            axes[1, i].text(
                j, y_pos, row['sig'], ha='center', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
            )
        
        axes[1, i].set_title(f'Avg Standardized Event Payout by {x.replace("_", " ").title()}', fontsize=11)
        axes[1, i].set_xlabel(x.replace('_', ' ').title())
        axes[1, i].set_ylabel('Average Standardized Event Payout')
        axes[1, i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
