import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns 

from scipy import stats
from scipy.stats import ttest_1samp, sem, t

def correlate_with_vegas(df_results):

    vegas_acc = df_results.dropna().groupby('date')['vegas_acc_choice'].first()
    pct_returns = df_results.dropna().groupby('date')['bankroll_pct_change'].first()

    r = np.corrcoef(vegas_acc, pct_returns)[0,1]
    sns.regplot(x=vegas_acc, y=pct_returns)
    plt.title(f'Vegas Accuracy for Choice Fighters vs Percent Returns \n Correlation: {r:.2f}')
    plt.xlabel('vegas accuracy choice fighter')
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
    axes[0].set_title('PMF Binomial Poisson')

    r = np.corrcoef(best_k, event_returns)[1,0]
    sns.regplot(x=best_k, y=event_returns)
    axes[1].set_title(f'Number Most Likely Wins vs Percent Returns \n r={r:.2f}')

    plt.tight_layout()
    plt.show()


def plot_event_accuracy_distributions(moneylines, parlays=None, *, odds_type=None, bins=10):
    """Histogram event accuracy and percentage of total bankroll wagered.

    Vegas picks the shorter decimal price. Tied prices have no favorite and
    are excluded from Vegas accuracy; void/nonbinary outcomes are excluded
    from every accuracy. Positive moneyline stakes and, when supplied, positive
    parlay stakes identify wagered fights. Each fight counts once, not once per
    ticket leg. Post-ruin rows are excluded from wagers when balances exist.
    Events with no eligible observations have NaN accuracy, not zero accuracy.

    odds_type selects dec_<market>_red/blue when available. Ordinary backtest
    outputs instead expose execution prices as real_odds_red/blue. If no market
    is specified and multiple recorded markets exist, an explicit choice is
    required. 'All Vegas' covers all fights present in moneylines.
    Returns (figure, axes, event_accuracy), with accuracies as fractions and
    eligible fight counts and bankroll_wagered_pct (0-100). Inputs are not modified.
    """
    keys = ['date', 'fighter_red', 'fighter_blue']
    required = set(keys + ['winner', 'pred_winner', 'choice_fstar'])
    missing = required - set(moneylines.columns)
    if missing:
        raise ValueError(f'Missing accuracy columns: {sorted(missing)}')
    if moneylines.empty:
        raise ValueError('No fights to plot')
    frame = moneylines.copy().reset_index(drop=True)
    frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
    if frame[keys].isna().any().any():
        raise ValueError('Fight dates and names must be present')
    if odds_type is not None and odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('odds_type must be open, close1, or close2')
    if odds_type is not None and {f'dec_{odds_type}_red', f'dec_{odds_type}_blue'} <= set(frame):
        red, blue = f'dec_{odds_type}_red', f'dec_{odds_type}_blue'
    elif {'real_odds_red', 'real_odds_blue'} <= set(frame):
        red, blue = 'real_odds_red', 'real_odds_blue'
    else:
        markets = [m for m in ('open', 'close1', 'close2')
                   if {f'dec_{m}_red', f'dec_{m}_blue'} <= set(frame)]
        if odds_type is not None or len(markets) != 1:
            raise ValueError('Specify odds_type matching available decimal price columns')
        red, blue = f'dec_{markets[0]}_red', f'dec_{markets[0]}_blue'
    prices = frame[[red, blue]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 1).any():
        raise ValueError('Favorite comparisons require finite decimal odds above 1')
    stakes = frame.choice_fstar.to_numpy(float)
    if not np.isfinite(stakes).all() or (stakes < 0).any():
        raise ValueError('Moneyline stakes must be finite and nonnegative')
    active = np.ones(len(frame), dtype=bool)
    if {'bankroll_postevent', 'event_payout_money_line', 'parlay_net'} <= set(frame):
        before = frame.bankroll_postevent - frame.event_payout_money_line - frame.parlay_net
        if not np.isfinite(before).all():
            raise ValueError('Starting bankrolls must be finite')
        active = before.to_numpy() > 0
    frame['_active'] = active
    frame['_wagered'] = (stakes > 0) & active
    frame['_pick'] = frame.pred_winner

    if parlays is not None and not parlays.empty:
        if not {'date', 'fstar_parlay'} <= set(parlays):
            raise ValueError('Parlays require date and fstar_parlay')
        tickets = parlays.copy()
        tickets['date'] = pd.to_datetime(tickets.date, errors='raise').dt.normalize()
        ticket_stakes = tickets.fstar_parlay.to_numpy(float)
        if not np.isfinite(ticket_stakes).all() or (ticket_stakes < 0).any():
            raise ValueError('Parlay stakes must be finite and nonnegative')
        for _, leg in tickets.loc[ticket_stakes > 0].iterrows():
            match = frame.date.eq(leg.date)
            if {'fighter_red', 'fighter_blue'} <= set(tickets):
                match &= frame.fighter_red.eq(leg.fighter_red) & frame.fighter_blue.eq(leg.fighter_blue)
            elif 'fight_index' in tickets:
                idx = leg.fight_index
                if pd.isna(idx) or int(idx) != idx or not 0 <= int(idx) < len(frame):
                    raise ValueError('Invalid parlay fight_index; pass unfiltered simulation outputs')
                match &= frame.index == int(idx)
            elif 'choice_fighter_name' in tickets:
                match &= frame.fighter_red.eq(leg.choice_fighter_name) | frame.fighter_blue.eq(leg.choice_fighter_name)
            else:
                raise ValueError('Parlay legs require fighter names, fight_index, or choice_fighter_name')
            ids = frame.index[match]
            if not len(ids) or len(frame.loc[ids, keys].drop_duplicates()) != 1:
                raise ValueError('Parlay leg does not uniquely match a fight; pass matching simulation outputs')
            side = frame.loc[ids, 'pred_winner'].to_numpy()
            if 'choice_fighter_name' in tickets:
                names = frame.loc[ids]
                if not ((names.fighter_red == leg.choice_fighter_name) | (names.fighter_blue == leg.choice_fighter_name)).all():
                    raise ValueError('Parlay fighter does not match its fight')
                side = names.fighter_red.eq(leg.choice_fighter_name).astype(int).to_numpy()
            elif 'pred_winner' in tickets:
                side = np.full(len(ids), leg.pred_winner)
            if ((frame.loc[ids, '_wagered'].to_numpy()) & (frame.loc[ids, '_pick'].to_numpy() != side)).any():
                raise ValueError('Opposing wagered picks on one fight cannot count as one pick')
            frame.loc[ids, '_pick'] = side
            frame.loc[ids, '_wagered'] = frame.loc[ids, '_wagered'].to_numpy() | active[ids]

    # Collapse repeated records while rejecting contradictory fight data.
    consistent = ['winner', '_pick', red, blue]
    if (frame.groupby(keys)[consistent].nunique(dropna=False) > 1).any().any():
        raise ValueError('Conflicting records for the same fight')
    frame['_wagered'] = frame.groupby(keys)['_wagered'].transform('any')
    frame = frame.drop_duplicates(keys)
    settled = frame.winner.isin([0, 1])
    if not frame.loc[settled & frame._wagered, '_pick'].isin([0, 1]).all():
        raise ValueError('Wagered picks must be binary: blue=0, red=1')
    favorite = (frame[red] < frame[blue]).astype(int)
    has_favorite = frame[red] != frame[blue]
    series = {
        'your_picks': (frame._pick == frame.winner, settled & frame._wagered),
        'vegas_all': (favorite == frame.winner, settled & has_favorite),
        'vegas_wagered': (favorite == frame.winner, settled & has_favorite & frame._wagered),
    }
    events = pd.DataFrame(index=pd.Index(sorted(frame.date.unique()), name='date'))
    for name, (correct, eligible) in series.items():
        grouped = correct.loc[eligible].groupby(frame.loc[eligible, 'date'])
        events[name] = grouped.mean().reindex(events.index)
        events[name + '_n'] = grouped.size().reindex(events.index, fill_value=0).astype(int)
    # Simulator event totals already count each parlay ticket once, whereas
    # the parlay frame commonly repeats its stake on every leg.
    if 'event_total_fstar' in frame:
        if (frame.groupby('date').event_total_fstar.nunique(dropna=False) > 1).any():
            raise ValueError('Conflicting total exposure within an event')
        exposure = frame.groupby('date').event_total_fstar.first()
    else:
        exposure = frame.groupby('date').choice_fstar.sum()
        if parlays is not None and not parlays.empty:
            ticket_keys = ['date'] + (['parlay_strategy'] if 'parlay_strategy' in tickets else [])
            if (tickets.groupby(ticket_keys).fstar_parlay.nunique(dropna=False) > 1).any():
                raise ValueError('Ambiguous parlay stakes; supply event_total_fstar')
            ticket_exposure = tickets.drop_duplicates(ticket_keys).groupby('date').fstar_parlay.sum()
            exposure = exposure.add(ticket_exposure, fill_value=0)
    exposure = exposure.reindex(events.index, fill_value=0).astype(float)
    if not np.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError('Event exposure must be finite and nonnegative')
    exposure = exposure.where(frame.groupby('date')._active.any(), 0.)
    events['bankroll_wagered_pct'] = 100 * exposure
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('bins must be a positive integer')
    fig, grid = plt.subplots(2, 2, figsize=(13, 9))
    axes = grid.ravel()
    labels = ('Your wagered picks', 'Vegas favorites: all fights', 'Vegas favorites: wagered fights')
    for ax, name, label in zip(axes, series, labels):
        values = events[name].dropna() * 100
        if len(values):
            ax.hist(values, bins=np.linspace(0, 100, bins + 1), edgecolor='black', alpha=.75)
            ax.axvline(values.mean(), color='tab:orange', linestyle='--',
                       label=f'Mean event accuracy: {values.mean():.1f}%')
            ax.legend()
        else:
            ax.text(.5, .5, 'No eligible events', transform=ax.transAxes, ha='center')
        ax.set(title=f'{label}\n{len(values)} events', xlabel='Event accuracy (%)', xlim=(0, 100))
        ax.grid(axis='y', alpha=.25)
    values = events['bankroll_wagered_pct']
    axes[3].hist(values, bins=np.linspace(0, max(100., values.max()), bins + 1),
                 edgecolor='black', alpha=.75, color='tab:green')
    axes[3].axvline(values.mean(), color='tab:orange', linestyle='--',
                    label=f'Mean wagered: {values.mean():.1f}%')
    axes[3].set(title=f'Total bankroll wagered: singles + parlays\n{len(values)} events',
                xlabel='Starting total bankroll wagered (%)', xlim=(0, max(100., values.max())))
    axes[3].legend()
    axes[3].grid(axis='y', alpha=.25)
    for ax in axes:
        ax.set_ylabel('Number of events')
    fig.tight_layout()
    plt.show()
    return fig, axes, events


def summary_stats(df_kelly):

    bets_df = df_kelly[df_kelly['choice_ev'] > 0]

    # Group by date
    bets_group = bets_df.groupby('date')

    # --- Event accuracy per event ---
    # 1 if pred_winner == winner else 0, then mean per event
    scored = bets_df.loc[bets_df.winner.isin([0, 1])]
    event_acc = scored.pred_winner.eq(scored.winner).groupby(scored.date).mean()

    # --- Event total sizing per event ---
    event_sizing = bets_group['choice_fstar'].sum()
    event_returns = bets_group.apply(lambda g: g['bankroll_pct_change'].iloc[0])


    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    sns.histplot(event_acc, ax=axes[0,0], bins=20, kde=False)
    axes[0,0].set_xlabel('Event Accuracy')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title(f'Event Choice Fighter Accuracy: Total Avg={np.mean(event_acc):.2f}')

    scored_returns = event_returns.reindex(event_acc.index)
    r = event_acc.corr(scored_returns)
    sns.regplot(x=event_acc, y=scored_returns, ax=axes[0,1])
    axes[0,1].set_xlabel('Event Accuracy')
    axes[0,1].set_ylabel('Bankroll Pct Return')
    axes[0,1].set_title(f'AI Pred Fighter Accuracy vs Bankroll Pct Return \n r={r:.2f}')

    sns.histplot(event_sizing, ax=axes[1,0], bins=20, kde=False)
    axes[1,0].set_xlabel('Total Percentage Wagered per Event')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Distribution of Event Sizing')

    r = event_sizing.corr(event_returns)
    sns.regplot(x=event_sizing, y=event_returns, ax=axes[1,1])
    axes[1,1].set_xlabel('Event total Fstar')
    axes[1,1].set_ylabel('Bankroll Pct Return')
    axes[1,1].set_title(f'Pct Total Bankroll Wagered vs Bankroll Pct Return \n r={r:.2f}') 

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


def plot_event_odds_stats(df_results, path=None):


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

        if col_idx == 1: 
            mean_win = values[values>0].mean()
            mean_loss = values[values<0].mean()
            axs[1, col_idx].set_title(
                f"{titles[col_idx]}\n"
                f"p={p_one_sided:.4f}, SD={std_val:.3f}, "
                f"95% CI=[{ci_lower:.3f}, {ci_upper:.3f}] \n"
                f"Mean Win: {mean_win:.2%}, Mean Loss: {mean_loss:.2%}"
            )

        else: 
            axs[1, col_idx].set_title(
                f"{titles[col_idx]}\n"
                f"p={p_one_sided:.4f}, SD={std_val:.3f}, "
                f"95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]"
            )

        axs[1, col_idx].set_xlabel("Average Value")
        axs[1, col_idx].set_ylabel("Frequency")
        axs[1, col_idx].legend()

    plt.tight_layout()

    if path is not None: 
        fig.savefig(path,
            dpi=300,
            bbox_inches="tight")

    plt.show()

    # return {
    #     "mean_avg_net_odds": mean_val,
    #     "std_dev": std_val,
    #     "ci_95": (ci_lower, ci_upper),
    #     "t_stat": t_stat,
    #     "p_value_one_sided": p_one_sided
    # }


def plot_backtest(df_results, init_bankroll, path=None):
    # Older moneyline-only backtests may omit the parlay columns entirely.
    df_results = df_results.copy()
    for column in ('parlay_net', 'parlay_net_odds'):
        if column not in df_results:
            df_results[column] = 0.0
    import numpy as np
    import matplotlib.pyplot as plt

    # GROUP BANKROLL
    df_group = df_results.groupby('date').first().reset_index()
    dates = df_group['date'].tolist()
    x_positions = np.arange(len(dates))

    bankroll_history = df_group['bankroll_postevent'].values

    proportion_profitable_ml_events = (
        df_group['event_payout_money_line'].gt(0).mean()
    )

    # Use the event's actual return instead of differences between recorded
    # bankroll balances, which are distorted by bankroll replenishments.
    proportion_profitable_total_events = (
        df_group['bankroll_pct_change'].gt(0).mean()
    )

    proportion_profitable_parlay_events = (
        df_group['parlay_net'].gt(0).mean()
    )

    # MONEY LINE NET PROFIT
    df_group_profit = df_results.groupby('date')['event_payout_money_line'].first().reset_index()
    per_fight_net = np.cumsum(df_group_profit['event_payout_money_line'].values)

    # COMPUTE CUMULATIVE PARLAY
    df_parlay_daily = df_results.groupby('date')['parlay_net'].first().reset_index()
    parlay_vals = df_parlay_daily['parlay_net'].values
    parlay_cumsum = np.cumsum(parlay_vals)
    x_parlay = np.arange(len(df_parlay_daily))


    # CREATE FIGURE WITH 3 SUBPLOTS
    fig, axs = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    label_every = 5  # show every 5th date

    axs[0].plot(x_positions, bankroll_history, marker='o', label='Bankroll')
    axs[0].axhline(init_bankroll, color='gray', linestyle='--', label='Initial Bankroll')
    axs[0].set_ylabel("Bankroll")
    axs[0].set_title(f"TOTAL Bankroll Over Time | Profitable Events: {proportion_profitable_total_events:.2%} | Final Bankroll: {bankroll_history[-1]:.2f}")
    axs[0].legend(loc='upper left')

    # RESTORE MILESTONE BOX 
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

    axs[1].plot(
        x_parlay,
        parlay_cumsum,
        marker='o',
        color='orange',
        label='Cumulative Combined Parlay Profit',
    )
    axs[1].set_title(
        "Combined Parlay Event Profit | "
        f"Profitable Events: {proportion_profitable_parlay_events:.2%}"
    )
    axs[1].set_ylabel("Cumulative Profit")
    axs[1].legend()

    axs[2].plot(per_fight_net, marker='o', label='Money Line Bet Profit')
    axs[2].set_title(f"Moneyline Event Profit | Profitable Events: {proportion_profitable_ml_events:.2%}")
    axs[2].set_ylabel("Cumulative Profit")
    axs[2].legend()

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

    if path is not None: 
        fig.savefig(path,
            dpi=300,
            bbox_inches="tight")

    plt.show()
    return fig, axs


def plot_bankroll_distributions(group_stats_, df_parlay=None, *, bins=30,
                                include_diagnostics=False):
    """Plot event bankroll-return distributions from simulate_and_plot outputs.

    Event totals repeat on fight rows, so retain one row per date. Returns are
    normalized from fractional or percent input to percentages on the plots. The parlay contribution
    is already in group_stats_; never sum the repeated ticket legs in df_parlay.
    Moneyline/parlay bottom panels show cumulative actual dollar profit divided
    by initial bankroll. Combined overlays compounded return and geometric growth
    per event. Component contributions add to combined return when there are no
    deposits/withdrawals; combined chains event returns to exclude cash flows.
    No-bet events count. Initial bankroll is inferred from the first event.
    Legacy edge/odds diagnostics are optional and require df_parlay.
    Returns the distribution figure and its axes.
    """
    required = {'date', 'bankroll_pct_change', 'bankroll_postevent',
                'event_payout_money_line', 'parlay_net'}
    missing = required - set(group_stats_.columns)
    if missing:
        raise ValueError(f'Missing event-return columns: {sorted(missing)}')
    if group_stats_.empty:
        raise ValueError('No backtest events to plot')
    group_stats = group_stats_.copy()
    group_stats['date'] = pd.to_datetime(group_stats['date'], errors='raise')
    if group_stats['date'].isna().any():
        raise ValueError('Event dates must not be missing')
    group_stats = group_stats.sort_values('date')
    daily = group_stats.drop_duplicates('date').copy()
    values = daily[list(required - {'date'})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Event returns and bankrolls must be finite')
    # Reconstruct the actual starting balance, including any replenishments.
    before = (daily['bankroll_postevent'] - daily['event_payout_money_line']
              - daily['parlay_net']).to_numpy(float)
    ml = daily['event_payout_money_line'].to_numpy(float)
    pl = daily['parlay_net'].to_numpy(float)
    if np.any((before <= 0) & ((ml != 0) | (pl != 0))):
        raise ValueError('Nonpositive starting bankroll with nonzero event profit')
    if before[0] <= 0:
        raise ValueError('Initial bankroll must be positive')
    daily['moneyline_return'] = np.divide(ml, before, out=np.zeros_like(ml), where=before > 0)
    daily['parlay_return'] = np.divide(pl, before, out=np.zeros_like(pl), where=before > 0)
    daily['total_return'] = daily['bankroll_pct_change']
    combined = daily.moneyline_return + daily.parlay_return
    if not np.allclose(combined, daily.total_return):
        if np.allclose(combined * 100, daily.total_return):
            daily['total_return'] = daily.total_return / 100
        else:
            raise ValueError('Moneyline and parlay returns do not match bankroll_pct_change')

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for col, title, top, bottom in zip(
            ('moneyline_return', 'parlay_return', 'total_return'),
            ('Moneylines', 'Parlays', 'Combined'), axes[0], axes[1]):
        percentages = daily[col] * 100
        sns.histplot(percentages, bins=bins, kde=percentages.nunique() > 1,
                     ax=top, color='skyblue', edgecolor='black')
        top.axvline(0, color='black', linestyle='--', linewidth=1)
        top.set(title=f'{title}: event return distribution',
                xlabel='Event bankroll return (%)', ylabel='Number of events')
        if col != 'total_return':
            profits = ml if col == 'moneyline_return' else pl
            contribution = 100 * np.cumsum(profits) / before[0]
            bottom.plot(daily['date'], contribution, color='tab:blue',
                        label='Cumulative actual profit / initial bankroll')
            bottom.set(title=f'{title}: actual profit contribution',
                       xlabel='Event date', ylabel='Contribution to initial bankroll (%)')
            bottom.legend(loc='best')
        else:
            returns = daily[col].to_numpy(float)
            wealth = np.cumprod(1 + returns)
            compounded = 100 * (wealth - 1)
            # Geometric growth is undefined after losses beyond the bankroll.
            valid = np.logical_and.accumulate(1 + returns >= 0)
            geometric = np.full(len(returns), np.nan)
            geometric[valid] = 100 * (np.power(wealth[valid], 1 / np.arange(1, len(returns) + 1)[valid]) - 1)
            cumulative_line, = bottom.plot(daily['date'], compounded, color='tab:blue',
                                           label='Cumulative compounded return')
            growth_axis = bottom.twinx()
            growth_line, = growth_axis.plot(daily['date'], geometric, color='tab:orange',
                                           linestyle='--', label='Geometric growth per event')
            bottom.set(title='Combined: compounded growth',
                       xlabel='Event date', ylabel='Cumulative compounded return (%)')
            growth_axis.set_ylabel('Geometric average growth per event (%)', color='tab:orange')
            bottom.tick_params(axis='y', labelcolor='tab:blue')
            growth_axis.tick_params(axis='y', labelcolor='tab:orange')
            bottom.legend(handles=[cumulative_line, growth_line], loc='best')
        bottom.axhline(0, color='gray', linestyle=':', linewidth=1)
        bottom.xaxis.set_major_locator(mdates.AutoDateLocator())
        bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        bottom.tick_params(axis='x', rotation=45)
        top.grid(True, alpha=.3)
        bottom.grid(True, alpha=.3)
    fig.tight_layout()
    plt.show()
    if not include_diagnostics:
        return fig, axes
    distribution_figure, distribution_axes = fig, axes
    if df_parlay is None:
        raise ValueError('df_parlay is required for include_diagnostics=True')

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


    return distribution_figure, distribution_axes


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


def parlay_analysis(df_parlay, path=None):
    if df_parlay.empty:
        return
    df_parlay['ev_bin'] = pd.cut(df_parlay['parlay_ev'], bins=10)

    # Create subplot
    fig, axes = plt.subplots(3, 1, figsize=(12,10))

    # ---- 1. Barplot of average parlay_net_odds per bin with CI ----
    sns.barplot(
        data=df_parlay,
        x='ev_bin',
        y='parlay_net_odds',
        estimator='mean',
        errorbar=('ci', 95),
        ax=axes[0]
    )
    axes[0].set_title("Average Net Odds per EV Bin")
    axes[0].set_xlabel("EV Bin")
    axes[0].set_ylabel("Average Net Odds")
    axes[0].tick_params(axis='x', rotation=45)

    # ---- 2. Barplot of counts per EV bin ----
    ax = axes[1]

    sns.countplot(
        data=df_parlay,
        x='ev_bin',
        ax=ax
    )

    # scale bar heights
    for p in ax.patches:
        p.set_height(p.get_height() / 2)

    # rescale y-axis to match
    ax.set_ylim(0, max(p.get_height() for p in ax.patches) * 1.1)
    ax.set_title("Count of Samples per EV Bin")
    ax.set_xlabel("EV Bin")
    ax.set_ylabel("Count (scaled by 1/2)")
    ax.tick_params(axis='x', rotation=45)

    # A zero net return is a void ticket, not a win or loss.
    net = df_parlay['parlay_net_odds']
    scored = net.gt(0).astype(float).where(net.notna() & net.ne(0))
    accuracy_per_bin = scored.groupby(df_parlay['ev_bin']).mean().reset_index()
    accuracy_per_bin.rename(columns={'parlay_net_odds': 'accuracy'}, inplace=True)

    sns.barplot(
        data=accuracy_per_bin,
        x='ev_bin',
        y='accuracy',
        ax=axes[2]
    )
    axes[2].set_title("Accuracy per EV Bin")
    axes[2].set_xlabel("EV Bin")
    axes[2].set_ylabel("Accuracy")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if path is not None: 
        fig.savefig(path,
            dpi=300,
            bbox_inches="tight")


    plt.show()



def per_bet_analysis(df_kelly, path=None):
    import math
    choice_kelly = df_kelly[df_kelly['choice_ev'] > 0]

    features = [
        'choice_ev',
        'choice_edge',
        'choice_bet_sigma',
        'choice_bet_sharpe',
        'choice_proba',
        'choice_juice'
    ]

    y = choice_kelly['fstar_net']

    n = len(features)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        x = choice_kelly[col]
        sns.regplot(x=x, y=y, ax=axes[i])
        axes[i].set_title(col)

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if path is not None: 
        fig.savefig(path,
            dpi=300,
            bbox_inches="tight")

    plt.show()
