import streamlit as st

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats

st.title("Profit Page")

def plot_backtest(df_results, init_bankroll):
    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------------------
    # GROUP BANKROLL
    # ----------------------------
    df_group = df_results.groupby('date').first().reset_index()
    dates = df_group['date'].tolist()
    x_positions = np.arange(len(dates))
    bankroll_history = df_group['bankroll_postevent'].values
    group_profit_history = df_group['event_payout_per_fight'].values
    proportion_wins = np.mean(group_profit_history > 0)

    # ----------------------------
    # COMPUTE CUMULATIVE PARLAY
    # ----------------------------
    df_parlay_daily = df_results.groupby('date')['parlay_net'].first().reset_index()
    parlay_vals = df_parlay_daily['parlay_net'].values
    parlay_cumsum = np.cumsum(parlay_vals)
    x_parlay = np.arange(len(df_parlay_daily))

    # ----------------------------
    # PER-FIGHT NET PROFIT
    # ----------------------------
    df_group_profit = df_results.groupby('date')['event_payout_per_fight'].first().reset_index()
    print(df_group_profit)
    per_fight_net = np.cumsum(df_group_profit['event_payout_per_fight'].values)

    # ----------------------------
    # EVENT NET ODDS (new)
    # ----------------------------
    df_event_odds = df_results.groupby('date')['event_net_odds'].first().reset_index()
    event_net_vals = df_event_odds['event_net_odds'].values
    event_odds_cumsum = np.cumsum(event_net_vals)

    # ----------------------------
    # PARLAY NET ODDS (new)
    # ----------------------------
    df_parlay_odds = df_results.groupby('date')['parlay_net_odds'].first().reset_index()
    parlay_net_vals = df_parlay_odds['parlay_net_odds'].values
    parlay_odds_cumsum = np.cumsum(parlay_net_vals)



    # ----------------------------
    # CREATE FIGURE WITH 5 SUBPLOTS
    # ----------------------------
    # Create figure
    fig, axs = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    label_every = 5  # show every 5th date

    # ---------------------------------------------------------
    # SUBPLOTS (same as before)
    # ---------------------------------------------------------
    axs[0].plot(x_positions, bankroll_history, marker='o', label='Bankroll')
    axs[0].axhline(init_bankroll, color='gray', linestyle='--', label='Initial Bankroll')
    axs[0].set_ylabel("Bankroll")
    axs[0].set_title(f"TOTAL Bankroll Over Time | Event Win Rate: {proportion_wins:.2%}")
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
    axs[1].set_title("Cumulative Parlay Net Profit")
    axs[1].set_ylabel("Cumulative Profit")
    axs[1].legend()

    axs[2].plot(per_fight_net, marker='o', label='Money Line Bet Profit')
    axs[2].set_title("Per Fight Bet Profit")
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
    st.pyplot(fig,use_container_width=False)


def kelly_analysis(df_kelly, x_vars, y_var, bins=8):

    # Standardize y_var
    fig, axes = plt.subplots(2, len(x_vars), figsize=(16, 12))

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
    st.pyplot(fig, use_container_width=False)


def calc_net_odds(df_test_results, odds_type):
    # Choose the probabilities and decimal odds according to prediction
    df_test_results['proba_choice'] = np.where(df_test_results['pred_winner']==1,
                                               df_test_results['proba_red'], 
                                               df_test_results['proba_blue'])
    df_test_results['dec_choice'] = np.where(df_test_results['pred_winner']==1,
                                             df_test_results[f'dec_{odds_type}_red'],
                                             df_test_results[f'dec_{odds_type}_blue'])
    # EV
    df_test_results['ev'] = df_test_results['proba_choice'] * (df_test_results['dec_choice']-1) - (1-df_test_results['proba_choice'])
    
    # Keep only positive EV bets
    df_ev = df_test_results[df_test_results['ev'] > 0].copy()

    print(df_ev.shape, df_test_results.shape)
    
    # Calculate adjusted profit per bet
    df_ev['adj_profit'] = np.where(df_ev['pred_winner'] == df_ev['winner'],
                                   df_ev['dec_choice']-1, -1)
    
    # Separate fav vs dog
    df_ev['is_fav'] = np.where(
        ((df_ev['pred_winner']==1) & (df_ev[f'{odds_type}_red'] <= df_ev[f'{odds_type}_blue'])) |
        ((df_ev['pred_winner']==0) & (df_ev[f'{odds_type}_blue'] < df_ev[f'{odds_type}_red'])),
        1, 0
    )
    
    fav_sum = df_ev.loc[df_ev['is_fav']==1, 'adj_profit'].tolist()
    dog_sum = df_ev.loc[df_ev['is_fav']==0, 'adj_profit'].tolist()
    
    # Compute stats
    def summarize(profits):
        n_total = len(profits)
        n_correct = sum(1 for x in profits if x != -1)
        n_incorrect = n_total - n_correct
        pct_correct = n_correct/n_total*100 if n_total>0 else 0
        return n_total, n_correct, n_incorrect, pct_correct

    fav_stats = summarize(fav_sum)
    dog_stats = summarize(dog_sum)
    
    # Sum of decimal odds
    st.markdown(f"""
    <p style="font-size:24px; font-weight:bold;">
    Sum of decimal odds for favorite predictions (adjusted): {sum(fav_sum)}
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-size:24px; font-weight:bold;">
    Sum of decimal odds for underdog predictions (adjusted): {sum(dog_sum)}
    </p>
    """, unsafe_allow_html=True)

    # Favorite bets stats
    st.markdown(f"""
    <p style="font-size:22px; color:blue;">
    Fav bets: {fav_stats[0]} total, {fav_stats[1]} correct, {fav_stats[2]} incorrect, {fav_stats[3]:.2f}% correct
    </p>
    """, unsafe_allow_html=True)

    # Underdog bets stats
    st.markdown(f"""
    <p style="font-size:22px; color:green;">
    Dog bets: {dog_stats[0]} total, {dog_stats[1]} correct, {dog_stats[2]} incorrect, {dog_stats[3]:.2f}% correct
    </p>
    """, unsafe_allow_html=True)


def plot_juice_histogram(df_, odds_type):
    df = df_.copy()

    # Determine if the predicted winner is favorite
    df['is_fav'] = np.where(
        ((df['pred_winner']==1) & (df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'])) |
        ((df['pred_winner']==0) & (df[f'{odds_type}_blue'] < df[f'{odds_type}_red'])),
        1, 0
    )

    # Get juice of the chosen fighter
    df['choice_juice'] = np.where(
        df['pred_winner'] == 1,
        df[f'juice_{odds_type}_red'],
        df[f'juice_{odds_type}_blue']
    )

    juice_fav = df.loc[df['is_fav']==1, 'choice_juice']
    juice_dog = df.loc[df['is_fav']==0, 'choice_juice']

    # Compute stats
    fav_mean, fav_std = juice_fav.mean(), juice_fav.std()
    dog_mean, dog_std = juice_dog.mean(), juice_dog.std()

    # Plot histograms with KDE
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)

    sns.histplot(juice_fav, bins=60, kde=True, color='blue', alpha=0.6, ax=axes[0])
    axes[0].set_title(f'Favorite {odds_type} Juice\nMean={fav_mean:.2f}, Std={fav_std:.2f}')
    axes[0].set_xlabel('Juice')
    axes[0].set_ylabel('Count')
    sns.despine(ax=axes[0])

    sns.histplot(juice_dog, bins=60, kde=True, color='red', alpha=0.6, ax=axes[1])
    axes[1].set_title(f'Underdog {odds_type} Juice\nMean={dog_mean:.2f}, Std={dog_std:.2f}')
    axes[1].set_xlabel('Juice')
    sns.despine(ax=axes[1])

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)


def plot_line_movement(df_, odds_type):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    import numpy as np

    df = df_.copy()

    # Determine if the predicted winner is favorite
    df['is_fav'] = np.where(
        ((df['pred_winner']==1) & (df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'])) |
        ((df['pred_winner']==0) & (df[f'{odds_type}_blue'] < df[f'{odds_type}_red'])),
        1, 0
    )

    # Get juice of the chosen fighter
    df['choice_close_line'] = np.where(
        df['pred_winner'] == 1,
        df[f'{odds_type}_red'],
        df[f'{odds_type}_blue']
    )

    df['choice_open_line'] = np.where(
        df['pred_winner'] == 1,
        df[f'open_red'],
        df[f'open_blue']
    )

    df['choice_line_movement'] = df['choice_close_line'] - df['choice_open_line'] 

    # Compute stats
    mean, std = df['choice_line_movement'].mean(), df['choice_line_movement'].std()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))  # You can adjust width and height here

    # Plot histogram with KDE
    sns.histplot(df['choice_line_movement'].values, bins=60, kde=True, color='blue', alpha=0.6, ax=ax)

    ax.set_title(f'Line Movement {odds_type} \nMean={mean:.2f}, Std={std:.2f}')
    ax.set_xlabel('Line Movement')
    ax.set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)  # Pass the figure object to Streamlit


def plot_event_win_pcts(df_results):
    from scipy.stats import shapiro  # if you need it later

    # ----------------------------------------------------
    # 1. Filter: choice_ev > 0 AND choice_fstar > 0
    # ----------------------------------------------------
    df_filt = df_results[
        (df_results["choice_ev"] > 0) &
        (df_results["choice_fstar"] > 0)
    ].copy()

    df_filt["correct"] = (df_filt["pred_winner"] == df_filt["winner"]).astype(int)

    # ----------------------------------------------------
    # 2. Compute per-event accuracy
    # ----------------------------------------------------
    df_event_acc = (
        df_filt.groupby("date")["correct"]
        .mean()
        .reset_index(name="event_pred_accuracy")
    )

    avg_event_acc = df_event_acc["event_pred_accuracy"].mean()
    p_lose_all = np.mean(df_event_acc["event_pred_accuracy"] == 0)

    # ----------------------------------------------------
    # 3. Create figure and axes
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_event_acc["event_pred_accuracy"], bins=20, edgecolor='black')
    ax.axvline(avg_event_acc, linestyle='--', color='red')

    ax.set_title(
        f"Prediction Accuracy per Event (Filtered)\n"
        f"Avg Accuracy = {avg_event_acc:.2%} | "
        f"P(Lose All Fights) = {p_lose_all:.2%}"
    )
    ax.set_xlabel("Accuracy per Event")
    ax.set_ylabel("Number of Events")
    ax.grid(alpha=0.3)

    # ----------------------------------------------------
    # 4. Display in Streamlit
    # ----------------------------------------------------
    st.pyplot(fig,use_container_width=False)


def plot_event_odds_stats(df_results):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp, sem, t

    # ------------- Compute average net odds per event -------------
    df_event = df_results.groupby("date")["event_net_odds"].mean().reset_index()
    dates = df_event["date"]
    avg_odds = df_event["event_net_odds"].values
    x_positions = np.arange(len(dates))

    # ------------- One-sample t-test: mean > 0 -------------
    t_stat, p_two_sided = ttest_1samp(avg_odds, 0, nan_policy='omit')
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0

    # ------------- Compute standard deviation and 95% CI -------------
    n = len(avg_odds)
    mean_val = np.mean(avg_odds)
    std_val = np.std(avg_odds, ddof=1)
    confidence_level = 0.95
    ci_range = sem(avg_odds, nan_policy='omit') * t.ppf((1 + confidence_level)/2., n-1)
    ci_lower = mean_val - ci_range
    ci_upper = mean_val + ci_range

    # ------------- Create 2 subplots -------------
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # === SUBPLOT 1: average net odds over time ===
    axs[0].plot(x_positions, avg_odds, marker="o", label="Avg Net Odds per Event")
    axs[0].axhline(0, color='black', linestyle='--')

    axs[0].set_xticks(
        x_positions[::5],
        [str(d) for i,d in enumerate(dates) if i % 5 == 0],
        rotation=45
    )

    axs[0].set_title(f"Average Event Net Odds Over Time | Mean={mean_val:.3f}")
    axs[0].set_ylabel("Avg Net Odds")
    axs[0].legend()

    # === SUBPLOT 2: histogram of event net odds ===
    axs[1].hist(avg_odds, bins=20, alpha=0.7, edgecolor='black')
    axs[1].axvline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.3f}")
    axs[1].axvline(0, color='black', linestyle='--')

    axs[1].set_title(
        f"Histogram of Average Net Odds per Event\n"
        f"t-test p-value (H0: mean=0, one-sided) = {p_one_sided:.4f}, "
        f"Std Dev = {std_val:.3f}, "
        f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]"
    )
    axs[1].set_xlabel("Average Net Odds")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()

    plt.tight_layout()
    st.pyplot(fig,use_container_width=False)


def analyze_line_movement_perf(df, odds_type='close', filter_ev=True):
    df = df.copy()

    # -----------------------------
    # 0. Extract probability of chosen side
    # -----------------------------
    df['proba_choice'] = np.where(
        df['pred_winner'] == 1,
        df['proba_red'],
        df['proba_blue']
    )

    # -----------------------------
    # 1. Compute decimal odds
    # -----------------------------
    def american_to_decimal(x):
        if x < 0:
            return 100 / abs(x) + 1
        else:
            return x / 100 + 1

    df['red_decimal']  = df[f'{odds_type}_red'].apply(american_to_decimal)
    df['blue_decimal'] = df[f'{odds_type}_blue'].apply(american_to_decimal)

    df['choice_decimal_odds'] = np.where(
        df['pred_winner'] == 1,
        df['red_decimal'],
        df['blue_decimal']
    )

    # Convert to NET odds
    df['choice_net_odds'] = df['choice_decimal_odds'] - 1.0

    # -----------------------------
    # 2. Expected Value (EV)
    # -----------------------------
    df['EV'] = df['proba_choice'] * df['choice_net_odds'] - (1 - df['proba_choice'])

    # Filter only EV > 0 bets?
    if filter_ev:
        df = df[df['EV'] > 0].reset_index(drop=True)

    # -----------------------------
    # 3. Determine fav/dog at open & close
    # -----------------------------
    df['fav_open'] = np.where(df['open_red'] <= df['open_blue'], 1, 0)
    df['fav_close'] = np.where(df[f'{odds_type}_red'] <= df[f'{odds_type}_blue'], 1, 0)

    df['choice_open_fav'] = np.where(
        ((df['pred_winner']==1) & (df['fav_open']==1)) |
        ((df['pred_winner']==0) & (df['fav_open']==0)),
        1, 0
    )

    df['choice_close_fav'] = np.where(
        ((df['pred_winner']==1) & (df['fav_close']==1)) |
        ((df['pred_winner']==0) & (df['fav_close']==0)),
        1, 0
    )

    # -----------------------------
    # 4. Movement category
    # -----------------------------
    def movement_label(row):
        if row['choice_open_fav']==1 and row['choice_close_fav']==0:
            return "Fav → Dog"
        if row['choice_open_fav']==0 and row['choice_close_fav']==1:
            return "Dog → Fav"
        if row['choice_open_fav']==1 and row['choice_close_fav']==1:
            return "Stayed Fav"
        return "Stayed Dog"

    df['movement_category'] = df.apply(movement_label, axis=1)

    # -----------------------------
    # 5. Profit
    # -----------------------------
    df['correct'] = (df['pred_winner'] == df['winner']).astype(int)
    df['profit'] = np.where(df['correct'] == 1, df['choice_net_odds'], -1)

    # -----------------------------
    # 6. Stats by movement category
    # -----------------------------
    stats = df.groupby('movement_category').agg(
        accuracy=('correct', 'mean'),
        n=('correct', 'count'),
        total_profit=('profit', 'sum'),
        mean_EV=('EV', 'mean')
    ).reset_index()

    # -----------------------------
    # 7. PLOTS
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Accuracy plot
    sns.barplot(data=stats, x='movement_category', y='accuracy', ax=axs[0])
    axs[0].set_title(f"Accuracy by Line Movement Category\nEV filter = {filter_ev}")
    axs[0].set_xlabel("Category")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_ylim(0, 1)

    # Profit plot
    sns.barplot(data=stats, x='movement_category', y='total_profit', ax=axs[1])
    axs[1].set_title("Total Net Profit by Category")
    axs[1].set_xlabel("Category")
    axs[1].set_ylabel("Total Net Profit")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    return stats


def plot_accuracy_by_variable(df, variable, pred_col="correct_pred"):
    grouped = df.groupby(variable)[pred_col].agg(["mean", "count"])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped.index, grouped["mean"], color="skyblue")

    plt.title(f"Prediction Accuracy by {variable}")
    plt.ylabel("Accuracy")
    plt.xlabel(variable)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate with counts AND accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_bins(df_, variable, bins=10, pred_col="correct_pred"):
    df = df_.copy()
    # Create bins using pandas.cut
    df["_bins"] = pd.cut(df[variable], bins=bins)

    # Group by bins
    grouped = df.groupby("_bins")[pred_col].agg(["mean", "count"])

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(grouped.index.astype(str), grouped["mean"], color="skyblue")

    plt.title(f"Prediction Accuracy by Binned {variable}")
    plt.ylabel("Accuracy")
    plt.xlabel(variable)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    # Annotate counts + accuracy %
    for bar, (acc, count) in zip(bars, zip(grouped["mean"], grouped["count"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={count}\n{acc*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    # Clean temporary column
    df.drop(columns=["_bins"], inplace=True)


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# Path to this app
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

import os
import streamlit as st



# Read CSVs
# Read CSVs using relative paths
df_results_open       = pd.read_csv("./streamlit/data/kelly1.csv")
df_results_close1     = pd.read_csv("./streamlit/data/kelly_close1.csv")
df_results_close2     = pd.read_csv("./streamlit/data/kelly_close2.csv")

df_results_test_open  = pd.read_csv("./streamlit/data/test_logit_open.csv")
df_results_test_close = pd.read_csv("./streamlit/data/test_logit_close.csv")

df_parlay_open        = pd.read_csv("./streamlit/data/parlay_open.csv")
df_parlay_close1      = pd.read_csv("./streamlit/data/parlay_close1.csv")
df_parlay_close2      = pd.read_csv("./streamlit/data/parlay_close2.csv")

# Model/history CSV
df_history = pd.read_csv("./streamlit/data/entire_odds_stats_2025-12-04.csv")
df_model   = pd.read_csv("./streamlit/data/entire_odds_stats_2025-12-04.csv")
init_bankroll = 2000
x_vars = ['mu_portfolio', 'sharpe_portfolio', 'portfolio_sigma']
y_var = 'event_net_odds'


# Section header
st.markdown('<h2>Profit Data Results</h2>', unsafe_allow_html=True)

# First paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Three types of odds were scraped from the following website: 
<a href="https://www.bestfightodds.com/" target="_blank">BestFightOdds</a>.
</p>
""", unsafe_allow_html=True)

# Second paragraph with link
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Betting strategy is a risk-adjusted formulation of Kelly Criteria. 
Kelly Formula calculates the optimal percentage of bankroll to bet on a given fight. 
The key property of this formula is that it only produces a value &gt; 0 when the betting edge is positive. 
For the derivation of the basic formula, see this 
<a href="https://en.wikipedia.org/wiki/Kelly_criterion" target="_blank">Wikipedia article</a>.
</p>
""", unsafe_allow_html=True)

# Third paragraph
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Data included in plots below: Choice fighter betting Juice is defined as the difference between fair odds and real odds, or the bookmaker built-in edge. 
I use a devig power algorithm to estimate the fair odds from the real odds (see articles on Google explaining devigging). 
Tests showed using the fair odds in betting strategy greatly increased profit. 
Additionally, probability derived from fair odds was heavily weighted in the AI model (see the AI model page).
</p>
""", unsafe_allow_html=True)

# Fourth paragraph
st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Open Odds: best results were obtained with opening odds. 
These odds are largely unaffected by line movement and market betting behavior.
</p>
""", unsafe_allow_html=True)


plot_backtest(df_results_open, init_bankroll)
kelly_analysis(df_results_open, x_vars, y_var)
plot_event_win_pcts(df_results_open)
plot_event_odds_stats(df_results_open)
calc_net_odds(df_results_test_open, 'open')
plot_juice_histogram(df_results_test_open, 'open')


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Below is the saved data displayed in the plots. The key variable is choice fstar. This is the optimal percentage of bankroll to wager. 
</p>
""", unsafe_allow_html=True)



st.dataframe(
    df_results_open[['fighter_red', 'fighter_blue', 'open_red', 'open_blue',
                          'red_proba', 'blue_proba', 'pred_winner', 'winner',
                          'choice_ev', 'choice_fstar', 'fight_payout']]
)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)

st.dataframe(
    df_parlay_open
)


st.markdown("""
## Close1 Odds Analysis

These odds produced the worst results. See net odds and number of bets placed. 
The bets are filtered by expected value (probability win * net odds) greater than zero.

The data shows market behavior greatly diminishes the edge from AI predictions. 
Notice the dip in starting bankroll at the start; tests revealed that replenishing bankroll back to the initial 
greatly improved profits when the strategy started to succeed. 

Plots displaying juice and line movement are for choice fighters with positive EV.
""")

plot_backtest(df_results_close1, init_bankroll)
kelly_analysis(df_results_close1, x_vars, y_var)
plot_event_win_pcts(df_results_close1)
plot_event_odds_stats(df_results_close1)
calc_net_odds(df_results_test_close, 'close1')
plot_juice_histogram(df_results_test_close, 'close1')
analyze_line_movement_perf(df_results_test_close, 'close1')

st.dataframe(
    df_results_close1[['fighter_red', 'fighter_blue', 'open_red', 'open_blue',
                          'red_proba', 'blue_proba', 'pred_winner', 'winner',
                          'choice_ev', 'choice_fstar', 'fight_payout']]
)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)

st.dataframe(
    df_parlay_close1
)


st.markdown('## Close2 Odds: Better results from close1 but worse results from Open ')
plot_backtest(df_results_close2, init_bankroll)
kelly_analysis(df_results_close2, x_vars, y_var)
plot_event_win_pcts(df_results_close2)
plot_event_odds_stats(df_results_close2)
calc_net_odds(df_results_test_close, 'close2')
plot_juice_histogram(df_results_test_close, 'close2')
analyze_line_movement_perf(df_results_test_close, 'close2')


st.dataframe(
    df_results_close2[['fighter_red', 'fighter_blue', 'open_red', 'open_blue',
                          'red_proba', 'blue_proba', 'pred_winner', 'winner',
                          'choice_ev', 'choice_fstar', 'fight_payout']]
)


st.markdown("""
<p style="font-size:18px; line-height:1.6;">
Results Data for Parlay Strategy
</p>
""", unsafe_allow_html=True)

st.dataframe(
    df_parlay_close2
)