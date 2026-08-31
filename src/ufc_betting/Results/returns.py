import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from ufc_betting.config import settings
from ufc_betting.DataPipeline.dataframes.moneylines import (
    MoneylineDataFrame,
)
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame
from ufc_betting.DataPipeline.utils.github_utils import commit_if_changed


def commit_figure_if_changed(
    file_path: str | Path,
    message: str | None = None,
    branch: str = "main",
) -> bool:
    """Commit and push a saved figure when its file contents changed."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Figure does not exist: {file_path}")

    subprocess.run(["git", "add", str(file_path)], check=True)

    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", str(file_path)],
        check=False,
    )
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)

    subprocess.run(
        ["git", "config", "--global", "user.name", "github-actions[bot]"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--global",
            "user.email",
            "github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )

    commit_message = message or f"Updating {file_path.name}"
    subprocess.run(
        ["git", "commit", "-m", commit_message, "--", str(file_path)],
        check=True,
    )

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    subprocess.run(
        ["git", "push", push_url, f"HEAD:{branch}"],
        check=True,
    )
    return True


def returns_by_date(
    starting_bankroll: float = 500,
) -> pd.DataFrame:
    """
    Calculate bankroll history from settled moneyline and parlay data.

    Each betting type has an independent bankroll. Moneyline returns are
    summed across fights on each date. Each parlay is counted once even
    though its result is repeated across its leg rows.
    """
    df_ml = MoneylineDataFrame(
        pd.read_csv(settings.ml_history_file)
    ).frame

    df_parlay = ParlayDataFrame(
        pd.read_csv(settings.parlay_history_file)
    ).frame

    settled_types = (
        "open",
        "close1_stack",
        "close2_stack",
    )

    all_dates = (
        pd.concat(
            [
                df_ml["date"],
                df_parlay["date"],
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    bankroll_results = pd.DataFrame({
        "date": all_dates,
    })

    for bet_type in settled_types:
        net_stake_column = f"net_stake_{bet_type}"
        net_odds_column = f"net_odds_{bet_type}"

        ml_daily_return = (
            (
                df_ml[net_stake_column].abs()
                * df_ml[net_odds_column]
            )
            .groupby(df_ml["date"])
            .sum(min_count=1)
            .reindex(all_dates)
            .fillna(0.0)
        )

        parlay_daily = (
            df_parlay.groupby("date")[
                [
                    net_stake_column,
                    net_odds_column,
                ]
            ]
            .first()
            .reindex(all_dates)
        )

        parlay_daily_return = (
            parlay_daily[net_stake_column].abs()
            * parlay_daily[net_odds_column]
        ).fillna(0.0)

        combined_daily_return = (
            ml_daily_return
            + parlay_daily_return
        )

        bankroll = float(starting_bankroll)
        profit_history: list[float] = []
        bankroll_history: list[float] = []

        for return_fraction in combined_daily_return:
            profit = bankroll * float(return_fraction)
            bankroll += profit

            profit_history.append(profit)
            bankroll_history.append(bankroll)

        bankroll_results[f"profits_{bet_type}"] = profit_history
        bankroll_results[f"bankroll_{bet_type}"] = bankroll_history

    commit_if_changed(
        bankroll_results,
        settings.bankroll_returns_file,
        "Saving Bankroll Results",
    )

    return bankroll_results


def plot_returns(ml_results, parlay_results, bankroll_results):

    types = ['open', 'close1_stack', 'close2_stack']
    fig, axes = plt.subplots(2, len(types), figsize=(15,6))
    
    for i, type_ in enumerate(types):
        # ML percent returns
        no_draws = ml_results[ml_results[f'pred_winner_{type_}'] < 2]
        all_preds = no_draws.dropna(subset=[f'pred_winner_{type_}'])
        bets_only = all_preds[all_preds[f'net_stake_{type_}'] != 0]

        net_odds = bets_only[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[0, i], kde=True, color='skyblue', bins=50)
        avg_ml = np.mean(net_odds)
        total_ml = np.sum(net_odds)
        axes[0, i].set_title(f"ML {type_} — Avg: {avg_ml:.2f}, Total: {total_ml:.2f}")
        
        # Parlay percent returns
        net_odds = parlay_results[f'net_odds_{type_}']
        sns.histplot(net_odds, ax=axes[1, i], kde=True, color='salmon', bins=50)
        avg_parlay = np.mean(net_odds)
        total_parlay = np.sum(net_odds)
        axes[1, i].set_title(f"Parlay {type_} — Avg: {avg_parlay:.2f}, Total: {total_parlay:.2f}")

    plt.tight_layout()
    path = settings.data_dir / 'plot_pngs' / 'avg_returns_live.png'
    
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    commit_figure_if_changed(path)

    fig, axes = plt.subplots(nrows=3, figsize=(15,6))
    date = bankroll_results['date']
    for i, type_ in enumerate(types): 
        bankroll = bankroll_results[f'bankroll_{type_}']
        axes[i].plot(date, bankroll)
        axes[i].scatter(date, bankroll)
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Bankroll')
        axes[i].set_title(f'Bankroll for Odds Type {type_}, Current Total: {bankroll.iloc[-1]:.2f}')

    plt.tight_layout()

    path = settings.data_dir / 'plot_pngs' / 'bankroll_returns_live.png'
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    commit_figure_if_changed(path)


def accuracy_analysis(ml_results, parlay_results):

    accuracies = {}
    bet_types = {}

    type_to_odds = {
        'open': 'open',
        'close1_stack': 'close1',
        'close2_stack': 'close2',
    }
    parlay_events = parlay_results.groupby('date').first()
    
    for type_, odds_type in type_to_odds.items():

        no_draws = ml_results[ml_results[f'pred_winner_{type_}'] < 2].dropna(subset=[f'{odds_type}_red', f'{odds_type}_blue'])
        all_preds = no_draws.dropna(subset=[f'pred_winner_{type_}'])
        bets_only = all_preds[
            all_preds[f'net_stake_{type_}'].notna()
            & (all_preds[f'net_stake_{type_}'] != 0)
        ]

        accuracy_all = (all_preds[f'pred_winner_{type_}'] == all_preds['winner_bool']).mean()
        accuracies[f'preds_all_{type_}'] = accuracy_all

        accuracy_bets = (bets_only[f'pred_winner_{type_}'] == bets_only['winner_bool']).mean()
        accuracies[f'bets_{type_}'] = accuracy_bets

        parlay_net = parlay_events[f'net_odds_{type_}'].dropna()
        parlay_accuracy = (parlay_net > 0).mean()
        accuracies[f'parlays_{type_}'] = parlay_accuracy 

        avail_vegas = no_draws.copy()
        odds_vegas = avail_vegas[[f'{odds_type}_blue', f'{odds_type}_red']].to_numpy()
        winners = avail_vegas['winner_bool']

        vegas_preds = np.where(
            odds_vegas[:, 0] == odds_vegas[:, 1],
            1,
            np.argmin(odds_vegas, axis=1)
        )
        accuracies[f'vegas_{type_}'] = (vegas_preds == winners).mean()

        choice_odds = np.where(
            bets_only[f'pred_winner_{type_}'] == 1, 
            bets_only[f'{odds_type}_red'], 
            bets_only[f'{odds_type}_blue']
        )
        
        dog_bets = bets_only[choice_odds > 0]
        fav_bets = bets_only[choice_odds < 0]

        total_dog = dog_bets.shape[0]
        win_dog = dog_bets[f'pred_winner_{type_}'] == dog_bets['winner_bool']
        n_win_dog = win_dog.sum()

        total_fav = fav_bets.shape[0]
        win_fav = fav_bets[f'pred_winner_{type_}'] == fav_bets['winner_bool']
        n_win_fav = win_fav.sum()

        bet_types[f'total_fav_{type_}'] = total_fav
        bet_types[f'total_dog_{type_}'] = total_dog
        bet_types[f'n_win_fav_{type_}'] = n_win_fav
        bet_types[f'n_win_dog_{type_}'] = n_win_dog

        win_pct_fav = n_win_fav / total_fav if total_fav != 0 else np.nan
        bet_types[f'accuracy_fav_{type_}'] = win_pct_fav

        win_pct_dog = n_win_dog / total_dog if total_dog != 0 else np.nan
        bet_types[f'accuracy_dog_{type_}'] = win_pct_dog

    df_accuracies = (
        pd.Series(accuracies, name="Accuracies")
        .rename_axis("metric")
        .reset_index()
        .round(4)
    )
    
    df_bet_types = (
        pd.Series(bet_types, name="Bet Type Stats")
        .rename_axis("metric")
        .reset_index()
        .round(4)
    )
    return df_accuracies, df_bet_types
