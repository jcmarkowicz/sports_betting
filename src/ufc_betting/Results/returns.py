import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np 
import pandas as pd
from sklearn.metrics import brier_score_loss

from ufc_betting.config import settings
from ufc_betting.DataPipeline.dataframes.moneylines import (
    MoneylineDataFrame,
)
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame
from ufc_betting.DataPipeline.utils.github_utils import commit_if_changed, commit_figure_if_changed


SETTLED_TYPES = (
    "open",
    "close1_stack",
    "close2_stack",
)


def _event_return_components(
    ml_results: pd.DataFrame,
    parlay_results: pd.DataFrame,
) -> tuple[
    pd.Series,
    dict[str, pd.Series],
    dict[str, pd.Series],
    dict[str, pd.Series],
]:
    """Return moneyline, parlay, and combined returns by event date."""
    ml_results = ml_results.drop_duplicates(keep="last")
    parlay_results = parlay_results.drop_duplicates(keep="last")

    all_dates = (
        pd.concat(
            [ml_results["date"], parlay_results["date"]],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    ml_returns_by_type: dict[str, pd.Series] = {}
    parlay_returns_by_type: dict[str, pd.Series] = {}
    combined_returns_by_type: dict[str, pd.Series] = {}
    for bet_type in SETTLED_TYPES:
        net_stake_column = f"net_stake_{bet_type}"
        net_odds_column = f"net_odds_{bet_type}"

        ml_daily_return = (
            (
                ml_results[net_stake_column].abs()
                * ml_results[net_odds_column]
            )
            .groupby(ml_results["date"])
            .sum(min_count=1)
            .reindex(all_dates)
            .fillna(0.0)
        )
        parlay_daily = (
            parlay_results.groupby("date")[
                [net_stake_column, net_odds_column]
            ]
            .first()
            .reindex(all_dates)
        )
        parlay_daily_return = (
            parlay_daily[net_stake_column].abs()
            * parlay_daily[net_odds_column]
        ).fillna(0.0)

        ml_returns_by_type[bet_type] = ml_daily_return
        parlay_returns_by_type[bet_type] = parlay_daily_return

        # A bankroll cannot lose more than its full value on one event.
        combined_returns_by_type[bet_type] = (
            ml_daily_return + parlay_daily_return
        ).clip(lower=-1.0)

    return (
        all_dates,
        ml_returns_by_type,
        parlay_returns_by_type,
        combined_returns_by_type,
    )


def _event_return_fractions(
    ml_results: pd.DataFrame,
    parlay_results: pd.DataFrame,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Return deduplicated, loss-capped return fractions by event date."""
    all_dates, _, _, combined_returns = _event_return_components(
        ml_results,
        parlay_results,
    )
    return all_dates, combined_returns



def returns_by_date(
    starting_bankroll: float = 500,
    bankroll_floor: float | None = 100,
    replenishment_amount: float = 500,
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

    if starting_bankroll < 0:
        raise ValueError("starting_bankroll must be nonnegative")
    if bankroll_floor is not None and bankroll_floor < 0:
        raise ValueError("bankroll_floor must be nonnegative or None")
    if replenishment_amount <= 0:
        raise ValueError("replenishment_amount must be positive")

    all_dates, event_returns = _event_return_fractions(
        df_ml,
        df_parlay,
    )

    bankroll_results = pd.DataFrame({
        "date": all_dates,
    })

    for bet_type in SETTLED_TYPES:
        bankroll = float(starting_bankroll)
        replenishment_count = 0
        profit_history: list[float] = []
        bankroll_history: list[float] = []
        replenished_history: list[bool] = []
        replenishment_history: list[float] = []
        replenishment_count_history: list[int] = []

        for return_fraction in event_returns[bet_type]:
            multiplier = max(0.0, 1.0 + float(return_fraction))
            profit = bankroll * (multiplier - 1.0)
            bankroll += profit

            replenished = (
                bankroll_floor is not None
                and bankroll < bankroll_floor
            )
            amount_added = 0.0
            if replenished:
                amount_added = float(replenishment_amount)
                replenishment_count += 1

            profit_history.append(profit)
            bankroll_history.append(bankroll)
            replenished_history.append(replenished)
            replenishment_history.append(amount_added)
            replenishment_count_history.append(replenishment_count)

            if replenished:
                bankroll = amount_added

        event_return = event_returns[bet_type]
        bankroll_results[f"return_fraction_{bet_type}"] = (
            event_return.to_numpy()
        )
        bankroll_results[f"multiplier_{bet_type}"] = (
            1.0 + event_return.to_numpy()
        )
        bankroll_results[f"profitable_event_{bet_type}"] = (
            event_return.gt(0).to_numpy()
        )
        bankroll_results[f"profits_{bet_type}"] = profit_history
        bankroll_results[f"replenished_{bet_type}"] = replenished_history
        bankroll_results[
            f"replenishment_amount_{bet_type}"
        ] = replenishment_history
        bankroll_results[
            f"replenishment_count_{bet_type}"
        ] = replenishment_count_history
        bankroll_results[f"bankroll_{bet_type}"] = bankroll_history

    commit_if_changed(
        bankroll_results,
        settings.bankroll_returns_file,
        "Saving Bankroll Results",
    )

    return bankroll_results


def plot_returns(ml_results, parlay_results, bankroll_results):

    import matplotlib.pyplot as plt
    import seaborn as sns

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
        net_odds = (
            parlay_results
            .groupby('date', sort=False)[f"net_odds_{type_}"]
            .first()
            .dropna()
        )
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
    (
        event_dates,
        ml_event_returns,
        parlay_event_returns,
        event_returns,
    ) = _event_return_components(
        ml_results,
        parlay_results,
    )
    
    for type_, odds_type in type_to_odds.items():

        no_draws = ml_results[ml_results[f'winner_bool'] < 2].dropna(subset=[f'{odds_type}_red', f'{odds_type}_blue'])
        all_preds = no_draws.dropna(subset=[f'pred_winner_{type_}'])
        bets_only = all_preds[
            all_preds[f'net_stake_{type_}'].notna()
            & (all_preds[f'net_stake_{type_}'] != 0)
        ]

        accuracy_all = (all_preds[f'pred_winner_{type_}'] == all_preds['winner_bool']).mean()
        accuracies[f'preds_all_{type_}'] = accuracy_all

        accuracy_bets = (bets_only[f'pred_winner_{type_}'] == bets_only['winner_bool']).mean()
        accuracies[f'bets_{type_}'] = accuracy_bets

        probability_column = f'choice_proba_{type_}'
        scored_predictions = all_preds.dropna(
            subset=[probability_column, 'winner_bool']
        ).copy()
        scored_bets = bets_only.dropna(
            subset=[probability_column, 'winner_bool']
        ).copy()

        for label, scored_rows in (
            ('all', scored_predictions),
            ('bets', scored_bets),
        ):
            choice_probability = pd.to_numeric(
                scored_rows[probability_column],
                errors='coerce',
            )
            valid_probability = choice_probability.between(0, 1)
            scored_rows = scored_rows.loc[valid_probability]
            choice_probability = choice_probability.loc[valid_probability]

            red_probability = choice_probability.where(
                scored_rows[f'pred_winner_{type_}'].eq(1),
                1.0 - choice_probability,
            )
            brier_score = (
                brier_score_loss(
                    scored_rows['winner_bool'],
                    red_probability,
                )
                if not scored_rows.empty
                else np.nan
            )
            accuracies[f'brier_{label}_{type_}'] = brier_score

        parlay_net = parlay_events[f'net_odds_{type_}'].dropna()
        parlay_accuracy = (parlay_net > 0).mean()
        accuracies[f'parlays_{type_}'] = parlay_accuracy 

        profitable_events = event_returns[type_].gt(0)
        accuracies[f'profitable_events_{type_}'] = int(
            profitable_events.sum()
        )
        accuracies[f'total_events_{type_}'] = len(event_dates)
        accuracies[f'profitable_event_pct_{type_}'] = (
            profitable_events.mean()
        )
        accuracies[f'total_event_profit_pct_{type_}'] = (
            profitable_events.mean()
        )

        profitable_ml_events = ml_event_returns[type_].gt(0)
        accuracies[f'profitable_ml_events_{type_}'] = int(
            profitable_ml_events.sum()
        )
        accuracies[f'ml_event_profit_pct_{type_}'] = (
            profitable_ml_events.mean()
        )

        profitable_parlay_events = parlay_event_returns[type_].gt(0)
        accuracies[f'profitable_parlay_events_{type_}'] = int(
            profitable_parlay_events.sum()
        )
        accuracies[f'parlay_event_profit_pct_{type_}'] = (
            profitable_parlay_events.mean()
        )

        avail_vegas = no_draws.copy()
        odds_vegas = avail_vegas[[f'{odds_type}_blue', f'{odds_type}_red']].to_numpy()
        winners = avail_vegas['winner_bool']

        vegas_preds = np.where(
            odds_vegas[:, 0] == odds_vegas[:, 1],
            1,
            np.argmin(odds_vegas, axis=1)
        )
        accuracies[f'vegas_{type_}'] = (vegas_preds == winners).mean()

        red_american = pd.to_numeric(
            avail_vegas[f'{odds_type}_red'],
            errors='coerce',
        )
        blue_american = pd.to_numeric(
            avail_vegas[f'{odds_type}_blue'],
            errors='coerce',
        )
        valid_vegas = (
            red_american.notna()
            & blue_american.notna()
            & red_american.ne(0)
            & blue_american.ne(0)
            & avail_vegas['winner_bool'].notna()
        )
        red_american = red_american.loc[valid_vegas]
        blue_american = blue_american.loc[valid_vegas]
        vegas_winners = avail_vegas.loc[valid_vegas, 'winner_bool']

        red_decimal = (1 + red_american / 100).where(
            red_american > 0,
            1 + 100 / red_american.abs(),
        )
        blue_decimal = (1 + blue_american / 100).where(
            blue_american > 0,
            1 + 100 / blue_american.abs(),
        )
        red_implied = 1 / red_decimal
        blue_implied = 1 / blue_decimal
        vegas_red_probability = red_implied / (
            red_implied + blue_implied
        )
        accuracies[f'brier_vegas_{type_}'] = (
            brier_score_loss(vegas_winners, vegas_red_probability)
            if not vegas_winners.empty
            else np.nan
        )

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
