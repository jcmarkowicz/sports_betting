
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
    "close1",
    "close2",
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


def prepare_return_plot_data(ml_results, parlay_results, bankroll_results, odds_type):
    """Return event, moneyline and parlay frames for backtest distribution plots.

    Reconstruct pre-event balances to exclude replenishments from profit.
    When the full-loss cap applies, scale both contributions proportionally.
    Invalid market prices are excluded only from the accuracy inputs.
    """
    if odds_type not in SETTLED_TYPES:
        raise ValueError(f"odds_type must be one of {SETTLED_TYPES}")
    ml, pl, balances = (frame.copy() for frame in
                        (ml_results, parlay_results, bankroll_results))
    for frame in (ml, pl, balances):
        frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    dates, ml_returns, pl_returns, total_returns = _event_return_components(ml, pl)
    balances = balances.set_index('date').reindex(dates)
    before = balances[f'bankroll_{odds_type}'] - balances[f'profits_{odds_type}']
    raw_total = ml_returns[odds_type] + pl_returns[odds_type]
    scale = pd.Series(1.0, index=raw_total.index)
    capped = raw_total.lt(-1)
    scale.loc[capped] = -1.0 / raw_total.loc[capped]
    events = pd.DataFrame({
        'date': dates.to_numpy(),
        'bankroll_pct_change': total_returns[odds_type].where(before.gt(0), 0).to_numpy(),
        'bankroll_postevent': balances[f'bankroll_{odds_type}'].to_numpy(),
        'event_payout_money_line': (before * ml_returns[odds_type] * scale).to_numpy(),
        'parlay_net': (before * pl_returns[odds_type] * scale).to_numpy(),
    })
    ml['winner'] = ml['winner_bool']
    ml['pred_winner'] = ml[f'pred_winner_{odds_type}']
    # Negative settlement stakes are losing wagers, not absent wagers.
    ml['choice_fstar'] = ml[f'net_stake_{odds_type}'].abs().fillna(0)
    valid_prices = pd.Series(True, index=ml.index)
    for side in ('red', 'blue'):
        american = pd.to_numeric(ml[f'{odds_type}_{side}'], errors='coerce')
        decimal = (1 + american / 100).where(american > 0, 1 + 100 / american.abs())
        valid_prices &= np.isfinite(decimal) & decimal.gt(1)
        ml[f'dec_{odds_type}_{side}'] = decimal
    ml = ml.loc[valid_prices].merge(events, on='date', how='left', validate='many_to_one')
    pl['fstar_parlay'] = pl[f'net_stake_{odds_type}'].abs().fillna(0)
    pl['choice_fighter_name'] = pl[f'choice_fighter_name_{odds_type}']
    eligible_names = pd.concat([
        ml[['date', 'fighter_red']].rename(columns={'fighter_red': 'choice_fighter_name'}),
        ml[['date', 'fighter_blue']].rename(columns={'fighter_blue': 'choice_fighter_name'}),
    ]).drop_duplicates()
    pl = pl[['date', 'choice_fighter_name', 'fstar_parlay']].merge(
        eligible_names, on=['date', 'choice_fighter_name'], how='inner',
    )
    return events, ml, pl


def plot_returns(ml_results, parlay_results, bankroll_results):

    import matplotlib.pyplot as plt
    types = SETTLED_TYPES

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

    path = settings.bankroll_returns_plot_file
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    commit_figure_if_changed(path)
    plt.close(fig)
    return path


def accuracy_analysis(ml_results, parlay_results):

    accuracies = {}
    bet_types = {}

    type_to_odds = {
        'open': 'open',
        'close1': 'close1',
        'close2': 'close2',
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

        # Settlement stakes are signed: losses must remain in the denominator.
        placed_parlays = parlay_events[f'net_stake_{type_}'].abs().gt(0)
        parlay_net = parlay_events.loc[
            placed_parlays, f'net_odds_{type_}'
        ].dropna()
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



def generate_test_accuracy_results(*, prediction_dir=None, output_dir=None,
                                   model_type="open",
                                   prediction_template="{split}_logit_{model_type}_with_{odds_type}.csv"):
    """Save test-only event accuracy plots, OLS plots and coefficient CSVs.

    Returns per-market artifact paths and combined result DataFrames. Uses only
    test_<...> prediction files; never falls back to training or validation data.
    Files are local exports, without Git commits/pushes. Existing exports are
    replaced when this function is called. No simulations or parameter fitting
    beyond the descriptive event-accuracy OLS regressions are performed.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from ufc_betting.Results.vegas_accuracy import (
        load_accuracy_data, score_predictions, plot_accuracy,
        plot_accuracy_regression, save_png, save_dataframe,
    )

    prediction_dir = Path(prediction_dir) if prediction_dir is not None else settings.model_predictions_folder
    output_dir = Path(output_dir) if output_dir is not None else settings.betting_results_dir / "vegas_accuracy_test"
    datasets, coverage_rows = {}, []
    # Validate all inputs before replacing any artifacts.
    for market in SETTLED_TYPES:
        path = prediction_dir / prediction_template.format(
            split="test", model_type=model_type, odds_type=market)
        if not path.name.startswith("test_"):
            raise ValueError("Test accuracy reports require prediction filenames starting with test_")
        data, coverage = load_accuracy_data(path, market)
        event_scores = data.assign(correct=data.model_prediction.eq(data.winner)).groupby("date").size()
        vegas_events = data.assign(correct=data.vegas_prediction.eq(data.winner)).groupby("date").correct.mean()
        if len(event_scores) < 3 or vegas_events.nunique() < 2:
            raise ValueError(f"test/{market}: regression needs >=3 events and varying Vegas accuracy")
        datasets[market] = data
        coverage_rows.append(dict(split="test", market=market, **coverage))

    artifacts, coefficient_frames, fit_rows, score_rows = {}, [], [], []
    for market, data in datasets.items():
        figures = []
        try:
            accuracy_fig, event_metrics = plot_accuracy(data, "test", market, model_type=model_type)
            figures.append(accuracy_fig)
            fit, coefficients, regression_fig = plot_accuracy_regression(data, "test", market)
            figures.append(regression_fig)
            prefix = f"test_{model_type}_{market}"
            paths = {
                "accuracy_plot": save_png(accuracy_fig, output_dir / f"{prefix}_event_accuracy.png"),
                "regression_plot": save_png(regression_fig, output_dir / f"{prefix}_accuracy_regression.png"),
                "coefficients": save_dataframe(coefficients, output_dir / f"{prefix}_regression_coefficients.csv"),
                "event_metrics": save_dataframe(event_metrics, output_dir / f"{prefix}_event_metrics.csv", index=False),
            }
            artifacts[market] = paths
            coefficient_frames.append(coefficients.reset_index().assign(split="test", market=market))
            fit_rows.append(dict(split="test", market=market, events=int(fit.nobs),
                                 r_squared=fit.rsquared, adjusted_r_squared=fit.rsquared_adj,
                                 residual_df=fit.df_resid, f_statistic=fit.fvalue, f_p_value=fit.f_pvalue))
            for source in ("vegas", "model"):
                score_rows.append(dict(split="test", market=market, source=source,
                                       fights=len(data), **score_predictions(data, source)))
        finally:
            for fig in figures:
                plt.close(fig)

    tables = {
        "coefficients": pd.concat(coefficient_frames, ignore_index=True),
        "fit_summary": pd.DataFrame(fit_rows),
        "accuracy_summary": pd.DataFrame(score_rows),
        "coverage": pd.DataFrame(coverage_rows),
    }
    table_paths = {name: save_dataframe(frame, output_dir / f"{name}.csv", index=False)
                   for name, frame in tables.items()}
    return dict(output_dir=output_dir, artifacts=artifacts, tables=tables, table_paths=table_paths)


def generate_test_return_plots(*, output_dir=None, publish=True):
    """Save notebook-equivalent test plots for the read-only Live Results page."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from ufc_betting.Results.test_simulation import simulate_saved_predictions
    from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
        plot_backtest, plot_bankroll_distributions, plot_event_accuracy_distributions,
    )

    output_dir = Path(output_dir) if output_dir is not None else settings.data_dir / "images" / "plot_pngs"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for market in SETTLED_TYPES:
        moneylines, parlays, *_ = simulate_saved_predictions(
            market, initial_bankroll=500, use_decay=market != "open",
            split="test", model_type="open",
        )
        for suffix, plot in (
            ("backtest", lambda: plot_backtest(moneylines, init_bankroll=500)),
            ("bankroll_distributions", lambda: plot_bankroll_distributions(moneylines, parlays)),
            ("event_accuracy", lambda: plot_event_accuracy_distributions(moneylines, parlays, odds_type=market)),
        ):
            fig, *_ = plot()
            path = output_dir / f"test_{market}_{suffix}.png"
            try:
                fig.savefig(path, dpi=150, bbox_inches="tight")
            finally:
                plt.close(fig)
            if publish:
                commit_figure_if_changed(path)
            saved.append(path)
    return saved
