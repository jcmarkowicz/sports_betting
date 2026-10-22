import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import brier_score_loss

from ufc_betting.Results.returns import (
    _event_return_fractions,
    accuracy_analysis,
    prepare_return_plot_data,
    returns_by_date,
)


BET_TYPES = ("open", "close1", "close2")


def return_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    moneylines = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 2)})
    parlays = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"] * 2)})
    for bet_type in BET_TYPES:
        moneylines[f"net_stake_{bet_type}"] = [0.8, 0.8]
        moneylines[f"net_odds_{bet_type}"] = [-1.0, -1.0]
        parlays[f"net_stake_{bet_type}"] = [0.5, 0.5]
        parlays[f"net_odds_{bet_type}"] = [-1.0, -1.0]
    return moneylines, parlays


class ReturnTests(unittest.TestCase):
    def test_tracking_metrics_are_unchanged_by_draws_and_no_contests(self):
        moneylines, parlays = return_frames()
        moneylines['winner_bool'] = [1, 0]
        for market in BET_TYPES:
            moneylines[f'pred_winner_{market}'] = [1, 0]
            moneylines[f'choice_proba_{market}'] = [.8, .7]
            moneylines[f'{market}_red'] = [-150, -150]
            moneylines[f'{market}_blue'] = [130, 130]
        expected, expected_types = accuracy_analysis(moneylines, parlays)
        void_ml = moneylines.copy()
        void_ml['winner_bool'] = [2, 3]
        void_ml['date'] = pd.to_datetime(['2026-02-01', '2026-03-01'])
        void_pl = parlays.copy()
        void_pl['date'] = void_ml.date
        for market in BET_TYPES:
            void_ml[f'net_odds_{market}'] = 0.
            void_pl[f'net_odds_{market}'] = 0.
        actual, actual_types = accuracy_analysis(
            pd.concat([moneylines, void_ml], ignore_index=True),
            pd.concat([parlays, void_pl], ignore_index=True))
        expected = expected.set_index('metric').Accuracies
        actual = actual.set_index('metric').Accuracies
        for market in BET_TYPES:
            for metric in ('preds_all', 'bets', 'vegas', 'brier_all', 'brier_bets', 'brier_vegas', 'parlays'):
                key = f'{metric}_{market}'
                self.assertEqual(actual[key], expected[key], key)
        pd.testing.assert_frame_equal(actual_types, expected_types)
        void_scores, _ = accuracy_analysis(void_ml, void_pl)
        void_scores = void_scores.set_index('metric').Accuracies
        for market in BET_TYPES:
            for metric in ('preds_all', 'bets', 'vegas', 'brier_all', 'brier_bets', 'brier_vegas', 'parlays'):
                self.assertTrue(pd.isna(void_scores[f'{metric}_{market}']))

    def test_plot_inputs_preserve_losses_and_actual_bankroll_contributions(self):
        moneylines, parlays = return_frames()
        moneylines['fighter_red'] = ['a', 'c']
        moneylines['fighter_blue'] = ['b', 'd']
        moneylines['winner_bool'] = [0, 1]
        moneylines['pred_winner_open'] = [1, 0]
        moneylines['open_red'] = [-150, 130]
        moneylines['open_blue'] = [130, -150]
        moneylines['net_stake_open'] = [-0.8, -0.8]
        parlays['choice_fighter_name_open'] = ['a', 'd']
        parlays['net_stake_open'] = [-0.5, -0.5]
        balances = pd.DataFrame({
            'date': ['2026-01-01'], 'bankroll_open': [0.0],
            'profits_open': [-1000.0],
        })
        events, ml, pl = prepare_return_plot_data(
            moneylines, parlays, balances, 'open',
        )
        self.assertAlmostEqual(events.event_payout_money_line.iloc[0], -1000 * 1.6 / 2.1)
        self.assertAlmostEqual(events.parlay_net.iloc[0], -1000 * 0.5 / 2.1)
        self.assertEqual(events.bankroll_pct_change.iloc[0], -1)
        self.assertEqual(ml.choice_fstar.tolist(), [0.8, 0.8])
        self.assertEqual(pl.fstar_parlay.tolist(), [0.5, 0.5])
        self.assertAlmostEqual(ml.dec_open_red.iloc[0], 1 + 100 / 150)
        self.assertNotIn('choice_fstar', moneylines)

    def test_event_return_is_deduplicated_and_floored_at_full_loss(self) -> None:
        moneylines, parlays = return_frames()
        moneylines = pd.concat([moneylines, moneylines], ignore_index=True)
        parlays = pd.concat([parlays, parlays], ignore_index=True)

        dates, event_returns = _event_return_fractions(moneylines, parlays)

        self.assertEqual(len(dates), 1)
        for bet_type in BET_TYPES:
            self.assertEqual(event_returns[bet_type].iloc[0], -1.0)

    @patch("ufc_betting.Results.returns.commit_if_changed")
    @patch("ufc_betting.Results.returns.pd.read_csv")
    @patch("ufc_betting.Results.returns.ParlayDataFrame")
    @patch("ufc_betting.Results.returns.MoneylineDataFrame")
    def test_bankroll_replenishment_is_counted(
        self,
        moneyline_class,
        parlay_class,
        read_csv,
        commit_if_changed,
    ) -> None:
        moneylines, parlays = return_frames()
        read_csv.side_effect = [moneylines, parlays]
        moneyline_class.return_value.frame = moneylines
        parlay_class.return_value.frame = parlays

        results = returns_by_date(
            starting_bankroll=500,
            bankroll_floor=100,
            replenishment_amount=1000,
        )

        for bet_type in BET_TYPES:
            self.assertEqual(results[f"multiplier_{bet_type}"].iloc[0], 0.0)
            self.assertFalse(results[f"profitable_event_{bet_type}"].iloc[0])
            self.assertTrue(results[f"replenished_{bet_type}"].iloc[0])
            self.assertEqual(
                results[f"replenishment_count_{bet_type}"].iloc[0],
                1,
            )
            self.assertEqual(results[f"bankroll_{bet_type}"].iloc[0], 0.0)
        commit_if_changed.assert_called_once()

    def test_accuracy_reports_profitable_event_percentage(self) -> None:
        moneylines, parlays = return_frames()
        for bet_type in BET_TYPES:
            moneylines[f"pred_winner_{bet_type}"] = [1, 0]
            moneylines[f"choice_proba_{bet_type}"] = [0.8, 0.7]
            moneylines[f"winner_bool"] = [1, 0]
            moneylines[f"open_red"] = [-150, -150]
            moneylines[f"open_blue"] = [130, 130]
            moneylines[f"close1_red"] = [-150, -150]
            moneylines[f"close1_blue"] = [130, 130]
            moneylines[f"close2_red"] = [-150, -150]
            moneylines[f"close2_blue"] = [130, 130]

        accuracies, _ = accuracy_analysis(moneylines, parlays)
        values = accuracies.set_index("metric")["Accuracies"]

        for bet_type in BET_TYPES:
            self.assertEqual(values[f"profitable_events_{bet_type}"], 0)
            self.assertEqual(values[f"total_events_{bet_type}"], 1)
            self.assertEqual(values[f"profitable_event_pct_{bet_type}"], 0)
            self.assertEqual(values[f"total_event_profit_pct_{bet_type}"], 0)
            self.assertEqual(values[f"profitable_ml_events_{bet_type}"], 0)
            self.assertEqual(values[f"ml_event_profit_pct_{bet_type}"], 0)
            self.assertEqual(
                values[f"profitable_parlay_events_{bet_type}"],
                0,
            )
            self.assertEqual(
                values[f"parlay_event_profit_pct_{bet_type}"],
                0,
            )
            self.assertEqual(values[f"brier_all_{bet_type}"], 0.065)
            self.assertEqual(values[f"brier_bets_{bet_type}"], 0.065)
            red_implied = 1 / (1 + 100 / 150)
            blue_implied = 1 / (1 + 130 / 100)
            vegas_red_probability = red_implied / (
                red_implied + blue_implied
            )
            expected_vegas_brier = brier_score_loss(
                [1, 0],
                [vegas_red_probability, vegas_red_probability],
            )
            self.assertAlmostEqual(
                values[f"brier_vegas_{bet_type}"],
                round(expected_vegas_brier, 4),
            )

    def test_brier_score_excludes_missing_probabilities(self) -> None:
        moneylines, parlays = return_frames()
        for bet_type, odds_type in (
            ("open", "open"),
            ("close1", "close1"),
            ("close2", "close2"),
        ):
            moneylines[f"pred_winner_{bet_type}"] = [1, 0]
            moneylines[f"choice_proba_{bet_type}"] = [1.0, pd.NA]
            moneylines["winner_bool"] = [1, 1]
            moneylines[f"{odds_type}_red"] = [-150, -150]
            moneylines[f"{odds_type}_blue"] = [130, 130]

        accuracies, _ = accuracy_analysis(moneylines, parlays)
        values = accuracies.set_index("metric")["Accuracies"]

        for bet_type in BET_TYPES:
            self.assertEqual(values[f"brier_all_{bet_type}"], 0.0)
            self.assertEqual(values[f"brier_bets_{bet_type}"], 0.0)


if __name__ == "__main__":
    unittest.main()
