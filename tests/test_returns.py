import unittest
from unittest.mock import patch

import pandas as pd

from ufc_betting.Results.returns import (
    _event_return_fractions,
    accuracy_analysis,
    returns_by_date,
)


BET_TYPES = ("open", "close1_stack", "close2_stack")


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
            self.assertEqual(results[f"bankroll_{bet_type}"].iloc[0], 1000.0)
        commit_if_changed.assert_called_once()

    def test_accuracy_reports_profitable_event_percentage(self) -> None:
        moneylines, parlays = return_frames()
        for bet_type in BET_TYPES:
            moneylines[f"pred_winner_{bet_type}"] = [1, 0]
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


if __name__ == "__main__":
    unittest.main()
