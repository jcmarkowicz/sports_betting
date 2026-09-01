import unittest

import pandas as pd

from ufc_betting.DataPipeline.dataframes.moneylines import (
    MoneylineDataFrame,
    MoneylineIntegrityError,
)


BET_TYPES = (
    "open",
    "close1",
    "close2",
    "close1_stack",
    "close2_stack",
)


def moneyline_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": ["2026-08-01", "2026-08-01"],
            "fighter_red": ["Red One", "Red Two"],
            "fighter_blue": ["Blue One", "Blue Two"],
            "open_red": [-150, 120],
            "open_blue": [130, -140],
            "close1_red": [-160, 110],
            "close1_blue": [140, -130],
            "close2_red": [-170, 100],
            "close2_blue": [150, -120],
        }
    )
    for bet_type in BET_TYPES:
        frame[f"pred_name_{bet_type}"] = ["Red One", "Blue Two"]
        frame[f"pred_winner_{bet_type}"] = [1, 0]
        frame[f"choice_proba_{bet_type}"] = [0.7, 0.6]
        frame[f"fstar_{bet_type}"] = [0.1, 0.2]
        frame[f"stake_{bet_type}"] = [10.0, 20.0]
        frame[f"edge_{bet_type}"] = [0.1, 0.1]
        frame[f"ev_{bet_type}"] = [0.1, 0.1]
    return frame


class MoneylineDataFrameTests(unittest.TestCase):
    def test_generated_moneylines_preserve_unsettled_rows(self) -> None:
        moneylines = MoneylineDataFrame.from_generated(moneyline_frame())

        self.assertEqual(len(moneylines.frame), 2)
        self.assertTrue(moneylines.frame["winner_bool"].isna().all())

    def test_results_match_by_fighters_and_drop_ties(self) -> None:
        moneylines = MoneylineDataFrame.from_generated(moneyline_frame())
        event = pd.DataFrame(
            {
                "fighter_red": ["Extra Red", " red two ", "RED ONE"],
                "fighter_blue": ["Extra Blue", "blue two", "BLUE ONE"],
                "winner": [0, 2, 1],
                "winner_name": ["Extra Blue", "DRAW", "Red One"],
            }
        )

        settled = moneylines.with_results(event)

        self.assertIsNotNone(settled)
        assert settled is not None
        self.assertEqual(settled.frame["fighter_red"].tolist(), ["Red One"])
        self.assertEqual(settled.frame["winner_bool"].tolist(), [1])
        self.assertEqual(settled.frame["win_bet_open"].tolist(), [True])

    def test_all_nonbinary_results_return_none(self) -> None:
        moneylines = MoneylineDataFrame.from_generated(moneyline_frame())
        event = pd.DataFrame(
            {
                "fighter_red": ["Red One", "Red Two"],
                "fighter_blue": ["Blue One", "Blue Two"],
                "winner": [2, pd.NA],
                "winner_name": ["DRAW", pd.NA],
            }
        )

        self.assertIsNone(moneylines.with_results(event))

    def test_missing_moneyline_fight_raises(self) -> None:
        moneylines = MoneylineDataFrame.from_generated(moneyline_frame())
        event = pd.DataFrame(
            {
                "fighter_red": ["Red One"],
                "fighter_blue": ["Blue One"],
                "winner": [1],
                "winner_name": ["Red One"],
            }
        )

        with self.assertRaisesRegex(MoneylineIntegrityError, "missing"):
            moneylines.with_results(event)

    def test_duplicate_event_fight_raises(self) -> None:
        moneylines = MoneylineDataFrame.from_generated(moneyline_frame())
        event = pd.DataFrame(
            {
                "fighter_red": ["Red One", " red one "],
                "fighter_blue": ["Blue One", "BLUE ONE"],
                "winner": [1, 1],
                "winner_name": ["Red One", "Red One"],
            }
        )

        with self.assertRaisesRegex(
            MoneylineIntegrityError,
            "duplicate fights",
        ):
            moneylines.with_results(event)

    def test_constructor_drops_nonbinary_but_preserves_nulls(self) -> None:
        frame = moneyline_frame()
        extra = frame.iloc[[0]].copy()
        extra["fighter_red"] = "Red Three"
        extra["fighter_blue"] = "Blue Three"
        frame = pd.concat([frame, extra], ignore_index=True)
        frame["winner_bool"] = pd.Series(
            [1, 0.5, pd.NA],
            dtype="Float64",
        )

        moneylines = MoneylineDataFrame(frame)

        self.assertEqual(
            moneylines.frame["fighter_red"].tolist(),
            ["Red One", "Red Three"],
        )
        self.assertEqual(moneylines.frame["winner_bool"].iloc[0], 1)
        self.assertTrue(pd.isna(moneylines.frame["winner_bool"].iloc[1]))

    def test_concatenate_replaces_event_and_keeps_all_fights(self) -> None:
        old_event = MoneylineDataFrame(moneyline_frame())
        other_frame = moneyline_frame()
        other_frame["date"] = "2026-07-01"
        other_event = MoneylineDataFrame(other_frame)
        history = MoneylineDataFrame.concatenate(other_event, old_event)
        assert history is not None

        updated_frame = moneyline_frame()
        updated_frame["open_red"] = [-200, 175]
        updated_event = MoneylineDataFrame(updated_frame)

        combined = MoneylineDataFrame.concatenate(history, updated_event)

        assert combined is not None
        august = combined.frame.loc[
            combined.frame["date"].eq(pd.Timestamp("2026-08-01"))
        ]
        self.assertEqual(len(combined.frame), 4)
        self.assertEqual(len(august), 2)
        self.assertEqual(august["open_red"].tolist(), [-200, 175])

    def test_concatenate_removes_exact_duplicate_rows(self) -> None:
        frame = moneyline_frame()
        duplicated = MoneylineDataFrame(
            pd.concat([frame, frame], ignore_index=True)
        )

        combined = MoneylineDataFrame.concatenate(duplicated)

        assert combined is not None
        self.assertEqual(len(combined.frame), 2)


if __name__ == "__main__":
    unittest.main()
