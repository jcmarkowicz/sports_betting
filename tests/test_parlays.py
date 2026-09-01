import unittest

import pandas as pd

from ufc_betting.DataPipeline.dataframes.parlays import (
    ParlayDataFrame,
    ParlayIntegrityError,
)


SETTLED_TYPES = ("open", "close1_stack", "close2_stack")


def parlay_frame() -> pd.DataFrame:
    frame = pd.DataFrame(index=range(2))

    frame["choice_fighter_name_open"] = ["Red One", "Blue Two"]
    frame["choice_fighter_bool_open"] = [1, 0]
    frame["parlay_fstar_open"] = [0.1, 0.1]
    frame["parlay_odds_open"] = [3.0, 3.0]

    frame["choice_fighter_name_close1_stack"] = [
        "Red One",
        "Blue Three",
    ]
    frame["choice_fighter_bool_close1_stack"] = [1, 0]
    frame["parlay_fstar_close1_stack"] = [0.2, 0.2]
    frame["parlay_odds_close1_stack"] = [2.5, 2.5]

    frame["choice_fighter_name_close2_stack"] = pd.Series(
        [pd.NA, pd.NA],
        dtype="string",
    )
    frame["choice_fighter_bool_close2_stack"] = pd.Series(
        [pd.NA, pd.NA],
        dtype="Int64",
    )
    frame["parlay_fstar_close2_stack"] = pd.Series(
        [pd.NA, pd.NA],
        dtype="Float64",
    )
    frame["parlay_odds_close2_stack"] = pd.Series(
        [pd.NA, pd.NA],
        dtype="Float64",
    )

    return frame


def event_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fighter_red": ["Red Three", " red one ", "Red Two"],
            "fighter_blue": ["Blue Three", "BLUE ONE", "Blue Two"],
            "winner": [0, 1, 2],
            "winner_name": ["Blue Three", "Red One", "DRAW"],
        },
        index=[30, 10, 20],
    )


class ParlayDataFrameTests(unittest.TestCase):
    def test_generated_parlays_preserve_unsettled_rows(self) -> None:
        parlays = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )

        self.assertEqual(len(parlays.frame), 2)
        for bet_type in SETTLED_TYPES:
            self.assertTrue(
                parlays.frame[f"winner_bool_{bet_type}"].isna().all()
            )

    def test_tie_voids_only_affected_parlay_type(self) -> None:
        parlays = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )

        settled = parlays.with_results(event_results())

        self.assertIsNotNone(settled)
        assert settled is not None
        self.assertTrue(settled.frame["winner_bool_open"].isna().all())
        self.assertTrue(settled.frame["net_odds_open"].isna().all())
        self.assertEqual(
            settled.frame["winner_bool_close1_stack"].tolist(),
            [1, 0],
        )
        self.assertEqual(
            settled.frame["win_parlay_close1_stack"].tolist(),
            [True, True],
        )
        self.assertEqual(
            settled.frame["net_odds_close1_stack"].tolist(),
            [2.5, 2.5],
        )

    def test_all_nonbinary_parlay_types_return_empty_dataframe(self) -> None:
        frame = parlay_frame()
        frame["choice_fighter_name_close1_stack"] = [
            "Red One",
            "Blue Two",
        ]
        frame["choice_fighter_bool_close1_stack"] = [1, 0]
        parlays = ParlayDataFrame.from_generated(
            frame,
            event_date="2026-08-01",
        )

        settled = parlays.with_results(event_results())

        self.assertIsInstance(settled, ParlayDataFrame)
        self.assertTrue(settled.frame.empty)

    def test_extra_and_reordered_event_rows_match_by_fighter(self) -> None:
        parlays = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )
        event = pd.concat(
            [
                event_results(),
                pd.DataFrame(
                    {
                        "fighter_red": ["Extra Red"],
                        "fighter_blue": ["Extra Blue"],
                        "winner": [1],
                        "winner_name": ["Extra Red"],
                    }
                ),
            ],
            ignore_index=True,
        ).sample(frac=1, random_state=3)

        settled = parlays.with_results(event)

        self.assertIsNotNone(settled)
        assert settled is not None
        self.assertEqual(
            settled.frame["winner_name_close1_stack"].tolist(),
            ["Red One", "Blue Three"],
        )

    def test_missing_selected_fighter_raises(self) -> None:
        parlays = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )
        event = event_results().loc[
            lambda frame: frame["fighter_blue"].ne("Blue Three")
        ]

        with self.assertRaisesRegex(ParlayIntegrityError, "missing"):
            parlays.with_results(event)

    def test_duplicate_parlay_leg_raises(self) -> None:
        frame = parlay_frame()
        frame["choice_fighter_name_close1_stack"] = [
            "Red One",
            "Red One",
        ]
        frame["choice_fighter_bool_close1_stack"] = [1, 1]
        parlays = ParlayDataFrame.from_generated(
            frame,
            event_date="2026-08-01",
        )

        with self.assertRaisesRegex(ParlayIntegrityError, "duplicate"):
            parlays.with_results(event_results())

    def test_invalid_choice_value_raises(self) -> None:
        frame = parlay_frame()
        frame.loc[0, "choice_fighter_bool_open"] = 2
        parlays = ParlayDataFrame.from_generated(
            frame,
            event_date="2026-08-01",
        )

        with self.assertRaisesRegex(ParlayIntegrityError, "0, 1, or null"):
            parlays.with_results(event_results())

    def test_settled_type_requires_valid_odds(self) -> None:
        frame = parlay_frame()
        frame["parlay_odds_close1_stack"] = pd.NA
        parlays = ParlayDataFrame.from_generated(
            frame,
            event_date="2026-08-01",
        )

        with self.assertRaisesRegex(ParlayIntegrityError, "positive numeric"):
            parlays.with_results(event_results())

    def test_history_voids_complete_type_for_affected_date(self) -> None:
        frame = parlay_frame()
        frame["date"] = "2026-08-01"
        frame["winner_bool_open"] = pd.Series(
            [1, 0.5],
            dtype="Float64",
        )
        frame["winner_name_open"] = ["Red One", "DRAW"]
        frame["win_parlay_open"] = [True, True]
        frame["net_stake_open"] = [0.1, 0.1]
        frame["net_odds_open"] = [3.0, 3.0]

        parlays = ParlayDataFrame(frame)

        for prefix in (
            "winner_bool",
            "winner_name",
            "win_parlay",
            "net_stake",
            "net_odds",
        ):
            self.assertTrue(
                parlays.frame[f"{prefix}_open"].isna().all()
            )

    def test_all_tie_result_returns_empty_dataframe(self) -> None:
        parlays = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )
        event = event_results().assign(winner=2, winner_name="DRAW")

        settled = parlays.with_results(event)

        self.assertIsInstance(settled, ParlayDataFrame)
        self.assertTrue(settled.frame.empty)

    def test_concatenate_ignores_empty_parlay(self) -> None:
        history_frame = parlay_frame()
        history_frame["date"] = "2026-07-01"
        history = ParlayDataFrame(history_frame)
        empty = ParlayDataFrame(history.frame.iloc[0:0].copy())

        combined = ParlayDataFrame.concatenate(history, empty)

        self.assertIsNotNone(combined)
        assert combined is not None
        self.assertEqual(len(combined.frame), 2)

    def test_concatenate_replaces_entire_event_and_keeps_all_legs(self) -> None:
        old_event = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-08-01",
        )
        other_event = ParlayDataFrame.from_generated(
            parlay_frame(),
            event_date="2026-07-01",
        )
        history = ParlayDataFrame.concatenate(other_event, old_event)
        assert history is not None

        updated_frame = parlay_frame()
        updated_frame["parlay_odds_open"] = [4.0, 4.0]
        updated_event = ParlayDataFrame.from_generated(
            updated_frame,
            event_date="2026-08-01",
        )

        combined = ParlayDataFrame.concatenate(history, updated_event)

        assert combined is not None
        august = combined.frame.loc[
            combined.frame["date"].eq(pd.Timestamp("2026-08-01"))
        ]
        self.assertEqual(len(combined.frame), 4)
        self.assertEqual(len(august), 2)
        self.assertEqual(august["parlay_odds_open"].tolist(), [4.0, 4.0])

    def test_concatenate_removes_exact_duplicate_rows(self) -> None:
        frame = parlay_frame()
        frame["date"] = "2026-08-01"
        duplicated = ParlayDataFrame(
            pd.concat([frame, frame], ignore_index=True)
        )

        combined = ParlayDataFrame.concatenate(duplicated)

        assert combined is not None
        self.assertEqual(len(combined.frame), 2)


if __name__ == "__main__":
    unittest.main()
