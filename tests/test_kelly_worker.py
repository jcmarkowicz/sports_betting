import unittest

import pandas as pd

from ufc_betting.BettingStrategy.kelly_worker import parlay_top_ev


def parlay_candidates(
    probabilities: list[float],
    odds: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "choice_fighter_name": ["A", "B", "C"],
            "choice_fighter_bool": [1, 0, 1],
            "choice_proba": probabilities,
            "choice_real_odds": odds,
            "choice_ev": [
                probability * real_odds - 1
                for probability, real_odds in zip(probabilities, odds)
            ],
        }
    )


class KellyWorkerTests(unittest.TestCase):
    def test_parlay_can_include_negative_ev_leg_when_combined_ev_positive(
        self,
    ) -> None:
        candidates = parlay_candidates(
            probabilities=[0.8, 0.45, 0.3],
            odds=[1.5, 2.0, 2.0],
        )

        parlay = parlay_top_ev(
            candidates,
            bankroll=500,
            type="open",
            parlay_mdd=None,
        )

        self.assertEqual(
            parlay["choice_fighter_name_open"].tolist(),
            ["A", "B"],
        )
        self.assertGreater(parlay["parlay_ev_open"].iloc[0], 0)
        self.assertGreater(parlay["parlay_fstar_open"].iloc[0], 0)
        self.assertGreater(parlay["stake_open"].iloc[0], 0)

    def test_nonpositive_combined_ev_receives_zero_stake(self) -> None:
        candidates = parlay_candidates(
            probabilities=[0.6, 0.45, 0.3],
            odds=[1.5, 2.0, 2.0],
        )

        parlay = parlay_top_ev(
            candidates,
            bankroll=500,
            type="open",
            parlay_mdd=None,
        )

        self.assertLessEqual(parlay["parlay_ev_open"].iloc[0], 0)
        self.assertEqual(parlay["parlay_fstar_open"].iloc[0], 0)
        self.assertEqual(parlay["stake_open"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
