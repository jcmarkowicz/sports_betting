import unittest

import pandas as pd

from ufc_betting.BettingStrategy.kelly_worker import cap_event_exposure, parlay_top_ev


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
    def test_cap_counts_ticket_once_and_scales_both_stakes(self):
        moneylines = pd.DataFrame({'fstar_open': [.6, .3], 'stake_open': [300., 150.]})
        parlays = pd.DataFrame({'parlay_fstar_open': [.3, .3], 'stake_open': [150., 150.]})
        ml, parlay = cap_event_exposure(moneylines, parlays, 'open')
        self.assertAlmostEqual(ml.fstar_open.sum() + parlay.parlay_fstar_open.iloc[0], 1.)
        self.assertAlmostEqual(ml.stake_open.sum() + parlay.stake_open.iloc[0], 500.)
        self.assertAlmostEqual(ml.fstar_open.iloc[0], .5)
        self.assertAlmostEqual(parlay.parlay_fstar_open.iloc[0], .25)
        self.assertEqual(parlay.parlay_fstar_open.nunique(), 1)
        self.assertEqual(moneylines.fstar_open.iloc[0], .6)

    def test_cap_leaves_under_budget_and_missing_parlay_unchanged(self):
        ml = pd.DataFrame({'fstar_open': [.2, .1], 'stake_open': [100., 50.]})
        for fraction in [.3, float('nan')]:
            parlay = pd.DataFrame({'parlay_fstar_open': [fraction, fraction],
                                  'stake_open': [fraction * 500, fraction * 500]})
            capped_ml, capped_parlay = cap_event_exposure(ml, parlay, 'open')
            pd.testing.assert_frame_equal(ml, capped_ml)
            pd.testing.assert_frame_equal(parlay, capped_parlay)

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
