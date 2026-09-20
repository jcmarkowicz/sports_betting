import unittest

import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class VoidSettlementTests(unittest.TestCase):
    def simulate(self, winners, regular=False, heavy=False):
        data = pd.DataFrame([
            dict(date='2026-01-01', fighter_red=f'red{i}', fighter_blue=f'blue{i}',
                 winner=winner, pred_winner=1, blue=.05, red=.95,
                 fair_blue=20., fair_red=1.1, odds_blue=5., odds_red=1.2,
                 open_red=-500, open_blue=400, se=.01)
            for i, winner in enumerate(winners)
        ])
        return simulate_kelly(
            data, prob_cols=['blue', 'red'],
            fair_decimal_cols=['fair_blue', 'fair_red'],
            real_decimal_cols=['odds_blue', 'odds_red'],
            pred_winner_col='pred_winner', init_bankroll=500,
            z=0, calc_parlay=regular, calc_heavy_favorite_parlay=heavy,
        )

    def test_void_moneyline_refunds_without_changing_allocation(self):
        won, _ = self.simulate([1, 1])
        void, _ = self.simulate([2, 1])
        pd.testing.assert_series_equal(won.choice_fstar, void.choice_fstar)
        self.assertGreater(void.choice_fstar.iloc[0], 0)
        self.assertEqual(void.fight_payout.iloc[0], 0)
        self.assertEqual(void.net_odds.iloc[0], 0)
        self.assertEqual(void.fstar_net.iloc[0], 0)
        self.assertAlmostEqual(
            void.bankroll_postevent.iloc[0], 500 + void.fight_payout.iloc[1]
        )

    def test_void_leg_refunds_regular_and_heavy_tickets_even_with_loss(self):
        for regular, heavy in [(True, False), (False, True), (True, True)]:
            with self.subTest(regular=regular, heavy=heavy):
                won, won_parlays = self.simulate([1, 1, 1], regular, heavy)
                void, void_parlays = self.simulate([2, 0, 1], regular, heavy)
                # Outcome changes cannot influence selection or allocation.
                pd.testing.assert_series_equal(won.choice_fstar, void.choice_fstar)
                pd.testing.assert_series_equal(
                    won_parlays.choice_fighter_name, void_parlays.choice_fighter_name
                )
                pd.testing.assert_series_equal(
                    won_parlays.fstar_parlay, void_parlays.fstar_parlay
                )
                self.assertTrue(void.parlay_net.eq(0).all())
                self.assertTrue(void.parlay_net_odds.eq(0).all())
                self.assertTrue(void_parlays.fstar_net.eq(0).all())
                self.assertAlmostEqual(
                    void.bankroll_postevent.iloc[0], 500 + void.fight_payout.sum()
                )

    def test_all_void_event_preserves_bankroll(self):
        results, parlays = self.simulate([2, 2], regular=True, heavy=True)
        self.assertTrue(results.bankroll_postevent.eq(500).all())
        self.assertTrue(results.bankroll_pct_change.eq(0).all())
        self.assertTrue(results.fstar_net.eq(0).all())
        self.assertTrue(parlays.fstar_net.eq(0).all())

    def test_binary_loss_still_loses_stakes(self):
        results, parlays = self.simulate([0, 0], regular=True)
        self.assertTrue(results.net_odds.eq(-1).all())
        self.assertTrue(parlays.parlay_net_odds.eq(-1).all())
        self.assertTrue(parlays.fstar_net.lt(0).all())
        self.assertLess(results.bankroll_postevent.iloc[0], 500)


if __name__ == '__main__':
    unittest.main()
