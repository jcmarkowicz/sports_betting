import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.DataPipeline.dataframes.moneylines import MoneylineDataFrame
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline
from ufc_betting.UpcomingPicks.model_helpers import logit_predict


class LiveBacktestParityTests(unittest.TestCase):
    def test_pipeline_caps_before_void_settlement_and_matches_simulator(self):
        data = pd.DataFrame([
            dict(date='2026-01-01', fighter_red=f'red{i}', fighter_blue=f'blue{i}',
                 winner=winner, winner_name=f'red{i}', signal=.85,
                 pred_winner=1, proba_red=.85, proba_blue=.15,
                 dec_fair_open_red=1.9, dec_fair_open_blue=2.1,
                 dec_open_red=1.8, dec_open_blue=2.,
                 open_red=-125., open_blue=100., close1_red=-125.,
                 close1_blue=100., close2_red=-125., close2_blue=100.)
            for i, winner in enumerate([2, 0, 1, 1, 1])
        ])
        predictions = (data.pred_winner, data.proba_red, data.proba_blue)
        with patch('ufc_betting.UpcomingPicks.betting_pipeline.logit_predict',
                   return_value=predictions), patch('builtins.print'):
            ml, parlay = betting_pipeline(
                data, [['signal']], [None], [None], [None], ['open'],
                [['dec_fair_open_blue', 'dec_fair_open_red']],
                [['dec_open_blue', 'dec_open_red']], 500, [.3], [.5], [250], [500],
            )
        results, tickets = simulate_kelly(
            data, prob_cols=['proba_blue', 'proba_red'],
            fair_decimal_cols=['dec_fair_open_blue', 'dec_fair_open_red'],
            real_decimal_cols=['dec_open_blue', 'dec_open_red'],
            pred_winner_col='pred_winner', init_bankroll=500,
            max_drawdown=.3, parlay_mdd=.5, N=250, N_parlay=500,
            calc_parlay=True, adjust_mdd_by_edge=False,
        )
        self.assertAlmostEqual(ml.fstar_open.sum() + parlay.parlay_fstar_open.iloc[0], 1.)
        np.testing.assert_allclose(ml.fstar_open, results.choice_fstar)
        np.testing.assert_allclose(parlay.parlay_fstar_open, tickets.fstar_parlay.astype(float))
        for frame in (ml, parlay):
            for column in list(frame.columns):
                if column.endswith('_open'):
                    for stage in ('close1', 'close2'):
                        frame[column[:-4] + stage] = frame[column]
        settled_ml = MoneylineDataFrame(ml).with_results(data).frame
        settled_parlay = ParlayDataFrame(
            parlay.assign(date='2026-01-01')
        ).with_results(data).frame
        self.assertTrue(settled_parlay.net_odds_open.isna().all())
        live_profit = (settled_ml.net_stake_open.abs() * settled_ml.net_odds_open).sum() * 500
        self.assertAlmostEqual(500 + live_profit, results.bankroll_postevent.iloc[0])

    def test_close_feature_replaces_open_without_duplicate_columns(self):
        class Scaler:
            def transform(self, frame):
                assert frame.columns.tolist() == ['proba_fair_open_diff']
                np.testing.assert_allclose(frame.iloc[:, 0], [.4, -.2])
                return frame.to_numpy()

        class Encoder:
            def transform(self, frame):
                return np.empty((len(frame), 0))

            def get_feature_names_out(self, columns):
                return []

        class Model:
            class model:
                exog_names = ['const', 'proba_fair_open_diff']

            def predict(self, frame):
                return .5 + frame.proba_fair_open_diff / 2

        data = pd.DataFrame({'proba_fair_open_diff': [.1, .1],
                             'proba_fair_close1_diff': [.4, -.2]})
        original = data.copy()
        _, red, _ = logit_predict(
            Model(), data, pd.Series(np.nan, index=data.index),
            ['proba_fair_close1_diff'], ['proba_fair_close1_diff'], [],
            pd.Series(True, index=data.index), Scaler(), Encoder(), data.index,
            rename_odds='close1',
        )
        np.testing.assert_allclose(red, [.7, .4])
        pd.testing.assert_frame_equal(data, original)


if __name__ == '__main__':
    unittest.main()
