import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.UpcomingPicks.betting_pipeline import betting_pipeline
from ufc_betting.config import settings


class UpcomingSavedPolicyTests(unittest.TestCase):
    stages = ('open', 'close1', 'close2')

    def frame(self):
        data = pd.DataFrame(dict(
            date=['2026-10-17'] * 4, fighter_red=['A', 'B', 'C', 'D'],
            fighter_blue=['E', 'F', 'G', 'H'], signal=[.85, .7, .25, .65],
            winner=[1, 0, 1, 2],
        ))
        for stage, red, blue in (
            ('open', [1.3, 2.1, 3., 2.2], [4., 1.8, 1.6, 1.8]),
            ('close1', [1.2, 1.9, 3.4, 2.4], [5., 2., 1.5, 1.7]),
            ('close2', [1.1, 2.3, 2.8, 2.], [7., 1.7, 1.8, 2.]),
        ):
            data[f'dec_{stage}_red'], data[f'dec_{stage}_blue'] = red, blue
            for side in ('red', 'blue'):
                data[f'dec_fair_{stage}_{side}'] = data[f'dec_{stage}_{side}'] * 1.05
                data[f'{stage}_{side}'] = 100.  # Display fields only.
        return data

    def run_pipeline(self, data, **options):
        def predict(**kwargs):
            red = data.signal.where(kwargs['valid_mask'])
            sides = (red >= .5).astype(float).where(red.notna())
            return sides, red, 1 - red, pd.Series(.025, index=data.index).where(red.notna())

        with patch('ufc_betting.UpcomingPicks.betting_pipeline.logit_predict', side_effect=predict):
            return betting_pipeline(
                data, [['signal']] * 3, [None] * 3, [None] * 3, [None] * 3,
                list(self.stages),
                [[f'dec_fair_{stage}_blue', f'dec_fair_{stage}_red'] for stage in self.stages],
                [[f'dec_{stage}_blue', f'dec_{stage}_red'] for stage in self.stages],
                500, **options,
            )

    def test_saved_policies_match_backtest_with_decay_uncertainty_and_bounds(self):
        data = self.frame()
        data['winner'] = [1, 0, 1, 0]
        original = data.copy(deep=True)
        ml, tickets, ev = self.run_pipeline(data, return_event_ev=True)
        pd.testing.assert_frame_equal(data, original)
        data['proba_red'], data['proba_blue'] = data.signal, 1 - data.signal
        data['pred_winner'], data['proba_se'] = (data.signal >= .5).astype(int), .025
        for stage in self.stages:
            with self.subTest(stage=stage):
                params = BettingParams.load(settings.data_dir / 'kelly_params' / f'{stage}.json')
                results, parlay = simulate_kelly(
                    data, prob_cols=['proba_blue', 'proba_red'],
                    fair_decimal_cols=[f'dec_fair_{stage}_blue', f'dec_fair_{stage}_red'],
                    real_decimal_cols=[f'dec_{stage}_blue', f'dec_{stage}_red'],
                    pred_winner_col='pred_winner', prediction_se_col='proba_se',
                    init_bankroll=500, calc_parlay=True, bankroll_floor=None,
                    opening_data=data if stage != 'open' else None,
                    **params.to_simulation_kwargs(),
                )
                np.testing.assert_allclose(ml[f'fstar_{stage}'], results.choice_fstar)
                np.testing.assert_allclose(tickets[f'parlay_fstar_{stage}'], parlay.fstar_parlay.astype(float))
                np.testing.assert_allclose(ml[f'stake_{stage}'], 500 * ml[f'fstar_{stage}'])
                self.assertLessEqual(ml[f'fstar_{stage}'].sum() + tickets[f'parlay_fstar_{stage}'].iloc[0], 1.)
                probability = np.maximum(data.signal, 1 - data.signal).to_numpy()
                odds = np.where(data.pred_winner == 1, data[f'dec_{stage}_red'], data[f'dec_{stage}_blue'])
                edges = probability * odds - 1
                legs = np.argsort(-edges, kind='stable')[:2]
                expected = (results.choice_fstar.to_numpy() * edges).sum()
                expected += float(parlay.fstar_parlay.iloc[0]) * (probability[legs].prod() * odds[legs].prod() - 1)
                self.assertAlmostEqual(ev.set_index('bet_type').loc['Combined', f'{stage}_ev_pct'], expected * 100)
        # The saved close2 bound excludes the heavy favorite from both blended policies.
        self.assertEqual(ml.fstar_close2.iloc[0], 0.)

    def test_live_stakes_do_not_depend_on_outcomes(self):
        data = self.frame()
        expected = self.run_pipeline(data)
        for frame in (data.drop(columns='winner'), data.assign(winner=np.nan), data.assign(winner=0)):
            actual = self.run_pipeline(frame)
            for left, right in zip(expected, actual):
                pd.testing.assert_frame_equal(left, right)

    def test_missing_rows_keep_original_ticket_indices(self):
        data = self.frame()
        data.loc[[0, 2], 'signal'] = np.nan
        ml, ticket = self.run_pipeline(data)
        self.assertTrue(ml.loc[[0, 2], 'stake_open'].isna().all())
        self.assertEqual(set(ticket.fight_index_open), {1, 3})
        for idx, name in zip(ticket.fight_index_open, ticket.choice_fighter_name_open):
            self.assertEqual(name, data.loc[idx, 'fighter_red'])

    def test_no_ticket_with_fewer_than_two_valid_fights(self):
        for valid_count in (0, 1):
            data = self.frame()
            data.loc[valid_count:, 'signal'] = np.nan
            ml, ticket, ev = self.run_pipeline(data, return_event_ev=True)
            self.assertEqual(ml.choice_proba_open.notna().sum(), valid_count)
            self.assertTrue(ticket.parlay_fstar_open.isna().all())
            self.assertEqual(ev.set_index('bet_type').loc['Parlay', 'open_ev_pct'], 0.)

    def test_decay_can_be_disabled_and_multiple_events_are_rejected(self):
        data = self.frame()
        decay, _ = self.run_pipeline(data)
        fixed, _ = self.run_pipeline(data, use_decay=False)
        self.assertFalse(np.allclose(decay.fstar_close1, fixed.fstar_close1))
        data.loc[0, 'date'] = '2026-10-24'
        with self.assertRaisesRegex(ValueError, 'one nonempty event'):
            self.run_pipeline(data)

    def test_stale_features_raise_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'Rebuild upcoming event features'):
            self.run_pipeline(self.frame().drop(columns='signal'))


if __name__ == '__main__':
    unittest.main()
