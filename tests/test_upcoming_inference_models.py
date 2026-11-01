"""Integration checks against exported inference predictions and backtest stakes."""
import unittest

import numpy as np
import pandas as pd

from ufc_betting.config import settings
from ufc_betting.UpcomingPicks.generate_bets import generate_bets
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


class UpcomingInferenceTests(unittest.TestCase):
    def test_market_predictions_and_saved_policy_backtest_parity(self):
        root = settings.data_dir / 'model_comparisons/lr_open_close'
        if not (root / 'trainval/open/inference/test_with_open.csv').exists():
            self.skipTest('Local inference artifacts are not available')
        data = pd.read_csv(root / 'cohort_test.csv')
        data = data.loc[data.date.eq(data.date.iloc[0])].reset_index(drop=True)
        original = data.copy(deep=True)
        ml, tickets, ev = generate_bets(data, bankroll=500, return_event_ev=True)
        pd.testing.assert_frame_equal(data, original)
        for stage in ('open', 'close1', 'close2'):
            saved = pd.read_csv(root / f'trainval/{stage}/inference/test_with_{stage}.csv')
            saved = data[['date', 'fighter_red', 'fighter_blue']].merge(
                saved, on=['date', 'fighter_red', 'fighter_blue'], validate='one_to_one')
            probability = np.maximum(saved.proba_red, 1 - saved.proba_red)
            np.testing.assert_allclose(ml[f'choice_proba_{stage}'], probability)
            params = BettingParams.load(settings.data_dir / f'kelly_params/inference_market_models/{stage}.json')
            saved['proba_blue'] = 1 - saved.proba_red
            singles, parlays = simulate_kelly(
                saved, prob_cols=['proba_blue', 'proba_red'],
                fair_decimal_cols=[f'dec_fair_{stage}_blue', f'dec_fair_{stage}_red'],
                real_decimal_cols=[f'dec_{stage}_blue', f'dec_{stage}_red'],
                pred_winner_col='pred_winner', prediction_se_col='proba_se',
                init_bankroll=500, calc_parlay=True, bankroll_floor=None,
                opening_data=saved if stage != 'open' else None,
                **params.to_simulation_kwargs())
            np.testing.assert_allclose(ml[f'fstar_{stage}'], singles.choice_fstar)
            np.testing.assert_allclose(tickets[f'parlay_fstar_{stage}'], parlays.fstar_parlay.astype(float))
        selected, _ = generate_bets(data, select_odds=0, bankroll=500)
        self.assertIn('stake_open', selected)
        self.assertNotIn('stake_close1', selected)
        self.assertEqual(len(ev), 3)


if __name__ == '__main__':
    unittest.main()
