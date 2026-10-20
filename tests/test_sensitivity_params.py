import unittest
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache, event_values


class ParameterTests(unittest.TestCase):
    def test_filter_and_separate_sizing(self):
        data = pd.DataFrame(dict(pred_winner=[1,1], winner=[1,0],
            proba_red=[.9,.8], proba_blue=[.1,.2], proba_se=[.02,.02],
            dec_fair_close1_red=[1.4,2.], dec_fair_close1_blue=[3.5,2.],
            dec_close1_red=[1.25,2.], dec_close1_blue=[4.,2.]))
        opening = BettingParams(z=None)
        closing = BettingParams(z=1., min_moneyline=-300, N=1000, mdd=.2)
        op = fight_cache(data,opening)
        cl = fight_cache(data,closing)
        self.assertGreater(op[0,0],0)
        self.assertEqual(cl[0,0],0)  # -400 excluded from singles
        self.assertGreater(cl[1,0],0)
        self.assertNotEqual(op[1,0],cl[1,0])
        self.assertLess(cl[1,4],op[1,4])
        # Filtering singles must retain the candidate's parlay probability/EV.
        np.testing.assert_array_equal(cl[:,5:8],op[:,5:8])
        before = op.copy()
        event_values(cl[None,:,:],data,closing,False)
        np.testing.assert_array_equal(op,before)

    def test_invalid_parameters(self):
        for kwargs in (dict(N=1),dict(N_parlay=2.5),dict(mdd=1),dict(mdd_parlay=0),dict(z=-1)):
            with self.assertRaises(ValueError): BettingParams(**kwargs)


if __name__ == '__main__':
    unittest.main()
