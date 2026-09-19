import unittest
import numpy as np
from ufc_betting.Models.RiskManagement.realized_ev_target_bands import aggregate, qualifying, advance


class RealizedBandsTests(unittest.TestCase):
    def test_equal_ev_different_realized_returns(self):
        ev,profit,_=aggregate(np.array([[.1,.1],[.1,.1]]),np.array([[.2,.2],[.2,.2]]),
            np.array([[1,-1],[2,-1]]),np.ones((2,2)),np.ones((2,2)))
        np.testing.assert_allclose(ev,[.04,.04])
        ids,returns=qualifying(ev,profit,.08,50)
        np.testing.assert_allclose(returns,[0,.1])

    def test_cap_and_uncertainty(self):
        e,r,exposure=aggregate(np.array([[.8,.8]]),np.array([[.2,.2]]),np.array([[1,1]]),np.ones((1,2)),np.ones((1,2))*.8)
        np.testing.assert_allclose(exposure,[1])
        np.testing.assert_allclose(e,[.2])

    def test_floor_no_topup(self):
        np.testing.assert_allclose(advance(np.array([90.,500.]),np.array([1.,-.9])),[90,50])

    def test_qualify_without_outcome_selection(self):
        a=qualifying(np.array([.1,.2]),np.array([-.5,.5]),.2,50)[0]
        b=qualifying(np.array([.1,.2]),np.array([.5,-.5]),.2,50)[0]
        np.testing.assert_array_equal(a,b)

    def test_filter_never_scales_settlement(self):
        expected=np.array([.049,.05,.051,.08])
        realized=np.array([-.2,.3,.4,.9])
        ids,returns=qualifying(expected,realized,.1,50)
        np.testing.assert_array_equal(ids,[0,1,2])
        np.testing.assert_array_equal(returns,realized[ids])

    def test_no_match_has_no_substitution(self):
        ids,returns=qualifying(np.array([.08]),np.array([.4]),.1,50)
        self.assertEqual(len(ids),0)
        self.assertEqual(len(returns),0)
