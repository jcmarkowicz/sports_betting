import unittest
import numpy as np
import pandas as pd
from ufc_betting.Models.RiskManagement.decay_sensitivity import decay_event_values
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Convergence.scenario_values import event_values


class DecaySensitivityTests(unittest.TestCase):
    def test_open_endpoint_and_zero_rates(self):
        group=pd.DataFrame(dict(winner=[1,0],fighter_red=['a','b'],fighter_blue=['c','d'],date=['2025-01-01']*2))
        op=np.array([[[.1,.4,1.,.2,.1,.7,2.,1.],[.2,.4,-1.,.2,.1,.7,2.,1.]]])
        cl=op.copy();cl[:,:,0]*=.4
        a=BettingParams();b=BettingParams(mdd_parlay=.2)
        rates=dict(k_down=3.,k_up=4.,k_parlay_down=5.,k_parlay_up=6.)
        for parlay in (False,True):
            expected=event_values(op,group,a,parlay)
            actual=decay_event_values(op,cl,np.full((2,2),2.),group,a,b,rates,parlay)
            np.testing.assert_allclose(actual,expected)
            zero=dict.fromkeys(rates,0.)
            actual=decay_event_values(op,cl,np.full((2,2),3.),group,a,b,zero,parlay)
            np.testing.assert_allclose(actual,expected)

    def test_large_rates_closing_limit(self):
        group=pd.DataFrame(dict(winner=[1,0],fighter_red=['a','b'],fighter_blue=['c','d'],date=['2025-01-01']*2))
        op=np.array([[[.1,.4,1.,.2,.1,.7,2.,1.],[.2,.4,-1.,.2,.1,.7,2.,1.]]])
        cl=op.copy();cl[:,:,0]*=.4
        a=BettingParams();b=BettingParams(mdd_parlay=.2)
        rates=dict.fromkeys(['k_down','k_up','k_parlay_down','k_parlay_up'],1000.)
        for parlay in (False,True):
            actual=decay_event_values(op,cl,np.full((2,2),3.),group,a,b,rates,parlay)
            np.testing.assert_allclose(actual,event_values(cl,group,b,parlay))
