import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'StreamlitPages'))
from upcoming_scaling import wager_tables,event_decision

class UpcomingScalingTests(unittest.TestCase):
    def test_stakes_and_ticket_count(self):
        ml=pd.DataFrame({'fighter_red':['a','c'],'fighter_blue':['b','d']})
        pl=pd.DataFrame(index=range(2));decisions={}
        for market in ['open','close1','close2']:
            ml[f'pred_name_{market}']=['a','d'];ml[f'stake_{market}']=[10.,20.]
            pl[f'choice_fighter_name_{market}']=['a','d'];pl[f'stake_{market}']=[8.,8.]
            decisions[market]=dict(multiplier=.5,label='Vol',trend='Uptrend',status='Reduced')
        original=ml.copy(deep=True)
        money,parlay,summary=wager_tables(ml,pl,decisions)
        np.testing.assert_allclose(money['Opening wager'],[5.,10.])
        self.assertEqual(parlay.Wager.iloc[0],4.)
        self.assertEqual(summary.Total_wager.iloc[0],19.)
        pd.testing.assert_frame_equal(ml,original)
        ml.loc[0,'stake_open']=np.nan
        money,_,summary=wager_tables(ml,pl,decisions)
        self.assertTrue(pd.isna(money['Opening wager'].iloc[0]))
        self.assertTrue(pd.isna(summary.Total_wager.iloc[0]))

    def test_raw_requires_no_history(self):
        self.assertEqual(event_decision('2026-11-14','Raw',{})['multiplier'],1.)

    def test_flat_gate_zero_and_future_outcomes_ignored(self):
        hist=pd.DataFrame(dict(date=pd.to_datetime(['2026-11-01','2026-11-20']),
            rolling_advantage=[.2,99.],advantage_slope=[0.,99.],vegas_actual=[.2,99.]))
        source={'event_history':hist,'trend_scale':1.}
        decision=event_decision('2026-11-14','Combined',{'open':source})
        self.assertEqual(decision['multiplier'],0.)
        self.assertEqual(decision['trend'],'Flat')

if __name__=='__main__':unittest.main()
