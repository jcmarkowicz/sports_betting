import unittest
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.Scaling import (
    ScalingConfig, calculate_brier_scores, calculate_event_history,
    calculate_event_trends, apply_event_filters,
)


class EventScalingTests(unittest.TestCase):
    def test_brier_order_and_event_windows(self):
        fights = pd.DataFrame(dict(date=['2020-01-01']*2+['2020-01-02'],
            winner=[1,0,1],proba_red=[.8,.6,.7],dec_open_red=[2.]*3,dec_open_blue=[2.]*3))
        config=ScalingConfig(fight_window=2,slope_events=2)
        scored=calculate_brier_scores(fights,config=config)
        np.testing.assert_allclose(scored.advantage,[.21,-.11,.16])
        np.testing.assert_allclose(scored.rolling_advantage.iloc[1:],[.05,.025])
        history=calculate_event_history(scored,config=config)
        np.testing.assert_allclose(history.advantage,[.05,.16])
        np.testing.assert_allclose(history.rolling_advantage,[.05,.025])

    def test_current_future_outcomes_do_not_change_features(self):
        history=pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=4),
            rolling_advantage=[.1,.2,.3,.4],advantage_slope=[.01,.03,-.04,.02],
            vegas_actual=[.2,.3,.4,.5]))
        events=pd.DataFrame({'date':['2020-01-03','2020-01-06']},index=[8,2])
        original=calculate_event_trends(events,history,trend_scale=1)
        changed=history.copy();changed.loc[2:,['rolling_advantage','advantage_slope','vegas_actual']]=99
        altered=calculate_event_trends(events,changed,trend_scale=1)
        pd.testing.assert_series_equal(original.iloc[0],altered.iloc[0])
        self.assertEqual(original.index.tolist(),[8,2])
        self.assertEqual(original.iloc[0].prior_brier,.3)
        self.assertEqual(original.iloc[0].trend,'Uptrend')
        self.assertEqual(original.iloc[1].trend,'Flat')

    def test_missing_history_is_unknown(self):
        history=pd.DataFrame(dict(date=pd.to_datetime(['2020-01-02']),rolling_advantage=[.1],
            advantage_slope=[.03],vegas_actual=[.2]))
        out=calculate_event_trends(pd.DataFrame({'date':['2020-01-01']}),history,trend_scale=1)
        self.assertEqual(out.trend.iloc[0],'Unknown')

    def test_flat_history_in_denominator_only_and_future_excluded(self):
        history=pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=4),
            trend=['Uptrend','Downtrend','Flat','Uptrend'],regime=['Above mean']*4,
            prior_brier=[.2]*4,return_pct=[10.,-10.,90.,999.]))
        target=pd.DataFrame(dict(date=['2020-01-04'],trend=['Uptrend'],regime=['Above mean'],prior_brier=[.2],stake_fraction=[.8]))
        cfg=ScalingConfig(volatility_events=2)
        out=apply_event_filters(target,history,strategy='volatility_only',config=cfg)
        self.assertAlmostEqual(out.stake_multiplier.iloc[0],.2)
        self.assertAlmostEqual(out.scaled_stake_fraction.iloc[0],.16)
        self.assertNotIn('scaled_return_pct',out)
        target['trend']='Flat';target['return_pct']=-50.
        self.assertEqual(apply_event_filters(target,history,config=cfg).scaled_return_pct.iloc[0],0.)
        self.assertEqual(apply_event_filters(target,history,strategy='raw',config=cfg).scaled_return_pct.iloc[0],-50.)

    def test_brier_downscale_and_target_returns_independence(self):
        history=pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=3),
            trend=['Uptrend']*3,regime=['Above mean']*3,prior_brier=[0.,1.,2.],return_pct=[0.,10.,20.]))
        target=pd.DataFrame(dict(date=['2020-01-04'],trend=['Uptrend'],regime=['Above mean'],prior_brier=[0.]))
        cfg=ScalingConfig(brier_min_events=3,brier_ridge=0)
        result=apply_event_filters(target,history,strategy='prior_brier_only',config=cfg)
        self.assertAlmostEqual(result.stake_multiplier.iloc[0],0.)
        target['return_pct']=1000.
        again=apply_event_filters(target,history,strategy='prior_brier_only',config=cfg)
        self.assertEqual(result.stake_multiplier.iloc[0],again.stake_multiplier.iloc[0])

    def test_duplicate_event_rejected(self):
        with self.assertRaises(ValueError):
            calculate_event_trends(pd.DataFrame({'date':['2020-01-01']*2}),pd.DataFrame(),trend_scale=1)


if __name__ == '__main__':
    unittest.main()
