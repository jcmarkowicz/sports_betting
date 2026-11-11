import unittest
import numpy as np
import pandas as pd
from ufc_betting.Results.live_scaling import scaled_live_paths, reconstruct_scaled_events
from ufc_betting.BettingStrategy.Scaling import apply_event_filters, ScalingConfig


class LiveScalingTests(unittest.TestCase):
    def source(self, trends, vol, brier):
        dates=pd.date_range('2026-01-01',periods=len(trends))
        return dict(features=pd.DataFrame({'trend':trends},index=dates),
            scalers=pd.DataFrame({'volatility_multiplier':vol,'brier_multiplier':brier},index=dates))

    def test_crossed_closing_scalers_use_open_gate(self):
        opening=self.source(['Flat','Uptrend'],[.5,.5],[.8,.8])
        closing=self.source(['Uptrend','Flat'],[.2,.2],[.3,.3])
        events=pd.DataFrame({'bankroll_pct_change':[-.5,.5]},index=opening['features'].index)
        result=scaled_live_paths(events,opening,closing)
        self.assertEqual(len(result['multipliers'].columns),8)
        np.testing.assert_allclose(result['multipliers']['Combined | gate open, scalers close2'],[0,.06])
        self.assertEqual(result['winner'],'Flat only | gate open')
        self.assertAlmostEqual(result['wealth'].iloc[-1][result['winner']],1.5)

    def test_recompound_excludes_deposits_and_components_add(self):
        dates=pd.date_range('2026-01-01',periods=2)
        events=pd.DataFrame(dict(date=dates,bankroll_pct_change=[.1,-.2],
            bankroll_postevent=[550.,800.],event_payout_money_line=[30.,-150.],parlay_net=[20.,-50.]))
        out=reconstruct_scaled_events(events,pd.Series([.5,.25],index=dates),500)
        np.testing.assert_allclose(out.bankroll_postevent,[525.,498.75])
        np.testing.assert_allclose(out.event_payout_money_line+out.parlay_net,[25.,-26.25])

    def test_total_loss_can_seed_scalers(self):
        history=pd.DataFrame(dict(date=pd.date_range('2026-01-01',periods=2),trend=['Uptrend']*2,
            regime=['Above mean']*2,prior_brier=[.1,.2],return_pct=[-100.,20.]))
        target=pd.DataFrame(dict(date=['2026-01-03'],trend=['Uptrend'],regime=['Above mean'],prior_brier=[.15]))
        out=apply_event_filters(target,history,config=ScalingConfig(volatility_events=2))
        self.assertTrue(out.volatility_ready.iloc[0])

if __name__=='__main__':unittest.main()
