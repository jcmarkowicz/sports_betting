import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import joblib
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import simulate
from ufc_betting.BettingStrategy.MonteCarlo.price_sampling import load_density_models, ConditionalPriceSampler


PREDICT='ufc_betting.Models.RiskManagement.open_model_close1_input.opening_model_at_close1'
KELLY='ufc_betting.BettingStrategy.Backtest.backtest_functions.simulate_kelly'


def predict(frame,prices,model,scaler):
    result=frame.copy()
    result[['dec_close1_red','dec_close1_blue']]=prices
    result['proba_red']=.6;result['proba_blue']=.4
    result['pred_winner']=1;result['proba_se']=.01
    result['dec_fair_close1_red']=2.;result['dec_fair_close1_blue']=2.
    return result


def kelly(frame,**kwargs):
    result=frame.copy()
    # Deterministic unit-bankroll event return driven by fresh execution prices.
    value=.01*result.dec_close1_red.sum()
    result['event_payout_money_line']=value
    result['parlay_net']=-.005
    return result,pd.DataFrame()


class McPriceTests(unittest.TestCase):
    def test_decay_requires_complete_saved_policies_before_sampling(self):
        from ufc_betting.BettingStrategy.betting_params import BettingParams
        params = BettingParams()
        with patch('ufc_betting.BettingStrategy.MonteCarlo.mc_engine._event_paths',
                   side_effect=AssertionError('Invalid policies must fail before sampling')):
            with self.assertRaisesRegex(ValueError, 'decay_rates and opening_params'):
                simulate(self.fights, closing_params=dict(close1=params, close2=params), **self.args)
            partial = params.with_decay(dict(k_down=1, k_up=1), params)
            with self.assertRaisesRegex(ValueError, 'all four saved decay rates'):
                simulate(self.fights, closing_params=dict(close1=partial, close2=partial), **self.args)

    @patch(PREDICT, side_effect=predict)
    def test_saved_joint_json_uses_shared_calculator_and_toggle(self, pred):
        from ufc_betting.BettingStrategy.betting_params import BettingParams
        from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator
        opening = BettingParams(z=.05)
        params = BettingParams(z=.6).with_decay(
            dict(k_down=2, k_up=3, k_parlay_down=4, k_parlay_up=5), opening)
        with tempfile.TemporaryDirectory() as directory:
            paths = {stage: Path(directory) / f'{stage}.json' for stage in ('close1', 'close2')}
            for path in paths.values():
                params.save(path)
            for use_decay in (True, False):
                *_, details = simulate(self.fights, closing_params=paths, use_decay=use_decay, **self.args)
                for run in details:
                    for date, group in run['inputs'].groupby('date'):
                        expected, _ = Calculator.calculate_market_event_returns(
                            group, dict(close1=params, close2=params), use_decay=use_decay)
                        actual = run['kelly'].loc[run['kelly'].date.eq(date)]
                        np.testing.assert_allclose(actual.choice_fstar, expected.choice_fstar)
                        self.assertLessEqual(actual.event_total_fstar.max(), 1 + 1e-12)

    @patch(KELLY, side_effect=kelly)
    @patch(PREDICT, side_effect=predict)
    def test_legacy_decay_toggle_removes_decay_controls(self, pred, backtest):
        simulate(self.fights, use_decay=False,
                 kelly_kwargs=dict(decay_rates=dict(k_down=1, k_up=1), opening_params={}), **self.args)
        for call in backtest.call_args_list:
            self.assertNotIn('decay_rates', call.kwargs)
            self.assertNotIn('opening_params', call.kwargs)

    @patch(PREDICT, side_effect=predict)
    def test_batch_bankruptcy_stops_later_monte_carlo_profits(self, pred):
        def losing_bets(frame, **kwargs):
            return frame.assign(event_payout_money_line=-.4, parlay_net=-.2), pd.DataFrame()

        with patch(KELLY, side_effect=losing_bets):
            moneylines, parlays, bankroll, _ = simulate(
                self.fights, 'close1', len_temp=2, **self.args
            )
        np.testing.assert_allclose(bankroll, [[500, -100, -100]] * 2)
        np.testing.assert_allclose(moneylines, [[-400, 0]] * 2)
        np.testing.assert_allclose(parlays, [[-200, 0]] * 2)

    def setUp(self):
        self.fights=pd.DataFrame([dict(date=date,fighter_red=f'r{i}',fighter_blue=f'b{i}',
            winner=1,title_fight=title,event_type='numbered' if title else 'fight_night',dec_open_red=2.,dec_open_blue=2.2,
            dec_close1_red=1.6,dec_close1_blue=2.6,dec_close2_red=1.8,dec_close2_blue=2.4)
            for i,(date,title) in enumerate([('2025-01-04',1),('2025-01-18',1),
                                           ('2025-02-01',0),('2025-02-15',0)])])
        self.args=dict(n_sims=2,random_state=12,model=object(),scaler=object(),return_details=True)

    @patch(KELLY,side_effect=kelly)
    @patch(PREDICT,side_effect=predict)
    def test_saved_paths_replay_without_sampling_and_use_current_prices(self,pred,sim):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'paths.json'
            saved=simulate(self.fights,'close1',
                           event_paths_mode='save',event_paths_file=path,**self.args)
            with patch('ufc_betting.BettingStrategy.MonteCarlo.mc_engine.sample_schedule',
                       side_effect=AssertionError('Must not regenerate saved schedules')):
                loaded=simulate(self.fights,'close1',
                                event_paths_mode='load',event_paths_file=path,**self.args)
                for a,b in zip(saved[3],loaded[3]):
                    pd.testing.assert_frame_equal(a['inputs'],b['inputs'])
                changed=self.fights.iloc[::-1].copy()
                changed['dec_close2_red']=3.
                replay=simulate(changed,'close2',event_paths_mode='load',event_paths_file=path,**self.args)
                for a,b in zip(saved[3],replay[3]):
                    self.assertEqual(a['inputs'].source_date.tolist(),b['inputs'].source_date.tolist())
                    self.assertTrue(b['inputs'].dec_close1_red.eq(3.).all())
            with self.assertRaisesRegex(ValueError,'match'):
                simulate(self.fights.iloc[:-1],'close1',event_paths_mode='load',event_paths_file=path,**self.args)
            with self.assertRaisesRegex(ValueError,'requested'):
                simulate(self.fights,'close1',event_paths_mode='load',event_paths_file=path,
                         **{**self.args,'n_sims':3})
            changed = self.fights.copy()
            changed.loc[0, 'event_type'] = 'fight_night'
            with self.assertRaisesRegex(ValueError, 'event types'):
                simulate(changed, 'close1', event_paths_mode='load', event_paths_file=path, **self.args)

    @patch(KELLY,side_effect=kelly)
    @patch(PREDICT,side_effect=predict)
    def test_replacement_varies_terminal_returns_and_replays(self,pred,sim):
        fights=self.fights.copy()
        fights['dec_close1_red']=[1.5,2.,3.,4.]
        args={**self.args,'n_sims':12,'sample_with_replacement':True}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bootstrap.json'
            a=simulate(fights,'close1',event_paths_mode='save',event_paths_file=path,**args)
            b=simulate(fights,'close1',event_paths_mode='load',event_paths_file=path,**args)
            np.testing.assert_allclose(a[2],b[2])
            self.assertGreater(np.ptp(np.asarray(a[2])[:,-1]),0)
            self.assertTrue(any(d['inputs'].source_date.nunique()<4 for d in a[3]))
            for detail in a[3]:
                self.assertEqual(detail['inputs'].date.nunique(),4)
            with self.assertRaisesRegex(ValueError,'sampling mode'):
                simulate(fights,'close1',event_paths_mode='load',event_paths_file=path,**self.args)

    @patch(KELLY,side_effect=kelly)
    @patch(PREDICT,side_effect=predict)
    def test_recorded_prices_schedule_and_partial_batch(self,pred,sim):
        for stage,price in [('open',2.),('close1',1.6),('close2',1.8)]:
            result=simulate(self.fights,stage,len_temp=3,**self.args)
            self.assertEqual(len(result[2][0]),3)  # initial + three-card batch + final card
            for detail in result[3]:
                self.assertTrue((detail['inputs'].dec_close1_red==price).all())
                self.assertEqual(len(detail['inputs'].source_date.unique()),4)
                self.assertTrue((detail['inputs'].groupby(detail['inputs'].date.dt.to_period('M')).title_fight.sum()>=1).all())
            expected=500*(1+3*(.01*price-.005))*(1+.01*price-.005)
            self.assertAlmostEqual(result[2][0][-1],expected)
        self.assertGreater(sim.call_count,0)

    @staticmethod
    def density_fit(day=5, stage='close1'):
        model = KDEMultivariateConditional(
            endog=np.log(np.array([1.5, 1.8, 2.1, 2.5, 2.8]) - 1),
            exog=np.array([[0., -.5], [0., -.2], [.2, .2], [.2, .4], [.4, .6]]),
            dep_type='c', indep_type='cc', bw=[.2, .3, .3])
        return dict(model=model, days_out=day, odds_type=stage, split='test')

    @patch(KELLY, side_effect=kelly)
    @patch(PREDICT, side_effect=predict)
    def test_intermediate_saved_models_replay_and_recompute_returns(self, pred, sim):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'saved_models').mkdir()
            for stage in ('close1', 'close2'):
                joblib.dump(self.density_fit(5, stage), root / f'saved_models/conditional_price_density_{stage}_5_days_test.joblib')
            kwargs = dict(data_dir=root, event_paths_file=root / 'paths.json', **self.args)
            a = simulate(self.fights, 'intermediate', event_paths_mode='save', **kwargs)
            b = simulate(self.fights, 'intermediate', event_paths_mode='load', **kwargs)
            np.testing.assert_allclose(a[2], b[2])
            for left, right in zip(a[3], b[3]):
                data = left['inputs']
                pd.testing.assert_frame_equal(data, right['inputs'])
                self.assertTrue(data.density_days_out.eq(5).all())
                self.assertTrue(data.sampled_endpoint_red.between(1.6, 1.8).all())
                self.assertTrue(data.sampled_endpoint_blue.between(2.4, 2.6).all())
                self.assertTrue((data.dec_close1_red > 1).all())
                self.assertFalse(data.dec_close1_red.eq(1.6).all())
                expected = .01 * data.groupby('date').dec_close1_red.sum().to_numpy()
                np.testing.assert_allclose(left['event_returns'].moneyline_return, expected)
            self.assertFalse(np.array_equal(a[3][0]['inputs'].dec_close1_red,
                                            a[3][1]['inputs'].dec_close1_red))

    @patch(KELLY, side_effect=kelly)
    @patch(PREDICT, side_effect=predict)
    def test_intermediate_multiple_days_share_event_day(self, pred, sim):
        fits = {day: {stage: self.density_fit(day, stage) for stage in ('close1', 'close2')} for day in (0, 6)}
        fights = pd.concat([self.fights, self.fights.assign(fighter_red='extra', fighter_blue='other')], ignore_index=True)
        result = simulate(fights, 'intermediate', days_out=[0, 6],
                          density_models=fits, **{**self.args, 'n_sims': 8})
        for detail in result[3]:
            data = detail['inputs']
            self.assertTrue(data.groupby('sampled_event_id').density_days_out.nunique().eq(1).all())
            self.assertTrue(data.sampled_endpoint_red.between(1.6, 1.8).all())
            self.assertTrue(data.sampled_endpoint_blue.between(2.4, 2.6).all())
        self.assertEqual(set(pd.concat([d['inputs'] for d in result[3]]).density_days_out), {0, 6})

    def test_density_loading_requires_both_markets_and_validates_metadata(self):
        fits = {stage: self.density_fit(5, stage) for stage in ('close1', 'close2')}
        loaded = load_density_models('.', 5, fits)
        self.assertEqual(set(loaded), {5})
        self.assertEqual(set(loaded[5]), {'close1', 'close2'})
        for stage in fits:
            self.assertIs(loaded[5][stage], fits[stage])
        with self.assertRaisesRegex(ValueError, 'both close1 and close2'):
            load_density_models('.', 5, {'close1': fits['close1']})
        with self.assertRaisesRegex(ValueError, 'refit'):
            load_density_models('.', 5, {'close1': {'training': {}}, 'close2': fits['close2']})
        with self.assertRaisesRegex(ValueError, 'metadata'):
            load_density_models('.', 6, fits)
        for days in (True, -1, [], [5, 5], [5.5]):
            with self.assertRaises(ValueError):
                load_density_models('.', days, fits)

    def test_nearest_selects_model_and_conditions_on_sampled_endpoint(self):
        fits = {5: {stage: self.density_fit(5, stage) for stage in ('close1', 'close2')}}
        models = load_density_models('.', [5], fits)
        sampler = ConditionalPriceSampler(models, random_state=42)
        frame = self.fights.copy().assign(sampled_event_id=np.arange(len(self.fights)))
        # Include equal prices and an exact midpoint to exercise close1 ties.
        frame.loc[2, 'dec_close2_red'] = frame.loc[2, 'dec_close1_red']
        with patch.object(sampler, 'rng') as rng, patch(
                'ufc_betting.BettingStrategy.MonteCarlo.price_sampling.sample_prices',
                return_value=np.array([2.1])) as draw:
            rng.choice.return_value = 5
            rng.random.return_value = np.array([.25, .75, .9, .5])
            prices = sampler.sample(frame)
        np.testing.assert_allclose(prices, 2.1)
        self.assertEqual(frame.density_source_red.tolist(), ['close1', 'close2', 'close1', 'close1'])
        self.assertEqual(frame.density_source_blue.tolist(), ['close1', 'close2', 'close2', 'close1'])
        for side_index, side in enumerate(('red', 'blue')):
            for row in range(len(frame)):
                source = frame.iloc[row][f'density_source_{side}']
                call = draw.call_args_list[side_index * len(frame) + row]
                self.assertIs(call.args[0], fits[5][source]['model'])
                self.assertEqual(call.args[1], frame.iloc[row][f'dec_open_{side}'])
                self.assertEqual(call.args[2], frame.iloc[row][f'sampled_endpoint_{side}'])

    def test_load_density_models_uses_folder_and_rejects_ambiguous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for day in (5, 6, 7):
                for stage in ('close1', 'close2'):
                    fit = self.density_fit(day, stage)
                    fit['split'] = 'validation'
                    joblib.dump(fit, folder / f'conditional_price_density_{stage}_{day}_days_validation.joblib')
            loaded = load_density_models(folder, [5, 6, 7])
            self.assertEqual(set(loaded), {5, 6, 7})
            self.assertTrue(all(set(value) == {'close1', 'close2'} for value in loaded.values()))
            with self.assertRaises(FileNotFoundError):
                load_density_models(folder, 8)
            joblib.dump(self.density_fit(), folder / 'conditional_price_density_close1_5_days.joblib')
            with self.assertRaisesRegex(ValueError, 'Multiple close1'):
                load_density_models(folder, 5)

    @patch(KELLY, side_effect=kelly)
    @patch(PREDICT, side_effect=predict)
    def test_nearest_default_runs_both_markets_and_replays(self, pred, sim):
        fits = {day: {stage: self.density_fit(day, stage) for stage in ('close1', 'close2')}
                for day in (5, 6, 7)}
        kwargs = dict(days_out=[5, 6, 7], density_models=fits, **self.args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paths.json'
            a = simulate(self.fights, 'intermediate', event_paths_mode='save', event_paths_file=path, **kwargs)
            b = simulate(self.fights, 'intermediate', event_paths_mode='load', event_paths_file=path, **kwargs)
        np.testing.assert_allclose(a[2], b[2])
        for left, right in zip(a[3], b[3]):
            pd.testing.assert_frame_equal(left['inputs'], right['inputs'])
            for side in ('red', 'blue'):
                data = left['inputs']
                expected = np.where(data.endpoint_fraction < .5, 'close1', 'close2')
                np.testing.assert_array_equal(data[f'density_source_{side}'], expected)


if __name__=='__main__':unittest.main()
