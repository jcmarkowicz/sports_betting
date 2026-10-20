import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import test_joint_kelly_decay_search as fixtures
from ufc_betting.BettingStrategy.MonteCarlo.compact_paths import CompactPaths
from ufc_betting.BettingStrategy.MonteCarlo.ordered_sampling import generate_ordered_paths
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import load_sampled_paths
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.Models.Search.frozen_betting import FrozenBettingSearch, JointDecayBettingSearch


class CompactPathTests(unittest.TestCase):
    def frame(self):
        return fixtures.JointKellyDecayTests().frame().assign(bout_position=[1, 2] * 3,
                                                    event_name=['UFC 300'] * 2 + ['UFC 301'] * 2 + ['UFC 302'] * 2)

    def test_roundtrip_modes_and_selective_loading(self):
        for sample_fights in (True, False):
            with self.subTest(sample_fights=sample_fights), tempfile.TemporaryDirectory() as tmp:
                args = dict(n_paths=4, odds_type=None, sample_fights=sample_fights,
                            strict_numbered_months=True, progress_every=0)
                expected = generate_ordered_paths(self.frame(), **args)
                directory = Path(tmp) / 'paths'
                actual = generate_ordered_paths(self.frame(), output_path=directory,
                                               storage='compact', **args)
                for a, b in zip(expected, actual):
                    pd.testing.assert_frame_equal(a, b)
                calls = []
                original = CompactPaths.__getitem__
                def get(instance, index):
                    calls.append(index)
                    return original(instance, index)
                with patch.object(CompactPaths, '__getitem__', get):
                    chosen = load_sampled_paths(directory, n_paths=2, random_state=42)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(chosen), 2)
                if not sample_fights:
                    for frame in actual:
                        self.assertEqual(sorted(frame.source_row_position), list(range(6)))
                        self.assertTrue(frame.groupby('sampled_event_id').source_date.nunique().eq(1).all())
                        cards = frame[['date', 'event_name']].drop_duplicates()
                        self.assertTrue(cards.groupby(cards.date.dt.to_period('M')).size().eq(1).all())
                actual.close()
                with self.assertRaises(FileExistsError):
                    generate_ordered_paths(self.frame(), output_path=directory, storage='compact', **args)

    def test_sampled_price_overrides_roundtrip(self):
        class Sampler:
            def __init__(self, *args, **kwargs):
                self.rng = np.random.default_rng(9)
            def sample(self, frame):
                return self.rng.uniform(1.5, 3., (len(frame), 2))
        with tempfile.TemporaryDirectory() as tmp, patch(
                'ufc_betting.BettingStrategy.MonteCarlo.price_sampling.load_density_models', return_value={}), patch(
                'ufc_betting.BettingStrategy.MonteCarlo.price_sampling.ConditionalPriceSampler', Sampler):
            args = dict(n_paths=3, sample_prices=True, odds_type=None, progress_every=0)
            expected = generate_ordered_paths(self.frame(), **args)
            actual = generate_ordered_paths(self.frame(), storage='compact', output_path=Path(tmp)/'paths', **args)
            for a, b in zip(expected, actual):
                pd.testing.assert_frame_equal(a, b)
            actual.close()

    def test_impossible_monthly_constraint_is_explicit(self):
        frame = self.frame().assign(event_name='UFC Fight Night')
        with self.assertRaisesRegex(ValueError, 'Exactly one numbered'):
            generate_ordered_paths(frame, strict_numbered_months=True, sample_fights=False)

    def test_shared_cache_matches_card_evaluation_with_price_variants(self):
        frame = self.frame()
        paths = generate_ordered_paths(frame, n_paths=4, odds_type=None, progress_every=0)
        # Identical donor identity, different execution prices must not be merged.
        paths[1]['dec_open_red'] *= 1.1
        paths[1]['dec_close1_red'] *= 1.1
        base = dict(mdd=.4, N=250, mdd_parlay=.35, N_parlay=300, z=.2,
                    moneyline_min_american=-400)
        for cls, market in ((FrozenBettingSearch, 'open'), (JointDecayBettingSearch, 'close1')):
            extra = {} if market == 'open' else dict(opening_params=BettingParams())
            params = dict(base)
            if market != 'open':
                params.update({f'log1p_{name}': .8 for name in cls.rate_names})
            search = cls(frame, market, sampled_paths=paths, **extra)
            result = search.simulate(params)
            expected = []
            for path in paths:
                child = cls(path, market, k=2, _fixed_path_indices=np.arange(3)[None], **extra)
                expected.append(child.simulate(params)['bankroll'][0])
            np.testing.assert_allclose(result['bankroll'], expected, rtol=1e-12)
            with patch.object(BettingReturnCalculator, 'calculate_scenario_cache',
                              wraps=BettingReturnCalculator.calculate_scenario_cache) as calc:
                search.simulate(params)
                self.assertEqual(calc.call_count, 1)


if __name__ == '__main__':
    unittest.main()
