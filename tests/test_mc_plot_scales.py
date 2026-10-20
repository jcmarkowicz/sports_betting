import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ufc_betting.BettingStrategy.MonteCarlo.mc_plots import (
    plot_bankroll_paths, plot_terminal_return_distributions,
    plot_terminal_return_violin, plot_max_drawdown_violin,
)


class PlotScaleTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_bankroll_scale_retains_zero_and_negative_balances(self):
        results = {'positive': (None, None, [[100, 200]]),
                   'ruin': (None, None, [[100, 0], [100, -20]])}
        _, axes = plot_bankroll_paths(results)
        self.assertEqual(axes[0].get_yscale(), 'log')
        self.assertEqual(axes[1].get_yscale(), 'symlog')
        np.testing.assert_array_equal(axes[1].lines[1].get_ydata(), [100, -20])

    def test_distribution_scales_and_drawdown_violin(self):
        results = {'sample': (None, None, [[100, 80, 120], [100, 0, 0], [100, 1000, 100]])}
        fig, axes = plot_terminal_return_distributions(results)
        self.assertTrue(all(ax.get_xscale() == 'symlog' for ax in axes.flat))
        self.assertEqual(sum(p.get_height() for p in axes[0, 0].patches), 3)
        fig.canvas.draw()
        for plot in (plot_terminal_return_violin, plot_max_drawdown_violin):
            fig, ax = plot(results)
            self.assertEqual(ax.get_yscale(), 'linear' if plot is plot_max_drawdown_violin else 'symlog')
            fig.canvas.draw()
        self.assertIn('%', ax.yaxis.get_major_formatter()(50))
        body = ax.collections[0].get_paths()[0].vertices[:, 1]
        self.assertAlmostEqual(body.min(), 20)
        self.assertAlmostEqual(body.max(), 100)

    def test_constant_drawdowns_render_without_density(self):
        fig, ax = plot_max_drawdown_violin({'flat': (None, None, [[100, 100]])})
        fig.canvas.draw()
        self.assertEqual(ax.get_ylim()[0], 0)


if __name__ == '__main__':
    unittest.main()
