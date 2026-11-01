import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ufc_betting.Results.live_plotly import to_plotly


class LivePlotlyTests(unittest.TestCase):
    def test_values_bins_secondary_axis_and_annotations_are_preserved(self):
        source, axes = plt.subplots(2, 2)
        axes[0, 0].plot([1, 2, 3], [10, 15, 12], label='Bankroll')
        counts, edges, _ = axes[0, 1].hist([0, 1, 1, 2, 3], bins=[0, 1, 2, 4])
        axes[1, 0].axvline(1.5, label='Mean')
        axes[1, 0].text(.5, .5, 'Milestone', transform=axes[1, 0].transAxes)
        twin = axes[1, 1].twinx()
        twin.plot([1, 2], [.01, .03], label='Growth')
        converted = to_plotly(source)
        np.testing.assert_allclose(converted.data[0].y, [10, 15, 12])
        bars = next(t for t in converted.data if t.type == 'bar')
        np.testing.assert_allclose(bars.y, counts)
        np.testing.assert_allclose(bars.width, np.diff(edges))
        self.assertEqual(converted.layout.yaxis5.side, 'right')
        self.assertEqual(converted.layout.shapes[0].x0, 1.5)
        self.assertIn('Milestone', [a.text for a in converted.layout.annotations])
        converted.to_json()
        plt.close(source)
