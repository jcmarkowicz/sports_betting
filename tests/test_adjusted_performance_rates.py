import unittest

import numpy as np
import pandas as pd

from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.rolling_stats import apply_rolling_stats


def fight_history():
    rows = []
    for i in range(5):
        row = dict(date=pd.Timestamp('2020-01-01') + pd.Timedelta(days=30 * i),
                   fighter_red='A', fighter_blue='B', winner='A',
                   weight_class='Lightweight', event_location='USA', title_fight=0,
                   performance_bonus_winner=0, fight_otn_bonus=0, method='U-DEC',
                   fight_minutes=10.)
        for color, offset in [('red', 0), ('blue', 1)]:
            for feature in ['kd', 'leg_str', 'head_str', 'body_str', 'clinch_str',
                            'td_landed', 'td_attempted', 'td_defended', 'td_absorbed',
                            'sub_att', 'reverse']:
                row[f'{feature}_{color}'] = 1.
            row[f'sig_str_landed_{color}'] = 20. + 7 * i + offset * 10
            row[f'sig_str_attempted_{color}'] = 100.
            row[f'control_{color}'] = 1. + .3 * i + offset
            for feature in ['age', 'height', 'reach']:
                row[f'{feature}_{color}'] = 30.
        row['sig_str_absorbed_red'] = row['sig_str_landed_blue']
        row['sig_str_absorbed_blue'] = row['sig_str_landed_red']
        rows.append(row)
    return pd.DataFrame(rows)


class AdjustedRateTests(unittest.TestCase):
    columns = ['adjusted_sig_str_red', 'adjusted_sig_str_blue',
               'adjusted_control_red', 'adjusted_control_blue']

    def test_equal_rates_with_different_durations_produce_equal_features(self):
        original = fight_history()
        scaled = original.copy()
        columns = ['fight_minutes'] + [f'{feature}_{color}'
            for feature in ['sig_str_landed', 'sig_str_absorbed', 'control']
            for color in ['red', 'blue']]
        scaled.loc[2, columns] *= 2
        expected = apply_rolling_stats(original)[self.columns].astype(float)
        actual = apply_rolling_stats(scaled)[self.columns].astype(float)
        self.assertTrue(expected.iloc[2:].notna().all().all())
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_current_bout_does_not_change_its_prefight_features(self):
        original = fight_history()
        changed = original.copy()
        changed.loc[3, 'sig_str_landed_red'] = 90.
        changed.loc[3, 'sig_str_absorbed_blue'] = 90.
        changed.loc[3, 'control_red'] = 8.
        expected = apply_rolling_stats(original)[self.columns].astype(float)
        actual = apply_rolling_stats(changed)[self.columns].astype(float)
        np.testing.assert_allclose(actual.iloc[:4], expected.iloc[:4], equal_nan=True)
        self.assertFalse(np.allclose(actual.iloc[4], expected.iloc[4]))


if __name__ == '__main__':
    unittest.main()
