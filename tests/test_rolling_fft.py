import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Notebooks' / 'Bets'))
from rolling_fft import rolling_fft_features, fit_fourier_window, rolling_fourier_features

class RollingFFTTests(unittest.TestCase):
    def test_full_fourier_reconstruction_and_derivative(self):
        t = np.arange(128) - 63.5
        wave = np.cos(2*np.pi*t/32) + .3*np.cos(2*np.pi*t/8)
        residual, fitted, velocity, _ = fit_fourier_window(wave + .01*t)
        np.testing.assert_allclose(fitted, residual, atol=1e-12)
        expected = -(2*np.pi/32)*np.sin(2*np.pi*t/32) - .3*(2*np.pi/8)*np.sin(2*np.pi*t/8)
        np.testing.assert_allclose(velocity, expected, atol=1e-12)

    def test_full_fourier_history_only(self):
        series = pd.Series(np.random.default_rng(42).normal(size=220), index=pd.date_range('2020-01-01',periods=220))
        full, _ = rolling_fourier_features(series)
        prefix, _ = rolling_fourier_features(series.iloc[:180])
        pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)

    def test_prefix_invariance_and_sinusoid_direction(self):
        t=np.arange(240)
        series=pd.Series(.02*np.sin(2*np.pi*t/32),index=pd.date_range('2020-01-01',periods=len(t)))
        features,_=rolling_fft_features(series,components=1)
        early,_=rolling_fft_features(series.iloc[:190],components=1)
        pd.testing.assert_frame_equal(features.loc[early.index],early)
        expected=.02*(2*np.pi/32)*np.cos(2*np.pi*t[127:]/32)
        self.assertGreater(np.corrcoef(features.velocity,expected)[0,1],.98)
        self.assertTrue(features.dominant_period.eq(32).all())

    def test_multiple_components_and_zero_signal(self):
        t=np.arange(256)
        y=pd.Series(np.sin(2*np.pi*t/32)+.4*np.cos(2*np.pi*t/16),index=pd.date_range('2020-01-01',periods=256))
        features,power=rolling_fft_features(y,components=2)
        expected=(2*np.pi/32)*np.cos(2*np.pi*t[127:]/32)-.4*(2*np.pi/16)*np.sin(2*np.pi*t[127:]/16)
        self.assertGreater(np.corrcoef(features.velocity,expected)[0,1],.98)
        zero,_=rolling_fft_features(y*0)
        np.testing.assert_allclose(zero.velocity,0,atol=1e-12)
