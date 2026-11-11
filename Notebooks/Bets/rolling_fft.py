"""History-only rolling Fourier decomposition of an event-indexed series."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def fit_fourier_window(values, harmonics=None):
    """Fit a Fourier series to one linearly detrended historical window.

    None retains the full spectrum (interpolation, not denoising). The derivative
    is that of the periodic interpolant; endpoint wraparound can affect it.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 4 or not np.isfinite(values).all():
        raise ValueError('Require at least four finite observations')
    t = np.arange(n)
    X = np.column_stack([np.ones(n), t])
    residual = values - X @ np.linalg.lstsq(X, values, rcond=None)[0]
    coef = np.fft.rfft(residual)
    if harmonics is not None:
        if not 1 <= harmonics <= n // 2:
            raise ValueError('Invalid harmonic count')
        coef[harmonics + 1:] = 0
    omega = 2 * np.pi * np.fft.rfftfreq(n)
    derivative_coef = 1j * omega * coef
    if n % 2 == 0:
        derivative_coef[-1] = 0  # Nyquist cosine derivative is zero at samples.
    fitted = np.fft.irfft(coef, n=n)
    velocity = np.fft.irfft(derivative_coef, n=n)
    return residual, fitted, velocity, np.abs(coef[1:]) ** 2 / n


def rolling_fourier_features(series, *, window=128, harmonics=None):
    """Rolling Fourier fits retaining all frequencies unless explicitly truncated."""
    series = series.sort_index()
    if not series.index.is_unique or len(series) < window:
        raise ValueError('Require unique dates and sufficient historical observations')
    rows, spectra = [], []
    for stop in range(window, len(series) + 1):
        values = series.iloc[stop-window:stop].to_numpy(float)
        residual, fitted, velocity, power = fit_fourier_window(values, harmonics)
        rows.append(dict(date=series.index[stop-1], raw_value=values[-1],
                         detrended_value=residual[-1], oscillation=fitted[-1],
                         velocity=velocity[-1], fit_rmse=np.sqrt(np.mean((residual-fitted)**2))))
        spectra.append(power)
    features = pd.DataFrame(rows).set_index('date')
    spectrum = pd.DataFrame(spectra, index=features.index,
                            columns=1 / np.fft.rfftfreq(window)[1:])
    spectrum.columns.name = 'period_events'
    return features, spectrum


def rolling_fft_features(series, *, window=128, min_period=8, max_period=64, components=3):
    """FFT selects frequencies; joint harmonic regression estimates endpoint motion.

    Period and velocity units are events. Each trailing window is linearly
    detrended and Hann-tapered for frequency selection. Harmonic coefficients are
    fitted to untapered detrended data to avoid dividing by a zero endpoint taper.
    Returned values use observations through their index date, never later data.
    """
    series = series.sort_index()
    if not series.index.is_unique or not np.isfinite(series.to_numpy()).all():
        raise ValueError('Require unique dates and finite observations')
    if window < 16 or not 2 <= min_period < max_period <= window / 2 or components < 1:
        raise ValueError('Invalid FFT window, periods, or component count')
    if len(series) < window:
        raise ValueError('Insufficient events for FFT window')
    t = np.arange(window, dtype=float)
    design_trend = np.column_stack([np.ones(window), t])
    frequencies = np.fft.rfftfreq(window)
    eligible = np.flatnonzero((frequencies >= 1/max_period) & (frequencies <= 1/min_period))
    rows, powers = [], []
    taper = np.hanning(window)
    for stop in range(window, len(series)+1):
        values = series.iloc[stop-window:stop].to_numpy(float)
        trend = design_trend @ np.linalg.lstsq(design_trend, values, rcond=None)[0]
        residual = values-trend
        power = np.abs(np.fft.rfft(residual*taper))**2 / np.sum(taper**2)
        peaks = find_peaks(power)[0]
        candidates = [j for j in peaks if j in eligible]
        # Include band edges if they are maxima inside the permitted band.
        for j in (eligible[0], eligible[-1]):
            if j not in candidates and power[j] >= max(power[k] for k in eligible if abs(k-j)<=1):
                candidates.append(j)
        chosen=[]
        for j in sorted(candidates,key=lambda j:power[j],reverse=True):
            if all(abs(j-k)>=2 for k in chosen):chosen.append(j)
            if len(chosen)==components:break
        row=dict(date=series.index[stop-1],raw_value=values[-1],
                 detrended_value=residual[-1],oscillation=0.,velocity=0.,amplitude=0.,dominant_period=np.nan)
        if chosen:
            omega=2*np.pi*frequencies[chosen]
            X=np.column_stack([part for w in omega for part in (np.cos(w*t),np.sin(w*t))])
            coef=np.linalg.lstsq(X,residual,rcond=None)[0]
            a,b=coef[::2],coef[1::2]
            phase=omega*t[-1]
            row.update(oscillation=float(np.sum(a*np.cos(phase)+b*np.sin(phase))),
                       velocity=float(np.sum(omega*(-a*np.sin(phase)+b*np.cos(phase)))),
                       amplitude=float(np.sqrt(np.sum(a*a+b*b))),
                       dominant_period=float(1/frequencies[chosen[0]]))
        rows.append(row);powers.append(power[eligible])
    features=pd.DataFrame(rows).set_index('date')
    spectrum=pd.DataFrame(powers,index=features.index,columns=1/frequencies[eligible])
    spectrum.columns.name='period_events'
    return features,spectrum
