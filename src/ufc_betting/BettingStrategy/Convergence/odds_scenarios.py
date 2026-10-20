"""Price interpolation and reproducible fight-level movement sampling."""
import numpy as np


def sample_prices(opening_prices, closing_prices, n_samples=50000, *,
                  random_state=None, levels=101):
    """Sample one event's prices between closing and opening quotes.

    Inputs: aligned (fights, 2) decimal prices, ordered red then blue.
    Returns prices shaped (scenarios, fights, 2) and movement_indices shaped
    (scenarios, fights). Index 0 is closing; levels - 1 is opening.
    Both sides of a fight share its sampled movement. random_state accepts
    a seed or Generator. The number of unique scenarios may be below n_samples.
    """
    opening = np.asarray(opening_prices, dtype=float)
    closing = np.asarray(closing_prices, dtype=float)
    if opening.ndim != 2 or opening.shape[1] != 2 or len(opening) == 0:
        raise ValueError('Prices must have shape (fights, 2), ordered red then blue')
    interpolate_prices(closing, opening, 0.)  # Shared price validation.
    for name, value in (('n_samples', n_samples), ('levels', levels)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
    indices = sample_movements(len(opening), n_samples,
                               np.random.default_rng(random_state), levels=levels)
    fractions = indices[:, :, None] / (levels - 1)
    prices = closing[None, :, :] + fractions * (opening - closing)[None, :, :]
    return dict(prices=prices, movement_indices=indices)


def interpolate_prices(closing, opening, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    closing, opening = np.asarray(closing, float), np.asarray(opening, float)
    if closing.shape != opening.shape or not np.isfinite([closing, opening]).all():
        raise ValueError("Prices must be finite and aligned")
    if (closing <= 1).any() or (opening <= 1).any():
        raise ValueError("Decimal odds must exceed 1")
    return closing + fraction * (opening - closing)


def sample_movements(n_fights, n_samples, rng, levels=101):
    """Mixture over movement centers and dispersion, rounded to the saved grid."""
    if n_fights < 1 or n_samples < levels or levels < 2:
        raise ValueError("Need fights and at least one sample per grid level")
    # Uniform reference vectors are included exactly once.
    vectors = np.repeat(np.arange(levels)[:, None], n_fights, axis=1)
    target = min(n_samples, levels ** n_fights)
    for _ in range(40):
        if len(vectors) >= target:
            break
        count = max(2 * (target - len(vectors)), 1000)
        mean = rng.uniform(.001, .999, (count, 1))
        concentration = rng.choice([2., 10., 50.], size=(count, 1))
        draws = rng.beta(mean * concentration, (1 - mean) * concentration,
                         size=(count, n_fights))
        draws = np.rint(draws * (levels - 1)).astype(int)
        candidates = np.unique(np.concatenate([vectors, draws]), axis=0)
        # Keep all reference vectors; randomly choose other unique vectors.
        nonuniform = candidates[~np.all(candidates == candidates[:, :1], axis=1)]
        take = min(target - levels, len(nonuniform))
        vectors = np.concatenate([np.repeat(np.arange(levels)[:, None], n_fights, axis=1),
                                  nonuniform[rng.choice(len(nonuniform), take, replace=False)]])
    return vectors


