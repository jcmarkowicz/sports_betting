"""Load historical conditional KDEs and sample execution prices for Monte Carlo."""
from pathlib import Path

import joblib
import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional

from ufc_betting.Models.LineMovement.conditional_price_density import sample_prices


def load_density_models(models_dir, days_out, density_models=None):
    """Return {day: {'close1': fit, 'close2': fit}} for every requested day.

    Read conditional_price_density_{stage}_{day}_days*.joblib in models_dir.
    Exactly one file must match each day/market; optional filename suffixes
    such as _test are accepted. Supplied models may be a day -> stage
    mapping, or a stage mapping for one day; each fit may also be a joblib path.
    Both markets are required and their day/market metadata is validated.
    """
    days = [days_out] if np.isscalar(days_out) else list(days_out)
    if (not days or any(isinstance(day, (bool, np.bool_)) or
                       not isinstance(day, (int, np.integer)) or day < 0 for day in days)
            or len(set(days)) != len(days)):
        raise ValueError('days_out must be a nonnegative integer or unique nonnegative integers')
    models = {}
    for day in days:
        supplied = None
        if density_models is not None:
            if not isinstance(density_models, dict):
                raise ValueError('density_models must map days/markets to fitted models')
            supplied = density_models.get(day, density_models if len(days) == 1 else None)
            if not isinstance(supplied, dict) or not {'close1', 'close2'} <= supplied.keys():
                raise ValueError(f'density_models must contain both close1 and close2 for day {day}')
        models[day] = {}
        for stage in ('close1', 'close2'):
            if supplied is not None:
                value = supplied[stage]
            else:
                folder = Path(models_dir)
                stem = f'conditional_price_density_{stage}_{day}_days'
                matches = sorted(path for path in folder.glob(f'{stem}*.joblib')
                                 if path.stem == stem or path.stem.startswith(stem + '_'))
                if not matches:
                    raise FileNotFoundError(f'No {stage} density for day {day} in {folder}')
                if len(matches) != 1:
                    raise ValueError(f'Multiple {stage} densities for day {day} in {folder}: '
                                     f'{[path.name for path in matches]}. Supply a folder with one model per day/market.')
                value = matches[0]
            if isinstance(value, (str, Path)):
                value = joblib.load(value)
            if not isinstance(value, dict) or not isinstance(value.get('model'), KDEMultivariateConditional):
                raise ValueError('Expected a fit_day_price_density result; refit old training-only saved models')
            if value.get('days_out') != day or value.get('odds_type') != stage:
                raise ValueError(f'Density metadata must match day={day}, odds_type={stage}')
            models[day][stage] = value
    return models


class ConditionalPriceSampler:
    """Sample intermediate prices using the nearer closing-market density."""

    def __init__(self, models, *, random_state=None):
        self.models = models
        self.rng = np.random.default_rng(random_state)

    def sample(self, frame):
        """Choose a day per event, interpolate endpoints, then sample each side.

        One uniform endpoint fraction per fight is shared by both sides. The
        nearer KDE is conditioned on that endpoint; ties select close1.
        Adds day, fraction, endpoint, and density-source audit columns to frame.
        Returns (fights, 2) decimal prices, ordered red then blue.
        """
        days = np.array(list(self.models))
        event_days = {event: int(self.rng.choice(days)) for event in frame.sampled_event_id.unique()}
        frame['density_days_out'] = frame.sampled_event_id.map(event_days)
        frame['endpoint_fraction'] = self.rng.random(len(frame))
        prices = np.empty((len(frame), 2))
        for column, side in enumerate(('red', 'blue')):
            first = frame[f'dec_close1_{side}'].to_numpy(float)
            second = frame[f'dec_close2_{side}'].to_numpy(float)
            endpoints = first + frame.endpoint_fraction.to_numpy() * (second - first)
            # Compare fractions to avoid rounding errors at the midpoint.
            sources = np.where((frame.endpoint_fraction.to_numpy() <= .5) | (first == second),
                               'close1', 'close2')
            frame[f'sampled_endpoint_{side}'] = endpoints
            frame[f'density_source_{side}'] = sources
            for position, (day, opening, closing, stage) in enumerate(zip(
                    frame.density_days_out, frame[f'dec_open_{side}'], endpoints, sources)):
                prices[position, column] = sample_prices(
                    self.models[day][stage]['model'], opening, closing, 1,
                    random_state=self.rng)[0]
        return prices
