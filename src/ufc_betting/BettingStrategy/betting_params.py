"""Validated Kelly controls, optional decay rates, and opening policy controls."""

from dataclasses import asdict, dataclass, fields, replace
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class BettingParams:
    """One Kelly policy, optionally bundled with decay and its opening policy.

    params = results['betting_params']
    params.save('Data/kelly_params/close1_combined.json')
    params = BettingParams.load('Data/kelly_params/close1_combined.json')

    DataFrames, bankroll, and reporting options belong to KellySimInput.
    """
    z: float | None = .5
    min_moneyline: float | None = None
    N: int = 250
    N_parlay: int = 1490
    mdd: float = .4
    mdd_parlay: float = .4
    moneyline_max_american: float | None = None
    decay_rates: dict | None = None
    opening_params: 'BettingParams | None' = None

    def __post_init__(self):
        for name in ('mdd', 'mdd_parlay'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{name} must be between 0 and 1')
            object.__setattr__(self, name, float(value))
        for name in ('N', 'N_parlay'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 1:
                raise ValueError(f'{name} must be an integer greater than 1')
            object.__setattr__(self, name, int(value))
        if self.z is not None and (not math.isfinite(self.z) or self.z < 0):
            raise ValueError('z must be nonnegative or None')
        if self.z is not None:
            object.__setattr__(self, 'z', float(self.z))
        for name in ('min_moneyline', 'moneyline_max_american'):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f'{name} must be finite or None')
            if value is not None:
                object.__setattr__(self, name, float(value))
        if (self.min_moneyline is not None and self.moneyline_max_american is not None
                and self.min_moneyline > self.moneyline_max_american):
            raise ValueError('Minimum moneyline must not exceed maximum')
        object.__setattr__(self, 'decay_rates', self.validate_decay_rates(self.decay_rates))
        if self.opening_params is not None:
            opening = self.from_value(self.opening_params)
            if opening.opening_params is not None or opening.decay_rates is not None:
                raise ValueError('Opening policy must contain only Kelly controls')
            object.__setattr__(self, 'opening_params', opening)

    @staticmethod
    def validate_decay_rates(rates):
        if rates is None:
            return None
        rates = dict(rates.get('best_params', rates))
        singles = {'k_down', 'k_up'}
        if set(rates) not in (singles, singles | {'k_parlay_down', 'k_parlay_up'}):
            raise ValueError('Decay rates require k_down/k_up and optionally both parlay rates')
        if any(not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError('Decay rates must be finite and nonnegative')
        return {key: float(value) for key, value in rates.items()}

    @classmethod
    def from_value(cls, value, *, require_complete=False):
        """Accept a BettingParams, parameter dictionary, or saved JSON path."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text(encoding='utf-8'))
        if not isinstance(value, Mapping):
            raise ValueError('Parameter file/input must contain a parameter dictionary')
        values = dict(value)
        if 'moneyline_min_american' in values:
            lower = values.pop('moneyline_min_american')
            if 'min_moneyline' in values and values['min_moneyline'] != lower:
                raise ValueError('Conflicting minimum moneyline values')
            values['min_moneyline'] = lower
        required = {'mdd', 'N', 'mdd_parlay', 'N_parlay', 'z', 'min_moneyline'}
        if require_complete and not required <= values.keys():
            raise ValueError(f'Parameter file/input must contain {sorted(required)}')
        for name, unbounded in (('min_moneyline', -math.inf), ('moneyline_max_american', math.inf)):
            if values.get(name) == unbounded:
                values[name] = None
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown betting parameters: {sorted(unknown)}')
        return cls(**values)

    def to_policy(self):
        """Calculator controls with canonical search names and numeric bounds."""
        return dict(mdd=self.mdd, N=self.N, mdd_parlay=self.mdd_parlay,
                    N_parlay=self.N_parlay, z=self.z,
                    moneyline_min_american=-math.inf if self.min_moneyline is None else self.min_moneyline,
                    moneyline_max_american=math.inf if self.moneyline_max_american is None else self.moneyline_max_american)

    def to_simulation_kwargs(self):
        """Translate controls to simulate_kelly's argument names."""
        bounds = None if self.min_moneyline is None and self.moneyline_max_american is None else (
            self.to_policy()['moneyline_min_american'], self.to_policy()['moneyline_max_american'])
        return dict(max_drawdown=self.mdd, N=self.N, parlay_mdd=self.mdd_parlay,
                    N_parlay=self.N_parlay, z=self.z, moneyline_american_bounds=bounds,
                    decay_rates=None if self.decay_rates is None else dict(self.decay_rates),
                    opening_params=None if self.opening_params is None else self.opening_params.to_policy())

    @classmethod
    def from_simulation_kwargs(cls, params):
        """Read the betting controls from a full set of simulation arguments."""
        bounds = params.get('moneyline_american_bounds')
        return cls.from_value(dict(
            mdd=params['max_drawdown'], N=params['N'],
            mdd_parlay=params['parlay_mdd'], N_parlay=params['N_parlay'],
            z=params['z'], min_moneyline=None if bounds is None else bounds[0],
            moneyline_max_american=None if bounds is None else bounds[1],
            decay_rates=params.get('decay_rates'), opening_params=params.get('opening_params'),
        ))

    def with_decay(self, rates, opening_params):
        return replace(self, decay_rates=rates, opening_params=self.from_value(opening_params))

    def to_dict(self):
        """JSON-safe parameters; datasets and simulation settings remain separate."""
        result = asdict(self)
        result['moneyline_min_american'] = result.pop('min_moneyline')
        result['opening_params'] = None if self.opening_params is None else self.opening_params.to_dict()
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_value(path, require_complete=True)
