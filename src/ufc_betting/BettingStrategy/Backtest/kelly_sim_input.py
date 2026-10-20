"""Reusable data and parameter container for simulate_kelly."""

from copy import deepcopy
from ufc_betting.BettingStrategy.betting_params import BettingParams

from ufc_betting.BettingStrategy.kelly_utils import (
    kelly_simulation_input, portable_kelly_input,
)


class KellySimInput:
    """Bundle simulation data and every supported Kelly betting control.

    Use the constructor for a prepared DataFrame with custom column names or
    simulation controls. Use from_predictions for the standard search prediction
    schema and Optuna parameter names. Preparation never runs a simulation.

    Example:
        inputs = KellySimInput.from_predictions(predictions, best_params, 'open')
        moneylines, parlays = simulate_kelly(inputs.data, **inputs.kwargs)
    """

    DEFAULTS = {
        'winner_col': 'winner', 'date_col': 'date',
        'init_bankroll': 1000, 'bankroll_floor': None,
        'max_drawdown': .30, 'parlay_mdd': .25, 'N': 1000, 'N_parlay': 2000,
        'calc_parlay': False, 'test_other_ev': False, 'test_bankroll_scaler': False,
        'z': None, 'prediction_se_col': 'se', 'moneyline_american_bounds': None,
        'calc_heavy_favorite_parlay': False, 'heavy_favorite_cutoff': -300,
        'heavy_parlay_max_legs': 4, 'use_walk_forward_params': False,
        'walk_forward_param_prefix': 'best_', 'monthly_params': None,
        'risk_prob_cols': None, 'heavy_parlay_mdd': None, 'adjust_mdd_by_edge': False,
        'decay_rates': None, 'opening_data': None, 'opening_params': None,
        'betting_params': None,
    }
    COLUMN_DEFAULTS = {
        'prob_cols': ['proba_blue', 'proba_red'],
        'fair_decimal_cols': ['dec_fair_open_blue', 'dec_fair_open_red'],
        'real_decimal_cols': ['dec_open_blue', 'dec_open_red'],
        'pred_winner_col': 'pred_winner',
    }

    def __init__(self, data, **simulation_params):
        unknown = set(simulation_params) - self.DEFAULTS.keys() - self.COLUMN_DEFAULTS.keys()
        if unknown:
            raise ValueError(f'Unknown Kelly simulation parameters: {sorted(unknown)}')
        self.data = data.copy(deep=True)
        self.kwargs = deepcopy({**self.DEFAULTS, **self.COLUMN_DEFAULTS, **simulation_params})
        controls = self.kwargs.pop('betting_params')
        if controls is not None:
            self.kwargs.update(BettingParams.from_value(controls).to_simulation_kwargs())
        # Validate both direct kwargs and any overrides supplied by the caller.
        self.betting_params = BettingParams.from_simulation_kwargs(self.kwargs)
        self._validate_columns()

    def _validate_columns(self):
        required = {'fighter_red', 'fighter_blue', 'open_red', 'open_blue'}
        for name in ('date_col', 'winner_col', 'pred_winner_col'):
            required.add(self.kwargs[name])
        for name in ('prob_cols', 'fair_decimal_cols', 'real_decimal_cols', 'risk_prob_cols'):
            columns = self.kwargs[name]
            if columns is None and name == 'risk_prob_cols':
                continue
            if not isinstance(columns, (list, tuple)) or len(columns) != 2:
                raise ValueError(f'{name} must contain two columns ordered blue, red')
            required.update(columns)
        if self.kwargs['z'] is not None:
            required.add(self.kwargs['prediction_se_col'])
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f'Missing Kelly simulation input columns: {sorted(missing)}')

    @classmethod
    def from_predictions(cls, predictions, betting_params, odds_type='open',
                         initial_bankroll=500., **simulation_overrides):
        """Prepare standard predictions and search parameters for fixed-control simulation.

        betting_params accepts BettingParams or Optuna names (mdd, mdd_parlay).
        The object can include decay rates and opening controls. All native
        simulator controls can additionally be supplied as simulation_overrides.
        Explicit overrides take precedence. Dataset price aliases allow controls
        from one market to be applied to another market's prepared data.
        """
        extra_names = ('heavy_parlay_mdd', 'heavy_favorite_cutoff', 'heavy_parlay_max_legs')
        extras = {name: betting_params[name] for name in extra_names
                  if isinstance(betting_params, dict) and name in betting_params}
        source = ({key: value for key, value in betting_params.items() if key not in extras}
                  if isinstance(betting_params, dict) else betting_params)
        controls = BettingParams.from_value(source)
        data = predictions.copy(deep=True)
        for name, value in controls.to_policy().items():
            data[f'best_{name}'] = value
        if 'proba_blue' not in data and 'proba_red' in data:
            data['proba_blue'] = 1 - data['proba_red']
        if 'pred_winner' not in data and 'proba_red' in data:
            data['pred_winner'] = (data['proba_red'] >= .5).astype(int)
        if 'risk_red_proba' not in data and 'proba_red' in data:
            data['risk_red_proba'] = data['proba_red']
        data, kwargs = kelly_simulation_input(data, odds_type, initial_bankroll)
        data, kwargs = portable_kelly_input(data, kwargs, controls)
        kwargs.update(extras)
        kwargs.update(simulation_overrides)
        return cls(data, **kwargs)

    def to_simulation_args(self):
        """Return independent copies suitable for simulate_kelly(data, **kwargs)."""
        return self.data.copy(deep=True), deepcopy(self.kwargs)

    @classmethod
    def from_params_file(cls, predictions, params_file, odds_type='open',
                         initial_bankroll=500., **simulation_overrides):
        """Load saved Kelly/decay parameters and prepare simulation inputs."""
        params = BettingParams.load(params_file)
        return cls.from_predictions(predictions, params, odds_type,
                                    initial_bankroll, **simulation_overrides)
