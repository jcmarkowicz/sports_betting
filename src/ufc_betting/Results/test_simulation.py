from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_backtest

from ufc_betting.config import settings
from dataclasses import replace

def simulate_saved_predictions(
    odds_type, initial_bankroll=500, *, use_decay=True,
    split="val", model_type="open",
    prediction_template="{split}_logit_{model_type}_with_{odds_type}.csv",
    predictions_path=None, opening_predictions_path=None,
):
    """Run the shared notebook/dashboard simulation from frozen predictions."""
    if odds_type not in ("open", "close1", "close2"):
        raise ValueError("odds_type must be open, close1, or close2")

    from pathlib import Path

    data_dir = settings.data_dir
    def resolve_prediction_path(value):
        path = Path(value)
        path = path if path.is_absolute() else settings.model_predictions_folder / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction CSV not found: {path}. Check split/model_type, "
                "edit prediction_template, or pass predictions_path explicitly.")
        return path
    params_path = data_dir / "kelly_params" / f"{odds_type}.json"
    prediction_path = resolve_prediction_path(
        predictions_path if predictions_path is not None else prediction_template.format(
            split=split, model_type=model_type, odds_type=odds_type))
    betting_params = BettingParams.load(params_path)
    decay_enabled = use_decay and odds_type != "open"
    if decay_enabled and (betting_params.decay_rates is None or betting_params.opening_params is None):
        raise ValueError(f"{params_path} must contain decay_rates and opening_params from the joint search")
    if not decay_enabled:
        betting_params = replace(betting_params, decay_rates=None, opening_params=None)
    predictions = read_frozen_predictions(prediction_path)

    opening_data = None
    if decay_enabled:
        # Use original opening prices for the same rows; the fixed opening
        # policy comes from this closing JSON, not a separately changed open.json.
        reference_cols = ["date", "fighter_red", "fighter_blue", "dec_open_red", "dec_open_blue"]
        if opening_predictions_path is not None:
            opening_data = read_frozen_predictions(resolve_prediction_path(opening_predictions_path))
        elif set(reference_cols).issubset(predictions.columns):
            opening_data = predictions[reference_cols].copy()
        else:
            raise ValueError(
                "Decay requires original opening odds for these fights. Supply "
                "dec_open_red/blue in the selected CSV or opening_predictions_path "
                "pointing to matching fights from the same split.")

    sim_input = KellySimInput.from_predictions(
        predictions,
        betting_params,
        odds_type=odds_type,
        initial_bankroll=initial_bankroll,
        calc_parlay=True,
        opening_data=opening_data,
    )
    betting_data, sim_kwargs = sim_input.to_simulation_args()
    moneylines, parlays = simulate_kelly(betting_data, **sim_kwargs)

    return moneylines, parlays, betting_params, prediction_path, betting_data, sim_kwargs
