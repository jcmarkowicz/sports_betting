"""Adapters for applying saved decay controls to scenario batches."""
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator


def validate_rates(rates):
    rates = BettingParams.validate_decay_rates(rates)
    if rates is None:
        raise ValueError('Decay rates are required')
    return rates


def singles_allocations(selected, params):
    return BettingReturnCalculator.adjusted_stakes(
        selected[:, :, 0], selected[:, :, 3], selected[:, :, 4],
        BettingParams.from_value(params).z is not None,
    )


def decay_event_values(open_selected, close_selected, opening_prices, group,
                       opening_params, closing_params, rates, parlay):
    return BettingReturnCalculator.calculate_decay_scenario_returns(
        open_selected, close_selected, opening_prices, group.winner.to_numpy(),
        opening_params, closing_params, rates, parlay,
    )
