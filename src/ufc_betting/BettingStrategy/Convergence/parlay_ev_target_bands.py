"""Parlay scenario adapter; all sizing and settlement lives in the calculator."""
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator


def evaluate_parlays(selected, group, validate=True, mdd_parlay=.4, N_parlay=1490, z=.5, combined=False):
    # validate is retained for old callers; calculator inputs are always validated.
    params = BettingParams(mdd_parlay=mdd_parlay, N_parlay=N_parlay, z=z)
    return BettingReturnCalculator.calculate_scenario_returns(
        selected, group.winner.to_numpy(), params,
        strategy='combined' if combined else 'parlay',
    )


