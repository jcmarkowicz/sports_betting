"""Data adapters for the shared betting calculator's scenario API."""
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator

def fight_cache(data, params, adjust_mdd_by_edge=False, *, odds_type='close1'):
    return BettingReturnCalculator.calculate_scenario_cache(data, params, odds_type)


def event_values(selected, group, params, parlay):
    return BettingReturnCalculator.calculate_scenario_returns(
        selected, group.winner.to_numpy(), params,
        strategy='parlay' if parlay else 'moneyline',
    )


