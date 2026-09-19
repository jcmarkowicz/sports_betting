"""Natural EV scenario bands for the backtest's top-two-EV parlay component.

Batch ticket sizing is checked against parlay_top_ev on every event. Shared
moneyline/parlay risk adjustments match simulate_kelly; only parlay P&L is
compounded here, so this is an isolated component account, not combined P&L.
"""
import numpy as np
import pandas as pd
from ufc_betting.BettingStrategy.Backtest.backtest_functions import parlay_top_ev
from ufc_betting.BettingStrategy.kelly_scaling import expected_max_drawdown, log_return_volatility


def evaluate_parlays(selected, group, validate=True):
    # Cache fields: single stake, unit EV, unit profit, raw edge,
    # conservative edge, probability, offered odds, predicted side.
    count, fights, _ = selected.shape
    if fights < 2:
        return np.zeros(count), np.zeros(count), np.zeros(count)
    # pandas descending quicksort reverses its ascending sort on reversed input.
    order = (fights - 1 - np.argsort(selected[:, ::-1, 1], axis=1))[:, ::-1][:, :2]
    rows = np.arange(count)[:, None]
    legs = selected[rows, order]
    probability = legs[:, :, 5].prod(axis=1)
    odds = legs[:, :, 6].prod(axis=1)
    ev = probability * odds - 1
    full = np.maximum(ev / (odds - 1), 0)
    low, high = np.zeros(count), np.ones(count)
    while np.max(high - low) > 1e-4:
        middle = (low + high) / 2
        sigma, _ = log_return_volatility(middle * full, odds - 1, probability)
        safe = expected_max_drawdown(sigma, 1490) <= -np.log1p(-.4)
        low = np.where(safe, middle, low)
        high = np.where(safe, high, middle)
    stake = low * full
    wins = np.all(legs[:, :, 7] == group.winner.to_numpy()[order], axis=1)
    unit_profit = np.where(wins, odds - 1, -1)
    if validate:
        for j in np.unique(np.linspace(0, count - 1, min(count, 12), dtype=int)):
            data = group.copy().reset_index(drop=True)
            data['choice_ev'] = selected[j, :, 1]
            data['choice_proba'] = selected[j, :, 5]
            data['choice_real_odds'] = selected[j, :, 6]
            data['pred_winner'] = selected[j, :, 7]
            profit, _, ticket = parlay_top_ev(data, 1., top_n=[0, 1], parlay_mdd=.4, N=1490)
            np.testing.assert_allclose(stake[j], ticket.fstar_parlay.iloc[0], atol=1e-12)
            np.testing.assert_allclose(stake[j] * unit_profit[j], profit, atol=1e-12)
            if ev[j] > 0:
                np.testing.assert_array_equal(order[j], ticket.index.to_numpy())
    # Preserve simulate_kelly's risk adjustment, including single-bet weights.
    weights = selected[:, :, 0].copy()
    weights[rows, order] += stake[:, None] / 2
    raw = (weights * selected[:, :, 3]).sum(axis=1)
    conservative = (weights * selected[:, :, 4]).sum(axis=1)
    uncertainty = np.divide(conservative, raw, out=np.zeros(count), where=raw > 0)
    uncertainty = np.clip(uncertainty, 0, 1)
    total = (selected[:, :, 0].sum(axis=1) + stake) * uncertainty
    final_stake = stake * uncertainty / np.maximum(total, 1)
    return final_stake * ev, final_stake * unit_profit, final_stake


if __name__ == '__main__':
    from ufc_betting.Models.RiskManagement.realized_ev_target_bands import run
    run(parlay=True)
