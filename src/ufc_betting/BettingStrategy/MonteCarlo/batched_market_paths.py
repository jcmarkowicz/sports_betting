"""Shared fight caches and bounded card batches for nearest-market evaluation."""
import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator as Calculator


def simulate_market_paths(paths, policies, *, use_decay, calc_parlay,
                          initial_bankroll, batch_size, progress_every, return_event_exposure=False):
    """Evaluate normalized paths with the same rules as calculate_market_event_returns.

    Deduplicate numerical inputs, not fighter names: sampled prices and model
    predictions may differ for repeated donor fights. Keep card order for EV ties.
    """
    from .mc_engine import settle_bankroll_paths

    columns = ['proba_red', 'prediction_se', 'winner']
    columns += [f'{prefix}_{side}' for prefix in (
        'dec_open', 'dec_close1', 'dec_fair_close1',
        'recorded_close1', 'recorded_close2') for side in ('blue', 'red')]
    inputs, cards, counts, offset = [], {}, [], 0
    for path_id, frame in enumerate(paths):
        inputs.append(frame[columns])
        groups = frame.groupby('date', sort=True).indices
        counts.append(len(groups))
        for event_id, ids in enumerate(groups.values()):
            cards.setdefault(len(ids), []).append((path_id, event_id, ids + offset))
        offset += len(frame)
    codes, unique = pd.factorize(pd.MultiIndex.from_frame(pd.concat(inputs, ignore_index=True)), sort=False)
    data = unique.to_frame(index=False)
    data.columns = columns
    data['proba_blue'] = 1 - data.proba_red
    data['pred_winner'] = (data.proba_red >= .5).astype(int)
    # Match the reference market evaluator's risk-probability convention.
    data['risk_red_proba'] = data.proba_red
    closing, opening, opening_policies = {}, {}, {}
    for stage, params in policies.items():
        closing[stage] = Calculator.calculate_scenario_cache(data, params)
        decayed = use_decay and params.decay_rates is not None
        opening_policies[stage] = params.opening_params if decayed else params
        opening[stage] = (Calculator.calculate_scenario_cache(data, opening_policies[stage])
                          if decayed else closing[stage])
    side = data.pred_winner.to_numpy()
    rows = np.arange(len(data))
    prices = {prefix: data[[f'{prefix}_blue', f'{prefix}_red']].to_numpy()[rows, side]
              for prefix in ('dec_open', 'dec_close1', 'recorded_close1', 'recorded_close2')}
    selected = Calculator.nearest_market(prices['dec_close1'], prices['recorded_close1'], prices['recorded_close2'])
    mask = (selected == 'close1')[:, None]
    mixed_close = np.where(mask, closing['close1'], closing['close2'])
    mixed_open = np.where(mask, opening['close1'], opening['close2'])
    winners = data.winner.to_numpy()
    ml = [np.zeros(n) for n in counts]
    pl = [np.zeros(n) for n in counts]
    exposure = [np.zeros(n) for n in counts] if return_event_exposure else None
    done, total = 0, sum(counts)
    if progress_every:
        print(f'Evaluating {total} cards using {len(data)} unique fight inputs', flush=True)
    for same_size in cards.values():
        for start in range(0, len(same_size), 1024):
            batch = same_size[start:start + 1024]
            ids = np.stack([codes[card[2]] for card in batch])
            legs = np.argsort(-mixed_close[ids, 1], axis=1, kind='stable')[:, :2]
            leg_ids = np.take_along_axis(ids, legs, axis=1)
            ticket_market = Calculator.nearest_market(
                prices['dec_close1'][leg_ids].prod(axis=1),
                prices['recorded_close1'][leg_ids].prod(axis=1),
                prices['recorded_close2'][leg_ids].prod(axis=1))
            for stage, params in policies.items():
                # z is enabled if any selected single or active ticket enables it.
                enabled = np.zeros(len(ids), dtype=bool)
                for market, policy in policies.items():
                    if policy.z is not None:
                        enabled |= (selected[ids] == market).any(axis=1)
                if calc_parlay and ids.shape[1] >= 2 and params.z is not None:
                    enabled[:] = True
                for use_z in (False, True):
                    positions = np.flatnonzero((ticket_market == stage) & (enabled == use_z))
                    if not len(positions):
                        continue
                    ix = ids[positions]
                    cl, op = mixed_close[ix], mixed_open[ix]
                    alloc = Calculator.scenario_allocations(cl, winners[ix], params, calc_parlay, apply_adjustments=False)
                    opening_alloc = Calculator.scenario_allocations(op, winners[ix], opening_policies[stage], calc_parlay, apply_adjustments=False)
                    stakes = alloc['moneyline_stakes']
                    for market, policy in policies.items():
                        if use_decay and policy.decay_rates is not None:
                            rates = policy.decay_rates
                            weight = Calculator.decay_weights(prices['dec_open'][ix], prices['dec_close1'][ix], rates['k_down'], rates['k_up'])
                            choose = selected[ix] == market
                            stakes[choose] = (weight * opening_alloc['moneyline_stakes'] + (1 - weight) * stakes)[choose]
                    if calc_parlay and ix.shape[1] >= 2 and use_decay and params.decay_rates is not None:
                        rates = params.decay_rates
                        movement = np.log(prices['dec_close1'][ix] / prices['dec_open'][ix])
                        movement = np.take_along_axis(movement, alloc['legs'], axis=1)
                        weight = np.exp(-rates['k_parlay_down'] * np.maximum(-movement, 0).sum(axis=1)
                                        -rates['k_parlay_up'] * np.maximum(movement, 0).sum(axis=1))
                        alloc['parlay_stakes'] = weight * opening_alloc['parlay_stakes'] + (1 - weight) * alloc['parlay_stakes']
                    alloc = Calculator.finalize_scenario_allocations(
                        cl, alloc, use_z, ticket_cache=closing[stage][ix],
                        single_enabled=np.where(selected[ix] == 'close1',
                                                policies['close1'].z is not None, policies['close2'].z is not None),
                        ticket_enabled=params.z is not None)
                    singles = (alloc['moneyline_stakes'] * cl[:, :, 2]).sum(axis=1)
                    tickets = alloc['parlay_stakes'] * alloc['multiplier'] * alloc['parlay_profit']
                    wagered = alloc['moneyline_stakes'].sum(axis=1) + alloc['parlay_stakes'] * alloc['multiplier']
                    for position, single, ticket, wager in zip(positions, singles, tickets, wagered):
                        path_id, event_id, _ = batch[position]
                        ml[path_id][event_id], pl[path_id][event_id] = single, ticket
                        if exposure is not None:
                            exposure[path_id][event_id] = wager
            done += len(batch)
            if progress_every and (done == total or done % (1024 * progress_every) == 0):
                print(f'Saved fight cards {done}/{total} evaluated', flush=True)
    # Group equal-length paths for vectorized settlement, preserving output order.
    output = ([None] * len(paths), [None] * len(paths), [None] * len(paths))
    for count in sorted(set(counts)):
        positions = [i for i, n in enumerate(counts) if n == count]
        singles = np.stack([ml[i] for i in positions])
        tickets = np.stack([pl[i] for i in positions])
        settled = settle_bankroll_paths(singles + tickets, initial_bankroll,
            family_returns=dict(moneyline=singles, parlay=tickets), batch_size=batch_size)
        for family, key in zip(output, ('moneyline_profits', 'parlay_profits', 'bankroll')):
            for position, values in zip(positions, settled[key]):
                family[position] = values
    if exposure is not None:
        for i, values in enumerate(exposure):
            active = np.repeat(np.asarray(output[2][i])[:-1] > 0, batch_size)[:len(values)]
            exposure[i] = np.where(active, values, 0.)
        return (*output, dict(event_exposure=exposure, event_counts=counts, batch_size=batch_size))
    return output
