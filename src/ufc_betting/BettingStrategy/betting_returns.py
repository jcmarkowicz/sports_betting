"""Settle fixed betting predictions into shared-bankroll event returns."""

import numpy as np
import pandas as pd

from ufc_betting.BettingStrategy.kelly_worker import run_per_bet_scaling, parlay_top_ev
from ufc_betting.BettingStrategy.kelly_scaling import (
    kelly_edge, expected_value, decimal_to_american, scale_kelly_for_mdd,
)
from ufc_betting.BettingStrategy.betting_params import BettingParams


class BettingReturnCalculator:
    """Calculate fixed-policy and allocated-stake betting returns.

    FrozenBettingSearch owns optimization and bootstrap scoring. This calculator
    settles fixed policies with uncertainty and exposure adjustments. Precomputed
    allocations can be settled directly without applying those adjustments again.
    Odds columns must be ordered blue, then red.
    """

    def __init__(self, real_odds, fair_odds, dates):
        self.real_odds = real_odds
        self.fair_odds = fair_odds
        self.dates = dates

    @staticmethod
    def moneyline_unit_profit(decimal_odds, selections, outcomes):
        """Net profit per unit staked: win=d-1, loss=-1, void=0.

        Outcomes 2 (draw/NC) and 3 (NC) are void; missing outcomes stay unsettled.
        """
        odds, sides, winners = np.broadcast_arrays(
            np.asarray(decimal_odds, float), np.asarray(selections, float),
            np.asarray(outcomes, float))
        if not np.isin(winners[~np.isnan(winners)], [0, 1, 2, 3]).all():
            raise ValueError("Outcomes must be 0/1, 2/3 for void, or missing")
        return np.where(np.isnan(winners) | np.isnan(sides), np.nan,
                        np.where(np.isin(winners, [2, 3]), 0.,
                                 np.where(sides == winners, odds - 1., -1.)))

    @staticmethod
    def parlay_unit_profit(decimal_odds, selections, outcomes):
        """Settle fixed ticket legs along the last axis; any void refunds the ticket."""
        sides, winners = np.broadcast_arrays(np.asarray(selections, float),
                                             np.asarray(outcomes, float))
        # Validate outcomes with the same rules used by singles.
        BettingReturnCalculator.moneyline_unit_profit(2., sides, winners)
        return np.where(np.isin(winners, [2, 3]).any(axis=-1), 0.,
                        np.where((np.isnan(winners) | np.isnan(sides)).any(axis=-1), np.nan,
                                 np.where((sides == winners).all(axis=-1),
                                          np.asarray(decimal_odds, float) - 1., -1.)))

    @staticmethod
    def settle_allocations(stakes, expected_profit, realized_profit):
        """Sum expected return, realized return, and exposure along the bet axis."""
        stakes, expected_profit, realized_profit = np.broadcast_arrays(
            np.asarray(stakes, float), np.asarray(expected_profit, float),
            np.asarray(realized_profit, float))
        return ((stakes * expected_profit).sum(axis=-1),
                (stakes * realized_profit).sum(axis=-1), stakes.sum(axis=-1))

    @staticmethod
    def risk_multiplier(weights, raw_edges, conservative_edges, enabled=True):
        """Independent conservative/raw edge ratios; weights only set the shape."""
        weights, raw, conservative = np.broadcast_arrays(
            np.asarray(weights, float), np.asarray(raw_edges, float),
            np.asarray(conservative_edges, float))
        if not enabled:
            return np.ones(weights.shape)
        return np.clip(np.divide(conservative, raw, out=np.zeros_like(raw), where=raw > 0), 0, 1)

    @classmethod
    def ticket_risk_multiplier(cls, raw_edges, conservative_edges, enabled=True):
        """One ticket's edge ratio using only its own legs (equal leg weights)."""
        raw = np.asarray(raw_edges, float).sum(axis=-1)
        conservative = np.asarray(conservative_edges, float).sum(axis=-1)
        return cls.risk_multiplier(np.ones_like(raw), raw, conservative, enabled)

    @classmethod
    def adjusted_stakes(cls, stakes, raw_edges, conservative_edges, enabled=True):
        """Apply per-bet uncertainty, then cap total exposure at one bankroll."""
        stakes = np.asarray(stakes, float)
        adjusted = stakes * cls.risk_multiplier(stakes, raw_edges, conservative_edges, enabled)
        return adjusted / np.maximum(adjusted.sum(axis=-1), 1)[..., None]

    @staticmethod
    def decay_weights(opening, current, k_down, k_up):
        """Opening-policy weights for movement in decimal prices."""
        if any(not np.isfinite(v) or v < 0 for v in (k_down, k_up)):
            raise ValueError('Decay rates must be finite and nonnegative')
        opening, current = np.asarray(opening, float), np.asarray(current, float)
        if (opening.shape != current.shape or not np.isfinite([opening, current]).all()
                or (opening <= 1).any() or (current <= 1).any()):
            raise ValueError('Matched finite decimal prices greater than 1 required')
        movement = np.log(current / opening)
        return np.exp(-np.where(movement < 0, k_down, k_up) * np.abs(movement))

    @staticmethod
    def nearest_market(sampled, close1, close2):
        """Absolute decimal-odds distance; ties select close1."""
        return np.where(np.abs(sampled - close1) <= np.abs(sampled - close2), 'close1', 'close2')

    @classmethod
    def calculate_market_event_returns(cls, group, policies, *, use_decay=True, calc_parlay=True):
        """Select singles by chosen-side price and the top-EV ticket by product odds.

        Blend raw allocations first, then apply the selected closing policies'
        z corrections and one shared event exposure cap to singles and parlay.
        """
        if set(policies) != {'close1', 'close2'}:
            raise ValueError('policies must supply close1 and close2')
        policies = {stage: BettingParams.from_value(value) for stage, value in policies.items()}
        data = group.reset_index(drop=True).copy()
        if not data.winner.isin([0, 1, 2, 3]).all():
            raise ValueError('Market-policy simulation requires outcomes 0, 1, or 2 (void)')
        side = data.pred_winner.to_numpy(int)
        rows = np.arange(len(data))
        current = data[['dec_close1_blue', 'dec_close1_red']].to_numpy()[rows, side]
        original = {
            stage: data[[f'recorded_{stage}_blue', f'recorded_{stage}_red']].to_numpy()[rows, side]
            for stage in ('close1', 'close2')
        }
        selected = cls.nearest_market(current, original['close1'], original['close2'])
        probability = np.where(side == 1, data.proba_red, data.proba_blue)
        legs = np.argsort(-(probability * current - 1), kind='stable')[:2]
        ticket_market = str(cls.nearest_market(current[legs].prod(),
                                          original['close1'][legs].prod(), original['close2'][legs].prod()))
        data['risk_red_proba'] = data.proba_red
        data['prediction_se'] = data.proba_se
        closing_cache = np.empty((len(data), 8))
        opening_cache = np.empty_like(closing_cache)
        opening_policies = {}
        closing_caches, opening_caches = {}, {}
        for stage, params in policies.items():
            mask = selected == stage
            closing_caches[stage] = cls.calculate_scenario_cache(data, params)
            closing_cache[mask] = closing_caches[stage][mask]
            if use_decay and params.decay_rates is not None:
                if params.opening_params is None:
                    raise ValueError(f'{stage} decay requires opening_params')
                opening_policies[stage] = params.opening_params
            else:
                opening_policies[stage] = params
            opening_caches[stage] = (cls.calculate_scenario_cache(data, opening_policies[stage])
                                     if use_decay and params.decay_rates is not None
                                     else closing_caches[stage])
            opening_cache[mask] = opening_caches[stage][mask]

        winners = data.winner.to_numpy()
        close = cls.scenario_allocations(closing_cache[None], winners, policies[ticket_market],
                                         calc_parlay, apply_adjustments=False)
        opening = cls.scenario_allocations(opening_cache[None], winners, opening_policies[ticket_market],
                                           calc_parlay, apply_adjustments=False)
        stakes = close['moneyline_stakes'][0].copy()
        opening_prices = data[['dec_open_blue', 'dec_open_red']].to_numpy()[rows, side]
        for stage, params in policies.items():
            mask = selected == stage
            if use_decay and params.decay_rates is not None and mask.any():
                rates = params.decay_rates
                weight = cls.decay_weights(opening_prices[mask], current[mask], rates['k_down'], rates['k_up'])
                stakes[mask] = weight * opening['moneyline_stakes'][0, mask] + (1 - weight) * stakes[mask]
        ticket_stake = float(close['parlay_stakes'][0] * close['multiplier'][0])
        rates = policies[ticket_market].decay_rates
        if calc_parlay and len(legs) == 2 and use_decay and rates is not None:
            if not {'k_parlay_down', 'k_parlay_up'} <= rates.keys():
                raise ValueError(f'{ticket_market} requires saved parlay decay rates')
            movement = np.log(current[legs] / opening_prices[legs])
            weight = np.exp(-rates['k_parlay_down'] * np.maximum(-movement, 0).sum()
                            - rates['k_parlay_up'] * np.maximum(movement, 0).sum())
            ticket_stake = weight * opening['parlay_stakes'][0] * opening['multiplier'][0] + (1 - weight) * ticket_stake
        enabled = any(policies[stage].z is not None for stage in set(selected))
        if calc_parlay and len(legs) == 2:
            enabled |= policies[ticket_market].z is not None
        close['moneyline_stakes'] = stakes[None]
        close['parlay_stakes'] = np.array([ticket_stake])
        close = cls.finalize_scenario_allocations(
            closing_cache[None], close, enabled,
            ticket_cache=closing_caches[ticket_market][None],
            single_enabled=np.array([[policies[stage].z is not None for stage in selected]]),
            ticket_enabled=policies[ticket_market].z is not None)
        stakes = close['moneyline_stakes'][0]
        ticket_stake = float(close['parlay_stakes'][0] * close['multiplier'][0])
        ml = float(np.sum(stakes * closing_cache[:, 2]))
        pl = float(ticket_stake * close['parlay_profit'][0])
        data['betting_policy'] = selected
        data['parlay_policy'] = ticket_market
        data['event_total_fstar'] = stakes.sum() + ticket_stake
        data['choice_fstar'] = stakes
        data['fight_payout'] = stakes * closing_cache[:, 2]
        data['event_payout_money_line'] = ml
        data['parlay_net'] = pl
        ticket = pd.DataFrame([dict(date=data.date.iloc[0], betting_policy=ticket_market,
                                   sampled_combined_odds=current[legs].prod(),
                                   close1_combined_odds=original['close1'][legs].prod(),
                                   close2_combined_odds=original['close2'][legs].prod(),
                                   fstar_parlay=ticket_stake, profit=pl)]) if calc_parlay and len(legs) == 2 else pd.DataFrame()
        return data, ticket

    @classmethod
    def calculate_scenario_cache(cls, data, params, odds_type='close1'):
        """Cache fight inputs: stake, EV, profit, edges, probability, price, side.

        Uses the same pre-event moneyline sizing as calculate_fight_returns.
        Event uncertainty and exposure adjustments are applied after sampling.
        """
        params = BettingParams.from_value(params)
        sides = data.pred_winner.to_numpy(int)
        if not np.isin(sides, [0, 1]).all() or not data.winner.isin([0, 1, 2, 3]).all():
            raise ValueError('Scenario inputs require binary predictions and settled outcomes')
        p = np.where(sides == 1, data.proba_red, data.proba_blue)
        fair = data[[f'dec_fair_{odds_type}_{s}' for s in ('blue', 'red')]].to_numpy()[np.arange(len(data)), sides]
        offered = data[[f'dec_{odds_type}_{s}' for s in ('blue', 'red')]].to_numpy()[np.arange(len(data)), sides]
        risk_red = data.get('risk_red_proba', data.get('risk_proba_red', data.proba_red))
        risk = np.where(sides == 1, risk_red, 1 - risk_red)
        se = data.get('prediction_se', data.get('proba_se', data.get('se')))
        if se is None:
            if params.z is not None:
                raise ValueError('Prediction standard errors are required when z is enabled')
            se = np.zeros(len(data))
        ev = expected_value(p, offered)
        american = np.array([round(decimal_to_american(v)) for v in offered])
        policy = params.to_policy()
        allowed = ((american >= policy['moneyline_min_american'])
                   & (american <= policy.get('moneyline_max_american', np.inf)))
        full = np.array([kelly_edge(probability, price) for probability, price in zip(p, fair)])
        active = allowed & (full > 0) & (ev > 0)
        stakes = np.zeros(len(data))
        # Same sizing as _moneyline_returns/run_per_bet_scaling, but solve all
        # eligible fights together instead of building pandas rows per policy.
        if active.any():
            stakes[active] = scale_kelly_for_mdd(
                p[active], fair[active], full[active], params.N, params.mdd)
        return np.column_stack([
            stakes, ev, cls.moneyline_unit_profit(offered, sides, data.winner.to_numpy()),
            np.maximum(risk - 1 / fair, 0),
            np.maximum(risk - (params.z or 0) * np.asarray(se) - 1 / fair, 0),
            p, offered, sides,
        ])

    @classmethod
    def scenario_allocations(cls, selected, winners, params, include_parlay=False, *,
                             apply_adjustments=True):
        """Size batches shaped (scenarios, fights, 8), with per-bet risk and a joint exposure cap.

        Top-two EV ties retain input order, as in the ordinary calculator.
        """
        params = BettingParams.from_value(params)
        selected = np.asarray(selected, float)
        winners = np.asarray(winners)
        if winners.ndim == 1 and selected.ndim == 3:
            winners = np.broadcast_to(winners, selected.shape[:2])
        if (selected.ndim != 3 or selected.shape[-1] != 8
                or winners.shape != selected.shape[:2] or not np.isfinite(selected).all()):
            raise ValueError('Expected finite scenario inputs shaped (scenarios, fights, 8)')
        if (selected[:, :, 0] < 0).any() or (selected[:, :, 6] <= 1).any():
            raise ValueError('Scenario stakes must be nonnegative and decimal prices greater than 1')
        count, fights, _ = selected.shape
        rows = np.arange(count)[:, None]
        order = np.argsort(-selected[:, :, 1], axis=1, kind='stable')[:, :2]
        stake, ev, profit = (np.zeros(count) for _ in range(3))
        if include_parlay and fights >= 2:
            legs = selected[rows, order]
            probability = legs[:, :, 5].prod(axis=1)
            odds = legs[:, :, 6].prod(axis=1)
            ev = expected_value(probability, odds)
            full = np.maximum(ev / (odds - 1), 0)
            stake = scale_kelly_for_mdd(probability, odds, full, params.N_parlay, params.mdd_parlay)
            profit = cls.parlay_unit_profit(odds, legs[:, :, 7], winners[rows, order])
        result = dict(moneyline_stakes=selected[:, :, 0].copy(),
                      parlay_stakes=stake, multiplier=np.ones(count),
                      parlay_ev=ev, parlay_profit=profit, legs=order)
        return (cls.finalize_scenario_allocations(selected, result, params.z is not None)
                if apply_adjustments else result)

    @classmethod
    def finalize_scenario_allocations(cls, selected, allocations, enabled=True, *, ticket_cache=None,
                                      single_enabled=None, ticket_enabled=None):
        """Apply each bet's z correction once, then cap combined event exposure."""
        singles = allocations['moneyline_stakes']
        ticket = allocations['parlay_stakes']
        single_risk = cls.risk_multiplier(singles, selected[:, :, 3], selected[:, :, 4], enabled)
        ticket_risk = np.ones(len(selected))
        legs = allocations['legs']
        if legs.shape[1] == 2:
            ticket_cache = selected if ticket_cache is None else ticket_cache
            rows = np.arange(len(selected))[:, None]
            ticket_risk = cls.ticket_risk_multiplier(
                ticket_cache[rows, legs, 3], ticket_cache[rows, legs, 4], enabled)
        if single_enabled is not None:
            single_risk = np.where(single_enabled, single_risk, 1.)
        if ticket_enabled is not None:
            ticket_risk = np.where(ticket_enabled, ticket_risk, 1.)
        adjusted_singles = singles * single_risk
        adjusted_ticket = ticket * ticket_risk
        raw_exposure = singles.sum(axis=1) + ticket
        exposure = adjusted_singles.sum(axis=1) + adjusted_ticket
        cap = 1 / np.maximum(exposure, 1.)
        # Diagnostic only: no shared uncertainty factor is applied to singles.
        average_risk = np.divide(exposure, raw_exposure, out=np.ones_like(exposure), where=raw_exposure > 0)
        return dict(allocations, moneyline_stakes=adjusted_singles * cap[:, None],
                    multiplier=ticket_risk * cap, uncertainty_multiplier=average_risk,
                    moneyline_risk_multiplier=single_risk, parlay_risk_multiplier=ticket_risk,
                    exposure_multiplier=cap)

    @classmethod
    def calculate_scenario_returns(cls, selected, winners, params, strategy='moneyline'):
        """Expected return, realized return, and exposure for each scenario."""
        if strategy not in ('moneyline', 'parlay', 'combined'):
            raise ValueError('Unknown betting strategy')
        selected = np.asarray(selected, float)
        allocations = cls.scenario_allocations(selected, winners, params, strategy != 'moneyline')
        singles = cls.settle_allocations(allocations['moneyline_stakes'], selected[:, :, 1], selected[:, :, 2])
        if strategy == 'moneyline':
            return singles
        tickets = cls.settle_allocations(
            (allocations['parlay_stakes'] * allocations['multiplier'])[:, None],
            allocations['parlay_ev'][:, None], allocations['parlay_profit'][:, None])
        return tickets if strategy == 'parlay' else tuple(a + b for a, b in zip(singles, tickets))

    @classmethod
    def decay_scenario_allocations(cls, opening, closing, opening_prices, winners,
                                   opening_params, closing_params, rates, include_parlay=False):
        """Blend raw stakes, apply closing-policy z, then cap joint exposure."""
        rates = BettingParams.validate_decay_rates(rates)
        if rates is None or (include_parlay and 'k_parlay_down' not in rates):
            raise ValueError('Parlay decay requires all four rates; moneylines require two')
        opening, closing = np.asarray(opening, float), np.asarray(closing, float)
        if opening.shape != closing.shape or not np.array_equal(opening[:, :, [1, 2, 5, 6, 7]], closing[:, :, [1, 2, 5, 6, 7]]):
            raise ValueError('Fixed policies must share current predictions, prices, and outcomes')
        closing_params = BettingParams.from_value(closing_params)
        op = cls.scenario_allocations(opening, winners, opening_params, include_parlay, apply_adjustments=False)
        cl = cls.scenario_allocations(closing, winners, closing_params, include_parlay, apply_adjustments=False)
        sides = closing[:, :, 7].astype(int)
        prices = np.asarray(opening_prices)
        if prices.ndim == 2:
            prices = np.broadcast_to(prices, (*sides.shape, 2))
        reference = np.take_along_axis(prices, (1 - sides)[..., None], axis=2)[..., 0]
        w = cls.decay_weights(reference, closing[:, :, 6], rates['k_down'], rates['k_up'])
        cl['moneyline_stakes'] = w * op['moneyline_stakes'] + (1 - w) * cl['moneyline_stakes']
        if include_parlay:
            movement = np.log(closing[:, :, 6] / reference)
            legs = movement[np.arange(len(movement))[:, None], cl['legs']]
            w = np.exp(-rates['k_parlay_down'] * np.maximum(-legs, 0).sum(axis=1)
                       -rates['k_parlay_up'] * np.maximum(legs, 0).sum(axis=1))
            cl['parlay_stakes'] = w * op['parlay_stakes'] + (1 - w) * cl['parlay_stakes']
        return cls.finalize_scenario_allocations(closing, cl, closing_params.z is not None)

    @classmethod
    def calculate_decay_scenario_returns(cls, opening, closing, opening_prices, winners,
                                         opening_params, closing_params, rates, parlay=False):
        allocations = cls.decay_scenario_allocations(opening, closing, opening_prices, winners,
                                                    opening_params, closing_params, rates, parlay)
        closing = np.asarray(closing, float)
        if not parlay:
            return cls.settle_allocations(allocations['moneyline_stakes'], closing[:, :, 1], closing[:, :, 2])
        return cls.settle_allocations(
            (allocations['parlay_stakes'] * allocations['multiplier'])[:, None],
            allocations['parlay_ev'][:, None], allocations['parlay_profit'][:, None])

    @classmethod
    def calculate_decay_portfolio(cls, opening, closing, opening_prices, winners, dates,
                                  opening_params, closing_params, rates, *, include_parlay=False, policy='decay'):
        """Shared frozen-event evaluation for decay searches and backtests.

        Caches align with fights; opening_prices is red/blue. Returned stakes
        are final bankroll fractions, adjusted only after the decay blend.
        """
        if policy not in ('decay', 'opening', 'closing', 'none'):
            raise ValueError('Unknown policy')
        rates = BettingParams.validate_decay_rates(rates)
        if rates is None:
            raise ValueError('Decay rates required')
        opening, closing, opening_prices = map(np.asarray, (opening, closing, opening_prices))
        winners = np.asarray(winners)
        dates = pd.to_datetime(dates).to_numpy()
        stakes = np.zeros(len(winners))
        events, tickets = [], []
        for date in np.unique(dates):
            ids = np.flatnonzero(dates == date)
            op, cl = opening[None, ids], closing[None, ids]
            if policy in ('opening', 'closing'):
                cache, params = (op, opening_params) if policy == 'opening' else (cl, closing_params)
                alloc = cls.scenario_allocations(cache, winners[ids], params, include_parlay)
            else:
                alloc = cls.decay_scenario_allocations(op, cl, opening_prices[ids], winners[ids],
                    opening_params, closing_params, rates, include_parlay and policy != 'none')
            stakes[ids] = alloc['moneyline_stakes'][0]
            ticket = float(alloc['parlay_stakes'][0] * alloc['multiplier'][0])
            ml = float(np.sum(stakes[ids] * cl[0, :, 2]))
            pl = float(ticket * alloc['parlay_profit'][0])
            events.append(dict(date=date, moneyline_return=ml, parlay_return=pl,
                               uncertainty_multiplier=float(alloc['uncertainty_multiplier'][0]),
                               event_return=ml + pl, parlay_stake=ticket, exposure=stakes[ids].sum() + ticket))
            if include_parlay and len(ids) >= 2 and policy != 'none':
                for idx in ids[alloc['legs'][0]]:
                    tickets.append(dict(date=date, fight_index=idx, fstar_parlay=ticket,
                                        parlay_net_odds=alloc['parlay_profit'][0]))
        return dict(stakes=stakes, events=pd.DataFrame(events), tickets=pd.DataFrame(tickets))

    def calculate_fight_returns(
        self, red_proba, risk_red_proba, se, winner_fold, val_idx, param,
    ):
        """Return moneyline, parlay, and combined returns for each event."""
        bets = self._prepare_bets(red_proba, risk_red_proba, se, winner_fold, val_idx)
        moneylines = self._moneyline_returns(bets, param)
        parlays = self.parlay_returns(bets, bets[['id', 'pred_win']], param)
        return self._event_returns(moneylines, parlays, param['z'])

    def calculate_allocated_returns(self, stakes, unit_profit):
        """Settle supplied fight allocations into chronological event returns.

        stakes are bankroll fractions; unit_profit is decimal odds minus one
        for wins and -1 for losses. Both arrays align positionally with dates.
        For already-finalized allocations only; decay callers must apply z and
        the exposure cap through decay_scenario_allocations before settlement.
        Events without bets remain in the output with zero return and exposure.
        """
        stakes = np.asarray(stakes, dtype=float)
        unit_profit = np.asarray(unit_profit, dtype=float)
        dates = pd.to_datetime(pd.Series(self.dates), errors='raise').dt.normalize()
        if stakes.shape != (len(dates),) or unit_profit.shape != stakes.shape:
            raise ValueError('Stakes and unit profits must align with event dates')
        if (dates.isna().any() or not np.isfinite(stakes).all()
                or not np.isfinite(unit_profit).all() or (stakes < 0).any()):
            raise ValueError('Need valid dates, finite unit profits, and finite nonnegative stakes')
        event_dates, event_ids = np.unique(dates.to_numpy(), return_inverse=True)
        return pd.DataFrame({
            'date': event_dates,
            'event_return': np.bincount(event_ids, weights=stakes * unit_profit,
                                        minlength=len(event_dates)),
            'exposure': np.bincount(event_ids, weights=stakes, minlength=len(event_dates)),
        })

    def _prepare_bets(self, red_proba, risk_red_proba, se, winners, val_idx):
        """Select each predicted winner and align its prices and probabilities."""
        red_proba = np.asarray(red_proba)
        risk_red_proba = np.asarray(risk_red_proba)
        choice = (red_proba >= .5).astype(int)
        rows = np.arange(len(choice))
        probability = np.where(choice == 1, red_proba, 1 - red_proba)
        real_odds = self.real_odds.iloc[val_idx].to_numpy()[rows, choice]
        fair_odds = self.fair_odds.iloc[val_idx].to_numpy()[rows, choice]
        return pd.DataFrame({
            'id': rows,
            'date': pd.to_datetime(pd.Series(self.dates).iloc[val_idx]).dt.normalize().to_numpy(),
            'choice_fighter_name': np.zeros(len(choice)),
            'choice_fighter_bool': choice,
            'choice_proba': probability,
            'choice_real_odds': real_odds,
            'choice_ev': expected_value(p=probability, o=real_odds),
            'choice_american_odds': [round(decimal_to_american(odds)) for odds in real_odds],
            'fair_odds': fair_odds,
            'risk_proba': np.where(choice == 1, risk_red_proba, 1 - risk_red_proba),
            'se': se,
            'pred_win': choice == np.asarray(winners),
            'is_void': np.isin(np.asarray(winners), [2, 3]),
        })

    def calculate_moneyline_stakes(self, red_proba, risk_red_proba, se,
                                   winner_fold, val_idx, param, *, include_parlay=False):
        """Return fixed moneyline allocations after uncertainty and event caps."""
        bets = self._prepare_bets(red_proba, risk_red_proba, se, winner_fold, val_idx)
        moneylines = self._moneyline_returns(bets, param)
        stakes = moneylines['fstar'].to_numpy(copy=True)
        no_parlay = pd.Series({'fstar': 0., 'selected_ids': []})
        tickets = (self.parlay_returns(bets, bets[['id', 'pred_win']], param)
                   if include_parlay else None)
        for date, group in moneylines.groupby('date', sort=False):
            ticket = no_parlay
            if tickets is not None:
                matching = tickets.loc[tickets['date'].eq(date)]
                if not matching.empty:
                    ticket = matching.iloc[0]
            single_risk, ticket_risk = self._bet_risk_multipliers(group, ticket, param['z'])
            exposure = np.sum(group['fstar'].to_numpy() * single_risk) + ticket['fstar'] * ticket_risk
            stakes[group.index.to_numpy()] *= single_risk * self._exposure_multiplier(exposure)
        return stakes

    def calculate_parlay_allocations(self, red_proba, risk_red_proba, se,
                                     winner_fold, val_idx, param):
        """Return top-EV ticket allocations after shared portfolio adjustments."""
        bets = self._prepare_bets(red_proba, risk_red_proba, se, winner_fold, val_idx)
        moneylines = self._moneyline_returns(bets, param)
        tickets = self.parlay_returns(bets, bets[['id', 'pred_win']], param)
        for index, ticket in tickets.iterrows():
            group = moneylines.loc[moneylines['date'].eq(ticket['date'])]
            single_risk, ticket_risk = self._bet_risk_multipliers(group, ticket, param['z'])
            exposure = np.sum(group['fstar'].to_numpy() * single_risk) + ticket['fstar'] * ticket_risk
            multiplier = ticket_risk * self._exposure_multiplier(exposure)
            tickets.loc[index, 'fstar'] *= multiplier
            tickets.loc[index, 'pct_return'] *= multiplier
        return tickets

    @staticmethod
    def _moneyline_returns(bets, param):
        """Apply price bounds, size moneylines, and settle their returns."""
        allowed = bets.choice_american_odds.between(
            param['moneyline_min_american'], param.get('moneyline_max_american', np.inf)
        )
        kelly = np.array([
            kelly_edge(p, odds) for p, odds in zip(bets.choice_proba, bets.fair_odds)
        ])
        sizing_input = pd.DataFrame({
            'f_star_unscaled': np.where(allowed, kelly, 0.),
            'p': bets.choice_proba,
            'fair_odds': bets.fair_odds,
            'real_odds': bets.choice_real_odds,
            'ev': bets.choice_ev,
        })
        sizing = run_per_bet_scaling(
            bets_df=sizing_input, max_drawdown=param['mdd'], bankroll=500, N=param['N']
        )
        results = bets.copy()
        stakes = sizing['fstar_scaled'].to_numpy()
        results['fstar'] = stakes
        results['pct_return'] = np.where(
            stakes > 0,
            stakes * BettingReturnCalculator.moneyline_unit_profit(
                bets.choice_real_odds, 1, np.where(bets.get('is_void', False), 2, bets.pred_win.astype(int))),
            np.nan,
        )
        return results

    def _event_returns(self, moneylines, parlays, z):
        """Apply individual risk corrections and one joint bankroll cap."""
        components = []
        for date, group in moneylines.groupby('date', sort=False):
            ticket = parlays.loc[parlays.date.eq(date)].iloc[0]
            single_risk, ticket_risk = self._bet_risk_multipliers(group, ticket, z)
            moneyline_return = np.sum(group.pct_return.fillna(0).to_numpy() * single_risk)
            parlay_return = (ticket.pct_return if pd.notna(ticket.pct_return) else 0.) * ticket_risk
            exposure = np.sum(group.fstar.to_numpy() * single_risk) + ticket.fstar * ticket_risk
            cap = self._exposure_multiplier(exposure)
            components.append({
                'date': date,
                'moneyline_return': float(moneyline_return * cap),
                'parlay_return': float(parlay_return * cap),
                'event_return': float((moneyline_return + parlay_return) * cap),
            })
        return pd.DataFrame(components)

    @classmethod
    def _bet_risk_multipliers(cls, bets, ticket, z):
        """Return independent singles ratios and the ticket-only leg ratio."""
        if z is None:
            return np.ones(len(bets)), 1.
        market_probability = 1 / bets.fair_odds.to_numpy()
        probability = bets.risk_proba.to_numpy()
        raw = np.maximum(probability - market_probability, 0.)
        conservative = np.maximum(probability - z * bets.se.to_numpy() - market_probability, 0.)
        singles = cls.risk_multiplier(bets.fstar, raw, conservative)
        chosen = np.isin(bets.id.to_numpy(), ticket.selected_ids)
        ticket_risk = float(cls.ticket_risk_multiplier(raw[chosen], conservative[chosen])) if chosen.any() else 1.
        return singles, ticket_risk

    @staticmethod
    def _exposure_multiplier(exposure):
        """Cap the adjusted event stake at the available bankroll."""
        return min(1., 1. / exposure) if exposure > 0 else 1.

    def parlay_returns(self, dat, winner_id, param):
        """Select and settle the two highest-EV legs per event."""

        pct_returns = []
        for date, group in dat.groupby('date'):
            selected = group.sort_values(
                'choice_ev', ascending=False, kind='stable'
            ).head(2)

            selected_ids = selected['id'].tolist()
            if len(selected) < 2:
                pct_returns.append({
                    'date': date,
                    'pct_return': np.nan,
                    'fstar': 0.0,
                    'selected_ids': [],
                })
                continue

            parlay_df = parlay_top_ev(
                data=selected,
                bankroll=500,
                type='open',
                top_n=list(range(len(selected))),
                parlay_mdd=param['mdd_parlay'],
                N=param['N_parlay']
            )
            if parlay_df.isna().all().all():
                pct_returns.append({
                    'date': date,
                    'pct_return': np.nan,
                    'fstar': 0.0,
                    'selected_ids': [],
                })
                continue

            # parlay_top_ev returns ticket fields, not fight IDs. Settle the
            # actual selected legs rather than merging on a discarded column.
            leg_outcomes = winner_id.set_index('id')['pred_win'].reindex(selected_ids)
            parlay_fstar = parlay_df['parlay_fstar_open'].iloc[0]
            parlay_net_odds = parlay_df['parlay_odds_open'].iloc[0]

            if not np.isfinite(parlay_fstar) or parlay_fstar <= 0:
                pct_return_parlay = np.nan
            else:
                outcomes = np.where(selected.get('is_void', False), 2,
                                    leg_outcomes.to_numpy(dtype=float))
                pct_return_parlay = parlay_fstar * float(self.parlay_unit_profit(
                    parlay_net_odds + 1., np.ones(len(selected)), outcomes))

            pct_returns.append({
                'date': date,
                'pct_return': pct_return_parlay,
                'fstar': parlay_fstar if pd.notna(pct_return_parlay) else 0.0,
                'selected_ids': selected_ids,
            })

        return pd.DataFrame(pct_returns)
