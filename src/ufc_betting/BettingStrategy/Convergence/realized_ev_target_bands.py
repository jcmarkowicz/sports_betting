"""Realized paths filtered by natural EV bands; no target-based stake changes."""
import json
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from ufc_betting.config import settings
from ufc_betting.BettingStrategy.betting_returns import BettingReturnCalculator
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.MonteCarlo.mc_engine import advance_bankroll
from ufc_betting.BettingStrategy.Convergence.model_input import predict_probabilities
from ufc_betting.Models.LogisticRegression.odds_input import replace_opening_odds
from ufc_betting.BettingStrategy.Convergence.odds_scenarios import interpolate_prices, sample_prices
from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache, event_values


def aggregate(weight, ev, profit, raw_edge, conservative_edge):
    """Compatibility adapter for shared calculator scenario settlement."""
    stakes = BettingReturnCalculator.adjusted_stakes(weight, raw_edge, conservative_edge)
    return BettingReturnCalculator.settle_allocations(stakes, ev, profit)


def qualifying(expected, realized, opening_ev, target_pct, tolerance_pp=1.):
    """Select natural EV within target +/- tolerance; NEVER adjust stakes."""
    if opening_ev <= 1e-12:
        return np.array([], dtype=int), np.array([])
    ratios = 100 * np.asarray(expected) / opening_ev
    ids = np.flatnonzero(np.abs(ratios - target_pct) <= tolerance_pp + 1e-10)
    return ids, np.asarray(realized)[ids]


def advance(bankroll, returns, floor=100):
    return advance_bankroll(bankroll, returns, floor=floor)


def build_price_cache(frame, opening_prices, closing_prices, model,
                      opening_params, closing_params, *, scaler=None,
                      closing_odds='close1', include_opening_policy=False):
    """Cache predictions and stakes at 101 points from closing to opening prices.

    Returns (closing_policy_cache, opening_policy_cache, opening_benchmark).
    Policy caches have shape (fights, 101, 8); the optional opening-policy
    cache is None when disabled. The benchmark has shape (fights, 8).
    Supply a loaded model/scaler and price arrays aligned with frame rows.
    """
    opening_params = BettingParams.from_value(opening_params)
    closing_params = BettingParams.from_value(closing_params)
    cache = np.zeros((len(frame), 101, 8))
    opening_policy_cache = np.zeros_like(cache) if include_opening_policy else None
    for shift in range(101):
        prices = interpolate_prices(closing_prices, opening_prices, shift / 100)
        features = replace_opening_odds(frame, closing_odds, prices=prices)
        data = predict_probabilities(features, model, scaler_path=scaler)
        cache[:, shift, :] = fight_cache(data, closing_params, odds_type=closing_odds)
        if include_opening_policy:
            opening_policy_cache[:, shift, :] = fight_cache(data, opening_params, odds_type=closing_odds)
    opening_benchmark = (opening_policy_cache[:, -1, :].copy() if include_opening_policy
                         else fight_cache(data, opening_params, odds_type=closing_odds))
    return cache, opening_policy_cache, opening_benchmark


def evaluate_event_scenarios(group, opening_prices, closing_prices, cache,
                             opening_cache, betting_params: BettingParams, *,
                             samples, rng, parlay=False, opening_policy_cache=None):
    """Sample one event's prices and calculate scenario and benchmark returns."""
    params = BettingParams.from_value(betting_params)
    if params.opening_params is None:
        raise ValueError('BettingParams must include opening_params for the benchmark')
    sampled = sample_prices(opening_prices, closing_prices, samples, random_state=rng)
    vectors = sampled['movement_indices']
    rows = np.arange(len(group))[None, :]
    selected = cache[rows, vectors, :]
    if params.decay_rates is None:
        expected, realized, exposure = event_values(selected, group, params, parlay)
    else:
        if opening_policy_cache is None:
            raise ValueError('Decay evaluation requires the opening-policy cache')
        expected, realized, exposure = BettingReturnCalculator.calculate_decay_scenario_returns(
            opening_policy_cache[rows, vectors, :], selected, opening_prices,
            group.winner.to_numpy(), params.opening_params, params, params.decay_rates, parlay,
        )
    opening_expected, opening_realized, _ = event_values(
        opening_cache[None, :, :], group, params.opening_params, parlay)
    return dict(movement_indices=vectors, expected=expected, realized=realized,
                exposure=exposure, opening_ev=float(opening_expected[0]),
                opening_return=float(opening_realized[0]))


def sample_target_paths(balances, expected, realized, opening_ev, target, rng):
    """Choose qualifying scenarios and settle paths; no match means no bet."""
    eligible, returns = qualifying(expected, realized, opening_ev, target)
    chosen = np.full(len(balances), -1, dtype=np.int32)
    updated = np.asarray(balances, dtype=float).copy()
    if len(eligible):
        draws = rng.integers(0, len(eligible), len(balances))
        chosen = eligible[draws]
        updated = advance(updated, returns[draws])
    return dict(balances=updated, chosen_ids=chosen, eligible=eligible, returns=returns)


def event_diagnostics(date, target, expected, opening_ev, eligible, returns):
    """Summarize EV qualification and realized-return ranges for one target."""
    ratios = 100 * expected[eligible] / opening_ev if len(eligible) else np.array([])
    status = ('qualified' if len(eligible) else
              'undefined_opening_ev' if opening_ev <= 1e-12 else 'no_sampled_match_no_bet')
    return dict(date=date, target_pct_opening_ev=target, candidate_count=len(expected),
                qualifying_unique_combinations=len(eligible), opening_ev_pct=opening_ev * 100,
                tolerance_pp=1., status=status,
                achieved_ev_ratio_min=ratios.min() if len(ratios) else np.nan,
                achieved_ev_ratio_max=ratios.max() if len(ratios) else np.nan,
                scenario_realized_return_min_pct=returns.min() * 100 if len(returns) else np.nan,
                scenario_realized_return_max_pct=returns.max() * 100 if len(returns) else np.nan)


def run(samples=50000, n_paths=5000, seed=42, parlay=False, opening_params=None, closing_params=None, output_dir=None, closing_odds='close1', decay_rates=None, *, model_path, validation_data, opening_data, scaler_path=None):
    """Evaluate supplied validation/opening DataFrames with a saved Logit model.

    run(model_path='model.pkl', validation_data=close_df, opening_data=open_df,
        scaler_path='scaler.pkl', opening_params=op, closing_params=cl)
    """
    if closing_odds not in ('close1', 'close2'):
        raise ValueError("closing_odds must be 'close1' or 'close2'")
    from pathlib import Path
    closing_params = BettingParams.from_value(closing_params) if closing_params is not None else BettingParams()
    opening_params = (BettingParams.from_value(opening_params) if opening_params is not None
                      else closing_params.opening_params or BettingParams())
    if decay_rates is None:
        decay_rates = closing_params.decay_rates
    betting_params = closing_params.with_decay(decay_rates, opening_params)
    decay_rates = betting_params.decay_rates
    if samples < 101 or n_paths < 1:
        raise ValueError("samples must be >=101 and n_paths >=1")
    if decay_rates is not None:
        if parlay and 'k_parlay_down' not in decay_rates:
            raise ValueError('Parlay convergence requires all four decay rates')
    root = settings.data_dir / "model_results"
    out = root / "opening_model_ev_targets" / "natural_ev_scenario_bands"
    if parlay:
        out = out.parent / "natural_parlay_ev_scenario_bands"
    if output_dir is not None:
        out = Path(output_dir)
    elif closing_odds == 'close2':
        out = out.parent / (out.name + '_close2')
    if decay_rates is not None and output_dir is None:
        out = out.parent / (out.name + "_decay")
    out.mkdir(parents=True, exist_ok=True)
    keys = ["date", "fighter_red", "fighter_blue"]
    frame = validation_data.copy()
    opening = opening_data.copy()
    for data in (frame, opening):
        data['date'] = pd.to_datetime(data.date, errors='raise').dt.normalize()
    frame = frame.sort_values(keys).reset_index(drop=True)
    opening = opening.sort_values(keys).reset_index(drop=True)
    if frame.duplicated(keys).any() or not frame[keys + ["winner"]].equals(opening[keys + ["winner"]]):
        raise ValueError("Fight inputs/outcomes must align")
    frame.date = pd.to_datetime(frame.date)
    if not frame.winner.isin([0, 1]).all():
        raise ValueError("Unsettled outcomes must be handled explicitly")
    close_prices = frame[[f"dec_{closing_odds}_red", f"dec_{closing_odds}_blue"]].to_numpy()
    open_prices = opening[["dec_open_red", "dec_open_blue"]].to_numpy()
    model = sm.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path is not None else None
    cache, opening_policy_cache, opening_cache = build_price_cache(
        frame, open_prices, close_prices, model, opening_params, closing_params,
        scaler=scaler, closing_odds=closing_odds,
        include_opening_policy=decay_rates is not None,
    )
    rng = np.random.default_rng(seed)
    targets = [50,60,70,80,90,100]
    dates = sorted(frame.date.unique())
    paths = np.full((len(targets),n_paths,len(dates)+1),500.)
    chosen_ids = np.full((len(targets),n_paths,len(dates)),-1,dtype=np.int32)
    benchmark = [500.]
    pools, summary, diagnostics = {}, [], []
    for event_no,date in enumerate(dates):
        ids = np.flatnonzero(frame.date.eq(date))
        event = evaluate_event_scenarios(
            frame.iloc[ids], open_prices[ids], close_prices[ids], cache[ids],
            opening_cache[ids], betting_params, samples=samples, rng=rng, parlay=parlay,
            opening_policy_cache=None if opening_policy_cache is None else opening_policy_cache[ids],
        )
        event_key = str(pd.Timestamp(date).date())
        pools[event_key] = event['movement_indices'].astype(np.uint8)
        benchmark.append(float(advance(benchmark[-1], event['opening_return'])))
        for k, target in enumerate(targets):
            selection = sample_target_paths(
                paths[k, :, event_no], event['expected'], event['realized'],
                event['opening_ev'], target, rng,
            )
            chosen_ids[k, :, event_no] = selection['chosen_ids']
            paths[k, :, event_no + 1] = selection['balances']
            diagnostics.append(event_diagnostics(
                date, target, event['expected'], event['opening_ev'],
                selection['eligible'], selection['returns'],
            ))
        if parlay:
            print(f'Completed parlay event {event_no+1}/{len(dates)}: {event_key}', flush=True)
    for k,target in enumerate(targets):
        for j in range(len(dates)+1):
            values=paths[k,:,j]
            q=np.quantile(values,[.025,.5,.95,.975])
            summary.append({"event_number":j,"date":str(pd.Timestamp(dates[j-1]).date()) if j else "start",
                "target_pct_opening_ev":target,"p025_bankroll":q[0],"median_bankroll":q[1],
                "p95_bankroll":q[2],"p975_bankroll":q[3],"mean_bankroll":values.mean(),
                "fraction_below_floor":float((values<100).mean())})
    summary=pd.DataFrame(summary)
    summary.to_csv(out / "bankroll_percentiles.csv",index=False)
    summary.loc[summary.event_number.eq(len(dates))].to_csv(out / "ending_bankroll_percentiles.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out / "event_qualification_counts.csv",index=False)
    np.savez_compressed(out / "bankroll_paths.npz",bankroll=paths,targets=targets,dates=np.array(dates,dtype='datetime64[D]'),scenario_ids=chosen_ids)
    np.savez_compressed(out / "candidate_movement_vectors.npz",**pools)
    frame[keys + ["winner"]].to_csv(out / "fight_order.csv",index=False)
    pd.DataFrame({"event_number":range(len(benchmark)),"realized_opening_bankroll":benchmark}).to_csv(out / "opening_benchmark.csv",index=False)
    metadata={"seed":seed,"candidate_combinations_per_event":samples,"paths_per_ev_target":n_paths,"events":len(dates),
        "model":str(model_path),"scaler":str(scaler_path) if scaler_path is not None else None,"mdd":.4,"N":250,"z":.5,"parlays":False,"initial_bankroll":500,"bankroll_floor":100,
        "settlement":"Actual historical winners at sampled hypothetical offered prices. No randomized outcomes.",
        "qualification":"Natural EV ratio within target +/-1 percentage point. NO target-based stake changes. Recompute only backtest uncertainty and exposure for each combination.",
        "missing_matches":"No sampled match means no bet for that event/target; bankroll stays unchanged. Does not prove mathematical impossibility. Coverage reported separately.",
        "sampling":"Unique ordered fight vectors on 1% grid using prior beta-mixture sampler, includes all uniform scenarios. Uniform draws with replacement from qualifying pools, independently per event.",
        "bands":"Pointwise 2.5th-97.5th percentiles, median, and 95th percentile across paths. Not simultaneous coverage or future-return confidence intervals.",
        "floor":"Stop placing bets after bankroll falls below $100; no top-ups. EV target applies while active.",
        "zero_opening_ev":"No bet and unchanged bankroll; no meaningful EV ratio.",
        "limitations":"Hypothetical prices, fixed known outcomes, sampling-design dependent bands. Endpoints and model were explored retrospectively. This is scenario sensitivity, not a probability forecast."}
    if parlay:
        metadata.update(parlays=True, parlay_mdd=.4, N_parlay=1490,
            selection='Top two choice_ev picks, reselected for each scenario; probabilities multiplied as in parlay_top_ev.',
            ev_definition='Parlay stake fraction times (product of model probabilities times product of offered odds minus one), divided by same-event opening parlay expected bankroll return.',
            account='Only parlay component P&L compounded; shared uncertainty/exposure sizing includes moneylines as in simulate_kelly. Not the combined strategy bankroll.',
            calculator='BettingReturnCalculator: shared moneyline/parlay sizing and settlement.')
    metadata.update(decay_rates=decay_rates, adjust_mdd_by_edge=False, closing_odds=closing_odds, opening_params=opening_params.to_dict(), closing_params=betting_params.to_dict(),
        parameter_scope="Opening parameters apply to benchmark; closing parameters apply to all interpolated scenarios. Same supplied model for both.")
    for key in ("mdd", "N", "z", "parlay_mdd", "N_parlay"):
        metadata.pop(key, None)
    (out / "methodology.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    coverage = pd.DataFrame(diagnostics).assign(qualified=lambda d:d.status.eq('qualified')).groupby('target_pct_opening_ev').qualified.agg(['sum','count'])
    coverage.columns=['qualifying_events','total_events']
    coverage.to_csv(out / 'event_coverage.csv')
    render(summary,benchmark,out,coverage,parlay=parlay, params=(opening_params,closing_params), n_paths=n_paths, closing_odds=closing_odds, decay_rates=decay_rates)
    print(summary.loc[summary.event_number.eq(len(dates))].to_string(index=False))
    print('Minimum qualifying combinations for positive targets:',pd.DataFrame(diagnostics).query('opening_ev_pct > 0').qualifying_unique_combinations.min())
    print(coverage.to_string())
    return dict(summary=summary, coverage=coverage, output_dir=out, methodology=metadata)


def render(summary, benchmark, out, coverage, parlay=False, params=None, n_paths=5000, closing_odds='close1', decay_rates=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,(target,g) in zip(axes.flat,summary.groupby('target_pct_opening_ev')):
        ax.fill_between(g.event_number,g.p025_bankroll,g.p975_bankroll,color='#2366af',alpha=.2,label='Central 95% of sampled paths')
        ax.plot(g.event_number,g.median_bankroll,color='#2366af',lw=2,label='Median')
        ax.plot(g.event_number,g.p95_bankroll,color='#c25916',ls=':',lw=2,label='95th percentile')
        ax.plot(range(len(benchmark)),benchmark,color='black',ls='--',lw=1.2,label='Actual opening-price benchmark')
        ax.set(title=f'{target}% +/- 1 pp | {coverage.loc[target,"qualifying_events"]}/{coverage.loc[target,"total_events"]} events qualify',xlabel='Completed test events',ylabel='Realized bankroll ($, log scale)',yscale='log')
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Actual winners, sampled odds combinations | 5,000 bankroll paths per EV target | $500 start',fontsize=16)
    fig.supxlabel('Natural EV bins only: NO target-based stake adjustment. No sampled match means no bet; opening benchmark uses all events.\nPointwise scenario percentiles, not forecasts. Backtest MDD .40, N 250, z .5, exposure cap; stop below $100.',fontsize=10)
    if parlay:
        fig.suptitle('Top-two-EV parlays | 5,000 sampled bankroll paths per EV target | $500 start',fontsize=16)
        fig.supxlabel('Natural parlay EV bins: no target-based stake changes. No match means no bet. Opening parlay benchmark included.\nParlay component only; shared backtest risk adjustments retained. Parlay MDD .40, N 1490, z .5; stop below $100. Scenario bands, not forecasts.',fontsize=9)
    if params is not None:
        op, cl = params
        label = 'Top-two-EV parlays' if parlay else 'Moneylines'
        fig.suptitle(f'{label} | {closing_odds} toward opening | {n_paths:,} paths per EV target | $500 start', fontsize=16)
        def describe(p):
            return f'z={p.z}, min ML={p.min_moneyline}, N={p.N}, parlay N={p.N_parlay}, MDD={p.mdd}, parlay MDD={p.mdd_parlay}'
        fig.supxlabel('Opening: '+describe(op)+'\nClosing scenarios: '+describe(cl)+'\nNatural EV bins +/-1 pp; no target stake adjustment. No match: no bet. Scenario bands, not forecasts; stop below $100.',fontsize=8)
    if decay_rates is not None:
        fig.suptitle(('Parlay' if parlay else 'Moneyline') + f' exponential decay | {closing_odds} toward opening | {n_paths:,} paths', fontsize=16)
        fig.supxlabel('Fitted rates: '+str(decay_rates)+'\nNatural EV bins; fixed endpoint policies; no post-blend cap/rescaling. Opening benchmark uses matching sizing.\nNo qualifying sample: no bet. $500 start; stop below $100. Scenario percentiles, not future confidence intervals.', fontsize=8)
    fig.savefig(out / 'realized_bankroll_bands.png',dpi=150)
    plt.close(fig)


