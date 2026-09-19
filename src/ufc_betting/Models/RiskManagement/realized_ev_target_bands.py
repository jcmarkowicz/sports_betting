"""Realized paths filtered by natural EV bands; no target-based stake changes."""
import json
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import interpolate_prices
from ufc_betting.Models.RiskManagement.uneven_odds_sensitivity import sample_movements
from ufc_betting.BettingStrategy.Backtest.backtest_functions import run_per_bet_scaling
from ufc_betting.BettingStrategy.kelly_scaling import kelly_edge


def aggregate(weight, ev, profit, raw_edge, conservative_edge):
    """Recompute uncertainty/exposure at event level for each uneven scenario."""
    raw = (weight * raw_edge).sum(axis=1)
    conservative = (weight * conservative_edge).sum(axis=1)
    uncertainty = np.divide(conservative, raw, out=np.zeros_like(raw), where=raw > 0)
    uncertainty = np.clip(uncertainty, 0, 1)
    exposure = weight.sum(axis=1) * uncertainty
    multiplier = uncertainty / np.maximum(exposure, 1)
    return ((weight * ev).sum(axis=1) * multiplier,
            (weight * profit).sum(axis=1) * multiplier, exposure / np.maximum(exposure, 1))


def qualifying(expected, realized, opening_ev, target_pct, tolerance_pp=1.):
    """Select natural EV within target +/- tolerance; NEVER adjust stakes."""
    if opening_ev <= 1e-12:
        return np.array([], dtype=int), np.array([])
    ratios = 100 * np.asarray(expected) / opening_ev
    ids = np.flatnonzero(np.abs(ratios - target_pct) <= tolerance_pp + 1e-10)
    return ids, np.asarray(realized)[ids]


def advance(bankroll, returns, floor=100):
    return np.where(bankroll >= floor, bankroll * (1 + returns), bankroll)


def run(samples=50000, n_paths=5000, seed=42, parlay=False):
    root = settings.data_dir / "model_results"
    out = root / "opening_model_ev_targets" / "natural_ev_scenario_bands"
    if parlay:
        out = out.parent / "natural_parlay_ev_scenario_bands"
        from ufc_betting.Models.RiskManagement.parlay_ev_target_bands import evaluate_parlays
    out.mkdir(parents=True, exist_ok=True)
    keys = ["date", "fighter_red", "fighter_blue"]
    frame = pd.read_csv(root / "test_logit_close1.csv").sort_values(keys).reset_index(drop=True)
    opening = pd.read_csv(root / "test_logit_open.csv").sort_values(keys).reset_index(drop=True)
    if frame.duplicated(keys).any() or not frame[keys + ["winner"]].equals(opening[keys + ["winner"]]):
        raise ValueError("Fight inputs/outcomes must align")
    frame.date = pd.to_datetime(frame.date)
    if not frame.winner.isin([0, 1]).all():
        raise ValueError("Unsettled outcomes must be handled explicitly")
    close_prices = frame[["dec_close1_red", "dec_close1_blue"]].to_numpy()
    open_prices = opening[["dec_open_red", "dec_open_blue"]].to_numpy()
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    # Cache fight x movement-grid quantities. Outcomes do not affect prediction,
    # stake sizing, candidate sampling, or EV qualification.
    cache = np.zeros((len(frame), 101, 8 if parlay else 5))
    for shift in range(101):
        prices = interpolate_prices(close_prices, open_prices, shift / 100)
        data = opening_model_at_close1(frame, prices, model, scaler)
        red = data.pred_winner.eq(1)
        p = np.where(red, data.proba_red, data.proba_blue)
        fair = np.where(red, data.dec_fair_close1_red, data.dec_fair_close1_blue)
        offered = np.where(red, data.dec_close1_red, data.dec_close1_blue)
        inputs = pd.DataFrame({"f_star_unscaled": [kelly_edge(a,b) for a,b in zip(p,fair)],
            "choice_proba":p,"choice_fair_odds":fair,"choice_real_odds":offered,
            "choice_ev":p*offered-1,"choice_idx":data.pred_winner,"winner":data.winner})
        sized,_ = run_per_bet_scaling(inputs,.4,500,250)
        cache[:,shift,0] = sized.f_star_scaled
        cache[:,shift,1] = p*offered-1
        cache[:,shift,2] = np.where(data.pred_winner.eq(data.winner),offered-1,-1)
        cache[:,shift,3] = np.maximum(p-1/fair,0)
        cache[:,shift,4] = np.maximum(p-.5*data.proba_se.to_numpy()-1/fair,0)
        if parlay:
            cache[:,shift,5] = p
            cache[:,shift,6] = offered
            cache[:,shift,7] = data.pred_winner
    rng = np.random.default_rng(seed)
    targets = [50,60,70,80,90,100]
    dates = sorted(frame.date.unique())
    paths = np.full((len(targets),n_paths,len(dates)+1),500.)
    chosen_ids = np.full((len(targets),n_paths,len(dates)),-1,dtype=np.int32)
    benchmark = [500.]
    pools, summary, diagnostics = {}, [], []
    for event_no,date in enumerate(dates):
        ids = np.flatnonzero(frame.date.eq(date))
        vectors = sample_movements(len(ids),samples,rng)
        event_key = str(pd.Timestamp(date).date())
        pools[event_key] = vectors.astype(np.uint8)
        selected = cache[ids[None,:],vectors,:]
        if parlay:
            expected, realized, exposure = evaluate_parlays(selected, frame.iloc[ids])
        else:
            expected, realized, exposure = aggregate(*[selected[:,:,i] for i in range(5)])
        opening_id = np.flatnonzero(np.all(vectors == 100,axis=1))[0]
        opening_ev = expected[opening_id]
        benchmark.append(float(advance(np.array([benchmark[-1]]),np.array([realized[opening_id]]))[0]))
        for k,target in enumerate(targets):
            eligible, returns = qualifying(expected,realized,opening_ev,target)
            if len(eligible):
                draws = rng.integers(0,len(eligible),n_paths)
                chosen_ids[k,:,event_no] = eligible[draws]
                paths[k,:,event_no+1] = advance(paths[k,:,event_no],returns[draws])
            else:
                # No price combination meets this strategy's EV filter: no bet.
                paths[k,:,event_no+1] = paths[k,:,event_no]
            ratios = 100 * expected[eligible] / opening_ev if len(eligible) else np.array([])
            diagnostics.append({"date":date,"target_pct_opening_ev":target,
                "candidate_count":len(vectors),"qualifying_unique_combinations":len(eligible) if opening_ev > 1e-12 else 0,
                "opening_ev_pct":opening_ev*100,"tolerance_pp":1.,
                "status":"qualified" if len(eligible) else ("undefined_opening_ev" if opening_ev<=1e-12 else "no_sampled_match_no_bet"),
                "achieved_ev_ratio_min":ratios.min() if len(ratios) else np.nan,
                "achieved_ev_ratio_max":ratios.max() if len(ratios) else np.nan,
                "scenario_realized_return_min_pct":returns.min()*100 if len(returns) else np.nan,
                "scenario_realized_return_max_pct":returns.max()*100 if len(returns) else np.nan})
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
        "model":"fixed saved opening model","mdd":.4,"N":250,"z":.5,"parlays":False,"initial_bankroll":500,"bankroll_floor":100,
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
            validation='Batch ticket results checked against parlay_top_ev for 12 scenarios per event.')
    (out / "methodology.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    coverage = pd.DataFrame(diagnostics).assign(qualified=lambda d:d.status.eq('qualified')).groupby('target_pct_opening_ev').qualified.agg(['sum','count'])
    coverage.columns=['qualifying_events','total_events']
    coverage.to_csv(out / 'event_coverage.csv')
    render(summary,benchmark,out,coverage,parlay=parlay)
    print(summary.loc[summary.event_number.eq(len(dates))].to_string(index=False))
    print('Minimum qualifying combinations for positive targets:',pd.DataFrame(diagnostics).query('opening_ev_pct > 0').qualifying_unique_combinations.min())
    print(coverage.to_string())


def render(summary, benchmark, out, coverage, parlay=False):
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
    fig.savefig(out / 'realized_bankroll_bands.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run()
