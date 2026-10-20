"""Unshifted historical event EV ratios and realized-return regression."""
from pathlib import Path
from dataclasses import asdict
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ufc_betting.config import settings, config
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache, event_values


def historical_event_returns(opening_params, closing_params, strategy='moneyline', closing_odds='close1'):
    """One row/event. EV is model-implied; realized returns use actual winners.

    Prices are read unchanged from historical CSVs. Both snapshots use the saved
    opening model, as in the sensitivity analysis. Returns are percentages of
    pre-event bankroll, not percentages of money staked or cumulative returns.
    """
    if strategy not in ('moneyline', 'parlay', 'combined') or closing_odds not in ('close1', 'close2'):
        raise ValueError('Use moneyline/parlay/combined and close1/close2')
    op = BettingParams.from_value(opening_params)
    cl = BettingParams.from_value(closing_params)
    root = settings.data_dir / 'model_results'
    keys = ['date','fighter_red','fighter_blue']
    opening = pd.read_csv(root/'test_logit_open.csv').sort_values(keys).reset_index(drop=True)
    closing = pd.read_csv(root/f'test_logit_{closing_odds}.csv').sort_values(keys).reset_index(drop=True)
    if opening.duplicated(keys).any() or closing.duplicated(keys).any() or not opening[keys+['winner']].equals(closing[keys+['winner']]):
        raise ValueError('Historical fights and outcomes must match exactly')
    if not closing.winner.isin([0,1]).all():
        raise ValueError('Only settled binary outcomes supported')
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    caches = []
    for source, stage, params in ((opening,'open',op),(closing,closing_odds,cl)):
        prices = source[[f'dec_{stage}_red',f'dec_{stage}_blue']].to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 1).any():
            raise ValueError('Invalid historical prices')
        data = opening_model_at_close1(closing,prices,model,scaler)
        np.testing.assert_array_equal(prices,data[['dec_close1_red','dec_close1_blue']].to_numpy())
        caches.append(fight_cache(data,params))
    rows = []
    for date,group in closing.groupby('date',sort=True):
        ids = group.index.to_numpy()
        row = dict(date=date,fights=len(group))
        for label,cache,params in zip(('opening','closing'),caches,(op,cl)):
            if strategy == 'combined':
                from ufc_betting.Models.RiskManagement.parlay_ev_target_bands import evaluate_parlays
                ev,realized,exposure = evaluate_parlays(cache[ids][None,:,:],group,
                    mdd_parlay=params.mdd_parlay,N_parlay=params.N_parlay,z=params.z,combined=True)
            else:
                ev,realized,exposure = event_values(cache[ids][None,:,:],group,params,strategy=='parlay')
            row.update({f'{label}_ev_pct':100*ev[0],f'{label}_return_pct':100*realized[0],f'{label}_exposure_pct':100*exposure[0]})
        row['pct_ev_of_open'] = 100*row['closing_ev_pct']/row['opening_ev_pct'] if row['opening_ev_pct'] > 1e-10 else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs.update(opening_params=asdict(op),closing_params=asdict(cl),strategy=strategy,closing_odds=closing_odds)
    return result


def fit_return_regression(events):
    """Descriptive event-level OLS with HC3 heteroskedasticity-robust errors."""
    valid = events.loc[np.isfinite(events.pct_ev_of_open) & np.isfinite(events.closing_return_pct)]
    if len(valid) < 3 or valid.pct_ev_of_open.nunique() < 2:
        raise ValueError('Regression needs at least 3 events and variation in EV ratio')
    return sm.OLS(valid.closing_return_pct,sm.add_constant(valid[['pct_ev_of_open']],has_constant='add')).fit(cov_type='HC3')


def plot_historical_ev(opening_params,closing_params,strategy='moneyline',closing_odds='close1',bins=20,output_dir=None):
    """Display histogram and regression; return event table, fit and figure path."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from IPython.display import Image, display
    events = historical_event_returns(opening_params,closing_params,strategy,closing_odds)
    valid = events.loc[np.isfinite(events.pct_ev_of_open)]
    fit = fit_return_regression(events)
    fig,axes = plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    axes[0].hist(valid.pct_ev_of_open,bins=bins,edgecolor='white',color='#2366af')
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set(xlabel='Closing EV / opening EV (%)',ylabel='Event count',title=f'Historical {closing_odds}: {len(valid)} events')
    axes[1].scatter(valid.pct_ev_of_open,valid.closing_return_pct,alpha=.65)
    x = np.linspace(valid.pct_ev_of_open.min(),valid.pct_ev_of_open.max(),200)
    prediction = fit.get_prediction(sm.add_constant(pd.DataFrame({'pct_ev_of_open':x}),has_constant='add')).summary_frame()
    axes[1].plot(x,prediction['mean'],color="#c25511",label='OLS fitted mean')
    axes[1].fill_between(x,prediction.mean_ci_lower,prediction.mean_ci_upper,alpha=.2,color='#c25916',label='95% HC3 mean confidence band')
    axes[1].set(xlabel='Closing EV / opening EV (%)',ylabel='Realized event return (% of bankroll)',title=f'R²={fit.rsquared:.3f}; slope={fit.params.iloc[1]:.3f}; p={fit.pvalues.iloc[1]:.3g}')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(100,color='gray',ls=':',lw=1)
        ax.grid(alpha=.15)
    fig.suptitle(f'{strategy.title()} | actual historical prices | fixed opening model')
    fig.supxlabel(f'{len(events)-len(valid)} events excluded: opening EV zero/undefined. Ratios above 100% retained.\nEV uses model probabilities; realized returns use actual outcomes. Descriptive association, not an out-of-sample forecast.',fontsize=9)
    out = Path(output_dir) if output_dir else settings.data_dir/'model_results'/'notebook_odds_sensitivity'/'historical'/strategy
    out.mkdir(parents=True,exist_ok=True)
    path = out/'historical_ev_histogram_regression.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    events.to_csv(out/'historical_event_returns.csv',index=False)
    (out/'regression_summary.txt').write_text(fit.summary().as_text(),encoding='utf-8')
    (out/'parameters.json').write_text(json.dumps(events.attrs,indent=2),encoding='utf-8')
    display(Image(filename=str(path)))
    return dict(events=events,regression=fit,plot_path=path)


def plot_open_vs_close_ev(opening_params, close1_params, close2_params=None,
                          strategy='combined', output_dir=None):
    """Regress opening event EV on actual closing event EV, in bankroll percent.

    No odds shifting, EV ratios, or decay blending. Zero EV events are retained.
    close2_params defaults to close1_params, making that comparison explicit.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True,sharey=True,constrained_layout=True)
    outputs={}
    for ax,stage,params in zip(axes,('close1','close2'),
                             (close1_params,close1_params if close2_params is None else close2_params)):
        events=historical_event_returns(opening_params,params,strategy,stage)
        valid=events.loc[np.isfinite(events.closing_ev_pct)&np.isfinite(events.opening_ev_pct)]
        if len(valid)<3 or valid.closing_ev_pct.nunique()<2:
            raise ValueError(f'{stage}: insufficient EV variation for regression')
        fit=sm.OLS(valid.opening_ev_pct,sm.add_constant(valid[['closing_ev_pct']],has_constant='add')).fit(cov_type='HC3')
        x=np.linspace(valid.closing_ev_pct.min(),valid.closing_ev_pct.max(),200)
        prediction=fit.get_prediction(sm.add_constant(pd.DataFrame({'closing_ev_pct':x}),has_constant='add')).summary_frame()
        ax.scatter(valid.closing_ev_pct,valid.opening_ev_pct,s=24,alpha=.65)
        ax.plot(x,prediction['mean'],color='#c25916',label='OLS fitted mean')
        ax.fill_between(x,prediction.mean_ci_lower,prediction.mean_ci_upper,alpha=.2,color='#c25916',label='95% HC3 mean band')
        ax.plot(x,x,color='gray',ls=':',label='Equal opening and closing EV')
        ax.set(xlabel=f'{stage} event EV (% of bankroll)',ylabel='Opening event EV (% of bankroll)',
            title=f'{stage} | n={len(valid)} | R²={fit.rsquared:.3f}\nOpen EV = {fit.params.iloc[0]:.3f} + {fit.params.iloc[1]:.3f} × close EV')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        outputs[stage]=dict(events=events,regression=fit)
    fig.suptitle(f'{strategy.title()} expected event returns: opening versus historical closing prices')
    fig.supxlabel('Fixed opening model; snapshot-specific betting parameters. Zero-EV events retained. No odds interpolation or decay. Descriptive regression.',fontsize=9)
    out=Path(output_dir) if output_dir else settings.data_dir/'model_results'/'notebook_odds_sensitivity'/'open_vs_close_ev'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'open_vs_close_ev.png';fig.savefig(path,dpi=150);plt.close(fig)
    for stage,result in outputs.items():
        result['events'].to_csv(out/f'{stage}_events.csv',index=False)
        (out/f'{stage}_regression.txt').write_text(result['regression'].summary().as_text(),encoding='utf-8')
    display(Image(filename=str(path)))
    return outputs


def plot_line_movement_histograms(data_dir, bins=40, output_dir=None):
    data_dir = Path(data_dir)
    keys = ["date", "fighter_red", "fighter_blue"]
    opening = pd.read_csv(data_dir / "test_logit_open.csv")
    if opening.duplicated(keys).any():
        raise ValueError("Duplicate opening fights")
    movements = []
    for stage in ("close1", "close2"):
        closing = pd.read_csv(data_dir / f"test_logit_{stage}.csv")
        if closing.duplicated(keys).any():
            raise ValueError(f"Duplicate {stage} fights")
        cols = [f"dec_{stage}_{side}" for side in ("red", "blue")]
        joined = opening[keys + ["dec_open_red", "dec_open_blue"]].merge(
            closing[keys + cols], on=keys, how="outer", validate="one_to_one", indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"Opening and {stage} fight cohorts differ")
        for side in ("red", "blue"):
            op = joined[f"dec_open_{side}"].to_numpy(float)
            cl = joined[f"dec_{stage}_{side}"].to_numpy(float)
            if not np.isfinite([op, cl]).all() or min(op.min(), cl.min()) <= 1:
                raise ValueError("Invalid historical decimal odds")
            part = joined[keys].copy()
            part["fighter"] = joined[f"fighter_{side}"]
            part["side"] = side
            part["closing_odds"] = stage
            part["opening_decimal"] = op
            part["closing_decimal"] = cl
            part["movement"] = cl - op
            movements.append(part)
    movements = pd.concat(movements, ignore_index=True)
    edges = np.histogram_bin_edges(movements.movement, bins=bins)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    for ax, stage in zip(axes, ("close1", "close2")):
        values = movements.loc[movements.closing_odds.eq(stage), "movement"]
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == len(values)
        ax.hist(values, bins=edges, color="#2366af", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=f"Opening to {stage} | {len(values):,} fighter prices",
               xlabel="Closing minus opening decimal odds", ylabel="Count")
        ax.text(.97, .95, f"Mean: {values.mean():.3f}\nMedian: {values.median():.3f}",
                transform=ax.transAxes, ha="right", va="top")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Historical line movements | matched test fights, both fighters")
    fig.supxlabel("Positive = larger payout at closing. Negative = smaller payout. Full observed range retained.", fontsize=9)
    out = Path(output_dir) if output_dir is not None else data_dir / "notebook_odds_sensitivity" / "line_movements"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "historical_line_movements.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    movements.to_csv(out / "historical_line_movements.csv", index=False)
    display(Image(filename=str(path)))
    return movements

def plot_six_day_moneyline_ev(snapshot_dir, opening_params, closing_params, bins=15,
                            exclude_event_dates=None):
    """Plot six-day event EV ratios; exclusions affect only the histogram."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    snapshot_dir = Path(snapshot_dir)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv")
    events = snapshot_moneyline_ev(snapshot_dir, opening_params, closing_params, manifest)
    excluded_dates = set(exclude_event_dates or [])
    plot_events = events.loc[~events.event_date.isin(excluded_dates)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True, layout="constrained")
    finite = plot_events.loc[np.isfinite(plot_events.pct_ev_of_open), "pct_ev_of_open"]
    edges = np.histogram_bin_edges(finite, bins=bins) if len(finite) else np.linspace(0, 100, bins+1)
    for ax, stage in zip(axes, ("close1", "close2")):
        group = events.loc[events.odds_type.eq(stage)]
        ratios = group.loc[np.isfinite(group.pct_ev_of_open)
                           & ~group.event_date.isin(excluded_dates), "pct_ev_of_open"]
        ax.hist(ratios, bins=edges, edgecolor="white", color="#3579ad")
        ax.axvline(100, color="gray", linestyle="--", label="Opening EV (100%)")
        if len(ratios):
            ax.axvline(ratios.median(), color="#c44e52", label=f"Median: {ratios.median():.1f}%")
        ax.set(title=f"{stage}: {len(ratios)} events | {len(group)-len(ratios)} excluded",
               xlabel="Six-day moneyline event EV / opening event EV (%)", ylabel="Event count")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Moneyline EV six days before event | recorded prices, fixed opening model")
    events.to_csv(snapshot_dir / "moneyline_event_ev.csv", index=False)
    fig.savefig(snapshot_dir / "moneyline_pct_open_ev_histogram.png", dpi=150)
    return {"events": events, "figure": fig, "manifest": manifest,
            "excluded_events": events.loc[events.event_date.isin(excluded_dates)]}


def snapshot_moneyline_ev(snapshot_dir, opening_params, closing_params, manifest=None):
    """Compare recorded close1/close2 to opening using the same eligible fights.

    Fixed saved opening model, recomputed market features/probabilities/fair odds;
    separate opening/closing sizing parameters. No decay or parlay allocations.
    Outcomes are not used for EV. Missing features/prices exclude matched fights.
    """
    import joblib
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from ufc_betting.config import config
    from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
    from ufc_betting.BettingStrategy.betting_params import BettingParams
    from ufc_betting.BettingStrategy.Convergence.scenario_values import fight_cache, event_values
    snapshot_dir = Path(snapshot_dir)
    manifest = pd.read_csv(snapshot_dir / "manifest.csv") if manifest is None else manifest.copy()
    op = BettingParams.from_value(opening_params)
    cl = BettingParams.from_value(closing_params)
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    numeric = [c for c in scaler.feature_names_in_ if c != "proba_fair_open_diff"]
    rows = []
    for record in manifest.itertuples():
        original = pd.read_csv(snapshot_dir / record.filename)
        for stage in ("close1", "close2"):
            price_cols = [f"dec_{s}_{side}" for s in ("open", stage) for side in ("red", "blue")]
            required = list(dict.fromkeys(numeric + ["elo_pred"] + price_cols))
            values = original.reindex(columns=required).apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(values).all(axis=1) & values[price_cols].gt(1).all(axis=1)
            frame = original.loc[valid].copy().reset_index(drop=True)
            row = dict(event_date=record.event_date, odds_type=stage, fights=len(frame),
                       excluded_fights=len(original)-len(frame), filename=record.filename,
                       opening_ev_pct=np.nan, six_day_ev_pct=np.nan, pct_ev_of_open=np.nan)
            if len(frame):
                # The shared cache includes an unused realized-return column.
                # Integer sentinel satisfies the legacy cache's outcome cast.
                # Its realized-return output is discarded; sizing does not use it.
                frame["winner"] = -1
                for label, price_stage, params in (("opening", "open", op), ("six_day", stage, cl)):
                    prices = frame[[f"dec_{price_stage}_red", f"dec_{price_stage}_blue"]].to_numpy(float)
                    data = opening_model_at_close1(frame, prices, model, scaler)
                    if not np.isfinite(data[["proba_red", "proba_blue", "proba_se"]]).all().all():
                        raise ValueError(f"Nonfinite predictions for {record.event_date}, {stage}")
                    cache = fight_cache(data, params)
                    ev, _, _ = event_values(cache[None, :, :], data, params, False)
                    row[f"{label}_ev_pct"] = 100 * ev[0]
                if row["opening_ev_pct"] > 1e-10:
                    row["pct_ev_of_open"] = 100 * row["six_day_ev_pct"] / row["opening_ev_pct"]
            rows.append(row)
    events = pd.DataFrame(rows)
    return events




def select_six_day_snapshots(source_dir, destination_dir):
    source, destination = Path(source_dir), Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source / "manifest.csv")
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    manifest["days_before"] = (pd.to_datetime(manifest.event_date, utc=True)
                               - manifest.timestamp.dt.floor("D")).dt.days
    selected = manifest.loc[manifest.days_before.eq(6)].sort_values(
        ["timestamp", "commit", "original_path"]).drop_duplicates("event_date", keep="last")
    for row in selected.itertuples():
        target = destination / row.filename
        if target.exists() and target.read_bytes() != (source / row.filename).read_bytes():
            raise ValueError(f"Existing snapshot differs: {target}")
        shutil.copyfile(source / row.filename, target)
    missing = pd.DataFrame({"event_date": sorted(set(manifest.event_date) - set(selected.event_date))})
    selected.to_csv(destination / "manifest.csv", index=False)
    missing.to_csv(destination / "missing_six_day_events.csv", index=False)
    return selected, missing

