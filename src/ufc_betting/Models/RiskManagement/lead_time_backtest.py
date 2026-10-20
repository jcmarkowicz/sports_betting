"""Replay a fixed joint decay strategy at archived pre-event lead times."""
from pathlib import Path
import contextlib
import io
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _name(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_days_out(days_out):
    """Accept one positive integer or an iterable of distinct lead times."""
    days = (days_out,) if isinstance(days_out, (int, np.integer)) else tuple(days_out)
    if not days or any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1 for d in days):
        raise ValueError('days_out must contain positive integers')
    if len(set(days)) != len(days):
        raise ValueError('days_out must not contain duplicates')
    return tuple(int(d) for d in days)


def load_lead_time_split(data_dir, split='test'):
    """Load actual saved split rows; do not infer membership from date cutoffs."""
    files = {'test': 'test_set_data_open_with_open.csv',
             'validation': 'val_set_data_open_with_close1.csv'}
    if split not in files:
        raise ValueError('split must be test or validation')
    return pd.read_csv(Path(data_dir) / 'model_results' / files[split], parse_dates=['date'])


def run_split_lead_time_backtests(data_dir, decay_rates, split='test', days_out=(7, 5, 3), **kwargs):
    """Run any requested lead time(s) for the saved test or validation cohort."""
    inputs = load_lead_time_split(data_dir, split)
    kwargs.setdefault('output_dir', Path(data_dir) / 'model_results' / 'lead_time_backtests' / split)
    return run_lead_time_backtests(data_dir, decay_rates, days_out=days_out,
                                  event_data=inputs, **kwargs)


def load_lead_time_params(data_dir):
    """Same frozen controls loaded by kelly_sim_test; reject mixed controls."""
    result = {}
    for stage in ("open", "close1", "close2"):
        frame = pd.read_csv(Path(data_dir) / "search_results" / stage / "betting_data.csv")
        params = {}
        for name in ("mdd", "N", "mdd_parlay", "N_parlay", "z",
                     "moneyline_min_american", "moneyline_max_american"):
            col = "best_" + name
            if col not in frame and name == "moneyline_max_american":
                continue
            if frame[col].isna().any() or frame[col].nunique() != 1:
                raise ValueError(f"Expected one fixed {stage} {col}")
            params[name] = int(frame[col].iloc[0]) if name in ("N", "N_parlay") else float(frame[col].iloc[0])
        result[stage] = params
    return result


def prepare_lead_time_data(data_dir, scaler, days_out=(7, 5, 3), event_data=None):
    """Exact UTC days; common settled fight cohort, outcome-only historical join."""
    from ufc_betting.config import settings
    source = Path(data_dir) / "intermediate_odds_stats"
    manifest = pd.read_csv(source / "manifest.csv")
    days_out = normalize_days_out(days_out)
    allowed_keys = None
    split_outcomes = {}
    requested_dates = sorted(manifest.event_date.unique())
    if event_data is not None:
        inputs = pd.read_csv(event_data) if isinstance(event_data, (str, Path)) else event_data.copy()
        date_col = 'date' if 'date' in inputs else 'event_date'
        inputs['date'] = pd.to_datetime(inputs[date_col], errors='raise').dt.strftime('%Y-%m-%d')
        requested_dates = sorted(inputs.date.unique())
        if {'fighter_red', 'fighter_blue'} <= set(inputs):
            allowed_keys = set()
            for row in inputs.itertuples():
                red, blue = _name(row.fighter_red), _name(row.fighter_blue)
                key = (row.date, *sorted((red, blue)))
                allowed_keys.add('|'.join(key))
                winner = getattr(row, 'winner', None)
                if winner in (0, 1):
                    split_outcomes[key] = red if winner == 1 else blue
        manifest = manifest.loc[manifest.event_date.isin(requested_dates)].copy()
    manifest["timestamp"] = pd.to_datetime(manifest.commit_datetime_utc, utc=True)
    manifest["days_out"] = (pd.to_datetime(manifest.event_date, utc=True)
                            - manifest.timestamp.dt.floor("D")).dt.days
    selected = manifest.loc[manifest.days_out.isin(days_out)].sort_values(
        ["timestamp", "commit", "original_path"]).drop_duplicates(["event_date", "days_out"], keep="last")
    outcomes = {}
    for path in (settings.stats_history_file, settings.non_merged_stats_file):
        for row in pd.read_csv(path).itertuples():
            red, blue, win = _name(row.fighter_red), _name(row.fighter_blue), _name(row.winner)
            date = str(row.event_date)[:10]
            if win not in (red, blue):
                continue
            key = (date, *sorted((red, blue)))
            if key in outcomes and outcomes[key] != win:
                raise ValueError(f"Conflicting outcomes: {key}")
            outcomes[key] = win
    outcomes.update(split_outcomes)
    required = [c for c in scaler.feature_names_in_ if c != "proba_fair_open_diff"] + ["elo_pred"]
    prices = [f"dec_{s}_{side}" for s in ("open", "close1", "close2") for side in ("red", "blue")]
    frames, audit = {d: [] for d in days_out}, []
    for row in selected.itertuples():
        frame = pd.read_csv(source / row.filename).copy()
        keys, winners = [], []
        for fight in frame.itertuples():
            red, blue = _name(fight.fighter_red), _name(fight.fighter_blue)
            key = (row.event_date, *sorted((red, blue)))
            keys.append("|".join(key))
            win = outcomes.get(key)
            winners.append(np.nan if win is None else int(win == red))
        frame["fight_key"], frame["winner"] = keys, winners
        frame["date"] = row.event_date
        numeric = frame.reindex(columns=list(dict.fromkeys(required + prices))).apply(pd.to_numeric, errors="coerce")
        features_ok = np.isfinite(numeric).all(axis=1) & numeric[prices].gt(1).all(axis=1)
        in_split = frame.fight_key.isin(allowed_keys) if allowed_keys is not None else pd.Series(True, index=frame.index)
        valid = features_ok & frame.winner.notna() & in_split
        audit.append(dict(event_date=row.event_date, days_out=row.days_out, filename=row.filename,
                          original_fights=len(frame), missing_inputs=int((~features_ok).sum()),
                          outside_input_split=int((~in_split).sum()),
                          unsettled_or_unmatched=int(frame.winner.isna().sum()), eligible_fights=int(valid.sum())))
        frame = frame.loc[valid].copy()
        frame["winner"] = frame.winner.astype(int)
        if frame.fight_key.duplicated().any():
            raise ValueError(f"Duplicate fight in {row.filename}")
        frames[row.days_out].append(frame)
    cohorts = {d: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["fight_key"])
               for d, parts in frames.items()}
    common = set.intersection(*(set(f.fight_key) for f in cohorts.values()))
    for d, frame in cohorts.items():
        cohorts[d] = frame.loc[frame.fight_key.isin(common)].copy()
        if len(cohorts[d]):
            cohorts[d] = cohorts[d].sort_values(["date", "fight_key"]).reset_index(drop=True)
    audit = pd.DataFrame(audit, columns=['event_date','days_out','filename','original_fights',
                                        'missing_inputs','outside_input_split','unsettled_or_unmatched','eligible_fights'])
    retained = cohorts[days_out[0]].groupby("date").size() if common else pd.Series(dtype=int)
    audit["common_fights_retained"] = audit.event_date.map(retained).fillna(0).astype(int)
    available = selected.groupby('event_date').days_out.agg(set).to_dict()
    missing = pd.DataFrame([dict(event_date=date,
                                missing_days=','.join(map(str, sorted(set(days_out)-available.get(date,set())))))
                           for date in requested_dates if set(days_out)-available.get(date,set())],
                          columns=['event_date','missing_days'])
    return cohorts, selected, audit, missing


def run_lead_time_backtests(data_dir, decay_rates, days_out=(7, 5, 3),
                            n_paths=20, seed=42, initial_bankroll=500,
                            params=None, output_dir=None, event_data=None):
    """Realized joint decay paths; uniform decimal interpolation per fight.

    One U(0,1) draw moves both sides of a fight between close1 and close2.
    Shared draws compare the two fixed closing policies at identical prices.
    No search/refit, EV-based cohort filter, rescaling, floor, or top-ups.
    """
    import joblib
    import statsmodels.api as sm
    from ufc_betting.config import config
    from .decay_sensitivity import validate_rates
    from .open_model_close1_input import opening_model_at_close1
    from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
    from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_backtest
    days_out = normalize_days_out(days_out)
    if n_paths < 1:
        raise ValueError("Positive path count and unique positive lead times required")
    rates = validate_rates(decay_rates)
    params = load_lead_time_params(data_dir) if params is None else params
    out = Path(output_dir) if output_dir else Path(data_dir) / "model_results" / "lead_time_backtests"
    out.mkdir(parents=True, exist_ok=True)
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    cohorts, selected, audit, missing = prepare_lead_time_data(data_dir, scaler, days_out, event_data=event_data)
    selected.to_csv(out / "selected_snapshots.csv", index=False)
    audit.to_csv(out / "cohort_audit.csv", index=False)
    missing.to_csv(out / "missing_exact_days.csv", index=False)
    if cohorts[days_out[0]].empty:
        message = 'No eligible common event cohort: exact-day archive snapshots are missing or fights lack usable inputs.'
        print(message)
        (out / 'coverage_status.json').write_text(json.dumps(dict(status='no_coverage',message=message,
            days_out=days_out,missing_event_count=len(missing)),indent=2),encoding='utf-8')
        return dict(status='no_coverage',message=message,summary=pd.DataFrame(),audit=audit,
                    missing_days=missing,dates=[],output_dir=out,simulation_results={},backtest_plots={})
    dates = sorted(cohorts[days_out[0]].date.unique())
    rng = np.random.default_rng(seed)
    deterministic, sampled, event_rows, summaries = {}, {}, [], []
    simulation_results, backtest_plots = {}, {}

    def evaluate(predicted, reference, stage):
        # The common adapter stores execution/fair prices in close1 columns;
        # stage chooses the frozen closing policy, not a second price rewrite.
        with contextlib.redirect_stdout(io.StringIO()):
            p = params[stage]
            kelly, parlay = simulate_kelly(
                predicted, prob_cols=['proba_blue','proba_red'],
                fair_decimal_cols=['dec_fair_close1_blue','dec_fair_close1_red'],
                real_decimal_cols=['dec_close1_blue','dec_close1_red'],
                pred_winner_col='pred_winner', prediction_se_col='proba_se',
                init_bankroll=initial_bankroll, bankroll_floor=None,
                max_drawdown=p['mdd'], N=p['N'], parlay_mdd=p['mdd_parlay'],
                N_parlay=p['N_parlay'], z=p['z'], calc_parlay=True,
                moneyline_american_bounds=(p['moneyline_min_american'],p.get('moneyline_max_american',np.inf)),
                adjust_mdd_by_edge=False, decay_rates=rates,
                opening_data=reference, opening_params=params['open'])
        grouped = kelly.groupby('date', sort=True).first().reset_index()
        events = grouped[['date', 'bankroll_postevent']].copy()
        events['event_return'] = grouped.bankroll_pct_change / 100
        assert list(pd.to_datetime(events.date).dt.strftime("%Y-%m-%d")) == dates
        return events, np.r_[initial_bankroll, grouped.bankroll_postevent.to_numpy()], kelly, parlay

    for day, frame in cohorts.items():
        print(f"{day} days: {len(frame)} fights / {len(dates)} events", flush=True)
        lo = frame[["dec_close1_red", "dec_close1_blue"]].to_numpy(float)
        hi = frame[["dec_close2_red", "dec_close2_blue"]].to_numpy(float)
        frame.to_csv(out / f"cohort_{day}_days.csv", index=False)
        for stage, prices in (("close1", lo), ("close2", hi)):
            predicted = opening_model_at_close1(frame, prices, model, scaler)
            events, path, kelly, parlay = evaluate(predicted, frame, stage)
            label = f'recorded_{day}_days_{stage}'
            simulation_results[label] = (kelly, parlay)
            kelly.to_csv(out / f'{label}_kelly.csv', index=False)
            parlay.to_csv(out / f'{label}_parlay.csv', index=False)
            backtest_path = out / f'{label}_backtest.png'
            with contextlib.redirect_stdout(io.StringIO()):
                plot_backtest(kelly, init_bankroll=initial_bankroll, path=str(backtest_path))
            plt.close('all')
            backtest_plots[label] = backtest_path
            deterministic[(day, stage)] = path
            events["days_out"], events["policy"], events["scenario"] = day, stage, "recorded"
            event_rows.append(events)
            summaries.append(dict(days_out=day, policy=stage, scenario="recorded", path=0, ending_bankroll=path[-1]))
        draws = rng.uniform(size=(n_paths, len(frame), 1))
        np.save(out / f"uniform_draws_{day}_days.npy", draws)
        sampled[day] = {stage: [] for stage in ("close1", "close2")}
        for i, draw in enumerate(draws):
            prices = lo + draw * (hi - lo)
            assert ((prices >= np.minimum(lo, hi)) & (prices <= np.maximum(lo, hi))).all()
            predicted = opening_model_at_close1(frame, prices, model, scaler)
            for stage in ("close1", "close2"):
                events, path, kelly, parlay = evaluate(predicted, frame, stage)
                label = f'uniform_{day}_days_{stage}_{i}'
                simulation_results[label] = (kelly, parlay)
                sampled[day][stage].append(path)
                events["days_out"], events["policy"], events["scenario"], events["path"] = day, stage, "uniform", i
                event_rows.append(events)
                summaries.append(dict(days_out=day, policy=stage, scenario="uniform", path=i, ending_bankroll=path[-1]))
            if (i+1) % 5 == 0:
                print(f"  {i+1}/{n_paths} sampled paths complete", flush=True)
    colors = {"close1": "#3579ad", "close2": "#df8332"}
    x = np.arange(len(dates)+1)
    fig, axes = plt.subplots(1, len(days_out), figsize=(16, 5), squeeze=False, layout="constrained")
    for ax, day in zip(axes[0], days_out):
        for stage, color in colors.items():
            ax.plot(x, deterministic[(day, stage)], color=color, label=stage)
        ax.set(title=f"{day} days before event", xlabel="Completed events", ylabel="Bankroll ($)")
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f"Recorded prices | joint moneyline + parlay decay | {len(dates)} common events")
    endpoint_plot = out / "recorded_lead_time_paths.png"
    fig.savefig(endpoint_plot, dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, len(days_out), figsize=(16, 5), squeeze=False, layout="constrained")
    for ax, day in zip(axes[0], days_out):
        for stage, color in colors.items():
            paths = np.asarray(sampled[day][stage])
            for path in paths:
                ax.plot(x, path, color=color, alpha=.12, linewidth=.8)
            ax.plot(x, np.median(paths, axis=0), color=color, linewidth=2.5, label=f"{stage} controls: median")
            np.save(out / f"uniform_paths_{day}_days_{stage}.npy", paths)
        ax.set(title=f"{day} days before event", xlabel="Completed events", ylabel="Bankroll ($)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(f"Uniform prices between close1 and close2 | {n_paths} scenarios per policy | seed {seed}")
    sampled_plot = out / "uniform_lead_time_paths.png"
    fig.savefig(sampled_plot, dpi=140); plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary["total_return_pct"] = 100 * (summary.ending_bankroll / initial_bankroll - 1)
    summary.to_csv(out / "path_summary.csv", index=False)
    joblib.dump(simulation_results, out / 'simulate_kelly_results.joblib', compress=3)
    pd.concat(event_rows, ignore_index=True).to_csv(out / "event_returns.csv", index=False)
    pd.DataFrame({"completed_event": np.arange(1,len(dates)+1), "date": dates}).to_csv(out / "event_sequence.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(dict(params=params, decay_rates=rates, days_out=days_out,
         n_paths=n_paths, seed=seed, initial_bankroll=initial_bankroll), indent=2), encoding="utf-8")
    return dict(status='complete', summary=summary, recorded_plot=endpoint_plot, sampled_plot=sampled_plot,
                audit=audit, missing_days=missing, dates=dates, output_dir=out,
                simulation_results=simulation_results, backtest_plots=backtest_plots)
