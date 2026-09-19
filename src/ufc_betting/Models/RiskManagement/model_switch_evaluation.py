"""Chronological switching-rule evaluation at identical hypothetical prices.

Models stay fixed. Policy inputs use current-vs-opening fair probability distance,
not the fraction of a future closing-price path. Each price scenario is a separate
backtest; repeated scenarios never multiply the number of independent fights.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import (
    KEY, predict, fair_probabilities, expected_bets, interpolate_prices,
)


def losses(p, y):
    p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), (p - y) ** 2


def choose_open(distance, rule):
    if rule == "always_open":
        return np.ones(len(distance), dtype=bool)
    if rule == "always_close1":
        return np.zeros(len(distance), dtype=bool)
    direction, threshold = rule.split(":")
    near = np.asarray(distance) <= float(threshold)
    return near if direction == "open_near" else ~near


def settle(rows, rule):
    """Apply the selector, then cap aggregate event stakes proportionally at 1."""
    selected = choose_open(rows.distance_pp, rule)
    p = np.where(selected, rows.p_open, rows.p_close1)
    f = np.where(selected, rows.f_open, rows.f_close1)
    profit = np.where(selected, rows.unit_profit_open, rows.unit_profit_close1)
    result = rows[["date", "scenario_pct", "fight_id"]].copy()
    result["selected_open"] = selected
    result["log_loss"], result["brier"] = losses(p, rows.winner)
    result["correct"] = (p >= .5) == rows.winner
    result["fstar"] = f
    total = result.groupby(["date", "scenario_pct"]).fstar.transform("sum")
    result["fstar"] /= np.maximum(total, 1)
    result["realized_return"] = result.fstar * profit
    return result


def audit(model, scaler, stage, data_dir):
    """Match saved training features back to dated source CSV rows."""
    numeric = list(scaler.feature_names_in_)
    source = pd.concat([pd.read_csv(data_dir / f"{s}_logit_{stage}.csv") for s in ("train", "test")], ignore_index=True)
    design = pd.DataFrame(scaler.transform(source[numeric]), columns=numeric)
    for name in model.model.exog_names:
        if name not in design:
            design[name] = 1.0 if name == "const" else source[name].to_numpy()
    names = model.model.exog_names
    hashes = pd.util.hash_pandas_object(pd.DataFrame(np.round(design[names].to_numpy(), 7)), index=False)
    saved = pd.util.hash_pandas_object(pd.DataFrame(np.round(np.asarray(model.model.exog), 7)), index=False)
    date_map = pd.DataFrame({"hash": hashes, "date": pd.to_datetime(source.date)}).groupby("hash").date.max()
    matched = saved.map(date_map)
    if matched.isna().any():
        raise ValueError(f"{stage}: cannot date {matched.isna().sum()} saved training rows; evaluation stopped")
    return {"training_rows": len(saved), "last_training_date": str(matched.max().date())}


def run():
    data_dir = settings.data_dir / "model_results"
    out = data_dir / "model_switch_evaluation"
    frames = {s: pd.read_csv(data_dir / f"test_logit_{s}.csv").sort_values(KEY).reset_index(drop=True) for s in ("open", "close1")}
    for frame in frames.values():
        frame.date = pd.to_datetime(frame.date)
        if frame.duplicated(KEY).any() or not frame.winner.isin([0, 1]).all():
            raise ValueError("Duplicate fights or unsettled outcomes")
    if not frames["open"][KEY + ["winner"]].equals(frames["close1"][KEY + ["winner"]]):
        raise ValueError("Fight cohorts/outcomes differ")
    models = {s: sm.load(getattr(config, f"model_{s}_path")) for s in frames}
    scalers = {s: joblib.load(getattr(config, f"scaler_{s}_path")) for s in frames}
    audits = {s: audit(models[s], scalers[s], s, data_dir) for s in frames}
    cutoff = max(pd.Timestamp(a["last_training_date"]) for a in audits.values())
    mask = frames["open"].date > cutoff
    excluded = int((~mask).sum())
    frames = {s: frame.loc[mask].reset_index(drop=True) for s, frame in frames.items()}
    opening = frames["open"]
    op = opening[["dec_open_red", "dec_open_blue"]].to_numpy()
    cl = frames["close1"][["dec_close1_red", "dec_close1_blue"]].to_numpy()
    opening_fair = fair_probabilities(op)
    observations = []
    for fraction in np.linspace(0, 1, 11):
        prices = interpolate_prices(op, cl, fraction)
        fair = fair_probabilities(prices)
        rows = opening[KEY + ["winner"]].copy()
        rows["fight_id"] = np.arange(len(rows))
        rows["scenario_pct"] = round(100 * fraction)
        rows["distance_pp"] = 100 * np.abs(fair[:, 0] - opening_fair[:, 0])
        for stage in frames:
            features = frames[stage].copy()
            features["proba_fair_open_diff"] = opening_fair[:, 0] - opening_fair[:, 1]
            features[f"proba_fair_{stage}_diff"] = fair[:, 0] - fair[:, 1]
            probability = predict(features, models[stage], scalers[stage])
            bets = expected_bets(probability, prices, fair, .4, 250)
            rows[f"p_{stage}"] = probability
            rows[f"f_{stage}"] = bets.fstar
            rows[f"unit_profit_{stage}"] = np.where(bets.pred_winner.eq(rows.winner), bets.choice_decimal - 1, -1)
        observations.append(rows)
    data = pd.concat(observations, ignore_index=True)
    rules = ["always_open", "always_close1"] + [f"{direction}:{threshold}" for direction in ("open_near", "close1_near") for threshold in [0, .5, 1, 2, 3, 5, 8, 12, 20]]
    settled = {rule: settle(data, rule) for rule in rules}
    events = {rule: value.groupby(["date", "scenario_pct"]).agg(
        realized_return=("realized_return", "sum"), stake=("fstar", "sum"),
        log_loss=("log_loss", "mean"), brier=("brier", "mean")) for rule, value in settled.items()}
    dates = np.sort(data.date.unique())
    if len(dates) <= 24:
        raise ValueError("Need more than 24 events for chronological evaluation")
    blocks, chosen, heldout = [], [], []
    for start in range(24, len(dates), 10):
        train_dates, test_dates = dates[:start], dates[start:start + 10]
        scores = {}
        for rule, event in events.items():
            train = event.loc[event.index.get_level_values("date").isin(train_dates)]
            scores[rule] = {"prediction": train.log_loss.mean(),
                            "return": -np.log1p(train.realized_return.clip(lower=-.999999999)).mean()}
        selections = {criterion: min(rules, key=lambda rule: scores[rule][criterion]) for criterion in ("prediction", "return")}
        for criterion, rule in selections.items():
            blocks.append({"criterion": criterion, "rule": rule, "training_events": len(train_dates),
                           "train_end": pd.Timestamp(train_dates[-1]), "test_start": pd.Timestamp(test_dates[0]),
                           "test_end": pd.Timestamp(test_dates[-1]), "test_events": len(test_dates)})
            selected = settled[rule].loc[settled[rule].date.isin(test_dates)].copy()
            selected["strategy"] = f"switch_{criterion}"
            selected["rule"] = rule
            heldout.append(selected)
        for rule in ("always_open", "always_close1"):
            baseline = settled[rule].loc[settled[rule].date.isin(test_dates)].copy()
            baseline["strategy"] = rule
            baseline["rule"] = rule
            heldout.append(baseline)
    heldout = pd.concat(heldout, ignore_index=True)
    event_results = heldout.groupby(["strategy", "scenario_pct", "date"]).agg(
        realized_return=("realized_return", "sum"), stake=("fstar", "sum"))
    metrics = []
    for (strategy, scenario), group in heldout.groupby(["strategy", "scenario_pct"]):
        event = event_results.loc[(strategy, scenario)].sort_index()
        bankroll = np.r_[1., (1 + event.realized_return).cumprod().to_numpy()]
        metrics.append({"strategy": strategy, "scenario_pct": scenario, "events": len(event), "fights": len(group),
            "log_loss": group.log_loss.mean(), "brier": group.brier.mean(), "accuracy": group.correct.mean(),
            "roi_on_stakes_pct": 100 * group.realized_return.sum() / group.fstar.sum() if group.fstar.sum() else np.nan,
            "compounded_return_pct": 100 * (bankroll[-1] - 1),
            "max_drawdown_pct": 100 * np.max(1 - bankroll / np.maximum.accumulate(bankroll)),
            "bets": int(group.fstar.gt(0).sum()), "open_selection_pct": 100 * group.selected_open.mean()})
    # Bin diagnostics use only held-out events; scenarios remain correlated.
    bins = pd.cut(data.distance_pp, [-1e-9, .5, 1, 2, 3, 5, 8, 12, np.inf], right=True)
    data["distance_bin"] = bins.astype(str)
    diagnostics = []
    for strategy in ("always_open", "always_close1"):
        part = heldout.loc[heldout.strategy.eq(strategy)].merge(data[["fight_id", "scenario_pct", "distance_bin"]], on=["fight_id", "scenario_pct"], validate="one_to_one")
        for label, group in part.groupby("distance_bin"):
            diagnostics.append({"strategy": strategy, "distance_bin_pp": label,
                "unique_fights": group.fight_id.nunique(), "scenario_rows": len(group),
                "log_loss": group.log_loss.mean(), "brier": group.brier.mean(), "accuracy": group.correct.mean(),
                "roi_on_stakes_pct": 100 * group.realized_return.sum() / group.fstar.sum() if group.fstar.sum() else np.nan})
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "matched_price_predictions.csv.gz", index=False)
    heldout.to_csv(out / "heldout_predictions_and_bets.csv.gz", index=False)
    event_results.to_csv(out / "heldout_event_returns.csv")
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "heldout_metrics_by_scenario.csv", index=False)
    block_performance = []
    for block in blocks:
        if block["criterion"] != "prediction":
            continue
        subset = heldout.loc[heldout.date.between(block["test_start"], block["test_end"])]
        for (strategy, scenario), group in subset.groupby(["strategy", "scenario_pct"]):
            ev = group.groupby("date").realized_return.sum()
            block_performance.append({"test_start": block["test_start"], "test_end": block["test_end"],
                "strategy": strategy, "scenario_pct": scenario, "events": len(ev),
                "log_loss": group.log_loss.mean(), "brier": group.brier.mean(),
                "mean_event_return_pct": 100 * ev.mean(),
                "compounded_return_pct": 100 * ((1 + ev).prod() - 1)})
    pd.DataFrame(block_performance).to_csv(out / "heldout_block_performance.csv", index=False)
    pd.DataFrame(blocks).to_csv(out / "selected_rules_by_block.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "heldout_distance_bins.csv", index=False)
    # Paired event bootstrap of policy gain, averaged across scenarios per event.
    rng = np.random.default_rng(42)
    bootstrap = []
    for switch in ("switch_prediction", "switch_return"):
        for baseline in ("always_open", "always_close1"):
            a = event_results.loc[switch].realized_return.unstack("scenario_pct")
            b = event_results.loc[baseline].realized_return.unstack("scenario_pct")
            delta = (a - b).mean(axis=1).to_numpy()
            draws = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(axis=1)
            bootstrap.append({"strategy": switch, "baseline": baseline,
                "mean_event_gain_pp": 100 * delta.mean(), "bootstrap_low_pp": 100 * np.quantile(draws, .025),
                "bootstrap_high_pp": 100 * np.quantile(draws, .975)})
    pd.DataFrame(bootstrap).to_csv(out / "paired_event_bootstrap.csv", index=False)
    metadata = {"training_audit": audits, "excluded_test_rows_at_or_before_training_end": excluded,
        "initial_selection_events": 24, "holdout_events": len(dates) - 24, "update_every_events": 10,
        "shared_mdd": .4, "shared_N": 250, "event_exposure_cap": 1.,
        "distance": "absolute red fair-probability change from actual opening, in percentage points; symmetric for blue",
        "selection": "expanding prior events only; log-loss selector and log-growth selector; tie order favors always-open then always-close1",
        "scenarios": "11 hypothetical shared offered prices: 0% opening through 100% close1. Evaluate each separately, equal scenario weight for training rules.",
        "limitations": ["Saved model weights fixed. Training-feature audit does not establish provenance of earlier feature engineering or hyperparameter decisions.",
            "Test outcomes have been inspected in prior analyses; this is retrospective chronological validation, not an untouched prospective holdout.",
            "Interpolated odds are hypothetical and rely on historical closing endpoints; live snapshots required before using returns as attainable performance.",
            "Distance-bin summaries pool correlated scenarios; unique fights reported separately.",
            "Paired event bootstrap preserves within-event scenarios but assumes independent events; intervals are exploratory."]}
    (out / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render(metrics, out)
    print(json.dumps(metadata, indent=2))
    print(metrics.loc[metrics.scenario_pct.isin([0, 50, 100])].to_string(index=False))
    print(pd.DataFrame(bootstrap).to_string(index=False))


def render(metrics, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for strategy, group in metrics.groupby("strategy"):
        group = group.sort_values("scenario_pct")
        for ax, metric in zip(axes, ["log_loss", "roi_on_stakes_pct", "compounded_return_pct"]):
            ax.plot(group.scenario_pct, group[metric], marker="o", markersize=3, label=strategy)
    for ax, title in zip(axes, ["Log loss (lower is better)", "Realized ROI on stakes (%)", "Realized compounded return (%)"]):
        ax.set(title=title, xlabel="Price movement: opening (0%) to close1 (100%)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Later-event evaluation | fixed models, equal risk settings, event exposure capped at 100%")
    fig.supxlabel("Each point is a separate hypothetical-price backtest. Rules selected only on earlier events; scenarios are not independent samples.", fontsize=10)
    fig.savefig(out / "heldout_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
