"""Fixed-model, dynamic-Kelly expected event returns along close1 -> open prices.

Run python -m ufc_betting.Models.RiskManagement.event_odds_sensitivity.
Only straight moneylines are included. Event dates identify available CSV cards.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.BettingStrategy.kelly_scaling import kelly_edge, scale_kelly_for_mdd
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.odds_features import devig_two_way

KEY = ["date", "fighter_red", "fighter_blue"]


def interpolate_prices(closing, opening, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    closing, opening = np.asarray(closing, float), np.asarray(opening, float)
    if closing.shape != opening.shape or not np.isfinite([closing, opening]).all():
        raise ValueError("Prices must be finite and aligned")
    if (closing <= 1).any() or (opening <= 1).any():
        raise ValueError("Decimal odds must exceed 1")
    return closing + fraction * (opening - closing)


def fair_probabilities(prices):
    return np.asarray([devig_two_way(red, blue, method="power")[:2]
                       for red, blue in prices])


def predict(frame, model, scaler):
    numeric = list(scaler.feature_names_in_)
    design = pd.DataFrame(scaler.transform(frame[numeric]), columns=numeric, index=frame.index)
    names = list(model.model.exog_names)
    for name in names:
        if name == "const":
            design[name] = 1.0
        elif name not in design:
            design[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(design[names].to_numpy()).all():
        raise ValueError("Nonfinite model features")
    probability = np.asarray(model.predict(design[names]))
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid model probabilities")
    return probability


def expected_bets(probability_red, prices, fair, mdd, horizon):
    """Match live straight-bet selection and fair-odds Kelly/MDD scaling."""
    red = np.asarray(probability_red) >= .5
    side = np.where(red, 0, 1)  # price arrays are [red, blue]
    index = np.arange(len(red))
    p = np.where(red, probability_red, 1 - np.asarray(probability_red))
    decimal = prices[index, side]
    fair_decimal = 1 / fair[index, side]
    ev = p * decimal - 1
    full = np.asarray([kelly_edge(a, b) for a, b in zip(p, fair_decimal)])
    fstar = np.asarray([scale_kelly_for_mdd(a, b, f, horizon, mdd)
                        if f > 0 and e > 0 else 0.0
                        for a, b, f, e in zip(p, fair_decimal, full, ev)])
    return pd.DataFrame({"pred_winner": red.astype(int), "proba_red": probability_red,
                         "choice_probability": p, "choice_decimal": decimal,
                         "choice_fair_decimal": fair_decimal, "ev": ev,
                         "fstar_full": full, "fstar": fstar,
                         "expected_bankroll_return_pct": 100 * fstar * ev})


def evaluate(frame, prices, stage, model, scaler, mdd, horizon):
    fair = fair_probabilities(prices)
    features = frame.copy()
    features[f"proba_fair_{stage}_diff"] = fair[:, 0] - fair[:, 1]
    p = predict(features, model, scaler)
    bets = expected_bets(p, prices, fair, mdd, horizon)
    bets[KEY] = frame[KEY].reset_index(drop=True)
    bets["decimal_red"], bets["decimal_blue"] = prices[:, 0], prices[:, 1]
    bets["fair_probability_red"], bets["fair_probability_blue"] = fair[:, 0], fair[:, 1]
    bets["n_bets"] = bets.fstar.gt(0).astype(int)
    events = bets.groupby("date").agg(
        expected_bankroll_return_pct=("expected_bankroll_return_pct", "sum"),
        stake_fraction=("fstar", "sum"), n_bets=("n_bets", "sum"),
        n_fights=("n_bets", "size"))
    events["expected_roi_on_stakes_pct"] = events.expected_bankroll_return_pct / events.stake_fraction.replace(0, np.nan)
    return events, bets


def render(curves, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    groups = list(curves.groupby("date", sort=True))
    def draw(ax, date, group):
        ax.plot(group.shift_pct, group.expected_bankroll_return_pct, color="#2366af", lw=2)
        target = group.open_target_pct.iloc[0]
        ax.axhline(target, color="#c25916", ls="--", label=f"Open target: {target:.2f}%")
        ax.set_title(f"{date:%Y-%m-%d} | {int(group.n_fights.iloc[0])} fights")
        ax.set_xlabel("Movement from close1 to opening prices (%)")
        ax.set_ylabel("Expected bankroll return (%)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    image_dir = output_dir / "events"
    image_dir.mkdir(exist_ok=True)
    with PdfPages(output_dir / "all_event_curves.pdf") as pdf:
        for offset in range(0, len(groups), 6):
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
            fig.suptitle("Close1 expected event return as prices move toward opening")
            selected = groups[offset:offset + 6]
            for ax, (date, group) in zip(axes.flat, selected):
                draw(ax, date, group)
            for ax in list(axes.flat)[len(selected):]:
                ax.set_visible(False)
            pdf.savefig(fig)
            plt.close(fig)
    for date, group in groups:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        draw(ax, date, group)
        fig.savefig(image_dir / f"{date:%Y-%m-%d}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (date, group) in zip(axes.flat, groups[-6:]):
        draw(ax, date, group)
    for ax in list(axes.flat)[min(len(groups), 6):]:
        ax.set_visible(False)
    fig.suptitle("Latest six events | fixed model weights, updated probabilities and Kelly stakes")
    fig.savefig(output_dir / "latest_events.png", dpi=140)
    plt.close(fig)


def run(data_dir=None, output_dir=None, steps=100, mdd=0.4, horizon=250):
    if not 0 < mdd < 1 or horizon <= 1:
        raise ValueError("Shared mdd must be between 0 and 1 and horizon must exceed 1")
    if steps < 1:
        raise ValueError("steps must be positive")
    data_dir = Path(data_dir) if data_dir else settings.data_dir / "model_results"
    output_dir = Path(output_dir) if output_dir else data_dir / "event_odds_sensitivity"
    frames = {stage: pd.read_csv(data_dir / f"test_logit_{stage}.csv") for stage in ("open", "close1")}
    for frame in frames.values():
        frame.date = pd.to_datetime(frame.date, errors="raise")
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError("Missing or duplicate fight keys")
    close = frames["close1"].sort_values(KEY).reset_index(drop=True)
    opening = frames["open"].sort_values(KEY).reset_index(drop=True)
    if not close[KEY].equals(opening[KEY]):
        raise ValueError("Opening and close1 test fights must match")
    models = {s: sm.load(getattr(config, f"model_{s}_path")) for s in frames}
    scalers = {s: joblib.load(getattr(config, f"scaler_{s}_path")) for s in frames}
    open_prices = opening[["dec_open_red", "dec_open_blue"]].to_numpy()
    close_prices = close[["dec_close1_red", "dec_close1_blue"]].to_numpy()
    # Use a single recomputed opening probability feature throughout the path.
    fair_open = fair_probabilities(open_prices)
    close["proba_fair_open_diff"] = fair_open[:, 0] - fair_open[:, 1]
    target, open_bets = evaluate(opening, open_prices, "open", models["open"], scalers["open"], mdd, horizon)
    curves, fight_rows, diagnostics = [], [], {}
    diagnostics["open_csv_probability_max_difference"] = float(np.max(np.abs(open_bets.proba_red.to_numpy() - opening.proba_red)))
    for fraction in np.linspace(0, 1, steps + 1):
        prices = interpolate_prices(close_prices, open_prices, fraction)
        event, bets = evaluate(close, prices, "close1", models["close1"], scalers["close1"], mdd, horizon)
        event["shift_pct"] = fraction * 100
        event["open_target_pct"] = target.expected_bankroll_return_pct
        event["remaining_gap_pp"] = event.open_target_pct - event.expected_bankroll_return_pct
        bets["shift_pct"] = fraction * 100
        bets["mean_absolute_decimal_change"] = np.mean(np.abs(prices - close_prices), axis=1)
        event["mean_absolute_decimal_change"] = bets.groupby("date").mean_absolute_decimal_change.mean()
        if fraction == 0:
            diagnostics["close1_csv_probability_max_difference"] = float(np.max(np.abs(bets.proba_red.to_numpy() - close.proba_red)))
            starting_gap = event.remaining_gap_pp.copy()
        event["gap_closed_pct"] = 100 * (starting_gap - event.remaining_gap_pp) / starting_gap.where(starting_gap.abs() > 1e-10)
        curves.append(event.reset_index())
        fight_rows.append(bets)
    curves = pd.concat(curves, ignore_index=True).sort_values(["date", "shift_pct"])
    curves["roi_change_pp_per_1pct_shift"] = curves.groupby("date").expected_bankroll_return_pct.diff() / curves.groupby("date").shift_pct.diff()
    summaries = []
    for date, group in curves.groupby("date"):
        first, last = group.iloc[0], group.iloc[-1]
        reached = group.expected_bankroll_return_pct.ge(group.open_target_pct - 1e-10)
        summaries.append({"date": date, "close1_start_pct": first.expected_bankroll_return_pct,
                          "open_target_pct": first.open_target_pct, "close1_at_open_prices_pct": last.expected_bankroll_return_pct,
                          "first_grid_shift_meeting_target_pct": group.loc[reached, "shift_pct"].min(),
                          "closest_grid_shift_pct": group.loc[group.remaining_gap_pp.abs().idxmin(), "shift_pct"],
                          "monotone_increasing": bool((group.expected_bankroll_return_pct.diff().dropna() >= -1e-10).all()),
                          "max_stake_fraction": group.stake_fraction.max()})
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / "event_curves.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "event_summary.csv", index=False)
    pd.concat(fight_rows, ignore_index=True).to_csv(output_dir / "fight_shift_details.csv", index=False)
    open_bets.to_csv(output_dir / "opening_benchmark_bets.csv", index=False)
    metadata = {"steps": steps, "formula": "d(t) = d_close1 + t * (d_open - d_close1)",
                "expected_return": "100 * sum(fstar * (model_probability * offered_decimal - 1))",
                "mdd_open": mdd, "mdd_close1": mdd,
                "N_open": horizon, "N_close1": horizon,
                "models": {s: str(getattr(config, f"model_{s}_path")) for s in frames},
                "diagnostics": diagnostics, "n_events": len(summaries), "n_fights": len(close),
                "notes": ["Saved weights and preprocessing stay fixed. Picks may switch.",
                          "Power devig; fair-odds Kelly; positive-EV gate and existing MDD scaling.",
                          "Shared risk limits for open and close1; no parlays or event exposure cap added.",
                          "Expectations are evaluated using each model's own probabilities, not observed outcomes or validated future ROI.",
                          "Dates group available CSV fights; a card may be incomplete.",
                          "First crossing is on the sampled grid; curves need not be monotone."]}
    (output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render(curves, output_dir)
    print(json.dumps(metadata, indent=2))
    return curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mdd", type=float, default=0.4, help="Shared opening and close1 MDD")
    parser.add_argument("--horizon", type=int, default=250, help="Shared opening and close1 N")
    run(**vars(parser.parse_args()))
