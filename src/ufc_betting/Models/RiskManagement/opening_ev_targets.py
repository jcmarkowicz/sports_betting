"""Required per-event uniform price movement to meet opening-model EV targets."""
import json
import numpy as np
import pandas as pd
import joblib
import statsmodels.api as sm
from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.open_model_close1_input import opening_model_at_close1
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import interpolate_prices
from ufc_betting.BettingStrategy.Backtest.backtest_functions import run_per_bet_scaling
from ufc_betting.BettingStrategy.kelly_scaling import kelly_edge


def expected_events(frame, z=.5):
    red = frame.pred_winner.eq(1)
    p = np.where(red, frame.proba_red, frame.proba_blue)
    fair = np.where(red, frame.dec_fair_close1_red, frame.dec_fair_close1_blue)
    real = np.where(red, frame.dec_close1_red, frame.dec_close1_blue)
    inputs = pd.DataFrame({"f_star_unscaled": [kelly_edge(a, b) for a, b in zip(p, fair)],
        "choice_proba": p, "choice_fair_odds": fair, "choice_real_odds": real,
        "choice_ev": p * real - 1, "choice_idx": frame.pred_winner, "winner": frame.winner})
    # Same per-bet scaling defaults as simulate_kelly, including edge-adjusted MDD.
    bets, _ = run_per_bet_scaling(inputs, max_drawdown=.4, bankroll=500, N=250)
    bets["date"] = pd.to_datetime(frame.date)
    bets["raw_weighted_edge"] = bets.f_star_scaled * np.maximum(p - 1 / fair, 0)
    bets["conservative_weighted_edge"] = bets.f_star_scaled * np.maximum(p - z * frame.proba_se.to_numpy() - 1 / fair, 0)
    rows = []
    for date, group in bets.groupby("date"):
        raw = group.raw_weighted_edge.sum()
        uncertainty = float(np.clip(group.conservative_weighted_edge.sum() / raw, 0, 1)) if raw > 0 else 0.
        total = group.f_star_scaled.sum() * uncertainty
        multiplier = uncertainty * (min(1., 1 / total) if total > 0 else 1.)
        rows.append({"date": date, "expected_return_pct": 100 * (group.f_star_scaled * multiplier * group.ev).sum(),
            "stake_fraction": group.f_star_scaled.sum() * multiplier,
            "uncertainty_multiplier": uncertainty, "n_fights": len(group)})
    return pd.DataFrame(rows)


def required_targets(curves, targets=(50, 60, 70, 80, 90, 100)):
    rows = []
    for date, group in curves.groupby("date"):
        group = group.sort_values("movement_pct")
        opening = float(group.iloc[-1].expected_return_pct)
        for target in targets:
            goal = opening * target / 100
            eligible = group.loc[group.expected_return_pct >= goal - 1e-10]
            if opening <= 1e-10:
                hit, status = None, "undefined_zero_opening_ev"
            elif eligible.empty:
                hit, status = None, "unreachable_on_grid"
            else:
                hit = eligible.iloc[0]
                status = "already_met" if hit.movement_pct == 0 else "reached_on_grid"
            rows.append({"date": date, "target_pct_opening_ev": target, "opening_expected_return_pct": opening,
                "starting_pct_opening_ev": 100 * group.iloc[0].expected_return_pct / opening if opening > 1e-10 else np.nan,
                "required_movement_pct": hit.movement_pct if hit is not None else np.nan,
                "achieved_pct_opening_ev": 100 * hit.expected_return_pct / opening if hit is not None else np.nan,
                "mean_absolute_decimal_change": hit.mean_absolute_decimal_change if hit is not None else np.nan,
                "status": status})
    return pd.DataFrame(rows)


def run():
    root = settings.data_dir / "model_results"
    out = root / "opening_model_ev_targets"
    source = pd.read_csv(root / "test_logit_close1.csv").sort_values(["date", "fighter_red", "fighter_blue"]).reset_index(drop=True)
    opening = pd.read_csv(root / "test_logit_open.csv").sort_values(["date", "fighter_red", "fighter_blue"]).reset_index(drop=True)
    keys = ["date", "fighter_red", "fighter_blue"]
    if source.duplicated(keys).any() or not source[keys].equals(opening[keys]):
        raise ValueError("Fight cohorts must match exactly")
    close_prices = source[["dec_close1_red", "dec_close1_blue"]].to_numpy()
    open_prices = opening[["dec_open_red", "dec_open_blue"]].to_numpy()
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    parts = []
    for percent in range(101):
        prices = interpolate_prices(close_prices, open_prices, percent / 100)
        prepared = opening_model_at_close1(source, prices, model, scaler)
        events = expected_events(prepared)
        moves = pd.DataFrame({"date": pd.to_datetime(source.date), "change": np.mean(np.abs(prices-close_prices), axis=1)})
        events["mean_absolute_decimal_change"] = events.date.map(moves.groupby("date").change.mean())
        events["movement_pct"] = percent
        parts.append(events)
    curves = pd.concat(parts, ignore_index=True)
    benchmark = curves.loc[curves.movement_pct.eq(100)].set_index("date").expected_return_pct
    curves["pct_opening_ev"] = 100 * curves.expected_return_pct / curves.date.map(benchmark).where(curves.date.map(benchmark) > 1e-10)
    results = required_targets(curves)
    out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out / "event_curves.csv", index=False)
    results.to_csv(out / "required_movement_by_event.csv", index=False)
    results.pivot(index="date", columns="target_pct_opening_ev", values="required_movement_pct").to_csv(out / "movement_target_matrix.csv")
    (out / "methodology.json").write_text(json.dumps({"model":str(config.model_open_path), "mdd":.4,"N":250,"z":.5,
        "event_exposure_cap":1., "parlays":False,"events":len(benchmark),"fights":len(source),
        "movement":"Uniform fraction of each side's decimal gap from close1 to open; 0%=close1, 100%=open",
        "target":"Opening model EV at opening prices using identical sizing and uncertainty treatment",
        "stakes":"simulate_kelly per-bet defaults including edge-adjusted MDD, then event uncertainty multiplier and exposure cap",
        "search":"First 1%-grid movement meeting or exceeding each target. Not necessarily exact equality or a sustained crossing.",
        "zero_targets":"Ratios undefined when opening EV is zero; omitted from ratio chart, retained in CSV",
        "interpretation":"Model-implied expected return, not realized returns or predicted typical wealth"},indent=2),encoding="utf-8")
    render(results, out)
    print(results.tail(12).to_string(index=False))
    return results


def render(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    dates = sorted(results.date.unique())
    for target, group in results.groupby("target_pct_opening_ev"):
        values = group.set_index("date").reindex(dates).required_movement_pct
        ax.plot(np.arange(1, len(dates)+1), values, marker=".", lw=1.3, label=f"{target}% opening EV", alpha=.85)
    ax.set(xlabel="Test event in chronological order", ylabel="Required movement toward opening prices (%)",
           title="Opening model: movement each event needs to reach opening-EV targets", ylim=(-3,103))
    ax.grid(alpha=.2)
    ax.legend(ncol=3)
    fig.supxlabel("0% = target already met at close1 prices. First crossing on a 1% grid; lines need not be monotone.\n"
                  "Fixed opening model; MDD 0.40, N 250, z 0.5; no parlays; 100% event exposure cap. Missing values: zero opening EV.",fontsize=10)
    fig.savefig(out / "required_movement_curves.png",dpi=150)
    plt.close(fig)
    from matplotlib.backends.backend_pdf import PdfPages
    groups = list(results.groupby("date", sort=True))
    def page(items):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (date, group) in zip(axes.flat, items):
            ax.plot(group.target_pct_opening_ev, group.required_movement_pct, marker="o", color="#2366af")
            ax.set(title=str(pd.Timestamp(date).date()), xlabel="Target (% of opening EV)",
                   ylabel="Required price movement (%)", ylim=(-3,103), xticks=[50,60,70,80,90,100])
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(items):]:
            ax.set_visible(False)
        fig.suptitle("How far each event's close1 prices must move to reach opening-model EV targets")
        fig.supxlabel("0% movement = already meets target at close1 prices; 100% = actual opening prices. First crossing on a 1% grid.\n"
                      "Opening model throughout. MDD 0.40, N 250, z 0.5. Expected returns, not realized performance.",fontsize=10)
        return fig
    with PdfPages(out / "all_event_required_movements.pdf") as pdf:
        for start in range(0,len(groups),6):
            fig=page(groups[start:start+6]); pdf.savefig(fig); plt.close(fig)
    fig=page(groups[-6:]); fig.savefig(out / "latest_event_required_movements.png",dpi=150); plt.close(fig)


if __name__ == "__main__":
    run()
