"""Event accuracy, OLS regression and export helpers for vegas_accuracy.ipynb.

Plot functions return figures without displaying or saving them. Call save_png
and save_dataframe explicitly from the notebook. Imports do not run analysis.
"""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from ufc_betting.config import settings
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly


def load_accuracy_data(path, market):
    path = Path(path)
    frame = read_frozen_predictions(path)
    odds_columns = [f"dec_{market}_red", f"dec_{market}_blue"]
    fair_column = f"dec_fair_{market}_red"
    required = {"date", "fighter_red", "fighter_blue", "winner", "proba_red", fair_column, *odds_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    numeric = ["winner", "proba_red", fair_column, *odds_columns]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (frame["date"].notna() & frame["winner"].isin([0, 1])
             & np.isfinite(frame[numeric]).all(axis=1)
             & frame["proba_red"].between(0, 1)
             & frame[[fair_column, *odds_columns]].gt(1).all(axis=1)
             & frame[["fighter_red", "fighter_blue"]].notna().all(axis=1))
    data = frame.loc[valid].copy().sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"{path.name}: no eligible decided fights")
    if data.duplicated(["date", "fighter_red", "fighter_blue"]).any():
        raise ValueError(f"{path.name}: duplicate fights would bias accuracy")
    data["vegas_probability"] = 1 / data[fair_column]
    data["vegas_prediction"] = (data[odds_columns[0]] <= data[odds_columns[1]]).astype(int)
    data["model_probability"] = data["proba_red"]
    data["model_prediction"] = (data["model_probability"] >= .5).astype(int)
    return data, {"source": str(path), "rows_loaded": len(frame),
                  "fights_used": len(data), "excluded": len(frame) - len(data),
                  "events": data["date"].nunique(),
                  "first_date": data["date"].min(), "last_date": data["date"].max()}


def score_predictions(data, source):
    return {
        "accuracy": accuracy_score(data["winner"], data[f"{source}_prediction"]),
        "f1": f1_score(data["winner"], data[f"{source}_prediction"], zero_division=0),
        "brier": brier_score_loss(data["winner"], data[f"{source}_probability"]),
    }


def metrics_over_time(data):
    period = data["date"]
    rows = []
    for value, group in data.groupby(period, sort=True):
        for source in ("vegas", "model"):
            rows.append(dict(period=value, source=source, fights=len(group),
                             **score_predictions(group, source)))
    return pd.DataFrame(rows)


def plot_accuracy(data, split, market, *, model_type="open"):
    trend = metrics_over_time(data)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    period_label = "Event date (one point per event)"
    unit = "event"
    for row, metric in enumerate(("accuracy", "f1", "brier")):
        left, right = axes[row]
        for source, color in (("vegas", "tab:blue"), ("model", "tab:orange")):
            values = trend.loc[trend.source == source].set_index("period")[metric]
            score = score_predictions(data, source)[metric]
            left.scatter(values.index, values, color=color, alpha=.65, s=28,
                         label=f"{source.title()} (overall {score:.3f})")
        left.set(title=f"{metric.title()} by {unit} over time", xlabel=period_label,
                 ylabel=metric.title(), ylim=(-.03, 1.03))
        left.legend()
        import matplotlib.dates as mdates
        left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        left.tick_params(axis="x", rotation=45)

        # Pair Vegas and model scores by the same event, never by row position.
        paired = trend.pivot(index="period", columns="source", values=metric).dropna()
        x, y = paired["vegas"], paired["model"]
        varying = len(paired) >= 2 and x.nunique() > 1 and y.nunique() > 1
        right.scatter(x, y, alpha=.65, s=30, label=f"{len(paired)} {unit}s")
        if varying:
            r, p_value = pearsonr(x, y)
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.array([x.min(), x.max()])
            right.plot(line_x, intercept + slope * line_x, color="tab:red", label="Linear fit")
            correlation_label = f"Pearson r={r:.3f}, p={p_value:.3g}"
        else:
            correlation_label = "Pearson r and p undefined (insufficient variation)"
        right.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Equal scores")
        right.set(title=f"Vegas vs model {metric} | {correlation_label}",
                  xlabel=f"Vegas {unit} {metric}", ylabel=f"Model {unit} {metric}",
                  xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        right.legend()
        for ax in (left, right):
            ax.grid(alpha=.2)
    fig.suptitle(f"{split.upper()} | {market} odds | {model_type} model | {len(data)} fights")
    return fig, trend


def plot_accuracy_regression(data, split, market):
    paired = metrics_over_time(data).pivot(
        index="period", columns="source", values="accuracy"
    ).dropna().rename(columns={"vegas": "vegas_accuracy", "model": "model_accuracy"})
    if len(paired) < 3 or paired["vegas_accuracy"].nunique() < 2:
        raise ValueError(f"{split}/{market}: regression needs >=3 events and varying Vegas accuracy")
    X = sm.add_constant(paired[["vegas_accuracy"]], has_constant="add")
    fit = sm.OLS(paired["model_accuracy"], X).fit()
    ci = fit.conf_int(alpha=.05)
    coefficients = pd.DataFrame({
        "coefficient": fit.params, "std_error": fit.bse, "t_statistic": fit.tvalues,
        "p_value": fit.pvalues, "ci_95_low": ci[0], "ci_95_high": ci[1],
    }).rename_axis("term")
    coefficients.insert(coefficients.columns.get_loc("p_value") + 1, "significance",
                        np.select([fit.pvalues < .001, fit.pvalues < .01, fit.pvalues < .05],
                                  ["***", "**", "*"], default=""))
    grid = pd.DataFrame({"vegas_accuracy": np.linspace(
        paired["vegas_accuracy"].min(), paired["vegas_accuracy"].max(), 200)})
    predictions = fit.get_prediction(sm.add_constant(grid, has_constant="add")).summary_frame(alpha=.05)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.scatter(paired["vegas_accuracy"], paired["model_accuracy"], alpha=.65,
               label=f"{len(paired)} events")
    ax.plot(grid["vegas_accuracy"], predictions["mean"], color="tab:red", label="OLS fit with intercept")
    ax.fill_between(grid["vegas_accuracy"], predictions["mean_ci_lower"],
                    predictions["mean_ci_upper"], color="tab:red", alpha=.15,
                    label="95% CI for mean")
    ax.set(xlabel="Vegas event accuracy", ylabel="Model event accuracy",
           title=f"{split.upper()} | {market}: model accuracy ~ intercept + Vegas accuracy",
           ylim=(-.03, 1.03))
    ax.autoscale(enable=True, axis="x")
    ax.text(.03, .97, f"$R^2$ = {fit.rsquared:.3f}",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=.85, edgecolor="gray"))
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    return fit, coefficients, fig


def event_accuracy_returns(data, market, *, initial_bankroll=500, use_decay=True, params_dir=None):
    params_dir = Path(params_dir) if params_dir is not None else settings.data_dir / "kelly_params"
    params = BettingParams.load(params_dir / f"{market}.json")
    opening_data = None
    if use_decay and market != "open":
        if params.decay_rates is None or params.opening_params is None:
            raise ValueError(f"{market}: saved policy needs decay_rates and opening_params")
        opening_data = data[["date", "fighter_red", "fighter_blue", "dec_open_red", "dec_open_blue"]].copy()
    else:
        params = replace(params, decay_rates=None, opening_params=None)
    inputs = KellySimInput.from_predictions(
        data, params, odds_type=market, initial_bankroll=initial_bankroll,
        calc_parlay=True, opening_data=opening_data,
    )
    betting_data, kwargs = inputs.to_simulation_args()
    moneylines, parlays = simulate_kelly(betting_data, **kwargs)
    # Event totals repeat on every fight row: retain exactly one per date.
    daily = moneylines.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.sort_values("date").drop_duplicates("date").set_index("date")
    event_start = daily["bankroll_postevent"] - daily["event_payout_money_line"] - daily["parlay_net"]
    event_accuracy = data.assign(
        vegas_correct=(data["vegas_prediction"] == data["winner"]).astype(float)
    ).groupby("date").agg(vegas_accuracy=("vegas_correct", "mean"), fights=("winner", "size"))
    returns = pd.DataFrame({
        "moneyline_return_pct": 100 * daily["event_payout_money_line"] / event_start.where(event_start > 0),
        "parlay_return_pct": 100 * daily["parlay_net"] / event_start.where(event_start > 0),
    })
    returns["combined_return_pct"] = returns["moneyline_return_pct"] + returns["parlay_return_pct"]
    events = event_accuracy.join(returns, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    return events, moneylines, parlays


def plot_return_correlations(events, split, market):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    rows = []
    for ax, family in zip(axes, ("moneyline", "parlay", "combined")):
        x = events["vegas_accuracy"] * 100
        y = events[f"{family}_return_pct"]
        varying = len(events) >= 2 and x.nunique() > 1 and y.nunique() > 1
        r = x.corr(y) if varying else np.nan
        ax.scatter(x, y, alpha=.65)
        if varying:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.array([x.min(), x.max()])
            ax.plot(line_x, intercept + slope * line_x, color="tab:red")
        label = f"r={r:.3f}" if varying else "r undefined (insufficient variation)"
        ax.set(title=f"{family.title()} returns | {label}", xlabel="Vegas event accuracy (%)",
               ylabel="Return (% of event-start bankroll)", xlim=(-2, 102))
        ax.axhline(0, color="gray", linestyle=":")
        ax.grid(alpha=.2)
        rows.append(dict(split=split, market=market, family=family, events=len(events), pearson_r=r))
    fig.suptitle(f"{split.upper()} | {market} | Event accuracy versus strategy returns")
    return fig, rows


def save_png(fig, path, *, dpi=300):
    """Save a returned figure as PNG, creating parent folders; return its path."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    return path


def save_dataframe(frame, path, *, index=True):
    """Save results to CSV; retain named indexes (including coefficient terms)."""
    path = Path(path).expanduser()
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path
