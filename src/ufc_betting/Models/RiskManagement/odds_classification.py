"""Classify historical choice-fighter rows by their source odds type.

Run with ``python -m ufc_betting.Models.RiskManagement.odds_classification``.
The configured fighter-model features retain their original red/blue orientation.
Only stage-specific current-price names are normalized before stacking rows.
Returns use one unit on each CSV's pred_winner at its actual source-stage odds;
the classifier partitions those bets, it does not replace their execution prices.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ufc_betting.config import config, settings


ODDS_TYPES = ("open", "close1", "close2")


def load_split(data_dir, split):
    """Build a shared feature matrix, integer labels, and aligned bet metadata."""
    if split not in ("train", "test"):
        raise ValueError("split must be train or test")
    features, metadata = [], []
    for odds_type in ODDS_TYPES:
        path = Path(data_dir) / f"{split}_logit_{odds_type}.csv"
        frame = pd.read_csv(path)
        configured = list(getattr(config, f"{odds_type}_feats"))
        required = configured + [
            "pred_winner", "winner", "date", "fighter_red", "fighter_blue",
            f"dec_{odds_type}_red", f"dec_{odds_type}_blue",
            "proba_fair_open_diff",
        ]
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if not frame.pred_winner.isin([0, 1]).all():
            raise ValueError(f"{path}: pred_winner must be 0 (blue) or 1 (red)")
        X = frame[configured].copy()
        # All stages describe the current price in the same feature column.
        X = X.rename(columns={
            f"proba_fair_{odds_type}_diff": "proba_fair_current_diff"
        })
        # For opens, opening and current probabilities are the same observation.
        X["proba_fair_open_diff"] = frame["proba_fair_open_diff"]
        features.append(X.apply(pd.to_numeric, errors="raise"))
        red = frame.pred_winner.eq(1)
        odds = np.where(red, frame[f"dec_{odds_type}_red"],
                        frame[f"dec_{odds_type}_blue"])
        if not (np.isfinite(odds) & (odds > 1)).all():
            raise ValueError(f"{path}: invalid selected decimal odds")
        settled = frame.winner.isin([0, 1])
        profit = np.where(settled,
                          np.where(frame.pred_winner.eq(frame.winner), odds - 1, -1),
                          np.nan)
        meta = frame[["date", "fighter_red", "fighter_blue", "pred_winner", "winner"]].copy()
        meta["date"] = pd.to_datetime(meta.date, errors="raise")
        meta["source_row"] = np.arange(len(frame))
        meta["actual_odds_type"] = odds_type
        meta["choice_fighter"] = np.where(red, frame.fighter_red, frame.fighter_blue)
        meta["choice_decimal_odds"] = odds
        meta["stake"] = np.where(settled, 1.0, 0.0)
        meta["net_profit"] = profit
        metadata.append(meta)
    # Reject unmatched schemas instead of exposing source labels via missingness.
    columns = list(features[0].columns)
    if any(set(X.columns) != set(columns) for X in features):
        raise ValueError("Configured feature lists must have a common normalized schema")
    X = pd.concat([X[columns] for X in features], ignore_index=True)
    meta = pd.concat(metadata, ignore_index=True)
    y = meta.actual_odds_type.map(dict(zip(ODDS_TYPES, range(3)))).astype(int)
    return X.replace([np.inf, -np.inf], np.nan), y, meta


def evaluate_returns(predictions):
    """Return all nine actual/predicted cells, with NaN ROI for empty cells."""
    grid = pd.MultiIndex.from_product(
        [ODDS_TYPES, ODDS_TYPES], names=["actual_odds_type", "predicted_odds_type"]
    )
    grouped = predictions.groupby(["actual_odds_type", "predicted_odds_type"])
    result = grouped.agg(
        n_predictions=("stake", "size"), n_bets=("net_profit", "count"),
        total_stake=("stake", "sum"), net_profit=("net_profit", "sum"),
    ).reindex(grid, fill_value=0)
    result["percent_return"] = 100 * result.net_profit / result.total_stake.replace(0, np.nan)
    return result.reset_index()


def run_classification(data_dir=None, output_dir=None, n_neighbors=25):
    data_dir = Path(data_dir) if data_dir is not None else settings.data_dir / "model_results"
    output_dir = Path(output_dir) if output_dir is not None else data_dir / "odds_classification"
    X_train, y_train, train_meta = load_split(data_dir, "train")
    X_test, y_test, test_meta = load_split(data_dir, "test")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns differ")
    # Existing CSVs can split one event at the holdout boundary. Preserve the
    # complete test set and purge contemporaneous/future training observations.
    before_test = train_meta.date < test_meta.date.min()
    purged_rows = int((~before_test).sum())
    X_train = X_train.loc[before_test].reset_index(drop=True)
    y_train = y_train.loc[before_test].reset_index(drop=True)
    train_meta = train_meta.loc[before_test].reset_index(drop=True)
    if set(y_train) != {0, 1, 2}:
        raise ValueError("Training must contain all three odds types before the test period")
    if not 1 <= n_neighbors <= len(X_train):
        raise ValueError("n_neighbors must be between 1 and the number of training rows")
    usable = X_train.columns[X_train.notna().any()]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(X_train[usable]))
    test_scaled = scaler.transform(imputer.transform(X_test[usable]))
    nonconstant = scaler.var_ > 0
    names = usable[nonconstant].tolist()
    train_scaled = train_scaled[:, nonconstant]
    test_scaled = test_scaled[:, nonconstant]
    train_design = sm.add_constant(pd.DataFrame(train_scaled, columns=names), has_constant="add")
    test_design = sm.add_constant(pd.DataFrame(test_scaled, columns=names), has_constant="add")
    if np.linalg.matrix_rank(train_design) != train_design.shape[1]:
        raise ValueError("Training features are linearly dependent; inspect the configured features")
    logit = sm.MNLogit(y_train.to_numpy(), train_design).fit(
        method="lbfgs", maxiter=2000, disp=False
    )
    if not logit.mle_retvals.get("converged", False):
        raise RuntimeError("Multinomial logistic regression did not converge")
    neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
    neighbors.fit(train_scaled, y_train)
    probabilities = {
        "mnlogit": np.asarray(logit.predict(test_design)),
        "nearest_neighbors": neighbors.predict_proba(test_scaled),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, returns = {}, []
    for model_name, proba in probabilities.items():
        if not np.isfinite(proba).all() or not np.allclose(proba.sum(axis=1), 1):
            raise RuntimeError(f"{model_name}: invalid predicted probabilities")
        predicted = proba.argmax(axis=1)
        rows = test_meta.copy()
        rows["predicted_odds_type"] = np.asarray(ODDS_TYPES)[predicted]
        for index, odds_type in enumerate(ODDS_TYPES):
            rows[f"probability_{odds_type}"] = proba[:, index]
        rows.to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)
        summary = evaluate_returns(rows)
        summary.insert(0, "model", model_name)
        returns.append(summary)
        metrics[model_name] = {
            "accuracy": accuracy_score(y_test, predicted),
            "classification_report": classification_report(
                y_test, predicted, labels=[0, 1, 2], target_names=ODDS_TYPES,
                output_dict=True, zero_division=0),
        }
        pd.DataFrame(confusion_matrix(y_test, predicted, labels=[0, 1, 2]),
                     index=pd.Index(ODDS_TYPES, name="actual_odds_type"),
                     columns=ODDS_TYPES).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
        print(f"\n{model_name}: accuracy={metrics[model_name]['accuracy']:.2%}")
        print(summary.pivot(index="actual_odds_type", columns="predicted_odds_type",
                            values="percent_return").reindex(index=ODDS_TYPES, columns=ODDS_TYPES).round(2))
    returns = pd.concat(returns, ignore_index=True)
    returns.to_csv(output_dir / "returns_by_actual_and_predicted.csv", index=False)
    logit.save(output_dir / "mnlogit.pkl")
    joblib.dump({"model": neighbors, "imputer": imputer, "scaler": scaler,
                 "input_columns": usable.tolist(), "nonconstant": nonconstant,
                 "model_columns": names, "classes": ODDS_TYPES},
                output_dir / "preprocessor_and_neighbors.joblib")
    metrics["data"] = {
        "train_rows": len(X_train), "test_rows": len(X_test), "features": names,
        "purged_train_rows": purged_rows,
        "n_neighbors": n_neighbors, "class_order": ODDS_TYPES,
        "train_end": str(train_meta.date.max()), "test_start": str(test_meta.date.min()),
        "return_definition": "100 * sum(net_profit) / sum(stake); one unit per settled choice",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"mnlogit": logit, "nearest_neighbors": neighbors,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "returns": returns, "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-neighbors", type=int, default=25)
    args = parser.parse_args()
    run_classification(args.data_dir, args.output_dir, args.n_neighbors)
