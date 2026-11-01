"""Training, saved-model inference, and comparable logistic-model experiments.

Run from lr_open_close.ipynb or import this module; importing never trains models.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from ufc_betting.config import (config, settings, TRAIN_START_YEAR, TRAIN_START_MONTH,
    TRAIN_START_DAY, VAL_START_DATE, TEST_START_DATE)
from ufc_betting.Models.LogisticRegression.train_test_builder import TrainTestBuilder
from ufc_betting.Models.LogisticRegression.inference import predict_logit
from ufc_betting.Models.LogisticRegression.statistical_functions import train_logit_model
from ufc_betting.Models.LogisticRegression.visual_functions import plot_model_metrics
from ufc_betting.Models.LogisticRegression.utils import save_results

winner_col, date_col = 'winner', 'date'

def score_predictions(frame, n_bins=10):
    """Binary red-win metrics; ECE uses fixed equal-width bins, including p=1."""
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    scored = frame.loc[frame.winner.isin([0, 1])]
    if scored.empty:
        raise ValueError('No settled fights available for evaluation')
    y = scored.winner.to_numpy(int)
    p = scored.proba_red.to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between zero and one')
    # Match sklearn.calibration.calibration_curve's boundary convention.
    bins = np.searchsorted(np.linspace(0, 1, n_bins + 1)[1:-1], p)
    errors = [(np.abs(p[bins == i].mean() - y[bins == i].mean()), (bins == i).sum())
              for i in range(n_bins) if (bins == i).any()]
    return dict(n_rows=len(frame), n_scored=len(scored), n_void_or_missing=len(frame)-len(scored),
                accuracy=accuracy_score(y, p >= .5), f1=f1_score(y, p >= .5, zero_division=0),
                brier=brier_score_loss(y, p), ece=sum(gap*n for gap,n in errors)/len(y),
                mean_calibration_error=float(np.mean([gap for gap,n in errors])),
                maximum_calibration_error=max(gap for gap,n in errors),
                mce=max(gap for gap,n in errors), n_bins=n_bins)


def run_comparison(history, test_data, output_dir, *, alphas=None, feature_sets=None,
                   val_start_date=VAL_START_DATE, test_start_date=TEST_START_DATE,
                   fit_modes=('train', 'trainval'), probability_models=('regularized', 'inference'),
                   n_bins=10):
    """Fit each market once per mode and save predictions/metrics for all splits.

    History supplies only pre-test rows; test_data supplies the entire test cohort.
    All candidates share complete-case fight keys so metric differences cannot
    arise from different evaluation samples. Validation is in-sample for trainval.
    Artifacts live beneath output_dir and never replace the canonical bet exports.
    """
    feature_sets = feature_sets or {'open': config.open_feats,
        'close1': config.open_close1_feats, 'close2': config.open_close2_feats}
    alphas = alphas or {mode: {m: (6.985 if mode == 'train' else 3.66)
                              for m in feature_sets} for mode in fit_modes}
    if set(feature_sets) != {'open', 'close1', 'close2'}:
        raise ValueError('Provide open, close1, and close2 feature sets')
    if not fit_modes or not set(fit_modes) <= {'train', 'trainval'}:
        raise ValueError('fit_modes must contain train and/or trainval')
    if not probability_models or not set(probability_models) <= {'regularized', 'inference'}:
        raise ValueError('Unknown probability model')
    for mode in fit_modes:
        for market in feature_sets:
            if not np.isfinite(alphas[mode][market]) or alphas[mode][market] < 0:
                raise ValueError('Each model needs a finite nonnegative alpha')
    val_start, test_start = pd.Timestamp(val_start_date), pd.Timestamp(test_start_date)
    if val_start >= test_start:
        raise ValueError('Validation must start before test')
    history, test_data = history.copy(), test_data.copy()
    for frame in (history, test_data):
        frame['date'] = pd.to_datetime(frame.date, errors='raise').dt.normalize()
        if frame.date.isna().any():
            raise ValueError('Missing fight dates')
    data = pd.concat([history.loc[history.date < test_start],
                      test_data.loc[test_data.date >= test_start]], ignore_index=True)
    keys = ['date', 'fighter_red', 'fighter_blue']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate fight keys in the combined data')
    required = set()
    for market, feats in feature_sets.items():
        required.update(make_builder(data, feats, market).valid_cols)
    eligible = data.dropna(subset=sorted(required))
    start = pd.Timestamp(TRAIN_START_YEAR, TRAIN_START_MONTH, TRAIN_START_DAY)
    eligible = eligible.loc[(eligible.date > start) & eligible.winner.isin([0,1,2,3])].copy()
    eligible = eligible.sort_values(keys, kind='stable').reset_index(drop=True)
    splits = {'train': eligible.loc[(eligible.date < val_start) & eligible.winner.isin([0,1])],
              'val': eligible.loc[(eligible.date >= val_start) & (eligible.date < test_start)],
              'test': eligible.loc[eligible.date >= test_start]}
    if any(not f.winner.isin([0,1]).any() for f in splits.values()):
        raise ValueError('Every split needs settled eligible fights')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for split, frame in splits.items():
        frame.to_csv(root / f'cohort_{split}.csv', index=False)
    manifest = dict(val_start=str(val_start), test_start=str(test_start), alphas=alphas,
                    features={k:list(v) for k,v in feature_sets.items()}, fit_modes=list(fit_modes),
                    probability_models=list(probability_models), n_bins=n_bins,
                    raw_rows=len(data), common_rows=len(eligible),
                    cohort_sizes={k:len(v) for k,v in splits.items()})
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summaries = []
    for mode in fit_modes:
        for model_type, feats in feature_sets.items():
            directory = root / mode / model_type
            artifacts = {name: directory / f'{name}.pkl'
                         for name in ('model', 'model_se', 'scaler', 'encoder')}
            kwargs = {f'{name}_path': path for name,path in artifacts.items()}
            workflow(feats, model_type, **kwargs, df_model=eligible,
                     alpha=alphas[mode][model_type], train_on_train_val=mode=='trainval',
                     save_model=True, show_plots=False, results_dir=directory/'fit_predictions',
                     plots_dir=directory/'plots', val_start_date=val_start, test_start_date=test_start,
                     **{f'{split}_split_path':directory/f'{split}_cohort.csv' for split in splits})
            markets = ('open','close1','close2') if model_type == 'open' else (model_type,)
            for odds_type in markets:
                for probability_model in probability_models:
                    for split, frame in splits.items():
                        path = directory / probability_model / f'{split}_with_{odds_type}.csv'
                        prediction = run_inference(frame, **kwargs, model_type=model_type,
                            odds_type=odds_type, probability_model=probability_model, pred_save_path=path)
                        if len(prediction) != len(frame):
                            raise ValueError('Inference dropped rows from the common cohort')
                        summaries.append(dict(fit_mode=mode, model=model_type, odds=odds_type,
                            probability_model=probability_model, split=split,
                            held_out=split=='test' or (split=='val' and mode=='train'),
                            alpha=alphas[mode][model_type], prediction_path=str(path.resolve()),
                            **score_predictions(prediction, n_bins),
                            **parlay_accuracy_metrics(prediction, odds_type)))
            pd.DataFrame(summaries).to_csv(root/'metrics.csv', index=False)
    return pd.DataFrame(summaries)

def make_builder(df, feats, odds_type):
    return TrainTestBuilder(
        df=df, feats=feats, target_col=winner_col, date_col=date_col,
        odds_type=odds_type, year=TRAIN_START_YEAR, month=TRAIN_START_MONTH, day=TRAIN_START_DAY,
        retain_void_evaluation=True,
    )

def workflow(
    feats, odds_type, model_path=None, model_se_path=None, scaler_path=None,
    encoder_path=None, save_model=False, alpha=0,
    val_start_date=VAL_START_DATE, test_start_date=TEST_START_DATE,
    train_split_path=None, val_split_path=None, test_split_path=None,
    pred_output_filename=None, train_on_train_val=False,
    df_model=None, results_dir=None, plots_dir=None, show_plots=True,
):
    """Optionally fit train + val, reserving test for held-out evaluation.

    Combined-fit prediction and plot filenames receive a trainval_fit_ prefix
    so they cannot replace the train-only workflow outputs.
    """
    import pandas as pd

    if df_model is None:
        raise ValueError('Pass df_model explicitly; workflow does not read notebook globals')
    results_dir = Path(results_dir) if results_dir is not None else settings.data_dir / 'model_predictions'
    plots_dir = Path(plots_dir) if plots_dir is not None else settings.images_dir / 'plot_pngs'
    pkt = make_builder(df_model, feats, odds_type).prepare_train_test(
        val_start_date=val_start_date, test_start_date=test_start_date, scale=not train_on_train_val,
        train_split_path=train_split_path, val_split_path=val_split_path,
        test_split_path=test_split_path,
    )
    if train_on_train_val:
        # Fit transforms on the complete fitting cohort, never on test rows.
        X_fit_raw = pd.concat([pkt['X_train'], pkt['X_val']])
        y_fit = pd.concat([pkt['y_train'], pkt['y_val']])
        fit_mask = y_fit.isin([0, 1])
        X_fit_raw, y_fit = X_fit_raw.loc[fit_mask], y_fit.loc[fit_mask]
        fitted = TrainTestBuilder.scale_encode(
            X_fit_raw, pkt['X_test'], pkt['cat_cols'], pkt['num_cols'],
        )
        pkt['scaler'], pkt['encoder'] = fitted['scaler'], fitted['encoder']
        X_fit = fitted['X_train']
        for split in ('train', 'val'):
            pkt[f'X_{split}'] = TrainTestBuilder.prepare_inference_data(
                pkt[f'X_{split}'], pkt['scaler'], pkt['encoder'])
        pkt['X_test'] = fitted['X_test']
        evaluation_split = 'test'
    else:
        X_fit, y_fit = pkt['X_train'], pkt['y_train']
        evaluation_split = 'val'
    results = train_logit_model(
        X_fit, y_fit, pkt[f'X_{evaluation_split}'], pkt[f'y_{evaluation_split}'],
        cov_type='HC3', reg=True, alpha=alpha,
        evaluation_label=evaluation_split.upper(),
    )
    if save_model:
        for path in (model_path, model_se_path, scaler_path, encoder_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        results['model'].save(model_path)
        results['model_se'].save(model_se_path)
        joblib.dump(pkt['scaler'], scaler_path)
        joblib.dump(pkt['encoder'], encoder_path)

    outputs = []
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Keep train+val diagnostics separate from held-out validation outputs.
    output_prefix = "trainval_fit_" if train_on_train_val else ""
    # A custom filename applies to validation only; the mode prefix still applies.
    prediction_paths = {
        split: results_dir / (output_prefix + (pred_output_filename if split == 'val' and pred_output_filename
                              else f'{split}_logit_{odds_type}.csv'))
        for split in ('train', 'val', 'test')
    }
    if len({path.resolve() for path in prediction_paths.values()}) != 3:
        raise ValueError('Prediction output paths for train, val, and test must be different')
    for split in ('train', 'val', 'test'):
        predictions = predict_logit(pkt[f'X_{split}'], results['model'], results['model_se'])
        rows = pkt[f'{split}_df'].copy()
        rows['proba_se'] = predictions['proba_se']
        # Predict every eligible fight; score only binary outcomes.
        scored = pkt[f'y_{split}'].isin([0, 1]).to_numpy()
        if show_plots:
            plot_model_metrics(
                pkt[f'y_{split}'].to_numpy()[scored], predictions['pred_winner'].to_numpy()[scored],
                predictions['proba_red'].to_numpy()[scored],
                suptitle=(f'{split.title()} Results for {odds_type} Odds'
                          + (' (in-sample)' if split == 'train' or
                             (train_on_train_val and split == 'val') else ' (held-out)')),
                path=plots_dir / f'{output_prefix}{split}_{odds_type}_model_metrics.png',
            )
        outputs.append(save_results(
            rows, predictions['proba_red'],
            prediction_paths[split],
        ))
    return tuple(outputs)

def parlay_accuracy_metrics(results, odds_type, target_col="winner", date_col="date"):
    """Score settled, positive-EV top-two tickets; no stake sizing or decay.

    Select legs before checking outcomes, so void/unlabelled fights are never
    replaced with other picks. Exclude such tickets from win-rate denominator.
    """
    import numpy as np
    import pandas as pd

    frame = results.copy().reset_index(drop=True)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
    if frame[date_col].isna().any():
        raise ValueError("Parlay evaluation requires nonmissing event dates")
    side = frame["pred_winner"].to_numpy()
    if not np.isin(side, [0, 1]).all():
        raise ValueError("Parlay predictions must be 0 or 1")
    p = frame["proba_red"].to_numpy(float)
    odds = frame[[f"dec_{odds_type}_blue", f"dec_{odds_type}_red"]].to_numpy(float)
    if (not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.isfinite(odds).all() or (odds <= 1).any()):
        raise ValueError("Parlay probabilities and decimal odds must be valid and finite")
    frame["_parlay_p"] = np.where(side == 1, p, 1 - p)
    frame["_parlay_odds"] = odds[np.arange(len(frame)), side.astype(int)]
    frame["_parlay_ev"] = frame["_parlay_p"] * frame["_parlay_odds"] - 1
    wins = tickets = skipped = excluded = 0
    for _, group in frame.groupby(date_col, sort=True):
        legs = group.sort_values("_parlay_ev", ascending=False, kind="stable").head(2)
        if len(legs) < 2 or legs["_parlay_p"].prod() * legs["_parlay_odds"].prod() - 1 <= 0:
            skipped += 1
            continue
        if not legs[target_col].isin([0, 1]).all():
            excluded += 1
            continue
        tickets += 1
        wins += int(legs["pred_winner"].eq(legs[target_col]).all())
    accuracy = wins / tickets if tickets else float("nan")
    return dict(parlay_accuracy=accuracy, parlay_accuracy_pct=100 * accuracy,
                parlay_wins=wins, parlay_tickets=tickets,
                parlay_skipped_events=skipped, parlay_excluded_tickets=excluded)

def run_inference(
    df, *, model_path, model_se_path, scaler_path, encoder_path,
    model_type, odds_type, probability_model="regularized",
    report_metrics=False, show_plots=False, plot_path=None,
    target_col="winner", plot_title=None, pred_save_path=None
):
    """Run a saved logistic model on supplied rows without fitting or splitting.

    model_type identifies the model's training market: open, close1, or close2.
    Opening models substitute closing features when odds_type is close1/close2.
    All other combinations retain the supplied feature values.
    probability_model="regularized" uses model_path for probabilities (default).
    probability_model="inference" uses model_se_path for probabilities.
    Standard errors always come from model_se_path. Metrics, classes, plots and
    saved probabilities all follow the selected probability model.
    report_metrics prints fight accuracy, Brier and positive-EV top-two parlay
    accuracy, and stores them in results.attrs['metrics']. Parlay accuracy excludes
    void/unsettled tickets and does not apply stake sizing or decay adjustments.
    show_plots displays plot_model_metrics; plot_path optionally saves that figure
    (and also enables plotting). Evaluation uses only outcomes 0/1, retaining all
    rows in the returned predictions. Unlabelled rows and outcomes 2/3 are excluded.
    """

    if model_type not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown model_type')
    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown odds_type')
    if probability_model not in ("regularized", "inference"):
        raise ValueError('probability_model must be "regularized" or "inference"')
    if model_se_path is None:
        raise ValueError("model_se_path is required for the inference model and standard errors")

    # Keep void outcomes for betting selection; exclude only unavailable model/price inputs.
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    required = list(scaler.feature_names_in_) + list(encoder.feature_names_in_)
    if model_type == 'open' and odds_type in ('close1', 'close2'):
        required = [name.replace('proba_fair_open_diff', f'proba_fair_{odds_type}_diff') for name in required]
    required += [f'dec_{odds_type}_{side}' for side in ('red', 'blue')]
    df = df.dropna(subset=required).copy()

    inputs = TrainTestBuilder.prepare_inference_data(
        df,
        scaler=joblib.load(scaler_path),
        encoder=joblib.load(encoder_path),
        switch_open_odds=model_type == 'open' and odds_type in ('close1', 'close2'),
        odds_type=odds_type,
    )
    inference_model = sm.load(model_se_path)
    selected_model = sm.load(model_path) if probability_model == "regularized" else inference_model
    predictions = predict_logit(inputs, selected_model, inference_model)
    results = df.copy()
    for column in predictions:
        results[column] = predictions[column].to_numpy()

    results["probability_model"] = probability_model
    results = save_results(
        df_model=results, 
        proba_red=predictions['proba_red'].to_numpy(), 
        fp=None,
        target_col=target_col,
    )
    # Preserve the selected feature set and metadata rather than requiring the
    # fixed feature schema of the old canonical opening-model betting exports.
    if pred_save_path is not None:
        Path(pred_save_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(pred_save_path, index=False)

    if report_metrics or show_plots or plot_path is not None:
        from sklearn.metrics import accuracy_score, brier_score_loss

        if target_col not in results:
            raise ValueError(f'Evaluation requires the {target_col!r} column')
        evaluated = results.loc[results[target_col].isin([0, 1])].reset_index(drop=True)
        if evaluated.empty:
            raise ValueError('Evaluation requires at least one outcome of 0 or 1')
        y_true = evaluated[target_col].astype(int)
        metrics = {
            'accuracy': accuracy_score(y_true, evaluated['pred_winner']),
            'brier_score': brier_score_loss(y_true, evaluated['proba_red']),
            'n_evaluated': len(evaluated),
            'n_excluded': len(results) - len(evaluated),
        }
        metrics.update(parlay_accuracy_metrics(results, odds_type, target_col=target_col))
        results.attrs['metrics'] = metrics
        if report_metrics:
            print(f"Probability model: {probability_model}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Brier score: {metrics['brier_score']:.4f}")
            print(f"Evaluated: {metrics['n_evaluated']}; excluded: {metrics['n_excluded']}")
            if metrics["parlay_tickets"]:
                print(f"Parlay accuracy (top-2 EV): {metrics['parlay_accuracy_pct']:.2f}% "
                      f"({metrics['parlay_wins']}/{metrics['parlay_tickets']} settled positive-EV tickets)")
            else:
                print("Parlay accuracy (top-2 EV): N/A (no settled positive-EV tickets)")
            print(f"Parlay skipped events (fewer than 2 legs or nonpositive EV): "
                  f"{metrics['parlay_skipped_events']}; void/unsettled tickets excluded: "
                  f"{metrics['parlay_excluded_tickets']}")
        if show_plots or plot_path is not None:
            if y_true.nunique() < 2:
                print('Metric plots skipped: ROC requires both outcomes 0 and 1.')
            else:
                if plot_path is not None:
                    plot_path = Path(plot_path)
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                plot_model_metrics(
                    y_true, evaluated['pred_winner'], evaluated['proba_red'],
                    suptitle=plot_title or f'{model_type.title()} Model Results for {odds_type} Odds',
                    path=plot_path,
                )
    return results
