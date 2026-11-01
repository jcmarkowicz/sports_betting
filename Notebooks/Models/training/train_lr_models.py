"""Train/save logistic comparisons, or refresh reports from existing predictions.

Run this script from the project environment. The manual notebook only reads
the reports. --reports-only never fits models or changes prediction CSVs.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from ufc_betting.config import config, settings
from lr_workflow import run_comparison, score_predictions

INDEX = ['fit_mode', 'split', 'model', 'odds', 'probability_model']
METRICS = ['accuracy', 'parlay_accuracy_pct', 'f1', 'brier',
           'expected_calibration_error', 'mean_calibration_error']


def placed_parlay_accuracy(parlays, moneylines):
    """Winning settled tickets / placed settled tickets, counting each ticket once."""
    if parlays.empty:
        return np.nan
    tickets = parlays.loc[parlays.fstar_parlay > 0].copy()
    if {'bankroll_postevent', 'event_payout_money_line', 'parlay_net'} <= set(moneylines):
        active = moneylines.assign(before=moneylines.bankroll_postevent -
            moneylines.event_payout_money_line - moneylines.parlay_net)
        active_dates = set(pd.to_datetime(active.loc[active.before > 0, 'date']).dt.normalize())
        tickets = tickets.loc[pd.to_datetime(tickets.date).dt.normalize().isin(active_dates)]
    keys = ['date'] + (['parlay_strategy'] if 'parlay_strategy' in tickets else [])
    if (tickets.groupby(keys).parlay_net_odds.nunique(dropna=False) > 1).any():
        raise ValueError('Inconsistent settlement across ticket legs')
    tickets = tickets.drop_duplicates(keys)
    settled = tickets.loc[tickets.parlay_net_odds.notna() & tickets.parlay_net_odds.ne(0)]
    return 100 * settled.parlay_net_odds.gt(0).mean() if len(settled) else np.nan


def save_reports(output_dir):
    """Recompute every metric from its saved predictions and save indexed CSVs/plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    root = Path(output_dir)
    source = pd.read_csv(root / 'metrics.csv')
    if source.duplicated(INDEX).any():
        raise ValueError('Duplicate model/data comparison rows')
    from ufc_betting.Results.test_simulation import simulate_saved_predictions
    rows, predictions = [], {}
    source_records = source.to_dict('records')
    def read_prediction(record):
        path = Path(record['prediction_path'])
        if not path.exists():
            path = root / record['fit_mode'] / record['model'] / record['probability_model'] / f"{record['split']}_with_{record['odds']}.csv"
        return pd.read_csv(path)
    for record in source_records:
        if record['fit_mode'] == 'trainval' and record['split'] == 'val':
            continue
        frame = read_prediction(record)
        if record['fit_mode'] == 'trainval' and record['split'] == 'train':
            validation = next(r for r in source_records if r['split']=='val' and
                all(r[k]==record[k] for k in ('fit_mode','model','odds','probability_model')))
            frame = pd.concat([frame, read_prediction(validation)], ignore_index=True)
            # These are the rows actually used to fit train+validation models.
            frame = frame.loc[frame.winner.isin([0, 1])].copy()
        path = root / 'report_predictions' / record['fit_mode'] / record['model'] / record['probability_model'] / f"{record['split']}_with_{record['odds']}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        if not frame.probability_model.eq(record['probability_model']).all():
            raise ValueError(f'Probability source mismatch: {path}')
        record.update(score_predictions(frame, int(record.get('n_bins', 10))))
        moneylines, parlays, *_ = simulate_saved_predictions(record['odds'],
            predictions_path=path.resolve(), model_type=record['model'], split=record['split'])
        record['parlay_accuracy_pct'] = placed_parlay_accuracy(parlays, moneylines)
        record['expected_calibration_error'] = record['ece']
        record['prediction_path'] = str(path.resolve())
        rows.append(record)
        predictions[tuple(record[k] for k in INDEX)] = frame
    metrics = pd.DataFrame(rows).sort_values(INDEX)
    # Keep the detailed machine-readable summary and explicitly indexed report.
    # Keep original split predictions as inputs so reports-only is idempotent.
    report_columns = METRICS
    metrics.set_index(INDEX)[report_columns].to_csv(root / 'comparison_metrics.csv')
    plot_rows = []
    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    for (mode, split), group in metrics.groupby(['fit_mode', 'split']):
        group.set_index(['model', 'odds', 'probability_model'])[report_columns].to_csv(
            root / f'{mode}_{split}_metrics.csv')
        for probability_model, panel in group.groupby('probability_model'):
            label = 'held-out' if panel.held_out.all() else 'in-sample'
            title = f'{split} | {mode} fit | {probability_model} | {label}'
            stem = f'{mode}_{split}_{probability_model}'
            fig, ax = plt.subplots(figsize=(8, 6))
            for row in panel.to_dict('records'):
                frame = predictions[tuple(row[k] for k in INDEX)]
                settled = frame.loc[frame.winner.isin([0, 1])]
                actual, predicted = calibration_curve(settled.winner, settled.proba_red,
                    n_bins=int(row['n_bins']), strategy='uniform')
                ax.plot(predicted, actual, marker="o", label=f"{row['model']} -> {row['odds']}")
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set(xlabel='Mean predicted red-win probability', ylabel='Observed red-win rate', title=title)
            ax.legend()
            calibration_path = plots / f'{stem}_calibration.png'
            fig.savefig(calibration_path, dpi=130, bbox_inches='tight')
            plt.close(fig)
            panel.set_index(['model', 'odds'])[report_columns].to_csv(
                root / f'{stem}_metrics.csv')
            plot_rows.append(dict(fit_mode=mode, split=split, probability_model=probability_model,
                calibration_plot=str(calibration_path.resolve())))
    pd.DataFrame(plot_rows).to_csv(root / 'plots.csv', index=False)
    (root / 'metric_definitions.json').write_text(json.dumps({
        'accuracy': 'Fraction of settled fights with correct predicted winner.',
        'f1': 'Binary F1 with red win as positive class and threshold 0.5.',
        'brier': 'Mean squared error of red-win probabilities.',
        'expected_calibration_error': 'Count-weighted mean absolute calibration gap across fixed equal-width bins.',
        'mean_calibration_error': 'Unweighted mean absolute calibration gap across nonempty bins.',
        'maximum_calibration_error': 'Largest absolute calibration gap across nonempty bins; legacy mce.',
        'parlay_accuracy_pct': 'Percentage of winning settled tickets with positive stakes from the shared Kelly simulation and current market-specific kelly_params; opening has no decay, closing uses configured decay. Count each date/strategy ticket once; exclude refunded and unplaced tickets.',
        'units': 'Parlay accuracy is a percentage [0,100]; other metrics are fractions [0,1]. Undefined parlay accuracy is blank (NaN).'
    }, indent=2), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=settings.all_feats_dir / 'entire_odds_stats_2026-10-11.csv')
    parser.add_argument('--test-data', type=Path, default=config.test_feats_file)
    parser.add_argument('--output-dir', type=Path, default=settings.data_dir / 'model_comparisons/lr_open_close')
    parser.add_argument('--reports-only', action='store_true')
    parser.add_argument('--alphas-json', type=Path, help='JSON mapping train/trainval to open/close1/close2 alpha values.')
    args = parser.parse_args()
    if not args.reports_only:
        alphas = json.loads(args.alphas_json.read_text()) if args.alphas_json else None
        run_comparison(pd.read_csv(args.history), pd.read_csv(args.test_data), args.output_dir, alphas=alphas)
    metrics = save_reports(args.output_dir)
    print(f'Saved {len(metrics)} metric rows and reports to {args.output_dir}')


if __name__ == '__main__':
    main()
