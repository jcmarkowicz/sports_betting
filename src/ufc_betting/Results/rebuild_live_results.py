"""Recalculate saved live results from local features without scraping or publishing."""
import shutil
import pandas as pd

from ufc_betting.config import config, settings
from ufc_betting.UpcomingPicks.generate_bets import generate_bets
from ufc_betting.DataPipeline.dataframes.moneylines import MoneylineDataFrame
from ufc_betting.DataPipeline.dataframes.parlays import ParlayDataFrame
from ufc_betting.Results.returns import returns_by_date, accuracy_analysis


def rebuild_live_results():
    """Use Event Scaling's trainval opening inference fit and market policies.

    Only rebuild dates already in the live archive. Back up existing outputs
    before writing; do not scrape, commit, push, or send notifications.
    """
    features = pd.read_csv(config.test_feats_file, parse_dates=['date'])
    existing = pd.read_csv(settings.ml_history_file, parse_dates=['date'])
    dates = set(existing.date.dt.normalize())
    features['date'] = features.date.dt.normalize()
    missing = dates - set(features.date)
    if missing:
        raise ValueError(f'Missing saved features for live dates: {sorted(missing)}')
    moneylines, parlays = [], []
    for date, event in features.loc[features.date.isin(dates)].groupby('date'):
        event = event.reset_index(drop=True)
        ml, pl = generate_bets(event, bankroll=500)
        moneylines.append(MoneylineDataFrame.from_generated(ml).with_results(event).frame)
        parlays.append(ParlayDataFrame.from_generated(pl, date).with_results(event).frame)
    ml, pl = pd.concat(moneylines, ignore_index=True), pd.concat(parlays, ignore_index=True)
    balances = returns_by_date(ml_results=ml, parlay_results=pl, persist=False)
    accuracy, bet_types = accuracy_analysis(ml_results=ml, parlay_results=pl)
    outputs = {
        settings.ml_history_file: ml, settings.parlay_history_file: pl,
        settings.bankroll_returns_file: balances,
        settings.betting_results_dir / 'prediction_accuracies.csv': accuracy,
        settings.betting_results_dir / 'bet_types.csv': bet_types,
    }
    backup = settings.betting_results_dir / 'backups' / pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, backup / path.name)
    for path, frame in outputs.items():
        frame.to_csv(path, index=False)
    return {'events': len(dates), 'fights': len(ml), 'backup': str(backup)}


if __name__ == '__main__':
    print(rebuild_live_results())
