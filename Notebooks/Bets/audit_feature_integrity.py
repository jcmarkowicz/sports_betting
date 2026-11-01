"""Offline integrity checks; never replace production data or model artifacts."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from ufc_betting.config import config, settings
from ufc_betting.DataPipeline.dataframes.stats import StatsRepository
from ufc_betting.DataPipeline.dataframes.odds import OddsRepository
from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering
from ufc_betting.DataPipeline.FeatureEngineering.BuildFeatures.odds_features import build_odds_features

OUT = Path('Data/backtest_results/val_test_audit/features')
KEYS = ['date', 'fighter_red', 'fighter_blue']


def predict(frame, test):
    suffix = '_test' if test else ''
    model = joblib.load(f'Data/saved_models/logit_model{suffix}_open.pkl')
    scaler = joblib.load(f'Data/saved_models/scaler_open{suffix}.pkl')
    columns = list(scaler.feature_names_in_)
    x = pd.DataFrame(scaler.transform(frame[columns]), columns=columns, index=frame.index)
    x['const'] = 1.
    x['elo_pred_1'] = (frame.elo_pred == 1).astype(float)
    return expit(x[model.params.index].to_numpy() @ model.params.to_numpy())


def impact(a, b, label, test):
    a, b = a.set_index(KEYS), b.set_index(KEYS)
    index = a.index.intersection(b.index)
    a, b = a.loc[index], b.loc[index]
    valid = a[config.open_feats].notna().all(axis=1) & b[config.open_feats].notna().all(axis=1)
    a, b = a.loc[valid], b.loc[valid]
    result = pd.DataFrame({'p_a': predict(a, test), 'p_b': predict(b, test)}, index=a.index)
    result['difference_pp'] = 100 * (result.p_b - result.p_a)
    result.to_csv(OUT / f'{label}_probabilities.csv')
    print(label, 'rows',len(result), 'max change pp',result.difference_pp.abs().max(),
          'mean change pp',result.difference_pp.abs().mean(), flush=True)
    print(result.reindex(result.difference_pp.abs().sort_values(ascending=False).index).head(5), flush=True)


def main():
    old = pd.read_csv(settings.all_feats_dir / 'entire_odds_stats_2026-10-11.csv')
    new = pd.read_csv(config.test_feats_file)
    baseline = pd.read_pickle(OUT / 'rebuilt_original.pkl')
    for frame in (old, new, baseline):
        frame['date'] = pd.to_datetime(frame.date)
    val_keys = pd.read_csv('Data/model_predictions/val_logit_open_with_open.csv', parse_dates=['date'])[KEYS]
    validation = old.merge(val_keys, on=KEYS, validate='one_to_one')
    impact(old.loc[old.date >= '2026-02-21'], new, 'old_to_new_test', True)
    impact(validation, baseline,
           'validation_to_raw_rebuild', False)
    engine = FeatureEngineering()
    raw = StatsRepository(settings.stats_history_file, settings.non_merged_stats_file).load_combined()
    odds = OddsRepository(settings.odds_history_file, settings.non_merged_odds_file).load_combined()
    def capture_conflicts(frame):
        keys = ['red_clean', 'blue_clean', 'date']
        quotes = [f'{m}_{s}' for m in ('open', 'close1', 'close2') for s in ('red', 'blue')]
        grouped = frame.groupby(keys, dropna=False)[quotes].nunique(dropna=False).gt(1).any(axis=1)
        conflicts = frame.set_index(keys).loc[grouped[grouped].index].reset_index()
        conflicts.to_csv(OUT / 'conflicting_odds.csv', index=False)
        return FeatureEngineering.unique_fight_odds(frame)
    engine.unique_fight_odds = capture_conflicts
    try:
        engine.standardized_merge(baseline[KEYS], build_odds_features(odds))
    except ValueError as error:
        print(str(error), flush=True)
    print('Rebuilding chronological features', flush=True)
    corrected = engine.standardize_features(raw)
    for column in ('fighter_red', 'fighter_blue'):
        corrected[column] = engine.clean_col(corrected[column])
    corrected.to_pickle(OUT / 'chronological_stats.pkl')
    # Hold odds fixed to isolate changes in statistical features.
    stats_cols = [c for c in config.open_feats if not c.startswith('proba_fair')]
    replacement = baseline.drop(columns=stats_cols).merge(corrected[KEYS+stats_cols], on=KEYS, validate='one_to_one')
    impact(new, replacement, 'test_chronology_only', True)
    impact(validation,
           replacement, 'validation_chronology_and_source_updates', False)


if __name__ == '__main__':
    main()
