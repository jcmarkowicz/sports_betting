"""Rebuild features locally without scraping, committing, or fitting models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ufc_betting.config import config, settings
from ufc_betting.DataPipeline.dataframes.stats import StatsRepository
from ufc_betting.DataPipeline.dataframes.odds import OddsRepository
from ufc_betting.DataPipeline.FeatureEngineering.features_pipeline import FeatureEngineering

OUT = Path('Data/backtest_results/val_test_audit/features')
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ['date', 'fighter_red', 'fighter_blue']
FEATURES = config.open_feats + ['proba_fair_close1_diff','proba_fair_close2_diff']

def compare(a, b, label):
    a, b = a.copy(), b.copy()
    a['date'], b['date'] = pd.to_datetime(a.date), pd.to_datetime(b.date)
    paired = a.merge(b, on=KEYS, suffixes=('_a','_b'), validate='one_to_one')
    rows = []
    examples = []
    for c in FEATURES:
        x, y = paired[c+'_a'].to_numpy(float), paired[c+'_b'].to_numpy(float)
        bad = ~np.isclose(x,y,atol=1e-9,rtol=1e-8,equal_nan=True)
        rows.append(dict(feature=c, differences=int(bad.sum()),
                         max_abs_difference=float(np.nanmax(np.abs(x-y))) if len(x) else 0))
        if bad.any():
            ex = paired.loc[bad,KEYS+[c+'_a',c+'_b']].copy()
            ex = ex.rename(columns={c+'_a':'a',c+'_b':'b'})
            ex['feature'] = c
            examples.append(ex)
    pd.DataFrame(rows).to_csv(OUT/f'{label}_summary.csv',index=False)
    if examples: pd.concat(examples).to_csv(OUT/f'{label}_differences.csv',index=False)
    print(label, 'rows',len(a),len(b),'matched',len(paired),
          'differences',[r for r in rows if r['differences']],flush=True)
    return paired

if __name__ == '__main__':
    stats = StatsRepository(settings.stats_history_file, settings.non_merged_stats_file).load_combined()
    odds = OddsRepository(settings.odds_history_file, settings.non_merged_odds_file).load_combined()
    print('raw combined',stats.shape,odds.shape,stats.event_date.min(),stats.event_date.max(),flush=True)
    full, _ = FeatureEngineering().build_all_stats(stats,None,odds,None,ignore_upcoming=True)
    full.to_pickle(OUT/'rebuilt_original.pkl')
    old = pd.read_csv(settings.all_feats_dir/'entire_odds_stats_2026-10-11.csv')
    new = pd.read_csv(config.test_feats_file)
    compare(old,new,'old_vs_test')
    compare(old,full,'old_vs_rebuilt')
    compare(new,full,'test_vs_rebuilt')
