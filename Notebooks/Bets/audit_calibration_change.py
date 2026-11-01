"""Describe the validation/test calibration gap without fitting or tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path('Data/backtest_results/val_test_audit')
OUT = ROOT / 'calibration_change'
OUT.mkdir(exist_ok=True)
fights = pd.read_csv(ROOT / 'vegas_exploration/fights.csv')
keys = ['date', 'fighter_red', 'fighter_blue']
rows, subgroups, periods, uncertainty = [], [], [], []
selected = {}
for (split, market), g in fights.groupby(['split', 'market']):
    h = g.loc[g.bet & g.settled & ((g.chosen_p-g.market_p) >= .10)].copy()
    selected[split,market] = h
    row = dict(split=split, market=market, n=len(h), predicted=h.chosen_p.mean(),
               actual=h.model_correct.mean(), market_probability=h.market_p.mean(),
               expected_extra_wins=(h.chosen_p-h.market_p).sum(),
               realized_extra_wins=(h.model_correct-h.market_p).sum())
    if split == 'test':
        old = pd.read_csv(ROOT / f'counterfactual_train_only_test_{market}.csv')
        z = h.merge(old[keys+['proba_red']], on=keys, suffixes=('', '_old'), validate='one_to_one')
        row['old_model_same_picks'] = np.where(z.pred_winner == 1,z.proba_red_old,1-z.proba_red_old).mean()
    rows.append(row)
    for agree, a in h.groupby('agree'):
        subgroups.append(dict(split=split,market=market,agrees_with_favorite=agree,n=len(a),
                              predicted=a.chosen_p.mean(),actual=a.model_correct.mean(),market_probability=a.market_p.mean()))
    h['period'] = pd.to_datetime(h.date).dt.to_period('Q').astype(str)
    for period, a in h.groupby('period'):
        periods.append(dict(split=split,market=market,period=period,n=len(a),predicted=a.chosen_p.mean(),actual=a.model_correct.mean()))

# Resample entire event dates, including those with no high-edge bets.
# Descriptive uncertainty only: cannot undo model/strategy selection or temporal drift.
rng = np.random.default_rng(124)
for market in ('open','close1','close2'):
    draws = {}
    for split in ('val','test'):
        h = selected[split,market].copy()
        h['residual'] = h.chosen_p - h.model_correct
        dates = fights.loc[(fights.split==split)&(fights.market==market),'date'].unique()
        grouped = h.groupby('date').agg(total=('residual','sum'),n=('residual','size')).reindex(dates,fill_value=0)
        sample = rng.integers(0,len(dates),(20000,len(dates)))
        totals = grouped.total.to_numpy()[sample].sum(axis=1)
        counts = grouped.n.to_numpy()[sample].sum(axis=1)
        draws[split] = totals/counts
    delta = draws['test']-draws['val']
    uncertainty.append(dict(market=market,lower_gap_change_pp=100*np.quantile(delta,.025),
                             upper_gap_change_pp=100*np.quantile(delta,.975)))

for name, data in [('summary',rows),('favorite_subgroups',subgroups),('quarters',periods),('event_bootstrap',uncertainty)]:
    frame = pd.DataFrame(data)
    frame.to_csv(OUT / f'{name}.csv',index=False)
    print(name, frame.to_string(index=False), sep='\n')

# Descriptive standardized input shifts among high-edge opening favorites.
# These are differences in inputs, not causal explanations.
scaler = joblib.load('Data/saved_models/scaler_open.pkl')
cols = list(scaler.feature_names_in_)
a = selected['val','open'].loc[lambda x:x.agree]
b = selected['test','open'].loc[lambda x:x.agree]
shift = pd.DataFrame({'feature':cols,'validation_mean':a[cols].mean().values,
                      'test_mean':b[cols].mean().values,
                      'shift_training_sd':(b[cols].mean().values-a[cols].mean().values)/scaler.scale_})
shift = shift.iloc[np.argsort(-shift.shift_training_sd.abs().to_numpy())]
shift.to_csv(OUT/'opening_favorite_feature_shifts.csv',index=False)
print('feature shifts (red/blue coordinates; descriptive)',shift.head(8).to_string(index=False),sep='\n')

high = fights.loc[fights.bet & fights.settled & ((fights.chosen_p-fights.market_p)>=.10)].copy()
high['period'] = pd.cut(pd.to_datetime(high.date),
    pd.to_datetime(['2024-01-01','2025-07-01','2026-02-21','2027-01-01']),right=False,
    labels=['validation before July 2025','validation July 2025 onward','test'])
high.groupby(['market','period'],observed=True).agg(n=('chosen_p','size'),
    predicted=('chosen_p','mean'),actual=('model_correct','mean'),market_probability=('market_p','mean')
    ).to_csv(OUT/'time_breakdown.csv')
