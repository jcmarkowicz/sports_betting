"""Descriptive attribution of validation/test moneyline event tails."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('Data/backtest_results/val_test_audit')
OUT = ROOT / 'vegas_exploration'
OUT.mkdir(parents=True, exist_ok=True)
keys = ['date', 'fighter_red', 'fighter_blue']
all_fights, event_rows = [], []
for split in ('val', 'test'):
    for market in ('open', 'close1', 'close2'):
        p = pd.read_csv(f'Data/model_predictions/{split}_logit_open_with_{market}.csv')
        ml = pd.read_csv(ROOT / f'{split}_{market}_moneylines.csv')
        f = p.merge(ml[keys + ['choice_fstar']], on=keys, validate='one_to_one')
        f['split'], f['market'] = split, market
        f['settled'] = f.winner.isin([0, 1])
        f['favorite'] = (f[f'dec_{market}_red'] < f[f'dec_{market}_blue']).astype(int)
        f['has_favorite'] = f[f'dec_{market}_red'] != f[f'dec_{market}_blue']
        f['vegas_correct'] = f.favorite.eq(f.winner)
        f['model_correct'] = f.pred_winner.eq(f.winner)
        f['agree'] = f.favorite.eq(f.pred_winner) & f.has_favorite
        f['bet'] = f.choice_fstar > 0
        f['pick'] = np.where(f.pred_winner == 1, f.fighter_red, f.fighter_blue)
        f['chosen_odds'] = np.where(f.pred_winner == 1, f[f'dec_{market}_red'], f[f'dec_{market}_blue'])
        f['chosen_p'] = np.where(f.pred_winner == 1, f.proba_red, f.proba_blue)
        f['market_p'] = np.where(f.pred_winner == 1, 1/f[f'dec_fair_{market}_red'], 1/f[f'dec_fair_{market}_blue'])
        f['favorite_p'] = np.where(f.favorite == 1, 1/f[f'dec_fair_{market}_red'], 1/f[f'dec_fair_{market}_blue'])
        f['unit_return'] = np.where(f.settled, np.where(f.model_correct, f.chosen_odds - 1, -1), 0)
        f['contribution_pct'] = 100*f.choice_fstar*f.unit_return
        f['market_expected_pct'] = 100*f.choice_fstar*(f.market_p*f.chosen_odds - 1)
        f['model_expected_pct'] = 100*f.choice_fstar*(f.chosen_p*f.chosen_odds - 1)
        f['opening_odds'] = np.where(f.pred_winner == 1, f.dec_open_red, f.dec_open_blue)
        f['movement'] = np.log(f.chosen_odds/f.opening_odds)
        all_fights.append(f)
        for date, g in f.groupby('date'):
            eligible = g.loc[g.settled & g.has_favorite]
            b = g.loc[g.settled & g.bet]
            agreed = b.loc[b.agree]
            disagreed = b.loc[~b.agree]
            stake = b.choice_fstar.sum()
            row = dict(split=split, market=market, date=date, fights=len(g),
                vegas_n=len(eligible), vegas_wins=int(eligible.vegas_correct.sum()),
                vegas_acc=eligible.vegas_correct.mean(), favorite_expected=eligible.favorite_p.mean(),
                bets=len(b), wins=int(b.model_correct.sum()), model_acc=b.model_correct.mean(),
                ml_return=g.contribution_pct.sum(), exposure=100*g.choice_fstar.sum(),
                favorite_bet_return=agreed.contribution_pct.sum(), other_bet_return=disagreed.contribution_pct.sum(),
                favorite_exposure=100*agreed.choice_fstar.sum(), other_exposure=100*disagreed.choice_fstar.sum(),
                agreed_bets=len(agreed), agreed_wins=int(agreed.model_correct.sum()),
                vegas_bet_acc=b.loc[b.has_favorite].vegas_correct.mean(),
                vegas_unbet_acc=eligible.loc[~eligible.bet].vegas_correct.mean(),
                stake_weighted_win=(b.choice_fstar*b.model_correct).sum()/stake if stake else np.nan,
                model_expected=g.loc[g.settled].model_expected_pct.sum(),
                market_expected=g.loc[g.settled].market_expected_pct.sum(),
                mean_bet_p=b.chosen_p.mean(), mean_market_p=b.market_p.mean(),
                effective_bets=stake**2/(b.choice_fstar**2).sum() if stake else 0)
            event_rows.append(row)
fights = pd.concat(all_fights, ignore_index=True)
events = pd.DataFrame(event_rows)
fights.to_csv(OUT/'fights.csv', index=False)
events.to_csv(OUT/'events.csv', index=False)
assert np.isfinite(events.ml_return).all()
for (split, market), e in events.groupby(['split', 'market']):
    original = pd.read_csv(ROOT/f'{split}_{market}_events.csv')
    paired = e.merge(original[['date', 'ml_return_pct']], on='date', validate='one_to_one')
    assert len(paired) == len(e)
    np.testing.assert_allclose(paired.ml_return, paired.ml_return_pct, atol=1e-9)

placed = fights.loc[fights.bet & fights.settled].copy()
placed['edge_pp'] = 100*(placed.chosen_p-placed.market_p)
placed['edge_band'] = pd.cut(placed.edge_pp, [-np.inf,5,10,15,np.inf], right=False,
                              labels=['<5','5-10','10-15','15+'])
bands, high_edge = [], []
for (split, market), g in placed.groupby(['split','market']):
    h = g.loc[g.edge_pp >= 10]
    high_edge.append(dict(split=split, market=market, n=len(h), wins=int(h.model_correct.sum()),
                          predicted=h.chosen_p.mean(), actual=h.model_correct.mean()))
    for band, h in g.groupby('edge_band', observed=True):
        bands.append(dict(split=split, market=market, edge_band=band, n=len(h),
                          predicted=h.chosen_p.mean(), actual=h.model_correct.mean(),
                          mean_stake=h.choice_fstar.mean(), sum_return_pp=h.contribution_pct.sum()))
pd.DataFrame(bands).to_csv(OUT/'edge_bands.csv', index=False)
pd.DataFrame(high_edge).to_csv(OUT/'high_edge_summary.csv', index=False)

# Check the distinction between plotted model-eligible fights and wider scraped cards.
raw = pd.read_csv('Data/test_data/test_data.csv')
coverage = []
for market in ('open','close1','close2'):
    p = fights.loc[(fights.split=='test') & (fights.market==market)]
    r = raw.merge(p[keys].assign(included=True), on=keys, how='left', validate='one_to_one')
    red, blue = f'dec_{market}_red', f'dec_{market}_blue'
    r = r.loc[r.winner.isin([0,1]) & r[[red,blue]].notna().all(axis=1) & r[red].ne(r[blue])].copy()
    r['correct'] = (r[red]<r[blue]).astype(int).eq(r.winner)
    for date, g in r.groupby('date'):
        for label, h in [('raw',g),('included',g.loc[g.included.eq(True)]),('excluded',g.loc[g.included.isna()])]:
            coverage.append(dict(market=market,date=date,cohort=label,n=len(h),
                                 wins=int(h.correct.sum()),accuracy=h.correct.mean()))
pd.DataFrame(coverage).to_csv(OUT/'raw_card_coverage.csv', index=False)
print('SUMMARY')
summaries = []
for (split, market), e in events.groupby(['split', 'market'], sort=False):
    f = fights.loc[(fights.split == split) & (fights.market == market)]
    b = f.loc[f.bet & f.settled]
    v = f.loc[f.settled & f.has_favorite]
    s = dict(split=split, market=market, vegas_event_mean=e.vegas_acc.mean(),
             vegas_fight_accuracy=v.vegas_correct.mean(), market_favorite_expected=v.favorite_p.mean(),
             vegas_bet_accuracy=b.loc[b.has_favorite].vegas_correct.mean(),
             vegas_unbet_accuracy=v.loc[~v.bet].vegas_correct.mean(),
             agreement=b.agree.mean(), favorite_stake_share=b.loc[b.agree].choice_fstar.sum()/b.choice_fstar.sum(),
             corr_vegas_return=e.vegas_acc.corr(e.ml_return),
             corr_bet_vegas_return=e.vegas_bet_acc.corr(e.ml_return),
             low_vegas_events=int(e.vegas_acc.lt(.5).sum()),
             low_vegas_mean_return=e.loc[e.vegas_acc.lt(.5), 'ml_return'].mean(),
             favorite_mean_return=e.favorite_bet_return.mean(), other_mean_return=e.other_bet_return.mean(),
             win_prob=b.chosen_p.mean(), market_prob=b.market_p.mean(), actual_win=b.model_correct.mean())
    summaries.append(s)
summary = pd.DataFrame(summaries)
summary.to_csv(OUT/'summary.csv', index=False)
print(summary.round(4).to_string(index=False))
print('\nTEST EVENTS BELOW -30%')
tails = events.loc[(events.split == 'test') & (events.ml_return < -30)]
print(tails.round(4).to_string(index=False))
tails.to_csv(OUT/'tail_events.csv', index=False)
print('\nLOW VEGAS VALIDATION EVENTS')
print(events.loc[(events.split == 'val') & (events.vegas_acc < .5)].sort_values(['market','vegas_acc']).round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for ax, market in zip(axes, ('open','close1','close2')):
    for split, color in (('val','tab:blue'),('test','tab:orange')):
        e = events.loc[(events.market == market) & (events.split == split)]
        ax.scatter(e.vegas_acc*100, e.ml_return, s=20+e.exposure*1.5,
                   color=color, alpha=.6, label=split)
        for _, r in e.loc[(e.split=='test') & (e.ml_return < -30)].iterrows():
            ax.annotate(r.date[5:], (r.vegas_acc*100,r.ml_return), xytext=(4,-10),
                        textcoords='offset points', fontsize=8)
    ax.axhline(0,color='gray',lw=1); ax.axvline(50,color='gray',lw=1,ls=':')
    ax.set(title=market, xlabel='Vegas favorite accuracy on available fights (%)',
           ylabel='Moneyline event return (% of bankroll)', ylim=(-70,65), xlim=(0,105))
    ax.grid(alpha=.2); ax.legend()
fig.suptitle('Bubble size represents moneyline exposure; labels identify test losses below 30%')
fig.savefig(OUT/'vegas_vs_returns.png', dpi=160)
plt.close(fig)
