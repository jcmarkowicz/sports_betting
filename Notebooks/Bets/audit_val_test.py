"""Reproduce validation/test diagnostics without fitting or overwriting predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ufc_betting.Results.test_simulation import simulate_saved_predictions
from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
    plot_event_accuracy_distributions, plot_bankroll_distributions,
)

OUT = Path('Data/backtest_results/val_test_audit')
OUT.mkdir(parents=True, exist_ok=True)
summaries = []
for split in ('val', 'test'):
    for market in ('open', 'close1', 'close2'):
        ml, pl, params, path, inputs, kwargs = simulate_saved_predictions(
            market, 500, split=split, use_decay=market != 'open')
        ml['date'] = pd.to_datetime(ml.date)
        pl['date'] = pd.to_datetime(pl.date)
        ml.to_csv(OUT / f'{split}_{market}_moneylines.csv', index=False)
        pl.to_csv(OUT / f'{split}_{market}_parlays.csv', index=False)
        daily = ml.groupby('date', sort=True).first()
        before = daily.bankroll_postevent - daily.event_payout_money_line - daily.parlay_net
        daily['ml_return_pct'] = 100 * daily.event_payout_money_line / before
        daily['pl_return_pct'] = 100 * daily.parlay_net / before
        daily['total_return_pct'] = daily.ml_return_pct + daily.pl_return_pct
        chosen_odds = (ml.choice_decimal_odds if 'choice_decimal_odds' in ml else
                       np.where(ml.pred_winner == 1, ml.bet_real_decimal_red, ml.bet_real_decimal_blue))
        chosen_proba = (ml.choice_proba if 'choice_proba' in ml else
                        np.where(ml.pred_winner == 1, ml.proba_red, ml.proba_blue))
        unit = np.where(ml.winner.isin([2, 3]), 0,
                        np.where(ml.pred_winner == ml.winner, chosen_odds - 1, -1))
        ml['independent_return'] = ml.choice_fstar * unit
        np.testing.assert_allclose(ml.groupby('date').independent_return.sum() * 100,
                                   daily.ml_return_pct, atol=1e-9)
        np.testing.assert_allclose(before.iloc[1:], daily.bankroll_postevent.iloc[:-1], atol=1e-8)
        assert not inputs.duplicated(['date', 'fighter_red', 'fighter_blue']).any()
        assert (ml.choice_fstar >= 0).all() and (daily.event_total_fstar <= 1 + 1e-10).all()
        # Independently settle one ticket per event, not one ticket per leg.
        ticket_returns = pd.Series(0., index=daily.index)
        wagered = ml.choice_fstar.gt(0).to_numpy(copy=True)
        for date, legs in pl.groupby('date', sort=True):
            ids = []
            for _, leg in legs.iterrows():
                match = (ml.date.eq(date) & (ml.fighter_red.eq(leg.choice_fighter_name)
                         | ml.fighter_blue.eq(leg.choice_fighter_name)))
                assert match.sum() == 1
                ids.append(np.flatnonzero(match)[0])
            outcomes = ml.iloc[ids].winner.to_numpy()
            sides = ml.iloc[ids].pred_winner.to_numpy()
            odds = np.asarray(chosen_odds)[ids]
            payout = (0. if np.isin(outcomes, [2, 3]).any() else
                      np.prod(odds) - 1 if np.array_equal(outcomes, sides) else -1.)
            fraction = float(legs.fstar_parlay.iloc[0])
            np.testing.assert_allclose(legs.fstar_parlay.to_numpy(float), fraction)
            ticket_returns.loc[date] = fraction * payout
            if fraction > 0:
                wagered[ids] = True
        np.testing.assert_allclose(100*ticket_returns, daily.pl_return_pct, atol=1e-9)
        fig, _, accuracy = plot_event_accuracy_distributions(ml, pl, odds_type=market)
        plt.close(fig)
        eligible = wagered & ml.winner.isin([0, 1]).to_numpy()
        expected_accuracy = (ml.pred_winner == ml.winner).loc[eligible].groupby(
            pd.to_datetime(ml.loc[eligible, 'date'])).mean()
        np.testing.assert_allclose(accuracy.your_picks.reindex(expected_accuracy.index),
                                   expected_accuracy, atol=1e-12)
        fig, _ = plot_bankroll_distributions(ml, pl)
        fig.savefig(OUT / f'{split}_{market}_distributions.png', dpi=100)
        plt.close('all')
        daily.join(accuracy).to_csv(OUT / f'{split}_{market}_events.csv')
        placed = ml.loc[(ml.choice_fstar > 0) & ml.winner.isin([0, 1])]
        summary = dict(split=split, market=market, fights=len(ml), events=len(daily),
                       first=str(daily.index.min()), last=str(daily.index.max()),
                       final_bankroll=daily.bankroll_postevent.iloc[-1],
                       ml_min=daily.ml_return_pct.min(), ml_max=daily.ml_return_pct.max(),
                       ml_mean=daily.ml_return_pct.mean(), ml_std=daily.ml_return_pct.std(),
                       ml_median=daily.ml_return_pct.median(),
                       mean_exposure_pct=100*daily.event_total_fstar.mean(),
                       placed=len(placed), placed_accuracy=(placed.pred_winner == placed.winner).mean(),
                       mean_pick_accuracy=accuracy.your_picks.mean(),
                       mean_vegas_accuracy=accuracy.vegas_all.mean(),
                       mean_wagered_vegas_accuracy=accuracy.vegas_wagered.mean(),
                       max_stake=ml.choice_fstar.max(),
                       max_proba=np.max(chosen_proba))
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        print(daily[['ml_return_pct', 'pl_return_pct', 'event_total_fstar']].sort_values('ml_return_pct').iloc[np.r_[0:2, -2:0]].to_string(), flush=True)
pd.DataFrame(summaries).to_csv(OUT / 'summary.csv', index=False)

# Use identical bin edges and density normalization to compare unequal cohorts.
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for column, market in enumerate(('open', 'close1', 'close2')):
    for split, color in (('val', 'tab:blue'), ('test', 'tab:orange')):
        events = pd.read_csv(OUT / f'{split}_{market}_events.csv')
        axes[0, column].hist(events.ml_return_pct, bins=np.arange(-70, 71, 10),
                            density=True, histtype='step', linewidth=2, color=color,
                            label=f'{split}: {len(events)} events')
        values = events.your_picks.dropna()*100
        axes[1, column].hist(values, bins=np.linspace(0, 100, 11), density=True,
                            histtype='step', linewidth=2, color=color, label=split)
    axes[0, column].set(title=f'{market}: moneyline event returns', xlabel='Starting bankroll return (%)', ylabel='Density')
    axes[1, column].set(title=f'{market}: wagered-pick event accuracy', xlabel='Correct picks (%)', ylabel='Density')
    for ax in axes[:, column]:
        ax.legend(); ax.grid(alpha=.2)
fig.savefig(OUT / 'comparison.png', dpi=150)
plt.close(fig)
