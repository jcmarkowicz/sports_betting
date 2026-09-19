"""Opposing decimal-odds paths for fixed opening and close1 parlay selections."""
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import (
    KEY, predict, fair_probabilities, interpolate_prices,
)
from ufc_betting.Models.RiskManagement.model_switch_evaluation import audit
from ufc_betting.Models.RiskManagement.parlay_switch_evaluation import select_legs, ticket, metrics


def switch_intervals(percent, gain, threshold=0.):
    """All switch regions on the evaluated grid; retain reversals and no-switch."""
    percent, gain = np.asarray(percent), np.asarray(gain)
    mask = gain > threshold
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(float(percent[g[0]]), float(percent[g[-1]])) for g in groups]


def build_paths(frame, model, scaler, steps=200):
    if steps < 2 or steps % 2:
        raise ValueError('steps must be even and >= 2 to include the midpoint')
    frame = frame.sort_values(KEY).reset_index(drop=True)
    op = frame[['dec_open_red', 'dec_open_blue']].to_numpy(float)
    cl = frame[['dec_close1_red', 'dec_close1_blue']].to_numpy(float)
    def probabilities(prices):
        fair = fair_probabilities(prices)
        features = frame.copy()
        features['proba_fair_open_diff'] = fair[:, 0] - fair[:, 1]
        return predict(features, model, scaler)
    p0, p1 = probabilities(op), probabilities(cl)
    cards = []
    for date, group in frame.groupby('date', sort=True):
        idx = group.index.to_numpy()
        group = group.reset_index(drop=True)
        opening_legs = select_legs(group, p0[idx], op[idx])
        closing_legs = select_legs(group, p1[idx], cl[idx])
        if set(opening_legs) != set(closing_legs):
            cards.append((date, idx, group, opening_legs, closing_legs))
    # Each market position is predicted once; the opposite path reads it in
    # reverse. Both red/blue quotes move so the fair probability stays defined.
    grid = np.linspace(0, 1, steps + 1)
    cache = [probabilities(interpolate_prices(op, cl, fraction)) for fraction in grid]
    rows, legs_rows = [], []
    for k, fraction in enumerate(grid):
        opening_prices = interpolate_prices(op, cl, fraction)
        closing_prices = interpolate_prices(cl, op, fraction)
        for date, idx, group, opening_legs, closing_legs in cards:
            row = dict(date=date, shift_pct=round(100 * fraction, 6))
            for label, probability, prices, legs in (
                ('opening', cache[k][idx], opening_prices[idx], opening_legs),
                ('close1', cache[steps-k][idx], closing_prices[idx], closing_legs),
            ):
                values = ticket(group, probability, prices, legs, mdd=.5, horizon=250)
                row.update({f'{label}_{name}': value for name, value in values.items()})
                for i, side in legs:
                    legs_rows.append(dict(date=date, shift_pct=row['shift_pct'], ticket=label,
                        fighter=group.iloc[i]['fighter_red' if side else 'fighter_blue'],
                        decimal_odds=float(prices[i, 1-side]),
                        probability=float(probability[i] if side else 1-probability[i])))
            for name in ('ev', 'expected_return', 'expected_log_growth', 'realized_return'):
                row[f'gain_{name}'] = row[f'close1_{name}'] - row[f'opening_{name}']
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(legs_rows)


def summarize(paths):
    crossings, aggregates = [], []
    for date, rows in paths.groupby('date', sort=True):
        rows = rows.sort_values('shift_pct')
        row = dict(date=date, opening_fighters=rows.opening_fighters.iloc[0],
                   close1_fighters=rows.close1_fighters.iloc[0])
        for name, threshold in [('positive_gain', 0.), ('two_point_gain', .02)]:
            zones = switch_intervals(rows.shift_pct, rows.gain_expected_return, threshold)
            row[f'{name}_intervals_pct'] = json.dumps(zones)
            row[f'{name}_first_switch_pct'] = zones[0][0] if zones else np.nan
            row[f'{name}_switch_at_start'] = bool(rows.gain_expected_return.iloc[0] > threshold)
            row[f'{name}_switch_at_end'] = bool(rows.gain_expected_return.iloc[-1] > threshold)
            row[f'{name}_regions'] = len(zones)
        for shift in (0, 50, 100):
            row[f'expected_gain_at_{shift}_pp'] = 100 * rows.loc[rows.shift_pct.eq(shift), 'gain_expected_return'].iloc[0]
        crossings.append(row)
    for shift, rows in paths.groupby('shift_pct', sort=True):
        rows = rows.sort_values('date')
        for policy, switch in (
            ('keep_opening', np.zeros(len(rows), bool)),
            ('always_close1', np.ones(len(rows), bool)),
            ('positive_expected_gain', rows.gain_expected_return.to_numpy() > 0),
            ('two_point_expected_gain', rows.gain_expected_return.to_numpy() > .02),
        ):
            realized = np.where(switch, rows.close1_realized_return, rows.opening_realized_return)
            expected = np.where(switch, rows.close1_expected_return, rows.opening_expected_return)
            aggregates.append(dict(shift_pct=shift, policy=policy, switches=int(switch.sum()),
                                   mean_expected_return_pct=100 * expected.mean(), **metrics(realized)))
    return pd.DataFrame(crossings), pd.DataFrame(aggregates)


def render(paths, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    pivot = paths.pivot(index='date', columns='shift_pct', values='gain_expected_return') * 100
    bound = max(abs(np.quantile(pivot.to_numpy(), [.05, .95])))
    plot = axes[0].imshow(pivot.to_numpy(), aspect='auto', origin='upper',
                          extent=[0, 100, len(pivot), 0], cmap='RdBu', vmin=-bound, vmax=bound)
    axes[0].set(title='Close1 minus opening expected bankroll return', xlabel='Movement toward opposite snapshot (%)',
                ylabel='Changed-selection events (chronological)')
    axes[0].axvline(50, color='black', linestyle=':', linewidth=1)
    fig.colorbar(plot, ax=axes[0], label='Percentage points (blue favors close1; scale clipped)')
    for policy, label in [('keep_opening', 'Keep opening fighters'), ('always_close1', 'Choose close1 fighters'),
                          ('positive_expected_gain', 'Switch if expected gain > 0'),
                          ('two_point_expected_gain', 'Switch if expected gain > 2 pp')]:
        rows = summary.loc[summary.policy.eq(policy)]
        axes[1].plot(rows.shift_pct, rows.compounded_return_pct, label=label)
    axes[1].set(title='Realized compounding at each hypothetical scenario',
                xlabel='Movement toward opposite snapshot (%)', ylabel='Compounded return (%)')
    axes[1].axvline(50, color='black', linestyle=':', linewidth=1)
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Fixed opening vs close1 parlay fighters | opposing decimal-odds paths')
    fig.supxlabel('0%: original snapshots. 50%: same midpoint market. 100%: exchanged snapshots. Each position is a separate scenario.', fontsize=9)
    fig.savefig(output / 'sensitivity.png', dpi=150)
    plt.close(fig)


def run():
    data_dir = settings.data_dir / 'model_results'
    output = data_dir / 'parlay_switch_sensitivity'
    op = pd.read_csv(data_dir / 'test_logit_open.csv').sort_values(KEY).reset_index(drop=True)
    cl = pd.read_csv(data_dir / 'test_logit_close1.csv').sort_values(KEY).reset_index(drop=True)
    if op.duplicated(KEY).any() or not op[KEY + ['winner']].equals(cl[KEY + ['winner']]):
        raise ValueError('Unmatched fight cohorts')
    for color in ('red', 'blue'):
        op[f'dec_close1_{color}'] = cl[f'dec_close1_{color}']
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    provenance = audit(model, scaler, 'open', data_dir)
    op['date'] = pd.to_datetime(op.date)
    op = op.loc[op.date > pd.Timestamp(provenance['last_training_date'])]
    paths, legs = build_paths(op, model, scaler)
    crossings, summary = summarize(paths)
    output.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output / 'event_paths.csv', index=False)
    legs.to_csv(output / 'leg_prices_and_probabilities.csv', index=False)
    crossings.to_csv(output / 'event_switch_intervals.csv', index=False)
    summary.to_csv(output / 'scenario_returns.csv', index=False)
    (output / 'methodology.json').write_text(json.dumps(dict(
        training_audit=provenance, changed_events=len(crossings), steps=200, resolution_pct=.5,
        opening_path='O + t*(C-O)', close1_path='C + t*(O-C)',
        probability='Frozen opening Logit, recomputed with interpolated two-sided power-devig feature',
        fixed_selections=True, mdd=.5, N=250, thresholds=[0., .02],
        limitations=['Two ticket-specific hypothetical price paths, not one contemporaneous market except at midpoint.',
                     'Shared fighters may have different ticket-specific quotes away from midpoint.',
                     'Scenario returns reuse the same 39 outcomes; not independent samples or chronological new tests.',
                     'The 2pp rule was selected retrospectively in the previous analysis, not validated here.',
                     'No reselection of fighters as odds move; no heavy-favorite parlays.',
                     'Interpolation is linear in decimal odds, not American odds or implied probabilities.']), indent=2), encoding='utf-8')
    render(paths, summary, output)
    print(summary.loc[summary.shift_pct.isin([0, 25, 50, 75, 100])].to_string(index=False))
    for name in ['positive_gain', 'two_point_gain']:
        print(name, 'switch at start:', int(crossings[f'{name}_switch_at_start'].sum()),
              'ever switch:', int(crossings[f'{name}_first_switch_pct'].notna().sum()),
              'multiple regions:', int((crossings[f'{name}_regions'] > 1).sum()))
    return paths, crossings, summary


if __name__ == '__main__':
    run()
