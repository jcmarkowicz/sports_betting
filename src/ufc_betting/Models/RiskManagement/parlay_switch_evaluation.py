"""Frozen opening Logit: switch top-two-EV fighters at current offered odds.

Only events whose opening and close1 selected fighter sets differ are scored.
Opening selection is repriced and re-estimated at the decision snapshot; no
already-placed ticket is cancelled. Heavy-favorite parlays are not included.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ufc_betting.config import config, settings
from ufc_betting.BettingStrategy.kelly_scaling import scale_kelly_for_mdd
from ufc_betting.Models.RiskManagement.event_odds_sensitivity import KEY, predict, fair_probabilities
from ufc_betting.Models.RiskManagement.model_switch_evaluation import audit


def select_legs(frame, probabilities, prices):
    side = (np.asarray(probabilities) >= .5).astype(int)
    chosen_p = np.where(side == 1, probabilities, 1 - probabilities)
    chosen_odds = prices[np.arange(len(side)), 1 - side]  # prices red, blue
    order = np.argsort(-(chosen_p * chosen_odds - 1), kind='stable')[:2]
    return [(int(i), int(side[i])) for i in order]


def ticket(frame, probabilities, prices, legs, mdd=.5, horizon=250):
    """Return ticket EV, bankroll fraction and realized return (all fractions)."""
    if len(legs) != 2:
        raise ValueError('Exactly two legs required')
    p = float(np.prod([probabilities[i] if side else 1 - probabilities[i]
                       for i, side in legs]))
    odds = float(np.prod([prices[i, 1 - side] for i, side in legs]))
    ev = p * odds - 1
    full = max(ev / (odds - 1), 0)
    f = scale_kelly_for_mdd(p, odds, full, horizon, mdd)
    won = all(frame.winner.iloc[i] == side for i, side in legs)
    realized = f * (odds - 1 if won else -1)
    names = sorted(frame.iloc[i]['fighter_red' if side else 'fighter_blue'] for i, side in legs)
    return dict(fighters=' | '.join(names), probability=p, decimal_odds=odds,
                ev=ev, stake_fraction=f, expected_return=f * ev,
                realized_return=realized, won=won,
                expected_log_growth=p * np.log1p(f * (odds - 1)) + (1 - p) * np.log1p(-f))


def build_events(frame, model, scaler, mdd=.5, horizon=250):
    frame = frame.sort_values(KEY, kind='stable').reset_index(drop=True)
    snapshots = {}
    for stage in ('open', 'close1', 'close2'):
        prices = frame[[f'dec_{stage}_red', f'dec_{stage}_blue']].to_numpy(float)
        if not np.isfinite(prices).all() or (prices <= 1).any():
            raise ValueError(f'Invalid {stage} prices')
        fair = fair_probabilities(prices)
        features = frame.copy()
        features['proba_fair_open_diff'] = fair[:, 0] - fair[:, 1]
        snapshots[stage] = (predict(features, model, scaler), prices)
    records = []
    for date, group in frame.groupby('date', sort=True):
        if len(group) < 2:
            raise ValueError(f'Event {date} has fewer than two fights')
        idx = group.index.to_numpy()
        group = group.reset_index(drop=True)
        stage_data = {stage: (p[idx], prices[idx]) for stage, (p, prices) in snapshots.items()}
        legs = {stage: select_legs(group, p, prices) for stage, (p, prices) in stage_data.items()}
        row = dict(date=date, fight_count=len(group),
                   changed_open_close1=set(legs['open']) != set(legs['close1']))
        for stage, (p, prices) in stage_data.items():
            for choice in (['selected'] if stage == 'open' else ['selected', 'keep_open']):
                selected = legs['open'] if choice == 'keep_open' else legs[stage]
                values = ticket(group, p, prices, selected, mdd, horizon)
                row.update({f'{stage}_{choice}_{name}': value for name, value in values.items()})
            if stage != 'open':
                row[f'{stage}_changed'] = set(legs['open']) != set(legs[stage])
                for metric in ('ev', 'expected_return', 'expected_log_growth'):
                    row[f'{stage}_delta_{metric}'] = row[f'{stage}_selected_{metric}'] - row[f'{stage}_keep_open_{metric}']
        records.append(row)
    return pd.DataFrame(records)


def apply_rule(events, stage, rule):
    kind, threshold = rule
    if kind == 'keep':
        switch = np.zeros(len(events), bool)
    elif kind == 'switch':
        switch = np.ones(len(events), bool)
    else:
        switch = events[f'{stage}_delta_{kind}'].to_numpy() > threshold
    switch &= events[f'{stage}_changed'].to_numpy(bool)
    returns = np.where(switch, events[f'{stage}_selected_realized_return'],
                       events[f'{stage}_keep_open_realized_return'])
    return switch, returns


def candidate_rules():
    # Fixed grids, not thresholds inferred from later outcomes. EV values are
    # fractions: .05 is five percentage points of expected return on stake.
    return [('keep', 0.), ('switch', 0.)] + [
        (kind, round(float(t), 8)) for kind, grid in (
            ('ev', np.arange(-.20, .501, .025)),
            ('expected_return', np.arange(-.05, .1001, .005)),
            ('expected_log_growth', [0.]),
        ) for t in grid
    ]


def growth(returns):
    return float(np.log1p(np.asarray(returns)).sum())


def metrics(returns):
    returns = np.asarray(returns)
    wealth = np.r_[1., np.cumprod(1 + returns)]
    return dict(events=len(returns), mean_event_return_pct=100 * returns.mean(),
                compounded_return_pct=100 * (wealth[-1] - 1),
                profitable_events=int((returns > 0).sum()),
                max_drawdown_pct=100 * np.max(1 - wealth / np.maximum.accumulate(wealth)))


def evaluate(events, stage, minimum_training_events=12):
    """Expanding earlier-event tuning, updated at calendar-quarter boundaries."""
    if len(events) <= minimum_training_events:
        raise ValueError('Insufficient changed-selection events for validation')
    rules = candidate_rules()
    scored = {rule: apply_rule(events, stage, rule) for rule in rules}
    first_date = pd.Timestamp(events.date.iloc[minimum_training_events])
    blocks, heldout = [], []
    for quarter in sorted(pd.to_datetime(events.date).dt.to_period('Q').unique()):
        start = max(quarter.start_time, first_date)
        train = pd.to_datetime(events.date) < start
        test = (pd.to_datetime(events.date) >= start) & (pd.to_datetime(events.date) < (quarter + 1).start_time)
        if train.sum() < minimum_training_events or not test.any():
            continue
        scores = {rule: growth(values[1][train]) for rule, values in scored.items()}
        best = max(rules, key=lambda rule: scores[rule])
        blocks.append(dict(stage=stage, test_start=str(start.date()),
                           training_events=int(train.sum()), test_events=int(test.sum()),
                           rule=best[0], threshold=best[1]))
        for policy, rule in [('learned', best), ('keep_open', ('keep', 0.)),
                              ('always_switch', ('switch', 0.)), ('positive_ev_gain', ('ev', 0.)),
                              ('positive_bankroll_ev_gain', ('expected_return', 0.)),
                              ('positive_expected_log_gain', ('expected_log_growth', 0.))]:
            switch, returns = apply_rule(events.loc[test], stage, rule)
            heldout.append(pd.DataFrame(dict(date=events.loc[test, 'date'], stage=stage,
                                             policy=policy, switched=switch,
                                             event_return=returns)))
    heldout = pd.concat(heldout, ignore_index=True)
    table = []
    for policy, group in heldout.groupby('policy'):
        table.append(dict(stage=stage, policy=policy, switches=int(group.switched.sum()),
                          **metrics(group.sort_values('date').event_return)))
    full_grid = pd.DataFrame([dict(stage=stage, rule=rule[0], threshold=rule[1],
                                  log_growth=growth(returns), switches=int(switch.sum()),
                                  **metrics(returns)) for rule, (switch, returns) in scored.items()])
    return pd.DataFrame(table), pd.DataFrame(blocks), heldout, full_grid


def run():
    data_dir = settings.data_dir / 'model_results'
    output = data_dir / 'parlay_switch_evaluation'
    frames = {s: pd.read_csv(data_dir / f'test_logit_{s}.csv').sort_values(KEY).reset_index(drop=True)
              for s in ('open', 'close1', 'close2')}
    for stage, frame in frames.items():
        if frame.duplicated(KEY).any() or not frame.winner.isin([0, 1]).all():
            raise ValueError(f'Invalid fight cohort: {stage}')
        if not frame[KEY + ['winner']].equals(frames['open'][KEY + ['winner']]):
            raise ValueError(f'Mismatched fight cohort: {stage}')
    model, scaler = sm.load(config.model_open_path), joblib.load(config.scaler_open_path)
    provenance = audit(model, scaler, 'open', data_dir)
    frame = frames['open'].copy()
    # Use each snapshot's own quoted odds; all non-market features stay fixed.
    for stage in ('close1', 'close2'):
        for color in ('red', 'blue'):
            frame[f'dec_{stage}_{color}'] = frames[stage][f'dec_{stage}_{color}']
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.loc[frame.date > pd.Timestamp(provenance['last_training_date'])]
    events = build_events(frame, model, scaler)
    eligible = events.loc[events.changed_open_close1].reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    # Identical choices are retained only in an exclusion audit, never scored.
    events.loc[~events.changed_open_close1, ['date', 'open_selected_fighters', 'close1_selected_fighters']].to_csv(output / 'excluded_same_fighters.csv', index=False)
    eligible.to_csv(output / 'changed_selection_events.csv', index=False)
    results = [evaluate(eligible, stage) for stage in ('close1', 'close2')]
    filenames = ['heldout_metrics', 'selected_rules_by_quarter', 'heldout_event_returns', 'retrospective_threshold_grid']
    tables = {}
    for i, name in enumerate(filenames):
        tables[name] = pd.concat([r[i] for r in results], ignore_index=True)
        tables[name].to_csv(output / f'{name}.csv', index=False)
    descriptive = pd.DataFrame([dict(stage=stage, **metrics(eligible[f'{stage}_selected_realized_return']))
                                 for stage in ('open', 'close1', 'close2')])
    descriptive.to_csv(output / 'selected_ticket_descriptive_returns.csv', index=False)
    paired = []
    rng = np.random.default_rng(42)
    for stage, group in tables['heldout_event_returns'].groupby('stage'):
        pivot = group.pivot(index='date', columns='policy', values='event_return').sort_index()
        for policy in pivot.columns.drop('keep_open'):
            delta = pivot[policy].to_numpy() - pivot.keep_open.to_numpy()
            draws = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(axis=1)
            paired.append(dict(stage=stage, policy=policy, mean_gain_pp=100 * delta.mean(),
                               bootstrap_low_pp=100 * np.quantile(draws, .025),
                               bootstrap_high_pp=100 * np.quantile(draws, .975)))
    pd.DataFrame(paired).to_csv(output / 'paired_event_bootstrap.csv', index=False)
    metadata = dict(training_audit=provenance, total_events=len(events),
                    changed_selection_events=len(eligible), excluded_same_fighters=len(events)-len(eligible),
                    mdd=.5, N=250, initial_rule_training_events=12,
                    selection='Top two individual EV fighters; no heavy-favorite parlays',
                    probability='Fixed opening model with snapshot fair-market feature; leg probabilities multiplied',
                    comparison='Keep opening fighters at current snapshot prices versus current selected fighters; both resized at same MDD/N',
                    evaluation='Earlier changed-selection events tune threshold, next quarter tests it; close2 never informs close1 choices',
                    limitations=['Conditional on the available modeled fight cohort, not necessarily every fight on each card.',
                                 'Retrospective threshold-grid winner is descriptive, not an independently validated optimum.',
                                 'Bootstrap is paired by event, assumes independent events, and is exploratory.',
                                 'These historical outcomes have been examined in previous analyses.'])
    (output / 'methodology.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    render(tables['heldout_event_returns'], output)
    print(json.dumps(metadata, indent=2))
    print(tables['heldout_metrics'].to_string(index=False))
    print(tables['selected_rules_by_quarter'].to_string(index=False))
    print('FULL SAMPLE BEST (DESCRIPTIVE ONLY)')
    print(tables['retrospective_threshold_grid'].sort_values('log_growth', ascending=False).groupby('stage').head(3).to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))
    return output, tables, eligible


def render(heldout, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels = {'keep_open': 'Keep opening fighters', 'always_switch': 'Always switch',
              'learned': 'Threshold selected on earlier events'}
    colors = {'keep_open': '#157a6e', 'always_switch': '#b85c38', 'learned': '#5261ac'}
    for ax, stage in zip(axes, ['close1', 'close2']):
        for policy, label in labels.items():
            rows = heldout.loc[heldout.stage.eq(stage) & heldout.policy.eq(policy)].sort_values('date')
            ax.plot(np.arange(len(rows) + 1), np.r_[100, 100 * np.cumprod(1 + rows.event_return)],
                    label=label, color=colors[policy], linewidth=2)
        ax.set(title=f'{stage}: both choices priced at {stage}', xlabel='Later evaluation events',
               ylabel='Bankroll (starting at 100)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Changed parlay selections | frozen opening Logit | MDD 0.50, N 250')
    fig.supxlabel('Only events with different opening/close1 fighter sets. Thresholds use earlier events only; no heavy-favorite parlays.', fontsize=9)
    fig.savefig(output / 'heldout_bankroll.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run()
