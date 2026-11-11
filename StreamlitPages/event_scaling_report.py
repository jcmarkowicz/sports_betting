"""Event Scaling page data and figures, using the shared dataframe scaling API.

Plot definitions mirror trend_smoothing_open.ipynb; no notebook execution or saved
notebook output is needed at runtime. Source CSVs are read only.
"""
from io import BytesIO
from threading import RLock

import matplotlib
matplotlib.use("Agg")  # Render server-side without a GUI backend.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ufc_betting.config import settings
from ufc_betting.BettingStrategy.Scaling import (
    ScalingConfig, calculate_brier_scores, calculate_event_history,
    fit_trend_scale, calculate_event_trends, apply_event_filters,
)

PREDICTIONS = settings.data_dir / 'model_comparisons/lr_open_close/report_predictions'
BACKTESTS = settings.data_dir / 'backtest_results/model_comparisons/inference_market_models'
PLOT_LOCK = RLock()
KEYS = ['date', 'fighter_red', 'fighter_blue']


def source_signature():
    """Invalidate the Streamlit cache when a prediction or backtest CSV changes."""
    paths = list(PREDICTIONS.glob('*/open/inference/*_with_*.csv'))
    paths += list(BACKTESTS.glob('*_open_with_*_moneylines.csv'))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(paths))


def metrics(values):
    r = np.asarray(values, float)/100
    if not len(r) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError('Metrics require finite event returns above -100%.')
    wealth = np.r_[1., np.cumprod(1+r)]
    return dict(events=len(r), geometric_growth_pct=100*np.expm1(np.log1p(r).mean()),
        average_return_pct=100*r.mean(), volatility_pp=100*r.std(ddof=1),
        worst_event_pct=100*r.min(), max_drawdown_pct=-100*(wealth/np.maximum.accumulate(wealth)-1).min(),
        final_wealth=wealth[-1])


def score_fights(frame, market):
    return calculate_brier_scores(frame, odds_type=market)


def load_market(market, config=ScalingConfig()):
    parts=[];histories={}
    train_end=pd.to_datetime(pd.read_csv(PREDICTIONS/'train/open/inference/train_with_open.csv').date).max()
    for label,fit,split,prefix in [('Validation','train','val','train_val'),('Test','trainval','test','trainval_test')]:
        root=PREDICTIONS/fit/'open/inference'
        warm=pd.read_csv(root/f'train_with_{market}.csv')
        pred=pd.read_csv(root/f'{split}_with_{market}.csv',parse_dates=['date'])
        tag='no_decay' if market=='open' else 'decay'
        saved=pd.read_csv(BACKTESTS/f'{prefix}_open_with_{market}_inference_{tag}_moneylines.csv',parse_dates=['date'])
        for frame in [saved,pred]:frame['date']=frame.date.dt.normalize()
        matched=saved[KEYS].merge(pred,on=KEYS,how='outer',indicator=True,validate='one_to_one',sort=False)
        if not matched['_merge'].eq('both').all():raise ValueError(f'{market}: saved fights and predictions differ')
        current=saved[KEYS].merge(pred,on=KEYS,validate='one_to_one',sort=False)
        reference=current.proba_red.to_numpy()
        saved_probability=saved.proba_red if 'proba_red' in saved else saved.choice_proba.where(saved.pred_winner.eq(1),1-saved.choice_proba)
        np.testing.assert_allclose(saved_probability,reference,atol=1e-10)
        if pd.to_datetime(warm.date).max()>=current.date.min():raise ValueError('Warmup overlaps evaluation')
        history=calculate_brier_scores(pd.concat([warm,current],ignore_index=True),odds_type=market,config=config)
        histories[label]=history
        event_history=calculate_event_history(history,config=config)
        scale=fit_trend_scale(event_history,training_end=train_end)
        cols=['bankroll_postevent','event_payout_money_line','parlay_net','event_total_fstar']
        if not saved.groupby('date')[cols].nunique(dropna=False).le(1).all().all():raise ValueError('Inconsistent event accounting')
        event=saved.groupby('date')[cols].first().sort_index()
        profit=event.event_payout_money_line+event.parlay_net;start=event.bankroll_postevent-profit
        if not start.gt(0).all():raise ValueError('Invalid initial event bankroll')
        np.testing.assert_allclose(start.iloc[1:],event.bankroll_postevent.iloc[:-1],rtol=1e-8,atol=1e-8)
        target=pd.DataFrame({'return_pct':100*profit/start,'stake_fraction':event.event_total_fstar,'split':label})
        features=calculate_event_trends(target,event_history,trend_scale=scale,config=config)
        parts.append(features)
    events=pd.concat(parts).sort_index()
    if not events.index.is_unique:raise ValueError('Overlapping validation and test dates')
    applied=apply_event_filters(events,events,config=config)
    scalers=applied[['volatility_multiplier','brier_multiplier']].rename(columns={'volatility_multiplier':'vol_multiplier'})
    scalers['combined_multiplier']=scalers.vol_multiplier*scalers.brier_multiplier
    events['opening_keep']=events.trend.isin(['Uptrend','Downtrend'])
    events['opening_prior_slope']=events.slope
    active=events.loc[events.opening_keep].join(scalers)
    for label,factor in [('Volatility only','vol_multiplier'),('Prior-Brier reductions','brier_multiplier'),('Combined','combined_multiplier')]:
        active[label]=active.return_pct*active[factor]
    active['Original']=active.return_pct
    return dict(events=events,filtered=active,histories=histories,all_scalers=scalers)


def _capture(fig, figures):
    buffer=BytesIO()
    fig.savefig(buffer,format='png',dpi=120,bbox_inches='tight')
    figures.append(buffer.getvalue())
    plt.close(fig)


def opening_report():
    with PLOT_LOCK:
        return _opening_report()


def _opening_report():
    figures=[]
    smoothing_results={};rows=[]
    for method in ['EWMA','Rolling mean']:
        for window in [5,10]:
            for cutoff in [.02,.03]:
                config=ScalingConfig(smoothing='ewma' if method=='EWMA' else 'rolling',smoothing_events=window,flat_cutoff=cutoff)
                result=load_market('open',config)
                smoothing_results[(method,window,cutoff)]=result
                for period,part in [('Combined',result['filtered']),*list(result['filtered'].groupby('split',sort=False))]:
                    for strategy,col in [('Flat filter only','Original'),('Prior-Brier reductions','Prior-Brier reductions'),('Volatility only','Volatility only'),('Combined','Combined')]:
                        rows.append(dict(smoothing=method,window=window,cutoff=cutoff,period=period,strategy=strategy,**metrics(part[col])))
    smoothing_metrics=pd.DataFrame(rows)
    raw_events=next(iter(smoothing_results.values()))['events']
    raw_metrics=metrics(raw_events.return_pct)

    # Plot family from opening notebook cell 3
    scenario_keys = list(smoothing_results)
    scenario_labels = [f'{method} {window}\n+/-{cutoff:.2f}' for method, window, cutoff in scenario_keys]
    strategies = ['Flat filter only', 'Prior-Brier reductions', 'Volatility only', 'Combined']
    fig, axes = plt.subplots(2, 2, figsize=(19, 11), constrained_layout=True)
    for ax, column, title in zip(axes.flat, ['geometric_growth_pct', 'average_return_pct', 'volatility_pp', 'max_drawdown_pct'], ['Geometric growth per included event (%)', 'Average event return (%)', 'Event return volatility (percentage points)', 'Maximum drawdown (%)']):
        for offset, strategy in zip([-0.3, -0.1, 0.1, 0.3], strategies):
            values = []
            for method, window, cutoff in scenario_keys:
                row = smoothing_metrics.loc[(smoothing_metrics.smoothing == method) & (smoothing_metrics.window == window) & (smoothing_metrics.cutoff == cutoff) & (smoothing_metrics.period == 'Combined') & (smoothing_metrics.strategy == strategy)].iloc[0]
                values.append(row[column])
            ax.bar(np.arange(8) + offset, values, width=0.19, label=strategy)
        ax.axhline(raw_metrics[column], color='gray', ls='--', label='Raw unfiltered baseline')
        ax.set_xticks(np.arange(8), scenario_labels, fontsize=8)
        ax.set(title=title)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    _capture(fig, figures)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True, constrained_layout=True)
    for ax, strategy in zip(axes.flat, strategies):
        for key, result in smoothing_results.items():
            method, window, cutoff = key
            col = 'Original' if strategy == 'Flat filter only' else strategy
            r = result['filtered'][col].reindex(raw_events.index, fill_value=0)
            ax.plot(r.index, (1 + r / 100).cumprod(), label=f'{method} {window}, +/-{cutoff:.2f}')
        ax.set_yscale('log')
        ax.set(title=strategy, ylabel='Wealth from 1 unit', xlabel='Event date')
        ax.axvline(raw_events.loc[raw_events.split.eq('Test')].index.min(), color='gray', ls=':')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    _capture(fig, figures)

    # Plot family from opening notebook cell 5
    vol_scaling_case = smoothing_results['EWMA', 5, 0.02]
    vol_scaling_events = vol_scaling_case['events'].sort_index()
    vol_scaling_audit = vol_scaling_case['filtered'].sort_index().copy()
    records = []
    for date, row in vol_scaling_audit.iterrows():
        prior = vol_scaling_events.loc[vol_scaling_events.index < date]
        target = prior.loc[prior.regime.eq('Above mean') & prior.trend.ne('Flat')].tail(10)
        comparison = prior.loc[prior.regime.eq(row.regime)].tail(10)
        assert (target.index < date).all() and (comparison.index < date).all()
        target_sd = target.return_pct.std(ddof=1) if len(target) == 10 else np.nan
        comparison_sd = comparison.return_pct.std(ddof=1) if len(comparison) == 10 else np.nan
        ready = np.isfinite(target_sd) and np.isfinite(comparison_sd)
        multiplier = min(1.0, target_sd / comparison_sd) if ready and comparison_sd > 1e-12 else 1.0
        records.append(dict(date=date, target_vol=target_sd, comparison_vol=comparison_sd, ready=ready, reconstructed_multiplier=multiplier))
    vol_scaling_audit = vol_scaling_audit.join(pd.DataFrame(records).set_index('date'))
    np.testing.assert_allclose(vol_scaling_audit.reconstructed_multiplier, vol_scaling_audit.vol_multiplier)
    assert vol_scaling_audit.trend.ne('Flat').all()
    vol_scaling_audit['return_change_pp'] = vol_scaling_audit['Volatility only'] - vol_scaling_audit.return_pct
    vol_scaling_audit['loss_avoided_pp'] = vol_scaling_audit.return_change_pp.clip(lower=0)
    vol_scaling_audit['profit_forgone_pp'] = -vol_scaling_audit.return_change_pp.clip(upper=0)
    vol_scaling_audit['log_growth_change'] = np.log1p(vol_scaling_audit['Volatility only'] / 100) - np.log1p(vol_scaling_audit.return_pct / 100)
    vol_scaling_audit['reduced'] = vol_scaling_audit.vol_multiplier.lt(1 - 1e-12)
    v = vol_scaling_audit
    test_marker = vol_scaling_events.index[vol_scaling_events.split.eq('Test')].min()
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True)
    for ax, column, label in zip(axes, ['Volatility only', 'Prior-Brier reductions', 'Combined'], ['Volatility only', 'Prior-Brier reductions only', 'Combined: prior-Brier reductions + volatility']):
        effect = v[column] - v.return_pct
        assert np.isfinite(effect).all()
        ax.bar(v.index, effect, width=4, color=np.where(effect >= 0, 'tab:green', 'tab:red'))
        ax.axhline(0, color='black', lw=0.7)
        ax.set(title=label, ylabel='Scaled minus raw return (pp)')
        ax.axvline(test_marker, color='gray', ls=':')
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('Event date')
    fig.suptitle('Effect per event: positive = loss avoided; negative = profit forgone\nEWMA-5, +/-0.02 | Non-flat events', fontsize=14)
    _capture(fig, figures)
    vol_scaling_group_effects = v.groupby(['regime', 'trend'], sort=True).agg(events=('return_pct', 'size'), reduced_events=('reduced', 'sum'), mean_multiplier=('vol_multiplier', 'mean'), losses_avoided_pp=('loss_avoided_pp', 'sum'), profits_forgone_pp=('profit_forgone_pp', 'sum'), log_wealth_effect=('log_growth_change', 'sum'))
    boxplot_events = vol_scaling_events.copy()
    groups = [(regime, trend) for trend in ['Flat', 'Uptrend', 'Downtrend'] for regime in ['Above mean', 'At/below mean']]
    accuracy_rows = []
    distribution_rows = []
    for row_idx, (regime, trend) in enumerate(groups):
        if row_idx % 2 == 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='col', constrained_layout=True)
        group = boxplot_events.loc[boxplot_events.regime.eq(regime) & boxplot_events.trend.eq(trend)]
        wins = int(group.return_pct.gt(0).sum())
        losses = int(group.return_pct.lt(0).sum())
        zeros = int(group.return_pct.eq(0).sum())
        accuracy = 100 * wins / len(group)
        accuracy_rows.append(dict(regime=regime, trend=trend, events=len(group), winning_events=wins, losing_events=losses, zero_return_events=zeros, event_accuracy_pct=accuracy))
        for col_idx, (outcome, mask) in enumerate([('Losing', group.return_pct.lt(0)), ('Winning', group.return_pct.gt(0))]):
            part = group.loc[mask]
            ax = axes[row_idx % 2, col_idx]
            ax.set(title=f'{regime} | {trend} | {outcome}: n={len(part)}\nGroup event accuracy: {wins}/{len(group)} = {accuracy:.1f}%', ylabel='Raw event return (%)', xticks=[1], xticklabels=['Unscaled returns'])
            ax.axhline(0, color='black', lw=0.7)
            ax.grid(axis='y', alpha=0.2)
            if part.empty:
                ax.text(0.5, 0.5, 'No events', transform=ax.transAxes, ha='center')
                continue
            box = ax.boxplot([part.return_pct.to_numpy()], positions=[1], widths=0.4, patch_artist=True, showfliers=False)
            box['boxes'][0].set_facecolor('mistyrose' if outcome == 'Losing' else 'honeydew')
            jitter = np.random.default_rng(42 + row_idx * 2 + col_idx).uniform(-0.1, 0.1, len(part))
            ax.scatter(1 + jitter, part.return_pct, s=20, color='tab:red' if outcome == 'Losing' else 'tab:green', zorder=3, alpha=0.7)
            ax.set_xticks([1], ['Unscaled returns'])
            ax.set_xlim(0.6, 1.4)
            distribution_rows.append(dict(regime=regime, trend=trend, outcome=outcome, events=len(part), raw_mean_pct=part.return_pct.mean(), raw_median_pct=part.return_pct.median(), raw_p10_pct=part.return_pct.quantile(0.1), raw_p90_pct=part.return_pct.quantile(0.9)))
        if row_idx % 2 == 1:
            fig.suptitle(f'{trend} event returns: above and at/below mean\nEWMA-5, +/-0.02 | Validation + test | Unscaled returns', fontsize=14)
            _capture(fig, figures)
    vol_scaling_distributions = pd.DataFrame(distribution_rows)
    raw_filtered_event_accuracy = pd.DataFrame(accuracy_rows)
    tradeoff_rows = []
    for label, part in [('All retained events', v), *[(f'{a} / {b}', v.loc[v.regime.eq(a) & v.trend.eq(b)]) for a, b in vol_scaling_group_effects.index]]:
        raw = metrics(part.return_pct)
        scaled = metrics(part['Volatility only'])
        tradeoff_rows.append(dict(group=label, events=len(part), raw_geometric_growth_pct=raw['geometric_growth_pct'], scaled_geometric_growth_pct=scaled['geometric_growth_pct'], raw_mdd_pct=raw['max_drawdown_pct'], scaled_mdd_pct=scaled['max_drawdown_pct'], raw_worst_pct=raw['worst_event_pct'], scaled_worst_pct=scaled['worst_event_pct'], losses_avoided_pp=part.loss_avoided_pp.sum(), profits_forgone_pp=part.profit_forgone_pp.sum(), log_wealth_effect=part.log_growth_change.sum()))
    vol_scaling_tradeoff = pd.DataFrame(tradeoff_rows)
    assert sum(vol_scaling_distributions.events) + boxplot_events.return_pct.eq(0).sum() == len(boxplot_events)
    assert raw_filtered_event_accuracy.events.sum() == len(boxplot_events)

    # Plot family from opening notebook cell 8
    import matplotlib.dates as mdates
    copied_case = smoothing_results['EWMA', 5, 0.02]
    copied_events = copied_case['events']
    copied_mean_summary = pd.DataFrame([dict(regime=regime, events=len(part), mean_return_pct=part.return_pct.mean(), std_return_pp=part.return_pct.std(ddof=1), geometric_return_per_event_pct=100 * np.expm1(np.log1p(part.return_pct / 100).mean())) for regime, part in copied_events.groupby('regime')]).set_index('regime').reindex(['Above mean', 'At/below mean'])
    assert copied_mean_summary.events.sum() == len(copied_events)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, col, title, ylabel in zip(axes, ['mean_return_pct', 'std_return_pp', 'geometric_return_per_event_pct'], ['Mean event return', 'Event return volatility', 'Geometric growth per event'], ['Return (%)', 'Standard deviation (pp)', 'Geometric return (%)']):
        values = copied_mean_summary[col]
        bars = ax.bar(values.index, values, color=['tab:green', 'tab:orange'])
        ax.bar_label(bars, fmt='%.2f')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set(title=title, ylabel=ylabel)
        ax.margins(y=0.2)
    fig.suptitle('Validation + test: above versus at/below historical mean (flat events included)')
    _capture(fig, figures)
    vegas_history = pd.read_csv(PREDICTIONS / 'trainval/open/inference/train_with_open.csv')
    vegas_history['date'] = pd.to_datetime(vegas_history.date)
    order = ['date'] + (['bout_position'] if 'bout_position' in vegas_history else []) + ['fighter_red', 'fighter_blue']
    vegas_history = score_fights(vegas_history.sort_values(order, kind='stable').reset_index(drop=True), 'open')
    for column in ['vegas_actual', 'vegas_expected', 'excess']:
        vegas_history['rolling_' + column] = vegas_history[column].rolling(100, min_periods=100).mean()
    np.testing.assert_allclose(vegas_history.rolling_excess, vegas_history.rolling_vegas_actual - vegas_history.rolling_vegas_expected, atol=1e-14, equal_nan=True)
    train_end = pd.to_datetime(pd.read_csv(PREDICTIONS / 'train/open/inference/train_with_open.csv').date).max()
    validation_start = copied_events.index[copied_events.split.eq('Validation')].min()
    test_start = copied_events.index[copied_events.split.eq('Test')].min()
    training_mean = copied_case['histories']['Validation'].loc[copied_case['histories']['Validation'].date.le(train_end), 'rolling_advantage'].mean()
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for stage, color in [('Train', 'tab:blue'), ('Validation', 'tab:orange'), ('Test', 'tab:green')]:
        history = copied_case['histories']['Test' if stage == 'Test' else 'Validation']
        endpoints = history.groupby('date').rolling_advantage.last()
        points = endpoints.loc[endpoints.index <= train_end] if stage == 'Train' else endpoints.reindex(copied_events.index[copied_events.split.eq(stage)])
        axes[0].plot(points.index, points, color=color, label=stage, lw=1.4)
    axes[0].hlines(training_mean, validation_start, copied_events.index.max(), colors='black', linestyles='--', label='Training mean')
    axes[0].axhline(0, color='gray', lw=0.6)
    axes[0].set(title='100-fight rolling Brier advantage: train, validation, and test', ylabel='Vegas Brier minus model Brier')
    axes[1].plot(vegas_history.date, vegas_history.rolling_vegas_actual, color='tab:orange', label='Actual Vegas Brier')
    axes[1].plot(vegas_history.date, vegas_history.rolling_vegas_expected, color='tab:blue', ls='--', label='Expected Brier if probabilities are correct: p(1-p)')
    axes[1].set(title='Open Vegas: train + validation actual versus expected Brier (100 fights)', ylabel='Rolling Brier score')
    axes[2].plot(vegas_history.date, vegas_history.rolling_excess, color='tab:purple', label='Actual minus expected Brier')
    axes[2].fill_between(vegas_history.date, 0, vegas_history.rolling_excess, where=vegas_history.rolling_excess.ge(0), color='tab:red', alpha=0.15)
    axes[2].fill_between(vegas_history.date, 0, vegas_history.rolling_excess, where=vegas_history.rolling_excess.lt(0), color='tab:green', alpha=0.15)
    axes[2].axhline(0, color='gray', ls='--', lw=1)
    axes[2].set(title='Open Vegas excess Brier: train + validation (100 fights)', ylabel='Actual minus expected', xlabel='Date')
    for ax in axes:
        ax.axvline(validation_start, color='tab:orange', ls='--', lw=0.8, label='Validation starts')
        ax.axvline(test_start, color='tab:green', ls=':', lw=0.8, label='Test starts / model refit')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _capture(fig, figures)

    # Plot family from opening notebook cell 10
    from scipy.stats import pearsonr, spearmanr
    
    def event_only_pearson_ci(x, y):
        rng = np.random.default_rng(42)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        count = len(x)
        block = min(5, count)
        draws = 2000
        starts = rng.integers(0, count - block + 1, size=(draws, int(np.ceil(count / block))))
        ids = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :count]
        a = x[ids]
        b = y[ids]
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        cor = np.divide((a * b).sum(axis=1), denominator, out=np.full(draws, np.nan), where=denominator > 0)
        return np.quantile(cor[np.isfinite(cor)], [0.025, 0.975])
    correlation_case = smoothing_results['EWMA', 5, 0.02]
    correlation_history = correlation_case['histories']['Validation']
    correlation_returns = correlation_case['events'].loc[lambda x: x.split.eq('Validation'), 'return_pct']
    event_scores = correlation_history.groupby('date')[['vegas_actual', 'advantage']].mean()
    event_score_dates = pd.Series(event_scores.index, index=event_scores.index)
    assert (event_score_dates.shift(1).reindex(correlation_returns.index) < correlation_returns.index).all()
    correlation_specs = [('Prior actual Vegas Brier: 0-fight window', event_scores.vegas_actual.shift(1), 'Previous event actual Vegas Brier'), ('Prior-event advantage: event only', event_scores.advantage.shift(1), 'Previous event Vegas Brier minus model Brier'), ('Same-event advantage: event only', event_scores.advantage, 'Current event Vegas Brier minus model Brier')]
    event_only_correlation_pairs = {}
    event_only_correlation_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)
    for ax, (label, predictor, xlabel) in zip(axes, correlation_specs):
        sample = pd.DataFrame({'brier': predictor.reindex(correlation_returns.index), 'return_pct': correlation_returns}).dropna()
        assert len(sample) == len(correlation_returns)
        event_only_correlation_pairs[label] = sample
        r = pearsonr(sample.brier, sample.return_pct).statistic
        rho = spearmanr(sample.brier, sample.return_pct).statistic
        lo, hi = event_only_pearson_ci(sample.brier, sample.return_pct)
        event_only_correlation_rows.append(dict(comparison=label, events=len(sample), pearson_r=r, ci_low=lo, ci_high=hi, spearman_r=rho))
        ax.scatter(sample.brier, sample.return_pct, alpha=0.65, s=25)
        slope, intercept = np.polyfit(sample.brier, sample.return_pct, 1)
        grid = np.linspace(sample.brier.min(), sample.brier.max(), 100)
        ax.plot(grid, intercept + slope * grid, color='tab:red')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set(title=f'{label}\nr={r:.3f}; 95% CI [{lo:.3f}, {hi:.3f}]\nSpearman={rho:.3f}; n={len(sample)}', xlabel=xlabel, ylabel='Current event bankroll return (%)')
        ax.grid(alpha=0.2)
    fig.suptitle('Opening inference: validation event returns, unscaled and including flat events')
    _capture(fig, figures)
    event_only_correlations = pd.DataFrame(event_only_correlation_rows)

    # Plot family from opening notebook cell 13
    bankroll_case = smoothing_results['EWMA', 5, 0.02]
    bankroll_all = bankroll_case['events'].sort_index()
    bankroll_active = bankroll_case['filtered'].sort_index()
    bankroll_variants = {'Raw unfiltered': bankroll_all.return_pct, 'Remove flat only': bankroll_active.return_pct.reindex(bankroll_all.index, fill_value=0), 'Volatility only': bankroll_active['Volatility only'].reindex(bankroll_all.index, fill_value=0), 'Prior-Brier only': bankroll_active['Prior-Brier reductions'].reindex(bankroll_all.index, fill_value=0), 'Combined': bankroll_active['Combined'].reindex(bankroll_all.index, fill_value=0)}
    bankroll_paths = {}
    bankroll_summary_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    colors = ['gray', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:green']
    for ax, period in zip(axes, ['Validation', 'Test']):
        dates = bankroll_all.index[bankroll_all.split.eq(period)]
        start_date = dates.min() - pd.Timedelta(days=1)
        for (label, series), color in zip(bankroll_variants.items(), colors):
            returns = series.reindex(dates)
            assert returns.notna().all() and returns.gt(-100).all()
            if label != 'Raw unfiltered':
                assert returns.loc[~bankroll_all.loc[dates, 'opening_keep']].eq(0).all()
            path = pd.Series(np.r_[500.0, 500.0 * (1 + returns / 100).cumprod().to_numpy()], index=pd.DatetimeIndex([start_date, *dates]))
            bankroll_paths[period, label] = path
            assert path.iloc[0] == 500.0
            ax.plot(path.index, path, color=color, label=f'{label}: {path.iloc[-1]:,.0f}', lw=1.6)
            bankroll_summary_rows.append(dict(period=period, strategy=label, calendar_events=len(dates), retained_events=len(dates) if label == 'Raw unfiltered' else int(bankroll_all.loc[dates, 'opening_keep'].sum()), starting_bankroll=500.0, ending_bankroll=path.iloc[-1], net_profit=path.iloc[-1] - 500.0, cumulative_return_pct=100 * (path.iloc[-1] / 500.0 - 1)))
        ax.axhline(500, color='black', ls=':', lw=0.8)
        ax.set(title=f'{period}: independent starting bankroll of 500', ylabel='Bankroll', xlabel='Event date')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle('Compounded event returns: EWMA-5, +/-0.02', fontsize=14)
    _capture(fig, figures)
    bankroll_period_summary = pd.DataFrame(bankroll_summary_rows)
    return dict(figures=figures, metrics=smoothing_metrics, bankroll=bankroll_period_summary, accuracy=raw_filtered_event_accuracy, correlations=event_only_correlations)


def closing_report(market):
    if market not in {'close1','close2'}:
        raise ValueError('Expected close1 or close2')
    with PLOT_LOCK:
        return _closing_report(market)


def _closing_report(market):
    figures=[]
    sources={'open':load_market('open'),market:load_market(market)}
    rows=[]
    target=sources[market]['events']
    for gate in ['open',market]:
        for scaler in ['open',market]:
            pd.testing.assert_index_equal(target.index,sources[gate]['events'].index)
            pd.testing.assert_index_equal(target.index,sources[scaler]['all_scalers'].index)
            keep=sources[gate]['events'].opening_keep
            returns=target.return_pct*sources[scaler]['all_scalers'].combined_multiplier
            rows.append(dict(gate=gate,scalers=scaler,**metrics(returns.loc[keep])))
    crossed=pd.DataFrame(rows)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,col,title in zip(axes,['geometric_growth_pct','max_drawdown_pct','average_return_pct'],
            ['Geometric growth/event (%)','Maximum drawdown (%)','Average return (%)']):
        bars=ax.bar(np.arange(4),crossed[col]);ax.bar_label(bars,fmt='%.2f')
        ax.set_xticks(range(4),[f'Gate: {r.gate}\nScalers: {r.scalers}' for r in crossed.itertuples()],fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
    fig.suptitle(f'{market}: combined strategy, crossed sources | EWMA-5 +/-0.02')
    _capture(fig,figures)

    # All five curves below use this closing market's own history, never opening filters.
    own=sources[market];active=own['filtered']
    variants={'Raw unfiltered':target.return_pct,'Remove flat only':active.return_pct,
              'Volatility only':active['Volatility only'],'Prior-Brier only':active['Prior-Brier reductions'],
              'Combined':active['Combined']}
    bankroll_rows=[]
    fig,axes=plt.subplots(2,1,figsize=(15,10),constrained_layout=True)
    for ax,period in zip(axes,['Validation','Test']):
        dates=target.index[target.split.eq(period)]
        for label,series in variants.items():
            returns=series.reindex(dates,fill_value=0)
            path=pd.Series(np.r_[500.,500*(1+returns/100).cumprod()],index=[dates.min()-pd.Timedelta(days=1),*dates])
            ax.plot(path.index,path,label=f'{label}: {path.iloc[-1]:,.0f}')
            bankroll_rows.append(dict(period=period,strategy=label,ending_bankroll=path.iloc[-1]))
        ax.set(title=f'{period}: closing history, initial bankroll 500',ylabel='Bankroll',xlabel='Event date')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle(f'{market}: EWMA-5 +/-0.02 | Closing history filters and scalers')
    _capture(fig,figures)

    train_end=pd.to_datetime(pd.read_csv(PREDICTIONS/'train/open/inference/train_with_open.csv').date).max()
    fig,ax=plt.subplots(figsize=(15,5),constrained_layout=True)
    for stage,color in [('Train','tab:blue'),('Validation','tab:orange'),('Test','tab:green')]:
        history=own['histories']['Test' if stage=='Test' else 'Validation']
        ends=history.groupby('date').rolling_advantage.last()
        points=ends.loc[ends.index<=train_end] if stage=='Train' else ends.reindex(target.index[target.split.eq(stage)])
        ax.plot(points.index,points,color=color,label=stage)
    ax.axvline(target.index[target.split.eq('Validation')].min(),color='gray',ls='--',label='Validation starts')
    ax.axvline(target.index[target.split.eq('Test')].min(),color='black',ls=':',label='Test / model refit')
    ax.axhline(0,color='gray',lw=.7)
    ax.set(title=f'{market}: 100-fight rolling Brier advantage — closing history',ylabel='Vegas Brier minus model Brier',xlabel='Event date')
    ax.legend();ax.grid(alpha=.2)
    _capture(fig,figures)
    return dict(figures=figures,crossed=crossed,bankroll=pd.DataFrame(bankroll_rows))
