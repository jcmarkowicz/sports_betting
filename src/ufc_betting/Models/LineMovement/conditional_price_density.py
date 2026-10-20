"""Conditional PDFs fitted to actual archived odds at an exact day before events."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import ndtr
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional
from ..RiskManagement.lead_time_backtest import _name


def exact_day_observations(data_dir, days_out, odds_type, split='test', *, snapshot_dir=None):
    """Match latest exact UTC-day archive quotes to saved final test/val quotes."""
    if isinstance(days_out, bool) or not isinstance(days_out, (int, np.integer)) or days_out < 0:
        raise ValueError('days_out must be a nonnegative integer')
    data_dir=Path(data_dir)
    prefix='test_set_data' if split=='test' else 'val_set_data'
    if split not in ('test','validation') or odds_type not in ('close1','close2'):
        raise ValueError('Use test/validation and close1/close2')
    frame=pd.read_csv(data_dir/'model_results'/f'{prefix}_open_with_{odds_type}.csv')
    endpoints={}
    for row in frame.itertuples():
        for side,other in [('red','blue'),('blue','red')]:
            key=(str(row.date)[:10],_name(getattr(row,f'fighter_{side}')),_name(getattr(row,f'fighter_{other}')))
            if key in endpoints:raise ValueError('Duplicate endpoint identity')
            endpoints[key]=(getattr(row,f'dec_open_{side}'),getattr(row,f'dec_{odds_type}_{side}'))
    source=Path(snapshot_dir) if snapshot_dir is not None else data_dir/'intermediate_odds_stats'
    m=pd.read_csv(source/'manifest.csv')
    m['timestamp']=pd.to_datetime(m.commit_datetime_utc,utc=True)
    m['days_out']=(pd.to_datetime(m.event_date,utc=True)-m.timestamp.dt.floor('D')).dt.days
    selected=m.loc[m.days_out.eq(days_out)&m.event_date.isin(frame.date)]
    selected=selected.sort_values(['timestamp','commit','original_path']).drop_duplicates('event_date',keep='last')
    rows=[]
    for record in selected.itertuples():
        for row in pd.read_csv(source/record.filename).itertuples():
            for side,other in [('red','blue'),('blue','red')]:
                key=(record.event_date,_name(getattr(row,f'fighter_{side}')),_name(getattr(row,f'fighter_{other}')))
                if key not in endpoints:continue
                opening,closing=endpoints[key];price=getattr(row,f'dec_{odds_type}_{side}')
                if not np.isfinite([opening,closing,price]).all() or min(opening,closing,price)<=1:continue
                rows.append(dict(date=key[0],fighter=key[1],opponent=key[2],opening=opening,
                                 closing=closing,price=price,source_file=record.filename))
    data=pd.DataFrame(rows)
    if data.empty:raise ValueError(f'No {split} snapshots available exactly {days_out} days out.')
    if data.duplicated(['date','fighter','opponent']).any():raise ValueError('Duplicate daily observation')
    return data


def fit_day_price_density(data_dir, days_out, odds_type='close1', split='test', *,
                          bw='normal_reference', snapshot_dir=None):
    """Fit historical intermediate odds conditioned on opening and closing odds.

    Returns a dictionary containing the Statsmodels model and its observations.
    bw is passed to Statsmodels: normal_reference, cv_ml, cv_ls, or three positive
    bandwidths ordered intermediate, opening, movement. No prices are sampled.
    """
    data = exact_day_observations(data_dir, days_out, odds_type, split, snapshot_dir=snapshot_dir)
    fitted = fit_price_density(data, days_out=days_out, odds_type=odds_type, bw=bw)
    fitted['split'] = split  # Compatibility for archive-based callers.
    return fitted


def fit_price_density(data, *, days_out, odds_type='close1', bw='normal_reference'):
    """Fit p(price | opening, closing) using only the supplied DataFrame.

    Each row is an observed intermediate quote with opening, closing, and price
    columns in decimal odds. No file loading, cohort selection, saving or plotting.
    days_out and odds_type label the supplied observations; they do not filter rows.
    Returns the fitted estimator under 'model', observations, and metadata.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    missing = {'opening', 'closing', 'price'} - set(data.columns)
    if missing:
        raise ValueError(f'Missing density training columns: {sorted(missing)}')
    if odds_type not in ('close1', 'close2'):
        raise ValueError('odds_type must be close1 or close2')
    if isinstance(days_out, bool) or not isinstance(days_out, (int, np.integer)) or days_out < 0:
        raise ValueError('days_out must be a nonnegative integer')
    data = data.copy(deep=True)
    values = data[['opening', 'closing', 'price']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 1).any():
        raise ValueError('Training odds must be finite decimal values greater than 1')
    data[['opening', 'closing', 'price']] = values
    x = np.column_stack([np.log(data.opening - 1),
                         np.log((data.closing - 1) / (data.opening - 1))])
    y = np.log(data.price.to_numpy() - 1)
    if len(data) <= 3:
        raise ValueError('Need more than three historical observations for conditional KDE')
    if isinstance(bw, str):
        if bw not in ('normal_reference', 'cv_ml', 'cv_ls'):
            raise ValueError('Unknown Statsmodels bandwidth method')
        if (np.std(np.column_stack([y, x]), axis=0) <= 0).any():
            raise ValueError('Constant transformed prices require explicit positive bandwidths')
    else:
        bw = np.asarray(bw, float)
        if bw.shape != (3,) or not np.isfinite(bw).all() or (bw <= 0).any():
            raise ValueError('Supply three finite positive bandwidths')
    model = KDEMultivariateConditional(endog=y, exog=x, dep_type='c', indep_type='cc', bw=bw)
    if not np.isfinite(model.bw).all() or (model.bw <= 0).any():
        raise ValueError('Statsmodels produced invalid bandwidths')
    return dict(model=model, observations=data, days_out=int(days_out), odds_type=odds_type,
                bandwidth_method=bw if isinstance(bw, str) else 'supplied',
                bandwidths=np.asarray(model.bw).tolist(),
                transform='log(price - 1); predictors log(opening - 1), log((closing - 1)/(opening - 1))')


def _query(opening_odds, closing_odds):
    if not np.isfinite([opening_odds, closing_odds]).all() or min(opening_odds, closing_odds) <= 1:
        raise ValueError('Opening and closing decimal odds must exceed 1')
    return np.array([np.log(opening_odds - 1), np.log((closing_odds - 1) / (opening_odds - 1))])


def price_pdf(model, prices, opening_odds, closing_odds):
    """Evaluate the fitted conditional PDF in decimal-odds units."""
    query = _query(opening_odds, closing_odds)
    prices = np.atleast_1d(np.asarray(prices, float))
    if prices.ndim != 1 or not np.isfinite(prices).all():
        raise ValueError('prices must be a finite one-dimensional grid')
    result = np.zeros_like(prices)
    valid = prices > 1
    if valid.any():
        values = model.pdf(endog_predict=np.log(prices[valid] - 1),
                           exog_predict=np.tile(query, (valid.sum(), 1)))
        # Change of variables from log(price - 1) back to decimal odds.
        result[valid] = np.asarray(values).reshape(-1) / (prices[valid] - 1)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Conditional density is undefined at these endpoints; check historical coverage')
    return result


def price_probability(model, lower, upper, opening_odds, closing_odds):
    """Conditional probability of a decimal-price interval using the fitted CDF."""
    query = _query(opening_odds, closing_odds)
    if np.isnan(lower) or np.isnan(upper) or lower > upper:
        raise ValueError('Interval bounds must be ordered and not NaN')
    def cdf(price):
        if price <= 1:
            return 0.
        if np.isposinf(price):
            return 1.
        value = float(np.asarray(model.cdf(endog_predict=[np.log(price - 1)],
                                           exog_predict=query[None, :])).item())
        if not np.isfinite(value):
            raise ValueError('Conditional CDF is undefined at these endpoints')
        return value
    return float(np.clip(cdf(upper) - cdf(lower), 0, 1))


def sample_prices(model, opening_odds, closing_odds, n_samples=1000, *,
                  random_state=None, grid_size=4096):
    """Return a 1D array of intermediate decimal odds from the fitted day model.

    Uses inverse-CDF sampling: direct quantile solving for small draws, and
    interpolation on a log-price grid for large draws. Prices can lie outside
    the opening/closing interval. random_state
    accepts a seed or NumPy Generator.

    prices = sample_prices(fitted['model'], 2.0, 1.6, 5000, random_state=42)
    """
    query = _query(opening_odds, closing_odds)
    for name, value, minimum in [('n_samples', n_samples, 0), ('grid_size', grid_size, 2)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if model.dep_type != 'c' or model.indep_type != 'cc':
        raise ValueError('Use the continuous conditional model from fit_day_price_density')
    if n_samples == 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(random_state)
    uniforms = np.maximum(rng.random(n_samples), np.nextafter(0., 1.))
    observed = np.asarray(model.endog, float).reshape(-1)
    bandwidth = float(model.bw[0])
    if not np.isfinite(observed).all() or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Model must have finite observations and a positive bandwidth')

    # For the continuous Gaussian product KDE, conditioning gives a mixture
    # of the observed dependent-variable kernels. Compute the endpoint weights
    # once, instead of recomputing them at every CDF evaluation in Statsmodels.
    predictors = np.asarray(model.exog, float)
    predictor_bw = np.asarray(model.bw[1:], float)
    if (predictors.shape != (len(observed), 2) or not np.isfinite(predictors).all()
            or not np.isfinite(predictor_bw).all() or (predictor_bw <= 0).any()):
        raise ValueError('Model must have finite predictors and positive bandwidths')
    log_weights = -.5 * np.sum(((predictors - query) / predictor_bw) ** 2, axis=1)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    def conditional_cdf(points):
        points = np.atleast_1d(points)
        values = np.empty(len(points))
        # Bound memory for grid sampling; a single quantile needs only one row.
        for start in range(0, len(points), 256):
            z = (points[start:start + 256, None] - observed) / bandwidth
            values[start:start + 256] = np.sum(ndtr(z) * weights, axis=1)
        return np.clip(values, 0., 1.)

    # Extend beyond historical values to include the KDE tails, not just the
    # observed price range. Ensure the bounds cover every requested quantile.
    margin = 8 * bandwidth
    for _ in range(10):
        low, high = observed.min() - margin, observed.max() + margin
        bounds = conditional_cdf(np.array([low, high]))
        if bounds[0] <= uniforms.min() and bounds[1] >= uniforms.max():
            break
        margin *= 2
    else:
        raise ValueError('Could not bracket the requested conditional quantiles')
    if n_samples <= 16:
        log_prices = np.array([brentq(lambda point: conditional_cdf(point)[0] - u,
                                     low, high, xtol=1e-12) for u in uniforms])
    else:
        grid = np.linspace(low, high, grid_size)
        cdf = np.maximum.accumulate(conditional_cdf(grid))
        # Flat tails contain no probability mass and cannot be inverted uniquely.
        probabilities, indices = np.unique(cdf, return_index=True)
        log_prices = np.interp(uniforms, probabilities, grid[indices])
    with np.errstate(over='ignore'):
        prices = 1 + np.exp(log_prices)
    if not np.isfinite(prices).all() or (prices <= 1).any():
        raise ValueError('Sampled prices cannot be represented as finite decimal odds above 1')
    return prices


def plot_price_density(model, opening_odds, closing_odds, *, prices=None, ax=None):
    """Display a conditional PDF from an already-fitted KDE; never fit or save.

    model is the estimator returned in fit_day_price_density(...)["model"].
    The opening/closing pair only conditions this plot, not the training data.
    Returns (figure, axes). Supply ax to draw without calling plt.show().
    """
    query = _query(opening_odds, closing_odds)
    if prices is None:
        observed = np.asarray(model.endog, float).reshape(-1)
        low = min(observed.min() - 4 * model.bw[0], np.log(min(opening_odds, closing_odds) - 1))
        high = max(observed.max() + 4 * model.bw[0], np.log(max(opening_odds, closing_odds) - 1))
        prices = 1 + np.exp(np.linspace(low, high, 700))
    pdf = price_pdf(model, prices, opening_odds, closing_odds)
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    ax.plot(prices, pdf, color='#3579ad')
    ax.fill_between(prices, pdf, alpha=.15, color='#3579ad')
    ax.axvline(opening_odds, color='#228765', linestyle='--', label='Opening')
    ax.axvline(closing_odds, color='#df8332', linestyle=':', label='Closing')
    observed_inputs = np.asarray(model.exog)
    if ((query < observed_inputs.min(axis=0)) | (query > observed_inputs.max(axis=0))).any():
        ax.text(.02, .97, 'Query outside observed predictor range',
                transform=ax.transAxes, va='top', color='#a33')
    ax.set(title=f'Conditional PDF | opening {opening_odds:.2f}, closing {closing_odds:.2f}',
           xlabel='Intermediate decimal odds', ylabel='Probability density')
    ax.legend()
    ax.grid(alpha=.2)
    if show:
        plt.show()
    return ax.figure, ax


def plot_day_price_density(data_dir, days_out=5, opening_odds=2., closing_odds=1.6,
                           split='test', output_dir=None, *, bw='normal_reference', snapshot_dir=None):
    """Plot conditional PDFs from historical observations; never generate prices."""
    query = _query(opening_odds, closing_odds)
    out = Path(output_dir) if output_dir else Path(data_dir) / 'plot_pngs' / 'conditional_day_price_density'
    out.mkdir(parents=True, exist_ok=True)
    results = {stage: fit_day_price_density(data_dir, days_out, stage, split, bw=bw, snapshot_dir=snapshot_dir)
               for stage in ('close1', 'close2')}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    rows = []
    try:
        for ax, (stage, fitted) in zip(axes, results.items()):
            model, observations = fitted['model'], fitted['observations']
            # Deterministic grid spans observed log prices plus four kernel widths.
            y = np.log(observations.price.to_numpy() - 1)
            low = min(y.min() - 4 * model.bw[0], np.log(min(opening_odds, closing_odds) - 1))
            high = max(y.max() + 4 * model.bw[0], np.log(max(opening_odds, closing_odds) - 1))
            grid = 1 + np.exp(np.linspace(low, high, 700))
            pdf = price_pdf(model, grid, opening_odds, closing_odds)
            ax.plot(grid, pdf, color='#3579ad')
            ax.fill_between(grid, pdf, alpha=.15, color='#3579ad')
            ax.set(title=f'{stage} | {len(observations)} observations / {observations.date.nunique()} events',
                   xlabel='Intermediate decimal odds', ylabel='Conditional density per decimal-price unit')
            x = np.column_stack([np.log(observations.opening - 1),
                                 np.log((observations.closing - 1) / (observations.opening - 1))])
            outside = bool(((query < x.min(axis=0)) | (query > x.max(axis=0))).any())
            if outside:
                ax.text(.02, .97, 'Query outside observed predictor range', transform=ax.transAxes, va='top', color='#a33')
            ax.axvline(opening_odds, color='#228765', linestyle='--', label='Opening')
            ax.axvline(closing_odds, color='#df8332', linestyle=':', label='Final closing')
            ax.legend(); ax.grid(alpha=.2)
            rows.append(dict(odds_type=stage, days_out=days_out, opening=opening_odds, closing=closing_odds,
                             observations=len(observations), outside_observed_range=outside,
                             probability_below_open=price_probability(model, 1, opening_odds, opening_odds, closing_odds),
                             probability_between_endpoints=price_probability(model, min(opening_odds, closing_odds),
                                                                             max(opening_odds, closing_odds), opening_odds, closing_odds)))
            observations.to_csv(out / f'{stage}_{days_out}_day_observations.csv', index=False)
        fig.suptitle(f'{days_out} days before event | opening {opening_odds:.2f}, final close {closing_odds:.2f}')
        fig.supxlabel(f'Historical {split} observations, exact UTC day. Statsmodels conditional KDE; '
                      f'bandwidth: {bw}. No random price sampling.', fontsize=9)
        path = out / f'conditional_price_{days_out}_days.png'
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f'conditional_price_{days_out}_days_summary.csv', index=False)
    return dict(models=results, summary=summary, plot_path=path)
