"""Validation-only joint and conditional price-movement kernel densities."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from scipy.special import logit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def fit_validation_odds_density(data_dir, bw_method='scott'):
    """Fit f(opening, magnitude | direction) with physical price boundaries.

    Transform opening x to log(x-1). For increases transform magnitude m to
    log(m); for decreases use logit(m/(x-1)), enforcing closing odds > 1.
    Both fighter sides are observations, not independent fight outcomes.
    """
    models, observations, counts = {}, [], []
    keys = ['date', 'fighter_red', 'fighter_blue']
    reference_keys = None
    for stage in ('close1', 'close2'):
        path = Path(data_dir) / 'model_results' / f'val_set_data_open_with_{stage}.csv'
        frame = pd.read_csv(path).sort_values(keys).reset_index(drop=True)
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate validation fights')
        if reference_keys is not None and not frame[keys].equals(reference_keys):
            raise ValueError('Validation cohorts differ between close1 and close2')
        reference_keys = frame[keys]
        parts = []
        for side in ('red', 'blue'):
            parts.append(pd.DataFrame(dict(date=frame.date, fighter=frame[f'fighter_{side}'],
                opening=pd.to_numeric(frame[f'dec_open_{side}'],errors='coerce'),
                closing=pd.to_numeric(frame[f'dec_{stage}_{side}'],errors='coerce'))))
        data = pd.concat(parts,ignore_index=True)
        data = data.loc[np.isfinite(data.opening) & np.isfinite(data.closing)
                        & data.opening.gt(1) & data.closing.gt(1)].copy()
        data['change'] = data.closing-data.opening
        data['magnitude'] = data.change.abs()
        data['direction'] = np.where(np.isclose(data.change,0,atol=1e-10,rtol=0), 'unchanged',
                                     np.where(data.change>0,'increase','decrease'))
        data['odds_type'] = stage
        observations.append(data)
        stage_models = {}
        pooled_u = np.log(data.opening.to_numpy()-1)
        pooled_bandwidth = float(np.std(pooled_u,ddof=1)*len(data)**(-1/5))
        for direction in ('increase','decrease','unchanged'):
            sub = data.loc[data.direction.eq(direction)]
            counts.append(dict(odds_type=stage,direction=direction,samples=len(sub),
                               fraction=len(sub)/len(data)))
            u = np.log(sub.opening.to_numpy()-1)
            entry = dict(weight=len(sub)/len(data),u=u)
            if direction != 'unchanged':
                if len(sub)<3:
                    raise ValueError(f'Insufficient {stage} {direction} observations')
                v = (np.log(sub.magnitude.to_numpy()) if direction=='increase'
                     else logit(sub.magnitude.to_numpy()/(sub.opening.to_numpy()-1)))
                entry['kde'] = gaussian_kde(np.vstack([u,v]),bw_method=bw_method)
                entry['bandwidth_u'] = float(np.sqrt(entry['kde'].covariance[0,0]))
            else:
                entry['bandwidth_u'] = pooled_bandwidth
            stage_models[direction] = entry
        models[stage] = stage_models
    return dict(models=models,observations=pd.concat(observations,ignore_index=True),
                counts=pd.DataFrame(counts),bw_method=bw_method)


def _marginal_u(entry, u):
    u = np.atleast_1d(u)
    if not len(entry['u']):
        return np.zeros(len(u))
    h = entry['bandwidth_u']
    return norm.pdf((u[:,None]-entry['u'][None,:])/h).mean(axis=1)/h


def movement_probability(fit, opening_odds, minimum_change, maximum_change,
                         odds_type='close1', direction='decrease'):
    """Probability of direction AND a decimal-magnitude interval given opening.

    Also returns probability conditional on the selected direction and its
    probability given opening. Exact unchanged prices form a separate atom.
    """
    if direction not in ('increase','decrease') or odds_type not in fit['models']:
        raise ValueError('Choose close1/close2 and increase/decrease')
    if not np.isfinite(opening_odds) or opening_odds<=1 or not 0<=minimum_change<maximum_change:
        raise ValueError('Need opening > 1 and 0 <= minimum_change < maximum_change')
    stages = fit['models'][odds_type]
    u = np.log(opening_odds-1)
    densities = {d:e['weight']*_marginal_u(e,u)[0] for d,e in stages.items()}
    denominator = sum(densities.values())
    if denominator<=np.finfo(float).tiny:
        raise ValueError('Opening price is outside numerically supported density range')
    p_direction = densities[direction]/denominator
    entry = stages[direction]; kde = entry['kde']; cov=kde.covariance
    weights = norm.pdf((u-kde.dataset[0])/entry['bandwidth_u'])
    if weights.sum()==0:
        raise ValueError('Opening price is outside supported directional density range')
    weights /= weights.sum()
    means = kde.dataset[1] + cov[1,0]/cov[0,0]*(u-kde.dataset[0])
    sd = np.sqrt(cov[1,1]-cov[1,0]**2/cov[0,0])
    def transform(m):
        if m<=0:return -np.inf
        if direction=='increase':return np.log(m)
        if m>=opening_odds-1:return np.inf
        return logit(m/(opening_odds-1))
    conditional = np.dot(weights,norm.cdf((transform(maximum_change)-means)/sd)
                         -norm.cdf((transform(minimum_change)-means)/sd))
    return dict(direction_probability=float(p_direction),
                interval_probability_given_direction=float(conditional),
                interval_and_direction_probability=float(p_direction*conditional))


def plot_validation_odds_density(data_dir, output_dir=None, bw_method='scott',
                                 plot_quantile=.99, grid_size=150):
    """Create joint PDFs and conditional PDFs, with separate direction panels.

    Plot limits may crop tails; all valid validation observations enter the fit.
    """
    if not 0<plot_quantile<=1 or grid_size<20:
        raise ValueError('Invalid plot limits/grid size')
    fit = fit_validation_odds_density(data_dir,bw_method)
    out = Path(output_dir) if output_dir else Path(data_dir)/'plot_pngs'/'validation_odds_density'
    out.mkdir(parents=True,exist_ok=True)
    data = fit['observations']
    x = np.linspace(max(1.001,data.opening.min()),data.opening.quantile(plot_quantile),grid_size)
    figures, paths = {}, {}
    for kind in ('joint','conditional'):
        fig, axes = plt.subplots(2,2,figsize=(14,10),layout='constrained')
        for i,stage in enumerate(('close1','close2')):
            for j,direction in enumerate(('decrease','increase')):
                entry = fit['models'][stage][direction]
                sub = data.loc[data.odds_type.eq(stage)&data.direction.eq(direction)]
                top = data.loc[data.direction.eq(direction),'magnitude'].quantile(plot_quantile)
                y = np.linspace(max(1e-5,top/(grid_size*10)),top,grid_size)
                xx,yy=np.meshgrid(x,y)
                valid = yy < xx-1 if direction=='decrease' else np.ones(xx.shape,bool)
                u = np.log(xx[valid]-1)
                if direction=='increase':
                    v=np.log(yy[valid]); jac_m=1/yy[valid]
                else:
                    fraction=yy[valid]/(xx[valid]-1)
                    v=logit(fraction); jac_m=1/((xx[valid]-1)*fraction*(1-fraction))
                density=entry['kde'](np.vstack([u,v]))*jac_m
                if kind=='joint':density/=xx[valid]-1
                else:
                    marginal=_marginal_u(entry,u)
                    density=np.divide(density,marginal,out=np.zeros_like(density),where=marginal>0)
                z=np.full(xx.shape,np.nan);z[valid]=density
                ax=axes[i,j]
                peak=float(np.nanmax(z))
                im=ax.pcolormesh(xx,yy,np.ma.masked_invalid(z),shading='auto',cmap='viridis',
                                 norm=LogNorm(vmin=max(peak*1e-4,1e-8),vmax=peak))
                ax.set(title=f'{stage}: {direction} | {len(sub)} fighter prices',
                       xlabel='Opening decimal odds',ylabel=f'Decimal-odds {direction} magnitude')
                if direction=='decrease':
                    line=x[x-1<=top];ax.plot(line,line-1,color='white',linestyle='--',linewidth=1)
                fig.colorbar(im,ax=ax,label=('Joint' if kind=='joint' else 'Conditional')+' density (log color scale)')
        formula='f(opening, change magnitude | direction)' if kind=='joint' else 'f(change magnitude | opening, direction)'
        fig.suptitle(f'Validation prices only: {formula}')
        fig.supxlabel(f'Boundary-aware KDE; all observations fitted. Display limits: {plot_quantile:.0%} quantiles. '
                      'Color is density, not point probability; unchanged prices are separate.',fontsize=9)
        path=out/f'validation_{kind}_odds_density.png'
        fig.savefig(path,dpi=140);plt.close(fig);paths[kind]=path
    fit['counts'].to_csv(out/'direction_counts.csv',index=False)
    fit['observations'].to_csv(out/'validation_price_changes.csv',index=False)
    fit['plot_paths']=paths
    return fit
