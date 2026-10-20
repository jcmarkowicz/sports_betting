"""Observed test-cohort opening-relative price changes over the last week."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ..RiskManagement.lead_time_backtest import _name


def plot_test_price_changes(data_dir, output_dir=None):
    data_dir = Path(data_dir)
    source = data_dir / 'intermediate_odds_stats'
    out = Path(output_dir) if output_dir else data_dir/'plot_pngs'/'test_price_changes'
    out.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(data_dir/'model_results'/'test_set_data_open_with_open.csv')
    keys = ['date','fighter_red','fighter_blue']
    if test.duplicated(keys).any():
        raise ValueError('Duplicate test fights')
    baseline = {}
    for row in test.itertuples():
        date = str(row.date)[:10]
        for side, other in [('red','blue'),('blue','red')]:
            key=(date,_name(getattr(row,f'fighter_{side}')),_name(getattr(row,f'fighter_{other}')))
            if key in baseline:raise ValueError('Ambiguous normalized fighter key')
            baseline[key]=getattr(row,f'dec_open_{side}')
    manifest = pd.read_csv(source/'manifest.csv')
    manifest['timestamp']=pd.to_datetime(manifest.commit_datetime_utc,utc=True)
    manifest['days_out']=(pd.to_datetime(manifest.event_date,utc=True)-manifest.timestamp.dt.floor('D')).dt.days
    selected=manifest.loc[manifest.days_out.between(1,7)&manifest.event_date.isin(test.date)]
    selected=selected.sort_values(['timestamp','commit','original_path']).drop_duplicates(['event_date','days_out'],keep='last')
    rows=[]
    def add(frame, date, day, stages, filename):
        for row in frame.itertuples():
            for side,other in [('red','blue'),('blue','red')]:
                key=(date,_name(getattr(row,f'fighter_{side}')),_name(getattr(row,f'fighter_{other}')))
                if key not in baseline:continue
                opening=baseline[key]
                for stage in stages:
                    price=getattr(row,f'dec_{stage}_{side}')
                    if not np.isfinite([opening,price]).all() or min(opening,price)<=1:continue
                    change=price-opening
                    direction='unchanged' if np.isclose(change,0,atol=1e-10,rtol=0) else ('decrease' if change<0 else 'increase')
                    rows.append(dict(date=date,fighter=key[1],opponent=key[2],days_out=day,
                                     odds_type=stage,opening=opening,price=price,change=change,
                                     direction=direction,source_file=filename))
    for row in selected.itertuples():
        add(pd.read_csv(source/row.filename),row.event_date,row.days_out,('close1','close2'),row.filename)
    for stage in ('close1','close2'):
        path=data_dir/'model_results'/f'test_set_data_open_with_{stage}.csv'
        closing=pd.read_csv(path)
        for date,group in closing.groupby('date'):
            add(group,str(date)[:10],0,(stage,),path.name)
    observations=pd.DataFrame(rows)
    if observations.duplicated(['date','fighter','opponent','days_out','odds_type']).any():
        raise ValueError('Duplicate daily fighter price')
    summary=observations.groupby(['direction','odds_type','days_out']).agg(
        mean_change=('change','mean'),samples=('change','size'),events=('date','nunique')).reset_index()
    fig,axes=plt.subplots(2,2,figsize=(14,7),sharex=True,layout='constrained',
                          gridspec_kw={'height_ratios':[3,1]})
    for col,direction in enumerate(('decrease','increase')):
        ax,count_ax=axes[:,col]
        for stage,color in [('close1','#3579ad'),('close2','#df8332')]:
            s=summary.loc[summary.direction.eq(direction)&summary.odds_type.eq(stage)].set_index('days_out').reindex(range(8))
            points=observations.loc[observations.direction.eq(direction)&observations.odds_type.eq(stage)]
            days=[day for day in range(8) if points.days_out.eq(day).any()]
            if days:
                ax.boxplot([points.loc[points.days_out.eq(day),'change'].to_numpy() for day in days],
                           positions=np.asarray(days)+(.18 if stage=='close1' else -.18),
                           widths=.28,manage_ticks=False,patch_artist=True,whis=1.5,
                           boxprops=dict(facecolor=color,edgecolor=color,alpha=.65),
                           medianprops=dict(color='black',linewidth=1.3),
                           whiskerprops=dict(color=color),capprops=dict(color=color),
                           flierprops=dict(marker='.',markersize=3,markeredgecolor=color,alpha=.4))
            count_ax.plot(s.index,s.samples.fillna(0),marker='.',color=color,label=stage)
        ax.set(title=f'{direction.title()} from opening',ylabel='Decimal price − opening price')
        ax.axhline(0,color='gray',linewidth=.7)
        ax.legend(handles=[Patch(facecolor=color,alpha=.65,label=stage)
                           for stage,color in [('close1','#3579ad'),('close2','#df8332')]])
        ax.grid(alpha=.2)
        count_ax.set(xlabel='Days until event',ylabel='Fighter samples',xticks=range(8),xlim=(7.5,-.5))
        count_ax.grid(alpha=.2)
    fig.suptitle('Test-set fights: price difference from fixed opening odds')
    fig.supxlabel('Days 7–1: archived UTC-day snapshots. Day 0: saved test closing prices. '
                  'Boxes: middle 50%; line: median; whiskers: 1.5 IQR; dots: outliers. Unchanged prices excluded.',fontsize=9)
    path=out/'test_price_changes_7_to_0.png'
    fig.savefig(path,dpi=140);plt.close(fig)
    observations.to_csv(out/'observations.csv',index=False)
    summary.to_csv(out/'daily_summary.csv',index=False)
    selected.to_csv(out/'selected_snapshots.csv',index=False)
    return dict(observations=observations,summary=summary,plot_path=path)
