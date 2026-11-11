"""Preserve the live report's calculated artists and subplot layout in Plotly."""
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import textwrap
from types import SimpleNamespace


def _color(value):
    red, green, blue, alpha = colors.to_rgba(value)
    return f'rgba({round(red*255)},{round(green*255)},{round(blue*255)},{alpha})'


def to_plotly(source):
    """Transfer lines, exact histogram bars, references and annotations unchanged.

    Uses axis domains rather than inferring a new subplot arrangement. In
    particular, the combined-return secondary axis remains on its own scale.
    """
    fig = go.Figure()
    legend_names = set()
    for index, ax in enumerate(source.axes, 1):
        suffix = '' if index == 1 else str(index)
        xref, yref = 'x' + suffix, 'y' + suffix
        spec = ax.get_subplotspec()
        rows, columns = spec.get_gridspec().get_geometry()
        # Browser-sized spacing instead of Matplotlib's print-sized margins.
        gap_x, gap_y = .12, .23 if rows == 2 else .16
        cell_width = (.94 - gap_x * (columns - 1)) / columns
        cell_height = (.94 - gap_y * (rows - 1)) / rows
        x0 = .02 + spec.colspan.start * (cell_width + gap_x)
        y1 = .96 - spec.rowspan.start * (cell_height + gap_y)
        pos = SimpleNamespace(x0=x0, x1=x0+cell_width, y0=y1-cell_height, y1=y1)
        is_date = isinstance(ax.xaxis.get_major_locator(), dates.DateLocator)
        def xvalues(values):
            values = np.asarray(values)
            if is_date and np.issubdtype(values.dtype, np.number):
                return [d.isoformat() for d in dates.num2date(values)]
            return values.tolist()
        xaxis = dict(domain=[pos.x0, pos.x1], anchor=yref,
                     title=dict(text=ax.get_xlabel(), standoff=12), showgrid=False,
                     zeroline=False, automargin=True)
        if is_date:
            xaxis.update(type='date', tickformat='%Y-%m-%d')
        else:
            xaxis['range'] = list(ax.get_xlim())
            from matplotlib.ticker import FixedLocator
            if isinstance(ax.xaxis.get_major_locator(), FixedLocator):
                xaxis.update(tickmode='array', tickvals=ax.get_xticks().tolist(),
                             ticktext=[t.get_text() for t in ax.get_xticklabels()], tickangle=-45)
        if not ax.get_xticklabels():
            xaxis['showticklabels'] = False
        yaxis = dict(domain=[pos.y0, pos.y1], anchor=xref, range=list(ax.get_ylim()),
                     title=dict(text=ax.get_ylabel(), standoff=10), showgrid=False,
                     gridcolor='rgba(148,163,184,0.13)', gridwidth=1,
                     zeroline=False, automargin=True)
        if ax.yaxis.get_label_position() == 'right':
            yaxis.update(side='right', showgrid=False)
        fig.update_layout(**{'xaxis'+suffix: xaxis, 'yaxis'+suffix: yaxis})
        if ax.get_title():
            fig.add_annotation(x=(pos.x0+pos.x1)/2, y=pos.y1, xref='paper', yref='paper',
                               text='<br>'.join(textwrap.fill(part, width=34 if columns == 3 else 65).replace('\n', '<br>')
                                                 for part in ax.get_title().split('\n')), showarrow=False,
                               yanchor='bottom', yshift=12, font=dict(size=13))
        for line in ax.lines:
            label = line.get_label()
            # Plotly uses one legend for the whole figure; Matplotlib's local
            # mean labels need their panel context retained in that legend.
            if label.startswith('Mean event accuracy:'):
                panel = ax.get_title().split('\n')[0]
                label = f'{panel} — {label}'
            elif label.startswith('Mean wagered:'):
                label = f'Total bankroll wagered — {label}'
            visible = bool(label and not label.startswith('_') and label not in legend_names)
            if visible:
                legend_names.add(label)
            dash = {'--': 'dash', ':': 'dot', '-.': 'dashdot'}.get(line.get_linestyle(), 'solid')
            x, y = line.get_xdata(orig=False), line.get_ydata(orig=False)
            if line.get_transform() == ax.get_xaxis_transform():
                fig.add_shape(type='line', x0=float(x[0]), x1=float(x[-1]), y0=0, y1=1,
                              xref=xref, yref=yref+' domain', line=dict(color=_color(line.get_color()), dash=dash, width=1))
            elif line.get_transform() == ax.get_yaxis_transform():
                fig.add_shape(type='line', x0=0, x1=1, y0=float(y[0]), y1=float(y[-1]),
                              xref=xref+' domain', yref=yref, line=dict(color=_color(line.get_color()), dash=dash, width=1))
            else:
                fig.add_trace(go.Scatter(x=xvalues(x), y=np.asarray(y).tolist(), xaxis=xref, yaxis=yref,
                    mode='lines+markers' if line.get_marker() not in ('None', None, '', ' ') else 'lines',
                    name=label if visible else None, showlegend=visible,
                    line=dict(color=_color(line.get_color()), width=2.2, dash=dash), marker=dict(size=5)))
                continue
            if visible:
                fig.add_trace(go.Scatter(x=[None], y=[None], xaxis=xref, yaxis=yref,
                    mode='lines', name=label, line=dict(color=_color(line.get_color()), dash=dash)))
        bars = [p for p in ax.patches if isinstance(p, Rectangle)]
        if bars:
            fig.add_trace(go.Bar(x=[p.get_x()+p.get_width()/2 for p in bars],
                y=[p.get_height() for p in bars], width=[p.get_width() for p in bars],
                base=[p.get_y() for p in bars], xaxis=xref, yaxis=yref, showlegend=False,
                marker=dict(color=[_color(p.get_facecolor()) for p in bars],
                            line=dict(color='white', width=.7)),
                hovertemplate='%{x:.3g}<br>Count: %{y}<extra></extra>'))
        for text in ax.texts:
            x, y = text.get_position()
            domain = text.get_transform() == ax.transAxes
            fig.add_annotation(x=x, y=y, xref=xref+' domain' if domain else xref,
                yref=yref+' domain' if domain else yref, text=text.get_text().replace('\n', '<br>'),
                showarrow=False, xanchor='left' if x > 1 else 'center',
                bgcolor='rgba(255,255,255,.9)', font=dict(size=11))
    fig.update_layout(template='plotly_white', height=1080 if rows == 3 else 900 if rows == 2 else 480,
        autosize=True,
        margin=dict(l=55, r=180 if len(source.axes)==3 else 80, t=85, b=120),
        font=dict(family='Arial, sans-serif', size=12, color='#334155'),
        legend=dict(orientation='h', y=-.08, x=0), hovermode='closest', bargap=0)
    if source._suptitle:
        fig.update_layout(title=dict(text=source._suptitle.get_text(), x=.5))
    # Override template defaults on every axis, including secondary axes.
    fig.update_xaxes(showgrid=False, zeroline=False, minor=dict(showgrid=False))
    fig.update_yaxes(showgrid=False, zeroline=False, minor=dict(showgrid=False))
    return fig


def live_figures(moneylines, parlays, balances, market):
    from ufc_betting.Results.live_calibration import plot_live_calibration
    from ufc_betting.Results.returns import prepare_return_plot_data
    from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
        plot_backtest, plot_bankroll_distributions, plot_event_accuracy_distributions,
    )
    events, ml, pl = prepare_return_plot_data(moneylines, parlays, balances, market)
    initial = events.iloc[0].bankroll_postevent - events.iloc[0].event_payout_money_line - events.iloc[0].parlay_net
    figures = {}
    for name, build in (
        ('calibration', lambda: plot_live_calibration(moneylines, market)),
        ('backtest', lambda: plot_backtest(events, initial, show=False)[0]),
        ('bankroll_distributions', lambda: plot_bankroll_distributions(events, show=False)[0]),
        ('event_accuracy', lambda: plot_event_accuracy_distributions(ml, pl, odds_type=market, show=False)[0]),
    ):
        source = build()
        try:
            figures[name] = to_plotly(source)
        finally:
            plt.close(source)
    return figures


def scaled_live_figures(moneylines, parlays, balances, market, opening, closing=None):
    """Compare live overlays; use the best ending compounded path for diagnostics."""
    from plotly.subplots import make_subplots
    from ufc_betting.Results.live_calibration import plot_live_calibration
    from ufc_betting.Results.returns import prepare_return_plot_data
    from ufc_betting.Results.live_scaling import scaled_live_paths, reconstruct_scaled_events
    from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
        plot_bankroll_distributions, plot_event_accuracy_distributions,
    )
    events,ml,pl=prepare_return_plot_data(moneylines,parlays,balances,market)
    events=events.sort_values('date')
    initial=events.iloc[0].bankroll_postevent-events.iloc[0].event_payout_money_line-events.iloc[0].parlay_net
    paths=scaled_live_paths(events.set_index('date'),opening,closing)
    best=paths['winner'];factor=paths['multipliers'][best]
    selected=reconstruct_scaled_events(events,factor,initial)
    # Exclude filtered/zero-allocation events from diagnostic denominators.
    selected=selected.loc[selected.date.map(factor).gt(0)]
    selected_dates=selected.date
    ml=ml.loc[ml.date.isin(selected_dates)].copy()
    pl=pl.loc[pl.date.isin(selected_dates)].copy()
    ml['choice_fstar']*=ml.date.map(factor)
    pl['fstar_parlay']*=pl.date.map(factor)
    for col in ['bankroll_postevent','event_payout_money_line','parlay_net','bankroll_pct_change']:
        ml[col]=ml.date.map(selected.set_index('date')[col])
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
        subplot_titles=['Compounded bankroll (no replenishments)','Cumulative compounded return (%)'],vertical_spacing=.16)
    for i,name in enumerate(paths['wealth']):
        wealth=paths['wealth'][name]
        dates=[events.date.min()-pd.Timedelta(days=1),*wealth.index]
        from plotly.colors import qualitative
        color=qualitative.Plotly[i%len(qualitative.Plotly)]
        for row,y in [(1,np.r_[initial,initial*wealth]),(2,np.r_[0,100*(wealth-1)])]:
            fig.add_trace(go.Scatter(x=dates,y=y,mode='lines',name=name,legendgroup=name,showlegend=row==1,
                line=dict(color=color,width=3 if name==best else 1.8,dash='dash' if 'scalers close2' in name else 'solid')),row=row,col=1)
    fig.update_layout(height=850,title=f'{market}: EWMA-5 +/-0.02 live scaling',legend=dict(orientation='h',y=-.15),margin=dict(b=170),hovermode='x unified')
    fig.update_yaxes(title_text='Bankroll',row=1,col=1);fig.update_yaxes(title_text='Return (%)',row=2,col=1)
    figures={'backtest':fig}
    for name,build in [('calibration',lambda:plot_live_calibration(moneylines,market)),
        ('bankroll_distributions',lambda:plot_bankroll_distributions(selected,show=False)[0]),
        ('event_accuracy',lambda:plot_event_accuracy_distributions(ml,pl,odds_type=market,show=False)[0])]:
        if name != 'calibration' and selected.empty:
            empty=go.Figure()
            empty.add_annotation(text='This path allocates no stakes; no retained events to summarize.',showarrow=False)
            figures[name]=empty
            continue
        source=build()
        try:figures[name]=to_plotly(source)
        finally:plt.close(source)
    return dict(figures=figures,winner=best,ending_return_pct=100*(paths['wealth'].iloc[-1][best]-1),
                selected_events=len(selected),total_events=len(events),multipliers=paths['multipliers'])
