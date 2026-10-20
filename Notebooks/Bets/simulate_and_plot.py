from ufc_betting.BettingStrategy.betting_params import BettingParams
from ufc_betting.Models.Search.prediction_input import read_frozen_predictions
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly
from ufc_betting.BettingStrategy.Backtest.backtest_plots import plot_backtest

from ufc_betting.config import settings
from IPython.display import display
from dataclasses import replace

def simulate_and_plot(
    odds_type, initial_bankroll=500, *, use_decay=True,
    split="val", model_type="open",
    prediction_template="{split}_logit_{model_type}_with_{odds_type}.csv",
    predictions_path=None, opening_predictions_path=None,
    show_conservative_edge_returns=False,
    plot_path=None,
):
    """Load saved market parameters and selected predictions, simulate, and plot.

    split/model_type select the prediction CSV, not new fitted betting parameters.
    prediction_template supports {split}, {model_type}, and {odds_type}.
    predictions_path overrides the template; relative paths use model_predictions.
    If closing data lacks opening prices, supply opening_predictions_path for
    the same fights/split explicitly. Never fall back to validation predictions.

    Opening runs without decay. Closing use_decay=True requires the joint
    search JSON containing decay_rates and fixed opening_params. False uses
    the saved closing Kelly controls with decay disabled.
    plot_path optionally saves the backtest figure (parent folders are created).
    Returns the moneyline and parlay result DataFrames.
    """
    if odds_type not in ("open", "close1", "close2"):
        raise ValueError("odds_type must be open, close1, or close2")

    from pathlib import Path

    data_dir = settings.data_dir
    def resolve_prediction_path(value):
        path = Path(value)
        path = path if path.is_absolute() else settings.model_predictions_folder / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Prediction CSV not found: {path}. Check split/model_type, "
                "edit prediction_template, or pass predictions_path explicitly.")
        return path
    params_path = data_dir / "kelly_params" / f"{odds_type}.json"
    prediction_path = resolve_prediction_path(
        predictions_path if predictions_path is not None else prediction_template.format(
            split=split, model_type=model_type, odds_type=odds_type))
    betting_params = BettingParams.load(params_path)
    decay_enabled = use_decay and odds_type != "open"
    if decay_enabled and (betting_params.decay_rates is None or betting_params.opening_params is None):
        raise ValueError(f"{params_path} must contain decay_rates and opening_params from the joint search")
    if not decay_enabled:
        betting_params = replace(betting_params, decay_rates=None, opening_params=None)
    predictions = read_frozen_predictions(prediction_path)

    opening_data = None
    if decay_enabled:
        # Use original opening prices for the same rows; the fixed opening
        # policy comes from this closing JSON, not a separately changed open.json.
        reference_cols = ["date", "fighter_red", "fighter_blue", "dec_open_red", "dec_open_blue"]
        if opening_predictions_path is not None:
            opening_data = read_frozen_predictions(resolve_prediction_path(opening_predictions_path))
        elif set(reference_cols).issubset(predictions.columns):
            opening_data = predictions[reference_cols].copy()
        else:
            raise ValueError(
                "Decay requires original opening odds for these fights. Supply "
                "dec_open_red/blue in the selected CSV or opening_predictions_path "
                "pointing to matching fights from the same split.")

    sim_input = KellySimInput.from_predictions(
        predictions,
        betting_params,
        odds_type=odds_type,
        initial_bankroll=initial_bankroll,
        calc_parlay=True,
        opening_data=opening_data,
    )
    betting_data, sim_kwargs = sim_input.to_simulation_args()
    moneylines, parlays = simulate_kelly(betting_data, **sim_kwargs)

    print(f"Predictions: {prediction_path}")
    print(f"{odds_type}: decay {'enabled' if betting_params.decay_rates is not None else 'disabled'}")
    display({
        key: value
        for key, value in betting_params.to_dict().items()
        if key != "opening_params"
    })
    if plot_path is not None:
        plot_path = Path(plot_path).expanduser()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_backtest(moneylines, init_bankroll=initial_bankroll, path=plot_path)
    if show_conservative_edge_returns:
        plot_conservative_edge_returns(
            moneylines, betting_data, sim_kwargs, z=betting_params.z, market=odds_type,
        )
    return moneylines, parlays


def plot_conservative_edge_returns(moneylines, betting_data, sim_kwargs, *, z, market):
    """Compare placed singles by signed conservative edge under the existing policy.

    Keep zero/negative edges before clipping. Event diagnostics summarize
    the individual adjustments; only the exposure cap is shared. Parlays have no single-fight conservative edge here.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    keys = ['date', 'fighter_red', 'fighter_blue']
    inputs = betting_data.copy()
    inputs['date'] = pd.to_datetime(inputs[sim_kwargs['date_col']]).dt.normalize()
    # Both standard and decay simulations select the supplied predicted side.
    sides = inputs[sim_kwargs['pred_winner_col']].to_numpy(dtype=int)
    rows = np.arange(len(inputs))
    risk_cols = sim_kwargs.get('risk_prob_cols')
    if risk_cols is None:
        risk_cols = (['risk_proba_blue', 'risk_proba_red']
                     if {'risk_proba_blue', 'risk_proba_red'}.issubset(inputs.columns)
                     else sim_kwargs['prob_cols'])
    probability = inputs[risk_cols].to_numpy(dtype=float)[rows, sides]
    fair = inputs[sim_kwargs['fair_decimal_cols']].to_numpy(dtype=float)[rows, sides]
    se = (inputs[sim_kwargs['prediction_se_col']].to_numpy(dtype=float)
          if z is not None else np.zeros(len(inputs)))
    signed_edge = probability - (0. if z is None else z) * se - 1 / fair
    if not np.isfinite(signed_edge).all():
        raise ValueError('Conservative-edge inputs must be finite')
    inputs['signed_conservative_edge'] = signed_edge
    bets = moneylines.copy()
    bets['date'] = pd.to_datetime(bets['date']).dt.normalize()
    bets = bets.merge(inputs[keys + ['signed_conservative_edge']], on=keys,
                      how='left', validate='one_to_one')
    if bets.signed_conservative_edge.isna().any():
        raise ValueError('Could not match every result to its prediction inputs')
    # Exclude unplaced bets and frozen post-ruin allocations; retain placed voids.
    event_start = bets.groupby('date')['bankroll_postevent'].first().shift(1)
    if not event_start.empty:
        event_start.iloc[0] = sim_kwargs['init_bankroll']
    # One observation per event, including combined moneyline and parlay P/L.
    # Derive returns from balances: standard and decay outputs use different
    # units for bankroll_pct_change.
    events = bets.groupby('date').agg(
        bankroll_postevent=('bankroll_postevent', 'first'),
        risk_multiplier=('uncertainty_multiplier', 'first'),
        exposure=('event_total_fstar', 'first'),
    )
    events['Event return (%)'] = 100 * (events.bankroll_postevent / event_start - 1)
    events = events.loc[(event_start > 0) & (events.exposure > 0)].copy()
    events['Outcome'] = np.select(
        [events['Event return (%)'] > 1e-10, events['Event return (%)'] < -1e-10],
        ['Winning event', 'Losing event'], default='Break-even event',
    )
    bets = bets.loc[(bets.choice_fstar > 0) & (bets.date.map(event_start) > 0)].copy()
    groups = ['Positive conservative edge', 'Zero/negative conservative edge']
    bets['edge_group'] = np.where(bets.signed_conservative_edge > 0, *groups)
    bets['Bankroll return (%)'] = 100 * bets.choice_fstar * bets.net_odds
    bets['Return on stake (%)'] = 100 * bets.net_odds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    if bets.empty:
        axes[0].text(.5, .5, 'No placed moneyline bets', ha='center', transform=axes[0].transAxes)
    else:
        sns.histplot(data=bets, x='Bankroll return (%)', hue='edge_group', hue_order=groups,
                     bins='auto', multiple='layer', element='step', alpha=.3, ax=axes[0])
    axes[0].set(xlabel='Bankroll return (%)', ylabel='Number of bets',
                title='Moneyline returns by conservative edge')
    if events.empty:
        axes[1].text(.5, .5, 'No wagered events', ha='center', transform=axes[1].transAxes)
    else:
        sns.scatterplot(data=events, x='Event return (%)', y='risk_multiplier',
                        hue='Outcome', style='Outcome', alpha=.65, s=45,
                        palette={'Winning event': 'tab:green', 'Losing event': 'tab:red',
                                 'Break-even event': 'tab:gray'}, ax=axes[1])
    axes[1].set(xlabel='Event return (% of starting bankroll)',
                ylabel='Mean per-bet risk multiplier (0–1)', ylim=(-.03, 1.03),
                title='Winning and losing events: moneylines + parlays')
    for ax in axes:
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    fig.suptitle(f'{market}: current policy (z={z})')
    plt.show()
    summary = bets.groupby('edge_group').agg(
        bets=('net_odds', 'size'),
        mean_bankroll_return_pct=('Bankroll return (%)', 'mean'),
        mean_return_on_stake_pct=('Return on stake (%)', 'mean'),
    ).reindex(groups)
    summary['bets'] = summary['bets'].fillna(0).astype(int)
    display(summary)
    return fig, axes, bets
