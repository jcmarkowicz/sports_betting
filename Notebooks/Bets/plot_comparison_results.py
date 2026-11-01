"""Plot and export any saved logistic-model comparison.

Import plot_comparison_results from notebooks or Python scripts. Importing this
module does not run a simulation or write files.
"""
from Notebooks.Bets.simulate_and_plot import simulate_and_plot
from ufc_betting.config import settings
from ufc_betting.BettingStrategy.Backtest.backtest_plots import (
    plot_bankroll_distributions,
    plot_event_accuracy_distributions,
)


def plot_comparison_results(
    fit_mode, model_type, probability_model, split, odds_type, *,
    initial_bankroll=500, use_decay=None,
    show_conservative_edge_returns=False,
    predictions_root=None, output_dir=None, params_dir=None,
):
    """Plot a saved comparison and export moneylines/parlays; return both frames.

    trainval/train means the combined train+validation fitting cohort.
    Opening models support all three odds markets; closing models use their own.
    Saved market-specific betting parameters are reused, not retuned.
    Repeating the same configuration replaces its files in output_dir.
    params_dir selects the folder containing the market JSON files. For the
    recent inference search use settings.data_dir / 'kelly_params/inference_market_models'.
    None preserves the original parameters. Custom parameter folders get a
    separate output subfolder when output_dir is omitted.
    """
    from pathlib import Path
    import pandas as pd

    if fit_mode not in ('train', 'trainval'):
        raise ValueError("fit_mode must be 'train' or 'trainval'")
    if probability_model not in ('regularized', 'inference'):
        raise ValueError("probability_model must be 'regularized' or 'inference'")
    if model_type not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown model_type')
    if odds_type not in ('open', 'close1', 'close2'):
        raise ValueError('Unknown odds_type')
    if model_type != 'open' and odds_type != model_type:
        raise ValueError('Closing models have exports only for their own odds market')
    allowed_splits = ('train', 'val', 'test') if fit_mode == 'train' else ('train', 'test')
    if split not in allowed_splits:
        raise ValueError(f'{fit_mode} supports splits {allowed_splits}')
    if initial_bankroll <= 0:
        raise ValueError('initial_bankroll must be positive')
    root = Path(predictions_root) if predictions_root is not None else (
        settings.data_dir / 'model_comparisons' / 'lr_open_close' / 'report_predictions')
    prediction_path = (root / fit_mode / model_type / probability_model /
                       f'{split}_with_{odds_type}.csv').resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(f'Missing predictions: {prediction_path}. Run train_lr_models.py first.')
    sources = pd.read_csv(prediction_path, usecols=['probability_model'])
    if sources.empty or not sources.probability_model.eq(probability_model).all():
        raise ValueError('Saved prediction probability source does not match the requested model')
    decay = odds_type != 'open' if use_decay is None else bool(use_decay) and odds_type != 'open'
    tag = f'{fit_mode}_{split}_{model_type}_with_{odds_type}_{probability_model}'
    tag += '_decay' if decay else '_no_decay'
    destination = Path(output_dir) if output_dir is not None else (
        settings.data_dir / 'backtest_results' / 'model_comparisons')
    if params_dir is not None:
        params_dir = Path(params_dir).expanduser().resolve()
        params_path = params_dir / f'{odds_type}.json'
        if not params_path.is_file():
            raise FileNotFoundError(f'Missing betting parameters: {params_path}')
        if output_dir is None:
            destination = destination / params_dir.name
    else:
        params_path = settings.data_dir / 'kelly_params' / f'{odds_type}.json'
    print(f'Betting parameters: {params_path}')
    destination.mkdir(parents=True, exist_ok=True)
    moneylines, parlays = simulate_and_plot(
        odds_type=odds_type, initial_bankroll=initial_bankroll,
        model_type=model_type, split=split, predictions_path=prediction_path,
        use_decay=decay, show_conservative_edge_returns=show_conservative_edge_returns,
        plot_path=destination / f'{tag}_backtest.png',
        params_dir=params_dir,
    )
    # Save simulation outputs before drawing optional diagnostic figures.
    moneylines.to_csv(destination / f'{tag}_moneylines.csv', index=False)
    parlays.to_csv(destination / f'{tag}_parlays.csv', index=False)
    distribution_fig, _ = plot_bankroll_distributions(moneylines, parlays)
    distribution_fig.savefig(destination / f'{tag}_bankroll_distributions.png',
                             dpi=300, bbox_inches='tight')
    accuracy_fig, _, event_accuracy = plot_event_accuracy_distributions(
        moneylines, parlays, odds_type=odds_type)
    accuracy_fig.savefig(destination / f'{tag}_event_accuracy.png',
                         dpi=300, bbox_inches='tight')
    print(f'Saved moneylines, parlays, and plots: {destination.resolve()} / {tag}_*')
    return moneylines, parlays
