# Delayed opening-logit strategy

The strategy has two independent Optuna searches. Model trials choose only L1
`alpha`, minimizing the equal-weight mean Brier score across sklearn
`TimeSeriesSplit` folds. The splitter operates on sorted event dates, then maps
each fold back to its fight rows. Scaling and encoding are fitted separately
inside every training fold. Model training always uses `config.open_feats`.

`TrainTestBuilder` filters dates strictly after February 26, 2010 and targets an
85% chronological training split. The complete boundary event goes into test,
so the exact percentage may be slightly lower. Missing closing prices do not
remove opening-training rows. The remaining history is evaluated sequentially;
settled outcomes can enter later retraining windows.

## Timing

Default `n_months=3`. At boundary T, tune betting parameters on `[T-n,T)` using
predictions produced by models available in that interval. Deploy those betting
parameters on `[T,T+n)`, using the opening model refitted at T. Alpha selection,
preprocessing, model fitting, and the inference-logit fit all precede the events
they predict. The first tuning block uses an earlier warm-up fit. Previously
predicted outcomes are never regenerated using a later-trained model.

Dates are event identifiers in this dataset: all fights sharing a date remain
together. If multiple distinct cards share a date, they must be given distinct
event identifiers before extending this implementation to that dataset.

## Betting simulations

Each betting trial creates one combined moneyline/parlay return per historical
event using frozen probabilities and inference SEs. Existing top-two-EV parlay
selection and the shared 100% event exposure cap apply; heavy-favorite parlays
remain disabled. No parlay crosses an event. The searched parameters are `mdd`,
`N`, `mdd_parlay`, `N_parlay`, `z`, and the moneyline American-odds bounds.

Each of `k=1000` paths samples whole events with replacement and shuffles the
sampled occurrences. Repeated events remain separate simulation steps. All paths
start at the same bankroll, and both bet families settle together after EVERY
event. There is no three-event batching. No-bet events stay in the sample with
zero returns. The paths share a deterministic sample-index matrix across trials
within a tuning window so candidate parameters face the same scenarios.

For path j, `R[j] = ending_bankroll[j] / initial_bankroll - 1`. The objective is
`mean(R) / std(R, ddof=1)`, not the Sharpe of event returns or bankroll levels.
Cash-only paths score zero. A nonzero constant terminal return has an undefined
ratio and scores negative infinity; if every trial is undefined, the run fails
explicitly. Bankruptcy stays at zero bankroll. Family return outputs attribute
profits to one shared bankroll rather than separately compounded accounts.

`odds_type` accepts `open`, `close1`, or `close2`. Selected real/fair odds and their
probability difference are substituted into the opening model's input slots
only at inference, and those same real odds price the bets. Closing odds never
enter model training or alpha validation.

The selected-feature unpenalized inference logit supplies its own probability
mean and delta-method probability SE. The regularized logit supplies the primary
prediction. `z` controls the existing uncertainty exposure adjustment. These SEs
are approximate post-selection estimates, not full model-selection uncertainty.

## Running from the repository root (PowerShell)

```powershell
$env:PYTHONPATH = 'src'
python -m ufc_betting.Models.RiskManagement.delayed_walk_forward_logit `
  --data Data/training_data/entire_odds_stats_2026-03-09.csv `
  --output-dir Data/model_results/delayed_open `
  --odds-type open --months 3 --paths 1000
```

Optional `--test-start` overrides the default 85% boundary. `--test-end` is
exclusive. `--model-trials`, `--betting-trials`, `--folds`, `--bankroll`, and
`--seed` control the searches. `--mode alpha` writes only the dated alpha search
and its models. `--mode betting --alpha-history PATH` reuses a historical alpha
CSV without optimizing alpha again. All modes use the same training date filter.

Python callers can use `search_alpha_history` for the model search and pass its
DataFrame to `run_delayed_walk_forward(..., alpha_history=history,
odds_type='close2')`. `FrozenBettingSearch` consumes already frozen predictions;
it never fits a model. `predict_from_alpha_history` reconstructs opening-only
models from the historical training rows before handing predictions to it.

The alpha-history schema is:

| Column | Meaning |
| --- | --- |
| `effective_date` | First date on which this selected alpha was available |
| `training_cutoff` | Exclusive cutoff for both alpha selection and model training |
| `alpha` | Positive regularization strength |
| `training_start` | Optional inclusive lower training bound |
| `cv_brier` | Optional model-search score |

For example, an alpha selected at `2024-02-11` using only rows before that date
has both `effective_date` and `training_cutoff` equal to `2024-02-11`.
Effective dates must be unique; training cutoff cannot exceed effective date.
The first prediction must have an available historical alpha. When supplying a
CSV, its dates must truthfully describe when alpha was selected; the program
cannot audit an external optimizer's provenance. Include every intended refit
boundary, including the warm-up boundary. As-of lookup never selects future rows.

Outputs include `alpha_history.csv`, model bundles and fold audits, run settings,
per-window betting trials, tuning/test predictions (with model cutoffs and SEs),
event returns and family profits, and `bootstrap_paths.npz`. The archive contains
sampled event indices, bankroll paths including their starting balance, terminal
returns, and cumulative family profit contributions. Indices reference the rows
of that window's `tuning_events.csv`. A separate CSV lists all terminal returns.
Top-level `windows.csv`, `events.csv`, and `predictions.csv` contain the sequential
out-of-sample results. Simulated path returns are tuning results, not independent
out-of-sample performance estimates.

## Input to the existing Kelly simulator

The returned `betting_data` is a pandas DataFrame containing only chronological
out-of-sample fights, their selected-market odds, probabilities, inference SEs,
fighter names, outcomes, and the applicable `best_*` betting parameters.

```python
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly

# results is the dictionary returned by run_delayed_walk_forward(...).
df_kelly = results['betting_data']
kelly_results, parlay_results = simulate_kelly(
    df_kelly, **results['kelly_sim_kwargs']
)
```

The matching arguments select the requested odds type, use the inference risk
probabilities and SEs, enable regular parlays, and apply each event's optimized
parameters. Heavy-favorite parlays stay disabled; the DataFrame includes inert
defaults for their fields because the simulator's parameter resolver requires
them. All probability and odds column lists are ordered blue, red (winner 0, 1).

The same inputs are saved as `betting_data.csv` and `kelly_sim_kwargs.json`:

```python
import json
import pandas as pd
from pathlib import Path

output = Path('Data/model_results/delayed_open')
df_kelly = pd.read_csv(output / 'betting_data.csv', parse_dates=['date'])
kwargs = json.loads((output / 'kelly_sim_kwargs.json').read_text())
kelly_results, parlay_results = simulate_kelly(df_kelly, **kwargs)
```
