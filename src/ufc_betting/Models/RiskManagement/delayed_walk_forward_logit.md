# Delayed monthly opening-logit strategy

The initial chronological split defaults to 70% training and 30% test.
TrainTestBuilder keeps the entire boundary date in test when its fights would
otherwise straddle the split, so the actual training fraction can be slightly
below 70%. Dates are filtered strictly after 2010-02-26. Missing closing odds
never remove otherwise eligible opening-training rows.

## Independent schedules

- `model_refit_months=1`: at each monthly boundary, use all eligible history
  strictly before that cutoff. Search only alpha using sklearn TimeSeriesSplit
  on ordered event dates and minimize mean fold Brier. Preprocessing is fitted
  separately inside each fold. Refit the opening prediction logit and its
  selected-feature inference logit using the full available training history.
- `betting_lookback_months=n`: reserve the first n months of TEST predictions
  for calibration. These fights do not appear in tracked betting results.
- `betting_refit_months=k`: after calibration, search betting parameters every
  k months, using only the trailing n months of original out-of-sample
  predictions and outcomes. Hold the betting parameters fixed until the next
  search; the underlying models continue updating monthly.
- `n_paths`: number of bootstrap paths, independent of either monthly schedule.

For test starting January 1, n=6 and k=2: January-June supplies the initial
calibration; July-August uses parameters selected on January-June;
September-October uses parameters selected on March-August. Monthly models
train on all available prior history throughout, including settled prior test
outcomes. Historical predictions are never replaced by predictions from a
later-trained model.

Model boundaries are anchored to the initial test date. Betting boundaries are
anchored to the first betting date. Month-end schedules retain their original
anchor rather than drifting permanently after February. The final betting
period may be shorter. An insufficient initial holdout or an empty tuning/test
window raises an explicit error.

## Model inputs and betting objective

All alpha selection and model fitting use opening features only.
`odds_type` is `open`, `close1`, or `close2`. At inference, the selected
market's odds enter the opening-feature slots and price the bets. The inference
logit's `get_prediction(...).summary_frame()` supplies its probability mean and
SE; the regularized logit supplies the primary probability.

Each trial samples whole event portfolios with replacement and shuffles their
occurrences. The sample-index matrix is fixed across trials in one window.
Each path starts with the same bankroll. Moneyline bets and regular top-two-EV
parlays share one bankroll and settle after each sampled event. No-bet events
remain zero-return occurrences. Heavy-favorite parlays remain disabled.

For each path, R = final_bankroll / initial_bankroll - 1. Maximize:

```python
score = R.mean() - risk_penalty * R.std(ddof=1)
```

`risk_penalty` is fixed per run, must be finite and nonnegative, and is not
searched by Optuna. Zero gives pure mean-return optimization. The penalty
measures dispersion of final returns, not within-path drawdown. Bankruptcy
stays at zero bankroll.

The betting ranges live in FrozenBettingSearch.suggest: mdd, N, mdd_parlay,
N_parlay, z, and the moneyline American-odds bounds.

**Event identity limitation:** chronological folds and the split deliberately
group by date. The existing betting engines also group by date; separate cards
sharing a date are still combined. The boundary split fix does not resolve that
separate event-identity issue.

## Run from Python or an editor

Edit RUN_SETTINGS near the bottom of delayed_walk_forward_logit.py.
The current editor settings preserve the user's six-month lookback and 0.6
risk penalty, and use a monthly model and three-month betting refit.
The no-argument script runs opening search first and reuses its monthly alpha
history for close1 and close2 in separate output subfolders.

```python
from ufc_betting.Models.RiskManagement.delayed_walk_forward_logit import run_search

results = run_search(
    train_size=.7,
    model_refit_months=1,
    betting_lookback_months=6,
    betting_refit_months=2,
    n_paths=1000,
    risk_penalty=.6,
    odds_type='open',
)
df_kelly = results['betting_data']
```

`data` can be a DataFrame or CSV path. By default, run_search reads
Data/training_data/entire_odds_stats_2026-03-09.csv. `test_start` overrides the
initial test/calibration boundary, not the tracked betting start. `test_end`
is exclusive. Outputs default to Data/search_results. An explicit output_dir
contains all exports.

The legacy `n_months` keyword sets both betting intervals; it no longer controls
model retraining. Legacy `k` remains an alias for path count. New code should use
the distinct names above to avoid confusing paths with months.

The CLI supports --model-refit-months, --betting-lookback-months,
--betting-refit-months, --paths, --risk-penalty and --train-size.
--mode alpha runs only monthly model search over the entire test interval.
--mode betting requires --alpha-history. Neither mode needs a pre-test warm-up.

## Reusing alpha history

Supply `alpha_history` as a DataFrame or CSV path to skip alpha optimization.
The history must contain EVERY model boundary in the requested test interval,
with exclusive training_cutoff equal to effective_date. Old histories from the
85% split or a three/six-month model schedule must be regenerated. Model
reconstruction uses the recorded alphas and cutoffs with opening features only.

Required columns: effective_date, training_cutoff, alpha.
Optional: training_start, cv_brier, training_rows. Dates must truthfully describe
when alpha was selected; external optimizer provenance cannot be verified.
A reused history must use the same training data and full-history lower bound.
Use the same test/model schedule for all markets; betting schedules may differ.

## Saved outputs

No training-row CSV is written.

- alpha_history.csv: one selected alpha and training metadata per model period.
- models/DATE/best_alpha.csv, folds.json, opening_model.joblib: winning alpha,
  fold boundaries, and fitted models.
- model_periods.csv: monthly training bounds, alpha, CV Brier and test Brier.
- model_predictions.csv: all monthly out-of-sample predictions, including the
  initial calibration holdout.
- calibration_predictions.csv: only that initial test holdout.
- best_params.csv: one winning betting parameter set per deployment period.
- predictions.csv and betting_data.csv: tracked betting predictions only,
  with each period's best_* settings, selected odds, and inference SEs.
- windows.csv and events.csv: sequential tracked betting results.
- kelly_sim_kwargs.json and run_config.json: simulator inputs and run settings.

Each tracked betting date folder contains best_params.csv, summary.json,
tuning_predictions.csv, tuning_events.csv, test_predictions.csv, test_events.csv,
bootstrap_terminal_returns.csv and bootstrap_paths.npz. Bootstrap event indices
refer to tuning_events.csv. Path arrays include the starting bankroll and family
profit contributions to the same combined bankroll.

Optuna per-trial INFO logging is suppressed. Console output shows model
retraining ranges, alpha effective dates, and the tuning/validation periods for
each odds type. Full trial CSVs are not written. Existing outputs from old runs
are not automatically deleted: use a fresh output folder when retaining both
versions of the strategy.

## Kelly simulation

### Reuse saved opening models for closing searches

`EDITOR_MODE = 'betting_only'` (the default) runs open, close1, and close2 betting
searches concurrently using saved opening models, with no model fitting or alpha
search. Edit `OPEN_BETTING_RUN_SETTINGS`, `CLOSE1_RUN_SETTINGS`, and
`CLOSE2_RUN_SETTINGS` independently. Results are saved under `open`, `close1`, and
`close2` within the search results folder, separate from the saved models.

The saved-model searches also write `open_best_params.csv`,
`close1_best_params.csv`, and `close2_best_params.csv` directly in
`opening_results_dir` (normally `Data/search_results`). Each CSV retains one
winning parameter set per betting period per completed run. Columns include
the run ID, UTC start time, output folder, zero-based winning Optuna trial
number, tuning/validation dates, search settings, objective score, and best
parameters. Subsequent runs append their period winners, preserving earlier
runs. Full trial lists are not exported. `best_params_dir` can override this
summary location.

```python
betting_results = run_all_betting_searches(
    OPEN_BETTING_RUN_SETTINGS, CLOSE1_RUN_SETTINGS, CLOSE2_RUN_SETTINGS)
open_results = betting_results['open']
close1_results = betting_results['close1']
close2_results = betting_results['close2']
```

`EDITOR_MODE = 'model_then_closing'` trains opening models using
`OPEN_MODEL_SETTINGS`, then immediately runs the close1 and close2 betting
searches using their independent settings. Both searches automatically use the
newly trained models' output folder and input data. No opening betting search runs.

`EDITOR_MODE = 'model_only'` runs only `OPEN_MODEL_SETTINGS` when
executing the file without arguments. It searches alpha with TimeSeriesSplit
and saves monthly expanding-history opening models, alpha history, fold details,
and run configuration. It does not run betting searches or export training rows.
You can also call it directly:

```python
model_results = run_opening_model(**OPEN_MODEL_SETTINGS)
```

Use `EDITOR_MODE = 'closing'` to reuse those saved models for both closing
searches, or `'all'` for the combined opening model/betting run followed by
both closing searches. Edit `CLOSE1_RUN_SETTINGS` and `CLOSE2_RUN_SETTINGS` independently:
each exposes its betting lookback, betting refit interval, path count, risk
penalty, trial count, initial bankroll, seed, and output folder.

`opening_results_dir` points to the completed opening run. It supplies
`run_config.json`, `alpha_history.csv`, and `models/DATE/opening_model.joblib`.
The closing search inherits the saved test dates, split, feature list, and model
schedule, and loads the fitted opening models without retraining or searching
alpha. Missing models raise an error instead of triggering a fit. Use the same
historical input data as the opening run. Model trial counts and CV fold counts
are not closing-search settings.

```python
close1_results = run_search(**CLOSE1_RUN_SETTINGS)
close2_results = run_search(**CLOSE2_RUN_SETTINGS)
```

Supplying only `alpha_history` through the older API still reconstructs models
by fitting with those alphas; use `opening_results_dir` to avoid fitting.

```python
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly

kelly_results, parlay_results = simulate_kelly(
    results['betting_data'], **results['kelly_sim_kwargs']
)
```

To load saved results:

```python
import json
from pathlib import Path
import pandas as pd

output = Path('Data/search_results')
df_kelly = pd.read_csv(output / 'betting_data.csv', parse_dates=['date'])
kwargs = json.loads((output / 'kelly_sim_kwargs.json').read_text())
kelly_results, parlay_results = simulate_kelly(df_kelly, **kwargs)
```

The initial calibration holdout is excluded from these inputs. Column lists are
blue first, red second. Inert heavy-favorite defaults are included only because
the simulator's parameter resolver expects them; that strategy remains disabled.
