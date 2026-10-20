# Frozen validation betting search

`optimize_kelly_params` tunes once on each complete validation dataset. Results are
in-sample betting-tuning results. Supplied predictions are never refitted.

For joint closing Kelly/decay tuning, first run an opening-only search with
`validation_data={"open": open_validation}`, `use_decay=False`, and
`params_dir="Data/kelly_params"`. Then call the same function with closing
datasets, `use_decay=True`, and `opening_params="Data/kelly_params/open.json"`.

Each closing market has its own study. Trials vary `mdd`, `N`, `mdd_parlay`,
`N_parlay`, `z`, the minimum American moneyline, and four decay rates together.
Opening controls stay fixed. Raw stakes are blended before the closing `z`
correction and the shared moneyline/parlay exposure cap. Trials reuse fixed
event paths and optimize mean terminal fractional return minus the SD penalty.

`opening_data` optionally supplies original opening odds matched by fight;
otherwise closing data must already include opening decimal prices. Supplied
opening and closing frames must have the same fights and outcomes, but can
have different row orders. Closing predictions are used throughout each trial.

`params_dir` saves complete `<market>.json` policies. `betting_params` and
`best_params` contain decoded decay rates; `study.best_params` and
`trial_params` retain the `log1p_*` search coordinates. Use `kelly_sim_input`
for a matching backtest. For decay exports, `kelly_sim_kwargs.json` refers to
`opening_data.csv`; load that CSV before passing those kwargs to a backtest.

```python
from ufc_betting.config import settings
from ufc_betting.Models.Search.engine import optimize_kelly_params

validation_data = {
    "open": open_validation,      # DataFrame or path to a CSV
    "close1": close1_validation,
    "close2": close2_validation,
}
path_set = settings.data_dir / "bootstrapped_events" / "validation_v1"

results = optimize_kelly_params(
    validation_data=validation_data,
    n_trials=100,
    k=1000,
    risk_penalty={"open": 0.25, "close1": 0.4, "close2": 0.4},
    bootstrap_mode="save",       # generate, save, or load
    bootstrap_dir=path_set,
    sample_with_replacement=True,
)

open_df = results["open"]["betting_data"]
close1_df = results["close1"]["betting_data"]
close2_df = results["close2"]["betting_data"]
```

To replay, use `bootstrap_mode="load"` and the same `bootstrap_dir` and dataset.
Changing optimization settings or risk penalties is allowed. `k` can select
the first k samples of a larger saved set. A new seed does not change loaded
paths, but still controls the Optuna sampler. To sample another saved set, use
a new directory or omit `bootstrap_dir` for an automatically named directory.
`generate` samples in memory without writing files. `save` rejects an existing
market directory instead of overwriting it.

Each `open/`, `close1/`, and `close2/` subfolder contains:

- `paths.npy`: integer array shaped `(n_samples, n_events)`. Values index the
  chronological `event_dates` array in metadata; repeated events remain separate
  settlement steps.
- `metadata.json`: schema version, creation time, market, seed, replacement flag,
  sample count, event dates, fight counts, fighter identities, dataset fingerprint,
  fingerprint columns, and a path-file checksum.

Loading verifies dates, fight identity/order within events, outcomes, execution
and fair prices, probabilities, SEs, and title metadata when supplied. Changes to
these inputs require a new saved set. Unused feature columns do not affect the
fingerprint. Loaded paths are reused across all trials.

## Validation columns

Provide `date`, `fighter_red`, `fighter_blue`, and settled `winner` (0=blue,
1=red), plus both sides of `dec_<market>_*` and `dec_fair_<market>_*`.
Kelly report preparation also requires `dec_open_blue` and `dec_open_red`,
unless `open_blue` and `open_red` are already provided.

Predictions can use either format:

- `proba_red` and `prediction_se`, with optional `risk_red_proba`.
- `choice_proba`, `choice_fighter_bool` (or `pred_winner`, 0=blue, 1=red),
  and `se` (or `proba_se`), with optional `risk_choice_proba` or `risk_red_proba`.

Supplied choice probabilities take precedence over a coexisting `proba_red`
column. Explicit risk probabilities are used only for uncertainty adjustment;
when omitted, they default to the supplied betting probabilities. There is no
probability-source selection or model-based replacement of betting probabilities.

SE remains the same when converting a blue probability to its red complement.
`title_fight` is optional with replacement and required without replacement.

`FrozenBettingSearch` also accepts these inputs directly and exposes the same
`bootstrap_mode`, `bootstrap_dir`, and `sample_with_replacement` controls. For that class, `bootstrap_dir` is the single
market directory itself. The three-market entry point adds the market subfolders.

## Run the Kelly backtest separately

The search returns a KellySimInput object and never invokes the backtest.

```python
from ufc_betting.BettingStrategy.Backtest.backtest_functions import simulate_kelly

inputs = results["open"]["kelly_sim_input"]
kelly_results, parlay_results = simulate_kelly(inputs.data, **inputs.kwargs)
```

To prepare inputs independently, use
`KellySimInput.from_predictions(predictions, best_params, odds_type="open")`.
The generic `KellySimInput(data, **simulation_params)` constructor accepts every
`simulate_kelly` argument except the DataFrame, which is its first argument.
This includes custom columns, bankroll settings, moneyline and parlay sizing,
heavy-favorite controls, uncertainty, monthly parameters, and decay inputs.
`to_simulation_args()` returns copies of the data and kwargs.

## Optional Logit preparation

`optimize_kelly_params` requires supplied `validation_data` and contains no model
loading, alpha discovery, or fitting options. For the separate model workflow:

```python
from ufc_betting.Models.Search.logit_search import optimize_kelly_params_with_logit

results = optimize_kelly_params_with_logit(
    input_paths=validation_csv_paths,
    training_input=training_csv,
    use_monthly_models=True,
    model_refit_months=1,
    n_trials=100,
)
```

The wrapper prepares predictions for all markets first, then calls
`optimize_kelly_params(validation_data=...)`. Set `use_monthly_models=False`
to use saved models without monthly refitting.

## Save optimized parameters and reload simulation inputs

```python
import json
from ufc_betting.config import settings
from ufc_betting.Models.Search.engine import optimize_kelly_params
from ufc_betting.BettingStrategy.Backtest.kelly_sim_input import KellySimInput

results = optimize_kelly_params(
    validation_data={
        "open": open_validation,
        "close1": close1_validation,
        "close2": close2_validation,
    },
    n_trials=100,
    risk_penalty={"open": 0.25, "close1": 0.4, "close2": 0.4},
)

params_dir = settings.data_dir / "kelly_params"
params_dir.mkdir(parents=True, exist_ok=True)
kelly_inputs = {}
for market, result in results.items():
    params_file = params_dir / f"{market}.json"
    params_file.write_text(json.dumps(result["best_params"], indent=2), encoding="utf-8")
    kelly_inputs[market] = KellySimInput.from_params_file(
        result["validation_predictions"], params_file, odds_type=market,
    )

open_df = kelly_inputs["open"].data
close1_df = kelly_inputs["close1"].data
close2_df = kelly_inputs["close2"].data
# Run separately: simulate_kelly(open_df, **kelly_inputs["open"].kwargs)
```

The input variables are your existing validation DataFrames or CSV paths.
Each run replaces open.json, close1.json, and close2.json with the latest best
parameters. Later, from_params_file can reload these files without rerunning
optimization; supply the prediction DataFrame to simulate alongside the file.

For `score_type="event_log_growth"`, the objective is mean event log growth
minus `drawdown_penalty * worst_tail_mean(max_drawdown)` minus
`ulcer_penalty * mean(path_ulcer_indices)`. Both weights default to zero.
The Ulcer Index is the RMS fractional drawdown of each path over post-event
balances. Initial wealth establishes the peak but is excluded from the mean;
no-bet, unfinished-drawdown and post-ruin events all count. Actual balances
are used without the log wealth floor or clipping. An index of 0.2 with
`ulcer_penalty=0.01` subtracts 0.002 from the score. The mean is taken after
computing each path's RMS, not by pooling squared drawdowns across paths.
Only event-log-growth scoring accepts a nonzero Ulcer penalty. Results include
per-path indices, the weight, and the mean and penalty amount in
`objective_breakdown`; trial/study metadata and progress output expose these too.
