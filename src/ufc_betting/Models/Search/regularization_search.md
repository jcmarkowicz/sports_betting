# Walk-forward regularization search

Edit `input_path`, `output_dir`, `features`, and the search settings at the bottom
of `regularization_search.py`, then run the file in your editor. There are no
command-line arguments or separate holdout dates.

Point `input_path` directly to `Data/train_val_test/splits/train.csv` (or another
prepared split CSV). All supplied rows are used: no additional date cutoff,
cleaning pass, or outer holdout is applied. Invalid/missing required inputs raise
an error instead of silently dropping rows. Extra metadata columns are ignored.
CSV exports must contain unscaled features, the target, and event dates.

Each fold uses `TrainTestBuilder.scale_encode`; validation uses its fitted
transforms through `prepare_inference_data`.

The runner uses that dataset for expanding-window cross-validation:

1. Sort unique event dates chronologically.
2. Use sklearn `TimeSeriesSplit` to build `n_splits` folds. Each fold trains on
   earlier events and validates on the next block; same-date fights stay together.
3. Fit scaling and encoding on each training fold only.
4. For every Optuna trial, minimize the equal-weight mean validation Brier score
   across these folds.
5. Refit the winning parameters on the complete supplied dataset for future use.

Two independent studies run: Logit searches `alpha`; binomial GLM elastic net
searches `alpha` and `L1_wt`. `n_trials` is per model, not per month.
`gap` excludes event dates between each training and validation block.
The initial training block is never scored; later validation blocks become
training history for subsequent folds. No data outside the input CSV is used.

```python
from ufc_betting.Models.Search.regularization_search import run_from_csv

results = run_from_csv(
    input_path=r"C:/your_folder/training_data.csv",
    output_dir=r"C:/your_folder/search_results",
    features=['proba_fair_open_diff', 'reach_diff'],
    n_trials=50,
    n_splits=5,
    logit_alpha_bounds=(0.2, 7.0),
    glm_alpha_bounds=(1e-5, 1.0),
    l1_wt_bounds=(0.0, 1.0),
)
print(results['summary'])
print(results['bundles']['glm']['study'].best_params)
```

Only the best parameters are written to disk:

- `logit_best_params.json`: `alpha`.
- `glm_best_params.json`: `alpha` and `L1_wt`.

These files contain plain parameter dictionaries, ready to load:

```python
import json
params = json.loads((output_dir / 'glm_best_params.json').read_text())
# Pass **params to fit_glm_bundle, or to GLM.fit_regularized(method='elastic_net', ...).
```

No folds, trials, model files, or summary files are exported. Existing files
from previous runs are left untouched. Best parameters are also returned as
`results['best_params']`; scores and studies remain available in memory through
`results['summary']` and `results['bundles']`. CV scores are tuning scores,
not an independent test estimate. Use the GLM notebook with the selected
parameters to make simulator-ready exports.

Input features must be unscaled and available at prediction time. Numeric-coded
categories are restored with `categorical_cols`, defaulting to the same names
used by the builder. At least
`n_splits + 1` event dates are needed (more with a gap), and each training fold
must contain both outcome classes. Logit and GLM scale their likelihood penalties
differently, so their alpha ranges differ.

The separate `walk_forward_regularization(builder, ...)` function retains the
optional nested monthly holdout workflow for existing callers. The file runner
calls `search_regularization` directly and does not use that outer workflow.
