# Compact Monte Carlo paths

Open `src/ufc_betting/BettingStrategy/MonteCarlo/generate_paths.py`, edit the `generate_path_run(...)` arguments under
`if __name__ == "__main__":`, then run the file in your editor. No command-line
arguments are needed. Outputs are new directories under `Data/bootstrapped_paths`,
not `.pkl` files. Existing directories are never overwritten; change `output_path`
for another run.

| Desired paths | `sample_fights` | `sample_prices` |
|---|---|---|
| Resample fights, retain recorded prices | `True` | `False` |
| Resample fights and prices | `True` | `True` |
| Shuffle intact cards, retain recorded prices | `False` | `False` |
| Shuffle intact cards and sample prices | `False` | `True` |

`n_paths` controls the number of paths and `seed` makes generation reproducible.
`days_out` selects the price models when `sample_prices=True`.

`strict_numbered_months=False` preserves all numbered cards and spreads them as
evenly as possible across months. The current source has 25 numbered cards in
22 months, so three months must contain two numbered cards. Setting this argument
to `True` requires exactly one per occupied month and raises an error for this
source. No cards are silently dropped, duplicated, or moved into invented months.
Intact-card mode preserves every fight once per path and order within each card.

The low-level `generate_ordered_paths` preserves legacy storage defaults; request
`storage="compact"` and an output directory explicitly when using it directly.

## Use recorded prices in search

```python
from ufc_betting.config import settings
from Notebooks.Search.run_search import run_search

results = run_search(
    "open", n_trials=100, n_paths=100,
    paths_path=settings.data_dir / "bootstrapped_paths" / "sampled_fights_recorded_prices_val",
)
```

`run_search` attaches the selected market's prediction CSV. It now defaults to the
compact recorded-price directory. Pass an old `.pkl` path explicitly to reuse a
legacy dataset; legacy pickle loading still reads that entire file.

## Use sampled prices

```python
from ufc_betting.BettingStrategy.MonteCarlo.sampled_paths import (
    load_sampled_paths, predict_intermediate_paths,
)

selected = load_sampled_paths(
    settings.data_dir / "bootstrapped_paths" / "sampled_fights_sampled_prices_val",
    n_paths=100, random_state=42,
)
# Pass your existing inference artifacts, just as in the intermediate-price workflow:
# paths = predict_intermediate_paths(
#     selected, prediction_model=model, prediction_model_type="open",
#     prediction_scalers=scalers, prediction_encoders=encoders,
#     prediction_se_model=se_model, search_market="close1",
# )
# optimize_kelly_params(validation_data={"close1": reference_predictions},
#     sampled_paths={"close1": paths}, use_decay=True, opening_params=opening_params)
```

Sampled prices need matching predictions. Recorded-price prediction attachment
deliberately rejects them. Distinct price/prediction inputs remain distinct in the
search cache, even when they come from the same donor fight.

## Storage and evaluation

The directory stores the source table once, memory-mapped donor IDs, compact
per-occurrence overrides, and a versioned manifest with a source checksum.
String overrides use integer dictionaries. Generation writes each path immediately;
it does not collect all path DataFrames in memory. A manifest is published only
after all paths finish. Load only trusted datasets, as the source table uses pickle.

`generate_ordered_paths(..., storage="compact")` returns a lazy `CompactPaths`
sequence. Close it with `paths.close()` when finished, or open it with a context
manager; this releases Windows memory-map file handles. Selecting paths through
`load_sampled_paths(..., n_paths=100)` reconstructs only those selected frames.

Search computes fight-level stakes once per unique set of numerical inputs per
trial. It batches card-level parlay selection, uncertainty and exposure adjustment,
then compounds each path's returns. The opening-policy cache is reused across
closing trials. Final card returns cannot be cached across different parameter
trials. Optuna prints a completion line for each trial.

## Evaluate saved paths

Edit the run settings in `src/ufc_betting/BettingStrategy/MonteCarlo/run_simulation.py`,
then run `python -m ufc_betting.BettingStrategy.MonteCarlo.run_simulation` from an
environment with the project installed. This replaces the former notebook-folder
`generate_results.py` script. Saved results remain at
`Data/montecarlo_results/intermediate_results.pkl`; open `results.ipynb` to analyze them.
Path generation can also run with `python -m ufc_betting.BettingStrategy.MonteCarlo.generate_paths`.
