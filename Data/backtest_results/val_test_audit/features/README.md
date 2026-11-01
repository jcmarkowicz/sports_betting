# Feature integrity audit

The >=10 percentage-point model/market disagreement identifies a poorly performing subset of test bets. It does not establish whether miscalibration, changing fighter relationships, sampling variation, or bad inputs caused the errors. Large stakes translate those errors into extreme event returns.

## What was verified

- Historical and new test data use the same `FeatureEngineering.build_all_stats` pipeline. The test builder computes features with full historical context before filtering to the test dates.
- Rebuilding from locally saved raw history with the pre-fix implementation reproduces all 417 test rows on every selected model feature, including the three market probability differences. See `test_vs_rebuilt_summary.csv`. This establishes reproducibility, not the correctness of every source observation.
- Applying saved models/scalers reproduces all 295 saved test predictions and all 671 validation predictions, with maximum absolute error 1.11e-16. Train-only preprocessing/model artifacts are used for validation; separately fitted train-plus-validation artifacts are used for test. The transformation procedure is shared, but fitted parameters are intentionally different.
- Among 392 overlapping historical/test rows, selected input differences are limited to small Elo changes and two reach changes. The reach changes follow a raw Felipe Franco reach revision from 76 to 77 inches. Among 282 overlapping rows with complete model features, the largest prediction change is 0.02686 percentage points, average 0.00079 points. Those changes cannot explain ten-point edges.
- Focused tests cover unchanged pre-fight model features when current/future results are modified, prefix stability, input shuffle stability, tournament card order, and odds identity/date checks. An additional real-history Elo prefix comparison over 8,517 rows has exactly zero difference after imposing deterministic chronology.
- A full rebuild with corrected chronology, holding market quotes fixed, changes the 295 usable test predictions by at most 0.02338 percentage points (mean 0.00036). On October 17, Joaquin Buckley / Mike Malott moves from 69.7212% to 69.6980% for red. This fix is important for reproducibility but cannot account for the extreme event losses. See `test_chronology_only_probabilities.csv`.

## Bugs and unresolved data conflicts

1. Date-only unstable sorting could reorder fights when history was appended. This affects stateful ratings and early same-day tournament histories. Feature generation now sorts by date, descending scraped bout position, and fighter names. All 8,934 raw fights have bout positions. Historical pre-fix train/validation/test datasets should not be mixed with selectively rebuilt features after changing this convention; rebuild them together and refit after resolving the source conflicts.
2. The odds scraper compared character membership rather than complete name tokens and ignored requested fight dates. It now checks name tokens, requested opponent/date pairs (allowing the existing one-day source-date tolerance), and the requested fighter even on direct result pages.
3. The merge silently kept the first conflicting odds record after aliases and dates were normalized. It now rejects conflicting quotes explicitly. Identical duplicate quotes remain allowed.

`conflicting_odds.csv` contains five canonical fights with conflicting source records:

| Fight | Date | Problem |
|---|---|---|
| Joanne Wood / Taila Santos | 2021-11-20 | Opening quotes differ across aliases |
| Chris Barnett / Martin Buday | 2022-04-16 | Opening quotes differ across source dates |
| Damon Blackshear / Alatengheili | 2025-04-26 | Opening quotes reverse the favorite |
| Andre Fili / Kai Kamaka III | 2026-10-10 | Closing quotes conflict; one record's original names refer to other fighters |
| Brendan Allen / Christian Leroy Duncan | 2026-10-10 | Closing quotes conflict; one record's original opponent is Chris Curtis |

For Blackshear/Alatengheili, the two opening-price alternatives change the saved validation model prediction from about 83.31% to 48.92% for red. This is an alternative-input sensitivity calculation, not a claim that either quote is verified correct. The two test conflicts do not occur on the previously identified extreme-loss dates (April 25, May 16, October 17). Thus these conflicts do not directly explain those events, but prevent declaring all source data clean.

Existing production CSVs, model artifacts, and predictions were not replaced. A full rebuild now intentionally fails until conflicting source prices are reconciled. The audit exports alternatives for review rather than inventing a correction. Glicko2 emits numerical overflow warnings during a complete rebuild; Glicko2 features are not inputs to this logistic model.

## Reproduce

`Notebooks/Bets/audit_feature_integrity.py` measures prediction sensitivity and exports canonical odds conflicts without scraping or training. `tests/test_feature_generation_integrity.py` contains the new integrity regressions. The local focused run passed 17 tests including existing scraper navigation, adjusted rates, attribute imputation, and notebook cohort checks. This is not a claim that the entire repository test suite passes.
