# Validation versus test audit

Audit of the current local prediction CSVs, October 24, 2026. No model was trained, no betting controls were retuned, and original prediction files were not overwritten.

## Conclusion

The test moneyline loss outliers are present in the underlying wagers; they are not caused by a different return formula, repeated event totals, or percentage scaling in these plots. The same market-specific simulation and plotting functions run for validation and test. Independent settlement checks passed for all 315 market-events (2,898 fight-market rows).

One genuine notebook-state bug was fixed: the last opening-test inference cell in `lr_open_close.ipynb` passed `df_inference`, a variable used earlier for validation, instead of `df_test`. Its plot title also said Validation. All three test inference cells now use `df_test`. A regression test executes those cells with deliberately stale validation state and verifies that all three receive the fresh test cohort. This was a latent bug: all three current saved test exports already match fresh inference on the new CSV, to floating-point precision.

## Comparable summaries

Validation: 671 fights across 72 event dates, May 11, 2024–February 7, 2026. Test: 295 fights across 33 dates, February 21–October 17, 2026. All three markets use identical fight keys within each split. There are seven validation voids and four test voids. Only feature-complete fights are represented; these are not necessarily complete event cards.

Moneyline returns below are net profit divided by the total bankroll at the start of that event, expressed as percentages. Standard deviation is across events.

| Market | Val mean | Test mean | Val SD | Test SD | Val worst | Test worst |
|---|---:|---:|---:|---:|---:|---:|
| open | 10.53% | 5.60% | 19.01% | 28.78% | -28.92% | -63.62% |
| close1 | 4.91% | 0.29% | 10.51% | 18.84% | -17.05% | -50.54% |
| close2 | 7.28% | 3.55% | 14.43% | 23.11% | -19.63% | -60.17% |

| Market | Val mean event pick accuracy | Test mean event pick accuracy | Val average total exposure | Test average total exposure |
|---|---:|---:|---:|---:|
| open | 74.06% | 71.23% | 73.00% | 78.96% |
| close1 | 73.10% | 69.84% | 22.76% | 29.88% |
| close2 | 70.99% | 71.10% | 37.06% | 46.73% |

The event-accuracy statistic includes unique fights wagered through singles OR parlays, excludes void outcomes, and averages each event equally. Events without eligible wagers have missing accuracy and are omitted from that histogram. It is not stake-weighted, moneyline-only, or a profitability metric. A two-pick event and an eight-pick event each contribute one observation.

Actual moneyline-only win rates: open 72.49% validation versus 71.97% test; close1 69.86% versus 67.50%; close2 68.94% versus 69.85%. Weighting wins by the fraction of bankroll staked gives 74.47% versus 70.48%, 75.48% versus 63.34%, and 73.80% versus 67.88%, respectively. These latter figures weight relative stakes, not dollar stakes across different bankroll levels. Losing test picks received larger average fractions than winning test picks in every market; validation showed the reverse.

## What produces the outliers

On October 17, 2026, opening moneylines selected Melissa Croden, Kyle Daukaus, and Erin Blanchfield. All three recorded outcomes were losses:

| Pick | Decimal odds | Model probability | Bankroll staked | Return contribution |
|---|---:|---:|---:|---:|
| Melissa Croden | 1.4000 | 78.56% | 24.85% | -24.85% |
| Kyle Daukaus | 1.4444 | 73.45% | 15.52% | -15.52% |
| Erin Blanchfield | 1.4348 | 76.19% | 23.24% | -23.24% |

Sum: -63.62% in moneylines. The parlay lost another 9.99%, for a combined event loss of 73.60%. This is exactly the allocated exposure; no extra loss was invented by the plot.

On May 16, all five closing moneyline selections lost: Melquizael Costa, Jacqueline Cavalcanti, Nikolay Veretennikov, Timmy Cuamba, and Tuco Tokkos. Their stakes sum to 50.54% for close1 and 60.17% for close2. The parlay contributions were -0.09% and -4.05%. On April 25, four of five placed opening picks and three of four placed closing picks lost, producing moneyline losses of 49.10%, 35.03%, and 44.55%.

Validation had zero moneyline events below -30%. Test had five opening, two close1, and three close2 events below -30%. This is an observed difference in the sampled outcomes and exposures, not proof of a permanent distributional change.

The configured `mdd` is an approximate per-bet sizing input. It is not an event loss limit or a guaranteed realized portfolio drawdown limit. The shared event exposure cap is 100% of bankroll. Several losing singles plus a losing parlay on the same card can therefore produce the observed losses while respecting every implemented cap. Parlays can also contain losing moneyline selections, concentrating the event's loss further.

## Why accuracy and bankroll tell different stories

Accuracy ignores stake size and payout odds. A winning favorite at decimal 1.40 earns only 0.40 times its stake; a loss costs the entire stake. Concentrating several such losses in one event heavily damages compounded wealth. A positive arithmetic mean event return does not guarantee positive compounded growth.

With the same $500 initial bankroll, current test combined balances finish at $386.98 (open), $281.80 (close1), and $643.30 (close2). Validation finishes at $1,116,283.04, $11,867.44, and $80,774.99, respectively, over a longer history. Those large historical figures reflect the simulated fractional staking and repeated compounding; they are not independent evidence that the strategy generalizes.

The models are also different: validation uses a 3,994-row train-only fit with alpha 6.985; test uses a 4,658-row train-plus-validation fit with alpha 3.66. The scaler and SE model sample sizes match their respective fitted models. Fresh inference reproduced all six saved probability and SE columns to about 1e-16. The search workflow uses validation-reference predictions, so validation is not an untouched test of the entire selection/tuning process; current search paths may also be sampled/synthetic rather than the historical sequence itself.

As a diagnostic, applying the original train-only model and its preprocessing to the same test fights still produces worst moneyline losses of 56.96%, 43.23%, and 53.47%, on the same worst dates. The refit changes stake magnitudes but does not explain away the extreme events. Those counterfactual exports and their summary are saved separately here.

## Calculation audit

Both notebooks call `Notebooks/Bets/simulate_and_plot.py`, then `Results/test_simulation.py::simulate_saved_predictions`, then `KellySimInput.from_predictions` and `simulate_kelly`. For each market both load the same `Data/kelly_params/<market>.json`, start at $500, enable parlays, disable opening decay, and enable closing decay. Closing calculations delegate to the common decay portfolio calculator; the branch differs by market, not by validation/test split.

Checks performed on all six fresh runs:

- No duplicate fight keys in simulation inputs.
- Independent moneyline settlement: `stake_fraction * (decimal_odds - 1)` for wins, `-stake_fraction` for losses, zero for voids; sum once per event.
- Independent ticket settlement and one stake per ticket, not per repeated leg. The repository's policy refunds the whole ticket when any leg is void; that policy is shared across both splits.
- Positive stakes, event exposure at most 100%, continuous chronological bankrolls, and matching moneyline-plus-parlay event totals.
- Independently reconstructed wagered-fight accuracy, including parlay-only selections and excluding voids.
- Execution odds are used for favorite comparisons. Closing-market remapping does not change the recorded execution prices used in these checks.
- All test exported fights and winners match the new test CSV. Fresh probabilities/SEs match saved artifacts; no saved-model/scaler mismatch was detected.

There is a pre-existing schema inconsistency: ordinary simulation stores fractional `bankroll_pct_change`; decay stores percentage points. `plot_bankroll_distributions` explicitly detects and normalizes this using the independently reconstructed component returns. Thus it does not create a 100x error here. Code directly consuming this column elsewhere should not assume uniform units.

The bottom moneyline/parlay plots show cumulative actual profit divided by initial bankroll, using their shared combined bankroll. They are not independently compounded moneyline-only and parlay-only portfolios. Separate notebook histograms also select their own return bins and use raw event counts; `comparison.png` uses common bins and densities to compare 72 versus 33 events fairly.

## Verification and scope

The targeted unittest run executed 26 tests: 25 passed, including the new notebook-state regression, return plots, accuracy eligibility, void settlement, joint exposure cap, post-decay risk adjustment, and live/backtest parity. One existing adapter test fails on the dtype of `best_moneyline_min_american` (normalized float64 versus legacy int64); it does not demonstrate differing numeric controls or payouts. It was left unchanged rather than relaxing the assertion during this audit.

The full-data independent checks pass. These checks establish consistency with recorded CSV outcomes and prices, not an independent verification of scraped sports results or a full point-in-time feature/leakage audit. Historical inputs and saved artifacts can change when rerun.

To regenerate the six backtests, checks, summaries, and comparison figure, run `Notebooks/Bets/audit_val_test.py` from the repository root with `src` and the root on PYTHONPATH. It writes only into this audit directory and does not execute notebook training or the bootstrap optimization cells. The train-only counterfactual CSVs are separate saved diagnostic outputs.

The next useful strategy experiment would be a predeclared lower event exposure cap, evaluated on a fresh holdout or walk-forward periods. Changing controls to remove these already-observed losses and then treating the same test data as unseen would overstate the evidence. No betting-policy changes were made here.
