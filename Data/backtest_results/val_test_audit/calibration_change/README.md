# Why the aggregate calibration gap changed

This analysis uses settled moneyline bets whose model probability exceeds the fair market probability by at least 10 percentage points. It is descriptive, using already-observed outcomes; the time breakdown is exploratory, not a pre-specified change-point test.

## Temporal aggregation concealed earlier deterioration

| Market | Validation before July 2025: predicted / actual (n) | Later validation: predicted / actual (n) | Test: predicted / actual (n) |
|---|---|---|---|
| Opening | 68.0% / 74.2% (66) | 71.5% / 60.7% (28) | 68.9% / 57.1% (42) |
| Close1 | 73.2% / 78.9% (38) | 70.5% / 57.1% (21) | 69.9% / 54.3% (35) |
| Close2 | 73.2% / 80.0% (40) | 70.3% / 57.1% (21) | 70.2% / 52.9% (34) |

The full validation aggregate combines early underconfidence with later overconfidence. Its near-zero average residual does not establish stable calibration. This undermines the premise that the failure began suddenly at the test boundary. The quarterly CSV also shows the late-validation weakness without relying only on this July split.

## The refit contributes little on these same test selections

The fitted model and scaler differ between validation and test. Holding the current high-edge test fights and chosen sides fixed, the older model predicts 68.0%, 68.9%, and 69.2% for opening/close1/close2, versus 68.9%, 69.9%, and 70.2% from the new model. Thus the refit adds about one point of confidence; it does not explain a 12–17-point residual. This fixed-selection diagnostic is not a simulation of which bets the older model would select.

## The market-relative advantage stopped appearing in outcomes

For closing high-edge test bets the market assigns about 57%, the model about 70%, and actual wins are 53–54%. The market is much closer on this selected cohort. Validation had realized extra wins above market expectations approximately matching the model's claimed advantage; test does not. This describes the failed model signal, without identifying a causal feature or proving that market probabilities are true probabilities.

Opening favorites are especially informative: validation high-edge favorite bets won 40/55 (72.7%), versus 12/24 (50.0%) in test. Their average model probabilities are similar (77.4% and 76.1%), as are market probabilities (64.3% and 62.7%). Opening high-edge bets opposing the market favorite won 66.7% in both periods. Thus the opening deterioration is concentrated in favorites the model considered especially safe, not a general increase in contrarian picking.

Across all eligible settled test fights, mean pick confidence and accuracy remain close (opening 69.4%/69.1%; close1 70.4%/70.8%; close2 70.1%/70.4%). These averages do not prove overall calibration: subgroup errors can cancel.

## Limits of the explanation

The evidence supports unstable performance of the model's market-relative signal over time, with early validation making the pooled estimate look stronger. It does not identify a particular feature relationship as the cause or separate persistent drift from sampling variation. An event-cluster bootstrap gives wide 95% intervals for the increase in average calibration residual: opening [-3.7, 29.0] points; close1 [-6.1, 34.6]; close2 [-3.8, 37.0]. These intervals include zero, assume sampled events are exchangeable within each period, and do not correct for exploratory subgroup selection or model/strategy tuning. Markets reuse many fights and are not independent replications.

The earlier integrity audit ruled out material prediction changes from the chronology bug and verified feature/prediction reproduction. Unresolved source-odds conflicts remain documented separately. Excluding the known ambiguous validation fight does not change these high-edge aggregates because that fight was not in this placed-bet subset.

The decisive next evaluation is chronological walk-forward prediction and calibration using only prior data, reporting market-relative residuals by period and edge band. This test set has now been examined repeatedly and cannot serve as untouched evidence for a corrective model fitted using these findings.
