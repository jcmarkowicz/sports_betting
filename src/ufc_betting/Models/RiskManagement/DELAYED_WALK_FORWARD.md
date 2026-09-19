# Delayed opening Logit with close1 betting

Entry point: `ufc_betting.Models.RiskManagement.delayed_walk_forward_logit`.
The original search retains its default behavior; the delayed search explicitly
disables heavy-favorite parlays in the shared event settlement engine.

For a holdout beginning September 1:

1. Use all eligible historical rows dated **before June 1**. Choose opening
   Logit's L1 `alpha` with shuffled, event-date-grouped K-fold validation,
   minimizing pooled out-of-fold **Brier score**. Scaling and categorical
   encoding are fitted separately inside each fold. Fit the selected model on
   all those historical rows and save its complete preprocessing/model bundle.
2. Load and freeze that bundle. Predict June 1–August 31 with close1 prices in
   the opening model's market-feature slots. Recompute power-devig fair odds
   from close1 prices. Optimize betting parameters on these three months.
3. Keep both model and betting parameters frozen for September 1–November 30.
   These are the independent test events. Repeat at December 1, advancing all
   boundaries by three months. Actual bankroll carries across test windows.

All intervals include their start and exclude their end. A requested final
interval shorter than three months is marked `partial_test_window`.
The grouped K-fold splits in step 1 are deliberately not chronological;
every row in them still precedes the three-month betting-tuning interval.

## Search

`delayed_bayesian_search.FrozenBettingSearch` accepts a prediction dataframe;
it never trains a model. Each trial jointly searches:

| Parameter | Search range |
| --- | --- |
| Moneyline MDD | 0.15–0.70, step 0.01 |
| Moneyline N | 200–1500, logarithmic sampling |
| Parlay MDD | 0.15–0.70, step 0.01 |
| Parlay N | 200–2500, logarithmic sampling |
| Moneyline minimum American odds | −600 to −100, step 25 |
| Moneyline maximum American odds | +100 to +500, step 25 |

Only top-two-EV parlays are enabled. Heavy-favorite parlays are excluded from
both parameter tuning and holdout evaluation. The searched parlay MDD/N applies
to the top-two-EV parlay. Moneyline bounds apply only to moneylines. `z=0` is
fixed; the existing event settlement engine and its total exposure cap are
reused. No independent heavy-parlay MDD, alpha, z, or leg-selection parameters
are searched during the betting stage.

The objective maximizes `sum(log1p(event_return))` over the entire three-month
tuning period. This ranks settings identically to final compounded bankroll.
All tickets from the same event are settled together before compounding to the
next event. No-bet events contribute zero; bankruptcy scores negative infinity.
Events are not shuffled because shuffling fixed returns cannot change this
objective. `N` is the existing sizing formula's horizon, not the CV fold count.

## Run

From an environment with the repository and its scientific Python dependencies
installed:

```powershell
python -m ufc_betting.Models.RiskManagement.delayed_walk_forward_logit `
  --data Data/training_data/entire_odds_stats_2026-03-09.csv `
  --test-start 2023-09-01 --test-end 2026-03-01 `
  --output-dir Data/model_results/delayed_open_close1 `
  --model-trials 30 --betting-trials 100 --folds 5 --bankroll 500
```

Use a new output directory for each experiment. Each window saves
`opening_model.joblib`, both trial tables, tuning and test predictions, event
returns, and a summary. Root-level `events.csv` contains the continuous test
bankroll; `windows.csv` separates tuning growth from holdout growth. Saved
bundles can be loaded with joblib and passed to `predict_close1`, followed by
`FrozenBettingSearch`, without any retraining. Production model files are not
overwritten.

Training uses `config.open_feats` and drops rows with missing opening features
or unsettled/nonbinary outcomes. Historical training does not require close1
odds. Tuning/test rows must have valid close1 prices. Feature values must be
pre-event values; this pipeline cannot undo lookahead in an input CSV.

The `delayed_open_close1_smoke` output predates the exclusion of heavy-favorite
parlays and must not be used to evaluate this revised strategy. It was only a
small execution check, not a completed strategy search.
