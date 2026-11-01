# Upcoming betting policies

`generate_bets(df, return_event_ev=True)` in `UpcomingPicks/generate_bets.py` uses
each market's inference model: open with open, close1 with close1, and close2
with close2. Each `model_se.pkl` supplies both probabilities and standard errors,
with its own scaler and encoder. Artifacts default to
`Data/model_comparisons/lr_open_close/trainval/<market>/`. Pass `fit_mode="train"`
to use the train-only fits used during validation search, or `comparison_dir`
to select another artifact root. The existing `DataPipeline.utils.bets_utils`
import forwards to this entry point. Models are loaded when generating picks.
Model
inputs come from the fitted scaler and encoder rather than the older config
feature lists. Input rows must contain the current model features and decimal
offered/fair odds; rebuild older event feature files before using the new model.

`betting_pipeline` loads `open.json`, `close1.json`, and `close2.json` from
`settings.data_dir / "kelly_params/inference_market_models"` on each call.
These are the policies exported by `run_inference_search.py`; no search is run
when generating picks. JSON files are not modified, and missing artifacts raise
an error instead of falling back to the older models or parameters.
It reuses `BettingReturnCalculator` for moneyline limits, Kelly sizing, top-two-EV
parlay selection, uncertainty adjustment, and the joint event exposure cap.

Opening bets use the opening policy. Each closing stage uses its corresponding
saved policy and embedded opening controls. Both policies are evaluated using
current predictions and prices; price movement from opening determines the
moneyline and parlay decay blend. The closing uncertainty adjustment and shared
bankroll cap are applied once, after blending. `use_decay=False` selects fixed
stage policies instead. No historical fight outcomes are needed to generate bets.

The existing moneyline, two-row parlay, and optional event-EV output schemas are
preserved. Parlay stake and fraction repeat on both leg rows and represent one
ticket. Missing rows retain their positions and cannot enter the ticket. A card
with fewer than two eligible rows has no parlay. Each call accepts one event.

For a custom evaluation, pass `params_dir=...` or
`betting_params={"open": ..., "close1": ..., "close2": ...}` to
`generate_bets` or `betting_pipeline`. Mapping values may be `BettingParams`, dictionaries, or JSON
paths. The old positional `mdd_ml_arr`, `mdd_parlay_arr`, `N_ml_arr`, and
`N_parlay_arr` arguments remain accepted but are ignored with a deprecation
warning; use the saved policies or explicit policy mapping instead.
