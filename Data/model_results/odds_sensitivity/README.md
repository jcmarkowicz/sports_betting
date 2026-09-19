# Odds sensitivity

Each row groups settled closing choices by their original American odds in a fixed [low, high) bin. Opening ROI uses the opening model's own choices on those same fights. Both strategies stake one unit per fight.

`required_adjustment` is the smallest common favorable shift applied to every closing choice in the bin that reaches the opening ROI. Picks and outcomes stay fixed. Zero means the closing ROI already meets the target. Missing means no solution within the allowed adjustment.

The shift removes the artificial discontinuity at even money: -150 to +150 is 100 continuous points. `mean_raw_american_change` also reports the literal signed-number difference; it includes that discontinuity. `mean_decimal_increase` reports the average payout increase.

`average_gap_closed_pp_per_point` measures the average ROI improvement per continuous point up to the target. The convergence CSV contains the full adjustment path, remaining gap, and percentage of gap closed. Favorite payout changes are nonlinear.

Use `n_bets` and `n_events` to assess sample support. These are historical sensitivity estimates using observed outcomes, not prospective thresholds validated on another holdout. No models are retrained or stakes recalculated.
