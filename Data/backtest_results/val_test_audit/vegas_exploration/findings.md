# What the data says about Vegas accuracy and the loss outliers

Exploration of the same saved validation/test predictions and audited stakes. No betting controls or models changed. Returns below are moneyline contributions as a percentage of the total starting event bankroll, not dollar P/L or return on stake.

## 1. There is no general decline in closing-market accuracy

Mean per-event favorite accuracy, using the same eligible fights as the notebook:

| Market | Validation, 72 events | Test, 33 events |
|---|---:|---:|
| Open | 68.74% | 65.80% |
| Close1 | 70.69% | 72.04% |
| Close2 | 70.87% | 72.40% |

Fight-weighted favorite accuracy reaches the same directional conclusion: open 68.24% to 65.59%, close1 70.14% to 71.38%, close2 70.39% to 71.78%. Draws/NCs and tied prices are excluded from these favorite accuracies.

Opening accuracy's apparent drop is concentrated: remove April 25 and October 17 as a descriptive sensitivity check and the test event mean becomes 68.44%, almost identical to validation's 68.74%. This is not a proposed rule for removing losses; those events remain part of the results.

Opening favorites actually had a higher average fair implied win probability in test (68.54% versus 66.09%), yet won less often (65.59% versus 68.24%, fight-weighted). Thus the opening deterioration is not simply a greater number of closely matched fights. Closing favorites' average fair probabilities were 69.09%/68.62%, and observed accuracy 71.38%/71.78%, so the analogous shortfall is absent there.

## 2. Different cards have different explanations

| Test date | Market | Vegas wins on eligible fights | Model wins on placed singles | Moneyline return | Interpretation |
|---|---|---:|---:|---:|---|
| Oct 17 | Open | 1/6 | 0/3 | -63.62% | Favorite collapse; all three bets agreed with Vegas |
| Apr 25 | Close2 | 2/10 | 1/4 | -44.55% | Favorite collapse; the three losing favorite bets cost 48.36 points, one nonfavorite win returned 3.82 |
| May 16 | Close2 | 7/9 | 0/5 | -60.17% | Vegas mostly right; four of the five selections opposed Vegas |
| Jul 11 | Open | 6/10 | 1/4 | -43.12% | Market accuracy moderate; losses concentrated among the selected bets |
| Oct 10 | Open | 5/7 | 3/5 | -30.78% | Neither accuracy catastrophic; two heavily staked losses overwhelm three low-payout wins |

May 16 is the strongest counterexample to the Vegas-decline explanation. In close2, the model backed Melquizael Costa, Nikolay Veretennikov, Tuco Tokkos, and Timmy Cuamba against market favorites. All four lost, costing 52.03% of bankroll. The fifth pick, favorite Jacqueline Cavalcanti, lost another 8.14%. On those five fights, market favorites won four. Close1 shows the same selection failure: nonfavorite losses cost 44.28 points and the favorite loss 6.26 points.

The word favorite means the side with the shorter of the two recorded prices. A decimal price below 2.0 is not sufficient to establish favorite status when both quoted prices are below 2.0.

October 10 illustrates why broad accuracy is inadequate. Felipe Franco and Matheus Camilo lost stakes of 24.54% and 22.19%. The three winning bets returned only 6.99%, 7.79%, and 1.17%, for net -30.78%. A 60% win rate on singles and 71.4% Vegas accuracy can still produce a large loss.

## 3. The apparent large-edge bets are a more specific concern

Among placed, settled singles, define model-market edge as chosen-side model probability minus the recorded fair market probability. For bets with an edge of at least 10 percentage points:

| Market | Val predicted | Val actual (wins/bets) | Test predicted | Test actual (wins/bets) |
|---|---:|---:|---:|---:|
| Open | 69.03% | 70.21% (66/94) | 68.87% | 57.14% (24/42) |
| Close1 | 72.21% | 71.19% (42/59) | 69.88% | 54.29% (19/35) |
| Close2 | 72.17% | 72.13% (44/61) | 70.22% | 52.94% (18/34) |

These bets attract substantial stakes. The 10–15-point edge band was especially poor in test: open 13/28 wins, close1 15/29, close2 14/28, versus mean predicted probabilities of about 69–70%. Mean bankroll fractions were 18.2%, 15.3%, and 16.5%. Smaller-edge test bets performed much better. However, the 15+-point group did not show a monotonic deterioration and is very small (six closing bets per market). This does not justify choosing 10 points as a new cutoff based on this sample.

The distinction is economically meaningful in the recorded calculations: the strategy interprets disagreement as an advantage and allocates more bankroll. When that advantage fails to materialize, errors dominate returns even if overall classification accuracy remains acceptable. This is evidence consistent with overestimated advantages on a subset of fights, not proof of persistent model miscalibration or a causal diagnosis. Markets reuse many of the same fights, so the three rows are not three independent replications.

## 4. The selected subset matters more than the whole card

For closing singles opposing Vegas (including tied-price picks), win rates changed from 17/29 to 11/22 for close1 and 19/34 to 12/24 for close2. Weighting those outcomes by relative stakes gives a larger drop: 68.3% to 45.2% and 65.4% to 43.6%. There are few observations, and May 16 contributes heavily.

Opening favorite bets contributed an average +6.12 percentage points per event in validation, versus -0.15 in test. Nonfavorite/tied picks contributed +4.41 versus +5.75. This supports a favorite-loss explanation for opening performance, while the closing tails have a mixture of favorite upsets and failed contrarian selections.

Poor Vegas cards were not automatically poor model cards in validation. For the five validation opening events with favorite accuracy below 50%, mean moneyline return was +12.44%; for six such test events it was -13.72%. For example, July 26, 2025 close2 had only 40% Vegas accuracy, but winning nonfavorite exposure offset losses and the strategy gained 11.46%. April 25, 2026 had 20% Vegas accuracy and much larger favorite losses, with too little offset from the winning nonfavorite.

Descriptive correlation between all-fight event favorite accuracy and moneyline return is only 0.04/-0.03/0.08 in validation (open/close1/close2), versus 0.45/0.30/0.35 in test. On the model's placed singles rather than all available fights, test correlations rise to 0.63/0.38/0.44. These relationships share outcomes with the return calculation and are not independent causal evidence.

## 5. Odds movement also distinguishes some failures

On May 16, the four contrarian close2 picks had all shortened from opening: Costa 2.30 to 2.06, Veretennikov 2.30 to 2.05, Tokkos 3.50 to 2.45, and Cuamba 2.70 to 2.08. The model still made them more likely winners than the market did. For example, Costa was 62.14% model probability versus 47.50% fair market probability, and received a 22.04% stake. Market movement toward a fighter did not mean that fighter had become the market favorite or would win.

Across test close2 placed singles, selections whose prices shortened had 48/76 wins (63.16%) versus mean model probability 73.14%. Those whose prices lengthened had 41/54 wins (75.93%) versus model probability 69.33%. Validation shortening selections won 107/148 (72.30%) versus model probability 71.62%. This suggests inspecting how price movement, model features, and decay translate into stakes; it does not establish that shortening prices are intrinsically bad. These are selected bets and a small, retrospective sample.

## 6. The plotted Vegas accuracy is not full-card accuracy

The new test CSV has 417 rows, while model exports contain 295 feature-complete rows. "Vegas all" in the notebook refers to all fights in those exports, not all scraped fights.

October 17 opening accuracy is 1/6 in the model subset, but 4/12 among all settled, non-tied opening-price rows in the source CSV. The opening card was still poor for favorites, but the subset makes it look even worse. Closing accuracy that date is 2/6 in the model subset versus 7/13 in the wider scraped card. On May 16, wider closing accuracy remains high at 8/11, so the model-versus-market disagreement explanation survives that broader comparison.

Across all raw test rows with usable settled non-tied odds, favorite accuracy is 66.21% open, 72.21% close1, and 72.51% close2. The corresponding model-subset figures are 65.59%, 71.38%, and 71.78%. Thus coverage differences matter more for particular events than the overall test averages. Wider raw-card counts still exclude missing odds, voids, and tied prices; they are not independently verified complete cards.

## Interpretation

The data supports three concrete mechanisms: opening favorite upsets on a few cards; losing contrarian closing picks where Vegas was largely correct; and disproportionate stakes on losing high-edge predictions. Aggressive exposure amplifies all three. A broad fall in Vegas accuracy cannot explain the closing outliers, and the opening average decline is mostly concentrated in two cards.

This analysis describes the recorded sample. It does not distinguish conclusively between transient outcome variance, unstable fitted relationships, stale/misestimated features, or overfitting. It narrows the most useful next investigation to calibration and feature quality for the high model-versus-market disagreement group, with event-clustered uncertainty and a fresh evaluation period. No new betting rule was fitted from these observations.

Reproduction: run `Notebooks/Bets/explore_vegas_outliers.py` from the repository root after the prior audit exports exist. Independent return calculations are asserted against the audited event totals. `events.csv`, `tail_events.csv`, `fights.csv`, `summary.csv`, `edge_bands.csv`, `high_edge_summary.csv`, and `raw_card_coverage.csv` provide the underlying observations. `movement_groups.csv` records the separate movement exploration. `vegas_vs_returns.png` visualizes the event relationship.
