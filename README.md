# CounterPredictor

CS2 match prediction. `PLAN.md` is the round-by-round plan (round model + simulator, still to build).
This README covers the **pre-match map-winner model** (`prematch.py`), which is already working.

## Data

Kaggle [`ektarr/counter-strike-pro-matches`](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches). Unzip it into `data/raw/` (that folder is gitignored).
Two quirks matter:
- `team1_win` is the **series** winner, repeated on every map row. The map winner comes from `score1_game > score2_game`.
- `team*_id` is a per-match entry id (Vitality has 135 of them). Teams are keyed by **name**. Player ids are stable.

Only per-map rows are used (`is_total == False`, map and score present, winning score ≥ 13): 10,670 maps from 2023-10-25 to 2026-06-28.

## Run

```
python3 prematch.py train                                              # ~2 min on a laptop; writes models/prematch.pkl
python3 prematch.py predict "Team Vitality" "Team Spirit" Mirage 2026-07-01
python3 test_prematch.py                                               # leakage + symmetry checks
```

`predict(team_a, team_b, map, date)` replays every map before `date`, uses each team's last known lineup, and returns `{team_a: p, team_b: 1-p}`.

## Model

Features are computed by replaying maps in time order. Every map in a series sees only the state from before that series started.
Each feature is a team A minus team B difference:
- overall Elo, per-map Elo
- last 10 maps: win rate and mean round differential
- lineup K/D, ADR and KAST over each player's last 20 maps
- days since last match, log maps played (overall and on this map)
- roster overlap with the previous map
- head-to-head win rate

Context features: log h2h count, event tier, best-of.

The models are trained on both orientations, and predictions average P(A,B) and 1−P(B,A), so P(A)+P(B)=1 exactly and there is no slot bias.

Time split:
- train 2024-01-15 → 2025-06-30 (6,641 maps; the first 3 months are dropped as Elo burn-in)
- validation 2025-07 → 2025-12 (1,471 maps)
- test 2026-01 → 2026-06 (1,861 maps)

Calibration is Platt scaling fitted on validation.

## Results (test, 2026-01-01 → 2026-06-28, n=1861)

| model | log loss | Brier | AUC | acc |
|---|---|---|---|---|
| coin flip | 0.6931 | 0.2500 | 0.500 | – |
| Elo only | 0.6685 | 0.2377 | 0.620 | 0.597 |
| Elo + map Elo | 0.7016 | 0.2483 | 0.613 | 0.590 |
| **logistic regression** | **0.6566** | **0.2324** | **0.634** | 0.601 |
| logreg + Platt | 0.6566 | 0.2324 | 0.634 | 0.601 |
| HistGradientBoosting | 0.6576 | 0.2328 | 0.633 | 0.602 |
| GBM + Platt | 0.6590 | 0.2334 | 0.633 | 0.602 |

Reliability for logistic regression on test:

| predicted | n | mean pred | actual |
|---|---|---|---|
| 0.2–0.3 | 76 | 0.261 | 0.237 |
| 0.3–0.4 | 193 | 0.358 | 0.420 |
| 0.4–0.5 | 438 | 0.454 | 0.516 |
| 0.5–0.6 | 483 | 0.550 | 0.571 |
| 0.6–0.7 | 401 | 0.648 | 0.646 |
| 0.7–0.8 | 229 | 0.742 | 0.738 |
| 0.8–0.9 | 30 | 0.819 | 0.733 |

Notes:
- GBM does not beat logistic regression here. With about 6.6k training maps, the signal is mostly linear in Elo and form.
- Isotonic calibration overfit the validation set and raised log loss (≈0.675), so Platt scaling replaced it. Its fitted slope is about 1, meaning logreg was already calibrated.
- Adding raw map Elo on top of Elo overweights it. In the learned models it helps only a little.
- The team listed first wins 56.9% of test maps. The model predicts 54.5% for that slot without seeing slot order, so the listing order (likely seeding) carries a bit of extra information that we deliberately don't use.

## Out-of-sample and cross-source check (2026-10-06)

Question: does the backtest hold on matches the model never saw, and on data from a different source?
The model in `models/prematch.pkl` was **not retrained** for this check. Its weights come from Kaggle data through 2026-06-28. Elo, form and h2h still update map by map as new results arrive (walk-forward).

- **C, same source, later dates:** the refreshed Kaggle dataset (v5, `data/kaggle_v5/`) adds 1,097 maps from 2026-06-29 to 2026-10-03.
- **B, independent source:** Liquipedia via its MediaWiki API (`liquipedia.py`, cached in `data/liquipedia/`), 33k maps for 2026, all tiers. Team codes are resolved to names with `{{TeamBracket}}` and matched to Kaggle names. Liquipedia has no player stats, so player features stay at their last Kaggle values. Only maps where both teams are known to the model are scored.
- HLTV was not used: its pages sit behind a Cloudflare "verify you are human" check.

Log loss with a 95% bootstrap CI (resampled by match):

| test set | n | coin flip | Elo only | frozen logreg | AUC | acc |
|---|---|---|---|---|---|---|
| Kaggle 2026H1 (original test) | 1861 | 0.693 | 0.669 | **0.657** [0.643, 0.670] | 0.633 | 0.601 |
| C: Kaggle Jul–Oct 2026 | 1097 | 0.693 | 0.682 | **0.658** [0.642, 0.676] | 0.654 | 0.621 |
| B: Liquipedia Jul–Oct 2026 | 2261 | 0.693 | 0.689 | **0.675** [0.666, 0.685] | 0.603 | 0.572 |
| B, S-tier only | 341 | | | 0.650 | | 0.619 |
| B, both teams with 20+ training maps | 727 | | | 0.660 | | 0.614 |

The model stays calibrated on both new sets. Predictions in the 0.5–0.8 range land within ~0.01 of the actual win rate (full tables in `data/oos_report.txt`).

Source-bias checks (Jan–Jun 2026 overlap):
- **Agreement:** where both sources have the same map, they agree on the winner 99.6% of the time and on the score 98.6% (1,746 maps). The few mismatches are mostly maps swapped within one series.
- **Coverage:** Kaggle covers 89% of Liquipedia's S-tier maps and 77% of A-tier, but only ~10% of B/C-tier. Kaggle is a top-tier sample.
- **Listed-first:** the team listed first wins 55–56% of maps on Liquipedia too, vs 57% on Kaggle. That's real seeding information, not a Kaggle artifact, and the model deliberately ignores it.

**Verdict:**
- **Same source:** the result holds out of sample. Log loss on unseen Jul–Oct maps (0.658) is the same as the original test (0.657).
- **Across sources:** it also beats Elo and coin flip on Liquipedia's independent results. The margin is smaller overall (0.675) because that sample is mostly lower-tier teams with thin history. On top-tier and well-covered teams it matches the Kaggle numbers (0.650–0.660).
- **What it is:** a modest edge (about 60% accuracy, AUC ~0.63), not a strong predictor.

### v2 model (all data)

`python3 oos_eval.py v2` retrains logreg on Kaggle v5 plus the 30.6k Liquipedia maps Kaggle doesn't have, and saves it as `models/prematch_v2.pkl`. The original is kept.

Check before adopting it: with the same features trained only on data before 2026-06-29, adding Liquipedia improved log loss on the Kaggle Jul–Oct maps from 0.659 to 0.646 (Elo alone: 0.682 → 0.663). The extra matches sharpen the ratings.

```
python3 oos_eval.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-07
```

## Model vs bookmaker (Pinnacle closing odds, Jul–Oct 2026)

`odds.py` fetches CS2 fixtures and Pinnacle price history from OddsPapi. The free plan needs a key in `.oddspapi_key`, which is gitignored, and every response is cached in `data/odds/cache/`. `market_eval.py` compares the **frozen** model (trained through 2026-06-28) with Pinnacle's closing price on the same matches.

- **Closing price:** the last snapshot where both teams are quoted, before the earlier of Kaggle's start time (UTC+3, converted to UTC) and OddsPapi's scheduled start. The bookmaker margin is removed proportionally.
- **Sample:** 184 Kaggle series matched to an OddsPapi fixture with Pinnacle odds.

| | Map 1 log loss | Map 1 acc | Series log loss | Series acc |
|---|---|---|---|---|
| coin flip | 0.693 | – | 0.693 | – |
| Elo only | 0.653 | 60.3% | 0.640 | 64.1% |
| **frozen model** | **0.642** | 63.6% | 0.627 | 65.2% |
| **Pinnacle closing** | 0.647 | 58.7% | **0.604** | 67.4% |
| model + market blend (leave-one-out) | 0.652 | 60.9% | 0.607 | 68.5% |

How to read it:
- **Series winner, the market's main product:** Pinnacle is better. Model minus market log loss is +0.024, 95% CI [−0.012, +0.060]. Blending the model in doesn't improve on the market.
- **Map 1:** tied (−0.005, CI [−0.033, +0.022]). Pinnacle's map-1 line is probably mostly set before the veto, while our model already knows which map is being played.
- **Where the edge comes from:** the model and the market pick the same favourite 80% of the time. Most of the predictable part is shared.
- **Bookmaker accuracy:** even Pinnacle is right only ~67% on series and ~77–79% on its 70%+ favourites. A claimed 83% across all matches would beat the sharpest market by a wide margin.

## Keeping data fresh

```
python3 update.py                                                          # just download
python3 oos_eval.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-07 --update   # download, then predict
python3 oos_eval.py v2 --update                                            # download, then retrain v2
```

- **Kaggle:** `--update` re-pulls the Kaggle dataset with kagglehub and copies it only when the version changes. Old versions stay in kagglehub's cache.
- **Liquipedia:** it re-fetches Liquipedia tournaments dated in the current and previous month, bypassing the cache, and merges them without dropping older rows.
- **Staleness warning:** it warns if the newest Kaggle match (the only player-stat source) is more than 3 weeks old.
- **Odds:** OddsPapi is never called by `--update`, to protect the free quota.
- **Prediction log:** every `predict` appends a row to `data/predictions.csv`, written before the match, so a forward track record builds up that can't be cherry-picked later.

## v3: player Elo, lineup continuity, map comfort, veto simulation (`v3.py`)

What's new:
- **Player Elo:** a rating per player and per (player, map). A team's strength is its lineup's average, so stand-ins and roster moves shift it right away.
- **Lineup continuity:** how many of today's 5 played in the team's last 5 maps.
- **Map comfort:** how often each team played this map recently, map-pool depth, and map 1/2 (picks) vs decider.
- **Veto simulation:** each team bans its worst remaining map and picks its best (by the model's own per-map probabilities). This gives a series probability before the maps are known, and `predict` works without a map.
- **No real veto data:** Liquipedia's free wikitext has no pick/ban data (0 of 28,101 matches), so who picked what is inferred.

```
python3 v3.py eval                                              # ablation + Pinnacle comparison (~4 min)
python3 v3.py train                                             # -> models/prematch_v3.pkl (v1/v2 kept)
python3 v3.py predict "Team Vitality" "Team Spirit" 2026-10-07  # series via simulated veto
python3 v3.py predict "Team Vitality" "Team Spirit" 2026-10-07 Mirage --bo 3 --update
```

Ablation: trained on everything before 2026-06-29 (Kaggle + Liquipedia), tested on Jul–Oct. Log loss change vs base, with a 95% CI clustered by match:

| feature group | Kaggle Jul–Oct (1,097 maps) | Liquipedia Jul–Oct (2,398 maps) |
|---|---|---|
| base (v2 features) | 0.6457 | 0.6678 |
| + player Elo | +0.0002 [−0.0008, +0.0013] | −0.0003 [−0.0009, +0.0003] |
| + continuity | −0.0001 [−0.0002, −0.0000] | −0.0000 |
| + map order/comfort | −0.0004 [−0.0015, +0.0007] | +0.0002 [−0.0005, +0.0009] |
| all (v3) | 0.6455, −0.0002 [−0.0018, +0.0014] | 0.6677, −0.0000 [−0.0010, +0.0009] |

Against Pinnacle closing on the same 184 series (model minus Pinnacle, log loss; positive means Pinnacle is better):

| | map 1 | series, map-1 shortcut | series, simulated veto |
|---|---|---|---|
| base | +0.016 [−0.021, +0.053] | +0.045 [−0.005, +0.094] | +0.040 [−0.006, +0.085] |
| v3 | +0.014 [−0.023, +0.052] | +0.043 [−0.008, +0.092] | **+0.037** [−0.010, +0.080] |

Verdict:
- **New features:** they add nothing measurable. Player Elo mostly duplicates team Elo, because Kaggle lineups are stable and most stand-in information arrives after our data does.
- **Veto simulation:** the only real gain is in how series odds are built. It's about 0.005–0.007 better than assuming every map plays like map 1, and it's honest before the veto.
- **Pinnacle:** still ahead on series.
- **Note on training data:** the frozen v1 (Kaggle-only training) does better on this top-tier Pinnacle subset (series 0.627, map 1 0.642) than models trained on Kaggle + Liquipedia. Liquipedia's lower-tier data helps overall but dilutes the player-stat signal at the top tier.
