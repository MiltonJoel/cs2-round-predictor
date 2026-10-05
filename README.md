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
