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

## Model vs Polymarket, Kalshi and other bookmakers (Jul–Oct 2026)

`polymarket.py` lists closed CS2 events from Polymarket's public Gamma API (no key) and downloads each outcome's price history from its CLOB API. Everything is cached in `data/polymarket/cache/`. `odds.py books bet365,1xbet,kalshi` fetches three more books for the OddsPapi fixtures already matched to Kaggle; OddsPapi's historical-odds calls didn't count against the free quota (still 15/250 after 193 calls). `market_compare.py` scores the **frozen** v1 against each source, using the same pre-start cutoff as the Pinnacle comparison. Run:

```
python3 polymarket.py events 2026-06-25 2026-10-04
python3 odds.py books bet365,1xbet,kalshi
python3 market_compare.py      # -> data/polymarket/model_vs_markets.csv
```

Series winner, log loss (lower is better):

| | n | frozen model | Polymarket | model + Polymarket (LOO blend) |
|---|---|---|---|---|
| all Polymarket-matched series | 509 | 0.655 | **0.619** | 0.611 |
| series Pinnacle also prices | 180 | 0.627 | 0.609 (Pinnacle 0.606) | – |

Every market against Pinnacle, on the series both price (positive means Pinnacle is better):

| | series n | series diff | map 1 n | map 1 diff |
|---|---|---|---|---|
| frozen model | 184 | +0.024 [−0.011, +0.058] | 184 | −0.005 [−0.032, +0.022] |
| Polymarket | 180 | +0.003 [−0.007, +0.014] | 157 | +0.005 [−0.006, +0.017] |
| Kalshi | 176 | +0.003 [−0.019, +0.028] | 154 | +0.016 [−0.005, +0.039] |
| bet365 | 141 | +0.006 [−0.008, +0.022] | 118 | +0.011 [−0.003, +0.025] |
| 1xBet | 182 | +0.002 [−0.006, +0.010] | 20 | too few |

How to read it:
- **All markets agree:** Polymarket, Kalshi, bet365 and 1xBet are within noise of Pinnacle on the series winner. Polymarket and Pinnacle correlate 0.99 and differ by 2 points on average, so any one of them is as good as the others.
- **Polymarket covers far more matches:** 509 of our series against Pinnacle's 184, down to tier-5 qualifiers. There the gap to the model is clear: +0.035 [+0.010, +0.059]. The model is too unsure about lopsided matches between little-known teams.
- **Adding market prices to the model doesn't help:** the blend's small gain over raw Polymarket comes from recalibrating Polymarket itself. A Polymarket-only recalibration scores 0.609, and adding the model makes it slightly worse (+0.002 [+0.000, +0.003]). The fitted model weight is 0.06 against 0.89 for Polymarket.
- **Map 1:** the model is still roughly tied with every market. The markets probably price map 1 before the veto, while the model knows the map.
- **Only exception, small sample:** on Polymarket's tier-1 events (n=55), the model beats Polymarket (0.593 vs 0.618). That is too few to rely on.

### Polymarket price next to a prediction (`--polymarket`)

Add `--polymarket` to any `predict` (prematch.py, oos_eval.py, v3.py, v4.py) to also print Polymarket's price for that exact match. You can also look it up alone:

```
python3 prematch.py predict fnatic "Team Liquid" Mirage 2026-10-19 --polymarket
python3 polymarket.py find "Team Vitality" "9z Team" 2026-10-24
```

- **Matching:** it searches Polymarket's CS2 events starting within 36 h of the date and matches both team names after normalising them. It ignores accents, a "Team" prefix and "Esports/Gaming" suffixes, and it accepts Polymarket's abbreviations (FaZe Clan → FaZe, Ninjas in Pyjamas → NIP). Close variants such as a prefix or a small typo still match, but `exact names` is then false. If the same two teams meet twice in the window, the extra events are listed under `other candidates`.
- **Prices:** before the start it shows the live midpoint for the series and each "Map N Winner" market. After the start it shows the last price before the start, never an in-play number. A `spread` above 0.2 means almost nobody is trading, and that price says little.

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

### v3 trained on Kaggle only (`python3 v3.py eval --kaggle-only`)

With top-tier data only, v3's features still add nothing. Kaggle Jul–Oct: base 0.6587, all v3 features +0.0013 [−0.0022, +0.0049]. Against Pinnacle, the Kaggle-only base is the best variant we have: series +0.021 [−0.011, +0.055], map 1 −0.006 (a tie).

## v4: expected rosters from Liquipedia transfers (`rosters.py`, `v4.py`)

`rosters.py` reads Liquipedia's "Player Transfers/YYYY/Month" pages (2024 → today, 14,241 player moves). It skips staff roles and treats inactive/benched players as not active. 80% of Kaggle's players appear in at least one transfer.

`v4.py` starts from each team's last lineup and applies every transfer dated before the match day. Its features:
- player Elo on that **expected roster**
- players changed in the last 30/90 days
- days since the last change
- newcomers (fewer than 5 maps with the team)

`predict` defaults to the current Liquipedia roster, and `--players a,b,c,d,e` overrides team A's five.

```
python3 rosters.py                                   # fetch/refresh transfers (cached)
python3 v4.py eval [--kaggle-only]
python3 v4.py train                                  # -> models/prematch_v4.pkl (Kaggle-only history, best top-tier variant)
python3 v4.py predict "Team Vitality" "Team Spirit" 2026-10-07 Mirage --players "zywoo,apex,flamez,mezii,kyousuke"
python3 v4.py predict ... --update                   # refresh Kaggle, Liquipedia results and transfers first
```

Log loss change vs **using the team's last lineup**, on unseen Jul–Oct maps (95% CI by match; negative = better). Training data is Kaggle + Liquipedia:

| lineup / features | Kaggle all (1,097) | Kaggle after a roster change (614) | Liquipedia all (2,398) | Liquipedia after a change (1,480) |
|---|---|---|---|---|
| actual lineup (known only at match time) | −0.0016 [−0.0034, −0.0000] | −0.0014 | −0.0006 | −0.0004 |
| expected roster | −0.0014 [−0.0034, +0.0004] | −0.0011 | −0.0006 | −0.0003 |
| + player Elo + change features | −0.0029 [−0.0058, +0.0002] | −0.0035 | −0.0019 | −0.0027 |
| all v3 + change features | **−0.0034** [−0.0062, −0.0003] | −0.0044 [−0.0093, +0.0004] | −0.0016 | −0.0034 [−0.0071, +0.0004] |

Kaggle-only training, compared with Pinnacle (184 series):
- **Map 1:** −0.010 [−0.035, +0.015], a tie.
- **Series:** +0.021 [−0.009, +0.052]. Pinnacle is still ahead.

Verdict:
- **Roster info is real but small.** Knowing the lineup in advance is worth about 0.002–0.003 log loss. Transfer-based expected rosters recover most of that, and the roster-change features add a little more, most right after changes.
- **Not enough to catch Pinnacle**, which very likely prices stand-ins and last-minute news we don't see.
- **Models can disagree a lot on single matchups:** v4 (Kaggle-only history) gives Vitality 44% vs Spirit, v3 (with Liquipedia) 62%.

## HLTV pages you save yourself (`hltv_import.py`)

Open a match page on hltv.org in your normal browser, save it (Ctrl+S, "Webpage, complete" or "HTML only") into `data/hltv_saved/`, then run:

```
python3 hltv_import.py      # offline: parses the saved .html files -> data/hltv_saved/maps.csv
```

- **What it reads:** date, event, teams, best-of, each map's score, the veto (who picked each map, the decider) and per-map player K/D/ADR/KAST when the saved page has them.
- **Team names** are mapped onto the Kaggle names.
- **No network:** it never contacts HLTV.
- **Not yet checked on real pages:** it's tested only on a synthetic page, so the CSS selectors may need small tweaks once a real saved page is available.

## Demos (`demos.py`)

Put `.dem` files you downloaded in `data/hltv_saved/` (or pass a folder), then run `python3 demos.py`. It parses them offline with demoparser2 and writes:
- `data/demos/rounds.csv`: one row per round, with CT/T team, winner, end reason, each side's equipment value and money at freeze-end, and a `complete` flag for the map.
- `data/demos/players.csv`: per player per map, kills, deaths, assists, ADR (damage capped at the victim's remaining HP) and KAST (kill, assist, survived or traded within 5 s).

How it handles messy files:
- **Zero-padded downloads:** demos that arrive zero-padded, with the real demo further inside the file, are recovered into `data/demos/recovered/`. The originals are left untouched.
- **Round restores:** technical-pause round restores, which kill all 10 players with "world" on one tick, are not counted as deaths.

Checked against Kaggle on 60 player-maps:
- kills match 100%
- deaths match 98%
- ADR is off by 1.2 on average
- KAST is off by 2.6 on average

Final scores match Kaggle and Liquipedia on all 7 complete maps.

### Round model from demos (`round_model.py`)

`demos.py` now walks subfolders (one folder per series), caches each parsed demo in `data/demos/parsed/` so reruns only parse new files, and fixes a bug where a round was dropped when the demo started recording after the buy phase. `python3 demos.py validate` compares final scores with Liquipedia.

On 122 demos (113 complete maps, 51 series, 2,404 rounds with economy):
- **Score check against Liquipedia:** 86 maps match, 6 differ (mostly by one round), and 21 aren't on Liquipedia.
- **Round winner model:** cross-validated so a whole series is held out each time, it predicts the round winner from the economy at freeze-end, side, score, pistol flag and map. These are in-round inputs, not pre-match.

| model | log loss | Brier | AUC | acc |
|---|---|---|---|---|
| CT win rate per map | 0.694 | 0.250 | 0.50 | 51.7% |
| logistic | **0.610** | 0.213 | 0.70 | 62.4% |
| gradient boosting | 0.624 | 0.219 | 0.69 | 62.2% |

The log-ratio of CT to T equipment is by far the strongest input. CT round win rate by equipment gap (non-pistol rounds):

| CT equipment vs T | CT round win rate |
|---|---|
| much poorer | 16.5% |
| poorer | 40.8% |
| even | 48.7% |
| richer | 59.5% |
| much richer | 78.6% |

CT-sided maps in this sample: Nuke 59%, Mirage 56%, Dust2 56%. Ancient and Inferno are about 49%.

## Map simulator (`sim.py`, PLAN.md step 4)

How it works:
- **Round model:** a logistic model of P(CT wins round) from the log CT/T equipment ratio, pistol and overtime flags, map, and the pre-match strength of the CT team. Strength is the logit of the frozen v1 map probability.
- **Economy:** buys are fitted from the demos (median equipment per resource level, per side). Next-round resources come from a linear fit on the demos, plus the CS2 loss-bonus ladder.
- **Match rules:** pistols at rounds 1 and 13, side switch after 12, MR3 overtime with $12,500.
- **Strength coefficient:** scaled ("anchored") so the simulated P(map win) reproduces v1. Fitted only from rounds, it under-weights team strength.
- **Map-to-map wobble:** each simulated map draws its own strength (σ fitted). Without it the simulator gives too many close scorelines.

```
python3 sim.py "Team Vitality" "Team Spirit" Mirage 2026-10-07 --ct A --sims 10000
python3 sim.py eval --sims 10000 --out results/ [--data-dir DIR]   # held-out series; writes summary/calibration/scorelines
```

Outputs:
- P(map win)
- top scorelines
- expected first-half score
- P(win) for each pistol round
- P(win) per round number

Local held-out check (113 demo maps, 5 folds by series, 3,000 sims per map):
- **P(map win):** simulator log loss 0.691 vs frozen v1 0.687. That's inside the noise at this sample size; the simulator mostly passes v1's opinion through.
- **Scorelines:** the simulated loser averages 8.12 rounds vs 8.03 on real demo maps and 8.03 across 3,008 Kaggle 2026 maps.
- **What it adds:** full scoreline and round-by-round odds, not better win probabilities.

Cloud run: the `cloud-data` branch holds everything under `data/` that the evaluation reads. That's the demo tables, the Kaggle CSVs, Liquipedia maps and transfers, and the frozen v1 model; no .dem files and no API caches.

## Announced lineups (`lineup.py`)

Before a match, `lineup.py` finds the tournament running that day whose Liquipedia participants list includes both teams. It reads each team's announced players from `{{TeamParticipants}}`, using the same cache and request pauses as `liquipedia.py`:
- **active five:** stand-ins are listed once announced
- **subs:** `status=sub` / `role=sub`
- **players who left:** `status=former`
- **notes:** e.g. "September 27th – bobz stands in for arT who misses the event due to health issues"

`v4.py predict` now picks rosters in this order:
1. `--players` (team A) or `--players-b` (team B) given by hand
2. the lineup announced on Liquipedia
3. last lineup updated with transfers

The output shows where each roster came from.

```
python3 lineup.py "Legacy" "PARIVISION" 2026-10-05
python3 v4.py predict "Legacy" "PARIVISION" 2026-10-05 Mirage                                   # Liquipedia: bobz for arT
python3 v4.py predict "Legacy" "PARIVISION" 2026-10-05 Mirage --players "latto,dumau,n1ssim,try,art"  # manual wins
```

Test case: Legacy at ESL Pro League S24, where bobz stands in for arT.
- **Lineups read:** Legacy's five (latto, dumau, n1ssim, try, bobz) and PARIVISION's (HObbit signed, slaxejezzz benched) both come from the tournament page.
- **On Mirage:** Legacy gets 64.5% with bobz and 60.1% with arT forced by hand. bobz's recent stats rate slightly higher, so the model doesn't penalise the stand-in.
