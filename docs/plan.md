# CounterPredictor — plan

Goal: before a CS2 map starts, predict how it will go round by round.

## The honest framing

You can't predict "round 7 goes to Team A" before the match starts. Round 7 depends on rounds 1–6 (economy, momentum).
What is predictable is **P(A wins round | pre-match info, round state)**. From that:

1. **Round model**: `p = f(pre_match_features, round_no, side, score, A_money_state, B_money_state)`.
2. **Simulator**: Monte Carlo the map 10k times, rolling economy forward with CS2 rules (loss bonus, pistol rounds, side switch at 12, MR3 OT).
3. **Outputs**: P(win) per round number (marginal), scoreline distribution, P(map win), P(pistol wins), expected half scores.

The round model plus the simulator gives everything from a single model. Skipped: a per-round sequence model (RNN/transformer). Add one only if the simulator is miscalibrated on later rounds.

## Data: what each dataset actually gives

| Dataset | What it has | Use |
|---|---|---|
| `ektarr/counter-strike-pro-matches` (29 MB, CC BY-SA, updated 2026-10-04) | 19k+ pro CS2 maps, T1–T3, teams, lineups, map, datetime, map score, player K/D/ADR/KAST | **Pre-match features**: team/map Elo, form, roster stability, player ratings. No round sequence. |
| `pr4nut/...-150-maps` (3 MB) | 202 anonymized matches, scoreboard only | **Skip**: anonymous, no teams, no rounds |
| `billpureskillgg/cs2-2023-11-23` (**51 GB**) | ~1.3k *matchmaking/FACEIT* matches, full telemetry | **Skip for now**: not pro, huge. Possible later to pretrain the economy→round-win prior. Ask before downloading. |
| **Pro demos (HLTV .dem)** parsed with `demoparser2` (already installed) | Round winner, side, equipment value, money per round | **Round labels**: the only source of pro round-by-round data. |
| `livanoff/csgo-esport-matches-round-data` (Kaggle) | 75k pro *CS:GO* rounds with buy type | Optional prior for economy effects. Different game and MR15, so don't mix it into training. |

Key gap: pro round-by-round labels need to come from demos. Start with ~500–1,000 recent T1/T2 maps. That's about 25k rounds, which is enough for a logistic/GBM round model.
HLTV: download demos by hand or at a polite rate, and check their ToS. Keep raw `.dem` files out of the repo and keep parsed parquet only (~KB per map).

## Steps (each one ends with a check)

1. **Pre-match ratings** (`ratings.py`): Elo per team and per (team, map) from the ektarr maps, in time order with no leakage.
   Check: Elo-only P(map win) log loss on 2026 maps beats a 50/50 baseline and the "higher-ranked team wins" baseline.
2. **Demo → rounds** (`parse.py`): demoparser2 → one row per round: `match, map, round, ct_team, winner_side, ct_eq_val, t_eq_val, ct_money, t_money, end_reason`.
   Check: parsed final score == the score in the ektarr row for the same map.
3. **Round model** (`round_model.py`): LightGBM (installed) or plain logistic.
   Features: Elo diff, map Elo diff, side, round_no, pistol flag, both teams' money bucket, loss-bonus count, score diff.
   Check: on the time split, round log loss and calibration plot vs. a baseline that uses only per-map side win rates.
4. **Simulator** (`sim.py`): apply the round model + economy rules, 10k sims.
   Check: simulated P(map win) is calibrated against held-out real results (Brier, reliability bins), and the simulated scoreline histogram matches the real one.
5. **CLI**: `predict.py TEAM_A TEAM_B MAP` prints the per-round P(A) table and the scoreline distribution.

## Evaluation rules
- Only walk-forward time splits: train on older maps, test on newer ones. No random K-fold, because rounds within a map are correlated, so split by **match**.
- Score with log loss and Brier, not accuracy.
- Compare against bookmaker map odds if we can get them. Beating Elo is the bar; beating bookmaker odds is the real test.

## Not doing (yet)
- In-play/live prediction. Same model, but it starts from the actual state; easy to add later.
- Player-level telemetry or positional features (PureSkill). Add only if the round model plateaus.
- Veto/map-pick prediction. Ask for the map as input.
