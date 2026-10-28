# CounterPredictor

Pre-match win probabilities for professional Counter-Strike 2, built from past results, player stats and demo telemetry.

Given two teams, a map and a date, the model replays every earlier map in order, builds each team's state at that moment (Elo, per-map Elo, recent form, lineup stats, roster changes, head-to-head) and returns P(team A wins). A round-level simulator built from parsed demos adds scorelines, pistol-round odds and per-round odds on top.

## Results

All numbers are on matches the models never saw during training. Full tables and method notes are in [docs/results.md](docs/results.md).

| Check | Sample | Result |
|---|---|---|
| Map winner, v1 logistic regression | 1,861 maps, Jan–Jun 2026 | log loss 0.657 vs Elo-only 0.669, 60% accuracy |
| Map winner, frozen v1 on later data | Jul–Oct 2026 | log loss 0.658, 62% accuracy, still calibrated |
| Series winner vs Pinnacle closing odds | 184 series | Pinnacle 0.584, model 0.602 |
| Series winner vs Polymarket | 509 series | Polymarket 0.589, model 0.625 |
| Round model from demos | 229 maps | 64% round accuracy, log loss 0.598 |

The honest summary: the model is well calibrated and clearly beats Elo, but the pre-match market price is still the better single number. Adding the model to the market price does not improve it.

## Project layout

```
src/            all code (run each script from the repository root)
tests/          leakage and symmetry checks
docs/           results and experiment log, original plan, dataset notes
results/        saved simulator evaluation runs
data/           local data, not tracked by git (Kaggle, Liquipedia, demos, odds, prediction log)
models/         trained models, not tracked by git
```

| Area | Scripts |
|---|---|
| Pre-match models | `prematch.py` (v1), `oos_eval.py` (v2, out-of-sample checks), `v3.py`, `v4.py`, `recency.py` |
| Data sources | `update.py`, `liquipedia.py`, `rosters.py`, `lineup.py`, `hltv_import.py` |
| Demos and simulation | `demos.py`, `import_demos.py`, `round_model.py`, `sim.py` |
| Markets | `odds.py`, `polymarket.py`, `market_eval.py`, `market_compare.py` |

## Setup

```
pip install -r requirements.txt
```

Data goes under `data/`, which git ignores:
- Kaggle [`ektarr/counter-strike-pro-matches`](https://www.kaggle.com/datasets/ektarr/counter-strike-pro-matches) in `data/raw/` (v1) and `data/kaggle_v5/` (newer models). `python3 src/update.py` downloads it and refreshes Liquipedia results.
- Liquipedia results, cached in `data/liquipedia/` by `src/liquipedia.py` through the official MediaWiki API.
- HLTV pages and demos saved by hand in `data/hltv_saved/`. Nothing here scrapes HLTV.

## Usage

```
python3 src/prematch.py train                                                  # train v1 -> models/prematch.pkl (~2 min)
python3 src/prematch.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-07
python3 src/prematch.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-07 --polymarket   # also show the market price
python3 src/v4.py predict "Team Vitality" "Team Spirit" 2026-10-07 Mirage      # v4: expected rosters from transfers
python3 src/sim.py "Team Vitality" "Team Spirit" Mirage 2026-10-07 --ct A      # scorelines, pistols, per-round odds
python3 src/import_demos.py --downloads ~/Downloads                            # import newly downloaded demos
python3 tests/test_prematch.py                                                 # leakage + symmetry checks
```

Add `--update` to an `oos_eval.py`, `v3.py` or `v4.py` command to download fresh Kaggle and Liquipedia data first. Every prediction is appended to `data/predictions.csv` before the match is played, so the forward track record can't be edited after the fact.

## Which model to use

- **Map winner:** v1 (`prematch.py`) is the default. v4 is marginally better after roster changes.
- **Series without a known map:** v3's veto simulation.
- **Scorelines and round-level odds:** the simulator (`sim.py`). It does not pick winners better than v1.
- **Betting decisions:** the Polymarket or Pinnacle price is the stronger estimate; use the model as a second opinion.

## Data notes

- In the Kaggle data, `team1_win` is the series winner repeated on every map row. The map winner comes from the round score, the series winner from the `is_total` row.
- Kaggle team ids are per match, so teams are keyed by name. Player ids are stable.
- Maps within a series are sometimes missing or out of order in Kaggle; Liquipedia and demos fill the order where needed.

## License

The code is released under the [MIT License](LICENSE).
