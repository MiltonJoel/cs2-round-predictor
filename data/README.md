# cloud-data

Inputs for `python3 sim.py eval --data-dir data --sims 10000 --out results/`. Everything here was derived on milton's PC:

- `demos/rounds.csv`, `demos/players.csv`: parsed from 122 CS2 demos (113 complete maps), produced by `demos.py`.
- `demos/demo_matches.csv`: maps each demo to the frozen-v1 pre-match strength, for reference. `sim.py` recomputes it.
- `kaggle_v5/`: Kaggle `ektarr/counter-strike-pro-matches` v5 (CC BY-SA).
- `liquipedia/maps_2026.csv`, `liquipedia/transfers.csv`: Liquipedia results and transfers (CC BY-SA), produced by `liquipedia.py` and `rosters.py`.
- `models/prematch.pkl`: frozen v1 pre-match model (trained on data through 2026-06-28).

There are no .dem files and no raw API caches.
