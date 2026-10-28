"""Download new data: python3 src/update.py  (or add --update to oos_eval.py commands).

- Kaggle ektarr/counter-strike-pro-matches: re-download via kagglehub; copied to data/kaggle_v5/ only if the version changed
  (every version also stays in ~/.cache/kagglehub/.../versions/N, so old pulls remain reproducible).
- Liquipedia: re-fetch tournaments dated in the current and previous month (cache bypassed), merge into maps_2026.csv.
Odds are not refreshed here (OddsPapi free plan has a 250 request/month budget); run odds.py by hand.
"""
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / "data/kaggle_v5"
LP = ROOT / "data/liquipedia/maps_2026.csv"


def kaggle():
    import kagglehub
    src = Path(kagglehub.dataset_download("ektarr/counter-strike-pro-matches"))
    stamp = KAGGLE / "VERSION"
    if stamp.exists() and stamp.read_text() == src.name:
        print(f"kaggle: already at {src.name}")
        return
    KAGGLE.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        shutil.copy2(f, KAGGLE / f.name)
    stamp.write_text(src.name)
    d = pd.read_csv(KAGGLE / "cs2_all_tiers_games.csv", usecols=["datetime"])
    print(f"kaggle: updated to {src.name}, {len(d)} rows, last {d.datetime.max()}")


def liquipedia(months_back=1):
    import liquipedia as L
    today = date.today()
    months = [f"{y}-{m:02d}" for y, m in
              ((today.year + (today.month - 1 - k) // 12, (today.month - 1 - k) % 12 + 1) for k in range(months_back, -1, -1))]
    L.FRESH = True
    try:
        new = L.collect(months)
    finally:
        L.FRESH = False
    old = pd.read_csv(LP, parse_dates=["datetime"]) if LP.exists() else new.iloc[:0]
    # Union, newest wins: a 2-month search misses long tournaments that started earlier, so never drop old rows.
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(
        ["datetime", "team1_lp", "team2_lp", "map_no"], keep="last")
    out.sort_values(["datetime", "team1", "map_no"]).to_csv(LP, index=False)
    print(f"liquipedia: refreshed {months}, {len(out) - len(old):+d} maps, last {out.datetime.max()}")


def staleness(max_days=21):
    """Player-stat features only come from Kaggle; warn if it has stopped updating."""
    last = pd.to_datetime(pd.read_csv(KAGGLE / "cs2_all_tiers_games.csv", usecols=["datetime"]).datetime).max()
    age = (pd.Timestamp.now() - last).days
    if age > max_days:
        print(f"WARNING: newest Kaggle match (with player stats) is {age} days old; "
              "lineup K/D/ADR/KAST features are stale and the model is drifting toward Elo+form.")


def run():
    for step in (kaggle, liquipedia, staleness):
        try:
            step()
        except Exception as e:  # a failed source must not block predicting from what we already have
            print(f"{step.__name__}: update failed ({e}); using cached data")


if __name__ == "__main__":
    run()
