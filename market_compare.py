"""Frozen model vs Polymarket (and Pinnacle) on Jul-Oct 2026 series the model never saw.

Run polymarket.py events first; this fetches (and caches) price history only for series that match a Kaggle match.
Pinnacle closing prices come from data/odds/model_vs_market.csv (market_eval.py), so this makes no OddsPapi requests.
Polymarket closing = last price point before the earlier of Kaggle's start (UTC+3 -> UTC) and Polymarket's start time.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import market_eval as E
import odds
import oos_eval as O
import polymarket as M
import prematch as P

ROOT = O.ROOT
ALIAS = {"fazeclan": "faze", "ninjasinpyjamas": "nip", "1w": "1win"}  # Kaggle key -> Polymarket key


def key(name):
    k = O.key(name)
    return ALIAS.get(k, k)


def series_table():
    """Map 1 row of every Kaggle series from CUT on, with the frozen model's map-1 and series probabilities."""
    v5 = P.load(O.V5)
    X = O.features(v5)
    new = (v5.datetime >= O.CUT).values
    model = pickle.load(open(P.MODEL, "rb"))["logreg"]
    v5["p_model"] = np.nan
    v5.loc[new, "p_model"] = P.sym_proba(model, X[new].fillna(0))
    v5["elo"] = P.elo_p(X.elo.values)
    d = v5[new]
    first = d.groupby("match_id").head(1).set_index("match_id")
    wins, maps = d.groupby("match_id").y.sum(), d.groupby("match_id").y.size()
    bo = first.bestOf.where(first.bestOf == first.bestOf, np.where(maps[first.index] == 1, 1, 3)).astype(int)
    return pd.DataFrame({"team1": first.team1, "team2": first.team2, "date": first.datetime, "bo": bo,
                         "model_map1": first.p_model, "elo_map1": first.elo, "y_map1": first.y,
                         "model_series": [E.series_p(p, b) for p, b in zip(first.p_model, bo)],
                         "elo_series": [E.series_p(p, b) for p, b in zip(first.elo, bo)],
                         "y_series": (wins > maps / 2).astype(int)[first.index]})


def attach_polymarket(t):
    pm = M.load()
    pm["t"] = pd.to_datetime(pm.start, format="ISO8601").dt.tz_localize(None)
    idx = {}
    for r in pm.itertuples():
        idx.setdefault(frozenset((key(r.team1), key(r.team2))), []).append(r)
    cols = {c: [] for c in ("pm_series", "pm_map1", "pm_volume", "pm_tier", "pm_event")}
    for mid, r in t.iterrows():
        start = r.date - pd.Timedelta(hours=3)
        c = [x for x in idx.get(frozenset((key(r.team1), key(r.team2))), []) if abs((x.t - start).total_seconds()) < 36 * 3600]
        e = min(c, key=lambda x: abs((x.t - start).total_seconds())) if c else None
        out = dict.fromkeys(cols)
        if e is not None:
            cutoff = min(start, e.t)
            flip = key(e.team1) != key(r.team1)  # orient to Kaggle team1
            for name in ("series", "map1"):
                tok = getattr(e, name + "_token")
                q = M.closing(M.history(tok, e.start), cutoff) if isinstance(tok, str) else None
                out["pm_" + name] = (1 - q if flip else q) if q is not None else np.nan
            out.update(pm_volume=e.series_volume, pm_tier=e.tier, pm_event=e.slug)
        for k in cols:
            cols[k].append(out[k])
    for k, v in cols.items():
        t[k] = v
    return t


BOOKS = ("bet365", "1xbet", "kalshi")  # one OddsPapi request per fixture (historical-odds takes up to 3 slugs)


def attach_books(t):
    """No-vig closing prices at BOOKS for series already matched to an OddsPapi fixture (same cutoff as Pinnacle)."""
    fx = pd.read_csv(ROOT / "data/odds/fixtures_jul_oct.csv").set_index("fixtureId")
    files = [ROOT / "data/odds/matched_fixtures.csv", ROOT / "data/odds/matched_extra_ok.csv"]
    matched = pd.concat([pd.read_csv(f) for f in files if f.exists()]).drop_duplicates("match_id")
    for b in BOOKS:
        for lv in ("series", "map1"):
            t[f"{b}_{lv}"] = np.nan
    for r in matched.itertuples():
        if r.match_id not in t.index:
            continue
        k = t.loc[r.match_id]
        h = odds.history(r.fixtureId, ",".join(BOOKS))
        start = min(k.date - pd.Timedelta(hours=3), pd.Timestamp(fx.startTime[r.fixtureId]).tz_localize(None))
        cutoff = start.strftime("%Y-%m-%dT%H:%M:%S")
        flip = O.key(fx.participant1Name[r.fixtureId]) != O.key(k.team1)
        for b in BOOKS:
            for lv, mk in (("series", odds.WINNER), ("map1", odds.MAP1)):
                q = odds.closing(h, mk, cutoff, b)
                if q is not None:
                    t.loc[r.match_id, f"{b}_{lv}"] = 1 - q if flip else q
    return t


def scores(s, level, cols):
    y = s["y_" + level].values
    res = pd.DataFrame([{"model": n, "log_loss": E.ll(y, p).mean(), "brier": ((p - y) ** 2).mean(),
                         "acc": ((p > .5) == y).mean()} for n, p in cols.items()]).set_index("model").round(4)
    return y, res


def diff_ci(y, a, b, rng):
    d = E.ll(y, a) - E.ll(y, b)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"


def loo_blend(y, *ps):
    Z = np.column_stack([P.logit(np.clip(p, .01, .99)) for p in ps])
    return np.array([LogisticRegression(C=1.0).fit(np.delete(Z, i, 0), np.delete(y, i)).predict_proba(Z[i:i + 1])[0, 1]
                     for i in range(len(y))])


def main():
    t = attach_polymarket(series_table())
    pin = pd.read_csv(ROOT / "data/odds/model_vs_market.csv").set_index("match_id")
    t["pin_series"], t["pin_map1"] = pin.mkt_series.reindex(t.index), pin.mkt_map1.reindex(t.index)
    t = attach_books(t)
    t.to_csv(ROOT / "data/polymarket/model_vs_markets.csv")
    rng = np.random.default_rng(0)
    print(f"{len(t)} Kaggle series since {O.CUT.date()}; {t.pm_series.notna().sum()} with a Polymarket series price, "
          f"{t.pm_map1.notna().sum()} with a map-1 price, {t.pin_series.notna().sum()} with Pinnacle")
    for level in ("series", "map1"):
        s = t.dropna(subset=["pm_" + level])
        y, res = scores(s, level, {"elo only": s["elo_" + level].values, "frozen model": s["model_" + level].values,
                                   "Polymarket closing": s["pm_" + level].values,
                                   "model + Polymarket (LOO)": loo_blend(s["y_" + level].values, s["pm_" + level].values,
                                                                         s["model_" + level].values)})
        print(f"\n== {level} winner, all Polymarket-matched series, n={len(s)} ==\n{res.to_string()}")
        print("model minus Polymarket:", diff_ci(y, s["model_" + level].values, s["pm_" + level].values, rng))
        for lo, hi, lab in ((0, 2e4, "volume < $20k"), (2e4, 1e5, "$20k-100k"), (1e5, 1e12, "> $100k")):
            g = s[(s.pm_volume >= lo) & (s.pm_volume < hi)]
            if len(g) > 10:
                print(f"  {lab:14s} n={len(g):3d}  model {E.ll(g['y_' + level], g['model_' + level]).mean():.4f}  "
                      f"Polymarket {E.ll(g['y_' + level], g['pm_' + level]).mean():.4f}")
        b = s.dropna(subset=["pin_" + level])
        if len(b) > 20:
            yb = b["y_" + level].values
            cols = {"frozen model": b["model_" + level].values, "Pinnacle closing": b["pin_" + level].values,
                    "Polymarket closing": b["pm_" + level].values,
                    "Pinnacle + Polymarket (LOO)": loo_blend(yb, b["pin_" + level].values, b["pm_" + level].values),
                    "all three (LOO)": loo_blend(yb, b["pin_" + level].values, b["pm_" + level].values,
                                                 b["model_" + level].values)}
            _, res = scores(b, level, cols)
            print(f"\n== {level} winner, series with Pinnacle and Polymarket, n={len(b)} ==\n{res.to_string()}")
            print("Polymarket minus Pinnacle:", diff_ci(yb, cols["Polymarket closing"], cols["Pinnacle closing"], rng))
            print("model minus Polymarket:  ", diff_ci(yb, cols["frozen model"], cols["Polymarket closing"], rng))
            print(f"corr Pinnacle/Polymarket {np.corrcoef(cols['Pinnacle closing'], cols['Polymarket closing'])[0, 1]:.3f}; "
                  f"mean |gap| {np.abs(cols['Pinnacle closing'] - cols['Polymarket closing']).mean():.3f}")

    for level in ("series", "map1"):
        print(f"\n== {level} winner: each source vs Pinnacle on the series both price ==")
        for src in ("model", "pm") + BOOKS:
            b = t.dropna(subset=["pin_" + level, f"{src}_{level}"])
            if len(b) < 20:
                print(f"  {src:8s} n={len(b)} (too few)")
                continue
            y = b["y_" + level].values
            print(f"  {src:8s} n={len(b):3d}  log loss {E.ll(y, b[f'{src}_{level}']).mean():.4f} vs Pinnacle "
                  f"{E.ll(y, b['pin_' + level]).mean():.4f}  diff {diff_ci(y, b[f'{src}_{level}'].values, b['pin_' + level].values, rng)}"
                  f"  acc {((b[f'{src}_{level}'] > .5) == y).mean():.1%}")


if __name__ == "__main__":
    main()
