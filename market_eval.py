"""Frozen model vs Pinnacle closing odds on Jul-Oct 2026 matches (python3 odds.py ... first; this makes 0 requests).

Pinnacle closing = last price before the scheduled start, vig removed proportionally.
Kaggle datetimes are UTC+3 (checked against Pinnacle/OddsPapi start times: median offset 0 h after shifting).
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import odds
import oos_eval as O
import prematch as P

ROOT = O.ROOT


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def series_p(p, bo):
    """P(win a best-of-`bo` series) if every map is won with probability p."""
    need = bo // 2 + 1
    from math import comb
    return sum(comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def main():
    v5 = P.load(O.V5)
    X = O.features(v5)
    new = (v5.datetime >= O.CUT).values
    model = pickle.load(open(P.MODEL, "rb"))["logreg"]
    v5["p_model"] = np.nan
    v5.loc[new, "p_model"] = P.sym_proba(model, X[new].fillna(0))
    v5["elo"] = P.elo_p(X.elo.values)
    first = v5[new].groupby("match_id").head(1).set_index("match_id")  # map 1 of each series
    wins = v5[new].groupby("match_id").y.sum()
    maps = v5[new].groupby("match_id").y.size()

    fx = pd.read_csv(ROOT / "data/odds/fixtures_jul_oct.csv").set_index("fixtureId")
    rows = []
    files = [ROOT / "data/odds/matched_fixtures.csv", ROOT / "data/odds/matched_extra_ok.csv"]
    matched = pd.concat([pd.read_csv(f) for f in files if f.exists()]).drop_duplicates("match_id")
    for r in matched.itertuples():
        h = odds.history(r.fixtureId)
        k = first.loc[r.match_id]
        # Earlier of Kaggle's start (UTC+3 -> UTC) and OddsPapi's scheduled start, so no in-play prices leak in.
        start = min(k.datetime - pd.Timedelta(hours=3), pd.Timestamp(fx.startTime[r.fixtureId]).tz_localize(None))
        cutoff = start.strftime("%Y-%m-%dT%H:%M:%S")
        flip = O.key(fx.participant1Name[r.fixtureId]) != O.key(k.team1)  # orient market to Kaggle team1
        out = {"match_id": r.match_id, "bo": k.bestOf, "team1": k.team1, "team2": k.team2, "date": k.datetime}
        for name, mk in (("series", odds.WINNER), ("map1", odds.MAP1)):
            q = odds.closing(h, mk, cutoff)
            out["mkt_" + name] = (1 - q if flip else q) if q is not None else np.nan
        out["model_map1"], out["elo_map1"], out["y_map1"] = k.p_model, k.elo, k.y
        bo = int(k.bestOf) if k.bestOf == k.bestOf else (1 if maps[r.match_id] == 1 else 3)  # a few rows lack bestOf
        out["bo"] = bo
        out["model_series"] = series_p(k.p_model, bo)
        out["elo_series"] = series_p(k.elo, bo)
        out["y_series"] = int(wins[r.match_id] > maps[r.match_id] / 2)
        rows.append(out)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "data/odds/model_vs_market.csv", index=False)

    rng = np.random.default_rng(0)
    for level in ("map1", "series"):
        s = d.dropna(subset=["mkt_" + level])
        y = s["y_" + level].values
        cols = {"coin flip": np.full(len(s), .5), "elo only": s["elo_" + level].values,
                "frozen model": s["model_" + level].values, "Pinnacle closing": s["mkt_" + level].values}
        # Does the model add anything on top of the market? Leave-one-out logistic blend on the two logits.
        Z = np.c_[P.logit(cols["Pinnacle closing"]), P.logit(cols["frozen model"])]
        blend = np.array([LogisticRegression(C=1.0).fit(np.delete(Z, i, 0), np.delete(y, i)).predict_proba(Z[i:i + 1])[0, 1]
                          for i in range(len(y))])
        cols["blend (LOO)"] = blend
        res = pd.DataFrame([{"model": n, "log_loss": ll(y, p).mean(), "brier": ((p - y) ** 2).mean(),
                             "acc": ((p > .5) == y).mean()} for n, p in cols.items()]).set_index("model").round(4)
        diff = ll(y, cols["frozen model"]) - ll(y, cols["Pinnacle closing"])
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        print(f"\n== {level} winner, n={len(s)} ==\n{res.to_string()}")
        print(f"model minus market log loss: {diff.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, "
              f"{np.percentile(bs, 97.5):+.4f}]  (positive = market better)")
        print(f"agree on favourite: {((cols['frozen model'] > .5) == (cols['Pinnacle closing'] > .5)).mean():.1%}; "
              f"corr of probabilities: {np.corrcoef(cols['frozen model'], cols['Pinnacle closing'])[0, 1]:.2f}")
        conf = np.maximum(cols["Pinnacle closing"], 1 - cols["Pinnacle closing"]) >= .7
        print(f"market's own accuracy on its 70%+ favourites: {((cols['Pinnacle closing'][conf] > .5) == y[conf]).mean():.1%}"
              f" (n={conf.sum()})")


if __name__ == "__main__":
    main()
