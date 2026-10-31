"""Hentosh-style player features on top of v1, computed from strictly pre-match data.

Adapted from https://github.com/Hentoshp/CS2_Win_Prediction (MIT License, Copyright (c) 2025 Pierce Hentosh).
That project averages each player's per-map win% and K/R over their last 100 FACEIT games, but fetches those games at
collection time, so the predicted match sits inside its own features. Here the window is the player's last 100 maps
*before* the series, per-map numbers are shrunk toward the player's overall numbers, and evaluation is a time holdout.

    python3 src/hentosh.py eval    # Jul-Oct 2026 holdout vs v1, Polymarket series, leaky variant; trains on < 2026-06-29
    python3 src/hentosh.py train   # refit the better family on all Kaggle v5 data -> models/prematch_hentosh.pkl
"""
import pickle, sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

import market_compare as MC
import market_eval as E
import oos_eval as O
import prematch as P

WINDOW, S_MAP, S_KR, KR0 = 100, 10, 100, 0.68  # shrinkage: 10 maps toward overall win%, 100 rounds toward overall K/R
NEW = ["h_mapwin", "h_kr"]
DIFF = P.DIFF + NEW
COLS = DIFF + P.CTX
MODEL = P.MODEL.with_name("prematch_hentosh.pkl")


class Players:
    def __init__(self):
        self.hist = defaultdict(lambda: deque(maxlen=WINDOW))  # player -> (map, won, kills, rounds)

    def player(self, p, m):
        h = self.hist.get(p)
        if not h:
            return np.nan, np.nan
        a = np.array([(x[0] == m, x[1], x[2], x[3]) for x in h], dtype=float)
        on = a[:, 0] == 1
        win_all = (a[:, 1].sum() + S_MAP * .5) / (len(a) + S_MAP)
        kr_all = (a[:, 2].sum() + S_KR * KR0) / (a[:, 3].sum() + S_KR)
        win = (a[on, 1].sum() + S_MAP * win_all) / (on.sum() + S_MAP)
        kr = (a[on, 2].sum() + S_KR * kr_all) / (a[on, 3].sum() + S_KR)
        return win, kr

    def team(self, lineup, m):
        v = np.array([self.player(p, m) for p in lineup] or [(np.nan, np.nan)])
        if np.isnan(v).all():
            return np.nan, np.nan
        return np.nanmean(v, axis=0)

    def features(self, la, lb, m):
        a, b = self.team(la, m), self.team(lb, m)
        return {"h_mapwin": a[0] - b[0], "h_kr": a[1] - b[1]}

    def update(self, r, la, lb):
        rounds = r.score1_game + r.score2_game
        for t, lineup, won in (("team1", la, r.y), ("team2", lb, 1 - r.y)):
            for i, p in enumerate(lineup, 1):
                k = getattr(r, f"{t}_player{i}_kills")
                if not pd.isna(k):
                    self.hist[p].append((r.map_name, won, k, rounds))


def build(d):
    """(pre, leaky): pre uses state before the series; leaky recomputes after the series' maps are added (Hentosh's setup)."""
    st, pre, leak = Players(), [], []
    for _, g in d.groupby(["datetime", "match_id"], sort=True):
        rs = [(r, *P.lineups(r)) for r in g.itertuples()]
        pre += [st.features(la, lb, r.map_name) for r, la, lb in rs]
        for r, la, lb in rs:
            st.update(r, la, lb)
        leak += [st.features(la, lb, r.map_name) for r, la, lb in rs]
    return pd.DataFrame(pre, index=d.index), pd.DataFrame(leak, index=d.index)


def symmetrize(X, y, diff):
    f = X.copy()
    f[diff] = -f[diff]
    return pd.concat([X, f]), np.concatenate([y, 1 - y])


def sym_proba(model, X, diff):
    f = X.copy()
    f[diff] = -f[diff]
    return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(f)[:, 1]) / 2


def logreg():
    return P.make_pipeline(P.StandardScaler(), P.LogisticRegression(max_iter=1000))


def svm():
    # ponytail: Platt-calibrated RBF SVC on a 6k-row subsample; full 18k symmetrized rows take far longer for no gain
    return P.make_pipeline(P.StandardScaler(), CalibratedClassifierCV(SVC(C=1.0), method="sigmoid", cv=3))


def fit(make, X, y, cols, n=None, seed=0):
    diff = [c for c in cols if c in DIFF]
    Xs, ys = symmetrize(X[cols].fillna(0), y, diff)
    if n and len(Xs) > n:
        i = np.random.default_rng(seed).choice(len(Xs), n, replace=False)
        Xs, ys = Xs.iloc[i], ys[i]
    m = make().fit(Xs, ys)
    return lambda Z: sym_proba(m, Z[cols].fillna(0), diff), m


def data():
    d = P.load(O.V5)
    pre, leak = build(d)
    X = pd.concat([O.features(d), pre], axis=1)[COLS]
    XL = X.copy()
    XL[NEW] = leak[NEW]
    return d, X, XL


def evaluate():
    d, X, XL = data()
    y = d.y.values
    tr = ((d.datetime >= P.BURN_IN_END) & (d.datetime < O.CUT)).values
    te = (d.datetime >= O.CUT).values
    v1 = pickle.load(open(P.MODEL, "rb"))["logreg"]
    preds = {"coin flip": np.full(te.sum(), .5), "elo only": P.elo_p(X.elo.values[te]),
             "v1 frozen (README)": P.sym_proba(v1, X[P.DIFF + P.CTX][te].fillna(0))}
    fns = {}
    for name, make, cols, n in (("v1 refit to cut", logreg, P.DIFF + P.CTX, None),
                                ("hentosh only, logreg", logreg, NEW + P.CTX, None),
                                ("v1+hentosh logreg", logreg, COLS, None),
                                ("v1+hentosh SVM+Platt", svm, COLS, 6000)):
        fns[name] = fit(make, X[tr], y[tr], cols, n)[0]
        preds[name] = fns[name](X[te])
    print(f"== Kaggle v5 holdout {O.CUT.date()} .. {d.datetime.max().date()}, n={te.sum()} maps "
          f"(train {tr.sum()} maps from {P.BURN_IN_END}) ==")
    print(O.table(y[te], preds, d.match_id[te]).to_string())
    print("\nlogreg coefs (scaled):", dict(zip(COLS, fit(logreg, X[tr], y[tr], COLS)[1][-1].coef_[0].round(3))))

    print("\n== Leaky variant: player window includes the match itself (Hentosh's setup) ==")
    leaky = fit(logreg, XL[tr], y[tr], COLS)[0]
    rows = [P.report("leaky, time holdout", y[te], leaky(XL[te]))]
    # Their evaluation: random K-fold over all rows, no time order.
    k, idx = 5, np.random.default_rng(0).permutation(np.flatnonzero(tr | te))
    for name, Z in (("leaky, random 5-fold CV", XL), ("pre-match, random 5-fold CV", X)):
        p = np.empty(len(idx))
        for f in range(k):
            out = np.zeros(len(idx), bool)
            out[f::k] = True
            p[out] = fit(logreg, Z.iloc[idx[~out]], y[idx[~out]], COLS)[0](Z.iloc[idx[out]])
        rows.append(P.report(name, y[idx], p))
    print(pd.DataFrame(rows).set_index("model").round(4).to_string())

    print("\n== Series winner vs Polymarket closing (map-1 probability -> series) ==")
    v5p = {n: np.full(len(d), np.nan) for n in ("v1 frozen", "v1+hentosh logreg", "v1+hentosh SVM+Platt")}
    v5p["v1 frozen"][te] = preds["v1 frozen (README)"]
    for n in ("v1+hentosh logreg", "v1+hentosh SVM+Platt"):
        v5p[n][te] = preds[n]
    t = MC.attach_polymarket(MC.series_table())
    s = t.dropna(subset=["pm_series"])
    first = d[te].assign(**{n: v[te] for n, v in v5p.items()}).groupby("match_id").head(1).set_index("match_id")
    cols = {n: np.array([E.series_p(p, b) for p, b in zip(first.loc[s.index, n], s.bo)]) for n in v5p}
    cols["Polymarket closing"] = s.pm_series.values
    _, res = MC.scores(s, "series", cols)
    print(f"n={len(s)}\n{res.to_string()}")
    rng = np.random.default_rng(0)
    for n in ("v1+hentosh logreg", "v1+hentosh SVM+Platt"):
        print(f"{n} minus v1:        ", MC.diff_ci(s.y_series.values, cols[n], cols["v1 frozen"], rng))
        print(f"{n} minus Polymarket:", MC.diff_ci(s.y_series.values, cols[n], cols["Polymarket closing"], rng))
    return preds, y[te]


def train(family):
    d, X, _ = data()
    tr = (d.datetime >= P.BURN_IN_END).values
    make, n = (svm, 6000) if family == "svm" else (logreg, None)
    _, m = fit(make, X[tr], d.y.values[tr], COLS, n)
    pickle.dump({"model": m, "family": family, "cols": COLS, "diff": DIFF, "history": "kaggle_v5"}, open(MODEL, "wb"))
    print("saved", MODEL, family)


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        train(sys.argv[2] if len(sys.argv) > 2 else "logreg")
    else:
        evaluate()
