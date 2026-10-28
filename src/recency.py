"""v1 with recency: recent maps count more, old results fade.

Three knobs, all chosen on 2026H1 validation and then scored once on the Jun 29+ maps v1 never saw:
- weight half-life: a training map that is H days older than the newest one counts 0.5x (sample weight)
- Elo half-life: a team's Elo (overall and per map) drifts back toward 1500 while it doesn't play
- h2h window: head-to-head only counts meetings from the last N days

    python3 src/recency.py eval     # tune on validation, score on Jun 29+ (Kaggle + Pinnacle), print tables
    python3 src/recency.py train    # refit the chosen setup on every map -> models/prematch_v1_recency.pkl (v1 kept)
    python3 src/recency.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-20
"""
import itertools, pickle, sys
from collections import defaultdict

import numpy as np
import pandas as pd

import market_eval as M
import oos_eval as O
import prematch as P

MODEL = P.MODEL.with_name("prematch_v1_recency.pkl")  # v1 recipe + chosen recency, refit on all data
VAL_END = O.CUT  # validation 2026-01-01 .. 2026-06-28, test from 2026-06-29
WEIGHT_HL = [None, 90, 180, 365, 730]
ELO_HL = [None, 180, 365]
H2H_DAYS = [None, 365]


class RState(P.State):
    def __init__(self, elo_hl=None, h2h_days=None):
        super().__init__()
        self.elo_hl, self.h2h_days = elo_hl, h2h_days
        self.t = {}  # Elo key -> time of its last update
        self.h2h_t = defaultdict(list)  # (a, b) sorted -> meeting times, parallel to self.h2h

    def _decayed(self, store, key, when):
        if self.elo_hl is None or key not in self.t:
            return store[key]
        return 1500 + (store[key] - 1500) * 0.5 ** ((when - self.t[key]).days / self.elo_hl)

    def side(self, t, m, lineup, when):
        s = super().side(t, m, lineup, when)
        s["elo"], s["map_elo"] = self._decayed(self.elo, t, when), self._decayed(self.map_elo, (t, m), when)
        return s

    def features(self, a, b, m, when, la=None, lb=None):
        f = super().features(a, b, m, when, la, lb)
        if self.h2h_days is not None:
            k = tuple(sorted((a, b)))
            h = [w for w, t in zip(self.h2h[k], self.h2h_t[k]) if (when - t).days <= self.h2h_days]
            f["h2h_n"] = np.log1p(len(h))
            f["h2h"] = (np.mean([w == a for w in h]) - 0.5) if h else 0.0
        return f

    def update(self, r, la, lb):
        a, b, m = r.team1_id, r.team2_id, r.map_name
        for store, key in ((self.elo, a), (self.elo, b), (self.map_elo, (a, m)), (self.map_elo, (b, m))):
            store[key] = self._decayed(store, key, r.datetime)
            self.t[key] = r.datetime
        self.h2h_t[tuple(sorted((a, b)))].append(r.datetime)
        super().update(r, la, lb)


def features(d, elo_hl=None, h2h_days=None):
    _, F = P.build(d, state=RState(elo_hl, h2h_days))
    return F.assign(tier=d.tier.fillna(2), bestOf=d.bestOf)[P.DIFF + P.CTX]


def fit(X, y, when, weight_hl):
    Xs, ys = P.symmetrize(X, y)
    w = np.ones(len(X)) if weight_hl is None else 0.5 ** ((when.max() - when).dt.days.values / weight_hl)
    model = P.make_pipeline(P.StandardScaler(), P.LogisticRegression(max_iter=1000))
    return model.fit(Xs.fillna(0), ys, logisticregression__sample_weight=np.r_[w, w])


def ll(y, p):
    return P.log_loss(y, p, labels=[0, 1])


def evaluate():
    d = P.load(O.V5)
    print(f"Kaggle maps: {len(d)} ({d.datetime.min():%Y-%m-%d} .. {d.datetime.max():%Y-%m-%d})")
    y, when = d.y.values, d.datetime
    tr = ((when >= P.BURN_IN_END) & (when < P.TEST_START)).values
    va = ((when >= P.TEST_START) & (when < VAL_END)).values
    te = (when >= VAL_END).values
    tier1 = (d.tier == 1).values

    # 1) tune on validation: every feature variant x every weight half-life
    feats, rows = {}, []
    for elo_hl, h2h in itertools.product(ELO_HL, H2H_DAYS):
        X = feats[elo_hl, h2h] = features(d, elo_hl, h2h)
        for whl in WEIGHT_HL:
            p = P.sym_proba(fit(X[tr], y[tr], when[tr], whl), X[va].fillna(0))
            rows.append({"elo_hl": elo_hl, "h2h_days": h2h, "weight_hl": whl, "val_ll": ll(y[va], p),
                         "val_ll_tier1": ll(y[va & tier1], p[tier1[va]]), "val_acc": ((p > .5) == y[va]).mean()})
    grid = pd.DataFrame(rows).sort_values("val_ll")
    print(f"\n== Validation 2026-01-01 .. {VAL_END:%Y-%m-%d} (train {tr.sum()} maps, val {va.sum()}) ==")
    print(grid.round(4).to_string(index=False))
    best = grid.iloc[0]
    cfg = {k: (None if pd.isna(best[k]) else int(best[k])) for k in ("elo_hl", "h2h_days", "weight_hl")}
    print("chosen:", cfg)

    # 2) test once: refit through Jun 28 and score the maps nobody tuned on
    trv = tr | va
    Xb, Xr = feats[None, None], feats[cfg["elo_hl"], cfg["h2h_days"]]
    frozen = pickle.load(open(P.MODEL, "rb"))["logreg"]
    preds = {
        "frozen v1 (trained to 2025-06)": P.sym_proba(frozen, Xb[te].fillna(0)),
        "v1 retrained to 2026-06, no recency": P.sym_proba(fit(Xb[trv], y[trv], when[trv], None), Xb[te].fillna(0)),
        "v1 retrained, weights only": P.sym_proba(fit(Xb[trv], y[trv], when[trv], cfg["weight_hl"]), Xb[te].fillna(0)),
        "v1 retrained, recency (chosen)": P.sym_proba(fit(Xr[trv], y[trv], when[trv], cfg["weight_hl"]),
                                                      Xr[te].fillna(0)),
    }
    # For reference only (not picked by validation): the strongest recency setting in the grid.
    Xs = feats[min(ELO_HL[1:]), H2H_DAYS[-1]]
    preds["strong recency, for reference"] = P.sym_proba(fit(Xs[trv], y[trv], when[trv], min(WEIGHT_HL[1:])),
                                                         Xs[te].fillna(0))
    res = []
    for name, p in preds.items():
        for sub, m in (("all", np.ones(te.sum(), bool)), ("tier 1", tier1[te])):
            res.append({"model": name, "maps": sub, "n": m.sum(), "log_loss": ll(y[te][m], p[m]),
                        "ll_95ci": "[%.3f, %.3f]" % tuple(O.boot(y[te][m], p[m], d.match_id[te][m])),
                        "acc": ((p[m] > .5) == y[te][m]).mean()})
    print(f"\n== Test: Kaggle maps {VAL_END:%Y-%m-%d} .. {d.datetime.max():%Y-%m-%d} ==")
    print(pd.DataFrame(res).round(4).to_string(index=False))
    base, new = preds["v1 retrained to 2026-06, no recency"], preds["v1 retrained, recency (chosen)"]
    diff = M.ll(y[te], new) - M.ll(y[te], base)
    g = pd.factorize(d.match_id[te])[0]
    rng = np.random.default_rng(0)
    bs = [diff[np.isin(g, rng.integers(0, g.max() + 1, g.max() + 1))].mean() for _ in range(1000)]
    print(f"recency minus no-recency log loss: {diff.mean():+.4f} (match bootstrap 95% CI "
          f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]; negative = recency better)")

    # 3) markets: Pinnacle (market_eval.py, 184 series) and Polymarket (market_compare.py, ~509 series), if present
    first = d[te].assign(**{k: v for k, v in preds.items()}).groupby("match_id").head(1).set_index("match_id")
    # Series label from Kaggle's own series result (team1_win), not from counting maps: Kaggle sometimes lacks map 3.
    official = d[te].groupby("match_id").team1_win.first().astype(int)
    for label, f, pre in (("Pinnacle closing", "data/odds/model_vs_market.csv", "mkt_"),
                          ("Polymarket closing", "data/polymarket/model_vs_markets.csv", "pm_")):
        if not (O.ROOT / f).exists():
            continue
        mk = pd.read_csv(O.ROOT / f)
        mk = mk[mk.match_id.isin(first.index)].assign(y_series=lambda x: official[x.match_id].values)
        print(f"\n== vs {label} (series with prices: map 1 {mk[pre + 'map1'].notna().sum()}, "
              f"series {mk[pre + 'series'].notna().sum()}) ==")
        out = []
        for name in [label] + list(preds):
            for lvl in ("map1", "series"):
                s = mk.dropna(subset=[pre + lvl])
                if name == label:
                    p = s[pre + lvl].values
                else:
                    p1 = first.loc[s.match_id, name].values
                    p = p1 if lvl == "map1" else np.array([M.series_p(q, int(b)) for q, b in zip(p1, s.bo)])
                out.append({"model": name, "level": lvl, "log_loss": M.ll(s["y_" + lvl].values, p).mean(),
                            "acc": ((p > .5) == s["y_" + lvl].values).mean()})
        print(pd.DataFrame(out).pivot(index="model", columns="level", values=["log_loss", "acc"]).round(4).to_string())
    return cfg


CHOSEN = {"elo_hl": None, "h2h_days": None, "weight_hl": None}  # `eval` 2026-10-18: no recency setting beat none


def train(cfg=CHOSEN):
    d = P.load(O.V5)
    X = features(d, cfg["elo_hl"], cfg["h2h_days"])
    tr = (d.datetime >= P.BURN_IN_END).values
    model = fit(X[tr], d.y.values[tr], d.datetime[tr], cfg["weight_hl"])
    pickle.dump({"logreg": model, "best": "logreg", "cfg": cfg, "data_last": d.datetime.max()}, open(MODEL, "wb"))
    print("saved", MODEL, cfg, "coefs:", dict(zip(P.DIFF + P.CTX, model[-1].coef_[0].round(3))))


def predict(team_a, team_b, map_name, date, tier=1, best_of=3):
    d, date = P.load(O.V5), pd.Timestamp(date)
    m = pickle.load(open(MODEL, "rb"))
    st, _ = P.build(d, until=date, state=RState(m["cfg"]["elo_hl"], m["cfg"]["h2h_days"]))
    known = {t.lower(): t for t in pd.concat([d.team1, d.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    X = pd.DataFrame([st.features(a, b, map_name, date)]).assign(tier=tier, bestOf=best_of)[P.DIFF + P.CTX]
    p = float(P.sym_proba(m["logreg"], X.fillna(0))[0])
    return {team_a: p, team_b: 1 - p}


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate()
    elif sys.argv[1] == "train":
        train()
    else:
        print(predict(*sys.argv[2:6]))
