"""v3: player-level Elo + lineup continuity + map-order/comfort features + veto simulation for pre-veto series odds.

Liquipedia's free wikitext has no pick/ban data (0 mapveto fields in 28k matches), so the veto is *inferred*:
map order (map 1/2 are picks, the last map of a Bo3/Bo5 is the decider) and each team's recent map share.

    python3 src/v3.py eval [--kaggle-only]  # ablation on Jul-Oct 2026 (Kaggle, Liquipedia) + Pinnacle comparison; trains on < 2026-06-29
    python3 src/v3.py train     # fit on all data -> models/prematch_v3.pkl (v1/v2 untouched)
    python3 src/v3.py predict "Team Vitality" "Team Spirit" 2026-10-07 [MAP] [--bo 3] [--update]
"""
import pickle, sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

import oos_eval as O
import prematch as P

K_PLAYER, CONT_N, SHARE_N = 32, 5, 30
NEW = ["pelo", "pmap", "cont", "share", "share_pick", "pool"]
GROUPS = {"base": [], "+player elo": ["pelo", "pmap"], "+continuity": ["cont"],
          "+map order/comfort": ["share", "share_pick", "pool", "map_no", "decider"]}
V3 = P.MODEL.with_name("prematch_v3.pkl")


class State3(P.State):
    def __init__(self):
        super().__init__()
        self.pelo = {}  # player -> rating
        self.pmap = {}  # (player, map) -> rating
        self.recent = defaultdict(lambda: deque(maxlen=CONT_N))  # team -> last lineups
        self.maps = defaultdict(lambda: deque(maxlen=SHARE_N))  # team -> last map names

    def _rating(self, store, key, mates):
        # ponytail: a newcomer starts at his teammates' average minus 25 (stand-ins are usually a bit weaker)
        if key not in store:
            known = [store[k] for k in mates if k in store]
            store[key] = (np.mean(known) - 25) if known else 1500.0
        return store[key]

    def strength(self, lineup, m):
        if not lineup:
            return np.nan, np.nan
        e = [self._rating(self.pelo, p, lineup) for p in lineup]
        pm = [self._rating(self.pmap, (p, m), [(q, m) for q in lineup]) for p in lineup]
        return np.mean(e), np.mean(pm)

    def side3(self, t, m, lineup):
        e, pm = self.strength(lineup, m)
        seen = set().union(*self.recent[t]) if self.recent[t] else set()
        ms = list(self.maps[t])
        return {"pelo": e, "pmap": pm, "cont": len(set(lineup) & seen) / 5 if seen else np.nan,
                "share": ms.count(m) / len(ms) if ms else np.nan, "pool": len(set(ms)) if ms else np.nan}

    def features3(self, a, b, m, when, la=None, lb=None, map_no=1, bo=3):
        la = la or self.lineup.get(a, [])
        lb = lb or self.lineup.get(b, [])
        f = self.features(a, b, m, when, la, lb)
        sa, sb = self.side3(a, m, la), self.side3(b, m, lb)
        f.update({k: sa[k] - sb[k] for k in sa})
        decider = int(map_no == bo or bo == 1)
        f["share_pick"] = f["share"] * (1 - decider) if f["share"] == f["share"] else np.nan
        f["map_no"], f["decider"] = map_no, decider
        return f

    def update3(self, r, la, lb):
        la = la or self.lineup.get(r.team1_id, [])
        lb = lb or self.lineup.get(r.team2_id, [])
        if la and lb:
            for store, keys_a, keys_b in ((self.pelo, la, lb),
                                          (self.pmap, [(p, r.map_name) for p in la], [(p, r.map_name) for p in lb])):
                ea = np.mean([self._rating(store, k, keys_a) for k in keys_a])
                eb = np.mean([self._rating(store, k, keys_b) for k in keys_b])
                d = K_PLAYER * (r.y - P.elo_p(ea - eb))
                for k in keys_a:
                    store[k] += d
                for k in keys_b:
                    store[k] -= d
        for t, lu in ((r.team1_id, la), (r.team2_id, lb)):
            if lu:
                self.recent[t].append(set(lu))
            self.maps[t].append(r.map_name)
        self.update(r, *P.lineups(r))


def pool_at(hist_maps):
    """Active map pool = 7 most played maps in the recent window."""
    return list(pd.Series(hist_maps).value_counts().index[:7])


def veto_series(pmap, bo, first=0):
    """Greedy veto: each team bans its worst remaining map, picks its best. pmap: {map: P(A wins)}.
    Returns (P(A wins series), maps in play order). Team A starts the veto (first=0)."""
    left = dict(pmap)
    order = {1: "bbbbbb", 3: "bbppbb", 5: "bbpppp"}[bo]
    played = []
    for i, step in enumerate(order):
        a_turn = (i % 2 == 0) == (first == 0)
        best_for_a = max(left, key=left.get)
        worst_for_a = min(left, key=left.get)
        pick = (worst_for_a if a_turn else best_for_a) if step == "b" else (best_for_a if a_turn else worst_for_a)
        if step == "p":
            played.append(pick)
        left.pop(pick)
        if len(left) == 1:
            break
    played += list(left)[:bo - len(played)]
    ps = [pmap[m] for m in played]
    need = bo // 2 + 1  # exact series probability for the fixed map order
    def win(i, wa, wb):
        if wa == need:
            return 1.0
        if wb == need:
            return 0.0
        return ps[i] * win(i + 1, wa + 1, wb) + (1 - ps[i]) * win(i + 1, wa, wb + 1)
    return win(0, 0, 0), played


def build3(d, until=None, simulate=()):
    """Like prematch.build, with v3 features. For match_ids in `simulate`, also returns per-pool-map feature rows."""
    st, rows, sims = State3(), [], {}
    simulate = set(simulate)
    recent_maps = deque(maxlen=600)
    for mid, g in d.groupby(["datetime", "match_id"], sort=True):
        if until is not None and g.datetime.iloc[0] >= until:
            break
        rs = list(g.itertuples())
        for k, r in enumerate(rs, 1):
            la, lb = P.lineups(r)
            bo = int(r.bestOf) if r.bestOf == r.bestOf else len(rs)
            rows.append(st.features3(r.team1_id, r.team2_id, r.map_name, r.datetime, la, lb, k, bo))
        if mid[1] in simulate:
            r = rs[0]
            la, lb = P.lineups(r)
            bo = int(r.bestOf) if r.bestOf == r.bestOf else len(rs)
            sims[mid[1]] = {m: st.features3(r.team1_id, r.team2_id, m, r.datetime, la, lb, 1, bo)
                            for m in pool_at(recent_maps)}
        for r in rs:
            st.update3(r, *P.lineups(r))
            recent_maps.append(r.map_name)
    return st, pd.DataFrame(rows, index=d.index[:len(rows)]), sims


def cols(group_names):
    extra = [c for g in group_names for c in GROUPS[g]]
    diff = P.DIFF + [c for c in extra if c in NEW]
    ctx = P.CTX + [c for c in extra if c not in NEW]
    return diff, ctx


def fit(X, y, diff, ctx):
    flip = X.copy()
    flip[diff] = -flip[diff]
    Xs, ys = pd.concat([X, flip])[diff + ctx], np.r_[y, 1 - y]
    return P.make_pipeline(P.StandardScaler(), P.LogisticRegression(max_iter=2000)).fit(Xs.fillna(0), ys)


def proba(m, X, diff, ctx):
    flip = X.copy()
    flip[diff] = -flip[diff]
    return (m.predict_proba(X[diff + ctx].fillna(0))[:, 1] + 1 - m.predict_proba(flip[diff + ctx].fillna(0))[:, 1]) / 2


def matrix(hist, F):
    return F.assign(tier=hist.tier.fillna(2), bestOf=hist.bestOf)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def diff_ci(y, p, q, groups, n=2000):
    """Mean log-loss difference p - q with a match-clustered bootstrap 95% CI (negative = p better)."""
    d = ll(y, p) - ll(y, q)
    g = pd.factorize(groups)[0]
    s = pd.Series(d).groupby(g).agg(["sum", "size"]).values
    rng = np.random.default_rng(0)
    bs = [(lambda i: s[i, 0].sum() / s[i, 1].sum())(rng.integers(0, len(s), len(s))) for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def evaluate(kaggle_only=False):
    hist = P.load(O.V5) if kaggle_only else O.v2_history()
    mvm = pd.read_csv(O.ROOT / "data/odds/model_vs_market.csv")
    _, F, sims = build3(hist, simulate=set(mvm.match_id))
    X, y = matrix(hist, F), hist.y.values
    tr = ((hist.datetime >= P.BURN_IN_END) & (hist.datetime < O.CUT)).values
    kg = set(P.load(O.V5).match_id)
    known = set(hist.team1[hist.match_id.isin(kg)]) | set(hist.team2[hist.match_id.isin(kg)])
    te_kg = ((hist.datetime >= O.CUT) & hist.match_id.isin(kg)).values
    te_lp = ((hist.datetime >= O.CUT) & ~hist.match_id.isin(kg) & hist.team1.isin(known) & hist.team2.isin(known)).values

    ladder = [["base"], ["base", "+player elo"], ["base", "+continuity"], ["base", "+map order/comfort"],
              ["base", "+player elo", "+continuity", "+map order/comfort"]]
    models, preds = {}, {}
    for g in ladder:
        name = " ".join(g) if len(g) < 4 else "all (v3)"
        diff, ctx = cols(g)
        models[name] = (fit(X[tr], y[tr], diff, ctx), diff, ctx)
        preds[name] = proba(models[name][0], X, diff, ctx)
    print(f"train {tr.sum()} maps (< {O.CUT:%Y-%m-%d}); test Kaggle {te_kg.sum()} maps, Liquipedia {te_lp.sum()} maps")
    for label, te in (("Kaggle Jul-Oct", te_kg), ("Liquipedia Jul-Oct (teams known)", te_lp)):
        print(f"\n== {label} ==")
        for name, p in preds.items():
            m, lo, hi = diff_ci(y[te], p[te], preds["base"][te], hist.match_id[te])
            acc = ((p[te] > .5) == y[te]).mean()
            print(f"{name:45s} log loss {ll(y[te], p[te]).mean():.4f}  acc {acc:.1%}  vs base {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # Pinnacle: same 184 series. Map 1 (map known) and series (pre-veto: simulated veto vs map-1-known shortcut).
    first = hist[hist.match_id.isin(mvm.match_id)].groupby("match_id").head(1)
    idx = dict(zip(first.match_id, first.index))
    out = []
    for r in mvm.itertuples():
        i = idx.get(r.match_id)
        if i is None:
            continue
        row = {"match_id": r.match_id, "y_map1": r.y_map1, "y_series": r.y_series, "mkt_map1": r.mkt_map1,
               "mkt_series": r.mkt_series}
        bo = int(r.bo)
        for name in ("base", "all (v3)"):
            m, diff, ctx = models[name]
            p1 = preds[name][hist.index.get_loc(i)]
            row[f"{name}|map1"] = p1
            row[f"{name}|series map1-known"] = _series_from_p(p1, bo)
            S = sims[r.match_id]
            tier = hist.tier[i] if hist.tier[i] == hist.tier[i] else 2
            pm = {mp: float(proba(m, pd.DataFrame([S[mp]]).assign(tier=tier, bestOf=bo), diff, ctx)[0]) for mp in S}
            row[f"{name}|series veto-sim"] = veto_series(pm, bo)[0]
        out.append(row)
    s = pd.DataFrame(out)
    print(f"\n== vs Pinnacle closing, {len(s)} series ==")
    for level, keys in (("map1", ["base|map1", "all (v3)|map1"]),
                        ("series", ["base|series map1-known", "base|series veto-sim", "all (v3)|series map1-known",
                                    "all (v3)|series veto-sim"])):
        t = s.dropna(subset=["mkt_" + level])
        yy = t["y_" + level].values
        print(f"-- {level} winner, n={len(t)}: Pinnacle log loss {ll(yy, t['mkt_' + level].values).mean():.4f}, "
              f"acc {((t['mkt_' + level] > .5) == yy).mean():.1%}")
        for k in keys:
            m, lo, hi = diff_ci(yy, t[k].values, t["mkt_" + level].values, t.match_id)
            print(f"   {k:32s} log loss {ll(yy, t[k].values).mean():.4f}  acc {((t[k] > .5) == yy).mean():.1%}"
                  f"  minus Pinnacle {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def _series_from_p(p, bo):
    from math import comb
    need = bo // 2 + 1
    return sum(comb(need - 1 + k, k) * p ** need * (1 - p) ** k for k in range(need))


def train():
    hist = O.v2_history()
    _, F, _ = build3(hist)
    X, y = matrix(hist, F), hist.y.values
    tr = (hist.datetime >= P.BURN_IN_END).values
    diff, ctx = cols(list(GROUPS))
    pickle.dump({"logreg": fit(X[tr], y[tr], diff, ctx), "diff": diff, "ctx": ctx}, open(V3, "wb"))
    print("saved", V3)


def predict(team_a, team_b, date, map_name=None, bo=3):
    hist, date = O.v2_history(), pd.Timestamp(date)
    st, _, _ = build3(hist, until=date)
    m = pickle.load(open(V3, "rb"))
    known = {t.lower(): t for t in pd.concat([hist.team1, hist.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    pool = pool_at(hist[hist.datetime < date].map_name.tail(600))
    def p(mp, k):
        f = pd.DataFrame([st.features3(a, b, mp, date, map_no=k, bo=bo)]).assign(tier=1, bestOf=bo)
        return float(proba(m["logreg"], f, m["diff"], m["ctx"])[0])
    if map_name:
        return {"map": map_name, team_a: p(map_name, 1), team_b: 1 - p(map_name, 1)}
    pm = {mp: p(mp, 1) for mp in pool}
    s, played = veto_series(pm, bo)
    return {"series": {team_a: s, team_b: 1 - s}, "likely maps": played,
            "per map": {k: round(v, 3) for k, v in sorted(pm.items(), key=lambda x: -x[1])}}


if __name__ == "__main__":
    pm_flag = "--polymarket" in sys.argv  # also print the Polymarket price for this exact match
    if pm_flag:
        sys.argv.remove("--polymarket")
    args = sys.argv[1:]
    if "--update" in args:
        args.remove("--update")
        import update
        update.run()
    bo = 3
    if "--bo" in args:
        i = args.index("--bo")
        bo = int(args[i + 1])
        del args[i:i + 2]
    if args[0] == "eval":
        evaluate(kaggle_only="--kaggle-only" in args)
    elif args[0] == "train":
        train()
    else:
        print(predict(*args[1:5], bo=bo))
        if pm_flag:
            import polymarket
            polymarket.show(args[1], args[2], args[3])
