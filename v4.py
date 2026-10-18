"""v4: expected roster from Liquipedia transfers (announced before the match) instead of the lineup actually played.

Lineup policies compared (same features, same splits):
  actual   - the 5 who actually played (from Kaggle; only known once the match starts)
  last     - the team's previous lineup (no transfer info)
  expected - previous lineup updated with every transfer dated before the match day, plus roster-change features
    python3 v4.py eval [--kaggle-only]
    python3 v4.py predict "Team Vitality" "Team Spirit" 2026-10-07 [MAP] [--players a,b,c,d,e] [--players-b ...] [--bo 3]
    (rosters default to the lineup announced on Liquipedia for that tournament, see lineup.py)
"""
import pickle, sys
from collections import defaultdict

import numpy as np
import pandas as pd

import oos_eval as O
import prematch as P
import v3

TRANSFERS = O.ROOT / "data/liquipedia/transfers.csv"
CHG = ["chg30", "chg90", "since_chg", "newcomers"]
V4 = P.MODEL.with_name("prematch_v4.pkl")
POLICIES = ("actual", "last", "expected")


def player_ids(d):
    """lower-case player name -> Kaggle player id (latest)."""
    ids = {}
    for s in (1, 2):
        for i in range(1, 6):
            # Kaggle names look like "ropz (robin kool), ee": keep the nickname only
            nick = d[f"team{s}_player{i}"].astype(str).str.split(" (", regex=False).str[0].str.strip().str.lower()
            ids.update(zip(nick, d[f"team{s}_player{i}_id"]))
    ids.pop("nan", None)
    return ids


def transfers(teams):
    """Transfer moves with team names mapped onto the history's team names."""
    t = pd.read_csv(TRANSFERS, parse_dates=["date"]).fillna({"from_team": "", "to_team": ""})
    known = {}
    for n in teams:
        known.setdefault(O.key(n), n)
    for c in ("from_team", "to_team"):
        t[c] = t[c].map(lambda x: known.get(O.key(x), x) if x else "")
    return t


class State4(v3.State3):
    def __init__(self, moves, ids):
        super().__init__()
        self.moves, self.ptr, self.ids = moves, 0, ids
        self.exp = {}  # team -> expected lineup
        self.changes = defaultdict(list)  # team -> dates of player moves in/out
        self.tenure = defaultdict(int)  # (team, player) -> maps played together

    def pid(self, name):
        return self.ids.get(name, "n:" + name)

    def advance(self, when):
        """Apply every transfer dated strictly before the match day."""
        day = when.normalize()
        while self.ptr < len(self.moves) and self.moves.date.iat[self.ptr] < day:
            r = self.moves.iloc[self.ptr]
            p = self.pid(r.player)
            if r.from_team and r.from_team != r.to_team:
                lu = self.exp.get(r.from_team, self.lineup.get(r.from_team, []))
                self.exp[r.from_team] = [x for x in lu if x != p]
                self.changes[r.from_team].append(r.date)
            if r.to_team:
                lu = [x for x in self.exp.get(r.to_team, self.lineup.get(r.to_team, [])) if x != p]
                self.exp[r.to_team] = (lu + [p])[-5:] if r.active_to else lu
                self.changes[r.to_team].append(r.date)
            self.ptr += 1

    def expected(self, t):
        return self.exp.get(t, self.lineup.get(t, []))

    def change_feats(self, t, when, lineup):
        ch = self.changes[t]
        return {"chg30": sum((when - c).days <= 30 for c in ch), "chg90": sum((when - c).days <= 90 for c in ch),
                "since_chg": min(((when - ch[-1]).days if ch else 365), 365),
                "newcomers": sum(self.tenure[(t, p)] < 5 for p in lineup)}

    def update3(self, r, la, lb):
        la = la or self.expected(r.team1_id)
        lb = lb or self.expected(r.team2_id)
        for t, lu in ((r.team1_id, la), (r.team2_id, lb)):
            for p in lu:
                self.tenure[(t, p)] += 1
            if lu:
                self.exp[t] = list(lu)  # resync with what was actually fielded
        super().update3(r, la, lb)


def build4(d, moves, ids, simulate=()):
    st, out, sims = State4(moves, ids), {p: [] for p in POLICIES}, {}
    simulate = set(simulate)
    recent_maps = []
    for mid, g in d.groupby(["datetime", "match_id"], sort=True):
        rs = list(g.itertuples())
        st.advance(rs[0].datetime)
        a, b = rs[0].team1_id, rs[0].team2_id
        last = {a: st.lineup.get(a, []), b: st.lineup.get(b, [])}
        exp = {a: st.expected(a), b: st.expected(b)}
        cf = {t: st.change_feats(t, rs[0].datetime, exp[t]) for t in (a, b)}
        for k, r in enumerate(rs, 1):
            la, lb = P.lineups(r)
            bo = int(r.bestOf) if r.bestOf == r.bestOf else len(rs)
            for pol, (xa, xb) in (("actual", (la or exp[a], lb or exp[b])), ("last", (last[a], last[b])),
                                  ("expected", (exp[a], exp[b]))):
                f = st.features3(a, b, r.map_name, r.datetime, xa, xb, k, bo)
                f.update({c: cf[a][c] - cf[b][c] for c in CHG})
                out[pol].append(f)
        if mid[1] in simulate:
            r = rs[0]
            bo = int(r.bestOf) if r.bestOf == r.bestOf else len(rs)
            sims[mid[1]] = {}
            for m in v3.pool_at(recent_maps[-600:]):
                f = st.features3(a, b, m, r.datetime, exp[a], exp[b], 1, bo)
                f.update({c: cf[a][c] - cf[b][c] for c in CHG})
                sims[mid[1]][m] = f
        for r in rs:
            st.update3(r, *P.lineups(r))
            recent_maps.append(r.map_name)
    idx = d.index[:len(out["actual"])]
    return st, {p: pd.DataFrame(v, index=idx) for p, v in out.items()}, sims


def evaluate(kaggle_only=False):
    hist = P.load(O.V5) if kaggle_only else O.v2_history()
    moves = transfers(pd.concat([hist.team1, hist.team2]).unique())
    ids = player_ids(P.load(O.V5))
    mvm = pd.read_csv(O.ROOT / "data/odds/model_vs_market.csv")
    st, F, sims = build4(hist, moves, ids, simulate=set(mvm.match_id))
    y = hist.y.values
    tr = ((hist.datetime >= P.BURN_IN_END) & (hist.datetime < O.CUT)).values
    kg = set(P.load(O.V5).match_id)
    known = set(hist.team1[hist.match_id.isin(kg)]) | set(hist.team2[hist.match_id.isin(kg)])
    te = {"Kaggle Jul-Oct": ((hist.datetime >= O.CUT) & hist.match_id.isin(kg)).values,
          "Liquipedia Jul-Oct": ((hist.datetime >= O.CUT) & ~hist.match_id.isin(kg) & hist.team1.isin(known)
                                 & hist.team2.isin(known)).values}
    recent = (np.abs(F["expected"].chg30.values) > 0) | (F["expected"].newcomers.abs().values > 0)
    print(f"transfers used: {st.ptr} of {len(moves)}; train {tr.sum()} maps (< {O.CUT:%Y-%m-%d})")

    diff_base, ctx_base = P.DIFF, P.CTX
    diff_all, ctx_all = v3.cols(list(v3.GROUPS))
    specs = {"base | actual": ("actual", diff_base, ctx_base), "base | last": ("last", diff_base, ctx_base),
             "base | expected": ("expected", diff_base, ctx_base),
             "base+player elo | expected": ("expected", diff_base + ["pelo", "pmap"], ctx_base),
             "base+player elo+changes | expected": ("expected", diff_base + ["pelo", "pmap"] + CHG, ctx_base),
             "all v3+changes | expected": ("expected", diff_all + CHG, ctx_all)}
    preds, models = {}, {}
    for name, (pol, diff, ctx) in specs.items():
        X = v3.matrix(hist, F[pol])
        models[name] = (v3.fit(X[tr], y[tr], diff, ctx), diff, ctx)
        preds[name] = v3.proba(models[name][0], X, diff, ctx)
    ref = "base | last"
    for label, mask in te.items():
        if not mask.any():
            continue
        for sub, m2 in (("all", mask), ("right after a roster change", mask & recent)):
            if m2.sum() < 30:
                continue
            print(f"\n== {label}, {sub} ({m2.sum()} maps) ==  [vs '{ref}', negative = better]")
            for name, p in preds.items():
                d, lo, hi = v3.diff_ci(y[m2], p[m2], preds[ref][m2], hist.match_id[m2])
                print(f"{name:38s} log loss {v3.ll(y[m2], p[m2]).mean():.4f}  acc {((p[m2] > .5) == y[m2]).mean():.1%}"
                      f"  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    first = hist[hist.match_id.isin(mvm.match_id)].groupby("match_id").head(1)
    idx = dict(zip(first.match_id, first.index))
    rows = []
    for r in mvm.itertuples():
        i = idx.get(r.match_id)
        if i is None or r.match_id not in sims:
            continue
        row = {"match_id": r.match_id, "y_map1": r.y_map1, "y_series": r.y_series, "mkt_map1": r.mkt_map1,
               "mkt_series": r.mkt_series}
        tier = hist.tier[i] if hist.tier[i] == hist.tier[i] else 2
        for name in ("base | last", "base | expected", "base+player elo+changes | expected"):
            m, diff, ctx = models[name]
            p1 = preds[name][hist.index.get_loc(i)]
            row[name + " | map1"] = p1
            row[name + " | series"] = v3._series_from_p(p1, int(r.bo))
        rows.append(row)
    s = pd.DataFrame(rows)
    print(f"\n== vs Pinnacle closing ==")
    for level in ("map1", "series"):
        t = s.dropna(subset=["mkt_" + level])
        yy = t["y_" + level].values
        print(f"-- {level}, n={len(t)}: Pinnacle {v3.ll(yy, t['mkt_' + level].values).mean():.4f}")
        for k in [c for c in t.columns if c.endswith("| " + level)]:
            d, lo, hi = v3.diff_ci(yy, t[k].values, t["mkt_" + level].values, t.match_id)
            print(f"   {k:48s} {v3.ll(yy, t[k].values).mean():.4f}  minus Pinnacle {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def predict(team_a, team_b, date, map_name=None, players=None, bo=3, players_b=None):
    """Rosters, in priority order: --players / --players-b given by hand > the lineup announced on Liquipedia for the
    tournament running that day (lineup.py: stand-ins included) > last lineup updated with transfers up to `date`."""
    hist, date = P.load(O.V5), pd.Timestamp(date)  # ponytail: Kaggle-only history, the best top-tier variant
    moves = transfers(pd.concat([hist.team1, hist.team2]).unique())
    ids = player_ids(hist)
    st, _, _ = build4(hist[hist.datetime < date], moves, ids)
    st.advance(date + pd.Timedelta(days=1))
    m = pickle.load(open(V4, "rb"))
    known = {t.lower(): t for t in pd.concat([hist.team1, hist.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    try:
        import lineup
        ann = lineup.announced(team_a, team_b, date)
    except Exception as e:  # no network / page format changed: fall back to transfers
        print(f"lineup: Liquipedia lookup failed ({e}); using transfer-based rosters")
        ann = None
    source = {}
    def roster(team, name, manual):
        if manual:
            source[name] = "manual (--players)"
            return [st.pid(p.strip().lower()) for p in manual]
        if ann and len(ann[name]["active"]) >= 5:
            info = ann[name]
            source[name] = {"from": "Liquipedia " + ann["tournament"], "subs": info["subs"],
                            "left": info["former"], "notes": info["notes"]}
            return [st.pid(p.lower()) for p in info["active"][:5]]
        source[name] = "last lineup + transfers"
        return st.expected(team)
    la, lb = roster(a, team_a, players), roster(b, team_b, players_b)
    cf = {t: st.change_feats(t, date, lu) for t, lu in ((a, la), (b, lb))}
    def p(mp):
        f = st.features3(a, b, mp, date, la, lb, 1, bo)
        f.update({c: cf[a][c] - cf[b][c] for c in CHG})
        return float(v3.proba(m["logreg"], pd.DataFrame([f]).assign(tier=1, bestOf=bo), m["diff"], m["ctx"])[0])
    inv = {v: k for k, v in ids.items()}  # id -> nickname
    names = lambda lu: [inv.get(x, str(x).removeprefix("n:")) for x in lu]
    out = {"rosters": {team_a: names(la), team_b: names(lb)}, "roster source": source}
    if map_name:
        q = p(map_name)
        out.update({"map": map_name, team_a: q, team_b: 1 - q})
    else:
        pm = {mp: p(mp) for mp in v3.pool_at(list(hist.map_name.tail(600)))}
        s, played = v3.veto_series(pm, bo)
        out.update({"series": {team_a: s, team_b: 1 - s}, "likely maps": played})
    return out


def train():
    hist = P.load(O.V5)
    moves = transfers(pd.concat([hist.team1, hist.team2]).unique())
    _, F, _ = build4(hist, moves, player_ids(hist))
    X, y = v3.matrix(hist, F["expected"]), hist.y.values
    tr = (hist.datetime >= P.BURN_IN_END).values
    diff, ctx = P.DIFF + ["pelo", "pmap"] + CHG, P.CTX
    pickle.dump({"logreg": v3.fit(X[tr], y[tr], diff, ctx), "diff": diff, "ctx": ctx}, open(V4, "wb"))
    print("saved", V4)


if __name__ == "__main__":
    pm_flag = "--polymarket" in sys.argv  # also print the Polymarket price for this exact match
    if pm_flag:
        sys.argv.remove("--polymarket")
    args = sys.argv[1:]
    if "--update" in args:
        args.remove("--update")
        import rosters, update
        update.run()
        rosters.L.FRESH = True
        rosters.collect(rosters.pages()[-2:])  # refresh the last two months of transfers
        rosters.L.FRESH = False
        rosters.collect().to_csv(rosters.OUT, index=False)
    players, players_b, bo = None, None, 3
    if "--players-b" in args:
        i = args.index("--players-b")
        players_b = args[i + 1].split(",")
        del args[i:i + 2]
    if "--players" in args:
        i = args.index("--players")
        players = args[i + 1].split(",")
        del args[i:i + 2]
    if "--bo" in args:
        i = args.index("--bo")
        bo = int(args[i + 1])
        del args[i:i + 2]
    if args[0] == "eval":
        evaluate(kaggle_only="--kaggle-only" in args)
    elif args[0] == "train":
        train()
    else:
        print(predict(*args[1:5], players=players, bo=bo, players_b=players_b))
        if pm_flag:
            import polymarket
            polymarket.show(args[1], args[2], args[3])
