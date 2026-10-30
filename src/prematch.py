"""Pre-match map-winner model: P(team A wins map M) from strictly prior matches.

    python3 src/prematch.py train            # build features, fit, calibrate, report, save models/prematch.pkl
    python3 src/prematch.py predict "Team Vitality" "Team Spirit" Mirage 2026-07-01
"""
import math, os, pickle, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data/raw"
MODEL = Path(__file__).resolve().parent.parent / "models/prematch.pkl"
VAL_START, TEST_START = "2025-07-01", "2026-01-01"
BURN_IN_END = "2024-01-15"  # ponytail: per-map rows start 2023-10-25; drop ~3 months from training while Elo settles
K, K_MAP, FORM_N, PLAYER_N = 32, 32, 10, 20
STATS = ["kills", "deaths", "adr", "kast"]

_SKLEARN_ATTRS = {
    "HistGradientBoostingClassifier": ("sklearn.ensemble", "HistGradientBoostingClassifier"),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "accuracy_score": ("sklearn.metrics", "accuracy_score"),
    "brier_score_loss": ("sklearn.metrics", "brier_score_loss"),
    "log_loss": ("sklearn.metrics", "log_loss"),
    "roc_auc_score": ("sklearn.metrics", "roc_auc_score"),
    "make_pipeline": ("sklearn.pipeline", "make_pipeline"),
    "StandardScaler": ("sklearn.preprocessing", "StandardScaler"),
}


def __getattr__(name):
    if name in _SKLEARN_ATTRS:
        mod_name, attr_name = _SKLEARN_ATTRS[name]
        import importlib
        val = getattr(importlib.import_module(mod_name), attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load(raw=RAW):
    tiers = pd.concat([pd.read_csv(raw / f"cs2_tier{t}_games.csv", usecols=["game_id", "is_total"]).assign(tier=t)
                       for t in (1, 2, 3)])
    tiers = tiers[~tiers.is_total].drop_duplicates("game_id")[["game_id", "tier"]]
    d = pd.read_csv(raw / "cs2_all_tiers_games.csv", low_memory=False)
    d = d[~d.is_total].dropna(subset=["map_name", "score1_game", "score2_game"])
    # team1_win is the SERIES winner repeated on every map row; the map winner comes from the score.
    d = d[(d[["score1_game", "score2_game"]].max(axis=1) >= 13) & (d.score1_game != d.score2_game)]
    d["y"] = (d.score1_game > d.score2_game).astype(int)
    d["datetime"] = pd.to_datetime(d.datetime)
    # team*_id is a per-match entry id (Vitality has 135 of them); the name is the stable team key.
    d["team1_id"], d["team2_id"] = d.team1, d.team2
    d = d.merge(tiers, on="game_id", how="left")
    return d.sort_values(["datetime", "match_id", "game_id"]).reset_index(drop=True)


def elo_p(diff):
    return 1 / (1 + 10 ** (-diff / 400))


class State:
    """Everything known about teams/players up to some point in time."""

    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.map_elo = defaultdict(lambda: 1500.0)  # (team, map)
        self.form = defaultdict(lambda: deque(maxlen=FORM_N))  # team -> (win, round_diff)
        self.player = defaultdict(lambda: deque(maxlen=PLAYER_N))  # player -> (k, d, adr, kast)
        self.lineup = {}  # team -> last lineup
        self.last = {}  # team -> last match datetime
        self.n = defaultdict(int)
        self.n_map = defaultdict(int)
        self.h2h = defaultdict(list)  # (a, b) sorted -> winners

    def side(self, t, m, lineup, when):
        f = self.form[t]
        nf = len(f)
        if nf:
            # FORM_N == 10: for <= 8 sequential sum, for 9..10 8-way accumulator matching np.mean
            if nf <= 8:
                sw = 0.0
                sr = 0.0
                for w, r in f:
                    sw += w
                    sr += r
            else:
                fl = list(f)
                sw = ((fl[0][0] + fl[1][0]) + (fl[2][0] + fl[3][0])) + ((fl[4][0] + fl[5][0]) + (fl[6][0] + fl[7][0]))
                sr = ((fl[0][1] + fl[1][1]) + (fl[2][1] + fl[3][1])) + ((fl[4][1] + fl[5][1]) + (fl[6][1] + fl[7][1]))
                for k in range(8, nf):
                    sw += fl[k][0]
                    sr += fl[k][1]
            form_v = sw / nf
            rdiff_v = sr / nf
        else:
            form_v = np.nan
            rdiff_v = np.nan

        pl_dict = self.player
        p_rows = []
        for p in lineup:
            dq = pl_dict.get(p)
            if dq:
                p_rows.extend(dq)
        if p_rows:
            ps = np.array(p_rows, dtype=np.float64)
            kd_v = ps[:, 0].sum() / max(ps[:, 1].sum(), 1)
            adr_v = ps[:, 2].mean()
            kast_v = ps[:, 3].mean()
        else:
            kd_v = adr_v = kast_v = np.nan

        prev = self.lineup.get(t, [])
        return {
            "elo": self.elo[t], "map_elo": self.map_elo[(t, m)],
            "form": form_v,
            "rdiff": rdiff_v,
            "kd": kd_v,
            "adr": adr_v,
            "kast": kast_v,
            "rest": (when - self.last[t]).days if t in self.last else np.nan,
            "exp": math.log1p(self.n[t]), "map_exp": math.log1p(self.n_map[(t, m)]),
            "stable": len(set(lineup) & set(prev)) if prev else np.nan,
        }

    def features(self, a, b, m, when, la=None, lb=None):
        la = la or self.lineup.get(a, [])  # no lineup in the row (e.g. Liquipedia) -> last known one
        lb = lb or self.lineup.get(b, [])
        sa, sb = self.side(a, m, la, when), self.side(b, m, lb, when)
        f = {k: sa[k] - sb[k] for k in sa}
        hk = (a, b) if a <= b else (b, a)
        h = self.h2h[hk]
        nh = len(h)
        f["h2h_n"] = math.log1p(nh)
        if nh:
            if nh <= 8:
                sh = 0.0
                for w in h:
                    sh += (1.0 if w == a else 0.0)
                f["h2h"] = sh / nh - 0.5
            else:
                f["h2h"] = np.mean([w == a for w in h]) - 0.5
        else:
            f["h2h"] = 0.0
        return f

    def _update_fast(self, a, b, m, y, s1, s2, when, la, s_a, lb, s_b):
        elo = self.elo
        delta = K * (y - 1.0 / (1.0 + 10.0 ** (-(elo[a] - elo[b]) / 400.0)))
        elo[a] += delta
        elo[b] -= delta
        melo = self.map_elo
        ka, kb = (a, m), (b, m)
        dm = K_MAP * (y - 1.0 / (1.0 + 10.0 ** (-(melo[ka] - melo[kb]) / 400.0)))
        melo[ka] += dm
        melo[kb] -= dm
        rd = s1 - s2
        self.form[a].append((y, rd))
        self.form[b].append((1 - y, -rd))
        self.h2h[(a, b) if a <= b else (b, a)].append(a if y else b)
        self.n[a] += 1
        self.n[b] += 1
        self.n_map[ka] += 1
        self.n_map[kb] += 1
        self.last[a] = when
        self.last[b] = when
        pl = self.player
        for p, sv in zip(la, s_a):
            if sv is not None:
                pl[p].append(sv)
        for p, sv in zip(lb, s_b):
            if sv is not None:
                pl[p].append(sv)
        if la:
            self.lineup[a] = la
        if lb:
            self.lineup[b] = lb

    def update(self, r, la, lb):
        a, b, m, y = r.team1_id, r.team2_id, r.map_name, r.y
        for key, store, k in (((a, b), self.elo, K), (((a, m), (b, m)), self.map_elo, K_MAP)):
            ka, kb = key
            delta = k * (y - elo_p(store[ka] - store[kb]))
            store[ka] += delta
            store[kb] -= delta
        rd = r.score1_game - r.score2_game
        self.form[a].append((y, rd))
        self.form[b].append((1 - y, -rd))
        self.h2h[tuple(sorted((a, b)))].append(a if y else b)
        for t in (a, b):
            self.n[t] += 1
            self.n_map[(t, m)] += 1
            self.last[t] = r.datetime
        for team, lineup in (("team1", la), ("team2", lb)):
            for i, p in enumerate(lineup, 1):
                s = [getattr(r, f"{team}_player{i}_{x}") for x in STATS]
                if not any(pd.isna(s)):
                    self.player[p].append(s)
        if la:
            self.lineup[a] = la
        if lb:
            self.lineup[b] = lb


def lineups(r):
    get = lambda t: [p for p in (getattr(r, f"{t}_player{i}_id", None) for i in range(1, 6)) if not pd.isna(p)]
    return get("team1"), get("team2")


def build(d, until=None, state=None):
    """Replay maps in time order. Features for every map in a match use only state from before that match."""
    st, rows = state or State(), []
    n = len(d)
    if n == 0:
        return st, pd.DataFrame(rows, index=d.index[:0])

    cols_set = set(d.columns)
    dt_list = d["datetime"].tolist()
    mid_list = d["match_id"].tolist()
    t1_list = d["team1_id"].tolist()
    t2_list = d["team2_id"].tolist()
    map_list = d["map_name"].tolist()
    y_list = d["y"].tolist()
    s1_list = d["score1_game"].tolist()
    s2_list = d["score2_game"].tolist()

    has_p = all(f"team1_player{i}_id" in cols_set for i in range(1, 6))
    if has_p:
        p1_ids = [d[f"team1_player{i}_id"].tolist() for i in range(1, 6)]
        p2_ids = [d[f"team2_player{i}_id"].tolist() for i in range(1, 6)]
        p1_sts = [[d[f"team1_player{i}_{x}"].tolist() for x in STATS] for i in range(1, 6)]
        p2_sts = [[d[f"team2_player{i}_{x}"].tolist() for x in STATS] for i in range(1, 6)]

    def _row_lu_stats(idx):
        if not has_p:
            return [], [], [], []
        la = [p1_ids[j][idx] for j in range(5) if p1_ids[j][idx] is not None and p1_ids[j][idx] == p1_ids[j][idx]]
        lb = [p2_ids[j][idx] for j in range(5) if p2_ids[j][idx] is not None and p2_ids[j][idx] == p2_ids[j][idx]]
        s_a = []
        for j in range(len(la)):
            sv = [p1_sts[j][x][idx] for x in range(4)]
            s_a.append(sv if all(v is not None and v == v for v in sv) else None)
        s_b = []
        for j in range(len(lb)):
            sv = [p2_sts[j][x][idx] for x in range(4)]
            s_b.append(sv if all(v is not None and v == v for v in sv) else None)
        return la, s_a, lb, s_b

    # Check if (datetime, match_id) is already grouped in sorted order (true for load() and v2_history())
    is_sorted = True
    for i in range(n - 1):
        d0, d1 = dt_list[i], dt_list[i + 1]
        if d0 > d1:
            is_sorted = False
            break
        if d0 == d1:
            m0, m1 = mid_list[i], mid_list[i + 1]
            t0_s, t1_s = isinstance(m0, str), isinstance(m1, str)
            if t0_s != t1_s:
                if t0_s and not t1_s:
                    is_sorted = False
                    break
            elif m0 > m1:
                is_sorted = False
                break

    if is_sorted:
        pos = 0
        while pos < n:
            g_dt = dt_list[pos]
            if until is not None and g_dt >= until:
                break
            g_mid = mid_list[pos]
            end = pos + 1
            while end < n and dt_list[end] == g_dt and mid_list[end] == g_mid and type(mid_list[end]) is type(g_mid):
                end += 1
            lu_cache = [_row_lu_stats(idx) for idx in range(pos, end)]
            for off, idx in enumerate(range(pos, end)):
                la, _, lb, _ = lu_cache[off]
                rows.append(st.features(t1_list[idx], t2_list[idx], map_list[idx], dt_list[idx], la, lb))
            for off, idx in enumerate(range(pos, end)):
                la, s_a, lb, s_b = lu_cache[off]
                st._update_fast(t1_list[idx], t2_list[idx], map_list[idx], y_list[idx],
                                s1_list[idx], s2_list[idx], dt_list[idx], la, s_a, lb, s_b)
            pos = end
    else:
        for (g_dt, _), grp_idx in d[["datetime", "match_id"]].groupby(["datetime", "match_id"], sort=True).indices.items():
            if until is not None and g_dt >= until:
                break
            lu_cache = [_row_lu_stats(int(idx)) for idx in grp_idx]
            for off, idx in enumerate(grp_idx):
                idx = int(idx)
                la, _, lb, _ = lu_cache[off]
                rows.append(st.features(t1_list[idx], t2_list[idx], map_list[idx], dt_list[idx], la, lb))
            for off, idx in enumerate(grp_idx):
                idx = int(idx)
                la, s_a, lb, s_b = lu_cache[off]
                st._update_fast(t1_list[idx], t2_list[idx], map_list[idx], y_list[idx],
                                s1_list[idx], s2_list[idx], dt_list[idx], la, s_a, lb, s_b)

    return st, pd.DataFrame(rows, index=d.index[:len(rows)])


DIFF = ["elo", "map_elo", "form", "rdiff", "kd", "adr", "kast", "rest", "exp", "map_exp", "stable", "h2h"]
CTX = ["h2h_n", "tier", "bestOf"]  # orientation-invariant


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(cal, p):
    return cal.predict_proba(logit(p)[:, None])[:, 1]


def symmetrize(X, y):
    flip = X.copy()
    flip[DIFF] = -flip[DIFF]
    return pd.concat([X, flip]), np.concatenate([y, 1 - y])


def sym_proba(model, X):
    flip = X.copy()
    flip[DIFF] = -flip[DIFF]
    return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(flip)[:, 1]) / 2


def report(name, y, p):
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
    return {"model": name, "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p), "acc": accuracy_score(y, p > 0.5), "n": len(y)}


def reliability(y, p, bins=10):
    b = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    return pd.DataFrame({"p": p, "y": y}).groupby(b, observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"),
                                                                        win_rate=("y", "mean"))


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = load()
    _, F = build(d)
    X = F.assign(tier=d.tier, bestOf=d.bestOf)[DIFF + CTX]
    y = d.y.values
    tr = (d.datetime >= BURN_IN_END) & (d.datetime < VAL_START)
    va = (d.datetime >= VAL_START) & (d.datetime < TEST_START)
    te = d.datetime >= TEST_START
    print(f"maps: {len(d)}  train {tr.sum()}  val {va.sum()}  test {te.sum()}  "
          f"({d.datetime.min():%Y-%m-%d} .. {d.datetime.max():%Y-%m-%d})")

    Xs, ys = symmetrize(X[tr], y[tr])
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xs.fillna(0), ys)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=0).fit(Xs, ys)

    preds = {
        "coin flip": lambda Z: np.full(len(Z), 0.5),
        "elo only": lambda Z: elo_p(Z.elo.values),
        "elo+map elo": lambda Z: elo_p(Z.elo.values + Z.map_elo.values),
        "logreg": lambda Z: sym_proba(logreg, Z.fillna(0)),
        "gbm": lambda Z: sym_proba(gbm, Z),
    }
    # Platt-scale on validation: one slope on the logit, no intercept, so P(A)+P(B)=1 still holds.
    # (Isotonic overfit the 1.5k validation maps and raised test log loss.)
    cal = {}
    for name in ("logreg", "gbm"):
        lv = logit(preds[name](X[va]))
        cal[name] = LogisticRegression(fit_intercept=False).fit(np.r_[lv, -lv][:, None], np.r_[y[va], 1 - y[va]])
    res, test_p = [], {}
    for name, f in preds.items():
        test_p[name] = f(X[te])
        res.append(report(name, y[te], test_p[name]))
        if name in cal:
            test_p[name + " +platt"] = calibrate(cal[name], test_p[name])
            res.append(report(name + " +platt", y[te], test_p[name + " +platt"]))
    res = pd.DataFrame(res).set_index("model").round(4)
    print("\nTest period", TEST_START, "onward:\n", res.to_string())
    best = res.drop("coin flip").log_loss.idxmin()
    print(f"\nReliability ({best}, test):\n", reliability(y[te], test_p[best]).round(3).to_string())
    print("\nSlot check: mean P(team1) on test =", round(test_p[best].mean(), 3), " actual team1 rate =",
          round(y[te].mean(), 3))

    MODEL.parent.mkdir(exist_ok=True)
    pickle.dump({"logreg": logreg, "gbm": gbm, "cal": cal, "best": best}, open(MODEL, "wb"))
    return res


def predict(team_a, team_b, map_name, date):
    """P(team_a wins map_name), using every map played before `date` and the teams' last known lineups."""
    d, date = load(), pd.Timestamp(date)
    st, _ = build(d, until=date)
    known = {t.lower(): t for t in pd.concat([d.team1, d.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    m = pickle.load(open(MODEL, "rb"))
    row = d[(d.team1_id == a) | (d.team2_id == a)].tail(1)
    X = pd.DataFrame([st.features(a, b, map_name, date)]).assign(tier=row.tier.values[0] if len(row) else 1,
                                                                bestOf=3)[DIFF + CTX]
    name = m["best"].replace(" +platt", "")
    model = m[name]
    p = sym_proba(model, X.fillna(0) if name == "logreg" else X)
    if m["best"].endswith("+platt"):
        p = calibrate(m["cal"][name], p)
    return {team_a: float(p[0]), team_b: float(1 - p[0])}


if __name__ == "__main__":
    pm_flag = "--polymarket" in sys.argv  # also print the Polymarket price for this exact match
    if pm_flag:
        sys.argv.remove("--polymarket")
    if sys.argv[1] == "train":
        train()
    else:
        print(predict(*sys.argv[2:6]))
        if pm_flag:
            import polymarket
            polymarket.show(sys.argv[2], sys.argv[3], sys.argv[5])
