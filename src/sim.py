"""Monte Carlo map simulator (docs/plan.md step 4): round model + CS2 economy rolled forward.

- Round model: logistic P(CT wins round | log equipment ratio, pistol, overtime, map, pre-match strength of the CT
  team), fitted on demo rounds (data/demos/rounds.csv). Strength = logit of the frozen v1 pre-match map probability.
- Economy: per-player resources (cash + equipment) per team. Buy = median equipment for that resource level and side
  (fitted from demos). Next-round resources = linear fit from demos (what was left, kept equipment if won, win/loss
  reward incl. CS2 loss-bonus ladder 1400..3400, kill money). Pistols at rounds 1 and 13 ($800), side switch after 12,
  MR3 overtime with $12,500 and a switch after 3.
    python3 src/sim.py eval                                   # held-out series: calibration, scorelines, vs v1
    python3 src/sim.py "Team Vitality" "Team Spirit" Mirage 2026-10-07 [--ct A|B] [--n 10000]
"""
import argparse, pickle, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold

import oos_eval as O
import prematch as P

ROUNDS = O.ROOT / "data/demos/rounds.csv"
SIM = P.MODEL.with_name("sim.pkl")
MAPS = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train"]


def norm(name):
    return O.key(re.sub(r"\(.*?\)|globant|clan", "", str(name).lower()))


# ---------- pre-match strength (frozen v1 weights on Kaggle + Liquipedia history) ----------
def prematch_strength(rounds):
    """{(file): logit P(first CT team wins map)} for demo maps found in the history; maps not found -> 0."""
    hist = O.v2_history()
    X = O.features(hist)
    p = P.sym_proba(pickle.load(open(P.MODEL, "rb"))["logreg"], X.fillna(0))
    idx = {}
    for i, (a, b, m, s1, s2) in enumerate(zip(hist.team1, hist.team2, hist.map_name, hist.score1_game,
                                               hist.score2_game)):
        idx.setdefault((frozenset((norm(a), norm(b))), m), []).append((i, norm(a), {norm(a): s1, norm(b): s2}))
    out = {}
    for f, g in rounds.groupby("file"):
        ct, t, m = g.ct_team.iat[0], g.t_team.iat[0], g["map"].iat[0]
        score = {norm(x): (g.winner == x).sum() for x in (ct, t)}
        cands = idx.get((frozenset((norm(ct), norm(t))), m), [])
        exact = [c for c in cands if c[2] == score]
        pick = (exact or cands or [None])[-1]
        if pick is None:
            continue
        i, first, _ = pick
        q = p[i] if first == norm(ct) else 1 - p[i]
        out[f] = float(np.log(q / (1 - q)))
    return out


# ---------- fitting ----------
def prepare(r, strength):
    r = r[r.complete].sort_values(["file", "round"]).copy()
    first_ct = r.groupby("file").ct_team.transform("first")
    s = r.file.map(strength).fillna(0.0)
    r["str_ct"] = np.where(r.ct_team == first_ct, s, -s)
    r["y"] = (r.winner_side == "CT").astype(int)
    r["pistol"] = r["round"].isin([1, 13]).astype(int)
    r["ot"] = (r["round"] > 24).astype(int)
    r["res_ct"], r["res_t"] = (r.ct_equip + r.ct_money) / 5, (r.t_equip + r.t_money) / 5
    r["eq_ct"], r["eq_t"] = r.ct_equip / 5, r.t_equip / 5
    return r


def x_round(eq_ct, eq_t, pistol, ot, m, str_ct):
    n = len(eq_ct)
    cols = [np.log((eq_ct + 200) / (eq_t + 200)), np.full(n, pistol), np.full(n, ot), str_ct]
    cols += [np.full(n, float(m == mp)) for mp in MAPS]
    return np.column_stack(cols)


def fit(r):
    r = r.dropna(subset=["ct_equip", "t_equip"])
    X = x_round(r.eq_ct.values, r.eq_t.values, 0, 0, None, r.str_ct.values)
    X[:, 1], X[:, 2] = r.pistol, r.ot
    for j, mp in enumerate(MAPS):
        X[:, 4 + j] = (r["map"] == mp).values
    rm = LogisticRegression(C=1.0, max_iter=2000).fit(X, r.y)
    # buy policy: median equipment per player by resource level, per side (non-pistol rounds)
    buy = {}
    nb = r[r.pistol == 0]
    bins = np.arange(0, 17001, 500)
    for side in ("ct", "t"):
        g = nb.groupby(pd.cut(nb["res_" + side], bins), observed=True)["eq_" + side].median().dropna()
        buy[side] = (np.array([iv.mid for iv in g.index]), g.values)
    # transition: next-round resources from this round (same half only)
    rows = []
    for _, g in r.groupby("file"):
        g = g.set_index("round")
        for k in g.index:
            if k + 1 in g.index and k not in (12, 24) and k < 24:
                for side in ("ct", "t"):
                    won = int((g.loc[k, "winner_side"] == "CT") == (side == "ct"))
                    rows.append({"side": side, "left": g.loc[k, "res_" + side] - g.loc[k, "eq_" + side],
                                 "eq": g.loc[k, "eq_" + side], "won": won, "nxt": g.loc[k + 1, "res_" + side]})
    tr = pd.DataFrame(rows)
    trans = {}
    for side in ("ct", "t"):
        d = tr[tr.side == side]
        Z = np.column_stack([d["left"], d["eq"] * d["won"], d["won"]])
        trans[side] = LinearRegression().fit(Z, d.nxt)
    model = {"round": rm, "buy": buy, "trans": trans}
    # Fit the per-map strength spread so simulated loser-rounds match the training maps, re-anchoring each time.
    strengths = r.groupby("file").str_ct.first().values
    real = r.groupby("file").apply(lambda g: min((g.winner == g.ct_team.iat[0]).sum(),
                                                  (g.winner != g.ct_team.iat[0]).sum()), include_groups=False).mean()
    best = None
    for sigma in (0.0, 0.75, 1.5, 2.25, 3.0):
        model["sigma"] = sigma
        anchor(model)
        sim_lr = np.mean([simulate(model, "Mirage", s, True, n=400, seed=i)["loser_rounds"].mean()
                          for i, s in enumerate(strengths[:60])])
        if best is None or abs(sim_lr - real) < best[0]:
            best = (abs(sim_lr - real), sigma, model["round"].coef_[0, 3])
    model["sigma"], model["round"].coef_[0, 3] = best[1], best[2]
    return model


def anchor(model, target=1.0):
    """Scale the per-round strength coefficient so that, averaged over maps and starting sides, the simulated
    P(map win) for a team with pre-match logit `target` equals sigmoid(target). The per-round fit on ~100 maps
    under-weights strength; this makes the simulator agree with the pre-match model on P(map win)."""
    want = 1 / (1 + np.exp(-target))
    coef = model["round"].coef_
    lo, hi = 0.0, 3.0
    for _ in range(9):
        mid = (lo + hi) / 2
        coef[0, 3] = mid
        got = np.mean([simulate(model, mp, target, ct, n=2000, seed=7)["p_win"]
                       for mp in ("Mirage", "Ancient", "Nuke") for ct in (True, False)])
        lo, hi = (mid, hi) if got < want else (lo, mid)
    coef[0, 3] = (lo + hi) / 2
    return model


# ---------- simulation ----------
def buy(model, side, res):
    x, y = model["buy"][side]
    return np.minimum(np.interp(res, x, y), res)


def next_res(model, side, res, eq, won, losses):
    lin = model["trans"][side]
    base = lin.predict(np.column_stack([res - eq, eq * won, won]))
    # the fit already contains the average loss bonus; add the deviation from it for long losing streaks
    bonus = np.where(won == 1, 0, (1400 + 500 * np.minimum(losses, 4)) - 2400)
    return np.clip(base + bonus, 0, 16000 + 6000)


def simulate(model, m, str_a, a_ct_first=True, n=10000, seed=0):
    """Team A vs B on map m. str_a = logit pre-match P(A wins map). Returns summary dict.
    Each simulated map draws its own strength (str_a + sigma*z): real teams' form varies map to map, and without
    this the simulator produces too many close scorelines."""
    rng = np.random.default_rng(seed)
    str_a = str_a + model.get("sigma", 0.0) * rng.standard_normal(n)
    sa, sb = np.zeros(n, int), np.zeros(n, int)
    res = {"a": np.full(n, 800.0), "b": np.full(n, 800.0)}
    loss = {"a": np.ones(n), "b": np.ones(n)}
    alive = np.ones(n, bool)
    win_by_round, half1 = {}, None
    rnd = 0
    while alive.any() and rnd < 60:
        rnd += 1
        if rnd <= 24:
            a_ct = a_ct_first == (rnd <= 12)
            if rnd in (1, 13):
                res = {"a": np.full(n, 800.0), "b": np.full(n, 800.0)}
                loss = {"a": np.ones(n), "b": np.ones(n)}
        else:
            k = (rnd - 25) % 6
            a_ct = (a_ct_first == ((rnd - 25) // 3 % 2 == 1))  # OT: start on 2nd-half sides, switch after 3
            if k in (0, 3):
                res = {"a": np.full(n, 12500.0), "b": np.full(n, 12500.0)}
        pistol = int(rnd in (1, 13))
        ct, t = ("a", "b") if a_ct else ("b", "a")
        eq = {ct: np.full(n, 700.0) if pistol else buy(model, "ct", res[ct]),
              t: np.full(n, 700.0) if pistol else buy(model, "t", res[t])}
        str_ct = str_a if ct == "a" else -str_a
        p_ct = model["round"].predict_proba(x_round(eq[ct], eq[t], pistol, int(rnd > 24), m, str_ct))[:, 1]
        ct_won = rng.random(n) < p_ct
        a_won = ct_won if ct == "a" else ~ct_won
        win_by_round[rnd] = (a_won[alive].mean(), alive.mean())
        sa += a_won & alive
        sb += ~a_won & alive
        for team, won in (("a", a_won), ("b", ~a_won)):
            side = "ct" if team == ct else "t"
            w = won.astype(int)
            res[team] = next_res(model, side, res[team], eq[team], w, loss[team])
            loss[team] = np.where(won, np.maximum(loss[team] - 1, 0), np.minimum(loss[team] + 1, 4))
        if rnd == 12:
            half1 = (sa.mean(), sb.mean())
        if rnd <= 24:
            alive &= (sa < 13) & (sb < 13)
        else:  # overtime: first to 4 in each block of 6 (above 12)
            base = 12 + 3 * ((rnd - 25) // 6)  # each 6-round OT block starts from 12-12, 15-15, ...
            alive &= (sa - base < 4) & (sb - base < 4)
    lines = pd.Series([f"{x}-{y}" for x, y in zip(sa, sb)]).value_counts(normalize=True)
    return {"p_win": float((sa > sb).mean()), "scorelines": lines.head(8).round(3).to_dict(),
            "half1_expected": tuple(round(x, 2) for x in half1),
            "p_pistol1": win_by_round[1][0], "p_pistol13": win_by_round.get(13, (np.nan,))[0],
            "p_round": {k: round(v[0], 3) for k, v in win_by_round.items() if k <= 24},
            "loser_rounds": np.minimum(sa, sb)}


# ---------- evaluation ----------
def evaluate(n=3000, out=None, folds=5, max_maps=None):
    """Held-out-series evaluation. Writes summary.txt, per_map.csv, calibration.csv, scorelines.csv to `out`."""
    out = Path(out or O.ROOT / "results")
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROUNDS)
    strength = prematch_strength(raw[raw.complete])
    r = prepare(raw, strength)
    maps = r.groupby("file").agg(series=("series", "first"), m=("map", "first"), ct=("ct_team", "first"),
                                 t=("t_team", "first")).reset_index()
    maps["series"] = maps.series.fillna(maps.file)
    if max_maps:  # smoke test
        maps = maps[maps.series.isin(maps.series.drop_duplicates().head(max_maps))]
        r = r[r.file.isin(maps.file)]
    final = r.groupby("file").apply(lambda g: (g.winner == g.ct_team.iat[0]).sum() > (g.winner != g.ct_team.iat[0]).sum(),
                                    include_groups=False)
    maps["y"] = maps.file.map(final).astype(int)
    maps["str"] = maps.file.map(strength)
    maps["known"] = maps["str"].notna()
    maps["p_v1"] = 1 / (1 + np.exp(-maps["str"].fillna(0)))
    maps["p_sim"] = np.nan
    real_lr, sim_lr = [], []
    for tr, te in GroupKFold(n_splits=min(folds, maps.series.nunique())).split(maps, maps.y, maps.series):
        model = fit(r[r.file.isin(maps.file.iloc[tr])])
        for i in te:
            row = maps.iloc[i]
            s = simulate(model, row.m, row["str"] if row.known else 0.0, True, n=n, seed=i)
            maps.loc[maps.index[i], "p_sim"] = s["p_win"]
            sim_lr.append(s["loser_rounds"])
            g = r[r.file == row.file]
            real_lr.append(min((g.winner == row.ct).sum(), (g.winner != row.ct).sum()))
    ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    rng = np.random.default_rng(0)
    lines = [f"{len(maps)} maps ({maps.known.sum()} with pre-match strength), {maps.series.nunique()} series, "
             f"{min(folds, maps.series.nunique())}-fold by series, {n} sims per map"]
    for name, d in (("all maps", maps), ("maps with strength", maps[maps.known])):
        lines.append(f"\n{name} (n={len(d)}):")
        for col in ("p_sim", "p_v1"):
            lines.append(f"  {col:6s} log loss {ll(d.y, d[col]):.4f}  Brier {np.mean((d[col] - d.y) ** 2):.4f}  "
                         f"acc {((d[col] > .5) == d.y).mean():.1%}")
        diff = [ll(d.y.values[i], d.p_sim.values[i]) - ll(d.y.values[i], d.p_v1.values[i])
                for i in [rng.integers(0, len(d), len(d)) for _ in range(2000)]]
        lines.append(f"  sim minus v1 log loss: {ll(d.y, d.p_sim) - ll(d.y, d.p_v1):+.4f}, 95% bootstrap CI "
                     f"[{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]; coin flip 0.6931")
    b = pd.cut(maps.p_sim, [0, .3, .4, .5, .6, .7, 1])
    cal = maps.groupby(b, observed=True).agg(n=("y", "size"), pred=("p_sim", "mean"), actual=("y", "mean")).round(3)
    simd = pd.Series(np.concatenate(sim_lr)).value_counts(normalize=True).sort_index()
    reald = pd.Series(real_lr).value_counts(normalize=True).sort_index()
    sc = pd.DataFrame({"real_demo_maps": reald, "simulated": simd}).fillna(0)
    k = pd.read_csv(O.V5 / "cs2_all_tiers_games.csv", usecols=["is_total", "datetime", "score1_game", "score2_game"])
    k = k[(~k.is_total) & (k.datetime >= "2026-01-01")].dropna()
    k = k[k[["score1_game", "score2_game"]].max(axis=1) >= 13]
    sc["real_kaggle_2026"] = k[["score1_game", "score2_game"]].min(axis=1).astype(int).value_counts(normalize=True)
    sc = sc.fillna(0).round(3).loc[lambda d: d.index <= 16]
    lines += ["\nreliability (sim):", cal.to_string(), "\nloser's rounds per map:", sc.to_string(),
              f"\nmean loser rounds: demo maps {np.mean(real_lr):.2f}, Kaggle 2026 ({len(k)} maps) "
              f"{k[['score1_game', 'score2_game']].min(axis=1).mean():.2f}, simulated {np.mean(np.concatenate(sim_lr)):.2f}"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    maps.to_csv(out / "per_map.csv", index=False)
    cal.to_csv(out / "calibration.csv")
    sc.to_csv(out / "scorelines.csv")
    final_model = fit(r)
    pickle.dump(final_model, open(out / "sim.pkl", "wb"))
    if Path(SIM).parent.exists():
        pickle.dump(final_model, open(SIM, "wb"))
    print("wrote", out)


def cli(team_a, team_b, m, date, ct="A", n=10000):
    """Pre-match strength from the frozen v1 model at `date`, then simulate (A starts CT unless --ct B)."""
    model = pickle.load(open(SIM, "rb"))
    hist, date = O.v2_history(), pd.Timestamp(date)
    st, _ = P.build(hist, until=date)
    known = {t.lower(): t for t in pd.concat([hist.team1, hist.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    X = pd.DataFrame([st.features(a, b, m, date)]).assign(tier=1, bestOf=3)[P.DIFF + P.CTX]
    q = float(P.sym_proba(pickle.load(open(P.MODEL, "rb"))["logreg"], X.fillna(0))[0])
    out = simulate(model, m, np.log(q / (1 - q)), ct.upper() == "A", n=n)
    out.pop("loser_rounds")
    return {"pre-match v1": round(q, 3), **out}


def use_data_dir(d):
    """Point every input at another folder (cloud runs): <d>/demos/rounds.csv, <d>/kaggle_v5/,
    <d>/liquipedia/maps_2026.csv, <d>/models/prematch.pkl."""
    global ROUNDS, SIM
    d = Path(d)
    ROUNDS = d / "demos/rounds.csv"
    O.V5, O.LP = d / "kaggle_v5", d / "liquipedia/maps_2026.csv"
    P.MODEL = d / "models/prematch.pkl"
    SIM = d / "models/sim.pkl"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="+", help='eval | "Team A" "Team B" MAP DATE')
    ap.add_argument("--data-dir")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--out", default=str(O.ROOT / "results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-series", type=int, help="smoke test: only the first N series")
    ap.add_argument("--ct", default="A")
    a = ap.parse_args()
    if a.data_dir:
        use_data_dir(a.data_dir)
    if a.cmd[0] == "eval":
        evaluate(n=a.sims, out=a.out, folds=a.folds, max_maps=a.max_series)
    else:
        print(cli(*a.cmd[:4], ct=a.ct, n=a.sims))
