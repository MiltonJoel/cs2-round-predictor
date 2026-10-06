"""Out-of-sample check of the frozen pre-match model (models/prematch.pkl, trained on Kaggle data through 2026-06-28).

    python3 liquipedia.py   # once: fetch + cache Liquipedia map results (data/liquipedia/maps_2026.csv)
    python3 oos_eval.py     # walk-forward scoring on (C) Kaggle refresh and (B) Liquipedia, source-bias checks
    python3 oos_eval.py v2  # then retrain on everything -> models/prematch_v2.pkl (original kept)
    python3 oos_eval.py predict "Team Vitality" "Team Spirit" Mirage 2026-10-07   # v2 prediction
"""
import pickle, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

import prematch as P

ROOT = Path(__file__).parent
CUT = pd.Timestamp("2026-06-29")  # first day the frozen model never saw
V5 = ROOT / "data/kaggle_v5"
LP = ROOT / "data/liquipedia/maps_2026.csv"


def key(name):
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return re.sub(r"^team|(esports?|gaming|club|team|cs|csgo|cs2)$", "", s) or s


def frozen_scores(d, X):
    m = pickle.load(open(P.MODEL, "rb"))
    return {"coin flip": np.full(len(X), 0.5), "elo only": P.elo_p(X.elo.values),
            "frozen logreg": P.sym_proba(m["logreg"], X.fillna(0))}


def features(d):
    _, F = P.build(d)
    return F.assign(tier=d.tier.fillna(2), bestOf=d.bestOf)[P.DIFF + P.CTX]


def boot(y, p, groups, n=1000, seed=0):
    """95% CI for log loss, resampling whole matches (maps within a series are correlated)."""
    rng, g = np.random.default_rng(seed), pd.factorize(groups)[0]
    idx = [np.flatnonzero(g == i) for i in range(g.max() + 1)]
    ll = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    stats = [ll[np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])].mean() for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


def table(y, preds, groups):
    rows = []
    for name, p in preds.items():
        r = P.report(name, y, p)
        r["ll_95ci"] = "[%.3f, %.3f]" % tuple(boot(y, p, groups))
        rows.append(r)
    return pd.DataFrame(rows).set_index("model").round(4)


def liquipedia_history(base, lp):
    """Kaggle history up to CUT + Liquipedia maps from CUT on, teams renamed to Kaggle names where they match."""
    known = {}
    for n in pd.concat([base.team1, base.team2]):
        known.setdefault(key(n), n)
    lp = lp.copy()
    for t in ("team1", "team2"):
        lp[t + "_k"] = lp[t].map(lambda x: known.get(key(x)))
    lp["matched"] = lp.team1_k.notna() & lp.team2_k.notna()
    new = lp[lp.datetime >= CUT].copy()
    for t in ("team1", "team2"):
        new[t] = new[t + "_k"].fillna(new[t])
        new[t + "_id"] = new[t]
    new["y"] = (new.score1_game > new.score2_game).astype(int)
    new["match_id"] = new.datetime.astype(str) + new.team1 + new.team2
    hist = pd.concat([base[base.datetime < CUT], new], ignore_index=True)
    return hist.sort_values(["datetime", "match_id", "map_no"], kind="stable").reset_index(drop=True), lp


def coverage(train, d):
    n = pd.concat([train.team1, train.team2]).value_counts()
    return np.where((d.team1.map(n).fillna(0) >= 20) & (d.team2.map(n).fillna(0) >= 20), "both teams 20+ maps", "thin")


def breakdown(d, p, y, by):
    out = pd.DataFrame({"by": by, "p": p, "y": y}).groupby("by").apply(
        lambda g: pd.Series({"n": len(g), "log_loss": P.log_loss(g.y, g.p, labels=[0, 1]),
                             "acc": ((g.p > .5) == g.y).mean()}), include_groups=False)
    return out.round(4)


def main():
    base = P.load()
    train_mask = (base.datetime >= P.BURN_IN_END) & (base.datetime < P.VAL_START)
    print("== Reference: 2026H1 test of the frozen model (should reproduce README) ==")
    Xb = features(base)
    te = base.datetime >= P.TEST_START
    print(table(base.y.values[te], frozen_scores(base[te], Xb[te]), base.match_id[te]).to_string())

    print(f"\n== C) Kaggle refresh (v5), maps from {CUT:%Y-%m-%d}, walk-forward ==")
    v5 = P.load(V5)
    Xv = features(v5)
    new = (v5.datetime >= CUT).values
    yv = v5.y.values[new]
    pv = frozen_scores(v5[new], Xv[new])
    print(table(yv, pv, v5.match_id[new]).to_string())
    print("\nreliability (frozen logreg):\n", P.reliability(yv, pv["frozen logreg"]).round(3).to_string())
    print("\nby tier:\n", breakdown(v5[new], pv["frozen logreg"], yv, v5.tier[new].values).to_string())
    print("\nby coverage:\n", breakdown(v5[new], pv["frozen logreg"], yv, coverage(base[train_mask], v5[new])).to_string())

    print(f"\n== B) Liquipedia (independent source), maps from {CUT:%Y-%m-%d}, walk-forward ==")
    lp = pd.read_csv(LP, parse_dates=["datetime"])
    lp = lp[(lp[["score1_game", "score2_game"]].max(axis=1) >= 13) & (lp.score1_game != lp.score2_game)]
    hist, lpk = liquipedia_history(base, lp)
    late = lpk[lpk.datetime >= CUT]
    teams = pd.concat([late.team1, late.team2])
    tk = pd.concat([late.team1_k, late.team2_k])
    print(f"team-name match: {tk.notna().mean():.1%} of team slots, {late.matched.mean():.1%} of maps both matched;"
          f" unmatched teams: {teams[tk.isna().values].nunique()} (top: "
          f"{teams[tk.isna().values].value_counts().head(12).to_dict()})")
    Xh = features(hist)
    m = ((hist.datetime >= CUT) & hist.page.notna()).values
    sel = m & hist.team1.isin(set(base.team1) | set(base.team2)).values & hist.team2.isin(
        set(base.team1) | set(base.team2)).values
    yl = hist.y.values[sel]
    pl = frozen_scores(hist[sel], Xh[sel])
    print(f"scored maps (both teams known to the model): {sel.sum()} of {m.sum()}")
    print(table(yl, pl, hist.match_id[sel]).to_string())
    print("\nreliability (frozen logreg):\n", P.reliability(yl, pl["frozen logreg"]).round(3).to_string())
    print("\nby Liquipedia tier:\n", breakdown(hist[sel], pl["frozen logreg"], yl, hist.tier_lp[sel].values).to_string())
    print("\nby coverage:\n", breakdown(hist[sel], pl["frozen logreg"], yl, coverage(base[train_mask], hist[sel])).to_string())

    print("\n== Source-bias checks ==")
    ov = lpk[(lpk.datetime >= P.TEST_START) & (lpk.datetime < CUT) & lpk.matched].copy()
    kg = base[base.datetime >= P.TEST_START].copy()
    for df, a, b in ((ov, "team1_k", "team2_k"), (kg, "team1", "team2")):
        df["day"] = df.datetime.dt.normalize()
        df["pair"] = [tuple(sorted(x)) for x in zip(df[a], df[b])]
        df["w"] = np.where(df.score1_game > df.score2_game, df[a], df[b])
        df["sc"] = [tuple(sorted(x)) for x in zip(df.score1_game, df.score2_game)]
    j = ov[["pair", "map_name", "day", "w", "sc", "event"]].merge(kg[["pair", "map_name", "day", "w", "sc"]],
                                                                 on=["pair", "map_name"], suffixes=("_lp", "_kg"))
    j = j[(j.day_lp - j.day_kg).abs() <= pd.Timedelta(days=1)].drop_duplicates(["pair", "map_name", "day_lp"])
    print(f"(a) Jan-Jun 2026 overlap: {len(ov)} Liquipedia maps with both teams known; {len(j)} found in Kaggle "
          f"(same pair+map within 1 day). Winner agrees {(j.w_lp == j.w_kg).mean():.1%}, "
          f"score agrees {(j.sc_lp == j.sc_kg).mean():.1%}.")
    bad = j[(j.w_lp != j.w_kg) | (j.sc_lp != j.sc_kg)]
    if len(bad):
        print(bad[["day_lp", "pair", "map_name", "sc_lp", "sc_kg", "w_lp", "w_kg", "event"]].head(15).to_string())
    found = ov.set_index(["pair", "map_name", "day"]).index.isin(j.set_index(["pair", "map_name", "day_lp"]).index)
    print("\n(b) share of Liquipedia Jan-Jun maps (teams known) that Kaggle has, by Liquipedia tier:")
    print(ov.assign(in_kaggle=found).groupby("tier_lp").in_kaggle.agg(["size", "mean"]).round(3).to_string())
    lpj = lpk[lpk.datetime >= P.TEST_START]
    print("\nmaps by tier, 2026: Kaggle", base[base.datetime >= P.TEST_START].tier.value_counts(normalize=True)
          .round(3).sort_index().to_dict(), "| Liquipedia", lpj.tier.value_counts(normalize=True).round(3)
          .sort_index().to_dict())
    print("listed-first win rate: Kaggle 2026H1 %.3f | Liquipedia 2026 %.3f | Liquipedia from cut %.3f" % (
        base.y[te].mean(), (lpj.score1_game > lpj.score2_game).mean(), (late.score1_game > late.score2_game).mean()))


def v2_history():
    """Kaggle v5 (through Oct 2026) + Liquipedia maps Kaggle doesn't have, teams renamed to Kaggle names."""
    d = P.load(V5)
    lp = pd.read_csv(LP, parse_dates=["datetime"])
    lp = lp[(lp[["score1_game", "score2_game"]].max(axis=1) >= 13) & (lp.score1_game != lp.score2_game)].copy()
    known = {}
    for n in pd.concat([d.team1, d.team2]):
        known.setdefault(key(n), n)
    for t in ("team1", "team2"):
        lp[t] = lp[t].map(lambda x: known.get(key(x), x))
        lp[t + "_id"] = lp[t]
    lp["y"] = (lp.score1_game > lp.score2_game).astype(int)
    lp["match_id"] = lp.datetime.astype(str) + lp.team1 + lp.team2
    # Drop Liquipedia maps Kaggle already has (same unordered pair + map within a day).
    kk = {(frozenset(p), m, day) for p, m, day in zip(zip(d.team1, d.team2), d.map_name, d.datetime.dt.normalize())}
    dup = [any((frozenset(p), m, day + pd.Timedelta(days=o)) in kk for o in (-1, 0, 1))
           for p, m, day in zip(zip(lp.team1, lp.team2), lp.map_name, lp.datetime.dt.normalize())]
    lp = lp[~np.array(dup) & (lp.datetime >= d.datetime.min())]
    print(f"v2 history: {len(d)} Kaggle + {len(lp)} extra Liquipedia maps")
    return pd.concat([d, lp], ignore_index=True).sort_values(["datetime", "match_id", "map_no"], kind="stable") \
        .reset_index(drop=True)


V2 = P.MODEL.with_name("prematch_v2.pkl")


def v2():
    """Retrain logreg on all data, saved alongside the frozen v1 model."""
    hist = v2_history()
    X = features(hist)
    tr = (hist.datetime >= P.BURN_IN_END).values
    Xs, ys = P.symmetrize(X[tr], hist.y.values[tr])
    logreg = P.make_pipeline(P.StandardScaler(), P.LogisticRegression(max_iter=1000)).fit(Xs.fillna(0), ys)
    pickle.dump({"logreg": logreg, "best": "logreg", "history": "kaggle_v5+liquipedia"}, open(V2, "wb"))
    print("saved", V2, "coefs:", dict(zip(P.DIFF + P.CTX, logreg[-1].coef_[0].round(3))))


def predict_v2(team_a, team_b, map_name, date, tier=1, best_of=3):
    hist, date = v2_history(), pd.Timestamp(date)
    st, _ = P.build(hist, until=date)
    known = {t.lower(): t for t in pd.concat([hist.team1, hist.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    X = pd.DataFrame([st.features(a, b, map_name, date)]).assign(tier=tier, bestOf=best_of)[P.DIFF + P.CTX]
    p = P.sym_proba(pickle.load(open(V2, "rb"))["logreg"], X.fillna(0))[0]
    return {team_a: float(p), team_b: float(1 - p)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["v2"]:
        v2()
    elif sys.argv[1:2] == ["predict"]:
        print(predict_v2(*sys.argv[2:6]))
    else:
        main()
