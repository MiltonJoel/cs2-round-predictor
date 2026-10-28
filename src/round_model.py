"""Round model from demos (docs/plan.md step 3): P(CT wins round | economy, side, score, round, map).

    python3 src/demos.py          # first: parse demos -> data/demos/rounds.csv
    python3 src/round_model.py    # cross-validated by series (no rounds of a held-out series in training)
Baseline = CT win rate per map from the training folds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oos_eval import ROOT

ROUNDS = ROOT / "data/demos/rounds.csv"


def features(r):
    r = r.sort_values(["file", "round"]).copy()
    r["y"] = (r.winner_side == "CT").astype(int)
    r["ct_won_prev"] = r.groupby("file").y.shift().fillna(0.5)
    # score from the CT team's perspective, before the round
    ct_pts, t_pts = [], []
    for _, g in r.groupby("file", sort=False):
        score = {}
        for row in g.itertuples():
            ct_pts.append(score.get(row.ct_team, 0))
            t_pts.append(score.get(row.t_team, 0))
            score[row.winner] = score.get(row.winner, 0) + 1
    r["score_diff"] = np.array(ct_pts) - np.array(t_pts)
    r["pistol"] = r["round"].isin([1, 13]).astype(int)
    r["overtime"] = (r["round"] > 24).astype(int)
    r["eq_diff"] = (r.ct_equip - r.t_equip) / 1000
    r["eq_ratio"] = np.log((r.ct_equip + 1000) / (r.t_equip + 1000))
    r["money_diff"] = (r.ct_money - r.t_money) / 1000
    r["ct_eq"], r["t_eq"] = r.ct_equip / 1000, r.t_equip / 1000
    for m in sorted(r["map"].unique()):
        r["map_" + m] = (r["map"] == m).astype(int)
    return r


def main():
    r = pd.read_csv(ROUNDS)
    r = features(r[r.complete])
    r = r.dropna(subset=["ct_equip", "t_equip"])  # rounds where the demo started after the buy phase
    group = np.where(r.series.fillna("") != "", r.series.fillna(""), r.file)
    cols = ["eq_diff", "eq_ratio", "money_diff", "ct_eq", "t_eq", "pistol", "overtime", "score_diff",
            "ct_won_prev"] + [c for c in r if c.startswith("map_")]
    y = r.y.values
    print(f"{r.file.nunique()} maps, {pd.Series(group).nunique()} series, {len(r)} rounds, CT won {y.mean():.1%}")
    preds = {k: np.zeros(len(r)) for k in ("CT rate per map", "logistic", "gradient boosting")}
    for tr, te in GroupKFold(n_splits=min(5, pd.Series(group).nunique())).split(r, y, group):
        rate = r.iloc[tr].groupby("map").y.mean()
        preds["CT rate per map"][te] = r.iloc[te]["map"].map(rate).fillna(y[tr].mean()).values
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(r.iloc[tr][cols], y[tr])
        preds["logistic"][te] = lr.predict_proba(r.iloc[te][cols])[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=40,
                                            random_state=0).fit(r.iloc[tr][cols], y[tr])
        preds["gradient boosting"][te] = gb.predict_proba(r.iloc[te][cols])[:, 1]
    print(pd.DataFrame([{"model": k, "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
                         "auc": roc_auc_score(y, p), "acc": ((p > .5) == y).mean()} for k, p in preds.items()])
          .set_index("model").round(4).to_string())
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(r[cols], y)
    print("\nlogistic coefficients (standardized, + favours CT):")
    print(pd.Series(lr[-1].coef_[0], cols).round(3).sort_values().to_string())
    b = pd.cut(r.eq_diff, [-99, -10, -3, 3, 10, 99], labels=["CT much poorer", "CT poorer", "even", "CT richer",
                                                             "CT much richer"])
    print("\nCT win rate by equipment gap (non-pistol):")
    print(r[r.pistol == 0].groupby(b[r.pistol == 0], observed=True).y.agg(["size", "mean"]).round(3).to_string())
    print("\nCT win rate by map:")
    print(r.groupby("map").y.agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
