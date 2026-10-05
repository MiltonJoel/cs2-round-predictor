"""python3 test_prematch.py — leakage + symmetry checks on a small sample."""
import numpy as np
import pandas as pd
import prematch as P

d = P.load().head(2000)
_, full = P.build(d)
_, part = P.build(d.head(1000))
# Features of a map must not change when later maps are removed (no look-ahead).
pd.testing.assert_frame_equal(full.head(1000), part)
# Maps within one match share the pre-match state: Elo diff must not move inside a match for the same pairing.
g = full.assign(m=d.match_id, a=d.team1).groupby(["m", "a"]).elo.nunique()
assert (g == 1).all()
X = full.assign(tier=d.tier, bestOf=d.bestOf)[P.DIFF + P.CTX]
Xs, ys = P.symmetrize(X, d.y.values)
m = P.LogisticRegression(max_iter=1000).fit(Xs.fillna(0), ys)
p = P.sym_proba(m, X.fillna(0))
flip = X.copy(); flip[P.DIFF] = -flip[P.DIFF]
assert np.allclose(p + P.sym_proba(m, flip.fillna(0)), 1)
print("ok")
