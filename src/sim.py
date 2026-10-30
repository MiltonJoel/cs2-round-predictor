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
import argparse, csv, ctypes, hashlib, importlib.util, os, pickle, re, subprocess, sys, tempfile, types
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_fast_c_lib():
    c_path = Path(__file__).resolve().with_name("_sim_fast.c")
    np_spec = importlib.util.find_spec("numpy")
    np_dir = Path(np_spec.submodule_search_locations[0])
    # numpy's random/math code is linked in statically, so a different numpy needs its own build
    h = hashlib.sha256(c_path.read_bytes() + str(np_dir.resolve()).encode()
                       + (np_dir / "version.py").read_bytes()).hexdigest()[:16]
    so_path = Path(tempfile.gettempdir()) / f"_sim_fast_{h}.so"
    if not so_path.exists():
        npy_rnd = np_dir / "random/lib/libnpyrandom.a"
        # numpy 2.x keeps libnpymath.a under _core/, numpy 1.x under core/
        npy_math = next(p for p in (np_dir / "_core/lib/libnpymath.a", np_dir / "core/lib/libnpymath.a") if p.exists())
        tmp_so = so_path.with_name(f"_sim_fast_{h}_{os.getpid()}.so")
        subprocess.check_call([
            "gcc", "-O2", "-march=native", "-fno-fast-math", "-ffp-contract=off",
            "-shared", "-fPIC", str(c_path), str(npy_rnd), str(npy_math),
            "-o", str(tmp_so), "-lm", "-ldl",
        ])
        os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(str(so_path))
    lib.c_sim_core.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int64, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.c_sim_core.restype = ctypes.c_int
    lib.c_anchor_only.argtypes = [
        ctypes.c_char_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.c_anchor_only.restype = ctypes.c_int
    lib.c_calibrate_sigma.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_double,
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_double,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.c_calibrate_sigma.restype = ctypes.c_int
    lib.c_prematch_features.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.c_prematch_features.restype = ctypes.c_int
    return lib


_C_LIB = _load_fast_c_lib()

import numpy as np
import pandas as pd

import oos_eval as O
import prematch as P

ROUNDS = O.ROOT / "data/demos/rounds.csv"
SIM = P.MODEL.with_name("sim.pkl")
MAPS = ["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train"]
_MAP_TO_IDX = {mp: i for i, mp in enumerate(MAPS)}


def _find_openblas_so():
    libs_dir = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    if libs_dir.exists():
        matches = sorted(libs_dir.glob("*openblas*.so*"))
        if matches:
            return str(matches[0]).encode("utf-8")
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                if "openblas" in line and ".so" in line:
                    return line.split()[-1].encode("utf-8")
    except OSError:
        pass
    return b""


_OPENBLAS_SO = _find_openblas_so()
_LBFGSB_MOD = None


def _get_lbfgsb():
    global _LBFGSB_MOD
    if _LBFGSB_MOD is None:
        sp_spec = importlib.util.find_spec("scipy")
        opt_dir = Path(sp_spec.submodule_search_locations[0]) / "optimize"
        so_file = next(opt_dir.glob("_lbfgsb*.so"))
        spec = importlib.util.spec_from_file_location("_lbfgsb", so_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LBFGSB_MOD = mod
    return _LBFGSB_MOD


class _StubObj:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _LightUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("builtins", "collections") or module.startswith("numpy"):
            return super().find_class(module, name)
        return type(name, (_StubObj,), {})


class LogisticRegression:
    """Lightweight L-BFGS-B binary logistic regression matching scikit-learn 1.7.2 bit-for-bit."""

    def __init__(self, penalty="l2", *, dual=False, tol=1e-4, C=1.0, fit_intercept=True,
                 intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs",
                 max_iter=100, multi_class="deprecated", verbose=0, warm_start=False,
                 n_jobs=None, l1_ratio=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        n_samples, n_features = X_arr.shape
        classes = np.unique(y_arr)
        pos_class = classes[1]
        y_bin = np.where(y_arr == pos_class, 1.0, 0.0)
        l2_reg = 1.0 / (self.C * n_samples)

        _lbfgsb = _get_lbfgsb()
        m = 10
        n_dof = n_features + 1
        x = np.zeros(n_dof, dtype=np.float64)
        low_bnd = np.zeros(n_dof, dtype=np.float64)
        upper_bnd = np.zeros(n_dof, dtype=np.float64)
        nbd = np.zeros(n_dof, dtype=np.int32)
        f = np.array(0.0, dtype=np.float64)
        g = np.zeros(n_dof, dtype=np.float64)
        factr = 64.0  # ftol = 64 * eps => factr = ftol / eps = 64.0
        pgtol = float(self.tol)
        wa = np.zeros(2 * m * n_dof + 5 * n_dof + 11 * m * m + 8 * m, dtype=np.float64)
        iwa = np.zeros(3 * n_dof, dtype=np.int32)
        task = np.zeros(2, dtype=np.int32)
        ln_task = np.zeros(2, dtype=np.int32)
        lsave = np.zeros(4, dtype=np.int32)
        isave = np.zeros(44, dtype=np.int32)
        dsave = np.zeros(29, dtype=np.float64)
        maxls = 50
        maxfun = 15000
        n_iter = 0
        nfev = 0

        loss_pt = np.empty(n_samples, dtype=np.float64)
        grad_pt = np.empty(n_samples, dtype=np.float64)

        while True:
            _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol,
                           wa, iwa, task, lsave, isave, dsave, maxls, ln_task)
            t0 = task[0]
            if t0 == 3:
                nfev += 1
                w = x[:n_features]
                b = x[-1]
                raw = X_arr @ w + b
                m1 = raw <= -37.0
                m2 = (~m1) & (raw <= -2.0)
                m3 = (~m1) & (~m2) & (raw <= 18.0)
                m4 = ~(m1 | m2 | m3)
                if m1.any():
                    loss_pt[m1] = -y_bin[m1] * raw[m1]
                    grad_pt[m1] = -y_bin[m1]
                if m2.any():
                    ez2 = np.exp(raw[m2])
                    loss_pt[m2] = np.log1p(ez2) - y_bin[m2] * raw[m2]
                    grad_pt[m2] = ((1.0 - y_bin[m2]) * ez2 - y_bin[m2]) / (1.0 + ez2)
                if m3.any():
                    ez3 = np.exp(-raw[m3])
                    loss_pt[m3] = np.log1p(ez3) + (1.0 - y_bin[m3]) * raw[m3]
                    grad_pt[m3] = ((1.0 - y_bin[m3]) - y_bin[m3] * ez3) / (1.0 + ez3)
                if m4.any():
                    ez4 = np.exp(-raw[m4])
                    loss_pt[m4] = ez4 + (1.0 - y_bin[m4]) * raw[m4]
                    grad_pt[m4] = (1.0 - y_bin[m4]) - ez4
                f = np.array(loss_pt.sum() / n_samples + 0.5 * l2_reg * (w @ w), dtype=np.float64)
                grad_pt /= n_samples
                g[:n_features] = X_arr.T @ grad_pt + l2_reg * w
                g[-1] = grad_pt.sum()
            elif t0 == 1:
                n_iter += 1
                if n_iter >= self.max_iter:
                    task[0] = 5
                    task[1] = 504
                elif nfev > maxfun:
                    task[0] = 5
                    task[1] = 502
            else:
                break

        self.n_features_in_ = int(n_features)
        self.classes_ = classes
        self.n_iter_ = np.array([n_iter], dtype=np.int32)
        self.coef_ = x[:n_features].reshape(1, n_features).copy()
        self.intercept_ = np.array([x[-1]], dtype=np.float64)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        raw = X_arr @ self.coef_[0] + self.intercept_[0]
        p1 = 1.0 / (1.0 + np.exp(-raw))
        out = np.empty((X_arr.shape[0], 2), dtype=np.float64)
        out[:, 1] = p1
        out[:, 0] = 1.0 - p1
        return out


class LinearRegression:
    """Lightweight OLS linear regression matching scikit-learn 1.7.2 bit-for-bit."""

    def __init__(self, *, fit_intercept=True, copy_X=True, tol=1e-6, n_jobs=None, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive

    def fit(self, X, y):
        X_arr = np.array(X, dtype=np.float64, copy=True)
        y_arr = np.array(y, dtype=np.float64, copy=True)
        X_offset = np.average(X_arr, axis=0)
        X_arr -= X_offset
        y_offset = np.average(y_arr, axis=0)
        y_arr -= y_offset
        cond = max(X_arr.shape) * np.finfo(X_arr.dtype).eps
        coef, _, rank, singular = np.linalg.lstsq(X_arr, y_arr, rcond=cond)
        coef = np.ravel(coef.T)
        self.n_features_in_ = int(X_arr.shape[1])
        self.coef_ = coef
        self.rank_ = int(rank)
        self.singular_ = singular
        self.intercept_ = np.float64(y_offset - X_offset @ coef)
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_


def _dump_sim_model(model, path):
    """Serialize `model` to `path` with exact scikit-learn 1.7.2 pickle opcodes without importing sklearn."""
    if "sklearn.linear_model._logistic" in sys.modules and "sklearn.linear_model._base" in sys.modules:
        sk_lr_cls = sys.modules["sklearn.linear_model._logistic"].LogisticRegression
        sk_lin_cls = sys.modules["sklearn.linear_model._base"].LinearRegression
        rm = sk_lr_cls.__new__(sk_lr_cls)
        rm.__dict__.update(model["round"].__dict__)
        trans = {}
        for side, ln in model["trans"].items():
            obj = sk_lin_cls.__new__(sk_lin_cls)
            obj.__dict__.update(ln.__dict__)
            trans[side] = obj
        with open(path, "wb") as f:
            pickle.dump({"round": rm, "buy": model["buy"], "trans": trans, "sigma": model["sigma"]}, f)
        return

    m_log = types.ModuleType("sklearn.linear_model._logistic")
    m_base = types.ModuleType("sklearn.linear_model._base")
    def _lr_getstate(self):
        st = {k.encode().decode(): v for k, v in self._d.items()}
        st["_sklearn_version"] = "1.7.2"
        return st

    StubLR = type("LogisticRegression", (), {
        "__module__": "sklearn.linear_model._logistic",
        "__getstate__": _lr_getstate,
    })
    StubLin = type("LinearRegression", (), {
        "__module__": "sklearn.linear_model._base",
        "__getstate__": lambda self: {**self.__dict__, "_sklearn_version": "1.7.2"},
    })
    m_log.LogisticRegression = StubLR
    m_base.LinearRegression = StubLin

    rm = StubLR()
    rm._d = {k: v for k, v in model["round"].__dict__.items() if k != "_sklearn_version"}
    trans = {}
    for side, ln in model["trans"].items():
        obj = StubLin()
        obj.__dict__.update({sys.intern(k): v for k, v in ln.__dict__.items() if k != "_sklearn_version"})
        trans[side] = obj
    payload = {"round": rm, "buy": model["buy"], "trans": trans, "sigma": model["sigma"]}

    mod_names = ("sklearn", "sklearn.linear_model", "sklearn.linear_model._logistic", "sklearn.linear_model._base")
    saved = {k: sys.modules.get(k) for k in mod_names}
    try:
        if saved["sklearn"] is None:
            sys.modules["sklearn"] = types.ModuleType("sklearn")
        if saved["sklearn.linear_model"] is None:
            sys.modules["sklearn.linear_model"] = types.ModuleType("sklearn.linear_model")
        sys.modules["sklearn.linear_model._logistic"] = m_log
        sys.modules["sklearn.linear_model._base"] = m_base
        data = pickle.dumps(payload)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    Path(path).write_bytes(data)


def _group_kfold_split(n_samples, groups, n_splits):
    if n_splits < 2:
        raise ValueError(
            f"k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}."
        )
    groups_arr = np.asarray(groups)
    unique_groups, group_idx = np.unique(groups_arr, return_inverse=True)
    n_groups = len(unique_groups)
    if n_splits > n_groups:
        raise ValueError(f"Cannot have number of splits n_splits={n_splits} greater than the number of groups: {n_groups}.")
    n_samples_per_group = np.bincount(group_idx)
    indices = np.argsort(n_samples_per_group)[::-1]
    n_samples_per_group = n_samples_per_group[indices]
    n_samples_per_fold = np.zeros(n_splits)
    group_to_fold = np.zeros(n_groups)
    for group_index, weight in enumerate(n_samples_per_group):
        lightest_fold = np.argmin(n_samples_per_fold)
        n_samples_per_fold[lightest_fold] += weight
        group_to_fold[indices[group_index]] = lightest_fold
    fold_indices = group_to_fold[group_idx]
    all_idx = np.arange(n_samples)
    for f in range(n_splits):
        te_mask = fold_indices == f
        yield all_idx[~te_mask], all_idx[te_mask]


def norm(name):
    return O.key(re.sub(r"\(.*?\)|globant|clan", "", str(name).lower()))


# ---------- pre-match strength (frozen v1 weights on Kaggle + Liquipedia history) ----------
def prematch_strength(rounds):
    """{(file): logit P(first CT team wins map)} for demo maps found in the history; maps not found -> 0."""
    # Group demo rounds by file in sorted order
    fl_col = rounds["file"].tolist()
    ct_col = rounds["ct_team"].tolist()
    t_col = rounds["t_team"].tolist()
    mp_col = rounds["map"].tolist()
    wn_col = rounds["winner"].tolist()

    file_info = {}
    for fl, ct, t, mp, wn in zip(fl_col, ct_col, t_col, mp_col, wn_col):
        info = file_info.get(fl)
        if info is None:
            info = [str(ct), str(t), str(mp), 0, 0]
            file_info[fl] = info
        if wn == info[0]:
            info[3] += 1
        elif wn == info[1]:
            info[4] += 1

    files = sorted(file_info.keys())
    nf = len(files)
    c_char_p_arr = ctypes.c_char_p * max(nf, 1)
    ct_arr = c_char_p_arr(*(file_info[f][0].encode("utf-8") for f in files))
    t_arr = c_char_p_arr(*(file_info[f][1].encode("utf-8") for f in files))
    m_arr = c_char_p_arr(*(file_info[f][2].encode("utf-8") for f in files))
    c_int_arr = ctypes.c_int * max(nf, 1)
    wct_arr = c_int_arr(*(file_info[f][3] for f in files))
    wt_arr = c_int_arr(*(file_info[f][4] for f in files))

    X_fort = np.empty((max(nf, 1), 15), dtype=np.float64, order="F")
    found = np.zeros(max(nf, 1), dtype=np.int32)
    ct_first = np.zeros(max(nf, 1), dtype=np.int32)
    kg_cnt = ctypes.c_int(0)
    lp_cnt = ctypes.c_int(0)

    rc = _C_LIB.c_prematch_features(
        str(O.V5).encode("utf-8"), str(O.LP).encode("utf-8"), nf,
        ct_arr, t_arr, m_arr, wct_arr, wt_arr,
        X_fort.ctypes.data, found.ctypes.data, ct_first.ctypes.data,
        ctypes.byref(kg_cnt), ctypes.byref(lp_cnt),
    )
    if rc != 0:
        raise RuntimeError(f"c_prematch_features failed with code {rc}")
    print(f"v2 history: {kg_cnt.value} Kaggle + {lp_cnt.value} extra Liquipedia maps")

    if nf == 0:
        return {}

    with open(P.MODEL, "rb") as f:
        pm = _LightUnpickler(f).load()["logreg"]
    sc_mean, sc_scale = pm.steps[0][1].mean_, pm.steps[0][1].scale_
    clf_coef, clf_inter = pm.steps[1][1].coef_[0], pm.steps[1][1].intercept_[0]

    X_use = X_fort[:nf]
    flip = X_use.copy(order="F")
    flip[:, :12] = -flip[:, :12]

    Zs = np.asfortranarray((X_use - sc_mean) / sc_scale)
    p1 = 1.0 / (1.0 + np.exp(-(Zs @ clf_coef + clf_inter)))
    Zf = np.asfortranarray((flip - sc_mean) / sc_scale)
    p2 = 1.0 / (1.0 + np.exp(-(Zf @ clf_coef + clf_inter)))
    p_arr = (p1 + 1.0 - p2) / 2.0

    out = {}
    for i, fl in enumerate(files):
        if found[i]:
            q = p_arr[i] if ct_first[i] else 1.0 - p_arr[i]
            out[fl] = float(np.log(q / (1.0 - q)))
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


_ANCHOR_RNG_CACHE = None
_CALIB_RNG_CACHE = {}


def _get_anchor_rng():
    global _ANCHOR_RNG_CACHE
    if _ANCHOR_RNG_CACHE is None:
        rg = np.random.default_rng(7)
        z7 = rg.standard_normal(2000)
        u7 = rg.random((60, 2000))
        lu7 = np.log(u7 / (1.0 - u7))
        _ANCHOR_RNG_CACHE = (z7, u7, lu7)
    return _ANCHOR_RNG_CACHE


_EXACT_U_CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int64, ctypes.c_int, ctypes.c_int64)


@_EXACT_U_CB_TYPE
def _exact_u400_cb(seed_id, rnd, idx):
    rg = np.random.default_rng(int(seed_id))
    rg.standard_normal(400)
    u = rg.random((int(rnd), 400))
    return float(u[int(rnd) - 1, int(idx)])


def _get_calib_rng(n_seeds):
    n_seeds = min(int(n_seeds), 60)
    cached = _CALIB_RNG_CACHE.get("full")
    if cached is not None and cached[0].shape[0] >= n_seeds:
        return cached
    z400 = np.empty((n_seeds, 400), dtype=np.float64)
    lu400 = np.empty((n_seeds, 60, 400), dtype=np.float64)
    for i in range(n_seeds):
        rg = np.random.default_rng(i)
        z400[i] = rg.standard_normal(400)
        u = rg.random((60, 400))
        lu400[i] = np.log(u / (1.0 - u))
    _CALIB_RNG_CACHE["full"] = (z400, lu400)
    return z400, lu400


def _clear_rng_caches():
    global _ANCHOR_RNG_CACHE
    _ANCHOR_RNG_CACHE = None
    _CALIB_RNG_CACHE.clear()


def _extract_c_model_arrays(model):
    rm_coef = np.ascontiguousarray(model["round"].coef_[0], dtype=np.float64)
    rm_inter = float(model["round"].intercept_[0])
    bx_ct = np.ascontiguousarray(model["buy"]["ct"][0], dtype=np.float64)
    by_ct = np.ascontiguousarray(model["buy"]["ct"][1], dtype=np.float64)
    bx_t = np.ascontiguousarray(model["buy"]["t"][0], dtype=np.float64)
    by_t = np.ascontiguousarray(model["buy"]["t"][1], dtype=np.float64)
    tc_ct = np.ascontiguousarray(model["trans"]["ct"].coef_, dtype=np.float64)
    ti_ct = float(model["trans"]["ct"].intercept_)
    tc_t = np.ascontiguousarray(model["trans"]["t"].coef_, dtype=np.float64)
    ti_t = float(model["trans"]["t"].intercept_)
    return rm_coef, rm_inter, bx_ct, by_ct, bx_t, by_t, tc_ct, ti_ct, tc_t, ti_t


def fit(r):
    r = r.dropna(subset=["ct_equip", "t_equip"])
    X = x_round(r.eq_ct.values, r.eq_t.values, 0, 0, None, r.str_ct.values)
    X[:, 1], X[:, 2] = r.pistol, r.ot
    map_vals = r["map"].values
    for j, mp in enumerate(MAPS):
        X[:, 4 + j] = map_vals == mp
    rm = LogisticRegression(C=1.0, max_iter=2000).fit(X, r.y.values)

    # buy policy: median equipment per player by resource level, per side (non-pistol rounds)
    buy_pol = {}
    nb_mask = r.pistol.values == 0
    bins = np.arange(0, 17001, 500, dtype=np.float64)
    mids_all = bins[:-1] + 250.0
    for side in ("ct", "t"):
        rv = r["res_" + side].values[nb_mask]
        ev = r["eq_" + side].values[nb_mask]
        valid_b = (rv > 0.0) & (rv <= 17000.0) & (~np.isnan(ev))
        rv_v = rv[valid_b]
        ev_v = ev[valid_b]
        b_idx = np.searchsorted(bins, rv_v, side="left") - 1
        order = np.lexsort((ev_v, b_idx))
        b_sorted = b_idx[order]
        e_sorted = ev_v[order]
        if len(b_sorted):
            b_starts = np.r_[0, np.flatnonzero(b_sorted[1:] != b_sorted[:-1]) + 1]
            b_lens = np.diff(np.r_[b_starts, len(b_sorted)])
            u_bins = b_sorted[b_starts]
            lo_i = b_starts + (b_lens - 1) // 2
            hi_i = b_starts + b_lens // 2
            meds = 0.5 * (e_sorted[lo_i] + e_sorted[hi_i])
            buy_pol[side] = (mids_all[u_bins].copy(), meds)
        else:
            buy_pol[side] = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))

    # transition: next-round resources from this round (same half only)
    fl_vals = r["file"].values
    rnd_vals = r["round"].values
    is_sorted_unique = (
        len(r) == 0
        or bool(np.all((fl_vals[:-1] < fl_vals[1:]) | ((fl_vals[:-1] == fl_vals[1:]) & (rnd_vals[:-1] < rnd_vals[1:]))))
    )
    trans = {}
    if is_sorted_unique and len(r) > 1:
        valid = (fl_vals[:-1] == fl_vals[1:]) & (rnd_vals[1:] == rnd_vals[:-1] + 1) & (rnd_vals[:-1] < 24) & (rnd_vals[:-1] != 12)
        idx = np.flatnonzero(valid)
        ct_won_arr = (r["winner_side"].values[idx] == "CT").astype(np.float64)
        for side, won_v in (("ct", ct_won_arr), ("t", 1.0 - ct_won_arr)):
            res_cur = r["res_" + side].values[idx]
            eq_cur = r["eq_" + side].values[idx]
            nxt = r["res_" + side].values[idx + 1]
            Z = np.column_stack([res_cur - eq_cur, eq_cur * won_v, won_v])
            trans[side] = LinearRegression().fit(Z, nxt)
    else:
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
        for side in ("ct", "t"):
            d = tr[tr.side == side]
            Z = np.column_stack([d["left"], d["eq"] * d["won"], d["won"]])
            trans[side] = LinearRegression().fit(Z, d.nxt)

    model = {"round": rm, "buy": buy_pol, "trans": trans}

    # Fit the per-map strength spread so simulated loser-rounds match the training maps, re-anchoring each time.
    if is_sorted_unique and len(r) > 0:
        starts = np.r_[0, np.flatnonzero(fl_vals[1:] != fl_vals[:-1]) + 1]
        strengths = np.ascontiguousarray(r.str_ct.values[starts], dtype=np.float64)
        first_cts = np.repeat(r.ct_team.values[starts], np.diff(np.r_[starts, len(r)]))
        is_ct_w = (r.winner.values == first_cts).astype(np.int64)
        w1 = np.add.reduceat(is_ct_w, starts)
        lens = np.diff(np.r_[starts, len(r)])
        real = float(np.minimum(w1, lens - w1).mean())
    else:
        strengths = np.ascontiguousarray(r.groupby("file").str_ct.first().values, dtype=np.float64)
        real = float(r.groupby("file").apply(lambda g: min((g.winner == g.ct_team.iat[0]).sum(),
                                                           (g.winner != g.ct_team.iat[0]).sum()), include_groups=False).mean())

    rm_coef, rm_inter, bx_ct, by_ct, bx_t, by_t, tc_ct, ti_ct, tc_t, ti_t = _extract_c_model_arrays(model)
    z7, u7, lu7 = _get_anchor_rng()
    n_str = min(len(strengths), 60)
    z400, lu400 = _get_calib_rng(max(n_str, 1))
    out_sig = ctypes.c_double(0.0)
    out_c3 = ctypes.c_double(0.0)
    rc = _C_LIB.c_calibrate_sigma(
        _OPENBLAS_SO, rm_coef.ctypes.data, rm_inter,
        len(bx_ct), bx_ct.ctypes.data, by_ct.ctypes.data,
        len(bx_t), bx_t.ctypes.data, by_t.ctypes.data,
        tc_ct.ctypes.data, ti_ct, tc_t.ctypes.data, ti_t,
        n_str, strengths.ctypes.data, real,
        z7.ctypes.data, u7.ctypes.data, lu7.ctypes.data,
        z400.ctypes.data, lu400.ctypes.data, ctypes.cast(_exact_u400_cb, ctypes.c_void_p),
        ctypes.byref(out_sig), ctypes.byref(out_c3),
    )
    if rc != 0:
        raise RuntimeError(f"c_calibrate_sigma failed with code {rc}")
    model["sigma"], model["round"].coef_[0, 3] = float(out_sig.value), float(out_c3.value)
    return model


def anchor(model, target=1.0):
    """Scale the per-round strength coefficient so that, averaged over maps and starting sides, the simulated
    P(map win) for a team with pre-match logit `target` equals sigmoid(target)."""
    rm_coef, rm_inter, bx_ct, by_ct, bx_t, by_t, tc_ct, ti_ct, tc_t, ti_t = _extract_c_model_arrays(model)
    z7, u7, lu7 = _get_anchor_rng()
    out_c3 = ctypes.c_double(0.0)
    rc = _C_LIB.c_anchor_only(
        _OPENBLAS_SO, float(target),
        rm_coef.ctypes.data, rm_inter, float(model.get("sigma", 0.0)),
        len(bx_ct), bx_ct.ctypes.data, by_ct.ctypes.data,
        len(bx_t), bx_t.ctypes.data, by_t.ctypes.data,
        tc_ct.ctypes.data, ti_ct, tc_t.ctypes.data, ti_t,
        z7.ctypes.data, u7.ctypes.data, lu7.ctypes.data,
        ctypes.byref(out_c3),
    )
    if rc != 0:
        raise RuntimeError(f"c_anchor_only failed with code {rc}")
    model["round"].coef_[0, 3] = float(out_c3.value)
    return model


# ---------- simulation ----------
def buy(model, side, res):
    x, y = model["buy"][side]
    return np.minimum(np.interp(res, x, y), res)


def next_res(model, side, res, eq, won, losses):
    lin = model["trans"][side]
    base = lin.predict(np.column_stack([res - eq, eq * won, won]))
    bonus = np.where(won == 1, 0, (1400 + 500 * np.minimum(losses, 4)) - 2400)
    return np.clip(base + bonus, 0, 16000 + 6000)


def _simulate_eval(c_arrays, sigma, m, str_a, a_ct_first, n, seed):
    rm_coef, rm_inter, bx_ct, by_ct, bx_t, by_t, tc_ct, ti_ct, tc_t, ti_t = c_arrays
    rng = np.random.default_rng(seed)
    midx = _MAP_TO_IDX.get(m, -1)
    out_p = ctypes.c_double(0.0)
    out_lrm = ctypes.c_double(0.0)
    lr_arr = np.empty(n, dtype=np.int64)
    rc = _C_LIB.c_sim_core(
        _OPENBLAS_SO, rng.bit_generator.ctypes.bit_generator,
        midx, float(str_a), 1 if a_ct_first else 0, int(n), float(sigma),
        rm_coef.ctypes.data, rm_inter,
        len(bx_ct), bx_ct.ctypes.data, by_ct.ctypes.data,
        len(bx_t), bx_t.ctypes.data, by_t.ctypes.data,
        tc_ct.ctypes.data, ti_ct, tc_t.ctypes.data, ti_t,
        2, ctypes.byref(out_p), ctypes.byref(out_lrm),
        lr_arr.ctypes.data, 0, 0, 0, 0, 0,
    )
    if rc != 0:
        raise RuntimeError(f"c_sim_core failed with code {rc}")
    return float(out_p.value), lr_arr


def simulate(model, m, str_a, a_ct_first=True, n=10000, seed=0):
    """Team A vs B on map m. str_a = logit pre-match P(A wins map). Returns summary dict.
    Each simulated map draws its own strength (str_a + sigma*z): real teams' form varies map to map, and without
    this the simulator produces too many close scorelines."""
    rm_coef, rm_inter, bx_ct, by_ct, bx_t, by_t, tc_ct, ti_ct, tc_t, ti_t = _extract_c_model_arrays(model)
    sigma = float(model.get("sigma", 0.0))
    rng = np.random.default_rng(seed)
    midx = _MAP_TO_IDX.get(m, -1)
    out_p = ctypes.c_double(0.0)
    out_lrm = ctypes.c_double(0.0)
    lr_arr = np.empty(n, dtype=np.int64)
    sa = np.empty(n, dtype=np.int64)
    sb = np.empty(n, dtype=np.int64)
    h1 = np.zeros(2, dtype=np.float64)
    has_rnd = np.zeros(61, dtype=np.int32)
    win_rnd = np.zeros(61, dtype=np.float64)

    rc = _C_LIB.c_sim_core(
        _OPENBLAS_SO, rng.bit_generator.ctypes.bit_generator,
        midx, float(str_a), 1 if a_ct_first else 0, int(n), sigma,
        rm_coef.ctypes.data, rm_inter,
        len(bx_ct), bx_ct.ctypes.data, by_ct.ctypes.data,
        len(bx_t), bx_t.ctypes.data, by_t.ctypes.data,
        tc_ct.ctypes.data, ti_ct, tc_t.ctypes.data, ti_t,
        3, ctypes.byref(out_p), ctypes.byref(out_lrm),
        lr_arr.ctypes.data, sa.ctypes.data, sb.ctypes.data,
        h1.ctypes.data, has_rnd.ctypes.data, win_rnd.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"c_sim_core failed with code {rc}")

    lines = pd.Series([f"{x}-{y}" for x, y in zip(sa, sb)]).value_counts(normalize=True)
    p_p1 = np.float64(win_rnd[1])
    p_p13 = np.float64(win_rnd[13]) if has_rnd[13] else np.nan
    p_round = {r_i: round(np.float64(win_rnd[r_i]), 3) for r_i in range(1, 25) if has_rnd[r_i]}
    return {
        "p_win": float(out_p.value),
        "scorelines": lines.head(8).round(3).to_dict(),
        "half1_expected": (round(np.float64(h1[0]), 2), round(np.float64(h1[1]), 2)),
        "p_pistol1": p_p1,
        "p_pistol13": p_p13,
        "p_round": p_round,
        "loser_rounds": lr_arr,
    }


# ---------- evaluation ----------
def evaluate(n=3000, out=None, folds=5, max_maps=None):
    """Held-out-series evaluation. Writes summary.txt, per_map.csv, calibration.csv, scorelines.csv to `out`."""
    out = Path(out or O.ROOT / "results")
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROUNDS)
    strength = prematch_strength(raw[raw.complete])
    r = prepare(raw, strength)
    del raw
    maps = r.groupby("file").agg(series=("series", "first"), m=("map", "first"), ct=("ct_team", "first"),
                                 t=("t_team", "first")).reset_index()
    maps["series"] = maps.series.fillna(maps.file)
    if max_maps:  # smoke test
        maps = maps[maps.series.isin(maps.series.drop_duplicates().head(max_maps))]
        r = r[r.file.isin(maps.file)]

    # Vectorized per-file winner and loser_rounds from sorted `r`
    fl_vals = r["file"].values
    starts = np.r_[0, np.flatnonzero(fl_vals[1:] != fl_vals[:-1]) + 1]
    u_files = fl_vals[starts]
    lens = np.diff(np.r_[starts, len(r)])
    first_cts = np.repeat(r.ct_team.values[starts], lens)
    ct_w_cnt = np.add.reduceat((r.winner.values == first_cts).astype(np.int64), starts)
    not_ct_w_cnt = lens - ct_w_cnt
    final = dict(zip(u_files, ct_w_cnt > not_ct_w_cnt))
    file_real_lr = dict(zip(u_files, np.minimum(ct_w_cnt, not_ct_w_cnt)))

    maps["y"] = maps.file.map(final).astype(int)
    maps["str"] = maps.file.map(strength)
    maps["known"] = maps["str"].notna()
    maps["p_v1"] = 1 / (1 + np.exp(-maps["str"].fillna(0)))
    maps["p_sim"] = np.nan

    n_splits = min(folds, maps.series.nunique())
    real_lr = []
    sim_lr_hist = np.zeros(64, dtype=np.int64)
    map_files = maps.file.values
    map_ms = maps.m.values
    map_strs = maps["str"].values
    map_known = maps.known.values

    for tr, te in _group_kfold_split(len(maps), maps.series.values, n_splits):
        model = fit(r[r.file.isin(map_files[tr])])
        c_arrays = _extract_c_model_arrays(model)
        sigma = float(model.get("sigma", 0.0))
        for i in te:
            p_win, lr_arr = _simulate_eval(
                c_arrays, sigma, map_ms[i], float(map_strs[i]) if map_known[i] else 0.0, True, n=n, seed=int(i)
            )
            maps.loc[maps.index[i], "p_sim"] = p_win
            bc = np.bincount(lr_arr, minlength=64)
            sim_lr_hist += bc[:64]
            real_lr.append(int(file_real_lr[map_files[i]]))

    final_model = fit(r)
    _clear_rng_caches()

    ll = lambda y, p: -np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    rng = np.random.default_rng(0)
    lines = [f"{len(maps)} maps ({maps.known.sum()} with pre-match strength), {maps.series.nunique()} series, "
             f"{n_splits}-fold by series, {n} sims per map"]
    for name, d in (("all maps", maps), ("maps with strength", maps[maps.known])):
        lines.append(f"\n{name} (n={len(d)}):")
        for col in ("p_sim", "p_v1"):
            lines.append(f"  {col:6s} log loss {ll(d.y, d[col]):.4f}  Brier {np.mean((d[col] - d.y) ** 2):.4f}  "
                         f"acc {((d[col] > .5) == d.y).mean():.1%}")
        yv = d.y.values
        ps_c = np.clip(d.p_sim.values, 1e-6, 1)
        pv_c = np.clip(d.p_v1.values, 1e-6, 1)
        pt_sim = -(yv * np.log(ps_c) + (1 - yv) * np.log(np.clip(1 - d.p_sim.values, 1e-6, 1)))
        pt_v1 = -(yv * np.log(pv_c) + (1 - yv) * np.log(np.clip(1 - d.p_v1.values, 1e-6, 1)))
        diff = [pt_sim[i].mean() - pt_v1[i].mean()
                for i in [rng.integers(0, len(d), len(d)) for _ in range(2000)]]
        lines.append(f"  sim minus v1 log loss: {ll(d.y, d.p_sim) - ll(d.y, d.p_v1):+.4f}, 95% bootstrap CI "
                     f"[{np.percentile(diff, 2.5):+.4f}, {np.percentile(diff, 97.5):+.4f}]; coin flip 0.6931")
    b = pd.cut(maps.p_sim, [0, .3, .4, .5, .6, .7, 1])
    cal = maps.groupby(b, observed=True).agg(n=("y", "size"), pred=("p_sim", "mean"), actual=("y", "mean")).round(3)

    tot_sim_lr = int(sim_lr_hist.sum())
    nz = np.flatnonzero(sim_lr_hist)
    simd = pd.Series(sim_lr_hist[nz] / tot_sim_lr, index=nz)
    sim_lr_mean = float((np.arange(64, dtype=np.int64) * sim_lr_hist).sum()) / tot_sim_lr
    reald = pd.Series(real_lr).value_counts(normalize=True).sort_index()
    sc = pd.DataFrame({"real_demo_maps": reald, "simulated": simd}).fillna(0)

    k_mins = []
    with open(O.V5 / "cs2_all_tiers_games.csv", newline="", encoding="utf-8") as kf:
        rdr = csv.reader(kf)
        hdr = next(rdr)
        i_tot = hdr.index("is_total")
        i_dt = hdr.index("datetime")
        i_s1 = hdr.index("score1_game")
        i_s2 = hdr.index("score2_game")
        for row in rdr:
            if row[i_tot] == "True":
                continue
            dt_s, s1_s, s2_s = row[i_dt], row[i_s1], row[i_s2]
            if not dt_s or dt_s < "2026-01-01" or not s1_s or not s2_s:
                continue
            s1_v, s2_v = float(s1_s), float(s2_s)
            if (s1_v if s1_v >= s2_v else s2_v) >= 13.0:
                k_mins.append(s1_v if s1_v <= s2_v else s2_v)
    k_mins_arr = np.array(k_mins, dtype=np.float64)
    sc["real_kaggle_2026"] = pd.Series(k_mins_arr.astype(int)).value_counts(normalize=True)
    sc = sc.fillna(0).round(3).loc[lambda d: d.index <= 16]
    lines += ["\nreliability (sim):", cal.to_string(), "\nloser's rounds per map:", sc.to_string(),
              f"\nmean loser rounds: demo maps {np.mean(real_lr):.2f}, Kaggle 2026 ({len(k_mins_arr)} maps) "
              f"{k_mins_arr.mean():.2f}, simulated {sim_lr_mean:.2f}"]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    maps.to_csv(out / "per_map.csv", index=False)
    cal.to_csv(out / "calibration.csv")
    sc.to_csv(out / "scorelines.csv")
    _dump_sim_model(final_model, out / "sim.pkl")
    if Path(SIM).parent.exists():
        _dump_sim_model(final_model, SIM)
    print("wrote", out)


def cli(team_a, team_b, m, date, ct="A", n=10000):
    """Pre-match strength from the frozen v1 model at `date`, then simulate (A starts CT unless --ct B)."""
    with open(SIM, "rb") as f:
        model = _LightUnpickler(f).load()
    hist, date = O.v2_history(), pd.Timestamp(date)
    st, _ = P.build(hist, until=date)
    known = {t.lower(): t for t in pd.concat([hist.team1, hist.team2])}
    a, b = known[team_a.lower()], known[team_b.lower()]
    X = pd.DataFrame([st.features(a, b, m, date)]).assign(tier=1, bestOf=3)[P.DIFF + P.CTX]
    with open(P.MODEL, "rb") as f:
        pm = _LightUnpickler(f).load()["logreg"]
    sc_mean, sc_scale = pm.steps[0][1].mean_, pm.steps[0][1].scale_
    clf_coef, clf_inter = pm.steps[1][1].coef_[0], pm.steps[1][1].intercept_[0]
    Xv = np.asfortranarray(X.fillna(0).values, dtype=np.float64)
    Xf = Xv.copy(order="F")
    Xf[:, :12] = -Xf[:, :12]
    p1 = 1.0 / (1.0 + np.exp(-(((Xv - sc_mean) / sc_scale) @ clf_coef + clf_inter)))
    p2 = 1.0 / (1.0 + np.exp(-(((Xf - sc_mean) / sc_scale) @ clf_coef + clf_inter)))
    q = float(((p1 + 1.0 - p2) / 2.0)[0])
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
