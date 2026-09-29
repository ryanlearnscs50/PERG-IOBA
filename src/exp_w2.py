"""Week-2 experiments: literature-protocol leakage check, stronger small-data feature models (MiniRocket, DWT,
functional PCA), and per-diagnosis detectability analysis from out-of-fold predictions.

python exp_w2.py leak | feats | diag
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import perg_features, raw_features
from data import cv_splits, load
from metrics import binary_metrics, evaluate

warnings.filterwarnings("ignore")
RUNS = Path(__file__).resolve().parents[1] / "runs"
FS_HZ = 1700.0


def summarize(res):
    df = pd.DataFrame(res)
    return {k: f"{df[k].mean():.3f}±{df[k].std():.3f}" for k in df}


# ---------------------------------------------------------------- leakage check (literature-style protocols)
def leak():
    """Same PCA-LR / HGB model under (a) our subject-grouped record-level CV, (b) record-level CV ignoring repeat
    visits, (c) eye-level random CV, (d) single-trace (repetition x eye) random CV as in several published works."""
    D = load()
    info, reps = D["info"], D["reps"]
    y = info.y_bin.values
    rows = {}
    mk = {"pca_lr": lambda: make_pipeline(StandardScaler(), PCA(20), LogisticRegression(C=0.3, max_iter=5000,
                                                                                        class_weight="balanced")),
          "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                        class_weight="balanced")}
    # trace-level samples
    tx, ty, trec = [], [], []
    for i, r in enumerate(reps):
        for k in range(r.shape[0]):
            for e in range(2):
                tx.append(r[k, e]), ty.append(y[i]), trec.append(i)
    tx = gaussian_filter1d(np.array(tx), 2.0, axis=-1)[:, ::3]
    ty, trec = np.array(ty), np.array(trec)
    ex = gaussian_filter1d(D["mean"].reshape(-1, 255), 2.0, axis=-1)[:, ::3]  # eye-level (rep-mean)
    ey, erec = np.repeat(y, 2), np.repeat(np.arange(len(y)), 2)
    Xr = raw_features(D["mean"], info)
    for name in mk:
        for proto in ["record_subject_grouped", "record_random", "eye_random", "eye_subject_grouped",
                      "trace_random", "trace_subject_grouped"]:
            res = []
            for seed in range(3):
                if proto.startswith("record"):
                    X, yy, g = Xr, y, info.group.values
                elif proto.startswith("eye"):
                    X, yy, g = ex, ey, info.group.values[erec]
                else:
                    X, yy, g = tx, ty, info.group.values[trec]
                if proto.endswith("grouped"):
                    spl = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yy, g)
                else:
                    spl = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy)
                p = np.zeros(len(yy))
                for tr, te in spl:
                    p[te] = mk[name]().fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
                acc = ((p > 0.5) == yy).mean()
                res.append({"auroc": roc_auc_score(yy, p), "acc": acc})
            rows[f"{name}/{proto}"] = summarize(res)
            print(name, proto, rows[f"{name}/{proto}"], flush=True)
    json.dump(rows, open(RUNS / "w2_leak.json", "w"), indent=1)


# ---------------------------------------------------------------- feature models
def sym_pool(F):
    """F: (N, 2, f) per-eye features -> eye-order-invariant (mean, max, min, |diff|)."""
    return np.concatenate([F.mean(1), F.max(1), F.min(1), np.abs(F[:, 0] - F[:, 1])], 1)


def dwt_feats(mean):
    """Per-eye DWT (db4, 5 levels) sub-band energies + band-limited time-domain stats (literature-style)."""
    x = mean.reshape(-1, 255).astype(np.float64)
    out = []
    coeffs = pywt.wavedec(x, "db4", level=5, axis=-1)
    for c in coeffs:
        out += [np.log1p((c ** 2).sum(1)), np.abs(c).max(1), c.std(1)]
    for lo, hi in [(1, 15), (15, 40), (40, 100)]:
        s = sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=FS_HZ, output="sos"), x, axis=-1)
        out += [s.std(1), s.max(1), s.min(1), s.argmax(1) / FS_HZ, s.argmin(1) / FS_HZ,
                np.abs(np.diff(np.sign(s), axis=1)).sum(1), np.abs(np.diff(s, axis=1)).sum(1)]
    return np.stack(out, 1).reshape(len(mean), 2, -1)


class FPCA:
    """Functional PCA per eye (fit on train eyes only) -> eye-symmetric pooling of scores."""

    def __init__(self, k=12):
        self.k = k

    def fit(self, mean):
        self.p = PCA(self.k).fit(gaussian_filter1d(mean, 2.0, axis=-1).reshape(-1, 255))
        return self

    def transform(self, mean):
        s = self.p.transform(gaussian_filter1d(mean, 2.0, axis=-1).reshape(-1, 255)).reshape(len(mean), 2, -1)
        return sym_pool(s)


class MiniRocketEyes:
    """MiniRocket kernels shared across eyes (fit on train eyes) -> eye-symmetric pooling."""

    def __init__(self, seed=0, n_kernels=10000):
        from aeon.transformations.collection.convolution_based import MiniRocket
        import os
        self.t = MiniRocket(n_kernels=n_kernels, random_state=seed, n_jobs=len(os.sched_getaffinity(0)))

    def fit(self, mean):
        self.t.fit(mean.reshape(-1, 1, 255).astype(np.float64))
        return self

    def transform(self, mean):
        f = np.asarray(self.t.transform(mean.reshape(-1, 1, 255).astype(np.float64)))
        return sym_pool(f.reshape(len(mean), 2, -1))


def tuned_lr(Xtr, ytr, gtr, seed, Cs=(1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)):
    """L2-LR with C chosen by inner subject-grouped 3-fold CV (nested; outer test fold never seen)."""
    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr, gtr))
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegressionCV(Cs=list(Cs), cv=inner, scoring="roc_auc_ovr" if len(set(ytr)) > 2 else "roc_auc",
                             class_weight="balanced", max_iter=3000, n_jobs=-1).fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict_proba(sc.transform(X))


def feature_run(task, feat, seeds=(0, 1, 2)):
    D = load()
    info, mean = D["info"], D["mean"]
    y = info["y_bin" if task == "bin" else "y_grp"].values
    g = info.group.values
    meta = info[["age_years", "sex_i"]].values
    res, P_all = [], []
    for seed in seeds:
        P = np.zeros((len(y), 2 if task == "bin" else 4))
        for tr, te in cv_splits(info, seed=seed):
            blocks_tr, blocks_te = [], []
            parts = feat.split("+")
            for part in parts:
                if part == "perg":
                    blocks_tr.append(perg_features(mean[tr], info.iloc[tr])[:, :-2])
                    blocks_te.append(perg_features(mean[te], info.iloc[te])[:, :-2])
                elif part == "raw":
                    blocks_tr.append(raw_features(mean[tr], info.iloc[tr])[:, :-2])
                    blocks_te.append(raw_features(mean[te], info.iloc[te])[:, :-2])
                elif part == "dwt":
                    blocks_tr.append(sym_pool(dwt_feats(mean[tr])))
                    blocks_te.append(sym_pool(dwt_feats(mean[te])))
                elif part == "fpca":
                    t = FPCA().fit(mean[tr])
                    blocks_tr.append(t.transform(mean[tr])), blocks_te.append(t.transform(mean[te]))
                elif part == "mrocket":
                    t = MiniRocketEyes(seed).fit(mean[tr])
                    blocks_tr.append(t.transform(mean[tr])), blocks_te.append(t.transform(mean[te]))
                elif part == "meta":
                    blocks_tr.append(meta[tr]), blocks_te.append(meta[te])
            Xtr, Xte = np.concatenate(blocks_tr, 1), np.concatenate(blocks_te, 1)
            P[te] = tuned_lr(Xtr, y[tr], g[tr], seed)(Xte)
        P_all.append(P)
        m = evaluate(task, y, P[:, 1] if task == "bin" else P)
        if task == "grp":
            m.update({f"screen_{k}": v for k, v in binary_metrics(info.y_bin.values, 1 - P[:, 0]).items()})
        res.append(m)
        print(task, feat, seed, {k: round(v, 3) for k, v in m.items()}, flush=True)
    out = RUNS / f"w2_{task}_{feat}"
    out.mkdir(exist_ok=True, parents=True)
    for s, P in zip(seeds, P_all):
        np.save(out / f"oof_seed{s}.npy", P)
    json.dump({"per_seed": res, "summary": summarize(res)}, open(out / "result.json", "w"), indent=1)
    return summarize(res)


def feats():
    rows = {}
    for task in ["bin", "grp"]:
        for feat in sys.argv[2].split(",") if len(sys.argv) > 2 else \
                ["raw", "fpca", "dwt", "perg+fpca", "perg+dwt+fpca", "mrocket", "mrocket+perg+fpca"]:
            rows[f"{task}/{feat}"] = feature_run(task, feat)
            print("SUMMARY", task, feat, rows[f"{task}/{feat}"], flush=True)
    json.dump(rows, open(RUNS / f"w2_feats_{'_'.join(sys.argv[2:]) or 'all'}.json".replace(",", "-"), "w"),
              indent=1)


# ---------------------------------------------------------------- per-diagnosis detectability
def diag():
    """Which diagnoses does the screening score actually separate from Normal? Uses seed-averaged OOF scores."""
    D = load()
    info = D["info"]
    y = info.y_bin.values
    run_dirs = sys.argv[2].split(",")
    for rd in run_dirs:
        ps = [np.load(f) for f in sorted((RUNS / rd).glob("oof_seed*.npy"))]
        P = np.mean(ps, 0)
        s = P[:, 1] if P.shape[1] == 2 else 1 - P[:, 0]
        thr = np.quantile(s[y == 0], 0.9)  # 90 % specificity on normals
        print(f"\n== {rd}: screening AUROC {roc_auc_score(y, s):.3f}; threshold @90%spec = {thr:.3f}")
        rows = []
        for d, idx in info[y == 1].groupby("diagnosis1").groups.items():
            idx = np.array([info.index.get_loc(i) for i in idx])
            if len(idx) < 3:
                continue
            auc = roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones(len(idx))], np.r_[s[y == 0], s[idx]])
            rows.append((d, len(idx), auc, (s[idx] > thr).mean()))
        df = pd.DataFrame(rows, columns=["diagnosis", "n", "auroc_vs_normal", "sens@90spec"]).sort_values("n",
                                                                                                         ascending=False)
        print(df.to_string(index=False, float_format="%.2f"))
        va = info[["va_re_logMar", "va_le_logMar"]].min(1)  # better eye
        for lab, m in [("better-eye VA logMAR<=0.1", (y == 1) & (va <= 0.1)),
                       ("better-eye VA logMAR>0.1", (y == 1) & (va > 0.1)), ("VA missing", (y == 1) & va.isna())]:
            m = m.values
            if m.sum() >= 3:
                auc = roc_auc_score(np.r_[np.zeros((y == 0).sum()), np.ones(m.sum())], np.r_[s[y == 0], s[m]])
                print(f"  abnormal with {lab}: n={m.sum()}, AUROC vs normal {auc:.3f}, sens@90spec "
                      f"{(s[m] > thr).mean():.2f}")
        df.to_csv(RUNS / f"w2_diag_{rd}.csv", index=False)


if __name__ == "__main__":
    {"leak": leak, "feats": feats, "diag": diag}[sys.argv[1]]()
