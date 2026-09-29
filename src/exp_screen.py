"""Binary screening tasks (see data.SCREEN_TASKS) with classical models vs the raw-waveform PCA-LR baseline.
Same protocol everywhere: 5-fold subject-grouped CV x 3 seeds; every hyperparameter tuned by inner subject-grouped
3-fold CV on the training fold only (nested). Writes runs/screen/<task>/<model>/{oof_seed*.npy,result.json}.

python exp_screen.py <task|all> [model,model,...]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines import perg_features, raw_features
from data import SCREEN_TASKS, load, screen_task
from exp_w2 import FPCA, MiniRocketEyes, dwt_feats, sym_pool

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "runs" / "screen"
SEEDS = tuple(int(s) for s in __import__("os").environ.get("PERG_SEEDS", "0 1 2").split())


def screen_metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    yh = (p > 0.5).astype(int)
    return {"auroc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, yh), "acc": accuracy_score(y, yh),
            "sens": yh[y == 1].mean(), "spec": 1 - yh[y == 0].mean(), "sens@90spec": float(np.interp(0.10, fpr, tpr))}


# ---------------------------------------------------------------- features (fit on the training fold only)
def worse_better(mean):
    """Order the two eyes per record by P50 amplitude (worse eye first): disease is often asymmetric."""
    x = gaussian_filter1d(mean, 2.0, axis=-1)
    s = x[:, :, 40:130].max(-1) - x[:, :, 30:75].min(-1)  # crude per-eye P50 amplitude
    order = np.argsort(s, 1)
    return np.take_along_axis(mean, order[:, :, None], 1)


def rel_feats(reps):
    """Per eye: reproducibility across repetitions (mean pairwise corr, log SNR), plus inter-ocular correlation.
    A flat/absent PERG is dominated by noise, so repetitions agree poorly. Single-repetition records -> NaN."""
    out = np.full((len(reps), 2, 2), np.nan)
    io = np.zeros((len(reps), 1))
    for i, r in enumerate(reps):
        x = gaussian_filter1d(r.astype(np.float64), 2.0, axis=-1)  # (k, 2, T)
        m = x.mean(0)
        io[i, 0] = np.corrcoef(m[0], m[1])[0, 1]
        if len(x) < 2:
            continue
        for e in range(2):
            c = np.corrcoef(x[:, e])
            out[i, e, 0] = c[np.triu_indices(len(x), 1)].mean()
            noise = ((x[:, e] - m[e]) ** 2).mean() / len(x)
            out[i, e, 1] = np.log((m[e] ** 2).mean() / (noise + 1e-6))
    return np.concatenate([sym_pool(out), io], 1)


class Template:
    """Normal-template matching per eye (template = mean normal waveform of the training fold):
    correlation, least-squares gain, residual RMS, best cross-correlation lag."""

    def fit(self, mean, y):
        self.t = gaussian_filter1d(mean[y == 0], 2.0, axis=-1).mean((0, 1))
        self.t = self.t - self.t.mean()
        return self

    def transform(self, mean):
        x = gaussian_filter1d(mean, 2.0, axis=-1)
        x = x - x.mean(-1, keepdims=True)
        t = self.t
        corr = (x * t).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(t) + 1e-9)
        gain = (x * t).sum(-1) / (t ** 2).sum()
        resid = np.sqrt(((x - gain[..., None] * t) ** 2).mean(-1))
        lags = np.arange(-15, 16)
        cc = np.stack([(np.roll(x, l, -1) * t).sum(-1) for l in lags], -1)
        lag = lags[cc.argmax(-1)].astype(float)
        return sym_pool(np.stack([corr, gain, resid, lag], -1))


def fft_feats(mean):
    x = gaussian_filter1d(mean, 1.0, axis=-1)
    return sym_pool(np.log(np.abs(np.fft.rfft(x, axis=-1))[..., 1:25] + 1e-3))


def no_meta(f):
    return lambda m: f(m, pd.DataFrame({"age_years": 0, "sex_i": 0}, index=range(len(m))))[:, :-2]


def block(part, tr, te, ctx, seed):
    """Feature block for global record indices tr / te; anything fitted uses the training records only."""
    mean, reps, info, ytr = ctx["mean"], ctx["reps"], ctx["info"], ctx["ytr"]
    fixed = {"raw": no_meta(raw_features), "perg": no_meta(perg_features), "fft": fft_feats,
             "wb": lambda m: gaussian_filter1d(worse_better(m), 2.0, axis=-1)[:, :, ::3].reshape(len(m), -1),
             "dwt": lambda m: sym_pool(dwt_feats(m))}
    if part in fixed:
        return fixed[part](mean[tr]), fixed[part](mean[te])
    if part.startswith("fm_"):  # frozen foundation-model embedding per eye (label-free), PCA fit on train eyes
        E = np.load(OUT.parent / f"{part}.npy")
        pca = make_pipeline(StandardScaler(), PCA(32)).fit(E[tr].reshape(-1, E.shape[-1]))
        f = lambda idx: sym_pool(pca.transform(E[idx].reshape(-1, E.shape[-1])).reshape(len(idx), 2, -1))
        return f(tr), f(te)
    if part == "rel":
        return rel_feats([reps[i] for i in tr]), rel_feats([reps[i] for i in te])
    if part == "meta":
        v = info[["age_years", "sex_i"]].values.astype(float)
        return v[tr], v[te]
    if part == "tmpl":
        t = Template().fit(mean[tr], ytr)
        return t.transform(mean[tr]), t.transform(mean[te])
    if part == "fpca":
        t = FPCA().fit(mean[tr])
        return t.transform(mean[tr]), t.transform(mean[te])
    if part == "mrocket":
        t = MiniRocketEyes(seed).fit(mean[tr])
        return t.transform(mean[tr]), t.transform(mean[te])
    raise ValueError(part)


def features(feat, tr, te, ctx, seed):
    bs = [block(p, tr, te, ctx, seed) for p in feat.split("+")]
    Xtr, Xte = np.concatenate([b[0] for b in bs], 1), np.concatenate([b[1] for b in bs], 1)
    med = np.nanmedian(Xtr, 0)  # impute (e.g. single-repetition records) with training medians
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)


# ---------------------------------------------------------------- models: (feature set, estimator, grid)
def lr(C=1.0):
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


CS = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
MODELS = {
    # the PCA baseline exactly as in baselines.py (fixed hyperparameters)
    "pca_base": ("raw", lambda: make_pipeline(StandardScaler(), PCA(20), lr(0.3)), None),
    # PCA baseline with nested-tuned #components and C (a fair, strongest version of the baseline)
    "pca_tuned": ("raw", lambda: make_pipeline(StandardScaler(), PCA(), lr()),
                  {"pca__n_components": [3, 5, 10, 20, 40], "logisticregression__C": CS}),
    "pca_wb": ("wb", lambda: make_pipeline(StandardScaler(), PCA(), lr()),
               {"pca__n_components": [3, 5, 10, 20, 40], "logisticregression__C": CS}),
    "lr_perg": ("perg", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_perg_fpca": ("perg+fpca", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_all": ("perg+dwt+fpca", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "svm_perg_fpca": ("perg+fpca", lambda: make_pipeline(StandardScaler(), SVC(probability=True,
                                                                               class_weight="balanced")),
                      {"svc__C": [0.1, 0.3, 1, 3, 10], "svc__gamma": ["scale", 0.003, 0.01, 0.03]}),
    "svm_all": ("perg+dwt+fpca", lambda: make_pipeline(StandardScaler(), SVC(probability=True,
                                                                             class_weight="balanced")),
                {"svc__C": [0.1, 0.3, 1, 3, 10], "svc__gamma": ["scale", 0.001, 0.003, 0.01]}),
    "hgb_all": ("perg+dwt+fpca", lambda: HistGradientBoostingClassifier(class_weight="balanced", max_iter=200),
                {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "min_samples_leaf": [10, 20]}),
    "lr_mrocket": ("mrocket", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_rel": ("rel", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_tmpl": ("tmpl", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_new": ("perg+fpca+rel+tmpl", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS}),
    "lr_new_fft": ("perg+fpca+rel+tmpl+fft", lambda: make_pipeline(StandardScaler(), lr()),
                   {"logisticregression__C": CS}),
    "lr_new_meta": ("perg+fpca+rel+tmpl+meta", lambda: make_pipeline(StandardScaler(), lr()),
                    {"logisticregression__C": CS}),
    "svm_new": ("perg+fpca+rel+tmpl", lambda: make_pipeline(StandardScaler(), SVC(probability=True,
                                                                                  class_weight="balanced")),
                {"svc__C": [0.1, 0.3, 1, 3, 10], "svc__gamma": ["scale", 0.003, 0.01, 0.03]}),
    "hgb_new": ("perg+dwt+fpca+rel+tmpl", lambda: HistGradientBoostingClassifier(class_weight="balanced",
                                                                                max_iter=200),
                {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "min_samples_leaf": [10, 20]}),
}
for _fm in ["moment", "mantis", "chronos", "tirex", "tirexaug"]:
    MODELS[f"lr_fm_{_fm}"] = (f"fm_{_fm}", lambda: make_pipeline(StandardScaler(), lr()), {"logisticregression__C": CS})
    MODELS[f"lr_fm_{_fm}_perg"] = (f"fm_{_fm}+perg+fft", lambda: make_pipeline(StandardScaler(), lr()),
                                   {"logisticregression__C": CS})
    MODELS[f"svm_fm_{_fm}"] = (f"fm_{_fm}", lambda: make_pipeline(StandardScaler(), SVC(probability=True,
                                                                                        class_weight="balanced")),
                               {"svc__C": [0.1, 0.3, 1, 3, 10], "svc__gamma": ["scale", 0.001, 0.003, 0.01]})
# fixed, a-priori ensemble (probability average) of diverse models; no per-task selection
ENSEMBLES = {"ensemble": ["pca_tuned", "lr_perg_fpca", "svm_all", "lr_mrocket"],
             # pre-registered final model (fixed before seeing mrocket / @all / ensemble results)
             "ens_final": ["lr_perg_fpca", "lr_new_fft", "tcn_mix_ens5", "lr_mrocket"],
             # session 4b, fixed after wave 2 and before wave 3: the 4 CNN configs (resnet/inception x direct/pool)
             "dl_cnn_ens": ["dl_resnet_direct", "dl_inception_direct", "dl_resnet_pool", "dl_inception_pool"],
             # CNN ensemble + the two all-record classical models (post hoc choice -> mild selection bias)
             "dl_cnn_classic": ["dl_resnet_direct", "dl_inception_direct", "dl_resnet_pool", "dl_inception_pool",
                                "lr_new_fft@all", "pca_tuned@all"]}


# pre-registered (session 4b, before the overnight sweep): average EVERY finished run matching the pattern
GLOB_ENS = {"dl_big_ens": r"dl_(resnet|inception)_(direct|pool)(_va|_ch64|_e300|_drop3|_mix1|_ens20)?",
            "dl_big_cwt": r"dl_(resnet|inception|cwt)_(direct|pool)(_va|_ch64|_e300|_drop3|_mix1|_ens20)?",
            # pre-registered session 5 (before wave 4 ran): every wave-4 run (all-record training / diagnosis aux head)
            "dl_w4_ens": r"dl_(resnet|inception)_(all|allpool|pool_diag|allpool_diag|pool_diag_va|pool_e300_diag)"}
# pre-registered session 5: task-dependent ensembles (classical trio = best-known classical family per task type)
TASK_ENS = {"cls_ens": lambda t: ["lr_mrocket", "lr_perg_fpca", "lr_new"] if t.startswith("mac_")
            else ["pca_tuned@all", "lr_new_fft@all", "lr_perg@all"],
            "grand_ens": lambda t: ["dl_big_ens", "dl_w4_ens", "cls_ens"]}


def glob_members(task, name):
    import re
    return sorted(d.name for d in (OUT / task).iterdir() if re.fullmatch(GLOB_ENS[name], d.name)
                  and all((d / f"oof_seed{s}.npy").exists() for s in SEEDS))


def run(task, name):
    """name = <model> or <model>@all. '@all' = train on every record whose subject is not in the test fold
    (all disease groups, label = abnormal), evaluate on the task's records: borrows strength from other diseases."""
    D = load()
    info, mean, reps = D["info"], D["mean"], D["reps"]
    sel, y = screen_task(info, task)
    g = info.group.values
    base, pooled = name.split("@")[0], name.endswith("@all")
    out = OUT / task / name
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for seed in SEEDS:
        if base in ENSEMBLES or base in GLOB_ENS or base in TASK_ENS:
            members = ENSEMBLES[base] if base in ENSEMBLES else TASK_ENS[base](task) if base in TASK_ENS \
                else glob_members(task, base)
            p = np.mean([oof(task, m, seed) for m in members], 0)
        else:
            feat, est, grid = MODELS[base]
            p = np.zeros(len(y))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(sel, y, g[sel]):
                te_g = sel[te]
                if pooled:
                    tr_g = np.flatnonzero(~np.isin(g, g[te_g]))
                    ytr = info.y_bin.values[tr_g]
                else:
                    tr_g, ytr = sel[tr], y[tr]
                ctx = {"mean": mean, "reps": reps, "info": info, "ytr": ytr}
                Xtr, Xte = features(feat, tr_g, te_g, ctx, seed)
                m = est()
                if grid:
                    inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr, g[tr_g]))
                    m = GridSearchCV(m, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
                p[te] = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        np.save(out / f"oof_seed{seed}.npy", p)
        res.append(screen_metrics(y, p))
    df = pd.DataFrame(res)
    summary = {k: [float(df[k].mean()), float(df[k].std())] for k in df}
    json.dump({"task": task, "model": name, "members": ENSEMBLES.get(base) or
               (glob_members(task, base) if base in GLOB_ENS else TASK_ENS[base](task) if base in TASK_ENS else None), "n": len(y), "n_pos": int(y.sum()), "per_seed": res,
               "summary": summary}, open(out / "result.json", "w"), indent=1)
    print(task, name, {k: f"{a:.3f}±{b:.3f}" for k, (a, b) in summary.items()}, flush=True)


def oof(task, model, seed):
    """Positive-class OOF probability of a finished run (classical: (n,), TCN: (n, 2))."""
    p = np.load(OUT / task / model / f"oof_seed{seed}.npy")
    return p[:, 1] if p.ndim == 2 else p


def protocol(tasks=("norm_mac", "mac_opt")):
    """How much do evaluation shortcuts common in the literature inflate traditional-ML scores? Same features
    (smoothed downsampled waveform + PERG peaks, per sample) and default models under: record-level subject-grouped
    CV (ours), record-level random CV (ignores repeat visits), eye-level random CV (the two eyes of one person can
    fall in train and test), trace-level random CV (single repetitions of the same eye in train and test)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    D = load()
    info, mean, reps = D["info"], D["mean"], D["reps"]
    g = info.group.values
    mk = {"rf": lambda: RandomForestClassifier(500, class_weight="balanced", n_jobs=-1, random_state=0),
          "svm": lambda: make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced")),
          "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
          "lr": lambda: make_pipeline(StandardScaler(), lr(0.1))}

    def eye_feats(x):  # x: (n, T) single-eye traces
        s = gaussian_filter1d(x, 2.0, axis=-1)
        pf = no_meta(perg_features)(np.stack([x, x], 1))[:, :10]  # per-eye peak feats (min-block = the eye itself)
        return np.concatenate([s[:, ::3], pf], 1)

    rows = []
    for task in tasks:
        sel, y = screen_task(info, task)
        rec = {"X": np.concatenate([no_meta(raw_features)(mean[sel]), no_meta(perg_features)(mean[sel])], 1),
               "y": y, "g": g[sel]}
        e_idx = np.repeat(sel, 2)
        eye = {"X": eye_feats(mean[sel].reshape(-1, 255)), "y": np.repeat(y, 2), "g": g[e_idx]}
        tx, ty, tg = [], [], []
        for i, yi in zip(sel, y):
            for k in range(reps[i].shape[0]):
                for e in range(2):
                    tx.append(reps[i][k, e]), ty.append(yi), tg.append(g[i])
        trace = {"X": eye_feats(np.array(tx)), "y": np.array(ty), "g": np.array(tg)}
        for lvl, d in [("record", rec), ("eye", eye), ("trace", trace)]:
            for grouped in [True, False]:
                for name, est in mk.items():
                    res = []
                    for seed in SEEDS:
                        spl = (StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d["X"], d["y"], d["g"])
                               if grouped else StratifiedKFold(5, shuffle=True, random_state=seed).split(d["X"], d["y"]))
                        p = np.zeros(len(d["y"]))
                        for tr, te in spl:
                            p[te] = est().fit(d["X"][tr], d["y"][tr]).predict_proba(d["X"][te])[:, 1]
                        res.append(screen_metrics(d["y"], p))
                    r = pd.DataFrame(res).mean()
                    rows.append({"task": task, "unit": lvl, "split": "subject-grouped" if grouped else "random",
                                 "model": name, **{k: round(r[k], 3) for k in ["auroc", "acc", "bal_acc"]}})
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "protocol_check.csv", index=False)


def table():
    """All finished runs (classical + TCN), metrics recomputed from OOF so every model is scored identically."""
    info = load()["info"]
    rows = []
    for task in SCREEN_TASKS:
        _, y = screen_task(info, task)
        for d in sorted((OUT / task).glob("*/oof_seed2.npy")):
            m = d.parent.name
            r = pd.DataFrame([screen_metrics(y, oof(task, m, s)) for s in SEEDS])
            rows.append({"task": task, "model": m, **{k: f"{r[k].mean():.3f}±{r[k].std():.3f}" for k in r},
                         "_auc": r.auroc.mean()})
    df = pd.DataFrame(rows).sort_values(["task", "_auc"], ascending=[True, False]).drop(columns="_auc")
    df.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] in ("table", "protocol"):
        {"table": table, "protocol": protocol}[sys.argv[1]]()
        sys.exit()
    tasks = list(SCREEN_TASKS) if sys.argv[1] == "all" else sys.argv[1].split(",")
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else list(MODELS) + list(ENSEMBLES)
    for t in tasks:
        for m in models:
            run(t, m)
