"""Classical baselines: clinical PERG features (N35/P50/N95) and raw-waveform models, same CV as the TCN."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import cv_splits, load
from metrics import evaluate

FS = 255 / 150.0  # samples per ms


def win(a, b):
    return slice(int(a * FS), int(b * FS))


def perg_features(mean, info):
    """Per eye: N35/P50/N95 amplitudes + latencies, P50 & N95 peak-to-peak, N95/P50 ratio; then
    eye-symmetric aggregation (min/max/mean over eyes) + age + sex."""
    x = gaussian_filter1d(mean, 2.0, axis=-1)
    feats = []
    for eye in range(2):
        s = x[:, eye]
        n35_i = s[:, win(20, 45)].argmin(1) + win(20, 45).start
        p50_i = s[:, win(40, 75)].argmax(1) + win(40, 75).start
        n95_i = s[:, win(75, 140)].argmin(1) + win(75, 140).start
        r = np.arange(len(s))
        n35, p50, n95 = s[r, n35_i], s[r, p50_i], s[r, n95_i]
        p50_amp, n95_amp = p50 - n35, p50 - n95
        feats.append(np.stack([n35, p50, n95, n35_i / FS, p50_i / FS, n95_i / FS, p50_amp, n95_amp,
                               n95_amp / (np.abs(p50_amp) + 0.1), s.std(1)], 1))
    F = np.stack(feats, 0)  # (2, N, f)
    agg = np.concatenate([F.min(0), F.max(0), F.mean(0), np.abs(F[0] - F[1])], 1)
    return np.concatenate([agg, info[["age_years", "sex_i"]].values], 1)


def raw_features(mean, info):
    x = gaussian_filter1d(mean, 2.0, axis=-1)[:, :, ::3]  # 85 pts/eye
    # eye-symmetric: sort-free -> mean and |diff| of the two eyes
    return np.concatenate([x.mean(1), np.abs(x[:, 0] - x[:, 1]), info[["age_years", "sex_i"]].values], 1)


MODELS = {
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced")),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                  class_weight="balanced"),
    "pca_lr": lambda: make_pipeline(StandardScaler(), PCA(20), LogisticRegression(C=0.3, max_iter=5000,
                                                                                  class_weight="balanced")),
}


def run(task, feat_name, model_name, seeds=(0, 1, 2)):
    D = load()
    info = D["info"]
    X = {"perg": perg_features, "raw": raw_features}[feat_name](D["mean"], info)
    y = info["y_bin" if task == "bin" else "y_grp"].values
    res = []
    for seed in seeds:
        oof = np.zeros(len(y)) if task == "bin" else np.zeros((len(y), 4))
        for tr, te in cv_splits(info, seed=seed):
            m = MODELS[model_name]().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            oof[te] = p[:, 1] if task == "bin" else p
        res.append(evaluate(task, y, oof))
    return pd.DataFrame(res).agg(["mean", "std"])


if __name__ == "__main__":
    rows = {}
    for task in ["bin", "grp"]:
        for feat, model in [("perg", "lr"), ("perg", "hgb"), ("raw", "pca_lr"), ("raw", "hgb")]:
            r = run(task, feat, model)
            rows[f"{task}/{feat}+{model}"] = {k: f"{r.loc['mean', k]:.3f}±{r.loc['std', k]:.3f}" for k in r}
            print(task, feat, model, rows[f"{task}/{feat}+{model}"], flush=True)
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "runs/baselines.json", "w"), indent=1)
