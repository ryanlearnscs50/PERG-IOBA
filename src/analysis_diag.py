"""Per-diagnosis breakdown of a screening run's OOF predictions (mean over seeds):
for each diagnosis1 in the positive/negative class, n and mean predicted P(positive) and AUROC vs the other class.
python analysis_diag.py <task> <model> [<model> ...]"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import load, screen_task
from exp_screen import SEEDS, oof

task, models = sys.argv[1], sys.argv[2:]
info = load()["info"]
sel, y = screen_task(info, task)
dg = info.diagnosis1.values[sel]
for m in models:
    p = np.mean([oof(task, m, s) for s in SEEDS], 0)
    rows = []
    for d in pd.unique(dg):
        k = dg == d
        cls = int(y[k][0])
        other = y != cls
        sc = np.r_[p[k], p[other]]
        lab = np.r_[np.ones(k.sum()), np.zeros(other.sum())] if cls else np.r_[np.zeros(k.sum()), np.ones(other.sum())]
        rows.append((cls, d, int(k.sum()), p[k].mean(), roc_auc_score(lab, sc)))
    df = pd.DataFrame(rows, columns=["cls", "diagnosis", "n", "mean_p", "auroc_vs_other"]).sort_values(["cls", "n"],
                                                                                                    ascending=[True, False])
    print(f"== {task} {m}  (overall AUROC {roc_auc_score(y, p):.3f})")
    print(df.to_string(index=False, float_format="%.2f"))
