"""Session 9 figures for mac_opt: ROC curves (baseline / classical / ResNet) and the ResNet confusion matrix,
from the 10 CV repeats. python w6_figs.py <resnet_model>  -> docs/fig_roc.png, docs/fig_confusion.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from data import load, screen_task
from exp_screen import oof

TASK, SEEDS = "mac_opt", range(10)
RESNET = sys.argv[1]
FIG = Path(__file__).resolve().parents[1] / "docs"
FIG.mkdir(exist_ok=True)
info = load()["info"]
_, y = screen_task(info, TASK)

models = [("PCA + logistic regression (baseline)", "pca_base", "#9aa0a6"),
          ("CNN (ResNet)", RESNET, "#2f6fdb")]
fig, ax = plt.subplots(figsize=(5, 5))
grid = np.linspace(0, 1, 201)
for label, m, c in models:
    tprs, aucs = [], []
    for s in SEEDS:
        p = oof(TASK, m, s)
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(y, p))
    ax.plot(grid, np.mean(tprs, 0), color=c, lw=2, label=f"{label}  AUROC {np.mean(aucs):.2f}")
ax.plot([0, 1], [0, 1], ls="--", color="#cccccc", lw=1)
ax.set_xlabel("False-alarm rate (macular called optic)")
ax.set_ylabel("Optic cases caught")
ax.set_title("Macular vs optic-nerve disease (10 repeats of 5-fold CV)", fontsize=10)
ax.legend(loc="lower right", fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_roc.png", dpi=160)

# confusion matrix at threshold 0.5, averaged over the 10 repeats
cm = np.zeros((2, 2))
for s in SEEDS:
    yh = (oof(TASK, RESNET, s) > 0.5).astype(int)
    for a in (0, 1):
        for b in (0, 1):
            cm[a, b] += ((y == a) & (yh == b)).sum() / len(SEEDS)
fig, ax = plt.subplots(figsize=(4, 3.4))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
names = ["Macular", "Optic"]
for a in (0, 1):
    for b in (0, 1):
        frac = cm[a, b] / cm[a].sum()
        ax.text(b, a, f"{cm[a, b]:.0f}\n({frac:.0%})", ha="center", va="center",
                color="white" if frac > 0.5 else "black", fontsize=11)
ax.set_xticks([0, 1], names)
ax.set_yticks([0, 1], names)
ax.set_xlabel("CNN says")
ax.set_ylabel("True diagnosis")
ax.set_title("CNN (ResNet), average of 10 repeats", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig_confusion.png", dpi=160)
print(cm)
