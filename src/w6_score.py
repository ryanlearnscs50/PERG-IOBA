"""Session 9: score mac_opt candidates over however many CV repeats exist for ALL of them (seeds 0-9 target).
python w6_score.py [model ...]"""
import sys

import numpy as np
import pandas as pd

from data import load, screen_task
from exp_screen import OUT, oof, screen_metrics

TASK = "mac_opt"
models = sys.argv[1:] or ["pca_base", "cls_ens", "simple_resnet_direct", "w6_resnet_e300_nomix",
                          "dl_resnet_direct_e300", "w6_resnet_e500", "dl_resnet_pool_e300"]
info = load()["info"]
_, y = screen_task(info, TASK)
seeds = [s for s in range(10) if all((OUT / TASK / m / f"oof_seed{s}.npy").exists() for m in models)]
print("seeds scored:", seeds, flush=True)
rows = []
for m in models:
    r = pd.DataFrame([screen_metrics(y, oof(TASK, m, s)) for s in seeds])
    rows.append({"model": m, **{k: f"{r[k].mean():.3f}±{r[k].std():.3f}" for k in r}})
print(pd.DataFrame(rows).to_string(index=False))
