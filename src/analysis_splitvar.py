"""How far can a single (lucky) 80/20 split or a narrower disease subset move accuracy? 100 random splits per setting."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from data import load, screen_task
from exp_screen import no_meta, lr, make_pipeline, StandardScaler, SVC
from baselines import raw_features, perg_features
D=load(); info, mean = D["info"], D["mean"]; g=info.group.values
IMD={"Macular dystrophy","Stargardt disease","X linked retinoschisis","Pattern macular dystrophy","Pseudovitelliform macular dystrophy"}
for task in ["norm_mac","mac_opt","norm_imd"]:
    if task=="norm_imd":
        sel=np.flatnonzero((info.diagnosis1=="Normal")|info.diagnosis1.isin(IMD)); y=(info.diagnosis1.values[sel]!="Normal").astype(int)
    else: sel,y=screen_task(info,task)
    X=np.concatenate([no_meta(raw_features)(mean[sel]),no_meta(perg_features)(mean[sel])],1)
    for name,mk in [("lr",lambda: make_pipeline(StandardScaler(),lr(0.1))),("svm",lambda: make_pipeline(StandardScaler(),SVC())),("rf",lambda: RandomForestClassifier(300,n_jobs=-1,random_state=0))]:
        for split,sp in [("grouped",GroupShuffleSplit(100,test_size=0.2,random_state=0).split(X,y,g[sel])),("random",ShuffleSplit(100,test_size=0.2,random_state=0).split(X,y))]:
            a=[accuracy_score(y[te],mk().fit(X[tr],y[tr]).predict(X[te])) for tr,te in sp]
            print(f"{task:9s} n={len(y)} pos={y.sum()} maj={max(y.mean(),1-y.mean()):.2f} {name:4s} {split:8s} single-80/20 acc: mean {np.mean(a):.3f} p90 {np.percentile(a,90):.3f} max {np.max(a):.3f}",flush=True)
