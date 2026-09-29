"""PERG-IOBA loading, label definitions and subject-grouped CV splits."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1] / "data" / "csv"
SCALE = 5.0  # µV; global amplitude scale (amplitude itself is diagnostic, so no per-trace z-scoring)

# 4-group taxonomy based on the PERG generator: P50 ~ macular cones/bipolars, N95 ~ ganglion cells.
MACULAR = {
    "Macular dystrophy", "Stargardt disease", "Pattern macular dystrophy", "X linked retinoschisis",
    "Central serous chorioretinopathy (CSCR)", "Torpedo maculopathy", "Pseudovitelliform macular dystrophy",
    "Macular telangiectasias", "Macula toxicity", "Acute macular neuroretinopathy", "Retinal toxicity",
    "Paracentral acute middle maculopathy type 2", "Foveal hypoplasia", "Pathologic myopia",
}
OPTIC_NEURO = {
    "Inherited optic atrophy", "Optic neuropathy", "Nonarteritic anterior ischemic optic neuropathy",
    "Orbital ischemia", "Bilateral optic nerve atrophy", "Bilateral Optic Atrophy", "Traumatic optic neuropathy",
    "Sarcoidosis neuropathy", "Arterio-venous malformation in right thalamus", "Anisometropic amblyopia",
    "External ophthalmoplegia", "Central nervous system disorder", "Infectious neuritis",
    "Central nervous system infection", "Pseudopapiledema", "Ischemic optic neuropathy", "Toxic neuropathy",
    "Retrobulbar neuritis", "Neurological disorder right side", "Affectation of the chiasm",
    "Systemic disorder with ocular manifestations",
}
# everything else non-Normal -> generalized / photoreceptor / chorioretinal disease
GROUP_NAMES = ["Normal", "Macular", "Generalized retinal", "Optic nerve / neuro"]


def diag_group(d: str) -> int:
    if d == "Normal":
        return 0
    if d in MACULAR:
        return 1
    if d in OPTIC_NEURO:
        return 3
    return 2


def _subject_groups(info: pd.DataFrame) -> np.ndarray:
    """Union-find over rep_record links so repeat visits of one subject share a CV group."""
    ids = list(info.id_record)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid, rep in zip(info.id_record, info.rep_record):
        if isinstance(rep, str):
            for other in re.findall(r"\d{4}", rep):
                if other in parent:
                    parent[find(other)] = find(rid)
    roots = [find(i) for i in ids]
    return pd.factorize(pd.Series(roots))[0]


def load():
    """Returns dict with
    reps: list of float32 arrays (k, 2, 255)  [k repetitions, eyes RE/LE, time], scaled
    mean: float32 (N, 2, 255) repetition-averaged
    info: DataFrame with y_bin, y_grp, group, age, sex
    """
    info = pd.read_csv(ROOT / "participants_info.csv", dtype={"id_record": str})
    reps = []
    for rid in info.id_record:
        df = pd.read_csv(ROOT / f"{rid}.csv")
        k = sum(c.startswith("RE_") for c in df.columns)
        arr = np.stack([np.stack([df[f"RE_{i}"].values, df[f"LE_{i}"].values]) for i in range(1, k + 1)])
        reps.append((arr / SCALE).astype(np.float32))
    info["y_bin"] = (info.diagnosis1 != "Normal").astype(int)
    info["y_grp"] = info.diagnosis1.map(diag_group)
    info["group"] = _subject_groups(info)
    info["sex_i"] = (info.sex == "Male").astype(int)
    mean = np.stack([r.mean(0) for r in reps])
    return {"reps": reps, "mean": mean, "info": info}


# Screening tasks: name -> (disease groups kept, positive groups). Records of other groups are dropped.
SCREEN_TASKS = {
    "norm_abn": ((0, 1, 2, 3), (1, 2, 3)),     # Normal vs any disease
    "norm_mac": ((0, 1), (1,)),                # Normal vs macular disease
    "norm_opt": ((0, 3), (3,)),                # Normal vs optic nerve / neuro
    "norm_macopt": ((0, 1, 3), (1, 3)),        # Normal vs (macular or optic nerve)
    "mac_opt": ((1, 3), (3,)),                 # differential: macular (0) vs optic nerve (1)
    "norm_imd": ((0, 1), (1,)),                # Normal vs inherited macular dystrophies only (see DIAG_FILTER)
    "mac_on": ((1, 3), (3,)),                  # macular vs optic neuropathies proper (no CNS/systemic/amblyopia)
}
# optional restriction of the positive group to specific diagnoses
IMD = {"Macular dystrophy", "Stargardt disease", "X linked retinoschisis", "Pattern macular dystrophy",
       "Pseudovitelliform macular dystrophy"}
# optic-nerve group minus diagnoses without an optic-nerve lesion (retro-chiasmal/CNS, systemic, amblyopia,
# ophthalmoplegia), chosen on clinical grounds before any mac_on result was computed
OPTIC_NEUROPATHY = {"Inherited optic atrophy", "Orbital ischemia", "Optic neuropathy",
                    "Nonarteritic anterior ischemic optic neuropathy", "Bilateral optic nerve atrophy",
                    "Traumatic optic neuropathy", "Retrobulbar neuritis", "Toxic neuropathy", "Ischemic optic neuropathy",
                    "Infectious neuritis", "Sarcoidosis neuropathy", "Bilateral Optic Atrophy"}
DIAG_FILTER = {"norm_imd": IMD, "mac_on": OPTIC_NEUROPATHY}


def screen_task(info: pd.DataFrame, task: str):
    """Returns (record indices kept, binary labels) for a screening task."""
    keep, pos = SCREEN_TASKS[task]
    m = info.y_grp.isin(keep).values.copy()
    if task in DIAG_FILTER:
        m &= (~info.y_grp.isin(pos) | info.diagnosis1.isin(DIAG_FILTER[task])).values
    sel = np.flatnonzero(m)
    return sel, np.isin(info.y_grp.values[sel], pos).astype(int)


def cv_splits(info: pd.DataFrame, n_splits=5, seed=0, strat_col="y_grp"):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(info)), info[strat_col], info["group"]))


if __name__ == "__main__":
    D = load()
    info = D["info"]
    print(info.y_grp.map(dict(enumerate(GROUP_NAMES))).value_counts())
    print("subjects:", info.group.nunique(), "records:", len(info))
    print(pd.Series([r.shape[0] for r in D["reps"]]).value_counts())
