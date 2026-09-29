"""Subject-grouped CV training of PERGNet (TCN) with optional metadata embedding and contrastive SSL pretraining.

python train.py --task bin --meta none --name tcn_bin
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import SCREEN_TASKS, cv_splits, load, screen_task
from metrics import binary_metrics, evaluate
from models import PERGNet, TCNEncoder

p = argparse.ArgumentParser()
p.add_argument("--task", default="bin", choices=["bin", "grp"] + list(SCREEN_TASKS))
p.add_argument("--meta", default="none", choices=["none", "concat", "film", "both"])
p.add_argument("--ssl", type=int, default=0, help="contrastive pretraining epochs on train-fold traces (0=off)")
p.add_argument("--ssl_temp", type=float, default=0.1)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--ch", type=int, default=64)
p.add_argument("--levels", type=int, default=6)
p.add_argument("--k", type=int, default=5)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--causal", type=int, default=0)
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--n_ens", type=int, default=1, help="independently initialised models per fold, probs averaged")
p.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a,a) alpha on the training batches (0=off)")
p.add_argument("--pool_all", type=int, default=0,
               help="screen tasks: train on ALL records outside the test subjects (label = abnormal), test on the task")
p.add_argument("--name", required=True)
p.add_argument("--out", default="../runs")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = load()
info = D["info"]
STRAT, SEL = "y_grp", np.arange(len(info))
if args.task in SCREEN_TASKS:  # binary screening on a subset of records; stratify on the task label
    sel, y_s = screen_task(info, args.task)
    if args.pool_all:  # keep every record for training; CV folds / evaluation on the task subset only
        info = info.assign(y_s=info.y_bin.values)
        SUB = info.iloc[sel].reset_index(drop=True).assign(y_s=y_s)
        SEL = sel
    else:
        info = info.iloc[sel].reset_index(drop=True).assign(y_s=y_s)
        D["reps"], D["mean"] = [D["reps"][i] for i in sel], D["mean"][sel]
    STRAT = "y_s"
if not args.pool_all:
    SUB = info
N, T = len(info), D["mean"].shape[-1]
KMAX = max(r.shape[0] for r in D["reps"])
R = np.zeros((N, KMAX, 2, T), np.float32)
M = np.zeros((N, KMAX), bool)
for i, r in enumerate(D["reps"]):
    R[i, : len(r)], M[i, : len(r)] = r, True
R, M = torch.tensor(R, device=dev), torch.tensor(M, device=dev)
XMEAN = torch.tensor(D["mean"], device=dev)
AGE = torch.tensor(info.age_years.values, dtype=torch.float32, device=dev)
SEX = torch.tensor(info.sex_i.values, device=dev)
y_np = info[{"bin": "y_bin", "grp": "y_grp"}.get(args.task, "y_s")].values
Y = torch.tensor(y_np, device=dev)
n_cls = 4 if args.task == "grp" else 2
EVAL = "grp" if args.task == "grp" else "bin"


def rep_average(idx, p_keep=0.6):
    """Random-subset average of repetitions: a physiologically realistic 'new recording' of the same eye."""
    w = (torch.rand(len(idx), KMAX, device=dev) < p_keep) & M[idx]
    empty = ~w.any(1)
    if empty.any():
        k = (torch.rand(int(empty.sum()), device=dev) * M[idx][empty].sum(1)).long()
        w[empty.nonzero()[:, 0], k] = True
    w = w.float()[:, :, None, None]
    return (R[idx] * w).sum(1) / w.sum(1)


def augment(x):
    B = x.shape[0]
    shift = torch.randint(-4, 5, (B,), device=dev)
    ar = (torch.arange(T, device=dev)[None] - shift[:, None]).clamp(0, T - 1)
    x = torch.gather(x, 2, ar[:, None].expand(-1, x.shape[1], -1))
    x = x * torch.exp(0.1 * torch.randn(B, x.shape[1], 1, device=dev))
    return x + 0.05 * torch.randn_like(x) + 0.05 * torch.randn(B, x.shape[1], 1, device=dev)


def ssl_pretrain(tr, epochs):
    """SimCLR/NT-Xent on single-eye traces; positives = two independent repetition-subset averages of the same
    eye (+ augmentation), so the embedding learns what is stable across trials and discards trial noise."""
    enc = TCNEncoder(args.ch, args.levels, args.k, args.dropout, bool(args.causal)).to(dev)
    proj = torch.nn.Sequential(torch.nn.Linear(enc.out_dim, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-2)
    tr = torch.tensor(tr, device=dev)
    for ep in range(epochs):
        for b in tr[torch.randperm(len(tr), device=dev)].split(128):
            if len(b) < 8:
                continue
            v1, v2 = augment(rep_average(b, 0.5)), augment(rep_average(b, 0.5))
            z = torch.cat([v1.reshape(-1, 1, T), v2.reshape(-1, 1, T)])  # (4B,1,T): eyes as separate samples
            z = F.normalize(proj(enc(z)), dim=1)
            n = z.shape[0] // 2
            sim = z @ z.T / args.ssl_temp
            sim.fill_diagonal_(-1e9)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
            loss = F.cross_entropy(sim, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return enc.state_dict()


def train_fold(tr, te, seed):
    return np.mean([train_one(tr, te, seed * 1000 + j) for j in range(args.n_ens)], 0)


def train_one(tr, te, seed):
    torch.manual_seed(seed)
    model = PERGNet(n_cls, args.meta, ch=args.ch, levels=args.levels, k=args.k, dropout=args.dropout,
                    causal=bool(args.causal)).to(dev)
    enc_lr = args.lr
    if args.ssl:
        sd = ssl_pretrain(tr, args.ssl)
        missing, _ = model.enc.load_state_dict(sd, strict=False)
        assert all("film" in m for m in missing), missing
        enc_lr = args.lr * 0.3
    enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.") and "film" not in n]
    other = [p for n, p in model.named_parameters() if not (n.startswith("enc.") and "film" not in n)]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": enc_lr}, {"params": other, "lr": args.lr}],
                            weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[enc_lr, args.lr], total_steps=steps, pct_start=0.1)
    cw = torch.tensor(len(tr) / (n_cls * np.bincount(y_np[tr], minlength=n_cls)), dtype=torch.float32, device=dev)
    tr_t = torch.tensor(tr, device=dev)
    for ep in range(args.epochs):
        model.train()
        for b in tr_t[torch.randperm(len(tr), device=dev)].split(args.bs):
            x = augment(rep_average(b))
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(len(b), device=dev)
                out = model(lam * x + (1 - lam) * x[perm], AGE[b], SEX[b])
                loss = lam * F.cross_entropy(out, Y[b], weight=cw, label_smoothing=0.05) + \
                    (1 - lam) * F.cross_entropy(out, Y[b][perm], weight=cw, label_smoothing=0.05)
            else:
                loss = F.cross_entropy(model(x, AGE[b], SEX[b]), Y[b], weight=cw, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        te_t = torch.tensor(te, device=dev)
        return torch.softmax(model(XMEAN[te_t], AGE[te_t], SEX[te_t]), 1).cpu().numpy()


out = Path(args.out) / args.name
out.mkdir(parents=True, exist_ok=True)
res, t0 = [], time.time()
for seed in [int(s) for s in args.seeds.split(",")]:
    P = np.zeros((len(SUB), n_cls))
    for f, (tr, te) in enumerate(cv_splits(SUB, seed=seed, strat_col=STRAT)):
        te_g = SEL[te]
        tr_g = np.flatnonzero(~np.isin(info.group.values, info.group.values[te_g])) if args.pool_all else tr
        P[te] = train_fold(tr_g, te_g, seed * 100 + f)
    np.save(out / f"oof_seed{seed}.npy", P)
    y_eval = SUB[{"bin": "y_bin", "grp": "y_grp"}.get(args.task, "y_s")].values
    m = evaluate(EVAL, y_eval, P[:, 1] if EVAL == "bin" else P)
    if args.task == "grp":  # screening score from the 4-class model: 1 - P(Normal)
        m.update({f"screen_{k}": v for k, v in binary_metrics(info.y_bin.values, 1 - P[:, 0]).items()})
    res.append(m)
    print(args.name, f"seed{seed}", {k: round(v, 3) for k, v in m.items()}, f"{time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(res)
summary = {k: [float(df[k].mean()), float(df[k].std())] for k in df}
json.dump({"args": vars(args), "per_seed": res, "summary": summary}, open(out / "result.json", "w"), indent=1)
print("SUMMARY", args.name, {k: f"{m:.3f}±{s:.3f}" for k, (m, s) in summary.items()}, flush=True)
