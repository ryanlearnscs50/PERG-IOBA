"""Deep models for binary screening tasks (data.SCREEN_TASKS), one seed per call so seeds can run on separate GPUs.

Encoders (shared across eyes, eye-symmetric pooling): tcn | inception | resnet.
--mode direct : train on the task's records only.
--mode pool   : multi-task on ALL records outside the test subjects: 4-group head on every record + task head on
                the task's records; prediction = mean of task head and the 4-group head's conditional probability.
--mode all    : like classical '@all': task head = Normal vs ANY disease on all records outside the test subjects
                (only for Normal-vs-X tasks); evaluated on the task's records.
--mode allpool: 'all' + the 4-group head (pred = mean of both, as in pool).
--hybrid 1    : hand-crafted PERG / spectral / reproducibility features (fold-standardised) enter the head too.
Folds are identical to exp_screen.py (StratifiedGroupKFold on the task subset), so OOFs can be ensembled.
Writes runs/screen/<task>/<name>/oof_seed<s>.npy (+ oofhead_/oofgrp_ variants in pool mode).

python train_dl.py --task norm_mac --arch inception --mode pool --hybrid 1 --seed 0 --name inc_pool_hyb
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from baselines import perg_features
from data import SCREEN_TASKS, load, screen_task
from exp_screen import fft_feats, no_meta, rel_feats, screen_metrics
from models import TCNEncoder

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=list(SCREEN_TASKS))
p.add_argument("--arch", default="tcn", choices=["tcn", "inception", "resnet", "cwt", "patchtst"])
p.add_argument("--mode", default="direct", choices=["direct", "pool", "all", "allpool"])
p.add_argument("--hybrid", type=int, default=0)
p.add_argument("--grp_w", type=float, default=0.5, help="weight of the 4-group loss in pool mode")
p.add_argument("--va_w", type=float, default=0.0,
               help="weight of auxiliary per-eye visual-acuity (logMAR) regression; training target only, never an input")
p.add_argument("--diag_w", type=float, default=0.0,
               help="aux head over diagnosis1 classes with >=8 records (+ 'other'), trained on every training record")
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--ch", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--mixup", type=float, default=0.4)
p.add_argument("--n_ens", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--name", required=True)
args = p.parse_args()
OUT = Path(__file__).resolve().parents[1] / "runs" / "screen" / args.task / args.name
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

D = load()
info = D["info"]
N, T = len(info), 255
sel, y_sel = screen_task(info, args.task)
keep, pos = SCREEN_TASKS[args.task]
neg_grp = [k for k in keep if k not in pos][0]
TASK_Y = np.full(N, -1)
TASK_Y[sel] = y_sel  # -1 = record not part of the task
if args.mode in ("all", "allpool"):
    assert neg_grp == 0, "all/allpool modes need a Normal-vs-X task"
    TASK_Y = (info.y_grp.values != 0).astype(int)
vc = info.diagnosis1.value_counts()
DIAG_CLS = sorted(vc[vc >= 8].index)
DIAG_Y = np.array([DIAG_CLS.index(d) if d in DIAG_CLS else len(DIAG_CLS) for d in info.diagnosis1])
N_DIAG = len(DIAG_CLS) + 1
DIAG = torch.tensor(DIAG_Y, device=dev)
POOLED = args.mode != "direct"
GRP_HEAD = args.mode in ("pool", "allpool")
KMAX = max(r.shape[0] for r in D["reps"])
R = np.zeros((N, KMAX, 2, T), np.float32)
M = np.zeros((N, KMAX), bool)
for i, r in enumerate(D["reps"]):
    R[i, : len(r)], M[i, : len(r)] = r, True
R, M = torch.tensor(R, device=dev), torch.tensor(M, device=dev)
XMEAN = torch.tensor(D["mean"], device=dev)
GRP = torch.tensor(info.y_grp.values, device=dev)
TY = torch.tensor(TASK_Y, device=dev)
VA_np = info[["va_re_logMar", "va_le_logMar"]].values.astype(np.float32)  # eye order matches R/XMEAN (RE, LE)
VA = torch.tensor(np.nan_to_num(np.clip(VA_np, -0.3, 2.0)), device=dev)
VA_M = torch.tensor(~np.isnan(VA_np), device=dev).float()
FEAT = np.concatenate([no_meta(perg_features)(D["mean"]), fft_feats(D["mean"]), rel_feats(D["reps"])], 1) \
    if args.hybrid else np.zeros((N, 0))


# ---------------------------------------------------------------- encoders (single eye (B,1,T) -> (B, out_dim))
class Ramp(nn.Module):
    """Adds a time-ramp channel so absolute latency is visible to translation-equivariant convolutions."""

    def __init__(self):
        super().__init__()
        self.register_buffer("r", torch.linspace(-1, 1, T)[None, None])

    def forward(self, x):
        return torch.cat([x, self.r.expand(x.shape[0], -1, -1)], 1)


class InceptionModule(nn.Module):
    def __init__(self, cin, nf, ks=(9, 19, 39)):
        super().__init__()
        self.bott = nn.Conv1d(cin, nf, 1, bias=False)
        self.convs = nn.ModuleList(nn.Conv1d(nf, nf, k, padding=k // 2, bias=False) for k in ks)
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))

    def forward(self, x):
        b = self.bott(x)
        return F.gelu(self.bn(torch.cat([c(b) for c in self.convs] + [self.pool(x)], 1)))


class InceptionEnc(nn.Module):
    """InceptionTime (Fawaz et al. 2020), depth 6 with residual every 3 modules, GAP + attention pooling."""

    def __init__(self, nf=32, dropout=0.1):
        super().__init__()
        self.ramp = Ramp()
        c = nf * 4
        self.mods = nn.ModuleList([InceptionModule(2 if i == 0 else c, nf) for i in range(6)])
        self.res = nn.ModuleList([nn.Sequential(nn.Conv1d(2, c, 1, bias=False), nn.BatchNorm1d(c)),
                                  nn.Sequential(nn.Conv1d(c, c, 1, bias=False), nn.BatchNorm1d(c))])
        self.drop = nn.Dropout(dropout)
        self.attn = nn.Conv1d(c, 1, 1)
        self.out_dim = 2 * c

    def forward(self, x):
        x = self.ramp(x)
        h, r = x, x
        for i, m in enumerate(self.mods):
            h = self.drop(m(h))
            if i % 3 == 2:
                h = F.gelu(h + self.res[i // 3](r))
                r = h
        w = torch.softmax(self.attn(h), -1)
        return torch.cat([(h * w).sum(-1), h.mean(-1)], 1)


class ResNetEnc(nn.Module):
    """ResNet1D (Wang et al. 2017 TSC baseline): 3 residual blocks of conv 7-5-3."""

    def __init__(self, nf=32, dropout=0.1):
        super().__init__()
        self.ramp = Ramp()
        chans = [2, nf, 2 * nf, 2 * nf]
        self.blocks = nn.ModuleList()
        for a, b in zip(chans[:-1], chans[1:]):
            self.blocks.append(nn.ModuleDict({
                "body": nn.Sequential(*[m for k in (7, 5, 3) for m in (
                    nn.Conv1d(a if k == 7 else b, b, k, padding=k // 2, bias=False), nn.BatchNorm1d(b), nn.GELU())]),
                "skip": nn.Sequential(nn.Conv1d(a, b, 1, bias=False), nn.BatchNorm1d(b))}))
        self.drop = nn.Dropout(dropout)
        self.attn = nn.Conv1d(chans[-1], 1, 1)
        self.out_dim = 2 * chans[-1]

    def forward(self, x):
        h = self.ramp(x)
        for blk in self.blocks:
            h = self.drop(F.gelu(blk["body"](h) + blk["skip"](h)))
        w = torch.softmax(self.attn(h), -1)
        return torch.cat([(h * w).sum(-1), h.mean(-1)], 1)


class PatchTSTEnc(nn.Module):
    """Transformer encoder on overlapping patches (PatchTST-style, Nie et al. 2023): patch 16 / stride 8 -> 31 tokens,
    linear patch embedding + learned positions (absolute latency matters), pre-norm layers, attention + mean pooling."""

    def __init__(self, d=64, dropout=0.1, patch=16, stride=8, layers=3, heads=4):
        super().__init__()
        self.patch, self.stride = patch, stride
        n_tok = (T - patch) // stride + 1
        self.emb = nn.Linear(patch, d)
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tok, d))
        self.inorm = nn.BatchNorm1d(1)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 1)
        self.out_dim = 2 * d

    def forward(self, x):
        tok = self.inorm(x).unfold(-1, self.patch, self.stride)[:, 0]  # (B, n_tok, patch)
        h = self.norm(self.tf(self.emb(tok) + self.pos))
        w = torch.softmax(self.attn(h), 1)
        return torch.cat([(h * w).sum(1), h.mean(1)], 1)


class CWTEnc(nn.Module):
    """Complex Morlet wavelet scalogram (fixed conv bank, 32 log-spaced freqs 8-250 Hz, 3 cycles) of one eye ->
    ImageNet-pretrained ResNet18 (transfer learning on time-frequency images, common for small biosignal sets)."""

    def __init__(self, dropout=0.1, fs=1700.0):
        super().__init__()
        import torchvision
        ks = []
        freqs = np.geomspace(8, 250, 32)
        L = int(3 * 3 / (2 * np.pi * freqs[0]) * fs) * 2 + 1
        t = (np.arange(L) - L // 2) / fs
        for f in freqs:
            sd = 3 / (2 * np.pi * f)
            w = np.exp(-t ** 2 / (2 * sd ** 2)) * np.exp(2j * np.pi * f * t)
            w /= np.abs(w).sum()
            ks += [w.real, w.imag]
        self.register_buffer("bank", torch.tensor(np.array(ks), dtype=torch.float32)[:, None])
        self.norm = nn.BatchNorm2d(1)
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        net.fc = nn.Identity()
        self.net = net
        self.drop = nn.Dropout(dropout)
        self.out_dim = 512

    def forward(self, x):
        c = F.conv1d(x, self.bank, padding=self.bank.shape[-1] // 2)  # (B, 64, T)
        mag = torch.sqrt(c[:, 0::2] ** 2 + c[:, 1::2] ** 2 + 1e-8)  # (B, 32, T)
        img = torch.log(mag + 1e-3)[:, None]
        img = F.interpolate(img, size=(64, 256), mode="bilinear", align_corners=False)
        img = self.norm(img).expand(-1, 3, -1, -1)
        return self.drop(self.net(img))


class Net(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.enc = {"tcn": lambda: TCNEncoder(args.ch, 6, 5, args.dropout),
                    "inception": lambda: InceptionEnc(args.ch, args.dropout),
                    "resnet": lambda: ResNetEnc(args.ch, args.dropout),
                    "cwt": lambda: CWTEnc(args.dropout),
                    "patchtst": lambda: PatchTSTEnc(2 * args.ch, args.dropout)}[args.arch]()
        d = 3 * self.enc.out_dim
        self.fproj = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_feat, 64), nn.GELU()) if n_feat else None
        d += 64 if n_feat else 0
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.3))
        self.task = nn.Linear(64, 2)
        self.grp = nn.Linear(64, 4)
        self.diag = nn.Linear(64, N_DIAG)
        self.va = nn.Sequential(nn.Linear(self.enc.out_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x, f):
        B = x.shape[0]
        e = self.enc(x.reshape(B * 2, 1, -1)).reshape(B, 2, -1)
        z = torch.cat([e.mean(1), e.max(1).values, (e[:, 0] - e[:, 1]).abs()], 1)
        if self.fproj is not None:
            z = torch.cat([z, self.fproj(f)], 1)
        h = self.trunk(z)
        return self.task(h), self.grp(h), self.va(e).squeeze(-1), self.diag(h)


# ---------------------------------------------------------------- augmentation (same as train.py)
def rep_average(idx, p_keep=0.6):
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


def soft_ce(logits, target, w):
    """Cross-entropy with soft targets (B, C) and per-sample weights w (B,)."""
    return -(w * (target * F.log_softmax(logits, 1)).sum(1)).sum() / w.sum().clamp_min(1e-6)


def train_one(tr, te, Ftr_all, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(FEAT.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    ty, gy = TASK_Y[tr], info.y_grp.values[tr]
    cw_t = len(ty[ty >= 0]) / (2 * np.bincount(ty[ty >= 0], minlength=2))
    cw_g = len(gy) / (4 * np.maximum(np.bincount(gy, minlength=4), 1))
    cw_t, cw_g = torch.tensor(cw_t, device=dev).float(), torch.tensor(cw_g, device=dev).float()
    tr_t = torch.tensor(tr, device=dev)
    for ep in range(args.epochs):
        model.train()
        for b in tr_t[torch.randperm(len(tr), device=dev)].split(args.bs):
            if len(b) < 4:
                continue
            x, f = augment(rep_average(b)), Ftr_all[b]
            t_on = TY[b] >= 0
            t_tgt = F.one_hot(TY[b].clamp_min(0), 2).float() * 0.95 + 0.025
            t_w = t_on.float() * cw_t[TY[b].clamp_min(0)]
            g_tgt = F.one_hot(GRP[b], 4).float() * 0.95 + 0.0125
            g_w = cw_g[GRP[b]]
            va, va_m = VA[b], VA_M[b]
            d_tgt = F.one_hot(DIAG[b], N_DIAG).float()
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                pm = torch.randperm(len(b), device=dev)
                x, f = lam * x + (1 - lam) * x[pm], lam * f + (1 - lam) * f[pm]
                tw1, tw2 = lam * t_w, (1 - lam) * t_w[pm]
                t_tgt = (tw1[:, None] * t_tgt + tw2[:, None] * t_tgt[pm]) / (tw1 + tw2).clamp_min(1e-6)[:, None]
                t_w = tw1 + tw2
                g_tgt, g_w = lam * g_tgt + (1 - lam) * g_tgt[pm], lam * g_w + (1 - lam) * g_w[pm]
                d_tgt = lam * d_tgt + (1 - lam) * d_tgt[pm]
                va, va_m = lam * va + (1 - lam) * va[pm], va_m * va_m[pm]
            lt, lg, lv, ld = model(x, f)
            loss = soft_ce(lt, t_tgt, t_w) if t_w.sum() > 0 else lt.sum() * 0
            if GRP_HEAD:
                loss = loss + args.grp_w * soft_ce(lg, g_tgt, g_w)
            if args.va_w > 0:
                loss = loss + args.va_w * (va_m * F.smooth_l1_loss(lv, va, reduction="none", beta=0.2)).sum() / \
                    va_m.sum().clamp_min(1)
            if args.diag_w > 0:
                loss = loss + args.diag_w * soft_ce(ld, d_tgt, torch.ones(len(b), device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    with torch.no_grad():
        te_t = torch.tensor(te, device=dev)
        lt, lg, _, _ = model(XMEAN[te_t], Ftr_all[te_t])
        ph = torch.softmax(lt, 1)[:, 1]
        pg = torch.softmax(lg, 1)
        pc = pg[:, pos[0]] / (pg[:, pos[0]] + pg[:, neg_grp] + 1e-9) if len(pos) == 1 else 1 - pg[:, neg_grp]
        return ph.cpu().numpy(), pc.cpu().numpy()


g = info.group.values
t0 = time.time()
PH, PC = np.zeros(len(sel)), np.zeros(len(sel))
for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=args.seed).split(sel, y_sel, g[sel])):
    te_g = sel[te]
    tr_g = np.flatnonzero(~np.isin(g, g[te_g])) if POOLED else sel[tr]
    Fz = FEAT.copy()
    if Fz.shape[1]:
        med = np.nanmedian(Fz[tr_g], 0)
        Fz = np.where(np.isnan(Fz), med, Fz)
        Fz = (Fz - Fz[tr_g].mean(0)) / (Fz[tr_g].std(0) + 1e-6)
        Fz = np.clip(Fz, -5, 5)
    Ft = torch.tensor(Fz, dtype=torch.float32, device=dev)
    outs = [train_one(tr_g, te_g, Ft, args.seed * 1000 + f * 50 + j) for j in range(args.n_ens)]
    PH[te] = np.mean([o[0] for o in outs], 0)
    PC[te] = np.mean([o[1] for o in outs], 0)
    print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
P = (PH + PC) / 2 if GRP_HEAD else PH
np.save(OUT / f"oof_seed{args.seed}.npy", P)
if GRP_HEAD:
    np.save(OUT / f"oofhead_seed{args.seed}.npy", PH)
    np.save(OUT / f"oofgrp_seed{args.seed}.npy", PC)
m = {k: round(float(v), 4) for k, v in screen_metrics(y_sel, P).items()}
extra = {"head_auroc": screen_metrics(y_sel, PH)["auroc"], "grp_auroc": screen_metrics(y_sel, PC)["auroc"]} \
    if GRP_HEAD else {}
json.dump({"args": vars(args), "metrics": m, **extra}, open(OUT / f"result_seed{args.seed}.json", "w"), indent=1)
print("RESULT", args.task, args.name, args.seed, m, extra, f"{time.time() - t0:.0f}s", flush=True)
