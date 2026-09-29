"""Frozen time-series foundation-model embeddings of every eye's repetition-averaged PERG (label-free, so computing
them once for all records leaks nothing). Saves runs/fm_<model>.npy with shape (N records, 2 eyes, d).
Run with the separate env: ../.venv_fm/bin/python fm_embed.py [moment,mantis,chronos]
TiRex (NX-AI, xLSTM) needs its own env: ../.venv_tirex/bin/python fm_embed.py tirex,tirexaug
"""
import sys
from pathlib import Path

import numpy as np
import torch

from data import load

OUT = Path(__file__).resolve().parents[1] / "runs"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def resample(x, L=512):
    t = np.linspace(0, 1, x.shape[-1])
    tn = np.linspace(0, 1, L)
    return np.stack([np.interp(tn, t, s) for s in x.reshape(-1, x.shape[-1])]).reshape(*x.shape[:-1], L)


def moment(x):
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"})
    m.init()
    m.to(dev).eval()
    X = torch.tensor(resample(x), dtype=torch.float32)[:, None]
    out = []
    with torch.no_grad():
        for b in X.split(64):
            out.append(m(x_enc=b.to(dev)).embeddings.cpu().numpy())
    return np.concatenate(out)


def mantis(x):
    from mantis.architecture import Mantis8M
    from mantis.trainer import MantisTrainer
    net = Mantis8M(device=dev).from_pretrained("paris-noah/Mantis-8M")
    return np.asarray(MantisTrainer(device=dev, network=net).transform(resample(x)[:, None].astype(np.float32)))


def chronos(x):
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-base", device_map=dev, torch_dtype=torch.float32)
    out = []
    for b in torch.tensor(x, dtype=torch.float32).split(64):
        emb, _ = pipe.embed(b)
        out.append(emb.float().mean(1).cpu().numpy())
    return np.concatenate(out)


def tirex(x, aug=False):
    """TiRex embedding API: per-layer hidden states averaged over patches, L2/layer-normed; aug adds the
    first-difference embedding + summary stats (dataset-standardised, label-free)."""
    from tirex.models.embedding import TiRexEmbedding
    m = TiRexEmbedding(device=dev, data_augmentation=aug, batch_size=128).eval()
    with torch.no_grad():
        return m(torch.tensor(x, dtype=torch.float32)[:, None]).cpu().numpy()


if __name__ == "__main__":
    mean = load()["mean"].astype(np.float64)  # (N, 2, 255), µV / 5
    x = mean.reshape(-1, 255)
    for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["moment", "mantis", "chronos"]):
        E = {"moment": moment, "mantis": mantis, "chronos": chronos, "tirex": tirex,
             "tirexaug": lambda v: tirex(v, True)}[name](x)
        E = E.reshape(len(mean), 2, -1)
        np.save(OUT / f"fm_{name}.npy", E)
        print(name, E.shape, flush=True)
