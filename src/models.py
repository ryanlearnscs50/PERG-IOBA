"""TCN encoder for PERG traces, clinical-metadata embedding (concat / FiLM), and eye-symmetric record head."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEmbedding(nn.Module):
    """Age (Fourier features of normalised age) + sex (lookup table) -> d-dim embedding."""

    def __init__(self, d=32, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2 ** torch.arange(n_freq).float() * math.pi / 8)
        self.age = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(), nn.Linear(d, d))
        self.sex = nn.Embedding(2, d)

    def forward(self, age, sex):
        a = ((age - 37.0) / 18.0)[:, None]
        f = torch.cat([a, torch.sin(a * self.freqs), torch.cos(a * self.freqs)], 1)
        return self.age(f) + self.sex(sex)


class TCNBlock(nn.Module):
    def __init__(self, ch, k, dil, dropout, causal, film_dim=0):
        super().__init__()
        self.pad = (k - 1) * dil
        self.causal = causal
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(ch, ch, k, dilation=dil))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(ch, ch, k, dilation=dil))
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(film_dim, 2 * ch) if film_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def _pad(self, x):
        if self.causal:
            return F.pad(x, (self.pad, 0))
        return F.pad(x, (self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x, cond=None):
        h = self.drop(F.gelu(self.c1(self._pad(x))))
        h = self.c2(self._pad(h))
        if self.film is not None and cond is not None:
            g, b = self.film(cond).unsqueeze(-1).chunk(2, 1)
            h = h * (1 + g) + b
        return F.gelu(x + self.drop(h))


class TCNEncoder(nn.Module):
    """Single-eye trace (B, 1, T) -> embedding (B, 2*ch). Input gets a time-ramp channel so absolute
    latency (P50 ~50 ms, N95 ~95 ms) is visible to the translation-equivariant convs."""

    def __init__(self, ch=64, levels=6, k=5, dropout=0.1, causal=False, film_dim=0, T=255):
        super().__init__()
        self.register_buffer("ramp", torch.linspace(-1, 1, T)[None, None])
        self.inp = nn.Conv1d(2, ch, 1)
        self.blocks = nn.ModuleList(TCNBlock(ch, k, 2 ** i, dropout, causal, film_dim) for i in range(levels))
        self.attn = nn.Conv1d(ch, 1, 1)
        self.out_dim = 2 * ch

    def forward(self, x, cond=None):
        h = self.inp(torch.cat([x, self.ramp.expand(x.shape[0], -1, -1)], 1))
        for blk in self.blocks:
            h = blk(h, cond)
        w = torch.softmax(self.attn(h), -1)
        return torch.cat([(h * w).sum(-1), h.mean(-1)], 1)


class PERGNet(nn.Module):
    """Shared encoder per eye; eyes combined symmetrically (mean, max, |diff|) so RE/LE order is irrelevant.
    meta: 'none' | 'concat' (metadata embedding into head) | 'film' (conditions every TCN block) | 'both'."""

    def __init__(self, n_out, meta="none", meta_dim=32, ch=64, levels=6, k=5, dropout=0.1, causal=False):
        super().__init__()
        self.meta = meta
        self.meta_emb = MetaEmbedding(meta_dim) if meta != "none" else None
        film_dim = meta_dim if meta in ("film", "both") else 0
        self.enc = TCNEncoder(ch, levels, k, dropout, causal, film_dim)
        d = 3 * self.enc.out_dim + (meta_dim if meta in ("concat", "both") else 0)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 64), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(64, n_out))

    def embed(self, x, age, sex):
        """x: (B, 2, T) -> record embedding before the head."""
        B = x.shape[0]
        m = self.meta_emb(age, sex) if self.meta_emb is not None else None
        cond = m.repeat_interleave(2, 0) if (m is not None and self.meta in ("film", "both")) else None
        e = self.enc(x.reshape(B * 2, 1, -1), cond).reshape(B, 2, -1)
        z = torch.cat([e.mean(1), e.max(1).values, (e[:, 0] - e[:, 1]).abs()], 1)
        if self.meta in ("concat", "both"):
            z = torch.cat([z, m], 1)
        return z

    def forward(self, x, age, sex):
        return self.head(self.embed(x, age, sex))
