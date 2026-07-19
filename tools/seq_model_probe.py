"""Sequence-model prototype (notes/27): the 'learned holistic reading'. Everything in our stack predicts
each row INDEPENDENTLY; a human reads the whole log at once. A per-well TCN (dilated temporal conv) over the
eval sequence of our existing signals learns SEQUENTIAL corrections the row-independent GBM/MLP can't. Tests
whether holistic sequence integration adds OOF on top of the current blend (CV-honest: a new decorrelated
model, like MLP/TabICL were). Guarded prior — the GR signal is at its information limit and the residual
drift direction was found unpredictable (notes/25); this is the one ARCHITECTURALLY-novel lever left.

    uv run python -u tools/seq_model_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from src.validation import score

torch.manual_seed(0)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
S = 256  # resample each well's eval zone to S stages


def build():
    d = np.load("/tmp/rogii_meta_oof.npz")
    truth, anchor, fusion, g = d["truth"], d["anchor"], d["fusion"], d["g"]
    gbm, mlp, hmm, pf, mds = d["oof_gbm"], d["oof_mlp"], d["hmm"], d["pf"], d["md_since"]
    uniq, first = np.unique(g, return_index=True); counts = np.diff(np.append(first, len(g)))
    # per-row feature block (anchor-relative), resample each well to S stages
    if os.environ.get("SEQ_DROP_GBMMLP") == "1":   # keep only honest per-well signals (no fit-on-train)
        cols = [hmm - anchor, pf - anchor, fusion - anchor, mds / 5000.0]
    else:
        cols = [hmm - anchor, pf - anchor, gbm, mlp, fusion - anchor, mds / 5000.0]
    if os.environ.get("SEQ_RICH") == "1":
        r = np.load("/tmp/rogii_rich.npz")
        for k in r.files:                       # raw GR-vs-typewell correlation surface (the "reading")
            v = np.nan_to_num(r[k].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(v)
        log(f"  + {len(r.files)} rich GR-match channels")
    if os.environ.get("SEQ_X") == "1":          # per-well trajectory + GR-envelope + tracker (CV-honest)
        from tools.frontier_fast import load_stack
        Xs = load_stack()["X"]
        for j in range(Xs.shape[1]):
            cols.append(np.nan_to_num(Xs[:, j].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        log(f"  + {Xs.shape[1]} stack-X per-well channels")
    feat_rows = np.column_stack(cols).astype(np.float32)
    F = feat_rows.shape[1]
    X = np.zeros((len(uniq), S, F), np.float32); Y = np.zeros((len(uniq), S), np.float32)
    meta = []  # (first, count, orig_ef) to map back
    for i, (f, c) in enumerate(zip(first, counts)):
        ef = np.linspace(0, 1, c); grid = np.linspace(0, 1, S)
        for j in range(F):
            X[i, :, j] = np.interp(grid, ef, feat_rows[f:f+c, j])
        Y[i] = np.interp(grid, ef, (truth - anchor)[f:f+c])
        meta.append((f, c))
    return X, Y, truth, anchor, fusion, g, uniq, meta, gbm, mlp


class TCN(nn.Module):
    def __init__(self, F, ch=48):
        super().__init__()
        def blk(i, o, d): return nn.Sequential(
            nn.Conv1d(i, o, 5, padding=2*d, dilation=d), nn.ReLU(), nn.BatchNorm1d(o))
        self.net = nn.Sequential(blk(F, ch, 1), blk(ch, ch, 2), blk(ch, ch, 4), blk(ch, ch, 8))
        self.head = nn.Conv1d(ch, 1, 1)
    def forward(self, x):            # x: (N, S, F)
        h = self.net(x.transpose(1, 2))
        return self.head(h).squeeze(1)   # (N, S)


def main():
    X, Y, truth, anchor, fusion, g, uniq, meta, gbm, mlp = build()
    log(f"{len(uniq)} wells, seq {X.shape}, feat {X.shape[2]}")
    # standardize features (per channel)
    mu = X.reshape(-1, X.shape[2]).mean(0); sd = X.reshape(-1, X.shape[2]).std(0) + 1e-6
    Xn = (X - mu) / sd
    ys = Y.std() + 1e-6
    dev = "cpu"
    # torch 2.2.2 (Intel-mac ceiling) can't bridge numpy 2.x -> go via Python lists
    tt = lambda a: torch.tensor(np.asarray(a, dtype=np.float32).tolist(), dtype=torch.float32)
    N_SEED = int(os.environ.get("SEQ_SEEDS", "3"))
    oof_seq_rows = np.zeros(len(truth), np.float32)   # full-res OOF residual prediction
    for fold, (tr, va) in enumerate(GroupKFold(5).split(uniq, groups=uniq)):
        Xt = tt(Xn[tr]); Yt = tt(Y[tr] / ys)
        Xv = tt(Xn[va])
        pv = np.zeros((len(va), S), np.float32)
        soff = int(os.environ.get("SEQ_SEED_OFFSET", "0"))
        for seed in range(soff, soff + N_SEED):
            torch.manual_seed(seed)
            net = TCN(X.shape[2]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
            lossf = nn.SmoothL1Loss()
            net.train()
            idx = np.arange(len(tr)); bs = 64
            for ep in range(60):
                np.random.RandomState(ep + 100 * seed).shuffle(idx)
                for b in range(0, len(idx), bs):
                    bi = idx[b:b+bs].tolist()
                    opt.zero_grad(); out = net(Xt[bi]); loss = lossf(out, Yt[bi]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv += np.asarray(net(Xv).tolist(), dtype=np.float32) * ys
        pv /= N_SEED                                  # (n_va, S) seed-averaged residual at stages
        # map each va well's S-stage prediction back to full rows
        for k, wi in enumerate(va):
            f, c = meta[wi]; grid = np.linspace(0, 1, S); ef = np.linspace(0, 1, c)
            oof_seq_rows[f:f+c] = np.interp(ef, grid, pv[k])
        log(f"fold {fold} done")
    # baseline and blends
    base_corr = 0.7 * gbm + 0.3 * mlp
    base = 0.75 * (anchor + base_corr) + 0.25 * fusion
    print(f"\n=== baseline pooled OOF: {score(truth, base):.4f} ===", flush=True)
    print("seq-model alone (as the correction):", flush=True)
    print(f"  {score(truth, 0.75*(anchor+oof_seq_rows)+0.25*fusion):.4f}", flush=True)
    print("blend seq into the correction: w_seq*seq + (1-w_seq)*base_corr", flush=True)
    for w in (0.1, 0.2, 0.3, 0.5):
        gp = w * oof_seq_rows + (1 - w) * base_corr
        print(f"  w_seq={w:.1f}: {score(truth, 0.75*(anchor+gp)+0.25*fusion):.4f}", flush=True)


if __name__ == "__main__":
    main()
