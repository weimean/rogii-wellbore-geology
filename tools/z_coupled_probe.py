"""z-velocity-coupled predictor (notes/24): a GEOMETRY prior, dTVT/dMD = beta*dZ/dMD + c, fit per well
on the known prefix, extrapolated over the eval zone using the KNOWN eval Z path. Distinct from all our
GR-driven trackers (HMM/PF) — the only genuinely-orthogonal candidate left (notes/22-23). This is the
deterministic mean of the sub-9 teams' run_pf_z. Test: does it beat ffill/anchor, and is it decorrelated
from our existing hmm/anchor?

    uv run python -u tools/z_coupled_probe.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from src.data import load_split
from src.validation import score

TI = "TVT_input"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def zc_predict(Zk, MDk, Tk, Ze, MDe, last, mode):
    """Fit on known increments, extrapolate cumulatively over eval using known eval Z/MD."""
    dTk = np.diff(Tk); dZk = np.diff(Zk); dMk = np.diff(MDk)
    ok = np.isfinite(dTk) & np.isfinite(dZk) & np.isfinite(dMk) & (dMk != 0)
    if ok.sum() < 20:
        return np.full(len(Ze), last)
    dTk, dZk, dMk = dTk[ok], dZk[ok], dMk[ok]
    if mode == "z":                       # dT = beta*dZ  (pure geometry, c=0)
        beta = np.dot(dZk, dTk) / (np.dot(dZk, dZk) + 1e-9); c = 0.0
    elif mode == "zmd":                   # dT = beta*dZ + c*dMD
        A = np.column_stack([dZk, dMk])
        coef = np.linalg.lstsq(A, dTk, rcond=None)[0]; beta, c = coef[0], coef[1]
    else:                                 # md only (free trend, the R^2~0 baseline)
        beta = 0.0; c = np.dot(dMk, dTk) / (np.dot(dMk, dMk) + 1e-9)
    # extrapolate cumulatively from the last known point over eval rows
    dZe = np.diff(np.concatenate([[Zk[-1]], Ze]))
    dMe = np.diff(np.concatenate([[MDk[-1]], MDe]))
    return last + np.cumsum(beta * dZe + c * dMe)


def main():
    wells = load_split(Path("data/raw/train"))
    aT = []; aFF = []; preds = {m: [] for m in ("z", "zmd", "md")}; aHMM = []
    from src.model import hmm_posterior, level_regression, fit_level_model
    lm = fit_level_model(wells)
    aANCH = []
    kept = 0
    for w in wells:
        h = w.horizontal
        if "TVT" not in h.columns:
            continue
        em = w.eval_mask.to_numpy()
        kn = h[h[TI].notna()];
        if em.sum() < 20 or len(kn) < 200:
            continue
        Z = h["Z"].to_numpy(float); MD = h["MD"].to_numpy(float); T = h["TVT"].to_numpy(float)
        ei = np.where(em)[0]; ki = np.where(h[TI].notna().to_numpy())[0]
        last = float(T[ki[-1]])
        for m in preds:
            preds[m].append(zc_predict(Z[ki], MD[ki], T[ki], Z[ei], MD[ei], last, m))
        aT.append(T[ei]); aFF.append(np.full(len(ei), last))
        hp, _ = hmm_posterior(w); aHMM.append(hp[em])
        aANCH.append(level_regression(w, lm)[em])
        kept += 1
    aT = np.concatenate(aT); aFF = np.concatenate(aFF); aHMM = np.concatenate(aHMM); aANCH = np.concatenate(aANCH)
    def rmse(p): return float(np.sqrt(np.mean((aT - p) ** 2)))
    log(f"{len(aT)} eval rows, {kept} wells")
    print(f"  ffill:          {rmse(aFF):.3f}", flush=True)
    print(f"  anchor (level): {rmse(aANCH):.3f}", flush=True)
    print(f"  HMM:            {rmse(aHMM):.3f}", flush=True)
    for m in ("md", "z", "zmd"):
        p = np.concatenate(preds[m])
        cH = np.corrcoef(p, aHMM)[0, 1]; cA = np.corrcoef(p, aANCH)[0, 1]
        print(f"  z-coupled[{m:3s}]: {rmse(p):7.3f}   corr(hmm)={cH:.3f} corr(anchor)={cA:.3f}", flush=True)
    # does the z-coupled pred help when blended with the anchor? (proxy for feature value)
    for m in ("z", "zmd"):
        p = np.concatenate(preds[m])
        for wgt in (0.15, 0.3, 0.5):
            print(f"    anchor + {wgt}*(zc[{m}]-anchor): {rmse(aANCH + wgt*(p-aANCH)):.3f}", flush=True)


if __name__ == "__main__":
    main()
