"""Interpretation diagnostic (notes/25): find WHERE our automated pipeline systematically fails, so an
interpreter (me) can look at those wells and diagnose the geological failure mode. Ranks wells by OOF
error, characterizes worst vs best on interpretable axes (GR variability, dip magnitude, eval length,
neighbour distance, formation availability), and saves GR-alignment plots of the worst wells to inspect.

    uv run python -u tools/err_interpret_diag.py
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from src.data import load_split
from src.validation import score

TI = "TVT_input"
OUT = "/private/tmp/claude-501/-Users-tsoiwaimun-kaggle-rogii-wellbore-geology/90bb14ac-958c-4de1-b12a-af1b4ba3d4d4/scratchpad"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    d = np.load("/tmp/rogii_meta_oof.npz")
    truth, anchor, fusion, g = d["truth"], d["anchor"], d["fusion"], d["g"]
    gbm, mlp = d["oof_gbm"], d["oof_mlp"]
    pred = 0.75 * (anchor + 0.7 * gbm + 0.3 * mlp) + 0.25 * fusion
    uniq, first = np.unique(g, return_index=True)
    counts = np.diff(np.append(first, len(g)))
    werr = np.array([np.sqrt(np.mean((truth[f:f+c] - pred[f:f+c])**2)) for f, c in zip(first, counts)])
    aerr = np.array([np.sqrt(np.mean((truth[f:f+c] - anchor[f:f+c])**2)) for f, c in zip(first, counts)])
    order = np.argsort(werr)[::-1]
    log(f"{len(werr)} wells. pooled {score(truth,pred):.3f}. per-well err: median {np.median(werr):.2f} "
        f"p90 {np.percentile(werr,90):.2f} max {werr.max():.2f}")
    print(f"  worst 15 wells (OOF RMSE): {np.round(werr[order[:15]],1)}", flush=True)
    print(f"  their anchor RMSE:         {np.round(aerr[order[:15]],1)}  (is the LEVEL or the WIGGLE wrong?)", flush=True)

    # load wells to characterize (align to g order: load_split with the _stack_features guard)
    wells = load_split(Path("data/raw/train"))
    FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    feats = []; kept_wells = []
    gi = 0
    for w in wells:
        h = w.horizontal; em = w.eval_mask.to_numpy()
        kn = h[h[TI].notna()]
        if em.sum() == 0 or len(kn) < 200 or "TVT" not in h.columns:
            continue
        ei = np.where(em)[0]
        gr = h["GR"].interpolate(limit_direction="both").to_numpy(float)
        T = h["TVT"].to_numpy(float)
        grstd = float(np.nanstd(gr[ei]))                       # GR variability in eval zone
        diprange = float(np.nanmax(T[ei]) - np.nanmin(T[ei]))  # structural relief over eval
        evlen = int(em.sum())
        hasf = int(all(c in h.columns for c in FORMS))
        feats.append((grstd, diprange, evlen, hasf)); kept_wells.append(w)
        gi += 1
    feats = np.array(feats)
    log(f"characterized {len(feats)} wells (aligned {len(feats)}=={len(werr)}? {len(feats)==len(werr)})")
    if len(feats) == len(werr):
        wq = np.argsort(werr); best = wq[:80]; worst = wq[-80:]
        names = ["GR_std_eval", "dip_range_eval", "eval_len", "has_formation"]
        print("\n  axis            best-80 median   worst-80 median", flush=True)
        for j, nm in enumerate(names):
            print(f"  {nm:15s} {np.median(feats[best,j]):12.1f}   {np.median(feats[worst,j]):12.1f}", flush=True)
        print(f"\n  corr(werr, GR_std)={np.corrcoef(werr,feats[:,0])[0,1]:+.2f}  "
              f"corr(werr, dip_range)={np.corrcoef(werr,feats[:,1])[0,1]:+.2f}  "
              f"corr(werr, eval_len)={np.corrcoef(werr,feats[:,2])[0,1]:+.2f}", flush=True)

    # save GR-alignment plots for the 6 worst wells
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(feats) == len(werr):
            fig, axes = plt.subplots(6, 2, figsize=(15, 20))
            for r, wi in enumerate(order[:6]):
                w = kept_wells[wi]; h = w.horizontal; t = w.typewell; em = w.eval_mask.to_numpy()
                md = h["MD"].to_numpy(float); T = h["TVT"].to_numpy(float)
                Ti = h[TI].to_numpy(float); gr = h["GR"].interpolate(limit_direction="both").to_numpy(float)
                pr = pred[first[wi]:first[wi]+counts[wi]]; an = anchor[first[wi]:first[wi]+counts[wi]]
                ei = np.where(em)[0]
                ax = axes[r, 0]
                ax.plot(md, T, "k-", lw=1, label="true TVT")
                ax.plot(md[~np.isnan(Ti)], Ti[~np.isnan(Ti)], "b-", lw=2, label="known (input)")
                ax.plot(md[ei], pr, "r-", lw=1, label="our pred")
                ax.plot(md[ei], an, "g--", lw=1, label="anchor")
                ax.set_title(f"well#{wi} OOF RMSE {werr[wi]:.0f} (anchorRMSE {aerr[wi]:.0f})"); ax.legend(fontsize=7)
                ax.set_xlabel("MD"); ax.set_ylabel("TVT")
                ax2 = axes[r, 1]
                ax2.plot(md, gr, "purple", lw=0.5, label="horizontal GR")
                tw_tvt = t["TVT"].to_numpy(float); tw_gr = t["GR"].ffill().bfill().to_numpy(float)
                ax2.plot(tw_tvt, tw_gr, "orange", lw=1, label="typewell GR(TVT)")
                ax2.set_title(f"GR: horiz vs typewell (GR_std_eval {np.nanstd(gr[ei]):.0f})"); ax2.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(f"{OUT}/worst_wells.png", dpi=70); plt.close()
            log(f"saved {OUT}/worst_wells.png")
    except Exception as e:
        log(f"plot failed: {e}")


if __name__ == "__main__":
    main()
