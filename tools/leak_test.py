"""Test the leak theory WITHOUT submitting: score our phase-2 predictions on the 3 test wells
against the truth in their train twins. If it reproduces our public score (13.601), the public
LB is exactly these 3 leaked wells.

    uv run python -u tools/leak_test.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

TEST = ["000d7d20", "00bbac68", "00e12e8b"]


def main():
    sub = pd.read_csv("outputs/submission_gbm_stack.csv")
    # parse id = {well}_{rowindex}
    sub["well"] = sub["id"].str.rsplit("_", n=1).str[0]
    sub["ridx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)

    all_pred, all_true, all_ff = [], [], []
    print(f"{'well':10s} {'eval_rows':>9s} {'our_RMSE':>9s} {'leak_RMSE':>9s} {'ffill_RMSE':>10s}", flush=True)
    for w in TEST:
        twin = pd.read_csv(f"data/raw/train/{w}__horizontal_well.csv")   # has true TVT
        s = sub[sub["well"] == w].sort_values("ridx")
        truth = twin["TVT"].to_numpy(float)[s["ridx"].to_numpy()]
        pred = s["tvt"].to_numpy(float)
        # ffill baseline for reference: last known TVT_input before the eval zone
        last_known = twin["TVT_input"].dropna().to_numpy(float)
        ff = np.full(len(truth), last_known[-1]) if len(last_known) else truth
        leak = truth  # perfect lookup
        def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
        print(f"{w:10s} {len(s):9d} {rmse(pred,truth):9.3f} {rmse(leak,truth):9.3f} {rmse(ff,truth):10.3f}", flush=True)
        all_pred.append(pred); all_true.append(truth); all_ff.append(ff)

    P = np.concatenate(all_pred); T = np.concatenate(all_true); F = np.concatenate(all_ff)
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"\n  POOLED over 3 wells:", flush=True)
    print(f"    our phase-2 RMSE : {rmse(P,T):.3f}   (our public score = 13.601)", flush=True)
    print(f"    leak-lookup RMSE : {rmse(T,T):.3f}   (copy the twin's TVT -> ~0, fully gameable)", flush=True)
    print(f"    ffill RMSE       : {rmse(F,T):.3f}", flush=True)


if __name__ == "__main__":
    main()
