"""Frontier feature groups, tested against the lockbox baseline (13.742).
Keep a group only if it beats the lockbox. Extend this file per group."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tools.frontier_cv import Harness

h = Harness()
print(f"dev={h.dev_mask.sum()} lock={h.lock_mask.sum()}", flush=True)
h.report("BASE")

d, hp, hs, pp, ps, anchor = h.d, h.hp, h.hs, h.pp, h.ps, h.anchor

# Group A: inter-signal uncertainty (transforms the GBM can't derive from raw cols)
absdiff = np.abs(hp - pp)
stdsum = hs + ps
norm_dis = absdiff / (stdsum + 1.0)
hmm_a = np.abs(hp - anchor); pf_a = np.abs(pp - anchor)
A = np.column_stack([absdiff, stdsum, norm_dis, hmm_a, pf_a])
h.report("+ uncertainty(A)", extra=A)

# Groups B (GR envelope/energy) and C (multi-scale NCC alignment quality)
fr = np.load("/tmp/rogii_frontier.npz")
B = np.column_stack([fr[k] for k in ["gmax51", "gmin51", "grms51", "gstd31", "gstd151", "gabsd"]])
C = np.column_stack([fr[k] for k in ["ncc8", "ncc15", "ncc25"]])
h.report("+ GR-envelope(B)", extra=B)
h.report("+ NCC-quality(C)", extra=C)
h.report("+ A+B", extra=np.column_stack([A, B]))
h.report("+ A+C", extra=np.column_stack([A, C]))
h.report("+ B+C", extra=np.column_stack([B, C]))
h.report("+ A+B+C", extra=np.column_stack([A, B, C]))
