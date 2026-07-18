# 17 — TabICL foundation model (a genuinely-untried lever that worked) — 2026-07-17

After the GR lever closed, tried the top team's TabICL (tabular foundation model, in-context learning).
Blocked earlier (no torch on Intel Mac); now runnable on Kaggle.

## Value (Kaggle CPU, offline, on our 113-feat stack subsample, GroupKFold)
GBM 10.77 · **TabICL 10.68** (beats GBM despite 20× less training data — in-context learning is data-efficient)
· 0.5 GBM + 0.5 TabICL **10.55** (−0.22), corr 0.91. Genuine different-function-class diversity (like the MLP).

## Delivery (the hard part — solved)
- Kaggle P100 GPU is UNUSABLE: torch 2.10+cu128 dropped the P100 arch (sm_60) → "no kernel image". TabICL
  runs fine on CPU, so enable_gpu=false (GBM boosters keep CPU fallback).
- Internet-off submission → OFFLINE bundle: `pip download tabicl einops` wheels + the 114MB regressor ckpt
  (from HF jingang/TabICL, grabbed via a helper kernel) → Kaggle dataset `gtwm98/rogii-tabicl-offline`.
  Kernel: `pip install --no-index --no-deps` the wheels; `TabICLRegressor(model_path=ckpt,
  allow_auto_download=False, device="cpu", n_estimators=4, kv_cache=True)`.
- Runtime: predict on ~1M scored-test rows on CPU would blow the 9h cap → predict per-well on a coarse
  ~40-row subsample + INTERPOLATE the smooth correction; fit on 10k context. Kernel ran ~1h24m.
- Blend: pred = 0.65·(0.7 GBM + 0.3 MLP) + 0.35·TabICL. Serialised with a lock (threaded predict-loop safe).

## Result: public 11.042 → **11.020** (−0.022), kernel v20 = new BEST.
Smaller than the −0.22 CV signal (MLP overlap + 0.35 weight + subsample smoothing). Likely more to extract:
higher blend (0.5), more context (15-20k), finer subsample (60-80/well), n_estimators=8.

## TabPFN (2nd foundation model) — BLOCKED by license (2026-07-18)
Value-tested TabPFN alongside TabICL on Kaggle: `TabPFNLicenseError` — TabPFN needs a one-time interactive
license acceptance / TABPFN_TOKEN (browser account at ux.priorlabs.ai) to download weights; a kernel has no
interactive terminal, and internet is off for the submission anyway. Requires the USER to accept the license
+ supply a token (I don't create accounts / accept licenses). Even unblocked it's a foundation model like
TabICL -> likely redundant. Not pursued. TabICL (open, no token) remains the foundation-model lever.
BEST STANDS: kernel v20, public 11.020.
