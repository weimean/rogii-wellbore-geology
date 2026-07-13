"""Frontier-build validation harness with a LOCKBOX.

Wells are split dev (80%) / lockbox (20%, fixed seed, never tuned on). New feature groups
are added to the base gbm_stack+PF matrix and kept only if they improve the LOCKBOX
(overall + deep-row RMSE, since the private set is ~74% deep). Reuses cached HMM/PF
posteriors (/tmp/rogii_feats.npz, /tmp/rogii_pf.npz).

API:
    from tools.frontier_cv import Harness
    h = Harness()
    h.report("base")                    # baseline dev-CV + lockbox
    h.report("base+X", extra=Xextra)    # Xextra: (n_rows, k) aligned to h.order
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from src.validation import score

STACK_FEATS = ["hmm_pred", "hmm_std", "z_from_ps", "md_from_ps", "horiz_from_ps", "dz",
               "tort", "gr50", "gr_dev", "az_sin", "az_cos", "eval_frac",
               "v50", "v200", "gr_off", "known_range", "eval_len"]
LVL = ["v50", "v200", "gr_off", "eval_len", "known_range"]


class Harness:
    def __init__(self, lockbox_frac=0.2, seed=42):
        d = np.load("/tmp/rogii_feats.npz")
        pf = np.load("/tmp/rogii_pf.npz")
        assert len(d["truth"]) == len(pf["pf_pred"]), "cache misalignment"
        self.d, self.pf = d, pf
        self.g = d["gidx"]; self.truth = d["truth"]; self.last = d["last_tvt"]
        self.hp, self.hs = d["hmm_pred"], d["hmm_std"]
        self.pp, self.ps = pf["pf_pred"], pf["pf_std"]
        self.eval_frac = d["eval_frac"]
        self.order = np.arange(len(self.truth))  # row order features must align to

        uniq = np.unique(self.g)
        rng = np.random.default_rng(seed)
        lock_wells = set(rng.choice(uniq, int(len(uniq) * lockbox_frac), replace=False).tolist())
        self.dev_mask = np.array([gi not in lock_wells for gi in self.g])
        self.lock_mask = ~self.dev_mask
        self.uniq, self.first = uniq, {gi: i for i, gi in enumerate(np.unique(self.g, return_index=True)[1])}
        self._Xl_well = np.column_stack([d[k] for k in LVL])
        self.gk = GroupKFold(5)

        # per-well level table
        u2, first_idx = np.unique(self.g, return_index=True)
        self.well_ids = u2
        self.well_X = self._Xl_well[first_idx]
        self.well_off = np.array([self.truth[self.g == gi].mean() - self.last[self.g == gi][0] for gi in u2])
        self.well_is_dev = np.array([bool(self.dev_mask[self.g == gi][0]) for gi in u2])

        self._build_anchors()

    def _build_anchors(self):
        # dev OOF anchor (5-fold within dev wells); lockbox anchor = dev-fit
        dev_w = np.where(self.well_is_dev)[0]
        oof_off = np.full(len(self.well_ids), np.nan)
        for tr, va in KFold(5, shuffle=True, random_state=0).split(dev_w):
            trw, vaw = dev_w[tr], dev_w[va]
            m = Ridge(1.0).fit(self.well_X[trw], self.well_off[trw])
            oof_off[vaw] = m.predict(self.well_X[vaw])
        m_all = Ridge(1.0).fit(self.well_X[dev_w], self.well_off[dev_w])
        lock_w = np.where(~self.well_is_dev)[0]
        oof_off[lock_w] = m_all.predict(self.well_X[lock_w])   # dev-fit for lockbox
        off_of = {gi: oof_off[i] for i, gi in enumerate(self.well_ids)}
        self.anchor = self.last + np.array([off_of[gi] for gi in self.g])

    def _base_X(self, anchor):
        d = self.d; hp, hs, pp, ps = self.hp, self.hs, self.pp, self.ps
        wt = 1 / (1 + (hs / 4) ** 2); fusion = wt * hp + (1 - wt) * anchor
        wtp = 1 / (1 + (ps / 4) ** 2); pf_fus = wtp * pp + (1 - wtp) * anchor
        cols = [d[k] for k in STACK_FEATS] + [anchor, hp - anchor, fusion, wt,
                                              pp, ps, pp - anchor, hp - pp, pf_fus]
        return np.column_stack(cols), fusion

    def _oof_gbm(self, X, y, mask):
        """5-fold OOF within the masked (dev) rows; returns full-length preds (0 outside)."""
        p = np.zeros(len(y)); idx = np.where(mask)[0]
        gm = self.g[idx]
        for tr, va in self.gk.split(idx, groups=gm):
            m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                              min_samples_leaf=200, l2_regularization=1.0, random_state=0)
            m.fit(X[idx[tr]], y[idx[tr]]); p[idx[va]] = m.predict(X[idx[va]])
        return p

    def report(self, tag, extra=None):
        anchor = self.anchor
        X, fusion = self._base_X(anchor)
        if extra is not None:
            X = np.column_stack([X, extra])
        y = self.truth - anchor
        dev, lock = self.dev_mask, self.lock_mask

        # dev CV (OOF within dev)
        dev_gbm = self._oof_gbm(X, y, dev)
        dev_pred = 0.6 * (anchor + dev_gbm) + 0.4 * fusion

        # lockbox: fit GBM on ALL dev rows, predict lockbox
        m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        m.fit(X[dev], y[dev])
        lock_pred = 0.6 * (anchor[lock] + m.predict(X[lock])) + 0.4 * fusion[lock]

        deep = self.eval_frac[lock] > 0.5
        dcv = score(self.truth[dev], dev_pred[dev])
        lcv = score(self.truth[lock], lock_pred)
        ldeep = score(self.truth[lock][deep], lock_pred[deep])
        print(f"  {tag:28s} dev-CV={dcv:.3f}  lockbox={lcv:.3f}  lockbox-deep={ldeep:.3f}", flush=True)
        return dict(dev=dcv, lock=lcv, lock_deep=ldeep)


if __name__ == "__main__":
    h = Harness()
    print(f"dev rows={h.dev_mask.sum()}  lockbox rows={h.lock_mask.sum()}  "
          f"(dev wells={h.well_is_dev.sum()}, lockbox wells={(~h.well_is_dev).sum()})", flush=True)
    h.report("BASE (gbm_stack+PF)")
