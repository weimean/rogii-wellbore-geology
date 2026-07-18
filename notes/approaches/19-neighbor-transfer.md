# 19 — Neighbour TVT-curve transfer — NEGATIVE (the wiggle doesn't transfer) — 2026-07-18

Tried the ROGII webinar's "precise" method: for close-neighbour wells, transfer the SINGLE nearest
other-well point's datum D=TVT+Z (wiggle-preserving), calibrated per-well on the known zone, to capture the
wiggle our smooth surfaces miss. On close-neighbour wells (<30 ft, ~3% of rows):
- ffill 16.47 · **smooth mean-surface (our formation-plane) 8.40** · **NN-transfer 25.21 (much WORSE)**.
**Physical conclusion: the WIGGLE is NOT spatially transferable between wells.** Two wells even 30 ft apart
have different local TVT wiggles (each well's own path through local heterogeneity), so a single neighbour's
D just injects noise; averaging (the smooth surface) is far better and already-optimal for the DATUM. The
webinar's "copy the neighbour's wiggle" is actually SELF-correlation within one well re-crossing a bed — not
two different wells — and that self-correlation we already closed as non-reproducible as a blind scan (notes/16).

**The 18% wiggle is genuinely irreducible for us:** not from GR (closed ×3, notes/15-16), not from neighbour
transfer (well-specific). Only the level/datum is recoverable (spatial formation-plane, done). tools/
neighbor_transfer_probe.py. **BEST STANDS: kernel v20, public 11.020.**
