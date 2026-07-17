# 16 — ROGII StarSteer webinar (domain ground-truth) — 2026-07-17

Watched the ROGII StarSteer geosteering webinar (VgzFt7xknGo, 51 min; ROGII = competition sponsor).
The domain method behind this exact problem.

## 3 levels of geosteering (we only have level-1 data: GR + typewell)
1. **Correlation** (logs) — two approaches (below). 2. **Image-log** apparent dips (borehole imager). 3.
**Resistivity inversion** (boundaries at a distance). We only have GR/typewell → level 1.

## The two correlation methods (explains our results)
- **MODEL-BASED**: propagate TYPEWELL GR along trajectory via a dip interpretation → synthetic GR; match
  vs horizontal GR, tune dips. Presenter: **"not so precise with lateral changes."** ⟵ this IS our
  HMM/PF/GR-match-grid family; the sponsor confirms it caps out.
- **STRAT-BASED (the precise one)**: correlate in TVT scale using the well's OWN high-res horizontal log,
  against the CLOSEST recent data ("~200 m before"), NOT the far typewell. On re-crossing a formation:
  **"I correlate the log on itself — don't need the typewell."**

## Key physics
- Horizontal log has FAR higher vertical resolution than typewell (undulating over km samples a 1 m interval
  ~10⁴× vs once) → sees thin layers the smoother typewell misses. Self-correlation exploits this.
- Everything reduces to BED DIP: dips change by fractions of a degree (89.9/90/91.4°) and that tiny change
  "places the well exactly." TVT ≈ integral of a near-constant dip along the trajectory.
- Multi-typewell: switch to the closest typewell as lateral changes occur (heel typewell vs TD typewell).

## Mapping to our pipeline
- VALIDATES our spatial/offset lever ("always the closest data point" = formation-plane KNN, −0.74/−1.11).
- CONFIRMS our GR negatives: typewell (model-based) matching is the imprecise method per the sponsor.
- Top ~5 ft = proprietary well-executed STRAT-BASED SELF-CORRELATION (horizontal-vs-self/nearby in TVT scale).
- Untried clean variant: build GR_known(TVT) from the well's OWN known zone (high-res), align each eval-GR
  window against THAT (not typewell) → read TVT directly. (Our agent only did HMM-path NCC refinement, −0.016.)
