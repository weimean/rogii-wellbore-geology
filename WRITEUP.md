# Reading a drill bit's geology from a gamma-ray log

A case study from the ROGII Wellbore Geology competition on Kaggle. The scores below all come from the
real hidden leaderboard, and the pipeline breaks no competition rules: no target leakage, no train-only
features at inference, licensed models only.

Final result: forward-fill baseline **15.91 RMSE** down to **10.57**, a 34% cut. The number matters less
than how I got there, so this writeup spends most of its time on the reasoning, the traps, and the things
I tried that failed.

## The problem

A horizontal oil well drills sideways through rock for thousands of feet. A gamma-ray (GR) tool on the
drill string logs the natural radioactivity of the rock it passes. A separate vertical "typewell" nearby
records GR as a function of **True Vertical Thickness (TVT)**, the bit's geological depth below a marker
layer. TVT is what a geosteering geologist tracks in real time to keep the bit inside the target rock.

The competition gives you TVT along the first stretch of each well and blanks it out for a long evaluation
zone, about 74% of the well on the median. Your job is to fill in that blank. Because the horizontal GR
is a stretched, shifted copy of the vertical typewell GR, recovering TVT reduces to log correlation, the
automated version of what a geologist does by eye.

One catch shaped everything that followed. The GR signal barely discriminates: its self-correlation at the
true depth sits around 0.65, and the per-step signal-to-noise runs near 0.02. Point-by-point GR matching
beats forward-fill by roughly one percent. The signal lives in the structure of the log, not its individual
samples, and most of the loss is a slow drift in the absolute level rather than the wiggle.

## The shape of the solution

The design principle held throughout: ensemble diversity has to come from a genuinely different algorithm,
never a restatement of one you already have. The pipeline pins the absolute level with a per-well anchor,
tracks trajectory-coupled motion with physics-style filters, and predicts the residual with a stack of
decorrelated learners.

- A Ridge level anchor fixes each well's absolute offset.
- A second-order Hidden Markov Model over (TVT, dip-rate) and a bank of particle filters track the motion.
- A residual stack of LightGBM, CatBoost, an MLP, the TabICL tabular foundation model, and a Temporal
  Convolutional Network predicts what the anchor and trackers miss.
- A U=TVT+Z projection and warm-up damping clean up the drift at the end.

The interesting parts were the two moments I declared victory too early, and the round where I checked a
competitor's claim by rebuilding it.

## False ceiling one: feature-poverty

I got stuck at 13.5 RMSE and convinced myself the inputs were exhausted. My evidence was a stack of probes,
each adding one hand-built feature and each buying about 0.01. So I concluded the signal was gone.

The conclusion was wrong, and the error was in the method. Testing features one at a time hides the value
of a feature *family*. When I reproduced the top teams' full set of multi-offset GR-match grids together
and paired it with a strong model, the group cleared 0.27 at a stroke. A ceiling inferred from
one-feature-at-a-time probes is not a ceiling. It is a measurement artifact.

## The spatial breakthrough, and a validation trap

The largest single gain, 1.85 RMSE, came from geology rather than modeling. TVT is deterministic in the
structural formation surfaces, which are train-only columns. Those surfaces are globally consistent
absolute depths, so I could impute each well's surface from its neighbours with a self-excluded KNN, which
stays test-legal. From the imputed surface, TVT falls out directly.

This exposed a trap that cost me a day. Grouped cross-validation *understated* the spatial features by
about 20 times. GroupKFold strips each well's pad-neighbours into the held-out fold, so the model never
sees the neighbours it will actually have at test time, where the scored wells genuinely are offset wells
to the training set. The lesson generalized: match the validator to the knob. Grouped CV understates
spatial and sequence features, so I trusted the public leaderboard for those. It *overstates* blend-weight
tuning, where I reverted four times after CV improved and the real board regressed.

## False ceiling two: architecture-poverty

After closing seven more levers I declared 11.0 the honest ceiling. Wrong again, and this time the gap was
architectural. Every model in the stack scored each row on its own. A geologist reads the whole log at once.
A Temporal Convolutional Network, a dilated convolution over the eval-zone sequence, is the learned version
of that holistic read, and it added 0.32. Grouped CV understated it 5 times, the same signature as the
spatial features. I trusted the board again and shipped it.

I also tried a transformer as a second sequence model. It correlated with the TCN at 0.9997, restated the
same reading, and diluted the blend when averaged in. Rejected. Self-attention's global receptive field
gave me nothing the dilated convolution had not already captured.

## Reading the competitors, and checking their homework

Late in the competition, several strong public notebooks appeared claiming an "honest" score near 7.5,
about 3 RMSE below mine. Chasing that would have meant weeks of work, so first I checked whether the number
was real.

I read six kernels. Most of the levers they used, I already had: likelihood-weighted particle filters,
segment-offset features, GR-residual features. Two ideas were new. One, a beam-search GR aligner, scored
43 alone on my data because raw GR without a level anchor drifts to the wrong rock layer, the same 0.02
signal-to-noise problem in a new costume. The other was real: predict in **U=TVT+Z** space, the geological
formation surface, which stays smooth while TVT wiggles with the trajectory. Fit a robust low-order
polynomial there, subtract the exact trajectory Z, and the slow drift error drops out. It validated at
0.17 on my out-of-fold predictions, transferred to 0.134 on the real board, and 76% of wells improved.
That became v29, my best at 10.57.

Then I tested the 7.5 claim by rebuilding it. I ported the one fully-reproducible public pipeline, particle
filter plus beam search plus selector plus polynomial, and ran it on my own 773 labeled wells. It scored 16
to 80, never near 7.5. The particle filter diverges on the weak GR signal, exactly as the physics predicts.
The reported 7.159 turned out to be a precomputed CSV the notebook loads in preference to running any
physics. The claimed headroom was an artifact. That one experiment saved me from chasing a mirage, and it
told me 10.57 sits near the honest reproducible ceiling.

## The round where I was wrong about being done

I told the reviewer the modeling was finished. He pushed back that avenues remained, and he was right. So I
laid out four I had skipped and tested each. Spatial dip as a feature: null, because a row-independent
gradient-boosted tree cannot use a per-row dip-rate it already holds as a surface. A global thin-plate-spline
structural surface: null, because the U-projection already supplies that smoothing. GR affine calibration:
negative, the extra columns let the model overfit. Feeding the spatial channels into the TCN: a regression
on the real board from 10.57 to 10.72, because 30 extra channels overfit a network trained on only 773
wells.

All four failed. The value was in testing them rather than assuming, and in the one that looked most
promising on paper, the spatial dip, correlating 0.48 with the true dip yet buying nothing once it met a
row-independent model.

## What made the numbers trustworthy

- **Group-aware validation.** GroupKFold on the well ID, always, so rows from one well never straddle the
  split. A leak-probe confirmed CV and the public board agree for row-independent features.
- **Error bars before conclusions.** The sequence-model sweeps ran the same config twice to measure seed
  variance, about 0.05 RMSE, larger than any config effect I was testing. That caught an earlier sweep
  whose rankings were pure CPU-contention noise.
- **Honest negatives, recorded with the measurement that killed them.** Around twenty levers were tested and
  rejected: a transformer that restated the TCN, a geometry prior that was geologically false, "raw GR" that
  lost to engineered features, a train-only feature that looked like a 0.53 CV win and turned out to be
  leakage the real board caught.
- **Reproduce a claimed ceiling before believing it.** Twice a self-reported number would have sent me
  chasing nothing.

## What I would tell a hiring manager

The modeling skill here is ordinary: a boosted-tree stack, a small neural net, a convolutional sequence
model. The skill that moved the score was knowing which signal to trust and which to ignore. Grouped CV
understated the two levers that mattered most and overstated the ones that burned me. A competitor's
headline number evaporated when I rebuilt it. My own "ceiling" was wrong twice. Getting to 10.57 took less
cleverness than restraint: run the experiment, read the error bar, and walk away from the mirage.

Full technical record: [`notes/CONCLUSION.md`](notes/CONCLUSION.md) for the recipe and the lever ledger,
[`notes/approaches/`](notes/approaches/) for per-lever narratives, and
[`notes/approaches/28-public-notebook-adopt.md`](notes/approaches/28-public-notebook-adopt.md) for the
competitor study and the ceiling reproduction.
