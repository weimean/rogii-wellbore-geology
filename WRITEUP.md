# Reading a drill bit's geology from a gamma-ray log

A case study from the ROGII Wellbore Geology competition on Kaggle. Every score below comes from the real
hidden leaderboard, and the pipeline breaks no competition rules: no target leakage, no train-only features at
inference, licensed models only.

Final result: **9.673 RMSE on the private leaderboard, rank 3116 of 6125, top 51%**, down from forward-fill's
15.91. The private shakeup moved the solution up 828 places from where it sat on the public board. That gain is
the part I am proudest of, and it is also where the real lesson lives, so this writeup spends most of its time
on the reasoning, the traps, and the things I tried that failed.

## The problem

A horizontal oil well drills sideways through rock for thousands of feet. A gamma-ray (GR) tool on the drill
string logs the natural radioactivity of the rock it passes. A separate vertical "typewell" nearby records GR
as a function of **True Vertical Thickness (TVT)**, the bit's geological depth below a marker layer. TVT is
what a geosteering geologist tracks in real time to keep the bit inside the target rock.

The competition gives you TVT along the first stretch of each well and blanks it out for a long evaluation
zone, about 74% of the well on the median. Your job is to fill in that blank. The horizontal GR is a stretched,
shifted copy of the vertical typewell GR, so recovering TVT reduces to log correlation, the automated version
of what a geologist does by eye.

One catch shaped everything that followed. The GR signal barely discriminates: its self-correlation at the true
depth sits around 0.65, and the per-step signal-to-noise runs near 0.02. Point-by-point GR matching beats
forward-fill by roughly one percent. The signal lives in the structure of the log, not its individual samples,
and most of the loss is a slow drift in the absolute level rather than the wiggle.

## The shape of the solution

The design principle held throughout: ensemble diversity has to come from a different algorithm, never a
restatement of one you already have. The pipeline pins the absolute level with a per-well anchor, tracks
trajectory-coupled motion with physics-style filters, and predicts the residual with a stack of decorrelated
learners.

- A Ridge level anchor fixes each well's absolute offset.
- A second-order Hidden Markov Model over (TVT, dip-rate) and a bank of particle filters track the motion.
- A residual stack of LightGBM, CatBoost, an MLP, the TabICL tabular foundation model, and a Temporal
  Convolutional Network predicts what the anchor and trackers miss.
- A U=TVT+Z projection and warm-up damping clean up the drift at the end.

The interesting parts were the two moments I declared victory too early, the round where I checked a
competitor's claim by rebuilding it, and what the final leaderboard reveal taught me about all of it.

## False ceiling one: feature-poverty

I got stuck at 13.5 RMSE and convinced myself the inputs were exhausted. My evidence was a stack of probes, each
adding one hand-built feature and each buying about 0.01. So I concluded the signal was gone.

The conclusion was wrong, and the error was in the method. Testing features one at a time hides the value of a
feature *family*. When I reproduced the top teams' full set of multi-offset GR-match grids together and paired
it with a strong model, the group cleared 0.27 at a stroke. A ceiling inferred from one-feature-at-a-time probes
is no ceiling. It is a measurement artifact.

## The spatial breakthrough, and a validation trap

The largest single gain, 1.85 RMSE, came from geology rather than modeling. TVT is deterministic in the
structural formation surfaces, which are train-only columns. Those surfaces are globally consistent absolute
depths, so I could impute each well's surface from its neighbours with a self-excluded KNN, which stays
test-legal. From the imputed surface, TVT falls out directly.

This exposed a trap that cost me a day. Grouped cross-validation *understated* the spatial features by about 20
times. GroupKFold strips each well's pad-neighbours into the held-out fold, so the model never sees the
neighbours it will have at test time, where the scored wells are themselves offset wells to the training set. The lesson generalized: match the validator to the knob. Grouped CV understates spatial and sequence
features, so I trusted the public leaderboard for those. It *overstates* blend-weight tuning, where I reverted
four times after CV improved and the real board regressed.

## False ceiling two: architecture-poverty

After closing seven more levers I declared 11.0 the honest ceiling. Wrong again, and this time the gap was
architectural. Every model in the stack scored each row on its own. A geologist reads the whole log at once. A
Temporal Convolutional Network, a dilated convolution over the eval-zone sequence, is the learned version of
that holistic read, and it added 0.32. Grouped CV understated it 5 times, the same signature as the spatial
features. I trusted the board again and shipped it.

I also tried a transformer as a second sequence model. It correlated with the TCN at 0.9997, restated the same
reading, and diluted the blend when averaged in. Rejected. Self-attention's global receptive field gave me
nothing the dilated convolution had not already captured.

## Reading the competitors, and checking their homework

Late in the competition, several strong public notebooks appeared claiming an "honest" score near 7.5, about 3
RMSE below mine. Chasing that would have meant weeks of work, so first I checked whether the number was real.

I read six kernels. Most of the levers they used, I already had: likelihood-weighted particle filters,
segment-offset features, GR-residual features. Two ideas were new. One, a beam-search GR aligner, scored 43
alone on my data because raw GR without a level anchor drifts to the wrong rock layer, the same 0.02
signal-to-noise problem in a new costume. The other was real: predict in **U=TVT+Z** space, the geological
formation surface, which stays smooth while TVT wiggles with the trajectory. Fit a robust low-order polynomial
there, subtract the exact trajectory Z, and the slow drift error drops out. It validated at 0.17 on my
out-of-fold predictions, transferred to 0.134 on the real board, and 76% of wells improved. That became v29, my
best at 10.57. A later feature that steers each particle filter toward its neighbours' structural dip took it to
10.42.

Then I tested the 7.5 claim by rebuilding it. I ported the one fully-reproducible public pipeline, particle
filter plus beam search plus selector plus polynomial, and ran it on my own 773 labeled wells. It scored 16 to
80, never near 7.5. The particle filter diverges on the weak GR signal, exactly as the physics predicts. The
reported 7.159 turned out to be a precomputed CSV the notebook loads in preference to running any physics. That
specific claim was an artifact, and the experiment saved me from chasing it. The mistake I made was in what I
concluded next.

## The round where I was wrong about being done

I told the reviewer the modeling was finished. He pushed back that avenues remained, and he was right. So I laid
out five I had skipped and tested each. Steering a particle filter with the neighbours' structural dip won,
moving the public score 0.14. The other four failed: a global thin-plate-spline surface the U-projection already
supplied, a GR affine calibration that let the model overfit, spatial channels fed into the data-starved TCN
that regressed the board, and a Huber loss that looked good on a single out-of-fold split and lost on the real
board.

That Huber result is worth pausing on. The gain it showed on one out-of-fold run sat inside the seed noise I had
measured, about 0.07 RMSE for a single boosted tree on this pipeline. I had the error bar in hand and shipped
the change anyway before the board corrected me. The winning dip feature cleared that noise floor by a wide
margin, which is why it transferred and Huber did not.

## What the leaderboard reveal taught me

The competition closed and the private board settled the questions the public board could not. Two things came
out of it.

First, the discipline paid off. The solution rose 828 places from public to private. A public board built on 50
wells was almost pure noise; one team's writeup measured the public-to-private rank correlation at −0.04.
Teams that tuned to that noise fell, and trusting group-aware cross-validation over it is what held up when the
real test wells arrived. The two-submission safety net mattered too. Kaggle kept my better private score across
the two best public entries, and the earlier v29 outscored the later dip-feature v32 on private by 0.05. That
last public gain had been public overfit, sitting inside the seed noise, and the safety net absorbed it.

Second, and this is the correction, my "honest ceiling" read was half wrong. When the winners posted their
solutions, the top score turned out to be honest and reachable, near 5.6 RMSE, and the gap to it was a model
class I never built. The winners treat geosteering as a 2D image-alignment problem. They render the GR-match
grid as an image, position along one axis and candidate TVT along the other, so the true path and every false
self-similar layer show up as parallel ridges. A U-Net then picks the correct ridge and reads out a distribution
over TVT. That class reaches 4.6 to 5.3 in cross-validation. A tabular gradient-boosted stack like mine
saturates near 9 to 10 no matter what features I pour in.

Catching the fake 7.5 was right. Generalizing it into "the whole gap is a leak I should not chase" was not.
Disproving one bad sub-claim did not license a verdict on the entire gap. The 1st-place solution even uses my
particle filter and my neighbour-surface feature, as input channels to its alignment network rather than as the
model itself. I had the right ingredients and built the wrong machine.

## What I would tell a hiring manager

The modeling skill here is ordinary: a boosted-tree stack, a small neural net, a convolutional sequence model.
The skill that moved the score was knowing which signal to trust and which to ignore. Grouped CV understated the
two levers that mattered most and overstated the ones that burned me, and the private shakeup rewarded reading
those error bars correctly. A competitor's headline number evaporated when I rebuilt it. My own "ceiling" was
wrong twice, and my read on the gap to the top was wrong a third time, which the reveal made plain. Getting here
took less cleverness than restraint, plus the honesty to name the one call I got wrong: run the experiment, read
the error bar, walk away from the mirage, and when a gap refuses to close, suspect a missing model class before
you blame a leak.

Full technical record: [`notes/CONCLUSION.md`](notes/CONCLUSION.md) for the recipe and the lever ledger,
[`notes/approaches/`](notes/approaches/) for per-lever narratives, and
[`notes/approaches/28-public-notebook-adopt.md`](notes/approaches/28-public-notebook-adopt.md) for the
competitor study, the ceiling reproduction, and the post-reveal breakdown of the winning solutions.
