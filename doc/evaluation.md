# Evaluation metrics

This document explains the metrics that appear in
`outputs/eval_*/summary.md` and in the head-to-head bar plots produced by
`scripts/eval_mamba3_attn_vs_da3.py`. Source code:
[`src/mamba3_attn/eval/metrics.py`](../src/mamba3_attn/eval/metrics.py).

Two families:

1. **Depth metrics** — standard monocular-depth evaluation on valid GT
   pixels. Comparable to the numbers reported by MiDaS, ZoeDepth, DA3.
2. **Representation metrics** — backbone feature quality without GT.
   They catch token collapse and reward 3D-consistent features.

---

## 1. Median alignment

Relative-depth models (DPT, DA3) are **scale-ambiguous**: the output is
only defined up to a global multiplicative factor. Before scoring, we
scale the prediction so its median matches the GT median:

```
pred := pred * (median(gt_valid) / median(pred_valid))
```

This is the MiDaS / DA3 convention. Implementation:
[`align_scale_median`](../src/mamba3_attn/eval/metrics.py). Without it, a
perfectly shaped prediction at the wrong scale would score near-zero on
every metric.

**Consequence:** a constant-depth prediction with the right median will
score reasonably on `|relative_depth_error|`. Median alignment tells you
about *shape*, not *absolute depth* — cross-check with `rmse` and the
representation metrics.

---

## 2. Depth metrics

All four are computed over **valid GT pixels only** (ETH3D provides a
per-pixel `valid_mask`; non-Lambertian, sky, and missing-LIDAR pixels
are excluded). Python identifier in parentheses.

### `|relative_depth_error|` (`abs_relative_depth_error`) ↓

Mean absolute relative error — the headline monocular-depth metric.

```
|relative_depth_error| = (1/N) · Σ  |pred_i − gt_i| / gt_i
```

- Divides by GT to make the error scale-invariant (a 10 cm error on a
  1 m subject matters more than on a 10 m subject).
- Typical values on ETH3D: DA3-SMALL ≈ 0.04, CM22@1000 ≈ 0.053.
- **Our primary acceptance metric** — the §9 gate requires ≤ 0.073.
- In Python the function name is `abs_relative_depth_error`
  (since `|...|` is not a valid identifier); the *display* form
  `|relative_depth_error|` appears in plot labels, markdown summaries,
  and dict keys.

### `delta<1.25` ↑

Threshold accuracy — the fraction of pixels where prediction and GT
agree within a factor of 1.25.

```
delta<1.25 = fraction of pixels with  max(pred/gt, gt/pred) < 1.25
```

- Symmetric in pred / gt by construction (uses `max(a/b, b/a)`).
- Complements `|relative_depth_error|`: absolute error punishes outliers;
  δ<1.25 measures the *bulk* of the distribution.
- `delta<1.25^2` is the same test with threshold 1.5625 — a looser bar
  that catches gross errors.
- Typical values on ETH3D: DA3-SMALL ≈ 0.974, CM22@1000 ≈ 0.997
  (SSM-3D actually *beats* DA3 on this metric).

### `rmse` ↓

Root-mean-squared error in **metres** (post-alignment).

```
rmse = √( (1/N) · Σ (pred_i − gt_i)² )
```

- Absolute (not ratio-based), so it's comparable across images only when
  they have similar scene depth. Weights outliers quadratically.
- Typical values: 0.10–0.15 m on ETH3D at 504 resolution.

### `log10` ↓

Mean absolute log₁₀ error.

```
log10 = (1/N) · Σ  |log₁₀(pred_i) − log₁₀(gt_i)|
```

- Scale-invariant *and* symmetric around equality (same penalty for 2×
  over- and 0.5× under-prediction).
- Small values even when `|relative_depth_error|` is large, because log
  compresses the range — treat it as a secondary check, not a headline.

---

## 3. Representation metrics (no GT required)

These probe the **backbone features** directly and ship regardless of
whether a depth head is attached. They are how we caught the token-
collapse failure modes during Phase B.

### `feat_cos_mean` ↓

Mean pairwise cosine similarity between the N patch tokens of a single
image.

```
feat_cos_mean(feats)  where feats: (N, C)
  = mean of the off-diagonal of normalize(feats) @ normalize(feats).T
```

- A healthy backbone gives tokens pointing in different directions —
  expect values near 0. Values **> 0.7** mean literal collapse (all
  tokens look alike → no spatial information).
- Implementation:
  [`feat_cos_mean`](../src/mamba3_attn/eval/metrics.py).

### `effective_rank` ↑

`exp(entropy(singular-value distribution))` — a continuous,
energy-weighted soft rank of the **per-image patch-token cloud**.

#### Input

The metric is computed **once per image**, with input `feats` of shape
`(T, C)`:

- `T` = number of patch tokens in that image, e.g. `36 × 36 = 1296` at
  `img_size=504, patch_size=14`.
- `C` = channel dimension, e.g. `384` for SSM-3D / DINOv2-S, `768` for
  DA3 (concat of two 384-dim streams).

So each row of `feats` is one patch's feature vector; each column is
one channel's activation across all patches. The reported number is
the **mean over the N images** in the eval set.

A single 384-D patch vector trivially has rank 1 — the interesting
question is *collective*: viewing the T patches as a point cloud in
`C`-D space, **how many independent directions does the cloud spread
along?**

#### Definition

```python
effective_rank(feats):                    # feats: (T, C) per image
    f = feats - feats.mean(axis=0)        # subtract per-channel mean
                                          # across T tokens (rank-1 DC offset removal)
    s = svd(f).singular_values            # (min(T, C),) singular values, descending
    p = s / sum(s)                        # normalize spectrum to a probability distribution:
                                          # p_i = share of total spectral energy along
                                          # the i-th principal direction
    H = -sum(p_i * log(p_i))              # Shannon entropy of that distribution
    return exp(H)                         # exp(H) = "effective number of equiprobable modes"
```

Implementation:
[`effective_rank`](../src/mamba3_attn/eval/metrics.py).

#### Properties

- **Range:** `[1, min(T, C)]`. With `T = 1296` and `C = 384`, the
  binding cap is `C`.
- **Mean-centering:** removes the rank-1 DC offset along the
  per-channel mean direction, so the metric measures the rank of the
  *variation* of the cloud, not the rank of a constant bias plus
  variation.
- **Limits:**
  - All `σ_i` equal (cloud uniformly fills the subspace) →
    `H = log K` → `effective_rank = K`. Maximum.
  - One `σ_1` dominates (rank-1 collapse) → `H ≈ 0` →
    `effective_rank ≈ 1`. Minimum.
  - 72 modes carrying ~equal energy + 312 nearly-zero modes →
    `effective_rank ≈ 72`.
- **Why entropy of the spectrum, not "count `σ_i > threshold`":**
  numerical rank is discrete and threshold-sensitive; many small
  singular values exist in trained nets due to noise. `exp(H(p))` is
  the **information-theoretic effective dimensionality** of the
  spectrum — continuous, energy-weighted, and matches "soft rank"
  intuition.
- **Scale-invariant:** multiplying `feats` by a constant leaves the
  ratios `p_i` unchanged.

#### Why we track it

Catches *low-rank collapse* that `feat_cos_mean` misses: tokens can
point in different directions (low cosine similarity) but still span
only a 2–3-dim subspace. For 3D reconstruction we want the model to
encode many independent scene attributes (geometry, texture,
viewpoint cues, semantics) per token, so a high effective_rank is a
**capacity metric** for the representation.

Acceptance gate: ≥ 150 is the stretch target. As of CM26 the SSM-3D
student sits at ~72 on ETH3D `terrains` while DA3-SMALL hits ~187
on the same images — i.e. the rank ceiling is not the teacher; the
bottleneck is inside the student. See `doc/PLAN.md §15.23–§15.24` for
the diagnostic ladder.

### `cross_view_nn_agreement` ↑

Fraction of patches whose best feature-NN across views lies near the
GT-warped pixel. Measures whether matching 3D points actually match in
feature space — the key check for 3D-consistent representations.

Pipeline (per image pair A, B):

1. Back-project each A-patch centre to camera-A 3D using GT depth.
2. Transform into camera-B, project to pixel coords: `uv_gt`.
3. For every A patch, find the B patch with max feature cosine
   similarity: `uv_nn`.
4. Report `fraction where ‖uv_nn − uv_gt‖ < radius_px`.

- Score in `[0, 1]`; higher = features track the underlying 3D structure
  across viewpoints.
- Implementation:
  [`cross_view_nn_agreement`](../src/mamba3_attn/eval/metrics.py).

---

## 4. Acceptance gates (PLAN §9)

A candidate modification is **kept** if it improves on the primary
metric by ≥ 2 % AND passes every gate below; otherwise it is reverted
(PLAN §13.5).

| Metric | Gate | Direction |
|---|---|---|
| `\|relative_depth_error\|` | ≤ 0.073 | lower is better |
| `delta<1.25` | ≥ 0.93 | higher is better |
| `rmse` | ≤ 0.15 | lower is better |
| `log10` | ≤ 0.035 | lower is better |
| `effective_rank` | ≥ 150 | higher is better |
| `feat_cos_mean` | ≤ 0.3 | lower is better |

The `|relative_depth_error|` threshold is the DA3-SMALL number on
ETH3D `terrains` scaled by the §9 allowance (≈ 1.75 ×). See
`doc/PLAN.md §9` for the full derivation.

---

## 5. Head-to-head snapshot (CM22@1000)

Numbers reproduced from `outputs/eval_cm22_1000/summary.md`:

| Metric | DA3-SMALL | SSM-3D (CM22@1000) | gap |
|---|---|---|---|
| `\|relative_depth_error\|` | 0.0417 | **0.0531** | 1.27× |
| `delta<1.25` | 0.9743 | **0.9972** | SSM-3D wins |
| `rmse` (m) | 0.0854 | 0.1012 | 1.19× |
| `log10` | 0.0189 | 0.0229 | 1.21× |
| `effective_rank` | 145 | 69.48 | 2.09× (SSM-3D worse) |
| `feat_cos_mean` | 0.18 | 0.46 | SSM-3D higher |

4 of 6 gates pass; the two that fail (`effective_rank`,
`feat_cos_mean`) tell us the Mamba backbone produces *less rich* but
*still geometrically usable* features at this parameter budget — depth
accuracy holds despite rank loss. See `doc/PLAN.md §15.13` for analysis.
