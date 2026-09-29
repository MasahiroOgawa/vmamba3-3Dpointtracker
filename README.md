# vmamba3-3Dpointtracker

[![arXiv](https://img.shields.io/badge/arXiv-2609.34035-b31b1b.svg)](https://arxiv.org/abs/2609.34035)

Metric 3D point tracking with a Mamba-3 state space model, on a single commodity GPU, monocular
and pose-free.

This is the official code of the paper
**[3D Point Tracking with State Space Models](https://arxiv.org/abs/2609.34035)**
(M. Ogawa, Q. An and A. Yamashita, arXiv:2609.34035, 2026).

## 1. What is this repository

Once a point's 2D image trajectory is fixed, what decides its metric 3D accuracy is the depth
along its pixel ray. So instead of learning tracking end to end, the tracker composes two frozen
front-ends and learns only what they cannot supply:

| Stage | Component | Trained? |
|---|---|---|
| 2D correspondence | [WAFT](https://github.com/princeton-vl/WAFT) dense optical flow, chained frame to frame into 2D tracks | frozen |
| Depth along the ray | [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) metric depth (`da3metric-large`, "DA3-l") | frozen |
| 3D refiner | a compact Mamba-3 model (the [visionMamba3](https://github.com/MasahiroOgawa/visionMamba3) VSSD-2pool mixer) conditioned on DINOv3 appearance features, correcting each track's pixel and depth | **trained** |
| Visibility | a flow-only visibility head | **trained** |

A state space model summarises a track in a fixed-size recurrent state, so its memory cost is
constant in the number of frames, whereas attention needs a key-value cache that grows with them.
That is what keeps the tracker within a single-GPU budget.

On the TAPVid-3D minival benchmark, the best configuration (WAFT + DA3-l + vmamba3-2pool with the
flow-only visibility head) reaches a mean **absolute metric Average Jaccard of 0.256**, the highest
among the methods evaluated under identical conditions. See the paper for the full comparison.

## 2. Structure of this repository

```
vmamba3-3Dpointtracker/
├── src/
│   ├── mamba3_tracker/            # the tracker
│   │   ├── model/                 #   refiners (depth_refined_tracker.py: Mamba3V35Refiner is the
│   │   │                          #   paper's model), flow_vis_head.py, dino_encoder.py, ...
│   │   ├── data/                  #   TAPVid-3D loader, official splits, depth-source registry
│   │   ├── train/                 #   config loader, losses, LR schedule
│   │   ├── eval/                  #   TAPVid-3D metrics (vendored official implementation)
│   │   └── viz/                   #   3D-track figures and videos
│   └── searaft_flow/              # flow chaining into 2D tracks (+ SEA-RAFT adapter)
├── configs/                       # one YAML per experiment: v<N>.yaml in chronological order
│                                  #   (v64 = the paper's model, v94 = its visibility head)
├── scripts/                       # download, cache, train, eval, figure and queue scripts
├── tests/unit/
├── doc/vmamba3_3dpointtrack/      # technical report (LaTeX) with the full experiment record
└── third_party/                   # git submodules, pinned
    ├── visionMamba3/              #   required: the Mamba-3 operators
    ├── WAFT/                      #   required: optical flow front-end
    ├── depth-anything-3/          #   required: metric depth front-end
    └── SEA-RAFT/, SpaTrackerV2/, TrackCraft3R/, DELTA_densetrack3d/, DenseTrack3Dv2/
                                   #   optional baselines, see 5. Evaluation
```

Configs are numbered in the order the experiments were run; a config is a complete recipe, and
the config path is the only argument a training script takes. Most of them are ablations recorded
in the technical report; the ones needed to reproduce the paper are named below.

WAFT and TrackCraft3R are pinned to the author's forks of the official repositories. The WAFT
fork adds only a `pyproject.toml` so its environment can be built with uv (model code unchanged);
the TrackCraft3R fork adds memory fixes for loading its 17 GB checkpoint.

## 3. How to set up

Requirements: Linux, Python 3.11–3.12, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU with
CUDA (the paper's runs used an RTX 4080 Laptop GPU, 12 GB), and a Hugging Face account.

```bash
git clone --recursive https://github.com/MasahiroOgawa/vmamba3-3Dpointtracker.git
cd vmamba3-3Dpointtracker
# if cloned without --recursive:
git submodule update --init --recursive

uv sync                                       # the tracker's environment
uv sync --project third_party/WAFT            # WAFT's own environment (torch 2.7), see below
uv run python scripts/download_weights.py     # WAFT checkpoints (~340 MB) into third_party/WAFT/
```

- `download_weights.py` **downloads** WAFT's a1 checkpoint (from the WAFT authors' Google Drive)
  and the Depth-Anything-V2-Small checkpoint WAFT's backbone loads (from Hugging Face).
- DA3 (`depth-anything/da3metric-large`) and DINOv3 (`facebook/dinov3-vits16-pretrain-lvd1689m`)
  are fetched from the Hugging Face Hub on first use. **DINOv3 is gated**: accept its license on
  its Hugging Face page, then log in with `uv run huggingface-cli login`.
- WAFT runs inside the tracker's environment during training and live evaluation. Its own
  environment is only used to precompute the WAFT track cache for evaluation (4.4).

Check the install:

```bash
uv run pytest
```

## 4. How to use

No trained checkpoints are released yet, so reproducing the result means training the refiner and
the visibility head. All paths below are the configs' defaults.

### 4.1 Data

TAPVid-3D (Aria Digital Twin, DriveTrack and Panoptic Studio). The official split is used:
training on the 4419 `full_eval` clips and evaluation on the 150 `minival` clips
(`src/mamba3_tracker/data/tapvid3d_splits.py`).

```bash
uv run python scripts/download_tapvid3d_all.py      # ~474 GB download, ~525 GB on disk
uv run python scripts/precompute_da3_depths.py      # DA3-l depth for every clip, ~22 GB
```

- `download_tapvid3d_all.py` **downloads** the `.npz` clips from the
  [ZhengGuangze/TAPVid-3D](https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D) Hugging Face
  mirror into `~/data/tapvid3d/<subset>/`, one tarball at a time (deleted after extraction). The
  mirror is used because Google's official
  [TAPVid-3D release](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) does not
  redistribute the ADT clips; `download_tapvid3d_all.py --subsets pstudio` fetches one subset.
- `precompute_da3_depths.py` runs DA3-l over every clip into `~/data/tapvid3d_da3/`.

### 4.2 Train the refiner (v64)

```bash
source scripts/cudnn_env.sh
uv run python scripts/train_depth_refined_tracker.py --config configs/v64.yaml
```

20000 steps with WAFT running live on every batch (~76 h). The checkpoint lands in
`result/v64_waft_live_da3l_2pool/ckpt_20000.pt`.

### 4.3 Train the visibility head (v94)

The head reads only forward/backward flow at each tracked point, so that flow is cached once and
the head is trained on the cache:

```bash
uv run python scripts/cache_flow_vis.py configs/v94_cache.yaml   # -> ~/data/tapvid3d_flowvis/
uv run python scripts/train_flow_vis_head.py configs/v94.yaml    # -> result/v94/best.pt
```

### 4.4 Evaluate on TAPVid-3D minival

Evaluation reads WAFT's 2D tracks from a cache, built once in WAFT's environment:

```bash
third_party/WAFT/.venv/bin/python scripts/eval_waft.py \
    --subsets drivetrack pstudio adt --split minival \
    --scale -1 --iters 4 --image-size 896 \
    --out-dir ~/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
```

Then score the refiner with the visibility head:

```bash
uv run python scripts/eval_metric3d.py --method v35 \
    --ckpt result/v64_waft_live_da3l_2pool/ckpt_20000.pt --depth da3l --split minival \
    --waft-pred-dir ~/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4 \
    --vis-source v94 --vis-head-ckpt result/v94/best.pt \
    --flowvis-dir ~/data/tapvid3d_flowvis/minival \
    --out-dir result/v64_waft_live_da3l_2pool/eval_v94
```

`<out-dir>/metrics.json` holds the scores; `overall.metric_average_jaccard` is the number the paper
reports (0.256). `--vis-source flow` scores the same refiner with the forward-backward consistency
mask instead of the head (0.255).

### 4.5 Track your own video

```bash
uv run python scripts/track_custom_video.py my_video.mp4 --ckpt <checkpoint> --out-dir result/my_video
```

This script runs the earlier v33 variant: SEA-RAFT flow (needs the `SEA-RAFT` baseline submodule,
see 5.1) with the depth-only refiner, not the paper's v64 model.

## 5. Evaluation

Re-running the comparison against other 3D trackers. None of this is needed to use the tracker.

### 5.1 Baselines

The baseline repositories are optional submodules, so a recursive clone skips them:

```bash
bash scripts/setup_baselines.sh                  # all of them
bash scripts/setup_baselines.sh SpaTrackerV2     # only the named ones
```

| Baseline | Submodule | Evaluation script |
|---|---|---|
| SEA-RAFT + DA3 | `third_party/SEA-RAFT` | `scripts/eval_metric3d.py --method searaft` |
| SpatialTrackerV2 | `third_party/SpaTrackerV2` | `scripts/eval_spatracker_v2.py` |
| TrackCraft3R | `third_party/TrackCraft3R` | `scripts/eval_trackcraft3r.py` |
| DELTA | `third_party/DELTA_densetrack3d` | `scripts/eval_delta.py` |
| DELTAv2 | `third_party/DenseTrack3Dv2` | `scripts/eval_deltav2.py` |

Each baseline pins its own torch/CUDA version, so each runs in **its own venv** inside its
submodule, built by following that repository's README; the version each script was run with is
in its docstring (e.g. DELTA: torch 2.2.2 / cu121). The scripts write predictions in a common
format that `eval_metric3d.py --method external` scores with the same absolute metric as the
tracker. The baseline scripts still locate their repositories and data at the author's absolute
paths (the `*_ROOT` constants at the top of each), so set those for your machine before running one.

### 5.2 Technical report

```bash
make -C doc/vmamba3_3dpointtrack          # vmamba3_3dpointtrack.pdf
```

## Citation

```bibtex
@article{ogawa2026pointtracking,
  title   = {3D Point Tracking with State Space Models},
  author  = {Ogawa, Masahiro and An, Qi and Yamashita, Atsushi},
  journal = {arXiv preprint arXiv:2609.34035},
  year    = {2026},
}
```

## License

MIT. See [LICENSE](LICENSE). Submodules and downloaded weights and data keep their own licenses.
