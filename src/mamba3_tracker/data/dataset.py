"""PyTorch Dataset wrappers for TAPVid-3D training and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tapvid3d import SUBSETS, list_clips, load_clip, peek_clip_F


@dataclass
class TrackingBatch:
    """One training batch. Variable N_q is padded to N_q_max with `query_mask`."""
    images: torch.Tensor          # (B, F, 3, H, W)
    queries_xyt: torch.Tensor     # (B, N_q_max, 3)
    tracks_XYZ: torch.Tensor      # (B, F, N_q_max, 3)
    visibility: torch.Tensor      # (B, F, N_q_max) bool
    query_mask: torch.Tensor      # (B, N_q_max) bool — True where the query slot is real
    K: torch.Tensor               # (B, 3, 3)
    clip_ids: list[str]
    subsets: list[str]
    depth: torch.Tensor | None = None  # v31: (B, F, Hd, Wd) cached DA3 metric depth


class TAPVid3DDataset(Dataset):
    """Yields fixed-length temporal windows sampled from full clips.

    For training: window length F (e.g. 24), random start within the clip,
    photometric augmentation only (no geometric augmentation — would
    invalidate `tracks_XYZ`).

    For evaluation: set `window_size=None` and the full clip is returned.
    """

    def __init__(
        self,
        clip_paths: Sequence[Path],
        window_size: int | None = 24,
        seed: int = 0,
        max_queries: int = 512,
        augment: bool = False,
        image_size: int = 448,
        da3_depth_root: str | Path | None = None,
    ) -> None:
        self.clip_paths = list(clip_paths)
        self.window_size = window_size
        self.max_queries = max_queries
        self.augment = augment
        self.image_size = image_size
        self.da3_depth_root = (
            Path(da3_depth_root).expanduser() if da3_depth_root is not None else None
        )
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int) -> dict:
        # Pick the window first, then decode only those JPEG frames. Decoding
        # the whole clip up-front and slicing afterwards burns ~566 MB per
        # drivetrack __getitem__ call (1280×1920×3 float32 × 24 frames) and
        # leaks into PyTorch's caching allocator when persistent_workers=True,
        # which is what killed v7 via systemd-oomd at ~step 200.
        path = self.clip_paths[idx]
        F_total = peek_clip_F(path)
        if self.window_size is not None and self.window_size < F_total:
            start = self._rng.randint(0, F_total - self.window_size)
            end = start + self.window_size
        else:
            start, end = 0, F_total

        clip = load_clip(path, frames=(start, end))
        images = clip.images
        tracks = clip.tracks_XYZ
        vis = clip.visibility
        H_orig, W_orig = int(clip.H), int(clip.W)

        # Resize every clip to a common (image_size, image_size) so collate
        # can stack across subsets with different native resolutions
        # (drivetrack 1280×1920, adt 512×512, pstudio 360×640). 3D GT tracks
        # are in camera-frame metres and are unaffected by image resize.
        # Query (x, y) coords ARE in pixel space, so they must be scaled to
        # match the resized image (the propagator's bilinear-sample step in
        # §8.3 of doc/attention/mamba3_attention.tex uses these to seed
        # each track's initial bank slot).
        sx = self.image_size / float(W_orig)
        sy = self.image_size / float(H_orig)
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = torch.nn.functional.interpolate(
                images, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

        if self.augment:
            images = _photometric_aug(images, self._rng)

        # Subselect queries if there are more than `max_queries`.
        N_q = clip.N_q
        if N_q > self.max_queries:
            picked = sorted(self._rng.sample(range(N_q), self.max_queries))
            queries = clip.queries_xyt[picked]
            tracks = tracks[:, picked]
            vis = vis[:, picked]
            N_q = self.max_queries
        else:
            queries = clip.queries_xyt

        # Shift query frame indices to the window's reference; only keep
        # queries whose anchor frame falls inside [start, end). If the cropped
        # window contains zero anchors, re-cast every query's anchor to
        # frame 0 of the window so we still have GT supervision.
        keep = (queries[:, 2].long() >= start) & (queries[:, 2].long() < end)
        if keep.sum().item() == 0:
            queries = queries.clone()
            queries[:, 2] = float(start)
        else:
            queries = queries[keep].clone()
            tracks = tracks[:, keep]
            vis = vis[:, keep]
        queries[:, 2] -= start
        # Scale (x, y) into the resized image's pixel coords.
        queries[:, 0] *= sx
        queries[:, 1] *= sy

        # Scale K to the resized-image pixel coords too, so the 2D
        # reprojection loss can compare predicted/GT pixel coords directly
        # (v6 addition; see src/mamba3_tracker/train/loss.py).
        K = clip.K.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        item = {
            "images": images,
            "queries_xyt": queries,
            "tracks_XYZ": tracks,
            "visibility": vis,
            "K": K,
            "clip_id": clip.clip_id,
            "subset": clip.subset,
        }
        if self.da3_depth_root is not None:
            depth_path = self.da3_depth_root / clip.subset / (clip.clip_id + ".npz")
            with np.load(depth_path) as dd:
                if "depth_q" in dd:
                    q = np.asarray(dd["depth_q"][start:end])             # (F, Hd, Wd) uint16
                    d_min = float(dd["d_min"])
                    d_max = float(dd["d_max"])
                    scale = max(d_max - d_min, 1e-6)
                    depth_window = torch.from_numpy(
                        d_min + q.astype(np.float32) * (scale / 65535.0)
                    )                                                    # (F, Hd, Wd) float32
                else:                                                    # legacy float32 cache
                    depth_window = torch.from_numpy(
                        np.asarray(dd["depth"][start:end])
                    ).float()
            item["depth"] = depth_window
        return item


def _photometric_aug(images: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Same per-frame color jitter applied across the whole window."""
    brightness = rng.uniform(0.7, 1.3)
    contrast = rng.uniform(0.7, 1.3)
    out = images * brightness
    mean = out.mean(dim=(-1, -2, -3), keepdim=True)
    out = (out - mean) * contrast + mean
    return out.clamp_(0.0, 1.0)


def collate_tracking(items: list[dict]) -> TrackingBatch:
    """Pad variable-N_q clips into a single batch."""
    B = len(items)
    Nmax = max(it["queries_xyt"].shape[0] for it in items)
    F, _, H, W = items[0]["images"].shape

    images = torch.stack([it["images"] for it in items], dim=0)
    K = torch.stack([it["K"] for it in items], dim=0)

    queries = torch.zeros(B, Nmax, 3, dtype=torch.float32)
    tracks = torch.zeros(B, F, Nmax, 3, dtype=torch.float32)
    vis = torch.zeros(B, F, Nmax, dtype=torch.bool)
    qmask = torch.zeros(B, Nmax, dtype=torch.bool)
    for b, it in enumerate(items):
        n = it["queries_xyt"].shape[0]
        queries[b, :n] = it["queries_xyt"]
        tracks[b, :, :n] = it["tracks_XYZ"]
        vis[b, :, :n] = it["visibility"]
        qmask[b, :n] = True

    depth = None
    if "depth" in items[0]:
        depth = torch.stack([it["depth"] for it in items], dim=0)

    return TrackingBatch(
        images=images,
        queries_xyt=queries,
        tracks_XYZ=tracks,
        visibility=vis,
        query_mask=qmask,
        K=K,
        clip_ids=[it["clip_id"] for it in items],
        subsets=[it["subset"] for it in items],
        depth=depth,
    )


def split_clips(
    clip_paths: Sequence[Path],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Random 90/10 train/val split, stratified by subset.

    Kept for backward compatibility with older configs that pre-date the
    official-minival split. Newer runs (v12+) should use `minival_split`
    instead, which uses the canonical TAPVid-3D minival files.
    """
    rng = random.Random(seed)
    by_subset: dict[str, list[Path]] = {s: [] for s in SUBSETS}
    for p in clip_paths:
        for s in SUBSETS:
            if s in p.parts:
                by_subset[s].append(p)
                break

    train: list[Path] = []
    val: list[Path] = []
    for sub_clips in by_subset.values():
        shuffled = sub_clips[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_frac)))
        val += shuffled[:n_val]
        train += shuffled[n_val:]
    return train, val


# Official TAPVid-3D minival file lists, copied verbatim from
#   https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/splits/tapvid3d_splits.py
# Each subset has exactly 50 clips. The published baselines in
# `configs/tapvid3d_baselines.yaml` are measured on these files.
MINIVAL_FILES: dict[str, list[str]] = {
    "pstudio": [
        "basketball_5.npz", "softball_25.npz", "boxes_22.npz", "boxes_19.npz",
        "juggle_8.npz", "boxes_12.npz", "boxes_6.npz", "basketball_29.npz",
        "tennis_28.npz", "tennis_22.npz", "basketball_9.npz", "basketball_24.npz",
        "football_3.npz", "tennis_17.npz", "softball_21.npz", "tennis_23.npz",
        "juggle_5.npz", "football_1.npz", "tennis_5.npz", "basketball_6.npz",
        "basketball_14.npz", "football_21.npz", "football_19.npz", "basketball_4.npz",
        "basketball_3.npz", "softball_2.npz", "boxes_11.npz", "juggle_4.npz",
        "softball_23.npz", "juggle_7.npz", "football_16.npz", "boxes_29.npz",
        "boxes_7.npz", "juggle_9.npz", "boxes_17.npz", "juggle_22.npz",
        "football_29.npz", "football_22.npz", "boxes_28.npz", "tennis_2.npz",
        "softball_9.npz", "basketball_13.npz", "tennis_4.npz", "football_7.npz",
        "softball_19.npz", "basketball_20.npz", "tennis_26.npz", "softball_14.npz",
        "boxes_5.npz", "boxes_27.npz",
    ],
    "drivetrack": [
        "tapvid3d_9142545919543484617_86_000_106_000_2_5AKc-TYQochsSWXpv376cA.npz",
        "tapvid3d_2681180680221317256_1144_000_1164_000_2_erT5IMWqaVRCzF6oN66E7Q.npz",
        "tapvid3d_10940952441434390507_1888_710_1908_710_2_ktWWj6EBhJZaj0IQHjSjZw.npz",
        "tapvid3d_6674547510992884047_1560_000_1580_000_2_sLjGdDksAJG7GgV0xFdh0g.npz",
        "tapvid3d_3872781118550194423_3654_670_3674_670_2_6upUeXn7HnQBNkgQ9ZomvQ.npz",
        "tapvid3d_5268267801500934740_2160_000_2180_000_2_ZU0XFHUBm0q8zqV8PBXuUQ.npz",
        "tapvid3d_10235335145367115211_5420_000_5440_000_2_4iry-kJsWTWnjLdVm-S8XA.npz",
        "tapvid3d_16105359875195888139_4420_000_4440_000_2_yqvi3P9YV-xoDyx3PZKEIw.npz",
        "tapvid3d_11967272535264406807_580_000_600_000_2_y5T1W9Gwcqsnzuc4pqX8Fw.npz",
        "tapvid3d_16331619444570993520_1020_000_1040_000_1_5KdC4474H0F_3c2pzLEG3g.npz",
        "tapvid3d_16578409328451172992_3780_000_3800_000_1_U-FL_r6V59Ml1puP3Ra7hQ.npz",
        "tapvid3d_18136695827203527782_2860_000_2880_000_2_3IPjUxMMOG3mFqNmMutzqA.npz",
        "tapvid3d_15062351272945542584_5921_360_5941_360_1_eiZ0zt164wCjt9lgslNFyg.npz",
        "tapvid3d_17993467596234560701_4940_000_4960_000_1_XbksitbrlYR9_DvWA-JKqg.npz",
        "tapvid3d_1022527355599519580_4866_960_4886_960_2_L58RM2TH_i-3sYbjr6JjQQ.npz",
        "tapvid3d_6771922013310347577_4249_290_4269_290_1_CQ9kW2zcKGsyhWnMn71VFw.npz",
        "tapvid3d_5459113827443493510_380_000_400_000_2_auYuu4nr89m2SJAXhx5csQ.npz",
        "tapvid3d_11940460932056521663_1760_000_1780_000_1_VvP3Pijy57rez0dxOvGajQ.npz",
        "tapvid3d_4967385055468388261_720_000_740_000_3_8I6i37GamzdjGA0aVXFyrw.npz",
        "tapvid3d_2475623575993725245_400_000_420_000_1_r_j-Erk4psYIRNnGvSSJzg.npz",
        "tapvid3d_6638427309837298695_220_000_240_000_1_SbPukwTCEiap4DzMUklocw.npz",
        "tapvid3d_5495302100265783181_80_000_100_000_1_Hb9oGoCRgYCdPA_DtouLKQ.npz",
        "tapvid3d_15578655130939579324_620_000_640_000_2_LcTHEaU9_O9lx0ktOeLrBg.npz",
        "tapvid3d_17386176497741125938_2180_000_2200_000_1_arpqjSyz9GYDTR9Eu7VQdw.npz",
        "tapvid3d_2656110181316327570_940_000_960_000_1__rolS4YsRwsEQqHw6BNVTg.npz",
        "tapvid3d_2681180680221317256_1144_000_1164_000_3_jnPMSEK1d86HLP-6zeEplQ.npz",
        "tapvid3d_13862220583747475906_1260_000_1280_000_2_-qclBYH8q_I-LnwAnxzzjA.npz",
        "tapvid3d_14250544550818363063_880_000_900_000_2_VDacMbTOddDXFtRVMF_m3w.npz",
        "tapvid3d_16608525782988721413_100_000_120_000_1_EpLEMxIlwLHsikWvlfy54Q.npz",
        "tapvid3d_3908622028474148527_3480_000_3500_000_1_-Jt1JVOOCOUNrj7A7zglBw.npz",
        "tapvid3d_2863984611797967753_3200_000_3220_000_1_4zC7IlkORDhkPdNS3V286Q.npz",
        "tapvid3d_2899357195020129288_3723_163_3743_163_1_ywJTCE1V98xZNGnNmY37Zg.npz",
        "tapvid3d_11940460932056521663_1760_000_1780_000_1_mvawqieCtHi2txLJCl5qEA.npz",
        "tapvid3d_5459113827443493510_380_000_400_000_1_bERYokPcM5jevThAHCLKsA.npz",
        "tapvid3d_10876852935525353526_1640_000_1660_000_2_H7txZoV9099pUCGqWjv2mQ.npz",
        "tapvid3d_6638427309837298695_220_000_240_000_2_F9t5HJfGP-79OPf7Us8eew.npz",
        "tapvid3d_14133920963894906769_1480_000_1500_000_3_418OdBqhvhfpsZyTBEPhPg.npz",
        "tapvid3d_12956664801249730713_2840_000_2860_000_2_dtmqCN3-JhK9Un7gCqs3Ug.npz",
        "tapvid3d_33101359476901423_6720_910_6740_910_3_JNFn13if2djGYt0wQ682cw.npz",
        "tapvid3d_17674974223808194792_8787_692_8807_692_2_rCRiNOIt-bOV910-j9YzxQ.npz",
        "tapvid3d_16042886962142359737_1060_000_1080_000_1_-UuWM0RiGK5o-uBrPllMrQ.npz",
        "tapvid3d_13207915841618107559_2980_000_3000_000_1_ZF1-vGaHNjltsRoVWjqYiQ.npz",
        "tapvid3d_15365821471737026848_1160_000_1180_000_1_Q5Yo2qc51Rcbvkdu37A3sQ.npz",
        "tapvid3d_7089765864827567005_1020_000_1040_000_3_qXPrSpe3gRXqM88J3-2zlw.npz",
        "tapvid3d_13862220583747475906_1260_000_1280_000_1_AvO7llz46ToGLClAYUV7yA.npz",
        "tapvid3d_8582923946352460474_2360_000_2380_000_2_zRIkIIu-FqXPx1B25304Gg.npz",
        "tapvid3d_14018515129165961775_483_260_503_260_3_AkhfkQrkuIYeFOVdMEji4A.npz",
        "tapvid3d_16105359875195888139_4420_000_4440_000_2_q-uok16avK_6CqAlNECcDw.npz",
        "tapvid3d_13862220583747475906_1260_000_1280_000_1_0igcubPJdfXj-jTWT0hKag.npz",
        "tapvid3d_6038200663843287458_283_000_303_000_2_jasa6rpTGZZo6fyug-U9jA.npz",
    ],
    "adt": [
        "Lite_release_recognition_GreenDecorationTall_seq031_6.npz",
        "Apartment_release_meal_seq136_8.npz",
        "Lite_release_recognition_WoodenBowl_seq032_1.npz",
        "Apartment_release_work_seq108_5.npz", "Apartment_release_decoration_seq138_6.npz",
        "Apartment_release_multiskeleton_party_seq122_5.npz",
        "Apartment_release_work_skeleton_seq138_5.npz",
        "Apartment_release_decoration_seq138_5.npz",
        "Apartment_release_clean_seq145_7.npz",
        "Apartment_release_meal_seq139_3.npz",
        "Apartment_release_multiuser_meal_seq134_0.npz",
        "Apartment_release_meal_seq147_3.npz",
        "Apartment_release_clean_seq140_3.npz",
        "Apartment_release_work_skeleton_seq136_2.npz",
        "Apartment_release_meal_seq133_5.npz",
        "Lite_release_recognition_BlackCeramicBowl_seq033_3.npz",
        "Apartment_release_work_seq106_2.npz",
        "Apartment_release_decoration_seq133_6.npz",
        "Apartment_release_work_skeleton_seq138_3.npz",
        "Apartment_release_decoration_seq135_7.npz",
        "Lite_release_recognition_WoodenFork_seq032_1.npz",
        "Apartment_release_meal_seq136_2.npz",
        "Apartment_release_work_seq107_4.npz",
        "Apartment_release_work_seq109_7.npz",
        "Lite_release_recognition_WoodenBoxSmall_seq031_2.npz",
        "Apartment_release_multiuser_clean_seq119_4.npz",
        "Lite_release_recognition_Flask_seq033_2.npz",
        "Lite_release_recognition_BookDeepLearning_seq032_5.npz",
        "Lite_release_recognition_DinoToy_seq031_3.npz",
        "Apartment_release_multiuser_clean_seq114_0.npz",
        "Apartment_release_multiuser_cook_seq144_4.npz",
        "Apartment_release_multiuser_party_seq134_7.npz",
        "Apartment_release_multiuser_party_seq133_3.npz",
        "Lite_release_recognition_BookDeepLearning_seq032_1.npz",
        "Apartment_release_multiskeleton_party_seq121_2.npz",
        "Lite_release_recognition_BirdHouseToy_seq030_6.npz",
        "Apartment_release_multiskeleton_party_seq121_1.npz",
        "Apartment_release_multiuser_clean_seq120_0.npz",
        "Apartment_release_multiuser_cook_seq115_8.npz",
        "Apartment_release_multiskeleton_party_seq126_8.npz",
        "Apartment_release_multiskeleton_party_seq117_4.npz",
        "Apartment_release_multiuser_cook_seq118_8.npz",
        "Apartment_release_decoration_skeleton_seq134_4.npz",
        "Apartment_release_meal_seq132_7.npz",
        "Apartment_release_multiuser_cook_seq117_4.npz",
        "Lite_release_recognition_WoodenBoxSmall_seq033_1.npz",
        "Apartment_release_multiuser_meal_seq139_5.npz",
        "Apartment_release_work_seq140_1.npz",
        "Apartment_release_multiuser_cook_seq111_7.npz",
        "Apartment_release_meal_seq138_4.npz",
    ],
}


def minival_split(
    data_root: str | Path = "~/data",
    subsets: Iterable[str] = SUBSETS,
    n_train: int = 40,
    n_val: int = 5,
    n_test: int = 5,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Stratified split of the TAPVid-3D minival into (train, val, test).

    Per subset, the 50 minival files are shuffled with `seed` then partitioned
    `n_train / n_val / n_test`. Defaults give 40/5/5 per subset = 120/15/15
    total across pstudio + drivetrack + adt. The split is deterministic.

    Raises if any minival file is missing on disk under `data_root/tapvid3d/<subset>/`.
    """
    if n_train + n_val + n_test > 50:
        raise ValueError(f"n_train + n_val + n_test = {n_train + n_val + n_test} > 50 per subset")
    root = Path(data_root).expanduser()
    if root.name != "tapvid3d":
        root = root / "tapvid3d"
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []
    for sub in subsets:
        if sub not in MINIVAL_FILES:
            raise ValueError(f"Subset {sub!r} not in MINIVAL_FILES")
        names = MINIVAL_FILES[sub][:]
        rng.shuffle(names)
        paths = [root / sub / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{sub}: {len(missing)} minival files missing under {root / sub}: "
                f"first missing = {missing[0].name}"
            )
        train += paths[:n_train]
        val   += paths[n_train : n_train + n_val]
        test  += paths[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def default_train_val(
    data_root: str | Path = "~/data",
    subsets: Iterable[str] = SUBSETS,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Backward-compat: returns (train, val) only. New code should call
    `minival_split` which also returns a held-out test set.

    Pre-v12 default — random 90/10 across whatever's on disk. Kept for older
    configs/checkpoints. v12+ configs explicitly call `minival_split`.
    """
    return split_clips(list_clips(data_root, subsets), val_frac=val_frac, seed=seed)


def filter_to_split(clips: list[Path], split: str) -> list[Path]:
    """Filter a clip-path list to a named TAPVid-3D split.

    split = "all"       → unchanged
    split = "minival"   → keep only files in MINIVAL_FILES (150 clips)
    split = "full_eval" → keep only files in FULL_EVAL_FILES (4419 clips)

    Clip filenames are unique across subsets (pstudio=basketball_*, drivetrack=
    tapvid3d_*, adt=Apartment_*/Lite_*), so a single flattened name-set suffices.
    """
    if split == "all":
        return clips
    from .tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES
    table = MINIVAL_FILES if split == "minival" else FULL_EVAL_FILES
    allow: set[str] = set()
    for names in table.values():
        allow.update(names)
    return [p for p in clips if p.name in allow]


def official_train_test_split(
    data_root: str | Path = "~/data",
    subsets: Iterable[str] = SUBSETS,
) -> tuple[list[Path], list[Path]]:
    """v19+ split following the official TAPVid-3D protocol (option A1).

    TAPVid-3D ships two named splits — MINIVAL_FILES (150 dev clips) and
    FULL_EVAL_FILES (4419 eval clips). They are DISJOINT (verified via
    the vendored splits file). The benchmark is officially eval-only, so
    published baselines train on external data (Kubric etc.). Lacking
    external data, we treat:

        train_clips  ←  FULL_EVAL_FILES   (4419 clips: 1906 adt + 2407 drivetrack + 106 pstudio)
        test_clips   ←  MINIVAL_FILES     ( 150 clips: 50 each)

    No leakage between train and test — they're disjoint by construction.
    Our reported numbers compare to the TAPVid-3D paper's Table 4 (minival)
    baseline column.

    `data_root` should point at `~/data` (default); the function appends
    `/tapvid3d/<subset>/` internally. Files not present on disk are silently
    skipped (no error) — useful for incremental downloads. The caller can
    check `len(train_clips)` against the expected totals to detect partial
    downloads.

    Returns:
        (train_clips, test_clips) — sorted lists of .npz paths.
    """
    from .tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES
    root = Path(data_root).expanduser()
    if root.name != "tapvid3d":
        root = root / "tapvid3d"
    train: list[Path] = []
    test: list[Path] = []
    for sub in subsets:
        if sub not in FULL_EVAL_FILES or sub not in MINIVAL_FILES:
            raise ValueError(f"Subset {sub!r} not in official TAPVid-3D splits")
        train += sorted(p for p in (root / sub / n for n in FULL_EVAL_FILES[sub]) if p.exists())
        test  += sorted(p for p in (root / sub / n for n in MINIVAL_FILES[sub])   if p.exists())
    return train, test
