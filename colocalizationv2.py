#!/usr/bin/env python3
"""
SLICE 60 ANALYSIS
Main thesis figure with 2x2 overview plus 4 zoom montages below.
Each zoom montage has 3 panels in this order
GCaMP overlay, DsRed overlay, DsRed classification

Outputs
registration_overlay.png
dsred_integrity.png
<strategy>_main_with_4zooms.png
masks_<strategy>.tif
results.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec

from tifffile import imread, imwrite, TiffFile
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


class Config:
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    DSRED_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\dsred_registered_SyN.tif"
    DSRED_ORIGINAL = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"
    GCAMP_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\gcamp_registered_SyN.tif"
    FUNC_TEMPLATE = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\functional_template.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\slice60_analysis_d7_cp-6"

    PLANE_IDX = 60
    PLANE_PATTERN = "aligned_p{:d}_nan.tif"

    CELLPOSE_MODEL = "cyto3"
    CELLPOSE_FLOW_THRESHOLD = 1.0

    STRATEGIES = [
        ("mean_d7_cp-6_restore", "mean", 7, -6, True),
    ]

    DSRED_PERCENTILE = 85

    N_ZOOM_REGIONS = 4
    ZOOM_SIZE = 140
    MIN_DSRED_POS = 3

    FONT_SIZE = 20
    FIGURE_DPI = 600  # Altíssima resolução


cfg = Config()


def load_plane_frames(plane_idx: int) -> np.ndarray:
    fname = cfg.PLANE_PATTERN.format(plane_idx + 1)
    fpath = Path(cfg.FUNC_TIFS_DIR) / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")

    print(f"  Loading plane {plane_idx} from {fpath.name}")
    with TiffFile(str(fpath)) as tif:
        frames = []
        for page in tif.pages:
            frame = page.asarray().astype(np.float32)
            if not np.any(np.isnan(frame)):
                frames.append(frame)

    stack = np.array(frames, dtype=np.float32)
    print(f"    {stack.shape[0]} valid frames, shape {stack.shape[1:]}")
    return stack


def compute_projections(stack: np.ndarray) -> dict:
    return {
        "mean": np.mean(stack, axis=0).astype(np.float32),
        "median": np.median(stack, axis=0).astype(np.float32),
        "max": np.max(stack, axis=0).astype(np.float32),
    }


def normalize_for_display(img: np.ndarray, plow=0.5, phigh=99.7) -> np.ndarray:
    v = img[np.isfinite(img)]
    v = v[v > 0]
    if v.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.percentile(v, [plow, phigh])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32)


def _get_z_slice(vol: np.ndarray, z: int) -> np.ndarray:
    if vol.ndim == 3:
        return vol[z]
    if vol.ndim == 4:
        if vol.shape[0] in (2, 3, 4):
            return vol[0, z]
        return vol[z, ...]
    raise ValueError(f"Unexpected volume shape {vol.shape}")


def create_registration_overlay(func_slice: np.ndarray, anat_slice: np.ndarray, output_path: Path):
    fs = cfg.FONT_SIZE
    func_norm = normalize_for_display(func_slice)
    anat_norm = normalize_for_display(anat_slice)

    rgb = np.zeros((*func_norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = anat_norm
    rgb[..., 1] = func_norm
    rgb[..., 2] = anat_norm

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(func_norm, cmap="gray", interpolation="bilinear")
    axes[0].set_title("GCaMP functional", fontsize=fs, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(anat_norm, cmap="gray", interpolation="bilinear")
    axes[1].set_title("GCaMP anatomical registered", fontsize=fs, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.clip(rgb, 0, 1), interpolation="bilinear")
    axes[2].set_title("Overlay", fontsize=fs, fontweight="bold")
    axes[2].axis("off")

    fig.legend(
        handles=[
            Patch(facecolor="green", label="Functional"),
            Patch(facecolor="magenta", label="Anatomical"),
            Patch(facecolor="white", label="Overlap"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=fs - 2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    for ax, letter in zip(axes, ["A", "B", "C"]):
        ax.text(
            0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=fs + 4, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def create_dsred_integrity_check(dsred_original: np.ndarray, dsred_registered: np.ndarray,
                                 plane_idx: int, output_path: Path):
    orig = _get_z_slice(dsred_original, plane_idx).astype(np.float32)
    reg = _get_z_slice(dsred_registered, plane_idx).astype(np.float32)

    orig_norm = normalize_for_display(orig)
    reg_norm = normalize_for_display(reg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fs = cfg.FONT_SIZE

    axes[0].imshow(orig_norm, cmap="gray", interpolation="bilinear")
    axes[0].set_title("DsRed original", fontsize=fs, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(reg_norm, cmap="gray", interpolation="bilinear")
    axes[1].set_title("DsRed registered", fontsize=fs, fontweight="bold")
    axes[1].axis("off")

    for ax, letter in zip(axes, ["A", "B"]):
        ax.text(
            0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=fs + 4, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {output_path.name}")


def segment_with_restore(image: np.ndarray, diameter: int, cellprob: float, use_restore: bool = True) -> np.ndarray:
    if use_restore:
        from cellpose import denoise
        model = denoise.CellposeDenoiseModel(
            gpu=True,
            model_type=cfg.CELLPOSE_MODEL,
            restore_type=f"oneclick_{cfg.CELLPOSE_MODEL}",
        )
    else:
        from cellpose import models
        model = models.Cellpose(model_type=cfg.CELLPOSE_MODEL, gpu=True)

    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        cellprob_threshold=cellprob,
        flow_threshold=cfg.CELLPOSE_FLOW_THRESHOLD,
        channels=[0, 0],
    )
    return masks


def classify_dsred(dsred_slice: np.ndarray, masks: np.ndarray, percentile: float = 85):
    ds = dsred_slice.astype(np.float32)
    m = masks.astype(np.int32)

    ds_positive = ds[ds > 0]
    if ds_positive.size == 0:
        ds_positive = ds.ravel()
    bright_thr = float(np.percentile(ds_positive, percentile))

    labels_unique = np.unique(m)
    labels_unique = labels_unique[labels_unique > 0]

    flat_m = m.ravel()
    flat_ds = ds.ravel()
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask]

    maxlab = int(labs.max()) if labs.size else 0
    counts = np.bincount(labs, minlength=maxlab + 1)
    bright_flags = (ds_vals > bright_thr).astype(np.float32)
    bright_counts = np.bincount(labs, weights=bright_flags, minlength=maxlab + 1)

    f_bright = {}
    for lab in labels_unique:
        area = counts[lab]
        if area > 10:
            f_bright[int(lab)] = float(bright_counts[lab] / max(area, 1) * 100.0)

    vals = np.array(list(f_bright.values()), dtype=np.float32)
    otsu_thr = float(threshold_otsu(vals)) if vals.size > 1 else float(np.median(vals)) if vals.size else 0.0

    positive = {lab for lab, fbr in f_bright.items() if fbr >= otsu_thr}
    negative = {lab for lab, fbr in f_bright.items() if fbr < otsu_thr}
    return positive, negative, otsu_thr, f_bright


def compute_segmentation_metrics(masks: np.ndarray, func_img: np.ndarray) -> dict:
    props = regionprops(masks.astype(np.int32), intensity_image=func_img)
    if len(props) == 0:
        return {"n_cells": 0}

    areas = [p.area for p in props]
    circularities = [4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8) for p in props]
    mean_intensities = [p.intensity_mean for p in props]

    bg_mask = masks == 0
    bg_mean = float(func_img[bg_mask].mean()) if bg_mask.any() else 0.0
    bg_std = float(func_img[bg_mask].std()) if bg_mask.any() else 1.0

    snrs = [(mi - bg_mean) / (bg_std + 1e-8) for mi in mean_intensities]

    centroids = np.array([p.centroid for p in props], dtype=np.float32)
    if len(centroids) > 1:
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        mean_nn_dist = float(np.mean(min_dists))
        min_nn_dist = float(np.min(min_dists))
    else:
        mean_nn_dist = np.nan
        min_nn_dist = np.nan

    total_px = masks.shape[0] * masks.shape[1]
    mask_px = int(np.sum(masks > 0))

    return {
        "n_cells": int(len(props)),
        "coverage_pct": float(mask_px / total_px * 100.0),
        "mean_area_px": float(np.mean(areas)),
        "std_area_px": float(np.std(areas)),
        "median_area_px": float(np.median(areas)),
        "mean_circularity": float(np.mean(circularities)),
        "std_circularity": float(np.std(circularities)),
        "mean_snr": float(np.mean(snrs)),
        "median_snr": float(np.median(snrs)),
        "mean_nn_distance": mean_nn_dist,
        "min_nn_distance": min_nn_dist,
    }


def _safe_crop_coords(cy: int, cx: int, h: int, w: int, zoom_size: int):
    y0 = int(np.clip(cy - zoom_size // 2, 0, max(0, h - zoom_size)))
    x0 = int(np.clip(cx - zoom_size // 2, 0, max(0, w - zoom_size)))
    y1 = y0 + zoom_size
    x1 = x0 + zoom_size
    return y0, y1, x0, x1


def select_dsred_rich_regions(dsred_img: np.ndarray, masks: np.ndarray,
                             positive: set, n_regions: int, zoom_size: int, min_dsred_pos: int):
    h, w = masks.shape
    props = regionprops(masks.astype(np.int32))
    if len(props) == 0:
        return []

    if len(positive) < min_dsred_pos:
        print(f"  ⚠️  Only {len(positive)} DsRed+ ROIs found")
        return []

    all_centroids = np.array([p.centroid for p in props], dtype=np.float32)
    all_labels = np.array([p.label for p in props], dtype=int)

    step = max(10, zoom_size // 3)
    candidates = []

    for y in range(zoom_size // 2, h - zoom_size // 2, step):
        for x in range(zoom_size // 2, w - zoom_size // 2, step):
            in_window = (
                (all_centroids[:, 0] >= y - zoom_size // 2) &
                (all_centroids[:, 0] < y + zoom_size // 2) &
                (all_centroids[:, 1] >= x - zoom_size // 2) &
                (all_centroids[:, 1] < x + zoom_size // 2)
            )
            labels_in = all_labels[in_window]
            n_pos = int(np.sum(np.isin(labels_in, list(positive))))
            n_tot = int(labels_in.size)
            if n_pos < min_dsred_pos or n_tot < 5:
                continue

            y0, y1, x0, x1 = _safe_crop_coords(y, x, h, w, zoom_size)
            mean_dsred = float(np.mean(dsred_img[y0:y1, x0:x1]))
            score = n_pos * 100.0 + mean_dsred * 10.0 + n_tot
            candidates.append((y, x, n_pos, n_tot, score))

    if len(candidates) == 0:
        rng = np.random.default_rng(0)
        pos_props = [p for p in props if p.label in positive]
        if len(pos_props) == 0:
            return []
        idxs = rng.choice(len(pos_props), size=min(n_regions, len(pos_props)), replace=False)
        out = []
        for i in idxs:
            cy, cx = pos_props[i].centroid
            out.append((int(cy), int(cx), 1, 1))
        return out

    candidates.sort(key=lambda t: t[4], reverse=True)

    selected = []
    for y, x, n_pos, n_tot, _ in candidates:
        if len(selected) == 0:
            selected.append((y, x, n_pos, n_tot))
        else:
            dists = [np.hypot(y - sy, x - sx) for sy, sx, _, _ in selected]
            if float(np.min(dists)) > zoom_size * 0.7:
                selected.append((y, x, n_pos, n_tot))
        if len(selected) >= n_regions:
            break

    print(f"  Selected {len(selected)} zoom regions")
    return selected


def overlay_boundaries(gray_norm: np.ndarray, masks_crop: np.ndarray, pos_set: set, neg_set: set,
                       neg_color_rgb, pos_color_rgb=(0, 1, 0)):
    rgb = np.stack([gray_norm] * 3, axis=-1).copy()
    labs = np.unique(masks_crop)
    labs = labs[labs > 0]
    for lab in labs:
        roi = masks_crop == lab
        b = find_boundaries(roi, mode="thick")
        if int(lab) in pos_set:
            rgb[b] = pos_color_rgb
        elif int(lab) in neg_set:
            rgb[b] = neg_color_rgb
    return rgb


def make_zoom_montage(func_crop_norm: np.ndarray, dsred_crop_norm: np.ndarray, masks_crop: np.ndarray,
                      pos_in_crop: set, neg_in_crop: set):
    pad = 3
    H, W = func_crop_norm.shape

    # Painel 1: GCaMP com contornos amarelos de todos os ROIs
    gcamp_rgb = overlay_boundaries(func_crop_norm, masks_crop, pos_in_crop, neg_in_crop,
                                   neg_color_rgb=(1, 1, 0), pos_color_rgb=(1, 1, 0))
    
    # Painel 2: DsRed apenas (sem contornos)
    dsred_rgb = np.stack([dsred_crop_norm] * 3, axis=-1).copy()
    
    # Painel 3: DsRed com classificação (verde = positivo, magenta = negativo)
    class_rgb = overlay_boundaries(dsred_crop_norm, masks_crop, pos_in_crop, neg_in_crop,
                                   neg_color_rgb=(1, 0, 1), pos_color_rgb=(0, 1, 0))

    montage = np.ones((H, W * 3 + pad * 2, 3), dtype=np.float32)
    montage[:, 0:W, :] = gcamp_rgb
    montage[:, W:W + pad, :] = 1.0
    montage[:, W + pad:2 * W + pad, :] = dsred_rgb
    montage[:, 2 * W + pad:2 * W + 2 * pad, :] = 1.0
    montage[:, 2 * W + 2 * pad:3 * W + 2 * pad, :] = class_rgb
    return np.clip(montage, 0, 1)


def create_thesis_figure_with_4zooms(func_img: np.ndarray, dsred_img: np.ndarray, masks: np.ndarray,
                                     positive: set, negative: set, projection_name: str, output_path: Path):
    fs = cfg.FONT_SIZE
    func_norm = normalize_for_display(func_img)
    dsred_norm = normalize_for_display(dsred_img)

    h, w = masks.shape
    regions = select_dsred_rich_regions(
        dsred_img, masks, positive,
        n_regions=cfg.N_ZOOM_REGIONS,
        zoom_size=cfg.ZOOM_SIZE,
        min_dsred_pos=cfg.MIN_DSRED_POS,
    )

    dsred_class_rgb = np.stack([dsred_norm] * 3, axis=-1).copy()
    for lab in positive:
        b = find_boundaries(masks == lab, mode="thick")
        dsred_class_rgb[b] = [0, 1, 0]
    for lab in negative:
        b = find_boundaries(masks == lab, mode="thick")
        dsred_class_rgb[b] = [1, 0, 1]

    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 0.95, 0.95], hspace=0.15, wspace=0.06)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])
    axG = fig.add_subplot(gs[3, 0])
    axH = fig.add_subplot(gs[3, 1])

    axA.imshow(func_norm, cmap="gray", interpolation="bilinear")
    axA.set_title(f"GCaMP functional ({projection_name})", fontsize=fs, fontweight="bold", pad=15)
    axA.axis("off")

    axB.imshow(dsred_norm, cmap="gray", interpolation="bilinear")
    axB.set_title("DsRed registered", fontsize=fs, fontweight="bold", pad=15)
    axB.axis("off")

    boundaries = find_boundaries(masks, mode="thick")
    func_rgb = np.stack([func_norm] * 3, axis=-1).copy()
    func_rgb[boundaries] = [1, 1, 0]
    n_total = len(positive) + len(negative)
    axC.imshow(func_rgb, interpolation="bilinear")
    axC.set_title(f"Segmentation ({n_total} ROIs)", fontsize=fs, fontweight="bold", pad=15)
    axC.axis("off")

    axD.imshow(dsred_class_rgb, interpolation="bilinear")
    axD.set_title("DsRed classification", fontsize=fs, fontweight="bold", pad=15)
    axD.axis("off")

    zoom_axes = [axE, axF, axG, axH]

    for i in range(4):
        if i >= len(regions):
            zoom_axes[i].axis("off")
            continue

        cy, cx, n_pos, n_tot = regions[i]
        y0, y1, x0, x1 = _safe_crop_coords(cy, cx, h, w, cfg.ZOOM_SIZE)

        # Rectangle em branco
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor="white", facecolor="none")
        axD.add_patch(rect)
        axD.text(
            x0 + 6, y0 + 22, f"{i + 1}", fontsize=fs, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.75, edgecolor="white", linewidth=2),
        )

        func_crop_norm = func_norm[y0:y1, x0:x1]
        dsred_crop_norm = dsred_norm[y0:y1, x0:x1]
        masks_crop = masks[y0:y1, x0:x1]

        labels_in_crop = set(np.unique(masks_crop).astype(int)) - {0}
        pos_in_crop = set([lab for lab in labels_in_crop if lab in positive])
        neg_in_crop = set([lab for lab in labels_in_crop if lab in negative])

        montage = make_zoom_montage(func_crop_norm, dsred_crop_norm, masks_crop, pos_in_crop, neg_in_crop)
        zoom_axes[i].imshow(montage, interpolation="nearest")
        # Título em preto
        zoom_axes[i].set_title(f"Zoom {i + 1}   DsRed+ {len(pos_in_crop)} of {len(labels_in_crop)}", 
                              fontsize=fs, fontweight="bold", color="black", pad=12)
        zoom_axes[i].axis("off")

        W = cfg.ZOOM_SIZE
        pad = 3
        zoom_axes[i].text(0.02, 0.96, "GCaMP", transform=zoom_axes[i].transAxes, fontsize=fs - 5, color="white",
                          va="top", bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))
        zoom_axes[i].text((W + pad) / (3 * W + 2 * pad) + 0.02, 0.96, "DsRed", transform=zoom_axes[i].transAxes,
                          fontsize=fs - 5, color="white", va="top",
                          bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))
        zoom_axes[i].text((2 * W + 2 * pad) / (3 * W + 2 * pad) + 0.02, 0.96, "Class", transform=zoom_axes[i].transAxes,
                          fontsize=fs - 5, color="white", va="top",
                          bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))

    # Add letters only to top 4 panels (A, B, C, D)
    for ax, letter in zip([axA, axB, axC, axD], list("ABCD")):
        ax.text(
            0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=fs + 6, fontweight="bold", color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )

    fig.legend(
        handles=[
            Patch(facecolor="green", label=f"DsRed+ ({len(positive)})"),
            Patch(facecolor="magenta", label=f"DsRed− ({len(negative)})"),
            Patch(facecolor="yellow", label="ROI contours"),
        ],
        loc="lower center",
        ncol=3,
        fontsize=fs - 1,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 100})
    plt.close()
    print(f"  ✓ Saved {output_path.name} at {cfg.FIGURE_DPI} DPI")


def main():
    print("=" * 80)
    print("SLICE 60 ANALYSIS")
    print("=" * 80)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ["figures", "masks", "projections", "metrics"]:
        (out_dir / d).mkdir(exist_ok=True)

    print("\n--- Loading data ---")
    stack = load_plane_frames(cfg.PLANE_IDX)
    projections = compute_projections(stack)

    for name, proj in projections.items():
        imwrite(str(out_dir / "projections" / f"plane{cfg.PLANE_IDX}_{name}.tif"), proj)

    dsred_vol = imread(cfg.DSRED_REGISTERED).astype(np.float32)
    dsred_slice = _get_z_slice(dsred_vol, cfg.PLANE_IDX)

    dsred_orig_vol = imread(cfg.DSRED_ORIGINAL).astype(np.float32)

    gcamp_anat_vol = imread(cfg.GCAMP_REGISTERED).astype(np.float32)
    gcamp_anat_slice = _get_z_slice(gcamp_anat_vol, cfg.PLANE_IDX)

    func_template_vol = imread(cfg.FUNC_TEMPLATE).astype(np.float32)
    func_template_slice = _get_z_slice(func_template_vol, cfg.PLANE_IDX)

    print("\n--- Registration overlay ---")
    create_registration_overlay(
        func_template_slice,
        gcamp_anat_slice,
        out_dir / "figures" / "registration_overlay.png",
    )

    print("\n--- DsRed integrity check ---")
    create_dsred_integrity_check(
        dsred_orig_vol,
        dsred_vol,
        cfg.PLANE_IDX,
        out_dir / "figures" / "dsred_integrity.png",
    )

    print("\n--- Running segmentation ---")
    all_results = []

    for strategy_name, proj_type, diam, cprob, use_restore in cfg.STRATEGIES:
        print(f"\n  Strategy {strategy_name}")
        print(f"  diameter {diam}  cellprob {cprob}  restore {use_restore}")

        img = projections[proj_type]

        try:
            masks = segment_with_restore(img, diam, cprob, use_restore)
        except Exception as e:
            print(f"  Segmentation failed {e}")
            continue

        positive, negative, otsu_thr, f_bright = classify_dsred(dsred_slice, masks, cfg.DSRED_PERCENTILE)
        seg_metrics = compute_segmentation_metrics(masks, img)

        result = {
            "strategy": strategy_name,
            "projection": proj_type,
            "diameter": diam,
            "cellprob": cprob,
            "restore": use_restore,
            "n_rois": seg_metrics.get("n_cells", 0),
            "n_dsred_pos": len(positive),
            "n_dsred_neg": len(negative),
            "otsu_threshold": float(otsu_thr),
            **seg_metrics,
        }
        all_results.append(result)

        print(f"  ROIs {result['n_rois']}  DsRed+ {len(positive)}  DsRed− {len(negative)}")

        imwrite(str(out_dir / "masks" / f"masks_{strategy_name}.tif"), masks.astype(np.uint16))

        create_thesis_figure_with_4zooms(
            img,
            dsred_slice,
            masks,
            positive,
            negative,
            proj_type,
            out_dir / "figures" / f"{strategy_name}_main_with_4zooms.png",
        )

    print("\n--- Saving results ---")
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "metrics" / "results.csv", index=False)

    if len(df) > 0:
        cols = ["strategy", "n_rois", "n_dsred_pos", "n_dsred_neg", "mean_area_px", "mean_circularity", "mean_snr"]
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(df[cols].to_string(index=False))

    print(f"\nDone. Output folder {out_dir}")


if __name__ == "__main__":
    main()