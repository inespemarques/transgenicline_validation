#!/usr/bin/env python3
"""
DsRed classification using:
  - Bright threshold = 85th percentile of DsRed pixels within each slice (global per-slice threshold)
  - Per-ROI pct_bright = % of ROI pixels above that bright threshold
  - Otsu threshold on pct_bright:
      (A) per-slice Otsu
      (B) global Otsu pooled across slices

Ground truth positives come from Fiji "Measure" CSVs with XM/YM coordinates.
Each point is mapped to a ROI label in the mask (with optional neighbor search).

Outputs:
  - Area histogram per slice
  - pct_bright histogram per slice (with both Otsu thresholds)
  - Global pct_bright histogram (with global Otsu)
  - Overlays for both methods (per-slice Otsu vs global Otsu)
  - CSV tables with ROI metrics + GT + predicted labels + TP/FP/FN/TN categories
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops


# =============================================================================
# CONFIG
# =============================================================================

PIXEL_SIZE_UM = 0.1  # XY um/px
NUCLEUS_DIAMETER_UM_RANGE = (3.0, 7.0)

# Area filter derived from diameter range (kept configurable).
# Expected nucleus area ~ pi*(d/2)^2 => converted to pixels.
AREA_MIN_PX = 400
AREA_MAX_PX = 6000

# Sweep these percentiles for the per-slice bright threshold.
BRIGHT_PERCENTILES = [60, 65, 70, 75, 80, 85, 90, 95, 99]

NEIGHBOR_RADIUS_PX = 1    # if GT point lands on background (label 0), search nearby

SLICES = [110, 300, 450]

TIFF_BY_SLICE = {
    110: r"C:\Users\OSVALDO\Downloads\results\Results_dsredslice110fish7tif.tif",
    300: r"C:\Users\OSVALDO\Downloads\results\Results_dsredfish7_slice300.tif",
    450: r"C:\Users\OSVALDO\Downloads\results\Results_dsredslice450fish7tif.tif",
}

GT_CSV_BY_SLICE = {
    110: r"C:\Users\OSVALDO\Downloads\results\Results110dsrednovo.csv",
    300: r"C:\Users\OSVALDO\Downloads\results\Results300dsrednovo.csv",
    450: r"C:\Users\OSVALDO\Downloads\results\Results450dsrednovo.csv",
}

OUT_DIR = r"C:\Users\OSVALDO\Downloads\results\03fev\overlays_realGT_otsu_hist_final"
os.makedirs(OUT_DIR, exist_ok=True)

# Display / QA
DSRED_DISPLAY_PCT = (0.2, 99.9)  # stronger contrast
DSRED_GAMMA = 0.6                # gamma < 1 boosts faint signal
ZOOM_PER_CATEGORY = 2            # number of zoom panels per category (TP/FP/FN)
ZOOM_MARGIN_PX = 60              # padding around ROI bbox (bigger context)
ZOOM_MIN_SIZE = 360              # minimum crop size in px (show regions, not just 1 cell)


# =============================================================================
# Helpers
# =============================================================================

def load_tiff_channels(tiff_path):
    """
    Load TIFF with 2 channels: channel 0 = DsRed intensity, channel 1 = label mask.
    Supports (2,H,W) or (H,W,2).
    """
    img = tiff.imread(tiff_path)
    if img.ndim == 3 and img.shape[0] == 2:
        dsred = img[0]
        mask = img[1]
    elif img.ndim == 3 and img.shape[-1] == 2:
        dsred = img[..., 0]
        mask = img[..., 1]
    else:
        raise ValueError(f"Unexpected TIFF shape {img.shape} for {tiff_path}")
    return dsred.astype(np.float32), mask.astype(np.int32)


def _maybe_fix_1_based(coords_xy, W, H):
    """
    Fiji coordinate exports are usually 0-based, but can sometimes behave like 1-based.
    We auto-detect by minimizing out-of-bounds fraction.
    """
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    out0 = np.mean((x < 0) | (x >= W) | (y < 0) | (y >= H))
    x1 = x - 1
    y1 = y - 1
    out1 = np.mean((x1 < 0) | (x1 >= W) | (y1 < 0) | (y1 >= H))

    if out1 + 1e-6 < out0:
        return np.stack([x1, y1], axis=1), True
    return coords_xy, False


def _label_at_or_near(mask, x, y, r=3):
    H, W = mask.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))

    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return 0

    lab = int(mask[yi, xi])
    if lab != 0:
        return lab

    if r <= 0:
        return 0

    x0 = max(0, xi - r)
    x1 = min(W - 1, xi + r)
    y0 = max(0, yi - r)
    y1 = min(H - 1, yi + r)

    patch = mask[y0 : y1 + 1, x0 : x1 + 1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0

    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def load_gt_positive_labels_from_fiji_csv(gt_csv_path, mask_img, neighbor_radius_px=3):
    df = pd.read_csv(gt_csv_path)
    if "XM" not in df.columns or "YM" not in df.columns:
        raise ValueError(f"GT CSV missing XM/YM: {gt_csv_path} columns={list(df.columns)}")

    coords = df[["XM", "YM"]].to_numpy(dtype=float)
    H, W = mask_img.shape
    coords, used_minus_one = _maybe_fix_1_based(coords, W, H)

    labels = []
    n_oob = 0
    n_zero = 0
    for x, y in coords:
        lab = _label_at_or_near(mask_img, x, y, r=int(neighbor_radius_px))
        if lab == 0:
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                n_oob += 1
            else:
                n_zero += 1
        else:
            labels.append(lab)

    gt_labels = set(int(v) for v in labels)
    info = {
        "n_points": int(coords.shape[0]),
        "n_gt_labels_unique": int(len(gt_labels)),
        "used_minus_one_correction": bool(used_minus_one),
        "n_points_out_of_bounds": int(n_oob),
        "n_points_no_label_found": int(n_zero),
        "neighbor_radius_px": int(neighbor_radius_px),
    }
    return gt_labels, info


def compute_roi_metrics(dsred_img, mask_img, bright_percentile=85.0):
    raise NotImplementedError("This helper is replaced by compute_roi_metrics_sweep().")


def compute_roi_metrics_sweep(dsred_img, mask_img, percentiles):
    """
    Compute per-ROI area once, and pct_bright for many bright-threshold percentiles.

    Returns:
      base_df: label, area_px
      pctbright_by_pct: dict[int -> np.ndarray aligned with base_df rows]
      bright_thr_by_pct: dict[int -> float] intensity threshold for each percentile
    """
    ds = dsred_img.astype(np.float32)
    m = mask_img.astype(np.int32)

    # ROI pixels only
    flat_m = m.reshape(-1)
    flat_ds = ds.reshape(-1)
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask].astype(np.float32)

    if labs.size == 0:
        raise RuntimeError("Mask contains no ROI pixels (all zeros?).")

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    present = np.nonzero(counts)[0]
    present = present[present != 0]

    area = counts[present].astype(np.int64)
    base_df = pd.DataFrame({"label": present.astype(np.int64), "area_px": area})

    # Bright thresholds computed on slice DsRed pixels > 0 if present, else all.
    ds_pos = ds[ds > 0]
    if ds_pos.size == 0:
        ds_pos = ds.reshape(-1)

    pctbright_by_pct = {}
    bright_thr_by_pct = {}

    # For fast bincount weights, we remap labels to a compact index (present labels only).
    # This avoids storing large arrays for label IDs up to 15k+.
    remap = -np.ones(maxlab + 1, dtype=np.int32)
    remap[present] = np.arange(present.size, dtype=np.int32)
    ridx = remap[labs]  # 0..n_labels-1 for ROI pixels

    for p in percentiles:
        thr = float(np.percentile(ds_pos, float(p)))
        bright_thr_by_pct[int(p)] = thr
        b = (ds_vals > thr).astype(np.float32)
        bright_counts = np.bincount(ridx, weights=b, minlength=present.size).astype(np.float64)
        pctbright = (bright_counts / np.maximum(area, 1) * 100.0).astype(np.float32)
        pctbright_by_pct[int(p)] = pctbright

    return base_df, pctbright_by_pct, bright_thr_by_pct


def apply_area_filter(df, min_area_px, max_area_px):
    keep = (df["area_px"] >= int(min_area_px)) & (df["area_px"] <= int(max_area_px))
    return df.loc[keep].copy()


def confusion_counts(df, pred_col):
    # df must have gt_positive boolean and pred boolean column
    pred = df[pred_col].astype(bool).to_numpy()
    gt = df["gt_positive"].astype(bool).to_numpy()
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    tn = int(np.sum((~pred) & (~gt)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def prf(cm):
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def make_overlay(mask_img, df, pred_col, gt_col="gt_positive"):
    """
    Build RGB overlay by LUT indexed by label mask.
    Categories:
      TP=green, FP=red, FN=blue, TN=gray
    """
    mask = mask_img.astype(np.int64)
    maxlab = int(mask.max())
    if len(df) > 0:
        maxlab = max(maxlab, int(df["label"].max()))

    lut = np.zeros((maxlab + 1, 3), dtype=np.uint8)
    lut[:] = np.array([100, 100, 100], dtype=np.uint8)  # TN default
    lut[0] = np.array([0, 0, 0], dtype=np.uint8)        # background

    # Compute per-label categories from df.
    df = df.copy()
    pred = df[pred_col].astype(bool)
    gt = df[gt_col].astype(bool)

    tp_l = df.loc[pred & gt, "label"].astype(int).to_numpy()
    fp_l = df.loc[pred & (~gt), "label"].astype(int).to_numpy()
    fn_l = df.loc[(~pred) & gt, "label"].astype(int).to_numpy()

    lut[tp_l] = np.array([0, 255, 0], dtype=np.uint8)
    lut[fp_l] = np.array([255, 0, 0], dtype=np.uint8)
    lut[fn_l] = np.array([0, 0, 255], dtype=np.uint8)

    return lut[mask]

def enhance_dsred_for_display(dsred_img, p_low=0.2, p_high=99.9, gamma=0.6):
    ds = dsred_img.astype(np.float32)
    lo, hi = np.percentile(ds, [float(p_low), float(p_high)])
    if hi <= lo:
        return np.zeros_like(ds, dtype=np.float32)
    out = (ds - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    # gamma correction to enhance faint structures
    out = np.power(out, float(gamma))
    return np.clip(out, 0.0, 1.0)


def build_boundary_rgb(mask_img, df, pred_col, gt_col="gt_positive"):
    """
    Color ROI boundaries by confusion category.
    This is cheaper to render than filling whole ROIs for giant images.
    """
    mask = mask_img.astype(np.int64)
    H, W = mask.shape

    # category code per label for lookup
    df = df.copy()
    pred = df[pred_col].astype(bool)
    gt = df[gt_col].astype(bool)

    maxlab = int(mask.max())
    code = np.zeros(maxlab + 1, dtype=np.uint8)  # 0=bg/unknown
    # default TN=4
    tn_l = df.loc[(~pred) & (~gt), "label"].astype(int).to_numpy()
    tp_l = df.loc[pred & gt, "label"].astype(int).to_numpy()
    fp_l = df.loc[pred & (~gt), "label"].astype(int).to_numpy()
    fn_l = df.loc[(~pred) & gt, "label"].astype(int).to_numpy()

    # Only assign labels that actually exist in this mask (avoid out-of-bounds)
    tn_l = tn_l[(tn_l >= 0) & (tn_l <= maxlab)]
    tp_l = tp_l[(tp_l >= 0) & (tp_l <= maxlab)]
    fp_l = fp_l[(fp_l >= 0) & (fp_l <= maxlab)]
    fn_l = fn_l[(fn_l >= 0) & (fn_l <= maxlab)]

    code[tn_l] = 4
    code[tp_l] = 1
    code[fp_l] = 2
    code[fn_l] = 3

    boundaries = find_boundaries(mask > 0, mode="outer")
    out = np.zeros((H, W, 3), dtype=np.uint8)

    # map mask -> code only where boundary
    cimg = code[np.clip(mask, 0, maxlab)]
    # colors: TP green, FP red, FN blue, TN gray
    out[(boundaries) & (cimg == 1)] = np.array([0, 255, 0], dtype=np.uint8)
    out[(boundaries) & (cimg == 2)] = np.array([255, 0, 0], dtype=np.uint8)
    out[(boundaries) & (cimg == 3)] = np.array([0, 0, 255], dtype=np.uint8)
    out[(boundaries) & (cimg == 4)] = np.array([160, 160, 160], dtype=np.uint8)
    return out


def build_bright_rgb(dsred_img, mask_img, bright_thr, df, pred_col, gt_col="gt_positive"):
    """
    Color only the BRIGHT pixels (>= bright_thr) inside masks by category.
    """
    mask = mask_img.astype(np.int64)
    bright = (dsred_img.astype(np.float32) >= float(bright_thr)) & (mask > 0)
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    df = df.copy()
    pred = df[pred_col].astype(bool)
    gt = df[gt_col].astype(bool)

    maxlab = int(mask.max())
    code = np.zeros(maxlab + 1, dtype=np.uint8)
    tp_l = df.loc[pred & gt, "label"].astype(int).to_numpy()
    fp_l = df.loc[pred & (~gt), "label"].astype(int).to_numpy()
    fn_l = df.loc[(~pred) & gt, "label"].astype(int).to_numpy()
    tn_l = df.loc[(~pred) & (~gt), "label"].astype(int).to_numpy()

    tp_l = tp_l[(tp_l >= 0) & (tp_l <= maxlab)]
    fp_l = fp_l[(fp_l >= 0) & (fp_l <= maxlab)]
    fn_l = fn_l[(fn_l >= 0) & (fn_l <= maxlab)]
    tn_l = tn_l[(tn_l >= 0) & (tn_l <= maxlab)]

    code[tp_l] = 1   # TP
    code[fp_l] = 2   # FP
    code[fn_l] = 3   # FN
    code[tn_l] = 4   # TN

    cimg = code[np.clip(mask, 0, maxlab)]
    out[(bright) & (cimg == 1)] = np.array([0, 255, 0], dtype=np.uint8)
    out[(bright) & (cimg == 2)] = np.array([255, 0, 0], dtype=np.uint8)
    out[(bright) & (cimg == 3)] = np.array([0, 0, 255], dtype=np.uint8)
    out[(bright) & (cimg == 4)] = np.array([160, 160, 160], dtype=np.uint8)
    return out


def compute_label_bboxes(mask_img):
    """
    Return dict[label -> (minr, minc, maxr, maxc)] for quick zoom crops.
    """
    props = regionprops(mask_img.astype(np.int32))
    out = {}
    for p in props:
        out[int(p.label)] = tuple(int(x) for x in p.bbox)
    return out


def _select_zoom_labels(df, pred_col, k_each=2):
    """
    Pick a few labels for TP/FP/FN inspection.
    Uses pct_bright as a relevance score.
    """
    out = []
    for cat_name, cond in [
        ("FP", (df[pred_col]) & (~df["gt_positive"])),
        ("TP", (df[pred_col]) & (df["gt_positive"])),
        ("FN", (~df[pred_col]) & (df["gt_positive"])),
    ]:
        sub = df.loc[cond].sort_values("pct_bright", ascending=False).head(int(k_each))
        for _, r in sub.iterrows():
            out.append((cat_name, int(r["label"]), float(r["pct_bright"])))
    return out


def _crop_bbox(bbox, H, W, margin, min_size):
    minr, minc, maxr, maxc = bbox
    minr = max(0, minr - margin)
    minc = max(0, minc - margin)
    maxr = min(H, maxr + margin)
    maxc = min(W, maxc + margin)

    # enforce minimum square crop size centered on bbox center
    h = maxr - minr
    w = maxc - minc
    size = max(h, w, int(min_size))
    cr = (minr + maxr) // 2
    cc = (minc + maxc) // 2
    half = size // 2
    r0 = max(0, cr - half)
    c0 = max(0, cc - half)
    r1 = min(H, r0 + size)
    c1 = min(W, c0 + size)
    # adjust back if clipped
    r0 = max(0, r1 - size)
    c0 = max(0, c1 - size)
    return int(r0), int(r1), int(c0), int(c1)


def save_inspection_figure(dsred_img, mask_img, df, pred_col, cm, metrics,
                           slice_num, method_name, bright_pct, bright_thr, otsu_slice, otsu_global, out_dir):
    """
    A QA figure with:
      - Full view DsRed (contrast-enhanced) + color-coded mask boundaries
      - Zoom panels of a few TP/FP/FN ROIs with boundaries colored
    """
    ds_disp = enhance_dsred_for_display(dsred_img, *DSRED_DISPLAY_PCT, gamma=DSRED_GAMMA)
    boundary_rgb = build_boundary_rgb(mask_img, df, pred_col)
    bright_rgb = build_bright_rgb(dsred_img, mask_img, bright_thr, df, pred_col)

    bboxes = compute_label_bboxes(mask_img)
    zoom_items = _select_zoom_labels(df, pred_col, k_each=ZOOM_PER_CATEGORY)

    n_zoom = len(zoom_items)
    ncols = 3
    nrows = 2 + int(np.ceil(n_zoom / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8 + 4 * (nrows - 2)))
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Overview panel: DsRed + boundaries
    ax0 = axes[0, 0]
    ax0.imshow(ds_disp, cmap="gray")
    ax0.imshow(boundary_rgb, alpha=0.9)
    ax0.set_title(f"Slice {slice_num} - boundaries (TP green / FP red / FN blue / TN gray)")
    ax0.axis("off")

    # Overview panel: DsRed + bright pixels by category
    ax1 = axes[0, 1]
    ax1.imshow(ds_disp, cmap="gray")
    ax1.imshow(bright_rgb, alpha=0.9)
    ax1.set_title("Bright pixels inside masks (colored by category)")
    ax1.axis("off")

    # Overview panel: bright pixels only (no DsRed)
    ax2 = axes[0, 2]
    ax2.imshow(bright_rgb)
    ax2.set_title("Bright pixels only")
    ax2.axis("off")

    # Text/legend row
    ax3 = axes[1, 0]
    ax3.axis("off")
    txt = (
        f"bright percentile={bright_pct}\n"
        f"bright_thr_intensity={bright_thr:.3f}\n"
        f"otsu_slice_pctbright={otsu_slice:.3f}\n"
        f"otsu_global_pctbright={otsu_global:.3f}\n\n"
        f"TP={cm['TP']} FP={cm['FP']}\n"
        f"FN={cm['FN']} TN={cm['TN']}\n\n"
        f"precision={metrics['precision']:.3f}\n"
        f"recall={metrics['recall']:.3f}\n"
        f"f1={metrics['f1']:.3f}\n"
        f"n_rois={len(df)}\n"
        f"method={method_name}"
    )
    ax3.text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=12)

    ax4 = axes[1, 1]
    ax4.axis("off")
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="green", label="TP"),
        Rectangle((0, 0), 1, 1, fc="red", label="FP"),
        Rectangle((0, 0), 1, 1, fc="blue", label="FN"),
        Rectangle((0, 0), 1, 1, fc="gray", label="TN"),
    ]
    ax4.legend(handles=legend_elements, loc="center", frameon=True, fontsize=12)

    axes[1, 2].axis("off")

    # Zoom panels
    for idx, (cat, lab, pb) in enumerate(zoom_items):
        r = 2 + (idx // ncols)
        c = idx % ncols
        ax = axes[r, c]
        bbox = bboxes.get(int(lab))
        if bbox is None:
            ax.axis("off")
            ax.set_title(f"{cat} label={lab} (bbox missing)")
            continue
        H, W = mask_img.shape
        r0, r1, c0, c1 = _crop_bbox(bbox, H, W, ZOOM_MARGIN_PX, ZOOM_MIN_SIZE)

        ds_crop = ds_disp[r0:r1, c0:c1]
        mask_crop = mask_img[r0:r1, c0:c1]
        bright_crop = bright_rgb[r0:r1, c0:c1]
        # Boundaries for ALL ROIs in window, colored by category
        boundary_crop = build_boundary_rgb(mask_crop, df, pred_col)

        ax.imshow(ds_crop, cmap="gray")
        ax.imshow(boundary_crop, alpha=0.9)
        ax.imshow(bright_crop, alpha=0.8)
        ax.set_title(f"{cat} region around label={lab} (pct_bright={pb:.2f}%)")
        ax.axis("off")

    # Hide any unused axes
    for rr in range(nrows):
        for cc in range(ncols):
            if rr == 0:
                continue
            k = (rr - 1) * ncols + cc
            if k >= n_zoom:
                axes[rr, cc].axis("off")

    fig.tight_layout()
    out = Path(out_dir) / f"slice_{slice_num}_inspect_{method_name}_p{bright_pct}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hist_area(df, slice_num, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(df["area_px"].to_numpy(), bins=60, color="#2b6cb0", alpha=0.85)
    ax.axvline(AREA_MIN_PX, color="red", linestyle="--", linewidth=1, label=f"min={AREA_MIN_PX}px")
    ax.axvline(AREA_MAX_PX, color="red", linestyle="--", linewidth=1, label=f"max={AREA_MAX_PX}px")
    ax.set_title(f"Slice {slice_num}: ROI area histogram (px)")
    ax.set_xlabel("area (px)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / f"hist_area_slice_{slice_num}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_hist_pctbright(df, slice_num, otsu_slice, otsu_global, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(df["pct_bright"].to_numpy(), bins=60, color="#2f855a", alpha=0.85)
    ax.axvline(otsu_slice, color="blue", linestyle="--", linewidth=1, label=f"Otsu slice={otsu_slice:.3f}")
    ax.axvline(otsu_global, color="black", linestyle="--", linewidth=1, label=f"Otsu global={otsu_global:.3f}")
    ax.set_title(f"Slice {slice_num}: pct_bright histogram")
    ax.set_xlabel("pct_bright (%)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / f"hist_pctbright_slice_{slice_num}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_hist_pctbright_global(all_df, otsu_global, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(all_df["pct_bright"].to_numpy(), bins=80, color="#805ad5", alpha=0.85)
    ax.axvline(otsu_global, color="black", linestyle="--", linewidth=1, label=f"Otsu global={otsu_global:.3f}")
    ax.set_title("Global pct_bright histogram (all slices pooled)")
    ax.set_xlabel("pct_bright (%)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / "hist_pctbright_global.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_overlay_figure(dsred_img, overlay_rgb, df, pred_col, cm, metrics, slice_num, method_name, out_dir):
    ds_norm = enhance_dsred_for_display(dsred_img, *DSRED_DISPLAY_PCT)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(ds_norm, cmap="gray")
    axes[0, 0].set_title(f"Slice {slice_num} DsRed")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_rgb)
    axes[0, 1].set_title(f"Overlay categories ({method_name})")
    axes[0, 1].axis("off")

    # Focus panels
    def only_color(color):
        out = np.zeros_like(overlay_rgb)
        m = np.all(overlay_rgb == np.array(color, dtype=np.uint8), axis=2)
        out[m] = np.array(color, dtype=np.uint8)
        return out

    fp_img = only_color([255, 0, 0])
    tp_img = only_color([0, 255, 0])
    fn_img = only_color([0, 0, 255])

    axes[1, 0].imshow(ds_norm, cmap="gray", alpha=0.5)
    axes[1, 0].imshow(fp_img, alpha=0.8)
    axes[1, 0].set_title(f"FP={cm['FP']}", color="red", fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ds_norm, cmap="gray", alpha=0.5)
    axes[1, 1].imshow(tp_img, alpha=0.8)
    axes[1, 1].set_title(f"TP={cm['TP']}", color="green", fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ds_norm, cmap="gray", alpha=0.5)
    axes[1, 2].imshow(fn_img, alpha=0.8)
    axes[1, 2].set_title(f"FN={cm['FN']}", color="blue", fontweight="bold")
    axes[1, 2].axis("off")

    # Text panel with summary stats
    axes[0, 2].axis("off")
    txt = (
        f"{method_name}\n"
        f"Pred col: {pred_col}\n\n"
        f"TP={cm['TP']}  FP={cm['FP']}\n"
        f"FN={cm['FN']}  TN={cm['TN']}\n\n"
        f"precision={metrics['precision']:.3f}\n"
        f"recall={metrics['recall']:.3f}\n"
        f"f1={metrics['f1']:.3f}\n\n"
        f"n_rois={len(df)}"
    )
    axes[0, 2].text(0.0, 1.0, txt, va="top", ha="left", family="monospace", fontsize=12)

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc="green", label="TP"),
        Rectangle((0, 0), 1, 1, fc="red", label="FP"),
        Rectangle((0, 0), 1, 1, fc="blue", label="FN"),
        Rectangle((0, 0), 1, 1, fc="gray", label="TN"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    out = Path(out_dir) / f"slice_{slice_num}_overlay_{method_name}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Save FP-only overlay as TIFF for quick inspection
    fp_only = fp_img
    out_fp = Path(out_dir) / f"slice_{slice_num}_FP_only_{method_name}.tif"
    tiff.imwrite(str(out_fp), fp_only)


def main():
    # Derived area estimate (informational)
    dmin, dmax = NUCLEUS_DIAMETER_UM_RANGE
    rmin_px = (dmin / 2.0) / PIXEL_SIZE_UM
    rmax_px = (dmax / 2.0) / PIXEL_SIZE_UM
    area_min_est = np.pi * rmin_px * rmin_px
    area_max_est = np.pi * rmax_px * rmax_px

    print("Config:")
    print(f"  bright_percentiles={BRIGHT_PERCENTILES}")
    print(f"  area_filter_px=[{AREA_MIN_PX}, {AREA_MAX_PX}]")
    print(f"  pixel_size_um={PIXEL_SIZE_UM} nucleus_diam_um={NUCLEUS_DIAMETER_UM_RANGE}")
    print(f"  est_area_px~[{area_min_est:.0f}, {area_max_est:.0f}] (pi*(d/2)^2)")
    print(f"  out_dir={OUT_DIR}")

    # Load slices once (big arrays), compute ROI base metrics and GT once.
    per_slice_static = {}
    for sl in SLICES:
        ds, mask = load_tiff_channels(TIFF_BY_SLICE[sl])
        base_df, pct_by_pct, thr_by_pct = compute_roi_metrics_sweep(ds, mask, BRIGHT_PERCENTILES)
        base_df_all = base_df.copy()
        base_df_filt = apply_area_filter(base_df, AREA_MIN_PX, AREA_MAX_PX)

        # Apply the same row-filter indices to pct_bright arrays
        keep_idx = base_df.index[(base_df["area_px"] >= AREA_MIN_PX) & (base_df["area_px"] <= AREA_MAX_PX)].to_numpy()
        pct_by_pct_filt = {p: pct[keep_idx] for p, pct in pct_by_pct.items()}
        base_df_filt = base_df_filt.reset_index(drop=True)

        gt_labels, gt_info = load_gt_positive_labels_from_fiji_csv(GT_CSV_BY_SLICE[sl], mask, neighbor_radius_px=NEIGHBOR_RADIUS_PX)
        base_df_filt["gt_positive"] = base_df_filt["label"].astype(int).isin(gt_labels)

        per_slice_static[sl] = {
            "ds": ds,
            "mask": mask,
            "base_all": base_df_all,
            "base": base_df_filt,
            "pct_by_pct": pct_by_pct_filt,
            "thr_by_pct": thr_by_pct,
            "gt_labels": gt_labels,
            "gt_info": gt_info,
        }

        print(f"\nSlice {sl}: mask shape={mask.shape} ROIs(all)={len(base_df_all)} ROIs(filtered)={len(base_df_filt)} "
              f"GT_points={gt_info['n_points']} GT_labels_unique={gt_info['n_gt_labels_unique']}")

    # Sweep percentiles
    sweep_summary = {}
    for p in BRIGHT_PERCENTILES:
        pct_out_dir = Path(OUT_DIR) / f"bright_p{int(p)}"
        pct_out_dir.mkdir(parents=True, exist_ok=True)

        # Pooled pct_bright across slices for global Otsu at this percentile.
        pooled_rows = []
        for sl in SLICES:
            df = per_slice_static[sl]["base"].copy()
            df["slice"] = sl
            df["pct_bright"] = per_slice_static[sl]["pct_by_pct"][int(p)]
            pooled_rows.append(df[["slice", "label", "area_px", "pct_bright", "gt_positive"]])
        pooled = pd.concat(pooled_rows, ignore_index=True)
        otsu_global = float(threshold_otsu(pooled["pct_bright"].to_numpy(dtype=np.float32)))

        plot_hist_pctbright_global(pooled, otsu_global, pct_out_dir)

        # Per-slice Otsu and outputs
        sweep_summary[str(p)] = {"otsu_global": otsu_global, "per_slice": {}}

        for sl in SLICES:
            ds = per_slice_static[sl]["ds"]
            mask = per_slice_static[sl]["mask"]
            df = per_slice_static[sl]["base"].copy()
            df["pct_bright"] = per_slice_static[sl]["pct_by_pct"][int(p)]

            vals = df["pct_bright"].to_numpy(dtype=np.float32)
            otsu_slice = float(threshold_otsu(vals))

            df["pred_pos_per_slice"] = df["pct_bright"] >= otsu_slice
            df["pred_pos_global"] = df["pct_bright"] >= otsu_global

            cm_slice = confusion_counts(df, "pred_pos_per_slice")
            cm_global = confusion_counts(df, "pred_pos_global")
            met_slice = prf(cm_slice)
            met_global = prf(cm_global)

            sweep_summary[str(p)]["per_slice"][str(sl)] = {
                "bright_thr_intensity": per_slice_static[sl]["thr_by_pct"][int(p)],
                "otsu_slice_pctbright": otsu_slice,
                "cm_per_slice": cm_slice,
                "metrics_per_slice": met_slice,
                "cm_global": cm_global,
                "metrics_global": met_global,
            }

            # Histograms (area once per percentile dir; pct_bright per percentile)
            plot_hist_area(per_slice_static[sl]["base_all"], sl, pct_out_dir)
            plot_hist_pctbright(df, sl, otsu_slice, otsu_global, pct_out_dir)

            # Overlays (filled ROIs)
            overlay_slice = make_overlay(mask, df, "pred_pos_per_slice")
            overlay_global = make_overlay(mask, df, "pred_pos_global")

            save_overlay_figure(ds, overlay_slice, df, "pred_pos_per_slice", cm_slice, met_slice, sl, "otsu_per_slice", pct_out_dir)
            save_overlay_figure(ds, overlay_global, df, "pred_pos_global", cm_global, met_global, sl, "otsu_global", pct_out_dir)

            # Inspection figures with boundaries + zooms
            save_inspection_figure(
                ds, mask, df, "pred_pos_per_slice", cm_slice, met_slice,
                sl, "otsu_per_slice", int(p), per_slice_static[sl]["thr_by_pct"][int(p)], otsu_slice, otsu_global, pct_out_dir
            )
            save_inspection_figure(
                ds, mask, df, "pred_pos_global", cm_global, met_global,
                sl, "otsu_global", int(p), per_slice_static[sl]["thr_by_pct"][int(p)], otsu_slice, otsu_global, pct_out_dir
            )

            # Save table for this slice+percentile
            out_csv = pct_out_dir / f"slice_{sl}_roi_table_p{int(p)}.csv"
            df_out = df.copy()
            df_out["bright_percentile"] = int(p)
            df_out["bright_thr_intensity"] = per_slice_static[sl]["thr_by_pct"][int(p)]
            df_out["otsu_thr_pct_bright_per_slice"] = otsu_slice
            df_out["otsu_thr_pct_bright_global"] = otsu_global
            df_out.to_csv(out_csv, index=False)

            print(f"p{int(p)} slice {sl}: per-slice F1={met_slice['f1']:.3f} global F1={met_global['f1']:.3f} "
                  f"(otsu_slice={otsu_slice:.3f} otsu_global={otsu_global:.3f})")

        with open(pct_out_dir / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump({
                "bright_percentile": int(p),
                "area_filter_px": [AREA_MIN_PX, AREA_MAX_PX],
                "neighbor_radius_px": NEIGHBOR_RADIUS_PX,
                "otsu_global_pct_bright": otsu_global,
                "otsu_per_slice_pct_bright": {str(sl): sweep_summary[str(p)]["per_slice"][str(sl)]["otsu_slice_pctbright"] for sl in SLICES},
                "tiff_by_slice": TIFF_BY_SLICE,
                "gt_csv_by_slice": GT_CSV_BY_SLICE,
                "dsred_display_percentiles": list(DSRED_DISPLAY_PCT),
            }, f, indent=2)

        # Per-percentile summary table
        rows = []
        for sl in SLICES:
            s = sweep_summary[str(p)]["per_slice"][str(sl)]
            rows.append({
                "slice": sl,
                "bright_percentile": int(p),
                "method": "otsu_per_slice",
                "bright_thr_intensity": s["bright_thr_intensity"],
                "otsu_threshold": s["otsu_slice_pctbright"],
                "TP": s["cm_per_slice"]["TP"],
                "FP": s["cm_per_slice"]["FP"],
                "FN": s["cm_per_slice"]["FN"],
                "TN": s["cm_per_slice"]["TN"],
                "precision": s["metrics_per_slice"]["precision"],
                "recall": s["metrics_per_slice"]["recall"],
                "f1": s["metrics_per_slice"]["f1"],
            })
            rows.append({
                "slice": sl,
                "bright_percentile": int(p),
                "method": "otsu_global",
                "bright_thr_intensity": s["bright_thr_intensity"],
                "otsu_threshold": sweep_summary[str(p)]["otsu_global"],
                "TP": s["cm_global"]["TP"],
                "FP": s["cm_global"]["FP"],
                "FN": s["cm_global"]["FN"],
                "TN": s["cm_global"]["TN"],
                "precision": s["metrics_global"]["precision"],
                "recall": s["metrics_global"]["recall"],
                "f1": s["metrics_global"]["f1"],
            })

        pd.DataFrame(rows).to_csv(pct_out_dir / "confusion_summary.csv", index=False)

    with open(Path(OUT_DIR) / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, indent=2)

    # Global summary CSV across all percentiles/slices
    all_rows = []
    for p in BRIGHT_PERCENTILES:
        pct_dir = Path(OUT_DIR) / f"bright_p{int(p)}"
        csv_path = pct_dir / "confusion_summary.csv"
        if csv_path.exists():
            all_rows.append(pd.read_csv(csv_path))
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(Path(OUT_DIR) / "confusion_summary_all.csv", index=False)

    print("\nDone.")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
