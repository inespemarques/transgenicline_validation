#!/usr/bin/env python3
"""
In situ (cytoplasm) classification using 2D shells (rings) around ROIs.
FIXED VERSION: Proper visualization with full contours, enhanced contrast, and legends.

Strategy (same as DsRed workflow, but applied to in situ intensity):
1) For each slice:
   - Load TIFF with 2 channels: intensity + label mask (2D).
   - Filter ROIs by area.
   - Build shells (rings) around ROIs using signed distance to the whole mask.
2) For each percentile p:
   - Bright threshold = percentile of the slice intensity (global per-slice threshold).
   - For each ROI: pct_bright = fraction of shell pixels above bright threshold.
   - Otsu on pct_bright across ROIs -> predict positive vs negative.
3) Compare against ground truth (Fiji XM/YM points -> labels).
4) Save full results and summary tables with PROPER VISUALIZATION.
"""

import os
import json
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================

# Update these paths if needed
BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "insitu_shell_validation_2d_FIXED")
os.makedirs(OUT_DIR, exist_ok=True)

# TIFFs with 2 channels: [0]=in situ intensity, [1]=label mask
TIFF_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7tif.tif"),
    300: os.path.join(BASE_RESULTS, "Results_insituslice300fish7tif.tif"),
    450: os.path.join(BASE_RESULTS, "Results_insituslice450fish7tif.tif"),
}

# Ground truth CSVs (Fiji XM/YM points) for in situ
GT_CSV_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7.csv"),
    300: os.path.join(BASE_RESULTS, "Results_insituslice300fish7.csv"),
    450: os.path.join(BASE_RESULTS, "Results_insituslice450fish7.csv"),
}

# Area filter (px) for 2D ROIs
AREA_MIN_PX = 400
AREA_MAX_PX = 6000

# Bright percentiles to test (per slice)
PERCENTILES = [60, 70, 75, 80, 85, 90, 95]

# Shells (start_dist, end_dist) in pixels
# Negative = inside (erode), positive = outside (dilate)
SHELL_CONFIGS = [
    (-5, 1), (-5, 3),
    (-3, 1),
    (-1, 3), (-1, 5), (-3, 5),
    (-5, 0), (-3, 0), (-1, 0),
    (0, 1), (0, 3), (0, 5),
]

# Figure settings - ENHANCED FOR VISIBILITY
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# In situ display: stronger contrast
INSITU_DISPLAY_PCT = (1, 99.5)  # More aggressive clipping
INSITU_GAMMA = 0.45  # Lower gamma for brighter image

# Contour settings
CONTOUR_THICKNESS = 3  # Thicker contours
CONTOUR_ALPHA = 1.0  # Fully opaque

# Zoom settings
ZOOM_PER_SLICE = 3
ZOOM_MARGIN_PX = 80  # More context
ZOOM_MIN_SIZE = 400  # Larger zooms

# Color scheme (consistent with DsRed)
COLORS = {
    "TP": np.array([0, 255, 0], dtype=np.uint8),      # Green
    "FP": np.array([255, 0, 0], dtype=np.uint8),      # Red
    "FN": np.array([0, 100, 255], dtype=np.uint8),    # Blue
    "TN": np.array([180, 180, 180], dtype=np.uint8),  # Gray
}

# =============================================================================
# IO / GT helpers
# =============================================================================

def load_tiff_channels(path):
    """Load TIFF with 2 channels: intensity and mask."""
    img = tiff.imread(path)
    print(f"    Loaded TIFF shape: {img.shape}, dtype: {img.dtype}")
    
    if img.ndim == 3 and img.shape[0] == 2:
        intensity = img[0]
        mask = img[1]
        print(f"    → Channel 0 (in situ intensity): shape={intensity.shape}")
        print(f"    → Channel 1 (label mask): shape={mask.shape}, unique labels={len(np.unique(mask))}")
    elif img.ndim == 3 and img.shape[-1] == 2:
        intensity = img[..., 0]
        mask = img[..., 1]
        print(f"    → Channel 0 (in situ intensity): shape={intensity.shape}")
        print(f"    → Channel 1 (label mask): shape={mask.shape}, unique labels={len(np.unique(mask))}")
    else:
        raise ValueError(f"Unexpected TIFF shape: {img.shape} for {path}")
    
    return intensity.astype(np.float32), mask.astype(np.int32)


def filter_by_area(mask, min_area, max_area):
    """Filter ROIs by area."""
    labels, counts = np.unique(mask, return_counts=True)
    valid = (labels > 0) & (counts >= min_area) & (counts <= max_area)
    lut = np.zeros(labels.max() + 1, dtype=mask.dtype)
    lut[labels[valid]] = labels[valid]
    filtered = lut[mask]
    print(f"    Area filter: {valid.sum()} / {(labels > 0).sum()} ROIs retained")
    return filtered


def load_positive_labels_from_csv(csv_path, label_slice):
    """Load ground truth positive labels from CSV."""
    df = pd.read_csv(csv_path)
    x_col = 'XM' if 'XM' in df.columns else next((c for c in df.columns if 'X' in c.upper()), None)
    y_col = 'YM' if 'YM' in df.columns else next((c for c in df.columns if 'Y' in c.upper()), None)
    if not x_col or not y_col:
        return set()
    xs = np.clip(df[x_col].astype(float).round().astype(int).values, 0, label_slice.shape[1] - 1)
    ys = np.clip(df[y_col].astype(float).round().astype(int).values, 0, label_slice.shape[0] - 1)
    pos_labels = {int(label_slice[y, x]) for x, y in zip(xs, ys) if int(label_slice[y, x]) > 0}
    print(f"    Ground truth: {len(pos_labels)} positive labels from CSV")
    return pos_labels


# =============================================================================
# Shells + brightness
# =============================================================================

def create_shell_2d(label_slice, start_dist, end_dist):
    """
    Create shell (ring) around the whole mask (all labels).
    Assigns label IDs to shell pixels. Outer shell uses nearest label.
    """
    binary = label_slice > 0
    if not binary.any():
        return np.zeros_like(label_slice)

    dist_out = distance_transform_edt(~binary)
    dist_in = distance_transform_edt(binary)
    dist_map = np.where(binary, -dist_in, dist_out)
    shell_mask = (dist_map >= start_dist) & (dist_map <= end_dist)

    shell = np.where(shell_mask, label_slice, 0).astype(label_slice.dtype)

    # For outer shell pixels, assign nearest label
    if end_dist > 0:
        outside = shell_mask & (dist_map > 0)
        if outside.any():
            _, idx = distance_transform_edt(~binary, return_indices=True)
            shell[outside] = label_slice[idx[0], idx[1]][outside]

    return shell


def compute_brightness_pct(label_slice, intensity_slice, shell_slice, bright_threshold):
    """
    Compute pct_bright for each label in shell.
    """
    shell = shell_slice.astype(np.int64)
    inten = intensity_slice.astype(np.float32)
    bright = (inten > bright_threshold).astype(np.uint8)

    valid = shell > 0
    labs = shell[valid].astype(np.int64)
    bvals = bright[valid].astype(np.float32)

    maxlab = int(labs.max()) if labs.size else 0
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    bright_counts = np.bincount(labs, weights=bvals, minlength=maxlab + 1).astype(np.float64)

    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present != 0]

    pct = (bright_counts[labels_present] / np.maximum(counts[labels_present], 1) * 100.0).astype(np.float32)
    return labels_present.astype(np.int64), pct


def compute_metrics(pred, gt, all_labels):
    """Compute classification metrics."""
    TP = len(pred & gt)
    FP = len(pred - gt)
    FN = len(gt - pred)
    TN = len(all_labels - pred - gt)
    total = len(all_labels)
    if total == 0:
        return None
    acc = (TP + TN) / total
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    mcc_num = (TP * TN) - (FP * FN)
    mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mcc": mcc}


# =============================================================================
# VISUALIZATION FUNCTIONS - ENHANCED
# =============================================================================

def enhance_display(img, p_low=1, p_high=99.5, gamma=0.45):
    """Enhanced contrast adjustment for in situ images."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x[x > 0], [float(p_low), float(p_high)]) if (x > 0).any() else (0, 1)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    y = np.power(y, float(gamma))
    return np.clip(y, 0, 1)


def build_full_contour_overlay(mask, labels_by_cat, thickness=3):
    """
    Build RGB overlay with FULL CONTOURS (not just boundaries).
    Each ROI gets a complete colored outline.
    """
    H, W = mask.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Build label -> category map
    maxlab = int(mask.max())
    label_to_cat = {}
    for cat, labs in labels_by_cat.items():
        for lab in labs:
            if 0 < lab <= maxlab:
                label_to_cat[int(lab)] = cat
    
    # Draw contours for each label
    for lab in np.unique(mask):
        if lab == 0:
            continue
        cat = label_to_cat.get(int(lab), "TN")
        
        # Create binary mask for this label
        label_mask = (mask == lab)
        
        # Find boundary and dilate for thickness
        boundary = find_boundaries(label_mask, mode="outer")
        if thickness > 1:
            boundary = binary_dilation(boundary, disk(thickness))
        
        # Color the boundary
        out[boundary] = COLORS[cat]
    
    return out


def build_bright_overlay(intensity, shell, bright_thr, labels_by_cat, alpha_factor=0.8):
    """
    Build RGB overlay showing bright pixels in shells.
    """
    H, W = intensity.shape
    out = np.zeros((H, W, 4), dtype=np.uint8)  # RGBA
    
    bright = (intensity > bright_thr) & (shell > 0)
    
    # Build label -> category map
    maxlab = int(shell.max()) if shell.max() > 0 else 0
    label_to_cat = {}
    for cat, labs in labels_by_cat.items():
        for lab in labs:
            if 0 < lab <= maxlab:
                label_to_cat[int(lab)] = cat
    
    # Color bright pixels by category
    for lab in np.unique(shell):
        if lab == 0:
            continue
        cat = label_to_cat.get(int(lab), "TN")
        bright_pixels = bright & (shell == lab)
        out[bright_pixels, :3] = COLORS[cat]
        out[bright_pixels, 3] = int(255 * alpha_factor)
    
    return out


def create_legend():
    """Create legend patches for the classification categories."""
    patches = [
        mpatches.Patch(color=np.array(COLORS["TP"]) / 255, label='TP (True Positive)'),
        mpatches.Patch(color=np.array(COLORS["FP"]) / 255, label='FP (False Positive)'),
        mpatches.Patch(color=np.array(COLORS["FN"]) / 255, label='FN (False Negative)'),
        mpatches.Patch(color=np.array(COLORS["TN"]) / 255, label='TN (True Negative)'),
    ]
    return patches


def select_zoom_labels(df_labels, k=3):
    """
    Pick up to 3 labels total for zooms: prefer FP, FN, TP (in that order).
    """
    picks = []
    for cat in ["FP", "FN", "TP"]:
        sub = df_labels[df_labels["category"] == cat].sort_values("pct_bright", ascending=False)
        for lab in sub["label"].tolist():
            if lab not in picks:
                picks.append(lab)
            if len(picks) >= k:
                break
        if len(picks) >= k:
            break
    if len(picks) < k:
        # fill with remaining highest-contrast ROIs
        rest = df_labels[~df_labels["label"].isin(picks)].sort_values("pct_bright", ascending=False)
        for lab in rest["label"].tolist():
            picks.append(lab)
            if len(picks) >= k:
                break
    return picks[:k]


def label_bboxes(mask):
    """Get bounding boxes for all labels."""
    from skimage.measure import regionprops
    props = regionprops(mask.astype(np.int32))
    return {int(p.label): p.bbox for p in props}


def crop_bbox(bbox, H, W, margin, min_size):
    """Crop image around bbox with margin and minimum size."""
    minr, minc, maxr, maxc = bbox
    minr = max(0, minr - margin)
    minc = max(0, minc - margin)
    maxr = min(H, maxr + margin)
    maxc = min(W, maxc + margin)
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
    r0 = max(0, r1 - size)
    c0 = max(0, c1 - size)
    return int(r0), int(r1), int(c0), int(c1)


# =============================================================================
# Main
# =============================================================================

def process_slice(slice_key):
    print(f"\n{'-'*80}\nSLICE {slice_key}\n{'-'*80}")

    tiff_path = TIFF_BY_SLICE[slice_key]
    csv_path = GT_CSV_BY_SLICE[slice_key]

    intensity, mask = load_tiff_channels(tiff_path)
    mask = filter_by_area(mask, AREA_MIN_PX, AREA_MAX_PX)

    label_slice = mask
    reference_labels = set(np.unique(label_slice)) - {0}
    gt_labels = load_positive_labels_from_csv(csv_path, label_slice)

    print(f"  ✓ Total labels={len(reference_labels)} | GT positive={len(gt_labels)}")

    results = []

    # Precompute per-slice bright thresholds
    intensity_pos = intensity[intensity > 0]
    if intensity_pos.size == 0:
        intensity_pos = intensity.reshape(-1)
    bright_thresholds = {p: float(np.percentile(intensity_pos, p)) for p in PERCENTILES}

    # cache shells to reuse for figures later
    shell_cache = {}

    for (start_dist, end_dist) in SHELL_CONFIGS:
        shell = create_shell_2d(label_slice, start_dist, end_dist)
        shell_cache[(start_dist, end_dist)] = shell

        for p in PERCENTILES:
            bright_thr = bright_thresholds[p]
            labels_present, pct = compute_brightness_pct(label_slice, intensity, shell, bright_thr)
            if len(pct) < 2:
                continue

            try:
                thr_otsu = float(threshold_otsu(pct))
            except Exception:
                continue

            predicted_positive = {int(lab) for lab, v in zip(labels_present, pct) if np.isfinite(v) and v > thr_otsu}
            metrics = compute_metrics(predicted_positive, gt_labels, reference_labels)
            if metrics is None:
                continue

            results.append({
                "slice": slice_key,
                "start": start_dist,
                "end": end_dist,
                "width": end_dist - start_dist,
                "inside": abs(min(start_dist, 0)),
                "outside": max(end_dist, 0),
                "percentile": p,
                "bright_threshold": bright_thr,
                "threshold_otsu": thr_otsu,
                **metrics
            })

    return results, {
        "intensity": intensity,
        "mask": label_slice,
        "gt_labels": gt_labels,
        "reference_labels": reference_labels,
        "bright_thresholds": bright_thresholds,
        "shell_cache": shell_cache,
    }


def main():
    print("\n" + "=" * 80)
    print("IN SITU CYTOPLASM CLASSIFICATION (2D SHELLS) - FIXED VISUALIZATION")
    print("=" * 80)
    print(f"Output: {OUT_DIR}")

    all_results = []
    slice_data = {}
    for sk in TIFF_BY_SLICE:
        results, data = process_slice(sk)
        all_results.extend(results)
        slice_data[sk] = data

    df_all = pd.DataFrame(all_results)
    out_csv = os.path.join(OUT_DIR, "results.csv")
    df_all.to_csv(out_csv, index=False)

    # Heatmap of F1 across all combinations (mean over slices)
    if not df_all.empty:
        df_all["shell_label"] = df_all.apply(lambda r: f"[{int(r['start']):+d},{int(r['end']):+d}]", axis=1)
        pivot_f1 = df_all.pivot_table(values="f1", index="shell_label", columns="percentile", aggfunc="mean")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "F1"})
        plt.title("In situ shell classification: F1 heatmap (mean across slices)")
        plt.xlabel("Percentile")
        plt.ylabel("Shell [inner, outer] px")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "F1_heatmap_all.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Per-slice figures: best config + overlays + zooms
    for sk, data in slice_data.items():
        if df_all.empty:
            continue
        df_slice = df_all[df_all["slice"] == sk].copy()
        if df_slice.empty:
            continue
        best = df_slice.loc[df_slice["f1"].idxmax()]
        best_key = (int(best["start"]), int(best["end"]))
        best_percentile = int(best["percentile"])

        intensity = data["intensity"]
        mask = data["mask"]
        gt_labels = data["gt_labels"]
        bright_thr = data["bright_thresholds"][best_percentile]
        shell = data["shell_cache"][best_key]

        labels_present, pct = compute_brightness_pct(mask, intensity, shell, bright_thr)
        if len(pct) < 2:
            continue
        try:
            thr_otsu = float(threshold_otsu(pct))
        except Exception:
            continue

        df_labels = pd.DataFrame({"label": labels_present, "pct_bright": pct})
        df_labels["gt_positive"] = df_labels["label"].astype(int).isin(gt_labels)
        df_labels["pred_positive"] = df_labels["pct_bright"] > thr_otsu
        df_labels["category"] = "TN"
        df_labels.loc[df_labels["pred_positive"] & df_labels["gt_positive"], "category"] = "TP"
        df_labels.loc[df_labels["pred_positive"] & (~df_labels["gt_positive"]), "category"] = "FP"
        df_labels.loc[(~df_labels["pred_positive"]) & df_labels["gt_positive"], "category"] = "FN"

        labels_by_cat = {
            "TP": set(df_labels[df_labels["category"] == "TP"]["label"].astype(int).tolist()),
            "FP": set(df_labels[df_labels["category"] == "FP"]["label"].astype(int).tolist()),
            "FN": set(df_labels[df_labels["category"] == "FN"]["label"].astype(int).tolist()),
            "TN": set(df_labels[df_labels["category"] == "TN"]["label"].astype(int).tolist()),
        }

        # Build overlays with ENHANCED visibility
        print(f"\n  Building visualizations for slice {sk}...")
        insitu_disp = enhance_display(intensity, *INSITU_DISPLAY_PCT, gamma=INSITU_GAMMA)
        contour_rgb = build_full_contour_overlay(mask, labels_by_cat, thickness=CONTOUR_THICKNESS)
        bright_rgba = build_bright_overlay(intensity, shell, bright_thr, labels_by_cat, alpha_factor=0.7)

        # Full overlays with LEGEND
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: In situ + contours
        axes[0].imshow(insitu_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].imshow(contour_rgb, alpha=CONTOUR_ALPHA)
        axes[0].set_title(f"Slice {sk}: In situ + Full Contours (best config)\n"
                         f"Shell=[{int(best['start'])},{int(best['end'])}]px, P{best_percentile}%, "
                         f"F1={best['f1']:.3f}", fontsize=11)
        axes[0].axis("off")
        
        # Right: Bright pixels in shells
        axes[1].imshow(insitu_disp, cmap="gray", vmin=0, vmax=1)
        # Convert RGBA to RGB for display
        bright_rgb_for_display = bright_rgba[:, :, :3]
        bright_alpha = bright_rgba[:, :, 3] / 255.0
        axes[1].imshow(bright_rgb_for_display, alpha=bright_alpha)
        axes[1].set_title(f"Bright pixels in shells (>{bright_thr:.1f})\n"
                         f"TP={int(best['TP'])} FP={int(best['FP'])} "
                         f"FN={int(best['FN'])} TN={int(best['TN'])}", fontsize=11)
        axes[1].axis("off")
        
        # Add legend
        legend_patches = create_legend()
        fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.savefig(os.path.join(FIG_DIR, f"slice_{sk}_overlay_best.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()

        # Zoom panels (3 per slice) with FULL context
        bbox_map = label_bboxes(mask)
        zoom_labels = select_zoom_labels(df_labels, k=ZOOM_PER_SLICE)
        
        fig, axes = plt.subplots(1, ZOOM_PER_SLICE, figsize=(7 * ZOOM_PER_SLICE, 7))
        if ZOOM_PER_SLICE == 1:
            axes = [axes]
        
        for ax_idx, lab in enumerate(zoom_labels):
            ax = axes[ax_idx]
            bbox = bbox_map.get(int(lab))
            if bbox is None:
                ax.axis("off")
                continue
            
            H, W = mask.shape
            r0, r1, c0, c1 = crop_bbox(bbox, H, W, ZOOM_MARGIN_PX, ZOOM_MIN_SIZE)
            
            # Crop all layers
            insitu_crop = insitu_disp[r0:r1, c0:c1]
            mask_crop = mask[r0:r1, c0:c1]
            shell_crop = shell[r0:r1, c0:c1]
            intensity_crop = intensity[r0:r1, c0:c1]
            
            # Rebuild overlays on crop
            contour_crop = build_full_contour_overlay(mask_crop, labels_by_cat, thickness=CONTOUR_THICKNESS)
            bright_crop = build_bright_overlay(intensity_crop, shell_crop, bright_thr, labels_by_cat, alpha_factor=0.6)
            
            # Display
            ax.imshow(insitu_crop, cmap="gray", vmin=0, vmax=1)
            ax.imshow(contour_crop, alpha=CONTOUR_ALPHA)
            bright_rgb_crop = bright_crop[:, :, :3]
            bright_alpha_crop = bright_crop[:, :, 3] / 255.0
            ax.imshow(bright_rgb_crop, alpha=bright_alpha_crop)
            
            cat = df_labels.loc[df_labels["label"] == lab, "category"].iloc[0]
            pb = df_labels.loc[df_labels["label"] == lab, "pct_bright"].iloc[0]
            ax.set_title(f"{cat} | Label={lab}\npct_bright={pb:.1f}%", fontsize=10, fontweight='bold')
            ax.axis("off")
        
        # Add legend to zoom figure
        fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, -0.01), fontsize=9, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.06)
        plt.savefig(os.path.join(FIG_DIR, f"slice_{sk}_zoom_best.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()

        # Save best config per slice
        best_out = {
            "slice": int(sk),
            "best_shell": [int(best["start"]), int(best["end"])],
            "best_percentile": int(best_percentile),
            "bright_threshold": float(bright_thr),
            "threshold_otsu": float(thr_otsu),
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "TP": int(best["TP"]),
            "FP": int(best["FP"]),
            "FN": int(best["FN"]),
            "TN": int(best["TN"]),
        }
        with open(os.path.join(FIG_DIR, f"slice_{sk}_best_config.json"), "w", encoding="utf-8") as f:
            json.dump(best_out, f, indent=2)

    # Summary of best config
    if not df_all.empty:
        best = df_all.loc[df_all['f1'].idxmax()]
        summary = {
            "best": best.to_dict(),
            "n_tests": int(len(df_all)),
            "n_configs": int(df_all[['start', 'end', 'percentile']].drop_duplicates().shape[0]),
        }
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("BEST CONFIG OVERALL:")
        print("=" * 80)
        print(f"  Slice: {int(best['slice'])}")
        print(f"  Shell: [{int(best['start'])}, {int(best['end'])}] px")
        print(f"  Percentile: {int(best['percentile'])}%")
        print(f"  F1={best['f1']:.4f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}")
        print(f"  TP={int(best['TP'])} FP={int(best['FP'])} FN={int(best['FN'])} TN={int(best['TN'])}")
        print("=" * 80)

    print(f"\n✓ Results saved: {out_csv}")
    print(f"✓ Figures saved: {FIG_DIR}")
    print(f"\n{'='*80}\nDONE!\n{'='*80}\n")


if __name__ == "__main__":
    main()