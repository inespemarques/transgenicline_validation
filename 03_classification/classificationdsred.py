#!/usr/bin/env python3
"""
DsRed Classification - AUTO-OPTIMIZED PUBLICATION VERSION
Automatically determines best bright percentile based on F1 score.

Features:
- Sweeps all percentiles and finds optimal threshold
- Plots F1 vs percentile to show optimization
- Generates publication figures with best parameters
- Complete performance analysis

Ines Marques - February 2026
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation


# =============================================================================
# CONFIG
# =============================================================================

PIXEL_SIZE_UM = 0.1
NUCLEUS_DIAMETER_UM_RANGE = (3.0, 7.0)

AREA_MIN_PX = 400
AREA_MAX_PX = 6000

# Percentile sweep range
BRIGHT_PERCENTILES = [60, 65, 70, 75, 80, 85, 90, 95, 99]

# Auto-select best based on F1, or set manually
AUTO_SELECT_BEST = True  # If False, uses MANUAL_BEST_PERCENTILE
MANUAL_BEST_PERCENTILE = 85

# Optimization criterion
OPTIMIZATION_METRIC = "f1"  # "f1", "accuracy", or "balanced" (f1+accuracy)/2
OPTIMIZATION_METHOD = "otsu_global"  # "otsu_global" or "otsu_per_slice"

NEIGHBOR_RADIUS_PX = 1

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

OUT_DIR = r"C:\Users\OSVALDO\Downloads\results\03fev\PUBLICATION_AUTO_OPTIMIZED"
os.makedirs(OUT_DIR, exist_ok=True)

# Display settings
DSRED_DISPLAY_PCT = (0.5, 99.8)
DSRED_GAMMA = 0.55
CONTOUR_WIDTH = 4  # Thicker contours (3-4 pixels)

# Full image settings
SAVE_FULL_IMAGE_SEPARATELY = True  # Save big overview image alone
FULL_IMAGE_DPI = 300

# Zoom settings
N_ZOOM_PER_CATEGORY = 3
ZOOM_MARGIN_PX = 80
ZOOM_MIN_SIZE = 400

# Colors (colorblind-friendly)
COLORS = {
    "TP": "#00D100",
    "FP": "#FF4136",
    "FN": "#0074D9",
    "TN": "#AAAAAA",
    "bright": "#FFDC00",
}

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("colorblind")


# =============================================================================
# Data Loading (same as before)
# =============================================================================

def load_tiff_channels(tiff_path):
    img = tiff.imread(tiff_path)
    if img.ndim == 3 and img.shape[0] == 2:
        dsred = img[0]
        mask = img[1]
    elif img.ndim == 3 and img.shape[-1] == 2:
        dsred = img[..., 0]
        mask = img[..., 1]
    else:
        raise ValueError(f"Unexpected TIFF shape {img.shape}")
    return dsred.astype(np.float32), mask.astype(np.int32)


def _maybe_fix_1_based(coords_xy, W, H):
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
    patch = mask[y0:y1+1, x0:x1+1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def load_gt_positive_labels_from_fiji_csv(gt_csv_path, mask_img, neighbor_radius_px=3):
    df = pd.read_csv(gt_csv_path)
    if "XM" not in df.columns or "YM" not in df.columns:
        raise ValueError(f"GT CSV missing XM/YM columns")
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


# =============================================================================
# Metrics (same as before)
# =============================================================================

def compute_roi_metrics_sweep(dsred_img, mask_img, percentiles):
    ds = dsred_img.astype(np.float32)
    m = mask_img.astype(np.int32)
    flat_m = m.reshape(-1)
    flat_ds = ds.reshape(-1)
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask].astype(np.float32)
    if labs.size == 0:
        raise RuntimeError("No ROI pixels")
    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    present = np.nonzero(counts)[0]
    present = present[present != 0]
    area = counts[present].astype(np.int64)
    base_df = pd.DataFrame({"label": present.astype(np.int64), "area_px": area})
    ds_pos = ds[ds > 0]
    if ds_pos.size == 0:
        ds_pos = ds.reshape(-1)
    pctbright_by_pct = {}
    bright_thr_by_pct = {}
    remap = -np.ones(maxlab + 1, dtype=np.int32)
    remap[present] = np.arange(present.size, dtype=np.int32)
    ridx = remap[labs]
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
    pred = df[pred_col].astype(bool).to_numpy()
    gt = df["gt_positive"].astype(bool).to_numpy()
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    tn = int(np.sum((~pred) & (~gt)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def compute_metrics(cm):
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / (total + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# =============================================================================
# Visualization functions (import from previous version)
# =============================================================================

def enhance_dsred_for_display(dsred_img, p_low=0.5, p_high=99.8, gamma=0.55):
    ds = dsred_img.astype(np.float32)
    lo, hi = np.percentile(ds, [float(p_low), float(p_high)])
    if hi <= lo:
        return np.zeros_like(ds, dtype=np.float32)
    out = (ds - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out = np.power(out, float(gamma))
    return np.clip(out, 0.0, 1.0)


def create_contour_overlay(mask_img, df, pred_col, dsred_img, bright_thr, 
                           contour_width=2, show_bright=True):
    H, W = mask_img.shape
    ds_disp = enhance_dsred_for_display(dsred_img, *DSRED_DISPLAY_PCT, gamma=DSRED_GAMMA)
    rgb = np.stack([ds_disp, ds_disp, ds_disp], axis=-1)
    mask = mask_img.astype(np.int64)
    maxlab = int(mask.max())
    if len(df) > 0:
        maxlab = max(maxlab, int(df["label"].max()))
    pred = df[pred_col].astype(bool)
    gt = df["gt_positive"].astype(bool)
    cat_map = np.zeros(maxlab + 1, dtype=np.uint8)
    tp_l = df.loc[pred & gt, "label"].astype(int).to_numpy()
    fp_l = df.loc[pred & (~gt), "label"].astype(int).to_numpy()
    fn_l = df.loc[(~pred) & gt, "label"].astype(int).to_numpy()
    tn_l = df.loc[(~pred) & (~gt), "label"].astype(int).to_numpy()
    cat_map[tp_l] = 1
    cat_map[fp_l] = 2
    cat_map[fn_l] = 3
    cat_map[tn_l] = 4
    boundaries = find_boundaries(mask > 0, mode='outer')
    if contour_width > 1:
        for _ in range(contour_width - 1):
            boundaries = binary_dilation(boundaries)
    cat_img = cat_map[np.clip(mask, 0, maxlab)]
    alpha = 0.9
    for cat_val, color_hex in [(1, COLORS["TP"]), (2, COLORS["FP"]), 
                                (3, COLORS["FN"]), (4, COLORS["TN"])]:
        color_rgb = np.array([int(color_hex[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
        mask_cat = boundaries & (cat_img == cat_val)
        for c in range(3):
            rgb[mask_cat, c] = alpha * color_rgb[c] + (1 - alpha) * rgb[mask_cat, c]
    if show_bright:
        bright_mask = (dsred_img >= bright_thr) & (mask > 0)
        bright_color = np.array([int(COLORS["bright"][i:i+2], 16) for i in (1, 3, 5)]) / 255.0
        alpha_bright = 0.6
        for c in range(3):
            rgb[bright_mask, c] = alpha_bright * bright_color[c] + (1 - alpha_bright) * rgb[bright_mask, c]
    return rgb


def compute_label_bboxes(mask_img):
    props = regionprops(mask_img.astype(np.int32))
    out = {}
    for p in props:
        out[int(p.label)] = tuple(int(x) for x in p.bbox)
    return out


def select_representative_rois(df, pred_col, n_per_category=3):
    examples = {}
    for cat_name, cond in [
        ("TP", (df[pred_col]) & (df["gt_positive"])),
        ("FP", (df[pred_col]) & (~df["gt_positive"])),
        ("FN", (~df[pred_col]) & (df["gt_positive"])),
    ]:
        sub = df.loc[cond].sort_values("pct_bright", ascending=False).head(n_per_category)
        examples[cat_name] = []
        for _, r in sub.iterrows():
            examples[cat_name].append({
                "label": int(r["label"]),
                "pct_bright": float(r["pct_bright"]),
                "area_px": int(r["area_px"])
            })
    return examples


def crop_with_margin(bbox, H, W, margin, min_size):
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
# NEW: Optimization Plot
# =============================================================================

def plot_percentile_optimization(sweep_results, best_percentile, optimization_metric, out_path):
    """
    Plot F1 (and other metrics) vs bright percentile to show optimization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall']
    titles = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        
        # Collect data per percentile
        percentiles = sorted(sweep_results.keys())
        
        # Per-slice and global for each percentile
        per_slice_means = []
        global_means = []
        per_slice_stds = []
        global_stds = []
        
        for p in percentiles:
            # Collect metric values across slices
            ps_vals = []
            gl_vals = []
            
            for sl_data in sweep_results[p]["slices"].values():
                ps_vals.append(sl_data["metrics_per_slice"][metric])
                gl_vals.append(sl_data["metrics_global"][metric])
            
            per_slice_means.append(np.mean(ps_vals))
            global_means.append(np.mean(gl_vals))
            per_slice_stds.append(np.std(ps_vals))
            global_stds.append(np.std(gl_vals))
        
        # Plot
        ax.errorbar(percentiles, per_slice_means, yerr=per_slice_stds,
                   marker='o', linestyle='-', linewidth=2, capsize=5,
                   label='Otsu Per-Slice', color='#2E86AB')
        ax.errorbar(percentiles, global_means, yerr=global_stds,
                   marker='s', linestyle='-', linewidth=2, capsize=5,
                   label='Otsu Global', color='#A23B72')
        
        # Mark best percentile
        ax.axvline(best_percentile, color='red', linestyle='--', linewidth=2,
                  label=f'Best = {best_percentile}')
        
        ax.set_xlabel('Bright Percentile', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} vs Bright Percentile', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    fig.suptitle(f'Percentile Optimization (Best = {best_percentile} based on {optimization_metric.upper()})', 
                fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_best_percentile_summary(sweep_results, best_percentile, out_path):
    """
    Table/summary showing why best percentile was chosen.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Percentile', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1', 'Score'])
    
    percentiles = sorted(sweep_results.keys())
    
    for p in percentiles:
        for method in ['per_slice', 'global']:
            # Average across slices
            accs = []
            precs = []
            recs = []
            f1s = []
            
            metric_key = f"metrics_{method}"
            
            for sl_data in sweep_results[p]["slices"].values():
                m = sl_data[metric_key]
                accs.append(m['accuracy'])
                precs.append(m['precision'])
                recs.append(m['recall'])
                f1s.append(m['f1'])
            
            avg_acc = np.mean(accs)
            avg_prec = np.mean(precs)
            avg_rec = np.mean(recs)
            avg_f1 = np.mean(f1s)
            
            # Combined score (if using balanced)
            score = (avg_f1 + avg_acc) / 2
            
            method_name = "Per-Slice" if method == "per_slice" else "Global"
            
            # Highlight best
            row = [f'{p}', method_name, 
                   f'{avg_acc:.3f}', f'{avg_prec:.3f}', 
                   f'{avg_rec:.3f}', f'{avg_f1:.3f}',
                   f'{score:.3f}']
            
            table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.12, 0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best row(s)
    row_idx = 1
    for p in percentiles:
        for method in ['per_slice', 'global']:
            if p == best_percentile:
                for col in range(7):
                    table[(row_idx, col)].set_facecolor('#FFE5B4')
            row_idx += 1
    
    ax.set_title(f'Percentile Sweep Summary\nBest: {best_percentile} (highlighted)', 
                fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# NEW: Save full overview image separately (BIGGER) - CONTOURS ONLY VERSION
# =============================================================================

def save_full_image_with_contours(dsred_img, mask_img, df, pred_col, bright_thr,
                                  cm, metrics, slice_num, method_name, out_path):
    """
    Save ONLY the full slice with THICK contours at high resolution
    Perfect for presentations and papers!
    """
    overlay = create_contour_overlay(mask_img, df, pred_col, dsred_img, 
                                    bright_thr, contour_width=CONTOUR_WIDTH,
                                    show_bright=True)
    
    # Big square figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    
    ax.imshow(overlay)
    
    # Title with key metrics
    title_text = (f"Slice {slice_num} - {method_name}\n"
                 f"Contours: TP(green) FP(red) FN(blue) TN(gray) | Bright pixels(yellow)\n"
                 f"F1={metrics['f1']:.3f} | Accuracy={metrics['accuracy']:.3f} | "
                 f"Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f}")
    
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add metrics box in corner
    metrics_box = (
        f"Confusion Matrix:\n"
        f"TP: {cm['TP']:4d}  FP: {cm['FP']:4d}\n"
        f"FN: {cm['FN']:4d}  TN: {cm['TN']:4d}\n"
        f"Total ROIs: {len(df)}"
    )
    
    ax.text(0.98, 0.02, metrics_box,
           transform=ax.transAxes,
           fontsize=13, verticalalignment='bottom', horizontalalignment='right',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.8))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["TP"], edgecolor='black', label=f'TP (n={cm["TP"]})'),
        Patch(facecolor=COLORS["FP"], edgecolor='black', label=f'FP (n={cm["FP"]})'),
        Patch(facecolor=COLORS["FN"], edgecolor='black', label=f'FN (n={cm["FN"]})'),
        Patch(facecolor=COLORS["TN"], edgecolor='black', label=f'TN (n={cm["TN"]})'),
        Patch(facecolor=COLORS["bright"], edgecolor='black', label='Bright pixels'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, 
             fontsize=12, facecolor='white', edgecolor='black', framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=FULL_IMAGE_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# NEW: Save full overview image separately (BIGGER)
# =============================================================================

def save_full_overview_image(dsred_img, mask_img, df, pred_col, bright_thr, 
                             slice_num, method_name, cm, metrics, out_path, dpi=300):
    """
    Save LARGE overview image separately showing:
    - DsRed enhanced background
    - ROI contours colored by category (TP/FP/FN/TN)
    - Bright pixels in yellow
    - Metrics overlay
    """
    overlay = create_contour_overlay(mask_img, df, pred_col, dsred_img,
                                    bright_thr, contour_width=CONTOUR_WIDTH, 
                                    show_bright=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))  # BIG square figure
    
    ax.imshow(overlay)
    ax.axis('off')
    
    # Add title with metrics
    title_text = (f'Slice {slice_num} - {method_name}\n'
                 f'Contours: TP(green) FP(red) FN(blue) TN(gray) | Bright pixels(yellow)\n'
                 f'Acc={metrics["accuracy"]:.3f} Prec={metrics["precision"]:.3f} '
                 f'Rec={metrics["recall"]:.3f} F1={metrics["f1"]:.3f}')
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics box in corner
    metrics_text = (
        f"TP: {cm['TP']:4d}  FP: {cm['FP']:4d}\n"
        f"FN: {cm['FN']:4d}  TN: {cm['TN']:4d}\n"
        f"n_rois: {len(df)}"
    )
    
    # Position text box in top-right corner
    ax.text(0.98, 0.98, metrics_text,
           transform=ax.transAxes,
           fontsize=12, verticalalignment='top', horizontalalignment='right',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
    
    # Legend in bottom-right
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=COLORS["TP"], label='TP', edgecolor='none'),
        Rectangle((0, 0), 1, 1, fc=COLORS["FP"], label='FP', edgecolor='none'),
        Rectangle((0, 0), 1, 1, fc=COLORS["FN"], label='FN', edgecolor='none'),
        Rectangle((0, 0), 1, 1, fc=COLORS["TN"], label='TN', edgecolor='none'),
        Rectangle((0, 0), 1, 1, fc=COLORS["bright"], label='Bright', edgecolor='none'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
             fontsize=11, ncol=5, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# Publication figures (abbreviated - use functions from previous version)
# =============================================================================

def plot_confusion_matrix(cm, slice_num, method_name, out_path):
    matrix = np.array([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'Confusion Matrix - Slice {slice_num}\n{method_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_performance_metrics(results_df, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        slices = results_df['slice'].unique()
        methods = results_df['method'].unique()
        x = np.arange(len(slices))
        width = 0.35
        for i, method in enumerate(methods):
            values = []
            for sl in slices:
                val = results_df.loc[(results_df['slice'] == sl) & 
                                     (results_df['method'] == method), metric].values
                values.append(val[0] if len(val) > 0 else 0)
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, values, width, 
                         label=method.replace('_', ' ').title(), alpha=0.8)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Slice', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} by Slice and Method', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(slices)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
    fig.suptitle('Classification Performance Metrics', fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_main_publication_figure(dsred_img, mask_img, df, pred_col, cm, metrics,
                                   bright_thr, slice_num, method_name, bboxes, 
                                   examples, out_path):
    """Main publication figure with BIG overview + zoom panels"""
    overlay = create_contour_overlay(mask_img, df, pred_col, dsred_img, 
                                    bright_thr, contour_width=CONTOUR_WIDTH, show_bright=True)
    H, W = mask_img.shape
    n_zooms = sum(len(v) for v in examples.values())
    n_zoom_rows = int(np.ceil(n_zooms / 3))
    
    # Changed figsize to make it bigger
    fig = plt.figure(figsize=(20, 8 + 4*n_zoom_rows))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3 + n_zoom_rows, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # BIGGER: Main image now spans ALL 4 columns for 2 rows
    ax_full = fig.add_subplot(gs[0:2, 0:4])
    ax_full.imshow(overlay)
    ax_full.set_title(f'Slice {slice_num} - {method_name}\nContours: TP(green) FP(red) FN(blue) TN(gray) | Bright pixels(yellow)', 
                     fontsize=15, fontweight='bold')
    ax_full.axis('off')
    
    # Draw rectangles on main image to show zoom regions
    zoom_idx = 0
    for cat in ["TP", "FP", "FN"]:
        for ex in examples.get(cat, []):
            lab = ex["label"]
            bbox = bboxes.get(lab)
            if bbox is None:
                continue
            r0, r1, c0, c1 = crop_with_margin(bbox, H, W, ZOOM_MARGIN_PX, ZOOM_MIN_SIZE)
            rect = Rectangle((c0, r0), c1-c0, r1-r0, linewidth=3, edgecolor=COLORS[cat], 
                           facecolor='none', linestyle='--')
            ax_full.add_patch(rect)
            ax_full.text(c0 + 5, r0 + 15, str(zoom_idx+1), color='white', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=COLORS[cat], alpha=0.8))
            zoom_idx += 1
    
    # Metrics panel (now in row 2)
    ax_metrics = fig.add_subplot(gs[2, 0:2])
    ax_metrics.axis('off')
    metrics_text = (f"Performance Metrics\n{'='*25}\n"
                   f"Accuracy:  {metrics['accuracy']:.3f}\n"
                   f"Precision: {metrics['precision']:.3f}\n"
                   f"Recall:    {metrics['recall']:.3f}\n"
                   f"F1 Score:  {metrics['f1']:.3f}\n\n"
                   f"Confusion Matrix\n{'='*25}\n"
                   f"TP: {cm['TP']:4d}  FP: {cm['FP']:4d}\n"
                   f"FN: {cm['FN']:4d}  TN: {cm['TN']:4d}\n"
                   f"Total ROIs: {len(df)}")
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Legend panel
    ax_legend = fig.add_subplot(gs[2, 2:4])
    ax_legend.axis('off')
    legend_elements = [Rectangle((0, 0), 1, 1, fc=COLORS["TP"], label='True Positive (TP)'),
                      Rectangle((0, 0), 1, 1, fc=COLORS["FP"], label='False Positive (FP)'),
                      Rectangle((0, 0), 1, 1, fc=COLORS["FN"], label='False Negative (FN)'),
                      Rectangle((0, 0), 1, 1, fc=COLORS["TN"], label='True Negative (TN)'),
                      Rectangle((0, 0), 1, 1, fc=COLORS["bright"], label='Bright Pixels')]
    ax_legend.legend(handles=legend_elements, loc='center', frameon=True, fontsize=10)
    
    zoom_idx = 0
    for cat in ["TP", "FP", "FN"]:
        for ex in examples.get(cat, []):
            lab = ex["label"]
            pb = ex["pct_bright"]
            bbox = bboxes.get(lab)
            if bbox is None:
                continue
            row = 3 + (zoom_idx // 4)  # Start at row 3, use 4 columns
            col = zoom_idx % 4
            ax_zoom = fig.add_subplot(gs[row, col])
            r0, r1, c0, c1 = crop_with_margin(bbox, H, W, ZOOM_MARGIN_PX, ZOOM_MIN_SIZE)
            crop_overlay = overlay[r0:r1, c0:c1]
            ax_zoom.imshow(crop_overlay)
            ax_zoom.set_title(f'{zoom_idx+1}. {cat} #{lab}\npb={pb:.1f}%', 
                            fontsize=10, fontweight='bold', color=COLORS[cat])
            ax_zoom.axis('off')
            for spine in ax_zoom.spines.values():
                spine.set_edgecolor(COLORS[cat])
                spine.set_linewidth(3)
            zoom_idx += 1
    
    fig.suptitle(f'DsRed Classification - Slice {slice_num}', fontsize=16, fontweight='bold')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =============================================================================
# MAIN WITH AUTO-OPTIMIZATION
# =============================================================================

def main():
    print("="*80)
    print("DsRed CLASSIFICATION - AUTO-OPTIMIZED")
    print("="*80)
    
    print(f"\nOptimization settings:")
    print(f"  Metric: {OPTIMIZATION_METRIC.upper()}")
    print(f"  Method: {OPTIMIZATION_METHOD}")
    print(f"  Auto-select: {AUTO_SELECT_BEST}")
    if not AUTO_SELECT_BEST:
        print(f"  Manual override: {MANUAL_BEST_PERCENTILE}")
    
    # Create directories
    sweep_dir = Path(OUT_DIR) / "percentile_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    opt_dir = Path(OUT_DIR) / "optimization"
    opt_dir.mkdir(parents=True, exist_ok=True)
    pub_dir = Path(OUT_DIR) / "final_figures"
    pub_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    per_slice_data = {}
    for sl in SLICES:
        print(f"Loading slice {sl}...")
        ds, mask = load_tiff_channels(TIFF_BY_SLICE[sl])
        base_df, pct_by_pct, thr_by_pct = compute_roi_metrics_sweep(ds, mask, BRIGHT_PERCENTILES)
        base_df_filt = apply_area_filter(base_df, AREA_MIN_PX, AREA_MAX_PX)
        keep_idx = base_df.index[(base_df["area_px"] >= AREA_MIN_PX) & 
                                  (base_df["area_px"] <= AREA_MAX_PX)].to_numpy()
        pct_by_pct_filt = {p: pct[keep_idx] for p, pct in pct_by_pct.items()}
        base_df_filt = base_df_filt.reset_index(drop=True)
        gt_labels, gt_info = load_gt_positive_labels_from_fiji_csv(
            GT_CSV_BY_SLICE[sl], mask, neighbor_radius_px=NEIGHBOR_RADIUS_PX
        )
        base_df_filt["gt_positive"] = base_df_filt["label"].astype(int).isin(gt_labels)
        per_slice_data[sl] = {
            "ds": ds, "mask": mask, "base": base_df_filt,
            "pct_by_pct": pct_by_pct_filt, "thr_by_pct": thr_by_pct, "gt_info": gt_info,
        }
        print(f"  ROIs: {len(base_df_filt)}, GT: {gt_info['n_gt_labels_unique']}")
    
    # SWEEP ALL PERCENTILES
    print(f"\n{'='*80}")
    print("PERCENTILE SWEEP")
    print(f"{'='*80}")
    
    sweep_results = {}
    
    for p in BRIGHT_PERCENTILES:
        print(f"\nProcessing percentile {p}...")
        sweep_results[p] = {"slices": {}}
        
        # Global Otsu for this percentile
        pooled_rows = []
        for sl in SLICES:
            df = per_slice_data[sl]["base"].copy()
            df["pct_bright"] = per_slice_data[sl]["pct_by_pct"][p]
            pooled_rows.append(df[["pct_bright", "gt_positive"]])
        pooled = pd.concat(pooled_rows, ignore_index=True)
        otsu_global = float(threshold_otsu(pooled["pct_bright"].to_numpy()))
        
        sweep_results[p]["otsu_global"] = otsu_global
        
        # Per-slice
        for sl in SLICES:
            df = per_slice_data[sl]["base"].copy()
            df["pct_bright"] = per_slice_data[sl]["pct_by_pct"][p]
            otsu_slice = float(threshold_otsu(df["pct_bright"].to_numpy()))
            
            df["pred_per_slice"] = df["pct_bright"] >= otsu_slice
            df["pred_global"] = df["pct_bright"] >= otsu_global
            
            cm_ps = confusion_counts(df, "pred_per_slice")
            cm_gl = confusion_counts(df, "pred_global")
            met_ps = compute_metrics(cm_ps)
            met_gl = compute_metrics(cm_gl)
            
            sweep_results[p]["slices"][sl] = {
                "otsu_slice": otsu_slice,
                "cm_per_slice": cm_ps,
                "metrics_per_slice": met_ps,
                "cm_global": cm_gl,
                "metrics_global": met_gl,
            }
            
            print(f"  Slice {sl}: per-slice F1={met_ps['f1']:.3f}, global F1={met_gl['f1']:.3f}")
    
    # DETERMINE BEST PERCENTILE
    print(f"\n{'='*80}")
    print("DETERMINING BEST PERCENTILE")
    print(f"{'='*80}")
    
    if AUTO_SELECT_BEST:
        best_score = -1
        best_percentile = None
        
        for p in BRIGHT_PERCENTILES:
            # Average across slices for chosen method
            metric_key = f"metrics_{OPTIMIZATION_METHOD.replace('otsu_', '')}"
            
            scores = []
            for sl_data in sweep_results[p]["slices"].values():
                m = sl_data[metric_key]
                if OPTIMIZATION_METRIC == "f1":
                    scores.append(m["f1"])
                elif OPTIMIZATION_METRIC == "accuracy":
                    scores.append(m["accuracy"])
                elif OPTIMIZATION_METRIC == "balanced":
                    scores.append((m["f1"] + m["accuracy"]) / 2)
            
            avg_score = np.mean(scores)
            
            print(f"  Percentile {p:2d}: {OPTIMIZATION_METRIC}={avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_percentile = p
        
        print(f"\n✓ Best percentile: {best_percentile} ({OPTIMIZATION_METRIC}={best_score:.4f})")
    else:
        best_percentile = MANUAL_BEST_PERCENTILE
        print(f"\n✓ Using manual percentile: {best_percentile}")
    
    # Save sweep results
    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    
    # OPTIMIZATION PLOTS
    print(f"\n{'='*80}")
    print("GENERATING OPTIMIZATION PLOTS")
    print(f"{'='*80}")
    
    plot_percentile_optimization(sweep_results, best_percentile, OPTIMIZATION_METRIC,
                                opt_dir / "percentile_optimization.png")
    plot_best_percentile_summary(sweep_results, best_percentile,
                                 opt_dir / "best_percentile_summary.png")
    
    # PUBLICATION FIGURES WITH BEST PERCENTILE
    print(f"\n{'='*80}")
    print(f"GENERATING PUBLICATION FIGURES (percentile={best_percentile})")
    print(f"{'='*80}")
    
    p = best_percentile
    results_rows = []
    
    # Global Otsu
    pooled_rows = []
    for sl in SLICES:
        df = per_slice_data[sl]["base"].copy()
        df["pct_bright"] = per_slice_data[sl]["pct_by_pct"][p]
        pooled_rows.append(df[["pct_bright", "gt_positive"]])
    pooled = pd.concat(pooled_rows, ignore_index=True)
    otsu_global = float(threshold_otsu(pooled["pct_bright"].to_numpy()))
    
    # Per-slice
    for sl in SLICES:
        print(f"\nProcessing slice {sl}...")
        ds = per_slice_data[sl]["ds"]
        mask = per_slice_data[sl]["mask"]
        df = per_slice_data[sl]["base"].copy()
        df["pct_bright"] = per_slice_data[sl]["pct_by_pct"][p]
        bright_thr = per_slice_data[sl]["thr_by_pct"][p]
        otsu_slice = float(threshold_otsu(df["pct_bright"].to_numpy()))
        
        df["pred_per_slice"] = df["pct_bright"] >= otsu_slice
        df["pred_global"] = df["pct_bright"] >= otsu_global
        
        bboxes = compute_label_bboxes(mask)
        
        for method, pred_col in [("otsu_per_slice", "pred_per_slice"),
                                  ("otsu_global", "pred_global")]:
            cm = confusion_counts(df, pred_col)
            metrics = compute_metrics(cm)
            
            results_rows.append({
                "slice": sl, "method": method, "bright_percentile": p,
                "bright_thr": bright_thr,
                "otsu_threshold": otsu_slice if method == "otsu_per_slice" else otsu_global,
                **cm, **metrics
            })
            
            examples = select_representative_rois(df, pred_col, N_ZOOM_PER_CATEGORY)
            
            # 1. FULL IMAGE WITH CONTOURS ONLY (HIGH-RES, SEPARATE)
            save_full_image_with_contours(
                ds, mask, df, pred_col, bright_thr,
                cm, metrics, sl, method.replace('_', ' ').title(),
                pub_dir / f"slice_{sl}_{method}_FULL_CONTOURS.png"
            )
            
            # 2. Multi-panel with overview + zooms
            create_main_publication_figure(
                ds, mask, df, pred_col, cm, metrics, bright_thr,
                sl, method.replace('_', ' ').title(), bboxes, examples,
                pub_dir / f"slice_{sl}_{method}_multi_panel.png"
            )
            
            # 3. Confusion matrix
            plot_confusion_matrix(cm, sl, method.replace('_', ' ').title(),
                                 pub_dir / f"slice_{sl}_{method}_confusion.png")
        
        df_out = df.copy()
        df_out["bright_percentile"] = p
        df_out["bright_thr"] = bright_thr
        df_out.to_csv(pub_dir / f"slice_{sl}_roi_table.csv", index=False)
    
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(pub_dir / "classification_results.csv", index=False)
    
    plot_performance_metrics(results_df, pub_dir / "performance_metrics.png")
    
    # Summary
    summary = {
        "optimization": {
            "metric": OPTIMIZATION_METRIC,
            "method": OPTIMIZATION_METHOD,
            "auto_select": AUTO_SELECT_BEST,
        },
        "best_bright_percentile": int(best_percentile),
        "global_otsu_threshold": float(otsu_global),
        "average_metrics": {
            "otsu_per_slice": {
                "accuracy": float(results_df[results_df["method"] == "otsu_per_slice"]["accuracy"].mean()),
                "precision": float(results_df[results_df["method"] == "otsu_per_slice"]["precision"].mean()),
                "recall": float(results_df[results_df["method"] == "otsu_per_slice"]["recall"].mean()),
                "f1": float(results_df[results_df["method"] == "otsu_per_slice"]["f1"].mean()),
            },
            "otsu_global": {
                "accuracy": float(results_df[results_df["method"] == "otsu_global"]["accuracy"].mean()),
                "precision": float(results_df[results_df["method"] == "otsu_global"]["precision"].mean()),
                "recall": float(results_df[results_df["method"] == "otsu_global"]["recall"].mean()),
                "f1": float(results_df[results_df["method"] == "otsu_global"]["f1"].mean()),
            }
        }
    }
    
    with open(pub_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\nBest percentile: {best_percentile}")
    print(f"Optimization metric: {OPTIMIZATION_METRIC}")
    print(f"\nOutputs:")
    print(f"  {opt_dir}/ - Optimization plots")
    print(f"  {pub_dir}/ - Publication figures")
    print(f"  {sweep_dir}/ - Full sweep data")
    
    print(f"\nAverage Performance (best percentile):")
    for method in ["otsu_per_slice", "otsu_global"]:
        avg = summary["average_metrics"][method]
        print(f"\n  {method.replace('_', ' ').title()}:")
        print(f"    Accuracy:  {avg['accuracy']:.3f}")
        print(f"    Precision: {avg['precision']:.3f}")
        print(f"    Recall:    {avg['recall']:.3f}")
        print(f"    F1 Score:  {avg['f1']:.3f}")


if __name__ == "__main__":
    main()