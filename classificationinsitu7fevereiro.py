#!/usr/bin/env python3
"""
IN-SITU SHELL OPTIMIZATION — REVISED THESIS FIGURES
=====================================================

Figures for:
  \subsubsection{In Situ Shell Geometry and Percentile Optimization}

MAIN BODY figures (3):
  Fig A: Feature distribution comparison
         Left:  bright-voxel fraction (BIMODAL → Otsu works)
         Right: mean intensity (UNIMODAL → Otsu fails)
         → Justifies feature choice
         
  Fig B: Precision / Recall / F1 vs percentile threshold
         Error bars across 3 slices. Vertical line at selected P75.
         → Justifies percentile choice (max precision, recall >= 0.80)
         
  Fig C: AUC bar chart by shell geometry
         Bar chart (not ROC curves — differences are small, bars are clearer)
         + hatched bars for classification F1 after Otsu
         → Shows shell geometry has minimal impact on separability
         → Justifies that the pipeline is robust to shell choice

APPENDIX figures (2):
  Fig D: Per-slice metrics breakdown (individual + pooled)
  Fig E: ROC curves for best shell (pooled across slices)
  
NOTES:
  - Mean/integrated intensity do NOT depend on percentile — 
    they are computed once (no percentile sweep)
  - Only bright-voxel fraction depends on percentile
  - All shells use Voronoi tessellation for exterior pixel assignment
"""

import os
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


# =============================================================================
# CONFIG
# =============================================================================

BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "insitu_optimization_figures_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_DIR = os.path.join(OUT_DIR, "main_body")
APPENDIX_DIR = os.path.join(OUT_DIR, "appendix")
os.makedirs(MAIN_DIR, exist_ok=True)
os.makedirs(APPENDIX_DIR, exist_ok=True)

TIFF_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7tif.tif"),
    300: os.path.join(BASE_RESULTS, "Results_insituslice300fish7tif.tif"),
    450: os.path.join(BASE_RESULTS, "Results_insituslice450fish7tif.tif"),
}

GT_CSV_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7.csv"),
    300: os.path.join(BASE_RESULTS, "Results_insituslice300fish7.csv"),
    450: os.path.join(BASE_RESULTS, "Results_insituslice450fish7.csv"),
}

AREA_MIN_PX = 400
AREA_MAX_PX = 6000

# Best shell from heatmap sweep (update after running heatmap script)
BEST_SHELL = (-5, 3)  # (d_start, d_end) = int 5 + ext 3 px
BEST_PERCENTILE = 75
EXCLUDE_ZEROS = True

# Shells to compare in bar chart
SHELLS_FOR_COMPARISON = [
    (-3,  0, "Int 3"),
    (-5,  0, "Int 5"),
    (-1,  1, "±1"),
    (-3,  3, "±3"),
    (-5,  3, "Int 5\n+Ext 3"),
    (-5,  5, "±5"),
    ( 0,  1, "Ext 1"),
    ( 0,  3, "Ext 3"),
    ( 0,  5, "Ext 5"),
]

PERCENTILES = [60, 65, 70, 75, 80, 85, 90, 95, 99]

FONT_SIZE = 20


# =============================================================================
# IO
# =============================================================================

def load_tiff_channels(path):
    img = tiff.imread(path)
    if img.ndim == 3 and img.shape[0] == 2:
        ch0, ch1 = img[0], img[1]
    elif img.ndim == 3 and img.shape[-1] == 2:
        ch0, ch1 = img[..., 0], img[..., 1]
    else:
        raise ValueError(f"Unexpected TIFF shape {img.shape}")
    return ch0.astype(np.float32), ch1.astype(np.int32)


def filter_by_area(mask, min_area, max_area):
    labels, counts = np.unique(mask, return_counts=True)
    valid = (labels > 0) & (counts >= min_area) & (counts <= max_area)
    if labels.size == 0:
        return mask
    lut = np.zeros(int(labels.max()) + 1, dtype=mask.dtype)
    lut[labels[valid]] = labels[valid]
    return lut[mask]


def load_positive_labels_from_csv(csv_path, label_slice):
    df = pd.read_csv(csv_path)
    x_col = "XM" if "XM" in df.columns else next((c for c in df.columns if "X" in c.upper()), None)
    y_col = "YM" if "YM" in df.columns else next((c for c in df.columns if "Y" in c.upper()), None)
    if not x_col or not y_col:
        return set()
    xs = np.clip(df[x_col].astype(float).round().astype(int).values, 0, label_slice.shape[1] - 1)
    ys = np.clip(df[y_col].astype(float).round().astype(int).values, 0, label_slice.shape[0] - 1)
    return {int(label_slice[y, x]) for x, y in zip(xs, ys) if int(label_slice[y, x]) > 0}


# =============================================================================
# VORONOI
# =============================================================================

def precompute_voronoi(label_slice):
    binary = label_slice > 0
    if not binary.any():
        H, W = label_slice.shape
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=label_slice.dtype)
    dist_out, idx = distance_transform_edt(~binary, return_indices=True)
    dist_in = distance_transform_edt(binary)
    dist_map = np.where(binary, -dist_in, dist_out).astype(np.float32)
    assigned = label_slice.copy()
    outside = ~binary
    if outside.any():
        assigned[outside] = label_slice[idx[0], idx[1]][outside]
    return dist_map, assigned


def extract_shell(dist_map, assigned, d_start, d_end):
    shell_mask = (dist_map >= float(d_start)) & (dist_map <= float(d_end))
    return np.where(shell_mask, assigned, 0)


# =============================================================================
# FEATURES
# =============================================================================

def compute_bright_threshold(intensity, percentile, exclude_zeros):
    vals = intensity.ravel()
    if exclude_zeros:
        vals = vals[vals > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, percentile))


def compute_fbright_per_roi(intensity, shell, bright_threshold):
    """Bright-voxel fraction: DEPENDS on percentile (via bright_threshold)."""
    shell_i64 = shell.astype(np.int64)
    valid = shell_i64 > 0
    if not valid.any():
        return {}
    labs = shell_i64[valid]
    ivals = intensity[valid]
    bright = (ivals > bright_threshold).astype(np.float32)
    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    bright_counts = np.bincount(labs, weights=bright, minlength=maxlab + 1)
    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present > 0]
    return {int(lab): float(bright_counts[lab] / max(counts[lab], 1) * 100.0)
            for lab in labels_present}


def compute_mean_intensity_per_roi(intensity, shell):
    """Mean intensity: does NOT depend on percentile."""
    shell_i64 = shell.astype(np.int64)
    valid = shell_i64 > 0
    if not valid.any():
        return {}
    labs = shell_i64[valid]
    ivals = intensity[valid]
    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    sum_int = np.bincount(labs, weights=ivals.astype(np.float64), minlength=maxlab + 1)
    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present > 0]
    return {int(lab): float(sum_int[lab] / max(counts[lab], 1))
            for lab in labels_present}


def compute_integrated_intensity_per_roi(intensity, shell):
    """Integrated intensity: does NOT depend on percentile."""
    shell_i64 = shell.astype(np.int64)
    valid = shell_i64 > 0
    if not valid.any():
        return {}
    labs = shell_i64[valid]
    ivals = intensity[valid]
    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    sum_int = np.bincount(labs, weights=ivals.astype(np.float64), minlength=maxlab + 1)
    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present > 0]
    return {int(lab): float(sum_int[lab]) for lab in labels_present}


# =============================================================================
# LOAD
# =============================================================================

def load_all_slices():
    slices = {}
    for sk in sorted(TIFF_BY_SLICE.keys()):
        print(f"  Loading slice {sk}...")
        intensity, mask = load_tiff_channels(TIFF_BY_SLICE[sk])
        mask = filter_by_area(mask, AREA_MIN_PX, AREA_MAX_PX)
        gt_labels = load_positive_labels_from_csv(GT_CSV_BY_SLICE[sk], mask)
        all_labels = set(np.unique(mask).astype(int)) - {0}
        dist_map, assigned = precompute_voronoi(mask)
        slices[sk] = {
            'intensity': intensity, 'mask': mask,
            'dist_map': dist_map, 'assigned': assigned,
            'gt_labels': gt_labels, 'all_labels': all_labels,
        }
        print(f"    Cells: {len(all_labels)}, GT+: {len(gt_labels)}, "
              f"GT-: {len(all_labels) - len(gt_labels)}")
    return slices


# =============================================================================
# HELPERS
# =============================================================================

def get_features_for_slice(sd, d_start, d_end, percentile, exclude_zeros):
    """Returns dict with all 3 features per ROI for one slice."""
    shell = extract_shell(sd['dist_map'], sd['assigned'], d_start, d_end)
    bright_thr = compute_bright_threshold(sd['intensity'], percentile, exclude_zeros)
    fbright = compute_fbright_per_roi(sd['intensity'], shell, bright_thr) if bright_thr else {}
    fmean = compute_mean_intensity_per_roi(sd['intensity'], shell)
    fintegrated = compute_integrated_intensity_per_roi(sd['intensity'], shell)
    return fbright, fmean, fintegrated


def classify_otsu(feat_dict, gt_labels, all_labels):
    """Otsu on feature values, return metrics."""
    if len(feat_dict) < 2:
        return None
    vals = np.array(list(feat_dict.values()), dtype=np.float32)
    if np.all(vals == vals[0]):
        return None
    try:
        otsu_thr = float(threshold_otsu(vals))
    except Exception:
        return None
    pred_pos = {l for l, v in feat_dict.items() if v >= otsu_thr}
    TP = len(pred_pos & gt_labels)
    FP = len(pred_pos - gt_labels)
    FN = len(gt_labels - pred_pos)
    TN = len(all_labels - pred_pos - gt_labels)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision': prec, 'recall': rec, 'f1': f1, 'otsu_thr': otsu_thr}


def shell_label_str(d_start, d_end):
    int_px = abs(min(0, d_start))
    ext_px = max(0, d_end)
    if int_px > 0 and ext_px > 0:
        return f"int {int_px} + ext {ext_px} px"
    elif ext_px == 0:
        return f"int {int_px} px"
    else:
        return f"ext {ext_px} px"


# =============================================================================
# FIG A: FEATURE DISTRIBUTION COMPARISON (bimodal vs unimodal)
# =============================================================================

def fig_a_feature_distributions(slices_data, out_path):
    """
    2×1 figure:
      Left:  bright-voxel fraction → BIMODAL → Otsu works
      Right: mean intensity → UNIMODAL → Otsu fails
    Pooled across all 3 slices.
    """
    fs = FONT_SIZE
    d_start, d_end = BEST_SHELL

    fbright_pos, fbright_neg = [], []
    fmean_pos, fmean_neg = [], []

    for sk, sd in sorted(slices_data.items()):
        fb, fm, _ = get_features_for_slice(sd, d_start, d_end,
                                            BEST_PERCENTILE, EXCLUDE_ZEROS)
        for lab in sd['gt_labels']:
            if lab in fb:
                fbright_pos.append(fb[lab])
            if lab in fm:
                fmean_pos.append(fm[lab])
        for lab in (sd['all_labels'] - sd['gt_labels']):
            if lab in fb:
                fbright_neg.append(fb[lab])
            if lab in fm:
                fmean_neg.append(fm[lab])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Left: bright-voxel fraction (BIMODAL) ----
    bins_b = np.linspace(0, 100, 50)
    ax1.hist(fbright_neg, bins=bins_b, alpha=0.55, color='#0173B2', density=True,
             edgecolor='black', linewidth=0.3, label=f'DsRed− (n={len(fbright_neg)})')
    ax1.hist(fbright_pos, bins=bins_b, alpha=0.55, color='#DE8F05', density=True,
             edgecolor='black', linewidth=0.3, label=f'DsRed+ (n={len(fbright_pos)})')

    all_fb = np.array(fbright_pos + fbright_neg, dtype=np.float32)
    if len(all_fb) > 1:
        try:
            otsu_b = float(threshold_otsu(all_fb))
            ax1.axvline(otsu_b, color='#D62728', linewidth=3, linestyle='--',
                        label=f'Otsu = {otsu_b:.1f}%')
        except Exception:
            pass

    ax1.set_xlabel("Bright-voxel fraction (%)", fontsize=fs, fontweight='bold')
    ax1.set_ylabel("Density", fontsize=fs, fontweight='bold')
    ax1.set_title("Bright-voxel fraction (bimodal)", fontsize=fs + 1, fontweight='bold')
    ax1.legend(fontsize=fs - 4, frameon=True, loc='upper center')
    ax1.tick_params(labelsize=fs - 4)
    ax1.grid(True, alpha=0.15)

    # ---- Right: mean intensity (UNIMODAL) ----
    all_mean = np.array(fmean_pos + fmean_neg, dtype=np.float32)
    p1, p99 = np.percentile(all_mean, [1, 99])
    bins_m = np.linspace(p1, p99, 50)

    ax2.hist(fmean_neg, bins=bins_m, alpha=0.55, color='#0173B2', density=True,
             edgecolor='black', linewidth=0.3, label=f'DsRed− (n={len(fmean_neg)})')
    ax2.hist(fmean_pos, bins=bins_m, alpha=0.55, color='#DE8F05', density=True,
             edgecolor='black', linewidth=0.3, label=f'DsRed+ (n={len(fmean_pos)})')

    if len(all_mean) > 1:
        try:
            otsu_m = float(threshold_otsu(all_mean))
            ax2.axvline(otsu_m, color='#D62728', linewidth=3, linestyle='--',
                        label=f'Otsu = {otsu_m:.1f}')
        except Exception:
            pass

    ax2.set_xlabel("Mean intensity (a.u.)", fontsize=fs, fontweight='bold')
    ax2.set_ylabel("Density", fontsize=fs, fontweight='bold')
    ax2.set_title("Mean intensity (unimodal)", fontsize=fs + 1, fontweight='bold')
    ax2.legend(fontsize=fs - 4, frameon=True, loc='upper right')
    ax2.tick_params(labelsize=fs - 4)
    ax2.grid(True, alpha=0.15)

    fig.suptitle(f"Feature Distribution — {shell_label_str(*BEST_SHELL)}, "
                 f"P{BEST_PERCENTILE}, exclude zeros",
                 fontsize=fs + 1, fontweight='bold', y=1.01)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# =============================================================================
# FIG B: PRECISION / RECALL / F1 vs PERCENTILE
# =============================================================================

def fig_b_metrics_vs_percentile(slices_data, out_path):
    """
    Precision, recall, F1 vs percentile threshold.
    Error bars = std across 3 slices (per-slice Otsu).
    Only for bright-voxel fraction.
    """
    fs = FONT_SIZE
    d_start, d_end = BEST_SHELL

    metric_data = {m: {'mean': [], 'std': []} for m in ['precision', 'recall', 'f1']}

    for p in PERCENTILES:
        slice_metrics = {'precision': [], 'recall': [], 'f1': []}
        for sk, sd in sorted(slices_data.items()):
            fb, _, _ = get_features_for_slice(sd, d_start, d_end, p, EXCLUDE_ZEROS)
            result = classify_otsu(fb, sd['gt_labels'], sd['all_labels'])
            if result is None:
                continue
            slice_metrics['precision'].append(result['precision'])
            slice_metrics['recall'].append(result['recall'])
            slice_metrics['f1'].append(result['f1'])

        for m in ['precision', 'recall', 'f1']:
            vals = slice_metrics[m]
            metric_data[m]['mean'].append(np.mean(vals) if vals else np.nan)
            metric_data[m]['std'].append(np.std(vals) if vals else 0)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_m = {'f1': '#0173B2', 'precision': '#DE8F05', 'recall': '#029E73'}
    markers_m = {'f1': 'o', 'precision': 's', 'recall': '^'}
    labels_m = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}

    for m in ['f1', 'precision', 'recall']:
        means = np.array(metric_data[m]['mean'])
        stds = np.array(metric_data[m]['std'])
        ax.errorbar(PERCENTILES, means, yerr=stds,
                    marker=markers_m[m], linewidth=2.5, markersize=9,
                    capsize=6, capthick=2, color=colors_m[m],
                    label=labels_m[m])

    ax.axvline(x=BEST_PERCENTILE, color='gray', linewidth=2.5, linestyle=':',
               alpha=0.7, zorder=0)
    ax.text(BEST_PERCENTILE + 0.8, 0.03, f'P{BEST_PERCENTILE}',
            fontsize=fs - 2, color='gray', fontweight='bold')

    ax.axhline(y=0.80, color='#029E73', linewidth=1.5, linestyle='--',
               alpha=0.4, zorder=0)
    ax.text(PERCENTILES[0] + 0.3, 0.815, 'Recall ≥ 0.80',
            fontsize=fs - 5, color='#029E73', alpha=0.7, fontweight='bold')

    ax.set_xlabel("Percentile threshold (%)", fontsize=fs, fontweight='bold')
    ax.set_ylabel("Score", fontsize=fs, fontweight='bold')
    ax.set_title("Classification Metrics vs Percentile Threshold",
                 fontsize=fs + 2, fontweight='bold')
    ax.legend(fontsize=fs - 2, frameon=True, loc='center left',
              bbox_to_anchor=(0.01, 0.45))
    ax.set_xlim(PERCENTILES[0] - 2, PERCENTILES[-1] + 2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=fs - 3)

    ax.text(0.99, 0.02, f"Shell: {shell_label_str(*BEST_SHELL)} | "
            f"Bright-voxel fraction | Per-slice Otsu | Exclude zeros",
            fontsize=fs - 6, ha='right', transform=ax.transAxes,
            color='gray', style='italic')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# =============================================================================
# FIG C: AUC + F1 BAR CHART BY SHELL GEOMETRY
# =============================================================================

def fig_c_shell_bar_chart(slices_data, out_path):
    """
    Grouped bar chart: AUC and F1 for each shell geometry.
    Uses bright-voxel fraction at BEST_PERCENTILE.
    """
    fs = FONT_SIZE

    aucs, f1s, f1_stds = [], [], []
    shell_labels = []

    for d_start, d_end, label in SHELLS_FOR_COMPARISON:
        y_true_all, y_score_all = [], []
        slice_f1s = []

        for sk, sd in sorted(slices_data.items()):
            fb, _, _ = get_features_for_slice(sd, d_start, d_end,
                                               BEST_PERCENTILE, EXCLUDE_ZEROS)
            for lab, score in fb.items():
                y_true_all.append(1 if lab in sd['gt_labels'] else 0)
                y_score_all.append(score)

            result = classify_otsu(fb, sd['gt_labels'], sd['all_labels'])
            if result is not None:
                slice_f1s.append(result['f1'])

        y_true_arr = np.array(y_true_all)
        y_score_arr = np.array(y_score_all)

        if len(y_true_arr) < 2 or len(np.unique(y_true_arr)) < 2:
            aucs.append(np.nan)
        else:
            fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
            aucs.append(auc(fpr, tpr))

        f1s.append(np.mean(slice_f1s) if slice_f1s else np.nan)
        f1_stds.append(np.std(slice_f1s) if slice_f1s else 0)
        shell_labels.append(label)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(shell_labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, aucs, width, label='AUC (ROC)',
                   color='#0173B2', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(
     x + width / 2, f1s, width,
     yerr=f1_stds,
     capsize=5,
     label='F1 (per-slice Otsu)',
     color='#DE8F05',
     edgecolor='black',
     linewidth=1.5)

    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f'{h:.3f}', ha='center', va='bottom',
                    fontsize=fs - 5, fontweight='bold')

    for i, bar in enumerate(bars2):
        h = bar.get_height()
        if not np.isnan(h):
            offset = f1_stds[i] + 0.008 if f1_stds[i] > 0 else 0.008
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f'{h:.3f}', ha='center', va='bottom',
                    fontsize=fs - 5, fontweight='bold')

    # Highlight best shell with red border
    best_label = None
    for d_start, d_end, label in SHELLS_FOR_COMPARISON:
        if (d_start, d_end) == BEST_SHELL:
            best_label = label
            break
    if best_label and best_label in shell_labels:
        best_idx = shell_labels.index(best_label)
        bars1[best_idx].set_edgecolor('#D62728')
        bars1[best_idx].set_linewidth(3.5)
        bars2[best_idx].set_edgecolor('#D62728')
        bars2[best_idx].set_linewidth(3.5)

    ax.set_xticks(x)
    ax.set_xticklabels(shell_labels, fontsize=fs - 3, fontweight='bold')
    ax.set_xlabel("Shell geometry", fontsize=fs, fontweight='bold')
    ax.set_ylabel("Score", fontsize=fs, fontweight='bold')
    ax.set_title(f"Shell Geometry Comparison — Bright-voxel Fraction, P{BEST_PERCENTILE}",
                 fontsize=fs + 1, fontweight='bold')
    ax.legend(fontsize=fs - 2, frameon=True, loc='lower right')
    ax.set_ylim(0, 1.12)
    ax.grid(axis='y', alpha=0.2)
    ax.tick_params(axis='y', labelsize=fs - 3)

    auc_vals = [a for a in aucs if not np.isnan(a)]
    f1_vals = [f for f in f1s if not np.isnan(f)]
    if auc_vals:
        ax.text(0.99, 0.15,
                f"AUC range: {min(auc_vals):.3f}–{max(auc_vals):.3f}\n"
                f"F1 range:  {min(f1_vals):.3f}–{max(f1_vals):.3f}",
                fontsize=fs - 5, ha='right', transform=ax.transAxes,
                color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# =============================================================================
# FIG D (APPENDIX): PER-SLICE METRICS BREAKDOWN
# =============================================================================

def fig_d_per_slice_metrics(slices_data, out_path):
    """Per-slice precision/recall/F1 vs percentile + pooled panel."""
    fs = FONT_SIZE
    d_start, d_end = BEST_SHELL
    slices_keys = sorted(slices_data.keys())

    fig, axes = plt.subplots(1, len(slices_keys) + 1,
                             figsize=(6 * (len(slices_keys) + 1), 6))

    colors_m = {'f1': '#0173B2', 'precision': '#DE8F05', 'recall': '#029E73'}
    markers_m = {'f1': 'o', 'precision': 's', 'recall': '^'}
    labels_m = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}

    all_slice_metrics = {p: {'precision': [], 'recall': [], 'f1': []}
                         for p in PERCENTILES}

    for idx, sk in enumerate(slices_keys):
        ax = axes[idx]
        sd = slices_data[sk]
        precs, recs, f1s = [], [], []

        for p in PERCENTILES:
            fb, _, _ = get_features_for_slice(sd, d_start, d_end, p, EXCLUDE_ZEROS)
            result = classify_otsu(fb, sd['gt_labels'], sd['all_labels'])
            if result is None:
                precs.append(np.nan); recs.append(np.nan); f1s.append(np.nan)
            else:
                precs.append(result['precision'])
                recs.append(result['recall'])
                f1s.append(result['f1'])
                all_slice_metrics[p]['precision'].append(result['precision'])
                all_slice_metrics[p]['recall'].append(result['recall'])
                all_slice_metrics[p]['f1'].append(result['f1'])

        for m, vals in [('f1', f1s), ('precision', precs), ('recall', recs)]:
            ax.plot(PERCENTILES, vals, marker=markers_m[m], linewidth=2.5,
                    color=colors_m[m], label=labels_m[m], markersize=7)

        ax.axvline(BEST_PERCENTILE, color='gray', linewidth=1.5, linestyle=':')
        ax.axhline(0.80, color='#029E73', linewidth=1, linestyle='--', alpha=0.3)
        ax.set_xlabel("Percentile (%)", fontsize=fs - 2, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Score", fontsize=fs - 2, fontweight='bold')
        ax.set_title(f"Slice {sk}\n(n={len(sd['all_labels'])}, "
                     f"GT+={len(sd['gt_labels'])})",
                     fontsize=fs - 1, fontweight='bold')
        ax.legend(fontsize=fs - 6, frameon=True, loc='center left')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(PERCENTILES[0] - 2, PERCENTILES[-1] + 2)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=fs - 5)

    # Pooled panel
    ax = axes[-1]
    for m in ['f1', 'precision', 'recall']:
        means = [np.mean(all_slice_metrics[p][m]) if all_slice_metrics[p][m]
                 else np.nan for p in PERCENTILES]
        stds = [np.std(all_slice_metrics[p][m]) if all_slice_metrics[p][m]
                else 0 for p in PERCENTILES]
        ax.errorbar(PERCENTILES, means, yerr=stds, marker=markers_m[m],
                    linewidth=2.5, markersize=7, capsize=4, capthick=2,
                    color=colors_m[m], label=labels_m[m])

    ax.axvline(BEST_PERCENTILE, color='gray', linewidth=1.5, linestyle=':')
    ax.axhline(0.80, color='#029E73', linewidth=1, linestyle='--', alpha=0.3)
    ax.set_xlabel("Percentile (%)", fontsize=fs - 2, fontweight='bold')
    ax.set_title("Pooled (mean ± std)", fontsize=fs - 1, fontweight='bold')
    ax.legend(fontsize=fs - 6, frameon=True, loc='center left')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(PERCENTILES[0] - 2, PERCENTILES[-1] + 2)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=fs - 5)

    fig.suptitle(f"Per-Slice Metrics — {shell_label_str(*BEST_SHELL)}, "
                 f"Exclude Zeros, Per-Slice Otsu",
                 fontsize=fs + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# =============================================================================
# FIG E (APPENDIX): ROC CURVES — 3 FEATURES
# =============================================================================

def fig_e_roc_features(slices_data, out_path):
    """
    ROC curves comparing 3 features at the best shell.
    
    NOTE: mean and integrated intensity do NOT depend on percentile.
    Only bright-voxel fraction uses BEST_PERCENTILE.
    """
    fs = FONT_SIZE
    d_start, d_end = BEST_SHELL

    fig, ax = plt.subplots(figsize=(9, 9))

    features = [
        ("pct_bright",           f"Bright-voxel fraction (P{BEST_PERCENTILE})", "#0173B2"),
        ("mean_intensity",       "Mean intensity",                               "#DE8F05"),
        ("integrated_intensity", "Integrated intensity",                         "#029E73"),
    ]

    for feat_name, feat_label, color in features:
        y_true_all, y_score_all = [], []

        for sk, sd in sorted(slices_data.items()):
            shell = extract_shell(sd['dist_map'], sd['assigned'], d_start, d_end)

            if feat_name == "pct_bright":
                bright_thr = compute_bright_threshold(sd['intensity'],
                                                       BEST_PERCENTILE, EXCLUDE_ZEROS)
                feat_dict = compute_fbright_per_roi(sd['intensity'], shell, bright_thr)
            elif feat_name == "mean_intensity":
                feat_dict = compute_mean_intensity_per_roi(sd['intensity'], shell)
            elif feat_name == "integrated_intensity":
                feat_dict = compute_integrated_intensity_per_roi(sd['intensity'], shell)

            for lab, score in feat_dict.items():
                y_true_all.append(1 if lab in sd['gt_labels'] else 0)
                y_score_all.append(score)

        y_true_arr = np.array(y_true_all)
        y_score_arr = np.array(y_score_all)

        if len(y_true_arr) < 2 or len(np.unique(y_true_arr)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, linewidth=2.5, color=color,
                label=f"{feat_label}\nAUC = {roc_auc:.3f}")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label="Random")

    ax.set_xlabel("False positive rate", fontsize=fs, fontweight='bold')
    ax.set_ylabel("True positive rate", fontsize=fs, fontweight='bold')
    ax.set_title(f"ROC — Feature Comparison ({shell_label_str(*BEST_SHELL)})",
                 fontsize=fs + 2, fontweight='bold')
    ax.legend(fontsize=fs - 4, loc='lower right', frameon=True, framealpha=0.95)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=fs - 4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("IN-SITU OPTIMIZATION — THESIS FIGURES v2")
    print("=" * 80)
    print(f"  Best shell: [{BEST_SHELL[0]:+d}, {BEST_SHELL[1]:+d}] "
          f"({shell_label_str(*BEST_SHELL)})")
    print(f"  Best percentile: P{BEST_PERCENTILE}")
    print(f"  Exclude zeros: {EXCLUDE_ZEROS}")
    print(f"  Output: {OUT_DIR}")

    print("\n--- Loading slices ---")
    slices_data = load_all_slices()

    total_cells = sum(len(sd['all_labels']) for sd in slices_data.values())
    total_gt = sum(len(sd['gt_labels']) for sd in slices_data.values())
    print(f"  Total: {total_cells} cells, {total_gt} GT+ "
          f"({total_gt / total_cells * 100:.1f}%)")

    # === MAIN BODY ===
    print("\n" + "=" * 80)
    print("MAIN BODY FIGURES")
    print("=" * 80)

    print("\n  [A] Feature distributions (bimodal vs unimodal)...")
    fig_a_feature_distributions(
        slices_data, os.path.join(MAIN_DIR, "fig_feature_distributions.png"))

    print("\n  [B] Metrics vs percentile...")
    fig_b_metrics_vs_percentile(
        slices_data, os.path.join(MAIN_DIR, "fig_metrics_vs_percentile.png"))

    print("\n  [C] Shell geometry bar chart...")
    fig_c_shell_bar_chart(
        slices_data, os.path.join(MAIN_DIR, "fig_shell_bar_chart.png"))

    # === APPENDIX ===
    print("\n" + "=" * 80)
    print("APPENDIX FIGURES")
    print("=" * 80)

    print("\n  [D] Per-slice metrics...")
    fig_d_per_slice_metrics(
        slices_data, os.path.join(APPENDIX_DIR, "fig_per_slice_metrics.png"))

    print("\n  [E] ROC features...")
    fig_e_roc_features(
        slices_data, os.path.join(APPENDIX_DIR, "fig_roc_features.png"))

    print(f"\n{'=' * 80}")
    print("✅ ALL FIGURES COMPLETE!")
    print(f"  Main body: {MAIN_DIR}")
    print(f"  Appendix:  {APPENDIX_DIR}")
    print(f"{'=' * 80}")

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║  THESIS LAYOUT FOR SUBSUBSECTION                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  MAIN BODY (3 figures):                                           ║
║                                                                   ║
║    Fig A — Feature distributions (bimodal vs unimodal)            ║
║      "All three features achieve AUC > 0.93, but only the        ║
║       bright-voxel fraction produces a bimodal distribution       ║
║       suitable for unsupervised Otsu thresholding."               ║
║                                                                   ║
║    Fig B — Metrics vs percentile                                  ║
║      "P75 was selected to maximise precision while maintaining    ║
║       recall >= 0.80 across all three annotated slices."          ║
║                                                                   ║
║    Fig C — Shell geometry bar chart (AUC + F1)                    ║
║      "Classification is robust to shell geometry (AUC range       ║
║       < 0.01). The selected shell [int 5 + ext 3] balances       ║
║       interior signal fidelity with cytoplasmic sampling."        ║
║                                                                   ║
║  APPENDIX:                                                        ║
║    Fig D — Per-slice metrics (inter-slice variability)            ║
║    Fig E — ROC curves for 3 features                              ║
║    Top-10 heatmap (from shell sweep script)                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()