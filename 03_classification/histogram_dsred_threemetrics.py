#!/usr/bin/env python3
"""
Feature comparison for DsRed classification.
Computes and compares three features:
  1. Fraction of bright pixels (pct_bright) - your existing feature
  2. Mean intensity inside mask
  3. Integrated intensity (sum of all pixels in mask)

Each feature is evaluated with Otsu threshold and performance metrics are compared.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import rcParams

from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table


# =============================================================================
# THESIS-QUALITY PLOT SETTINGS
# =============================================================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.width'] = 1.2


# =============================================================================
# CONFIG - ADAPT TO YOUR PATHS
# =============================================================================

PIXEL_SIZE_UM = 0.1
AREA_MIN_PX = 400
AREA_MAX_PX = 6000
NEIGHBOR_RADIUS_PX = 1
BRIGHT_PERCENTILE = 85  # Use your best percentile from sweep

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

OUT_DIR = r"C:\Users\OSVALDO\Downloads\results\03fev\feature_comparison"
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS FROM YOUR ORIGINAL SCRIPT
# =============================================================================

def load_tiff_channels(tiff_path):
    """Load TIFF with 2 channels: channel 0 = DsRed, channel 1 = mask."""
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
    """Auto-detect 1-based vs 0-based Fiji coordinates."""
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
    """Find label at coordinate (x,y) or nearby within radius r."""
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
    """Load ground truth positive labels from Fiji CSV."""
    df = pd.read_csv(gt_csv_path)
    if "XM" not in df.columns or "YM" not in df.columns:
        raise ValueError(f"GT CSV missing XM/YM: {gt_csv_path}")
    
    coords = df[["XM", "YM"]].to_numpy(dtype=float)
    H, W = mask_img.shape
    coords, used_minus_one = _maybe_fix_1_based(coords, W, H)
    
    labels = []
    for x, y in coords:
        lab = _label_at_or_near(mask_img, x, y, r=int(neighbor_radius_px))
        if lab != 0:
            labels.append(lab)
    
    gt_labels = set(int(v) for v in labels)
    return gt_labels


def apply_area_filter(df, min_area_px, max_area_px):
    """Filter ROIs by area."""
    keep = (df["area_px"] >= int(min_area_px)) & (df["area_px"] <= int(max_area_px))
    return df.loc[keep].copy()


# =============================================================================
# FEATURE COMPUTATION FUNCTIONS
# =============================================================================

def compute_pct_bright_feature(dsred_img, mask_img, bright_percentile=85):
    """
    Feature 1: Fraction of bright pixels.
    Computes % of pixels above bright threshold for each ROI.
    """
    ds = dsred_img.astype(np.float32)
    m = mask_img.astype(np.int32)
    
    # Compute bright threshold from positive pixels
    ds_pos = ds[ds > 0]
    if ds_pos.size == 0:
        ds_pos = ds.reshape(-1)
    bright_thr = float(np.percentile(ds_pos, float(bright_percentile)))
    
    # ROI pixels only
    flat_m = m.reshape(-1)
    flat_ds = ds.reshape(-1)
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask].astype(np.float32)
    
    if labs.size == 0:
        raise RuntimeError("Mask contains no ROI pixels.")
    
    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    present = np.nonzero(counts)[0]
    present = present[present != 0]
    
    area = counts[present].astype(np.int64)
    
    # Remap labels to compact index
    remap = -np.ones(maxlab + 1, dtype=np.int32)
    remap[present] = np.arange(present.size, dtype=np.int32)
    ridx = remap[labs]
    
    # Count bright pixels per ROI
    b = (ds_vals > bright_thr).astype(np.float32)
    bright_counts = np.bincount(ridx, weights=b, minlength=present.size).astype(np.float64)
    pctbright = (bright_counts / np.maximum(area, 1) * 100.0).astype(np.float32)
    
    df = pd.DataFrame({
        "label": present.astype(np.int64),
        "area_px": area,
        "pct_bright": pctbright
    })
    
    return df, bright_thr


def compute_mean_intensity_feature(dsred_img, mask_img):
    """
    Feature 2: Mean intensity inside each ROI.
    """
    props = regionprops_table(
        mask_img.astype(np.int32),
        intensity_image=dsred_img.astype(np.float32),
        properties=['label', 'area', 'intensity_mean']
    )
    
    df = pd.DataFrame(props)
    df.rename(columns={'area': 'area_px', 'intensity_mean': 'mean_intensity'}, inplace=True)
    return df


def compute_integrated_intensity_feature(dsred_img, mask_img):
    """
    Feature 3: Integrated intensity (sum of all pixel intensities in ROI).
    """
    mask = mask_img.astype(np.int32)
    ds = dsred_img.astype(np.float32)
    
    # Flatten and filter to ROI pixels only
    flat_m = mask.reshape(-1)
    flat_ds = ds.reshape(-1)
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask].astype(np.float32)
    
    if labs.size == 0:
        raise RuntimeError("Mask contains no ROI pixels.")
    
    maxlab = int(labs.max())
    
    # Sum intensity per label using bincount
    integrated = np.bincount(labs, weights=ds_vals, minlength=maxlab + 1)
    
    # Also get area
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    
    present = np.nonzero(integrated)[0]
    present = present[present != 0]
    
    df = pd.DataFrame({
        'label': present.astype(np.int64),
        'area_px': counts[present].astype(np.int64),
        'integrated_intensity': integrated[present].astype(np.float32)
    })
    
    return df


# =============================================================================
# CLASSIFICATION AND EVALUATION
# =============================================================================

def classify_with_otsu_and_evaluate(feature_values, gt_labels_series):
    """
    Apply Otsu threshold to feature values and compute metrics.
    
    Returns dict with: precision, recall, f1, confusion_matrix, threshold
    """
    vals = np.array(feature_values, dtype=np.float32)
    gt = np.array(gt_labels_series, dtype=bool)
    
    # Compute Otsu threshold
    threshold = threshold_otsu(vals)
    
    # Predict positive if above threshold
    pred = vals >= threshold
    
    # Confusion matrix
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    tn = int(np.sum((~pred) & (~gt)))
    
    # Metrics
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn},
        'threshold': float(threshold)
    }


# =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def create_feature_comparison_plot(metrics_data, output_path='feature_comparison.png'):
    """
    Create thesis-quality grouped bar chart comparing features.
    """
    features = list(metrics_data.keys())
    precision_vals = [metrics_data[f]['precision'] for f in features]
    recall_vals = [metrics_data[f]['recall'] for f in features]
    f1_vals = [metrics_data[f]['f1'] for f in features]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar positions and width
    x = np.arange(len(features))
    width = 0.25
    
    # Professional color scheme
    colors = {
        'Precision': '#2E86AB',  # Blue
        'Recall': '#A23B72',      # Purple-pink
        'F1': '#F18F01'           # Orange
    }
    
    # Create bars
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', 
                   color=colors['Precision'], edgecolor='black', linewidth=0.8, zorder=3)
    bars2 = ax.bar(x, recall_vals, width, label='Recall', 
                   color=colors['Recall'], edgecolor='black', linewidth=0.8, zorder=3)
    bars3 = ax.bar(x + width, f1_vals, width, label='F1 Score', 
                   color=colors['F1'], edgecolor='black', linewidth=0.8, zorder=3)
    
    # Add value labels on bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(bars1, precision_vals)
    autolabel(bars2, recall_vals)
    autolabel(bars3, f1_vals)
    
    # Customize plot
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_xlabel('Feature', fontweight='bold')
    ax.set_title('Classification Performance Comparison Across Features', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=0, ha='center')
    ax.set_ylim(0, 1.1)
    
    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, ncol=3, bbox_to_anchor=(1.0, 1.12))
    
    # Baseline at 0.5
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, 
               alpha=0.7, zorder=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved to: {output_path}")
    
    return fig, ax


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("DSRED FEATURE COMPARISON ANALYSIS")
    print("="*70)
    print(f"\nConfig:")
    print(f"  Bright percentile: {BRIGHT_PERCENTILE}")
    print(f"  Area filter: [{AREA_MIN_PX}, {AREA_MAX_PX}] px")
    print(f"  Slices: {SLICES}")
    print(f"  Output directory: {OUT_DIR}\n")
    
    # Storage for pooled results across all slices
    all_results = {
        'pct_bright': [],
        'mean_intensity': [],
        'integrated_intensity': [],
        'gt_positive': []
    }
    
    # Process each slice
    for sl in SLICES:
        print(f"\n{'='*70}")
        print(f"Processing Slice {sl}...")
        print(f"{'='*70}")
        
        # Load data
        ds, mask = load_tiff_channels(TIFF_BY_SLICE[sl])
        print(f"  Loaded TIFF: shape={mask.shape}")
        
        # Load ground truth
        gt_labels = load_gt_positive_labels_from_fiji_csv(
            GT_CSV_BY_SLICE[sl], mask, neighbor_radius_px=NEIGHBOR_RADIUS_PX
        )
        print(f"  Ground truth: {len(gt_labels)} positive labels")
        
        # Compute features
        print(f"\n  Computing features...")
        
        # Feature 1: pct_bright
        df1, bright_thr = compute_pct_bright_feature(ds, mask, BRIGHT_PERCENTILE)
        print(f"    ✓ pct_bright (bright_thr={bright_thr:.2f})")
        
        # Feature 2: mean_intensity
        df2 = compute_mean_intensity_feature(ds, mask)
        print(f"    ✓ mean_intensity")
        
        # Feature 3: integrated_intensity
        df3 = compute_integrated_intensity_feature(ds, mask)
        print(f"    ✓ integrated_intensity")
        
        # Merge all features
        df = df1.merge(df2[['label', 'mean_intensity']], on='label', how='left')
        df = df.merge(df3[['label', 'integrated_intensity']], on='label', how='left')
        
        # Apply area filter
        df = apply_area_filter(df, AREA_MIN_PX, AREA_MAX_PX)
        print(f"\n  After area filter: {len(df)} ROIs")
        
        # Add ground truth column
        df['gt_positive'] = df['label'].astype(int).isin(gt_labels)
        n_pos = df['gt_positive'].sum()
        print(f"  Ground truth positives in filtered ROIs: {n_pos}/{len(df)}")
        
        # Accumulate for pooled analysis
        all_results['pct_bright'].extend(df['pct_bright'].tolist())
        all_results['mean_intensity'].extend(df['mean_intensity'].tolist())
        all_results['integrated_intensity'].extend(df['integrated_intensity'].tolist())
        all_results['gt_positive'].extend(df['gt_positive'].tolist())
        
        # Save per-slice table
        slice_out_csv = Path(OUT_DIR) / f"slice_{sl}_all_features.csv"
        df.to_csv(slice_out_csv, index=False)
        print(f"  ✓ Saved: {slice_out_csv.name}")
    
    # Convert pooled results to arrays
    print(f"\n{'='*70}")
    print("POOLED ANALYSIS (all slices combined)")
    print(f"{'='*70}")
    print(f"Total ROIs: {len(all_results['pct_bright'])}")
    print(f"Total positives: {sum(all_results['gt_positive'])}")
    
    # Evaluate each feature with Otsu
    results = {}
    
    print("\nEvaluating features with Otsu thresholding...\n")
    
    # Feature 1: pct_bright
    results['Fraction of\nbright pixels'] = classify_with_otsu_and_evaluate(
        all_results['pct_bright'], 
        all_results['gt_positive']
    )
    print("Feature 1: Fraction of bright pixels")
    print(f"  Otsu threshold: {results['Fraction of\\nbright pixels']['threshold']:.3f}%")
    print(f"  Precision: {results['Fraction of\\nbright pixels']['precision']:.3f}")
    print(f"  Recall:    {results['Fraction of\\nbright pixels']['recall']:.3f}")
    print(f"  F1:        {results['Fraction of\\nbright pixels']['f1']:.3f}")
    cm = results['Fraction of\\nbright pixels']['confusion_matrix']
    print(f"  TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}\n")
    
    # Feature 2: mean_intensity
    results['Mean intensity\ninside mask'] = classify_with_otsu_and_evaluate(
        all_results['mean_intensity'],
        all_results['gt_positive']
    )
    print("Feature 2: Mean intensity inside mask")
    print(f"  Otsu threshold: {results['Mean intensity\\ninside mask']['threshold']:.3f}")
    print(f"  Precision: {results['Mean intensity\\ninside mask']['precision']:.3f}")
    print(f"  Recall:    {results['Mean intensity\\ninside mask']['recall']:.3f}")
    print(f"  F1:        {results['Mean intensity\\ninside mask']['f1']:.3f}")
    cm = results['Mean intensity\\ninside mask']['confusion_matrix']
    print(f"  TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}\n")
    
    # Feature 3: integrated_intensity
    results['Integrated\nintensity'] = classify_with_otsu_and_evaluate(
        all_results['integrated_intensity'],
        all_results['gt_positive']
    )
    print("Feature 3: Integrated intensity")
    print(f"  Otsu threshold: {results['Integrated\\nintensity']['threshold']:.3f}")
    print(f"  Precision: {results['Integrated\\nintensity']['precision']:.3f}")
    print(f"  Recall:    {results['Integrated\\nintensity']['recall']:.3f}")
    print(f"  F1:        {results['Integrated\\nintensity']['f1']:.3f}")
    cm = results['Integrated\\nintensity']['confusion_matrix']
    print(f"  TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}\n")
    
    # Save results JSON
    results_json_path = Path(OUT_DIR) / "feature_comparison_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {results_json_path.name}")
    
    # Create comparison plot
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON PLOT")
    print(f"{'='*70}\n")
    
    plot_path = Path(OUT_DIR) / "feature_comparison_thesis.png"
    create_feature_comparison_plot(results, output_path=plot_path)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"\nFiles generated:")
    print(f"  - feature_comparison_thesis.png (main figure)")
    print(f"  - feature_comparison_results.json (detailed metrics)")
    print(f"  - slice_XXX_all_features.csv (per-slice data tables)")
    
    return results


if __name__ == "__main__":
    main()