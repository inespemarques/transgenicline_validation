#!/usr/bin/env python3
"""
COMPLETE FINAL PUBLICATION FIGURES - OPTIMIZED
================================================

OTIMIZAÇÕES:
1. Cache de features (não recalcular para mesmos parâmetros)
2. Redução de shells testadas (apenas as mais relevantes)
3. Percentiles pré-filtrados (recall constraint)
4. Paralelização opcional
"""

import os
import warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

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
OUT_DIR = os.path.join(BASE_RESULTS, "final_publication_figures_OPTIMIZED")
os.makedirs(OUT_DIR, exist_ok=True)

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

BEST_PERCENTILE = 85
BEST_SHELL = (-5, 3)
EXCLUDE_ZEROS = True
GLOBAL_OTSU = True

YOUR_METRICS = {
    'TP': 1494, 'FP': 348, 'FN': 275, 'TN': 3376,
    'precision': 0.811, 'recall': 0.845, 'f1': 0.827,
    'accuracy': 0.887, 'specificity': 0.907
}

FONT_SIZE = 16

# ⚡ OTIMIZAÇÃO: Apenas shells relevantes (reduzido de 24 para 12)
SHELL_DISTANCES_TO_TEST = [
    (-5, -5), (-3, -3), (-1, -1),  # Interior puro (3 amostras)
    (-5, 0), (-5, 2), (-5, 3), (-5, 5),  # Mista: int 5 + ext variável (4 amostras)
    (-3, 2), (-3, 3),  # Mista: int 3 + ext variável (2 amostras)
    (0, 1), (0, 3), (0, 5),  # Exterior puro (3 amostras)
]

# ⚡ OTIMIZAÇÃO: Apenas percentiles que podem dar recall >= 0.80
PERCENTILES_TO_TEST = [70, 75, 80, 85, 90, 95]  # 6 instead of 9


# =============================================================================
# IO + VORONOI (unchanged from corrected version)
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


def precompute_voronoi(label_slice):
    """CORRECTED: Uses Voronoi for exterior pixel assignment."""
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
        assigned[outside] = label_slice[idx[0], idx[1]][outside]  # ← VORONOI
    
    return dist_map, assigned


def extract_shell(dist_map, assigned, d_start, d_end):
    shell_mask = (dist_map >= float(d_start)) & (dist_map <= float(d_end))
    return np.where(shell_mask, assigned, 0)


def compute_bright_threshold(intensity, percentile, exclude_zeros):
    vals = intensity.ravel()
    if exclude_zeros:
        vals = vals[vals > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, percentile))


def compute_fbright_per_roi(intensity, shell, bright_threshold):
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
    return slices


# =============================================================================
# ⚡ OPTIMIZED: FEATURE CACHE
# =============================================================================

class FeatureCache:
    """Cache para evitar recalcular features com mesmos parâmetros."""
    
    def __init__(self):
        self.cache = {}
    
    def get_key(self, slice_id, d_start, d_end, percentile, exclude_zeros):
        return (slice_id, d_start, d_end, percentile, exclude_zeros)
    
    def get(self, slice_id, d_start, d_end, percentile, exclude_zeros):
        key = self.get_key(slice_id, d_start, d_end, percentile, exclude_zeros)
        return self.cache.get(key, None)
    
    def set(self, slice_id, d_start, d_end, percentile, exclude_zeros, features):
        key = self.get_key(slice_id, d_start, d_end, percentile, exclude_zeros)
        self.cache[key] = features
    
    def get_or_compute(self, slice_data, slice_id, d_start, d_end, percentile, exclude_zeros):
        """Get from cache or compute if not cached."""
        cached = self.get(slice_id, d_start, d_end, percentile, exclude_zeros)
        if cached is not None:
            return cached
        
        # Compute
        shell = extract_shell(slice_data['dist_map'], slice_data['assigned'], d_start, d_end)
        bright_thr = compute_bright_threshold(slice_data['intensity'], percentile, exclude_zeros)
        if bright_thr is None:
            features = {}
        else:
            features = compute_fbright_per_roi(slice_data['intensity'], shell, bright_thr)
        
        # Cache and return
        self.set(slice_id, d_start, d_end, percentile, exclude_zeros, features)
        return features


# Global cache instance
FEATURE_CACHE = FeatureCache()


# =============================================================================
# HELPER: Test shell configurations (OPTIMIZED)
# =============================================================================

def test_shell_with_percentile_cached(slices_data, d_start, d_end, percentile,
                                      exclude_zeros, global_otsu):
    """
    ⚡ OPTIMIZED: Uses cache to avoid recomputing features.
    """
    y_true_all = []
    y_pred_all = []
    
    if global_otsu:
        all_features = []
        all_gt = []
        
        for sk, sd in sorted(slices_data.items()):
            # ⚡ Use cache
            fbright = FEATURE_CACHE.get_or_compute(sd, sk, d_start, d_end, percentile, exclude_zeros)
            
            for lab, feat in fbright.items():
                all_features.append(feat)
                all_gt.append(1 if lab in sd['gt_labels'] else 0)
        
        if len(all_features) < 2:
            return None
        
        try:
            otsu_thr = float(threshold_otsu(np.array(all_features, dtype=np.float32)))
        except:
            return None
        
        y_true_all = all_gt
        y_pred_all = [1 if f >= otsu_thr else 0 for f in all_features]
    
    else:
        for sk, sd in sorted(slices_data.items()):
            # ⚡ Use cache
            fbright = FEATURE_CACHE.get_or_compute(sd, sk, d_start, d_end, percentile, exclude_zeros)
            
            if len(fbright) < 2:
                continue
            
            try:
                feat_vals = np.array(list(fbright.values()), dtype=np.float32)
                otsu_thr = float(threshold_otsu(feat_vals))
            except:
                continue
            
            for lab, feat in fbright.items():
                y_true_all.append(1 if lab in sd['gt_labels'] else 0)
                y_pred_all.append(1 if feat >= otsu_thr else 0)
    
    if not y_true_all:
        return None
    
    tp = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_all, y_pred_all) if yt == 1 and yp == 0)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


# =============================================================================
# FIGURES (unchanged except using cached function)
# =============================================================================

def create_confusion_matrix_exact_style(metrics, output_path):
    """Confusion matrix with EXACT style from reference image."""
    fs = FONT_SIZE
    
    cm = np.array([
        [metrics['TP'], metrics['FP']],
        [metrics['FN'], metrics['TN']]
    ])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors_list = ['#FFFFFF', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue_gradient', colors_list, N=n_bins)
    
    im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=cm.max())
    
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            total = cm[i, :].sum()
            percentage = (count / total * 100) if total > 0 else 0
            
            text_color = 'white' if count > cm.max() * 0.5 else 'black'
            
            ax.text(j, i, f'{count}\n({percentage:.1f}%)',
                   ha='center', va='center',
                   fontsize=fs + 4, fontweight='bold',
                   color=text_color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\npositive', 'Predicted\nnegative'],
                       fontsize=fs - 1, fontweight='bold')
    ax.set_yticklabels(['Actual\npositive', 'Actual\nnegative'],
                       fontsize=fs - 1, fontweight='bold')
    
    ax.tick_params(length=0)
    
    ax.set_title('P85', fontsize=fs + 2, fontweight='bold', pad=15,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                         edgecolor='black', linewidth=1.5))
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=fs - 2, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=fs - 3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Confusion matrix saved")


def create_distribution_plots(slices_data, output_path):
    """Distribution plots per slice + pooled."""
    fs = FONT_SIZE - 2
    d_start, d_end = BEST_SHELL
    
    slice_data = {}
    for sk, sd in sorted(slices_data.items()):
        # ⚡ Use cache
        fbright = FEATURE_CACHE.get_or_compute(sd, sk, d_start, d_end, BEST_PERCENTILE, EXCLUDE_ZEROS)
        
        pos_vals = [fbright[lab] for lab in sd['gt_labels'] if lab in fbright]
        neg_vals = [fbright[lab] for lab in (sd['all_labels'] - sd['gt_labels']) if lab in fbright]
        
        all_vals = np.array(pos_vals + neg_vals, dtype=np.float32)
        otsu_thr = float(threshold_otsu(all_vals)) if len(all_vals) > 1 else None
        
        slice_data[sk] = {
            'pos': pos_vals, 'neg': neg_vals, 
            'otsu': otsu_thr,
            'n_pos': len(pos_vals), 'n_neg': len(neg_vals)
        }
    
    all_pos = []
    all_neg = []
    for sk in slice_data:
        all_pos.extend(slice_data[sk]['pos'])
        all_neg.extend(slice_data[sk]['neg'])
    
    pooled_otsu = float(threshold_otsu(np.array(all_pos + all_neg, dtype=np.float32)))
    
    fig = plt.figure(figsize=(20, 4))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)
    
    slice_keys = sorted(slice_data.keys())
    
    for idx, sk in enumerate(slice_keys):
        ax = fig.add_subplot(gs[0, idx])
        
        data = slice_data[sk]
        bins = np.linspace(0, 100, 40)
        
        ax.hist(data['neg'], bins=bins, alpha=0.6, color='#6BAED6', 
               edgecolor='black', linewidth=0.5, label=f"In situ− (n={data['n_neg']})")
        ax.hist(data['pos'], bins=bins, alpha=0.6, color='#FD8D3C',
               edgecolor='black', linewidth=0.5, label=f"In situ+ (n={data['n_pos']})")
        
        if data['otsu']:
            ax.axvline(data['otsu'], color='red', linewidth=2.5, linestyle='--',
                      label=f"Otsu = {data['otsu']:.1f}")
        
        ax.set_xlabel('Bright-voxel fraction (%)', fontsize=fs, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Density', fontsize=fs, fontweight='bold')
        ax.set_title(f'Slice {sk}', fontsize=fs + 1, fontweight='bold')
        ax.legend(fontsize=fs - 4, frameon=True, loc='upper center')
        ax.tick_params(labelsize=fs - 3)
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_xlim(0, 100)
    
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0, 100, 40)
    
    ax.hist(all_neg, bins=bins, alpha=0.6, color='#6BAED6',
           edgecolor='black', linewidth=0.5, label=f"In situ− (n={len(all_neg)})")
    ax.hist(all_pos, bins=bins, alpha=0.6, color='#FD8D3C',
           edgecolor='black', linewidth=0.5, label=f"In situ+ (n={len(all_pos)})")
    
    ax.axvline(pooled_otsu, color='red', linewidth=2.5, linestyle='--',
              label=f"Otsu = {pooled_otsu:.1f}")
    
    ax.set_xlabel('Bright-voxel fraction (%)', fontsize=fs, fontweight='bold')
    ax.set_title('Pooled (all slices)', fontsize=fs + 1, fontweight='bold')
    ax.legend(fontsize=fs - 4, frameon=True, loc='upper center')
    ax.tick_params(labelsize=fs - 3)
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Distribution plots saved")


def create_shell_distance_plot(slices_data, output_path):
    """
    ⚡ OPTIMIZED: Performance vs shell distance.
    Uses cache and reduced shell/percentile space.
    """
    fs = FONT_SIZE - 2
    
    preproc_configs = [
        (True, True, "Exclude zeros + Global Otsu", '#DE8F05'),
        (True, False, "Exclude zeros + Per-block Otsu", '#029E73'),
        (False, True, "Include zeros + Global Otsu", '#0173B2'),
        (False, False, "Include zeros + Per-block Otsu", '#999999'),
    ]
    
    results_by_preproc = {}
    
    total_tests = len(preproc_configs) * len(SHELL_DISTANCES_TO_TEST) * len(PERCENTILES_TO_TEST)
    print(f"\n  Testing {len(SHELL_DISTANCES_TO_TEST)} shells × "
          f"{len(PERCENTILES_TO_TEST)} percentiles × "
          f"{len(preproc_configs)} configs = {total_tests} combinations...")
    
    test_count = 0
    for excl_zeros, glob_otsu, label, color in preproc_configs:
        shell_positions = []
        precisions = []
        recalls = []
        f1s = []
        
        for d_start, d_end in SHELL_DISTANCES_TO_TEST:
            best_p = None
            best_prec = -1
            best_config = None
            
            for p in PERCENTILES_TO_TEST:
                test_count += 1
                if test_count % 20 == 0:
                    print(f"    Progress: {test_count}/{total_tests} ({100*test_count/total_tests:.1f}%)")
                
                config = test_shell_with_percentile_cached(
                    slices_data, d_start, d_end, p, excl_zeros, glob_otsu
                )
                
                if config and config['recall'] >= 0.80 and config['precision'] > best_prec:
                    best_prec = config['precision']
                    best_p = p
                    best_config = config
            
            if best_config:
                if d_start == d_end:
                    pos = d_start
                else:
                    pos = (d_start + d_end) / 2
                
                shell_positions.append(pos)
                precisions.append(best_config['precision'])
                recalls.append(best_config['recall'])
                f1s.append(best_config['f1'])
        
        results_by_preproc[label] = {
            'positions': shell_positions,
            'precision': precisions,
            'recall': recalls,
            'f1': f1s,
            'color': color
        }
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    for label, data in results_by_preproc.items():
        if not data['positions']:
            continue
        
        ax1.plot(data['positions'], data['precision'], 'o-',
                linewidth=2.5, markersize=7, color=data['color'], label=label)
        ax2.plot(data['positions'], data['recall'], 'o-',
                linewidth=2.5, markersize=7, color=data['color'], label=label)
        ax3.plot(data['positions'], data['f1'], 'o-',
                linewidth=2.5, markersize=7, color=data['color'], label=label)
    
    for ax in [ax1, ax2, ax3]:
        ax.axvline(0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
        ax.axhline(0.80, color='gray', linewidth=1.5, linestyle='--', alpha=0.4)
        ax.set_xlabel('Shell distance (px)', fontsize=fs + 2, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=fs)
        
        ax.text(-4, 0.02, '← interior', fontsize=fs - 2, ha='center', 
               color='gray', style='italic')
        ax.text(4, 0.02, 'exterior →', fontsize=fs - 2, ha='center',
               color='gray', style='italic')
    
    ax1.set_ylabel('Precision', fontsize=fs + 2, fontweight='bold')
    ax1.set_title('Precision', fontsize=fs + 3, fontweight='bold')
    
    ax2.set_ylabel('Recall', fontsize=fs + 2, fontweight='bold')
    ax2.set_title('Recall', fontsize=fs + 3, fontweight='bold')
    ax2.legend(fontsize=fs - 4, frameon=True, loc='lower left')
    
    ax3.set_ylabel('F1 Score', fontsize=fs + 2, fontweight='bold')
    ax3.set_title('F1 Score', fontsize=fs + 3, fontweight='bold')
    
    fig.suptitle('Performance vs Shell Distance (Voronoi + Optimized)\n(best percentile per shell, recall ≥ 0.80 constraint)',
                fontsize=fs + 4, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Shell distance plot saved")


def create_roc_and_metrics_panel(slices_data, output_path):
    """ROC + metrics panel (using cache)."""
    fs = FONT_SIZE - 1
    d_start, d_end = BEST_SHELL
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    
    # ========== LEFT: ROC CURVES ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Feature 1: Bright-pixels at P80%
    y_true_bright = []
    y_score_bright = []
    
    for sk, sd in sorted(slices_data.items()):
        # ⚡ Use cache
        fbright = FEATURE_CACHE.get_or_compute(sd, sk, d_start, d_end, 80, EXCLUDE_ZEROS)
        
        for lab, score in fbright.items():
            y_true_bright.append(1 if lab in sd['gt_labels'] else 0)
            y_score_bright.append(score)
    
    if len(y_true_bright) >= 2 and len(np.unique(y_true_bright)) == 2:
        fpr, tpr, _ = roc_curve(y_true_bright, y_score_bright)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=3, color='#0173B2',
                label=f'Bright-pixels\nP80% AUC {roc_auc:.3f}')
    
    # Feature 2: Mean intensity
    y_true_mean = []
    y_score_mean = []
    
    for sk, sd in sorted(slices_data.items()):
        shell = extract_shell(sd['dist_map'], sd['assigned'], d_start, d_end)
        
        shell_i64 = shell.astype(np.int64)
        valid = shell_i64 > 0
        if not valid.any():
            continue
        
        labs = shell_i64[valid]
        ivals = sd['intensity'][valid]
        maxlab = int(labs.max())
        counts = np.bincount(labs, minlength=maxlab + 1)
        sum_int = np.bincount(labs, weights=ivals.astype(np.float64), minlength=maxlab + 1)
        labels_present = np.nonzero(counts)[0]
        labels_present = labels_present[labels_present > 0]
        
        for lab in labels_present:
            mean_int = sum_int[lab] / max(counts[lab], 1)
            y_true_mean.append(1 if lab in sd['gt_labels'] else 0)
            y_score_mean.append(mean_int)
    
    if len(y_true_mean) >= 2 and len(np.unique(y_true_mean)) == 2:
        fpr, tpr, _ = roc_curve(y_true_mean, y_score_mean)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=3, color='#DE8F05',
                label=f'Mean intensity\nAUC {roc_auc:.3f}')
    
    # Feature 3: Integrated intensity
    y_true_int = []
    y_score_int = []
    
    for sk, sd in sorted(slices_data.items()):
        shell = extract_shell(sd['dist_map'], sd['assigned'], d_start, d_end)
        
        shell_i64 = shell.astype(np.int64)
        valid = shell_i64 > 0
        if not valid.any():
            continue
        
        labs = shell_i64[valid]
        ivals = sd['intensity'][valid]
        maxlab = int(labs.max())
        counts = np.bincount(labs, minlength=maxlab + 1)
        sum_int = np.bincount(labs, weights=ivals.astype(np.float64), minlength=maxlab + 1)
        labels_present = np.nonzero(counts)[0]
        labels_present = labels_present[labels_present > 0]
        
        for lab in labels_present:
            y_true_int.append(1 if lab in sd['gt_labels'] else 0)
            y_score_int.append(sum_int[lab])
    
    if len(y_true_int) >= 2 and len(np.unique(y_true_int)) == 2:
        fpr, tpr, _ = roc_curve(y_true_int, y_score_int)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=3, color='#029E73',
                label=f'Integrated intensity\nAUC {roc_auc:.3f}')
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    
    ax1.set_xlabel('False positive rate', fontsize=fs + 2, fontweight='bold')
    ax1.set_ylabel('True positive rate', fontsize=fs + 2, fontweight='bold')
    ax1.legend(fontsize=fs - 2, frameon=True, loc='lower right')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=fs)
    
    # ========== RIGHT: METRICS VS PERCENTILE ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 99]
    
    metric_data = {m: {'mean': [], 'std': []} for m in ['precision', 'recall', 'f1']}
    
    for p in percentiles:
        slice_metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for sk, sd in sorted(slices_data.items()):
            # ⚡ Use cache
            fbright = FEATURE_CACHE.get_or_compute(sd, sk, d_start, d_end, p, EXCLUDE_ZEROS)
            
            if len(fbright) < 2:
                continue
            
            try:
                feat_vals = np.array(list(fbright.values()), dtype=np.float32)
                otsu_thr = float(threshold_otsu(feat_vals))
            except:
                continue
            
            y_true = [1 if lab in sd['gt_labels'] else 0 for lab in fbright.keys()]
            y_pred = [1 if fbright[lab] >= otsu_thr else 0 for lab in fbright.keys()]
            
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
            
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            
            slice_metrics['precision'].append(prec)
            slice_metrics['recall'].append(rec)
            slice_metrics['f1'].append(f1)
        
        for m in ['precision', 'recall', 'f1']:
            vals = slice_metrics[m]
            metric_data[m]['mean'].append(np.mean(vals) if vals else np.nan)
            metric_data[m]['std'].append(np.std(vals) if vals else 0)
    
    colors_m = {'f1': '#0173B2', 'precision': '#DE8F05', 'recall': '#029E73'}
    markers_m = {'f1': 'o', 'precision': 's', 'recall': '^'}
    labels_m = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall'}
    
    for m in ['f1', 'precision', 'recall']:
        means = np.array(metric_data[m]['mean'])
        stds = np.array(metric_data[m]['std'])
        ax2.errorbar(percentiles, means, yerr=stds,
                    marker=markers_m[m], linewidth=2.5, markersize=9,
                    capsize=6, capthick=2, color=colors_m[m],
                    label=labels_m[m])
    
    ax2.axvline(x=BEST_PERCENTILE, color='gray', linewidth=2.5, linestyle=':',
               alpha=0.7, zorder=0)
    ax2.text(BEST_PERCENTILE + 0.8, 0.03, f'P{BEST_PERCENTILE}',
            fontsize=fs - 2, color='gray', fontweight='bold')
    
    ax2.set_xlabel('Percentile threshold (%)', fontsize=fs + 2, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=fs + 2, fontweight='bold')
    ax2.legend(fontsize=fs, frameon=True, loc='center left')
    ax2.set_xlim(percentiles[0] - 2, percentiles[-1] + 2)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=fs)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ ROC + metrics panel saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COMPLETE FINAL PUBLICATION FIGURES - OPTIMIZED")
    print("=" * 80)
    print(f"Output: {OUT_DIR}")
    print(f"Best method: P{BEST_PERCENTILE}, Shell {BEST_SHELL}, Excl.zeros, Global Otsu")
    print()
    print("⚡ OPTIMIZATIONS APPLIED:")
    print(f"  • Feature caching (avoid recomputing same parameters)")
    print(f"  • Reduced shells: {len(SHELL_DISTANCES_TO_TEST)} (was 24)")
    print(f"  • Reduced percentiles: {len(PERCENTILES_TO_TEST)} (was 9)")
    print(f"  • Expected speedup: ~10x faster")
    print()
    
    # Figure 1: Confusion Matrix
    print("[1/4] Creating confusion matrix...")
    cm_path = os.path.join(OUT_DIR, "fig1_confusion_matrix_P85.png")
    create_confusion_matrix_exact_style(YOUR_METRICS, cm_path)
    
    # Load data
    print("\n[2/4] Loading slices with Voronoi...")
    slices_data = load_all_slices()
    
    # Figure 2: Distribution plots
    print("\n[3/4] Creating distribution plots...")
    dist_path = os.path.join(OUT_DIR, "fig2_distribution_plots.png")
    create_distribution_plots(slices_data, dist_path)
    
    # Figure 6: Shell distance plot (OPTIMIZED)
    print("\n[4/4] Creating shell distance performance plot (OPTIMIZED)...")
    shell_dist_path = os.path.join(OUT_DIR, "fig6_shell_distance_performance.png")
    create_shell_distance_plot(slices_data, shell_dist_path)
    
    # Figure 7: ROC + Metrics dual panel
    print("\n[5/4] Creating ROC + metrics dual panel...")
    roc_metrics_path = os.path.join(OUT_DIR, "fig7_roc_and_metrics.png")
    create_roc_and_metrics_panel(slices_data, roc_metrics_path)
    
    print("\n" + "=" * 80)
    print("✅ ALL FIGURES COMPLETE!")
    print("=" * 80)
    print()
    print(f"Cache stats: {len(FEATURE_CACHE.cache)} feature computations cached")
    print()
    print("CREATED FILES:")
    print("  1. fig1_confusion_matrix_P85.png")
    print("  2. fig2_distribution_plots.png")
    print("  3. fig6_shell_distance_performance.png")
    print("  4. fig7_roc_and_metrics.png")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()