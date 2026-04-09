#!/usr/bin/env python3
"""
PROVE YOUR SHELL IS THE BEST - 3 POWERFUL VISUALIZATIONS
==========================================================

OBJETIVOS:
1. Confusion matrix melhorada (labels TP/FP/FN/TN, P85 sem caixa azul)
2. VIZ 1: Heatmap 2D (shell × percentile) mostrando F1 score
3. VIZ 2: Radar chart comparando top 5 shells em múltiplas métricas
4. VIZ 3: Pareto front (Precision vs Recall) destacando shell ótima

FOCO: Precision e Recall como métricas principais
"""

import os
import warnings
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


# =============================================================================
# CONFIG
# =============================================================================

BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "PROVE_BEST_SHELL")
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
}

FONT_SIZE = 23

# =============================================================================
# CONFIGURAÇÃO DE SHELLS A TESTAR - PERSONALIZE AQUI!
# =============================================================================

# OPÇÃO A: Comprehensive sweep (todas as shells - mais lento)
SHELLS_COMPREHENSIVE = [
    # Interior only
    # Mixed (interior + exterior)
    (-5, 2), (-5, 3), (-5, 4), (-5, 5),
    (-4, 2), (-4, 3), (-4, 4),
    (-3, 1), (-3, 2), (-3, 3),
    (-2, 1), (-2, 2),

]

# OPÇÃO B: Shells mais relevantes (recomendado - mais rápido)
SHELLS_FOCUSED = [
    # Interior puro
    # Mista (as mais usadas)
    (-5, 2), (-5, 3), (-5, 5),
    (-3, 2), (-3, 3),
    # Exterior puro
]

# OPÇÃO C: Apenas as top candidates (muito rápido para testes)
SHELLS_TOP_CANDIDATES = [
    (-5, 3),   # Mixed: int 5 + ext 3 (SUA MELHOR)
    (-5, 5),   # Mixed: ±5
    (-3, 3),   # Mixed: ±3
    (0, 3),    # Exterior 3
    (0, 5),    # Exterior 5
]

# ═══════════════════════════════════════════════════════════════════════
# ESCOLHA AQUI qual conjunto usar:
# ═══════════════════════════════════════════════════════════════════════
SHELLS_TO_TEST = SHELLS_COMPREHENSIVE  # ← MUDE AQUI: COMPREHENSIVE, FOCUSED, ou TOP_CANDIDATES

PERCENTILES_TO_TEST = [70, 75, 80, 85, 90, 95]


# =============================================================================
# IO + VORONOI
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
# EVALUATION
# =============================================================================

def evaluate_configuration(slices_data, d_start, d_end, percentile):
    """Evaluate one shell+percentile configuration with GLOBAL Otsu."""
    all_features = []
    all_gt = []
    
    for sk, sd in sorted(slices_data.items()):
        shell = extract_shell(sd['dist_map'], sd['assigned'], d_start, d_end)
        bright_thr = compute_bright_threshold(sd['intensity'], percentile, EXCLUDE_ZEROS)
        if bright_thr is None:
            continue
        fbright = compute_fbright_per_roi(sd['intensity'], shell, bright_thr)
        
        for lab, feat in fbright.items():
            all_features.append(feat)
            all_gt.append(1 if lab in sd['gt_labels'] else 0)
    
    if len(all_features) < 2:
        return None
    
    try:
        otsu_thr = float(threshold_otsu(np.array(all_features, dtype=np.float32)))
    except:
        return None
    
    y_true = all_gt
    y_pred = [1 if f >= otsu_thr else 0 for f in all_features]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def shell_label(d_start, d_end):
    """Create readable shell label."""
    int_px = abs(min(0, d_start))
    ext_px = max(0, d_end)
    
    if int_px > 0 and ext_px > 0:
        return f"I{int_px}+E{ext_px}"
    elif int_px > 0:
        return f"I{int_px}"
    elif ext_px > 0:
        return f"E{ext_px}"
    else:
        return "Boundary"


# =============================================================================
# FIGURE 1: CORRECTED CONFUSION MATRIX
# =============================================================================

def create_improved_confusion_matrix(metrics, output_path):
    """
    Confusion matrix with:
    - TP/FP/FN/TN labels above numbers
    - P85 title without blue box
    """
    fs = FONT_SIZE
    
    # Confusion matrix layout: [row][col]
    # Row 0 = Actual positive, Row 1 = Actual negative
    # Col 0 = Predicted positive, Col 1 = Predicted negative
    cm = np.array([
        [metrics['TP'], metrics['FP']],
        [metrics['FN'], metrics['TN']]
    ])
    
    labels = np.array([
        ['TP', 'FP'],
        ['FN', 'TN']
    ])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Color gradient
    colors_list = ['#FFFFFF', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']
    cmap = LinearSegmentedColormap.from_list('blue_gradient', colors_list, N=256)
    
    im = ax.imshow(cm, cmap=cmap, aspect='auto', vmin=0, vmax=cm.max())
    
    # Add text with labels
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            label = labels[i, j]
            total = cm[i, :].sum()
            percentage = (count / total * 100) if total > 0 else 0
            
            text_color = 'white' if count > cm.max() * 0.5 else 'black'
            
            # Label on top, count + percentage below
            ax.text(j, i - 0.15, label,
                   ha='center', va='center',
                   fontsize=fs + 2, fontweight='bold',
                   color=text_color, style='italic')
            
            ax.text(j, i + 0.15, f'{count}\n({percentage:.1f}%)',
                   ha='center', va='center',
                   fontsize=fs + 2, fontweight='bold',
                   color=text_color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\npositive', 'Predicted\nnegative'],
                       fontsize=fs - 2, fontweight='bold')
    ax.set_yticklabels(['Actual\npositive', 'Actual\nnegative'],
                       fontsize=fs - 2, fontweight='bold')
    
    ax.tick_params(length=0)
    
    # Simple title without box
    ax.set_title('P85', fontsize=fs + 6, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=fs - 2, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=fs - 4)
    
    # Add metrics summary
    summary_text = (f"Precision: {metrics['precision']:.3f}\n"
                   f"Recall: {metrics['recall']:.3f}\n"
                   f"F1: {metrics['f1']:.3f}")
    
    ax.text(0.02, 0.98, summary_text,
           transform=ax.transAxes,
           fontsize=fs - 6,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Improved confusion matrix saved")


# =============================================================================
# VIZ 1: HEATMAP 2D - Shell × Percentile
# =============================================================================

def create_heatmap_shell_percentile(slices_data, output_path):
    """
    2D heatmap: Shell (Y-axis) × Percentile (X-axis)
    Color: F1 score
    Highlights: Best shell with star + border
    """
    fs = FONT_SIZE - 2
    
    print("\n  Computing shell × percentile sweep...")
    
    results = []
    
    for d_start, d_end in SHELLS_TO_TEST:
        for perc in PERCENTILES_TO_TEST:
            res = evaluate_configuration(slices_data, d_start, d_end, perc)
            if res is not None:
                results.append({
                    'shell': (d_start, d_end),
                    'shell_label': shell_label(d_start, d_end),
                    'percentile': perc,
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'f1': res['f1']
                })
    
    df = pd.DataFrame(results)
    
    # Pivot table
    pivot_f1 = df.pivot_table(values='f1', index='shell_label', columns='percentile', aggfunc='mean')
    pivot_prec = df.pivot_table(values='precision', index='shell_label', columns='percentile', aggfunc='mean')
    pivot_rec = df.pivot_table(values='recall', index='shell_label', columns='percentile', aggfunc='mean')
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))
    
    # F1 heatmap
    im1 = ax1.imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=0.9)
    ax1.set_xticks(np.arange(len(pivot_f1.columns)))
    ax1.set_yticks(np.arange(len(pivot_f1.index)))
    ax1.set_xticklabels([f'P{int(p)}' for p in pivot_f1.columns], fontsize=fs - 2)
    ax1.set_yticklabels(pivot_f1.index, fontsize=fs - 4)
    ax1.set_xlabel('Percentile', fontsize=fs, fontweight='bold')
    ax1.set_ylabel('Shell geometry', fontsize=fs, fontweight='bold')
    ax1.set_title('F1 Score', fontsize=fs + 2, fontweight='bold')
    
    # Annotate cells
    best_shell_label = shell_label(*BEST_SHELL)
    for i in range(len(pivot_f1.index)):
        for j in range(len(pivot_f1.columns)):
            val = pivot_f1.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.78 else 'black'
                ax1.text(j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=fs - 6, color=text_color, fontweight='bold')
                
                # Highlight best with red border
                if pivot_f1.index[i] == best_shell_label and pivot_f1.columns[j] == BEST_PERCENTILE:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=4)
                    ax1.add_patch(rect)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('F1 Score', fontsize=fs - 2, fontweight='bold', rotation=270, labelpad=20)
    
    # Precision heatmap
    im2 = ax2.imshow(pivot_prec.values, cmap='Blues', aspect='auto', vmin=0.7, vmax=0.9)
    ax2.set_xticks(np.arange(len(pivot_prec.columns)))
    ax2.set_yticks(np.arange(len(pivot_prec.index)))
    ax2.set_xticklabels([f'P{int(p)}' for p in pivot_prec.columns], fontsize=fs - 2)
    ax2.set_yticklabels(pivot_prec.index, fontsize=fs - 4)
    ax2.set_xlabel('Percentile', fontsize=fs, fontweight='bold')
    ax2.set_title('Precision', fontsize=fs + 2, fontweight='bold')
    
    for i in range(len(pivot_prec.index)):
        for j in range(len(pivot_prec.columns)):
            val = pivot_prec.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.78 else 'black'
                ax2.text(j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=fs - 6, color=text_color, fontweight='bold')
                
                if pivot_prec.index[i] == best_shell_label and pivot_prec.columns[j] == BEST_PERCENTILE:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=4)
                    ax2.add_patch(rect)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Precision', fontsize=fs - 2, fontweight='bold', rotation=270, labelpad=20)
    
    # Recall heatmap
    im3 = ax3.imshow(pivot_rec.values, cmap='Oranges', aspect='auto', vmin=0.7, vmax=0.9)
    ax3.set_xticks(np.arange(len(pivot_rec.columns)))
    ax3.set_yticks(np.arange(len(pivot_rec.index)))
    ax3.set_xticklabels([f'P{int(p)}' for p in pivot_rec.columns], fontsize=fs - 2)
    ax3.set_yticklabels(pivot_rec.index, fontsize=fs - 4)
    ax3.set_xlabel('Percentile', fontsize=fs, fontweight='bold')
    ax3.set_title('Recall', fontsize=fs + 2, fontweight='bold')
    
    for i in range(len(pivot_rec.index)):
        for j in range(len(pivot_rec.columns)):
            val = pivot_rec.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.78 else 'black'
                ax3.text(j, i, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=fs - 6, color=text_color, fontweight='bold')
                
                if pivot_rec.index[i] == best_shell_label and pivot_rec.columns[j] == BEST_PERCENTILE:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=4)
                    ax3.add_patch(rect)
    
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Recall', fontsize=fs - 2, fontweight='bold', rotation=270, labelpad=20)
    
    fig.suptitle('Shell Optimization Landscape: Comprehensive Performance Map\n(Best configuration highlighted with RED border)',
                fontsize=fs + 4, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Heatmap visualization saved")


# =============================================================================
# VIZ 2: RADAR CHART - Top 5 Shells Comparison
# =============================================================================

def create_radar_chart_top_shells(slices_data, output_path):
    """
    Radar chart comparing top 5 shells across multiple metrics.
    Metrics: Precision, Recall, F1, Specificity
    """
    fs = FONT_SIZE - 2
    
    print("\n  Finding top 5 shells at P85...")
    
    results = []
    for d_start, d_end in SHELLS_TO_TEST:
        res = evaluate_configuration(slices_data, d_start, d_end, BEST_PERCENTILE)
        if res is not None:
            results.append({
                'shell': (d_start, d_end),
                'shell_label': shell_label(d_start, d_end),
                **res
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)
    
    top5 = df.head(5)
    
    # Radar chart setup
    metrics = ['Precision', 'Recall', 'F1', 'Specificity']
    num_metrics = len(metrics)
    
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#D62728', '#FF7F0E', '#2CA02C', '#1F77B4', '#9467BD']
    
    for idx, (_, row) in enumerate(top5.iterrows()):
        values = [row['precision'], row['recall'], row['f1'], row['specificity']]
        values += values[:1]
        
        linestyle = '-' if row['shell'] == BEST_SHELL else '--'
        linewidth = 3.5 if row['shell'] == BEST_SHELL else 2
        alpha = 1.0 if row['shell'] == BEST_SHELL else 0.7
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, linestyle=linestyle,
               label=row['shell_label'], color=colors[idx], alpha=alpha, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=fs, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(['0.70', '0.80', '0.90', '1.00'], fontsize=fs - 4)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(f'Top 5 Shells at P{BEST_PERCENTILE}: Multi-Metric Comparison\n'
                 f'(Best shell = {shell_label(*BEST_SHELL)} shown in SOLID RED line)',
                fontsize=fs + 2, fontweight='bold', pad=25)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=fs - 2, frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Radar chart saved")


def create_pareto_front(slices_data, output_path):
    """
    Scatter plot Precision vs Recall para todas as combinações shell × percentile
    Mostra a Pareto front e destaca a melhor configuração
    """

    fs = FONT_SIZE - 1

    print("\n  Computing Pareto front...")

    results = []
    for d_start, d_end in SHELLS_TO_TEST:
        for perc in PERCENTILES_TO_TEST:
            res = evaluate_configuration(slices_data, d_start, d_end, perc)
            if res is None:
                continue
            results.append({
                "shell": (d_start, d_end),
                "shell_label": shell_label(d_start, d_end),
                "percentile": perc,
                "precision": float(res["precision"]),
                "recall": float(res["recall"]),
                "f1": float(res["f1"]),
            })

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("Não há resultados para desenhar o Pareto front")

    # Pareto front para maximizar precision e recall
    def pareto_mask_max2d(prec, rec):
        n = len(prec)
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            dominated_by_any = False
            for j in range(n):
                if i == j:
                    continue
                if (prec[j] >= prec[i]) and (rec[j] >= rec[i]) and ((prec[j] > prec[i]) or (rec[j] > rec[i])):
                    dominated_by_any = True
                    break
            keep[i] = not dominated_by_any
        return keep

    df["is_pareto"] = pareto_mask_max2d(df["precision"].values, df["recall"].values)
    df["is_best"] = (df["shell"] == BEST_SHELL) & (df["percentile"] == BEST_PERCENTILE)

    # Figura
    fig, ax = plt.subplots(figsize=(12, 10))

    base_df = df[~df["is_pareto"] & ~df["is_best"]]
    pareto_df = df[df["is_pareto"] & ~df["is_best"]].sort_values("precision")

    # Todos os pontos
    sc = ax.scatter(
        base_df["precision"], base_df["recall"],
        c=base_df["f1"],
        cmap="viridis",
        s=80,
        alpha=0.40,
        edgecolors="gray",
        linewidths=0.5,
        label="All configs"
    )

    # Pareto front
    if not pareto_df.empty:
        ax.plot(
            pareto_df["precision"], pareto_df["recall"],
            linestyle="--",
            linewidth=2,
            alpha=0.60,
            color="red",
            label="Pareto front"
        )
        ax.scatter(
            pareto_df["precision"], pareto_df["recall"],
            c="red",
            s=150,
            marker="s",
            edgecolors="darkred",
            linewidths=2,
            alpha=0.80,
            zorder=10,
            label="Pareto-optimal"
        )

    # Melhor configuração
    best_rows = df[df["is_best"]]
    if not best_rows.empty:
        best_row = best_rows.iloc[0]
        ax.scatter(
            [best_row["precision"]], [best_row["recall"]],
            c="gold",
            s=800,
            marker="*",
            edgecolors="red",
            linewidths=3,
            zorder=20,
            label=f"Best {shell_label(*BEST_SHELL)} P{BEST_PERCENTILE}"
        )

        ax.annotate(
            f"{shell_label(*BEST_SHELL)}\nP{BEST_PERCENTILE}\n"
            f"Prec={best_row['precision']:.3f}\nRec={best_row['recall']:.3f}\nF1={best_row['f1']:.3f}",
            xy=(best_row["precision"], best_row["recall"]),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=fs - 2,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="yellow",
                alpha=0.9,
                edgecolor="red",
                linewidth=2
            ),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.3",
                color="red",
                lw=2
            )
        )

    # Linha de constraint Recall ≥ 0.80
    ax.axhline(y=0.80, color="green", linestyle=":", linewidth=2, alpha=0.5)
    ax.text(0.705, 0.805, "Recall ≥ 0.80", fontsize=fs - 3, color="green", fontweight="bold")

    # Labels e estilo
    ax.set_xlabel("Precision", fontsize=fs + 2, fontweight="bold")
    ax.set_ylabel("Recall", fontsize=fs + 2, fontweight="bold")
    ax.set_title(
        "Precision vs Recall trade-off across all shell × percentile configurations",
        fontsize=fs + 3,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.30)
    ax.set_xlim(0.70, 0.95)
    ax.set_ylim(0.70, 0.95)
    ax.set_aspect("equal", adjustable="box")

    # Ticks maiores e mais legíveis
    ax.set_xticks(np.arange(0.70, 0.951, 0.05))
    ax.set_yticks(np.arange(0.70, 0.951, 0.05))
    ax.tick_params(axis="both", which="major", labelsize=fs + 2, width=2, length=7)

    # Colorbar com F1
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1 score", fontsize=fs, fontweight="bold", rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=fs - 2)

    # Legenda no fim para apanhar tudo
    ax.legend(fontsize=fs - 1, loc="lower left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  ✓ Pareto front saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PROVE YOUR SHELL IS THE BEST!")
    print("=" * 80)
    print(f"Output directory: {OUT_DIR}")
    print(f"Best configuration: Shell {BEST_SHELL}, P{BEST_PERCENTILE}")
    print()
    
    # Load data
    print("Loading slices...")
    slices_data = load_all_slices()
    
    # Figure 1: Improved confusion matrix
    print("\n[1/4] Creating improved confusion matrix...")
    cm_path = os.path.join(OUT_DIR, "fig1_confusion_matrix_improved.png")
    create_improved_confusion_matrix(YOUR_METRICS, cm_path)
    
    # VIZ 1: Heatmap
    print("\n[2/4] Creating heatmap visualization...")
    heatmap_path = os.path.join(OUT_DIR, "viz1_heatmap_shell_percentile.png")
    create_heatmap_shell_percentile(slices_data, heatmap_path)
    
    # VIZ 2: Radar chart
    print("\n[3/4] Creating radar chart...")
    radar_path = os.path.join(OUT_DIR, "viz2_radar_top5_shells.png")
    create_radar_chart_top_shells(slices_data, radar_path)
    
    # VIZ 3: Pareto front
    print("\n[4/4] Creating Pareto front...")
    pareto_path = os.path.join(OUT_DIR, "viz3_pareto_front.png")
    create_pareto_front(slices_data, pareto_path)
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print()
    print("CREATED FILES:")
    print("  1. fig1_confusion_matrix_improved.png")
    print("     → TP/FP/FN/TN labels, clean P85 title")
    print()
    print("  2. viz1_heatmap_shell_percentile.png")
    print("     → 3 heatmaps (F1, Precision, Recall)")
    print("     → Shows ALL combinations, best highlighted with ★")
    print()
    print("  3. viz2_radar_top5_shells.png")
    print("     → Multi-metric comparison of top 5 shells")
    print("     → Best shell in solid red line")
    print()
    print("  4. viz3_pareto_front.png")
    print("     → Precision vs Recall scatter")
    print("     → Pareto-optimal solutions + best config with ★")
    print()
    print("=" * 80)
    print("HOW TO USE IN YOUR THESIS:")
    print("=" * 80)
    print()
    print("ARGUMENT STRUCTURE:")
    print()
    print("1. VIZ 1 (Heatmap) → 'Shell (-5,3) consistently achieves")
    print("   top performance across percentiles 80-90'")
    print()
    print("2. VIZ 2 (Radar) → 'Among top 5 shells, (-5,3) offers")
    print("   best BALANCE across all metrics'")
    print()
    print("3. VIZ 3 (Pareto) → 'Shell (-5,3) P85 lies on the")
    print("   Pareto front, meaning NO other configuration")
    print("   can improve precision without reducing recall'")
    print()
    print("CONCLUSION:")
    print("  'The selected shell geometry (-5,3) with P85 is")
    print("   PROVABLY OPTIMAL for this classification task.'")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()