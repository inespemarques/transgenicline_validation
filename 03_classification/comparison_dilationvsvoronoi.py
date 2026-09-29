#!/usr/bin/env python3
"""
In situ classification - FOCUSED Voronoi vs Dilation Comparison

FAST VERSION: Tests only specific, relevant shell configurations
instead of exhaustive grid search.

Focus: Show degradation as exterior distance increases (1, 2, 5, 10 px)
"""

import os
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 20


# =============================================================================
# CONFIG
# =============================================================================

BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "insitu_FOCUSED_COMPARISON")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TIFF_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7tif.tif"),
    # ONLY slice 110 for speed (1000 nuclei vs 4000 total)
}

GT_CSV_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results_insituslice110fish7.csv"),
}

AREA_MIN_PX = 400
AREA_MAX_PX = 6000

# FOCUSED: Only test EXTERIOR shells (where difference is most visible!)
# OPTIMIZED for reasonable runtime with 1000 nuclei: 3 shells × 3 percentiles
FOCUSED_SHELLS = [
    # (start, end) - ONLY pure exterior regions
    (0, 1),    # 1 pixel exterior - minimal difference
    (0, 3),
    (0, 7),       # 5 pixels exterior - moderate difference
    (0, 10)   # 10 pixels exterior - maximum difference
]

# Fewer percentiles for speed (most important ones)
PERCENTILES = [85]

FEATURES = ["pct_bright"]

MIN_RECALL = 0.80

PERCENTILE_EXCLUDE_ZEROS = True


# =============================================================================
# IO AND GT
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

    pos_labels = {int(label_slice[y, x]) for x, y in zip(xs, ys) if int(label_slice[y, x]) > 0}
    return pos_labels


# =============================================================================
# VORONOI METHOD
# =============================================================================

def precompute_voronoi_signed_distance(label_slice):
    binary = label_slice > 0
    if not binary.any():
        H, W = label_slice.shape
        dist_map = np.zeros((H, W), dtype=np.float32)
        assigned = np.zeros((H, W), dtype=label_slice.dtype)
        return dist_map, assigned

    dist_out, idx = distance_transform_edt(~binary, return_indices=True)
    dist_in = distance_transform_edt(binary)
    dist_map = np.where(binary, -dist_in, dist_out).astype(np.float32)

    assigned = label_slice.copy()
    outside = ~binary
    if outside.any():
        assigned[outside] = label_slice[idx[0], idx[1]][outside]

    return dist_map, assigned

def create_shell_voronoi(label_slice, dist_map, assigned, start_dist, end_dist):
    shell_mask = (dist_map >= float(start_dist)) & (dist_map <= float(end_dist))
    shell = np.where(shell_mask, assigned, 0).astype(label_slice.dtype)
    return shell


# =============================================================================
# DILATION METHOD
# =============================================================================

def precompute_dilation_signed_distance(label_slice):
    binary = label_slice > 0
    if not binary.any():
        H, W = label_slice.shape
        return np.zeros((H, W), dtype=np.float32)

    dist_out = distance_transform_edt(~binary)
    dist_in = distance_transform_edt(binary)
    return np.where(binary, -dist_in, dist_out).astype(np.float32)

def create_shell_dilation(label_slice, start_dist, end_dist):
    """
    Create shell with TRUE PIXEL OVERLAP (different from Voronoi!).
    
    Key difference:
    - Voronoi: each exterior pixel belongs to EXACTLY ONE nucleus  
    - This method: exterior pixels can belong to MULTIPLE nuclei
    
    For each nucleus, we dilate independently. Pixels in overlapping regions
    contribute to BOTH nuclei's features - this inflates metrics!
    
    Returns: dictionary mapping label -> array of (y,x) coordinates
    """
    import time
    
    # Get all unique labels
    all_labels = np.unique(label_slice)
    all_labels = all_labels[all_labels > 0]
    
    if len(all_labels) == 0:
        return {}
    
    # For each nucleus, track which pixels are in its shell
    nucleus_shells = {}
    
    # Progress tracking
    n_nuclei = len(all_labels)
    start_time = time.time()
    
    # Process each nucleus independently
    for idx, lbl in enumerate(all_labels):
        # Progress every 100 nuclei
        if idx > 0 and idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining = (n_nuclei - idx) / rate
            print(f"      Dilation progress: {idx}/{n_nuclei} nuclei ({idx/n_nuclei*100:.1f}%) - ETA: {remaining/60:.1f} min")
        
        # Mask for this nucleus only
        nucleus_mask = (label_slice == lbl)
        
        # Distance from THIS nucleus (independent of others!)
        dist_inside = distance_transform_edt(nucleus_mask)
        dist_outside = distance_transform_edt(~nucleus_mask)
        signed_dist = np.where(nucleus_mask, -dist_inside, dist_outside)
        
        # Shell for THIS nucleus
        shell_mask = (signed_dist >= float(start_dist)) & (signed_dist <= float(end_dist))
        
        # Store coordinates (allows overlap between nuclei!)
        shell_coords = np.argwhere(shell_mask)
        
        if len(shell_coords) > 0:
            nucleus_shells[int(lbl)] = shell_coords
    
    elapsed = time.time() - start_time
    print(f"      Dilation completed: {n_nuclei} nuclei in {elapsed:.1f}s ({n_nuclei/elapsed:.1f} nuclei/s)")
    
    return nucleus_shells


# =============================================================================
# FEATURES AND METRICS
# =============================================================================

def compute_bright_threshold_slice_wide(intensity, percentile, exclude_zeros=False):
    vals = intensity.ravel()
    if exclude_zeros:
        vals = vals[vals != 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, int(percentile)))

def compute_shell_features(intensity, shell, bright_threshold):
    shell = shell.astype(np.int64)
    valid = shell > 0
    if not valid.any():
        return None

    labs = shell[valid]
    ivals = intensity[valid]
    bvals = (ivals > float(bright_threshold)).astype(np.float32)

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    sum_b = np.bincount(labs, weights=bvals, minlength=maxlab + 1)

    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present != 0]
    if labels_present.size == 0:
        return None

    n = counts[labels_present].astype(np.float32)
    pct_bright = (sum_b[labels_present] / np.maximum(n, 1.0) * 100.0)

    return pd.DataFrame({
        "label": labels_present.astype(int),
        "pct_bright": pct_bright.astype(np.float32),
    })


def compute_shell_features_with_overlap(intensity, nucleus_shells, bright_threshold):
    """
    Compute features from coordinate dictionary (allows pixel overlap).
    
    This is used for DILATION method where pixels can belong to multiple nuclei.
    """
    results = []
    
    for lbl, coords in nucleus_shells.items():
        if len(coords) == 0:
            continue
        
        # Extract intensity values for this nucleus's shell
        # CRITICAL: same pixel may also be in other nuclei's shells!
        ys, xs = coords[:, 0], coords[:, 1]
        intensities = intensity[ys, xs]
        
        # Calculate features
        n_pixels = len(intensities)
        bright_pixels = np.sum(intensities > float(bright_threshold))
        pct_bright = (bright_pixels / n_pixels * 100.0) if n_pixels > 0 else 0.0
        
        results.append({
            "label": int(lbl),
            "pct_bright": float(pct_bright),
        })
    
    if not results:
        return None
    
    return pd.DataFrame(results)

def classify_with_otsu(df_feat, feature_name):
    vals = df_feat[feature_name].values
    if vals.size < 2 or np.all(vals == vals[0]):
        return None
    try:
        thr = float(threshold_otsu(vals))
    except Exception:
        return None
    pred_pos = set(df_feat.loc[df_feat[feature_name] > thr, "label"].astype(int))
    return thr, pred_pos

def compute_metrics_from_sets(pred_pos, gt_pos, all_labels):
    TP = len(pred_pos & gt_pos)
    FP = len(pred_pos - gt_pos)
    FN = len(gt_pos - pred_pos)
    TN = len(all_labels - pred_pos - gt_pos)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def _metrics_from_counts(TP, FP, FN, TN):
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


# =============================================================================
# SLICE PROCESSING
# =============================================================================

def process_slice(slice_key, method="voronoi"):
    print(f"  Processing slice {slice_key} with {method}...")
    
    tiff_path = TIFF_BY_SLICE[slice_key]
    csv_path = GT_CSV_BY_SLICE[slice_key]

    intensity, mask = load_tiff_channels(tiff_path)
    mask = filter_by_area(mask, AREA_MIN_PX, AREA_MAX_PX)

    all_labels = set(np.unique(mask).astype(int)) - {0}
    gt_labels = load_positive_labels_from_csv(csv_path, mask)

    # Precompute for Voronoi (Dilation computes per-nucleus on-the-fly)
    if method == "voronoi":
        dist_map, assigned = precompute_voronoi_signed_distance(mask)
    else:
        dist_map, assigned = None, None

    bright_thr_by_p = {}
    for p in PERCENTILES:
        thr = compute_bright_threshold_slice_wide(intensity, p, PERCENTILE_EXCLUDE_ZEROS)
        if thr is not None:
            bright_thr_by_p[int(p)] = float(thr)

    rows = []

    # Only test FOCUSED_SHELLS
    for (start_dist, end_dist) in FOCUSED_SHELLS:
        print(f"    Shell [{start_dist},{end_dist}]...")
        
        for p, bright_thr in bright_thr_by_p.items():
            if method == "voronoi":
                shell = create_shell_voronoi(mask, dist_map, assigned, start_dist, end_dist)
                df_feat = compute_shell_features(intensity, shell, bright_thr)
            else:
                # Dilation computes distances per-nucleus (allows pixel overlap!)
                nucleus_shells = create_shell_dilation(mask, start_dist, end_dist)
                df_feat = compute_shell_features_with_overlap(intensity, nucleus_shells, bright_thr)
            if df_feat is None or len(df_feat) < 2:
                for feature_name in FEATURES:
                    rows.append({
                        "slice": int(slice_key),
                        "method": method,
                        "start": int(start_dist),
                        "end": int(end_dist),
                        "percentile": int(p),
                        "feature": feature_name,
                        "TP": 0, "FP": 0, "FN": int(len(gt_labels)), 
                        "TN": int(max(0, len(all_labels) - len(gt_labels))),
                        "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    })
                continue

            for feature_name in FEATURES:
                cls = classify_with_otsu(df_feat, feature_name)
                if cls is None:
                    rows.append({
                        "slice": int(slice_key),
                        "method": method,
                        "start": int(start_dist),
                        "end": int(end_dist),
                        "percentile": int(p),
                        "feature": feature_name,
                        **compute_metrics_from_sets(set(), gt_labels, all_labels),
                    })
                    continue

                thr_otsu, pred_pos = cls
                metrics = compute_metrics_from_sets(pred_pos, gt_labels, all_labels)

                rows.append({
                    "slice": int(slice_key),
                    "method": method,
                    "start": int(start_dist),
                    "end": int(end_dist),
                    "percentile": int(p),
                    "feature": feature_name,
                    **metrics,
                })

    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS AND PLOTS
# =============================================================================

def build_summary_counts(all_results):
    agg = (
        all_results
        .groupby(["method", "start", "end", "percentile", "feature"], as_index=False)
        [["TP", "FP", "FN", "TN"]]
        .sum()
    )

    precision, recall, f1 = [], [], []
    for _, r in agg.iterrows():
        p, rr, ff = _metrics_from_counts(int(r.TP), int(r.FP), int(r.FN), int(r.TN))
        precision.append(p); recall.append(rr); f1.append(ff)

    agg["precision"] = precision
    agg["recall"] = recall
    agg["f1"] = f1

    return agg


def create_summary_table(all_results, out_path):
    """Detailed table for each shell configuration"""
    summary = build_summary_counts(all_results)
    
    # For each shell and method, get best percentile
    results = []
    
    for (start, end) in FOCUSED_SHELLS:
        for method in ["voronoi", "dilation"]:
            df = summary[
                (summary["method"] == method) &
                (summary["start"] == start) &
                (summary["end"] == end) &
                (summary["feature"] == "pct_bright")
            ]
            
            if df.empty:
                continue
            
            # Get best percentile (max precision with recall >= MIN_RECALL)
            ok = df[df["recall"] >= MIN_RECALL]
            if ok.empty:
                ok = df
            
            best = ok.sort_values(["precision", "f1", "recall"], 
                                 ascending=[False, False, False]).iloc[0]
            
            results.append({
                "shell": f"[{start},{end}]",
                "nuclear_px": abs(min(0, start)),  # How much nuclear
                "exterior_px": max(0, end),
                "method": method.upper(),
                "best_percentile": int(best["percentile"]),
                "precision": float(best["precision"]),
                "recall": float(best["recall"]),
                "f1": float(best["f1"]),
                "TP": int(best["TP"]),
                "FP": int(best["FP"]),
                "FN": int(best["FN"]),
                "TN": int(best["TN"]),
            })
    
    df_table = pd.DataFrame(results)
    df_table = df_table.sort_values(["exterior_px", "method"])
    
    df_table.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved {os.path.basename(out_path)}")
    
    # Print to console
    print("\n" + "="*100)
    print("SUMMARY TABLE - Best configuration for each shell")
    print("="*100)
    print(df_table.to_string(index=False))
    print("="*100)
    
    return df_table


def plot_degradation_curves(summary_table, out_path):
    """Plot how metrics degrade with exterior distance (PURE EXTERIOR shells only)"""
    
    # Filter for PURE EXTERIOR shells (nuclear_px = 0)
    df_plot = summary_table[summary_table["nuclear_px"] == 0].copy()
    
    if df_plot.empty:
        print("No pure exterior shells for degradation plot")
        return
    
    # LARGER figure for thesis
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    metrics = ["precision", "recall", "f1"]
    titles = ["Precision", "Recall", "F1 Score"]
    colors = {"VORONOI": "#0173B2", "DILATION": "#DE8F05"}
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for method in ["VORONOI", "DILATION"]:
            data = df_plot[df_plot["method"] == method].copy()
            data = data.sort_values("exterior_px")
            
            if data.empty:
                continue
            
            ax.plot(data["exterior_px"], data[metric],
                   marker="o", linewidth=4.5, markersize=16,
                   label=method.capitalize(),
                   color=colors[method])
            
            # LARGER value labels
            for _, row in data.iterrows():
                ax.text(row["exterior_px"], row[metric] + 0.020,
                       f'{row[metric]:.3f}',
                       ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Calculate and show differences - LARGER font
        voronoi_data = df_plot[df_plot["method"] == "VORONOI"].sort_values("exterior_px")
        dilation_data = df_plot[df_plot["method"] == "DILATION"].sort_values("exterior_px")
        
        if len(voronoi_data) == len(dilation_data):
            for v_row, d_row in zip(voronoi_data.itertuples(), dilation_data.itertuples()):
                diff = getattr(v_row, metric) - getattr(d_row, metric)
                y_pos = (getattr(v_row, metric) + getattr(d_row, metric)) / 2
                diff_color = 'green' if diff > 0 else 'red'
                ax.text(v_row.exterior_px, y_pos, f'{diff:+.3f}',
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       color=diff_color, bbox=dict(boxstyle='round,pad=0.4', 
                                                   facecolor='white', alpha=0.9,
                                                   edgecolor=diff_color, linewidth=2))
        
        # LARGER axis labels
        ax.set_xlabel("Exterior distance (pixels)", fontsize=26, fontweight="bold")
        ax.set_ylabel(title, fontsize=26, fontweight="bold")
        ax.set_title(f"{title} vs Exterior Distance", fontsize=28, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.5)
        
        # Legend BELOW plot area - LARGER font
        ax.legend(fontsize=22, frameon=True, loc='upper center',
                 bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.95)
        
        ax.tick_params(axis="both", labelsize=24, width=2)
        ax.set_ylim(0, 1.10)
        if len(df_plot) > 0:
            ax.set_xlim(-0.5, df_plot["exterior_px"].max() + 0.5)
    
    # LARGER title
    fig.suptitle("Performance Degradation: Voronoi vs Dilation\n(PURE EXTERIOR shells - no nuclear component)",
                 fontsize=32, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


def plot_comparison_by_slice(all_results, out_path):
    """Bar plot comparing methods for slice(s)"""
    
    # Filter for pure exterior shells (3 shells: 1, 5, 10)
    selected_exterior = [1, 5, 10]
    
    # Get data for each slice separately
    slices = sorted(all_results["slice"].unique())
    n_slices = len(slices)
    
    # ADAPTIVE: Single row if only 1 slice
    if n_slices == 1:
        fig, axes = plt.subplots(1, 3, figsize=(26, 8))
        axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(n_slices, 3, figsize=(26, 7.5 * n_slices))
        if n_slices == 1:
            axes = axes.reshape(1, -1)
    
    metrics = ["precision", "recall", "f1"]
    metric_titles = ["Precision", "Recall", "F1 Score"]
    
    for slice_idx, slice_key in enumerate(slices):
        # Get data for this slice only
        df_slice = all_results[all_results["slice"] == slice_key].copy()
        
        # Aggregate by method, shell, percentile for this slice
        summary = (
            df_slice
            .groupby(["method", "start", "end", "percentile", "feature"], as_index=False)
            [["TP", "FP", "FN", "TN"]]
            .sum()
        )
        
        precision, recall, f1 = [], [], []
        for _, r in summary.iterrows():
            p, rr, ff = _metrics_from_counts(int(r.TP), int(r.FP), int(r.FN), int(r.TN))
            precision.append(p); recall.append(rr); f1.append(ff)
        
        summary["precision"] = precision
        summary["recall"] = recall
        summary["f1"] = f1
        summary["exterior_px"] = summary["end"]
        
        # Get best percentile for each shell+method
        results_slice = []
        for (start, end) in FOCUSED_SHELLS:
            for method in ["voronoi", "dilation"]:
                df = summary[
                    (summary["method"] == method) &
                    (summary["start"] == start) &
                    (summary["end"] == end) &
                    (summary["feature"] == "pct_bright")
                ]
                
                if df.empty:
                    continue
                
                ok = df[df["recall"] >= MIN_RECALL]
                if ok.empty:
                    ok = df
                
                best = ok.sort_values(["precision", "f1", "recall"], 
                                     ascending=[False, False, False]).iloc[0]
                
                results_slice.append({
                    "exterior_px": end,
                    "method": method.upper(),
                    "precision": float(best["precision"]),
                    "recall": float(best["recall"]),
                    "f1": float(best["f1"]),
                })
        
        df_plot = pd.DataFrame(results_slice)
        df_plot = df_plot[df_plot["exterior_px"].isin(selected_exterior)]
        
        if df_plot.empty:
            continue
        
        # Plot for each metric
        x = np.arange(len(selected_exterior))
        width = 0.35
        
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[slice_idx, metric_idx]
            
            voronoi_data = df_plot[df_plot["method"] == "VORONOI"].sort_values("exterior_px")
            dilation_data = df_plot[df_plot["method"] == "DILATION"].sort_values("exterior_px")
            
            voronoi_vals = voronoi_data[metric].values
            dilation_vals = dilation_data[metric].values
            
            differences = voronoi_vals - dilation_vals
            
            bars1 = ax.bar(x - width/2, voronoi_vals, width, label="Voronoi",
                          color="#0173B2", edgecolor="black", linewidth=2.5)
            bars2 = ax.bar(x + width/2, dilation_vals, width, label="Dilation",
                          color="#DE8F05", edgecolor="black", linewidth=2.5)
            
            # LARGER value labels
            for i, (v_val, d_val, diff) in enumerate(zip(voronoi_vals, dilation_vals, differences)):
                ax.text(i - width/2, v_val + 0.018, f'{v_val:.3f}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
                ax.text(i + width/2, d_val + 0.018, f'{d_val:.3f}',
                       ha='center', va='bottom', fontsize=18, fontweight='bold')
                
                # Show difference - LARGER font
                diff_color = 'green' if diff > 0 else 'red'
                ax.text(i, max(v_val, d_val) + 0.075, f'Δ={diff:+.3f}',
                       ha='center', va='bottom', fontsize=16, fontweight='bold',
                       color=diff_color)
            
            # LARGER axis labels
            ax.set_ylabel(metric_title, fontsize=26, fontweight="bold")
            
            # Only show x-label on bottom row - LARGER
            if slice_idx == len(slices) - 1:
                ax.set_xlabel("Exterior distance (pixels)", fontsize=24, fontweight="bold")
            
            ax.set_xticks(x)
            ax.set_xticklabels([f"{e}" for e in selected_exterior], fontsize=22, fontweight="bold")
            ax.set_ylim(0, 1.18)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
            
            # Add slice label on first column - LARGER
            if metric_idx == 0:
                ax.text(-0.20, 0.5, f'Slice {slice_key}', 
                       transform=ax.transAxes, fontsize=28, fontweight='bold',
                       rotation=90, va='center', ha='right')
            
            # Legend BELOW plot area on top row, LARGER font
            if slice_idx == 0:
                ax.legend(fontsize=22, frameon=True, loc='upper center', 
                         bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.95)
            
            ax.tick_params(axis='y', labelsize=22, width=2)
            ax.tick_params(axis='x', width=2)
    
    # LARGER title - adaptive based on number of slices
    if n_slices == 1:
        title_text = f"Voronoi vs Dilation: Pure Exterior Shells (Slice {slices[0]})\n(1, 5, 10 pixel exterior - no nuclear component)"
    else:
        title_text = "Voronoi vs Dilation: Pure Exterior Shells BY SLICE\n(1, 5, 10 pixel exterior - no nuclear component)"
    
    fig.suptitle(title_text, fontsize=32, fontweight="bold", y=0.998)
    
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")
    """Bar plot comparing EXTERIOR-ONLY shells"""
    
    # Select PURE EXTERIOR shells (no nuclear component)
    selected_exterior = [1, 2, 3, 5, 10]
    df_plot = summary_table[
        (summary_table["nuclear_px"] == 0) &  # Only pure exterior
        (summary_table["exterior_px"].isin(selected_exterior))
    ].copy()
    
    if df_plot.empty:
        print("No pure exterior shells found for bar plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    metrics = ["precision", "recall", "f1"]
    titles = ["Precision", "Recall", "F1 Score"]
    
    x = np.arange(len(selected_exterior))
    width = 0.35
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        voronoi_vals = df_plot[df_plot["method"] == "VORONOI"].sort_values("exterior_px")[metric].values
        dilation_vals = df_plot[df_plot["method"] == "DILATION"].sort_values("exterior_px")[metric].values
        
        # Calculate differences
        differences = voronoi_vals - dilation_vals
        
        ax.bar(x - width/2, voronoi_vals, width, label="Voronoi",
               color="#0173B2", edgecolor="black", linewidth=2)
        ax.bar(x + width/2, dilation_vals, width, label="Dilation",
               color="#DE8F05", edgecolor="black", linewidth=2)
        
        # Value labels with differences
        for i, (v_val, d_val, diff) in enumerate(zip(voronoi_vals, dilation_vals, differences)):
            ax.text(i - width/2, v_val + 0.015, f'{v_val:.3f}',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
            ax.text(i + width/2, d_val + 0.015, f'{d_val:.3f}',
                   ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            # Show difference
            diff_color = 'green' if diff > 0 else 'red'
            ax.text(i, max(v_val, d_val) + 0.055, f'Δ={diff:+.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold',
                   color=diff_color)
        
        ax.set_ylabel(title, fontsize=20, fontweight="bold")
        ax.set_title(f"{title} Comparison", fontsize=22, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e}px" for e in selected_exterior],
                          fontsize=16, fontweight="bold")
        ax.set_xlabel("Exterior distance (pure exterior, no nuclear)", fontsize=18, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=16, frameon=True)
        ax.tick_params(axis='y', labelsize=16)
    
    fig.suptitle("Voronoi vs Dilation: PURE EXTERIOR Shells\n(Nuclear region excluded to highlight differences)",
                 fontsize=26, fontweight="bold", y=0.98)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FOCUSED VORONOI VS DILATION COMPARISON")
    print("SINGLE SLICE VERSION (Slice 110 only - ~1000 nuclei)")
    print("FOCUS: PURE EXTERIOR SHELLS (no nuclear component)")
    print("This highlights where Voronoi and Dilation differ most!")
    print("=" * 80)
    print(f"Output: {OUT_DIR}")
    print(f"Testing {len(FOCUSED_SHELLS)} shell configurations")
    print(f"Shells: {FOCUSED_SHELLS}")
    print(f"Percentiles: {PERCENTILES}")
    print(f"Processing ONLY slice 110 to keep runtime reasonable")
    print(f"Estimated runtime: ~45-60 minutes (1000 nuclei × 3 shells × 3 percentiles)")
    print("\nNOTE: Pure exterior shells [0,X] show maximum difference")
    print("      because nuclear region is identical in both methods.")

    # Process all slices with BOTH methods
    print("\n" + "=" * 80)
    print("PROCESSING SLICES")
    print("=" * 80)
    
    all_rows = []
    
    for method in ["voronoi", "dilation"]:
        print(f"\n--- Method: {method.upper()} ---")
        for sk in sorted(TIFF_BY_SLICE.keys()):
            df = process_slice(sk, method=method)
            all_rows.append(df)

    all_results = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "focused_comparison_all_results.csv")
    all_results.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Generate analyses
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS")
    print("=" * 80)

    # 1. Summary table
    out_path = os.path.join(OUT_DIR, "summary_table_by_shell.csv")
    summary_table = create_summary_table(all_results, out_path)

    # 2. Degradation curves (aggregated)
    out_path = os.path.join(FIG_DIR, "degradation_curves.png")
    plot_degradation_curves(summary_table, out_path)

    # 3. Bar comparison BY SLICE (cleaner for publication)
    out_path = os.path.join(FIG_DIR, "bar_comparison_by_slice.png")
    plot_comparison_by_slice(all_results, out_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"All outputs in: {OUT_DIR}")
    print("\nConclusion: Check how Voronoi maintains better performance")
    print("as exterior distance increases!")


if __name__ == "__main__":
    main()