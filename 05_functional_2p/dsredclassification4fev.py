#!/usr/bin/env python3
"""
DsRed Classification - Percentile Method Comparison

Compares:
1. Percentile calculated WITHIN mask only
2. Percentile calculated across ENTIRE slice
3. F1 score for each method across slices and percentiles
"""

import os
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.5


# =============================================================================
# CONFIG
# =============================================================================

BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "dsred_percentile_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SLICES = [110, 300, 450]

GT_CSV_BY_SLICE = {
    110: os.path.join(BASE_RESULTS, "Results110dsrednovo.csv"),
    300: os.path.join(BASE_RESULTS, "Results300dsrednovo.csv"),
    450: os.path.join(BASE_RESULTS, "Results450dsrednovo.csv"),
}

MIN_CELL_PIXELS = 400
AREA_MAX_PX = 6000

PERCENTILES = [60, 65, 70, 75, 80, 85, 90]

FEATURES = ["pct_bright"]

NEIGHBOR_RADIUS_PX = 2


# =============================================================================
# IO AND GT
# =============================================================================

def find_tiff_for_slice(results_dir, slice_num):
    from pathlib import Path
    results_path = Path(results_dir)
    patterns = [
        f"*dsredslice{slice_num}*fish7*.tif",
        f"*dsredslice_{slice_num}*.tif",
        f"*dsred{slice_num}*dsred*.tif",
        f"*dsred*{slice_num}*.tif",
    ]
    for pattern in patterns:
        files = list(results_path.glob(pattern))
        if files:
            return str(files[0])
    raise FileNotFoundError(f"No TIFF found for slice {slice_num}")

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

def _maybe_fix_1_based(coords, W, H):
    x, y = coords[:, 0], coords[:, 1]
    out0 = np.mean((x < 0) | (x >= W) | (y < 0) | (y >= H))
    x1, y1 = x - 1, y - 1
    out1 = np.mean((x1 < 0) | (x1 >= W) | (y1 < 0) | (y1 >= H))
    if out1 + 1e-6 < out0:
        return np.stack([x1, y1], axis=1), True
    return coords, False

def _label_at_or_near(mask, x, y, r=3):
    H, W = mask.shape
    xi, yi = int(round(float(x))), int(round(float(y)))
    if xi < 0 or xi >= W or yi < 0 or yi >= H:
        return 0
    lab = int(mask[yi, xi])
    if lab != 0:
        return lab
    if r <= 0:
        return 0
    x0, x1 = max(0, xi - r), min(W - 1, xi + r)
    y0, y1 = max(0, yi - r), min(H - 1, yi + r)
    patch = mask[y0:y1+1, x0:x1+1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0
    uniq, cnt = np.unique(vals, return_counts=True)
    return int(uniq[np.argmax(cnt)])

def load_positive_labels_from_csv(csv_path, label_slice, neighbor_radius_px=3):
    df = pd.read_csv(csv_path)
    coords = df[["XM", "YM"]].to_numpy(dtype=float)
    H, W = label_slice.shape
    coords, _ = _maybe_fix_1_based(coords, W, H)
    labels = []
    for (x, y) in coords:
        lab = _label_at_or_near(label_slice, x, y, r=int(neighbor_radius_px))
        if lab != 0:
            labels.append(lab)
    return set(int(v) for v in labels)


# =============================================================================
# PERCENTILE CALCULATION METHODS
# =============================================================================

def compute_percentile_within_mask(intensity, mask, percentile):
    """Calcula percentil APENAS dentro da máscara"""
    intensity_in_mask = intensity[mask > 0]
    if intensity_in_mask.size == 0:
        return None
    return float(np.percentile(intensity_in_mask, int(percentile)))

def compute_percentile_entire_slice(intensity, percentile):
    """Calcula percentil na SLICE INTEIRA (incluindo background)"""
    vals = intensity.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.percentile(vals, int(percentile)))


# =============================================================================
# FEATURES AND METRICS
# =============================================================================

def compute_nuclear_features(intensity, mask, bright_threshold):
    mask = mask.astype(np.int64)
    valid = mask > 0
    if not valid.any():
        return None

    labs = mask[valid]
    ivals = intensity[valid]
    bvals = (ivals > bright_threshold).astype(np.float32)

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    sum_b = np.bincount(labs, weights=bvals, minlength=maxlab + 1)

    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present != 0]
    if labels_present.size == 0:
        return None

    n = counts[labels_present]
    pct_bright = (sum_b[labels_present] / np.maximum(n, 1.0) * 100.0)

    df = pd.DataFrame({
        "label": labels_present.astype(int),
        "pct_bright": pct_bright,
    })
    return df

def classify_with_otsu(df_feat, feature_name):
    vals = df_feat[feature_name].values
    if vals.size < 2 or np.all(vals == vals[0]):
        return None
    try:
        thr = float(threshold_otsu(vals))
    except:
        return None
    pred_pos = set(df_feat.loc[df_feat[feature_name] > thr, "label"].astype(int))
    return thr, pred_pos

def compute_metrics(pred_pos, gt_pos, all_labels):
    TP = len(pred_pos & gt_pos)
    FP = len(pred_pos - gt_pos)
    FN = len(gt_pos - pred_pos)
    TN = len(all_labels - pred_pos - gt_pos)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# PROCESS SLICE WITH BOTH METHODS
# =============================================================================

def process_slice_both_methods(slice_key, tiff_path, csv_path):
    """Processa slice com ambos os métodos de percentil"""
    print(f"\nProcessing slice {slice_key}")
    
    intensity, mask = load_tiff_channels(tiff_path)
    mask = filter_by_area(mask, MIN_CELL_PIXELS, AREA_MAX_PX)

    all_labels = set(np.unique(mask).astype(int)) - {0}
    gt_labels = load_positive_labels_from_csv(csv_path, mask, NEIGHBOR_RADIUS_PX)

    print(f"  Cells: {len(all_labels)}  Positive: {len(gt_labels)}")

    results = []

    for p in PERCENTILES:
        # METHOD 1: Percentile within mask
        bright_thr_mask = compute_percentile_within_mask(intensity, mask, p)
        if bright_thr_mask is not None:
            df_feat = compute_nuclear_features(intensity, mask, bright_thr_mask)
            if df_feat is not None and len(df_feat) >= 2:
                cls = classify_with_otsu(df_feat, "pct_bright")
                if cls is not None:
                    _, pred_pos = cls
                    metrics = compute_metrics(pred_pos, gt_labels, all_labels)
                    results.append({
                        "slice": int(slice_key),
                        "percentile": int(p),
                        "method": "within_mask",
                        "bright_threshold": float(bright_thr_mask),
                        **metrics,
                    })

        # METHOD 2: Percentile entire slice
        bright_thr_slice = compute_percentile_entire_slice(intensity, p)
        if bright_thr_slice is not None:
            df_feat = compute_nuclear_features(intensity, mask, bright_thr_slice)
            if df_feat is not None and len(df_feat) >= 2:
                cls = classify_with_otsu(df_feat, "pct_bright")
                if cls is not None:
                    _, pred_pos = cls
                    metrics = compute_metrics(pred_pos, gt_labels, all_labels)
                    results.append({
                        "slice": int(slice_key),
                        "percentile": int(p),
                        "method": "entire_slice",
                        "bright_threshold": float(bright_thr_slice),
                        **metrics,
                    })

    return pd.DataFrame(results)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_f1_comparison_by_method(results_df, out_path):
    """
    Compara F1 dos dois métodos ao longo dos percentis
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    methods = results_df["method"].unique()
    colors = {
        "within_mask": "#0173B2",
        "entire_slice": "#DE8F05",
    }
    labels = {
        "within_mask": "Within mask only",
        "entire_slice": "Entire slice",
    }

    for method in methods:
        df_method = results_df[results_df["method"] == method].copy()
        
        # Aggregate across slices
        summary = df_method.groupby("percentile").agg({
            "f1": ["mean", "std"],
        }).reset_index()
        summary.columns = ["percentile", "f1_mean", "f1_std"]
        
        ax.errorbar(
            summary["percentile"], summary["f1_mean"], yerr=summary["f1_std"],
            marker="o", linewidth=3.5, markersize=12, capsize=6,
            label=labels.get(method, method),
            color=colors.get(method, "black")
        )

    ax.set_xlabel("Percentile threshold (%)", fontsize=22, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=22, fontweight="bold")
    ax.set_title("F1 comparison: Percentile calculation methods\nAveraged across all slices",
                 fontsize=24, fontweight="bold", pad=20)

    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)
    ax.legend(fontsize=20, frameon=True, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


def plot_f1_by_slice_for_each_method(results_df, out_path):
    """
    Mostra F1 para cada método separadamente, com curvas por slice
    """
    methods = sorted(results_df["method"].unique())
    
    fig, axes = plt.subplots(1, 2, figsize=(26, 9))

    colors_map = {
        110: "#E69F00",
        300: "#56B4E9",
        450: "#009E73",
    }

    method_labels = {
        "within_mask": "Percentile within mask only",
        "entire_slice": "Percentile across entire slice",
    }

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_method = results_df[results_df["method"] == method].copy()

        for slice_key in SLICES:
            df_slice = df_method[df_method["slice"] == slice_key].copy()
            if df_slice.empty:
                continue
            
            df_slice = df_slice.sort_values("percentile")
            
            ax.plot(df_slice["percentile"], df_slice["f1"],
                    marker="o", linewidth=3.5, markersize=12,
                    label=f"Slice {slice_key}",
                    color=colors_map.get(slice_key, "black"))

        ax.set_xlabel("Percentile threshold (%)", fontsize=20, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=20, fontweight="bold")
        ax.set_title(f"{method_labels.get(method, method)}",
                     fontsize=22, fontweight="bold", pad=15)

        ax.tick_params(axis="both", labelsize=18)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)
        ax.legend(fontsize=16, frameon=True, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


def plot_threshold_comparison(results_df, out_path):
    """
    Compara os valores de threshold entre os dois métodos
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    methods = results_df["method"].unique()
    colors = {
        "within_mask": "#0173B2",
        "entire_slice": "#DE8F05",
    }
    labels = {
        "within_mask": "Within mask",
        "entire_slice": "Entire slice",
    }

    for method in methods:
        df_method = results_df[results_df["method"] == method].copy()
        
        summary = df_method.groupby("percentile").agg({
            "bright_threshold": ["mean", "std"],
        }).reset_index()
        summary.columns = ["percentile", "threshold_mean", "threshold_std"]
        
        ax.errorbar(
            summary["percentile"], summary["threshold_mean"], yerr=summary["threshold_std"],
            marker="s", linewidth=3.5, markersize=12, capsize=6,
            label=labels.get(method, method),
            color=colors.get(method, "black")
        )

    ax.set_xlabel("Percentile (%)", fontsize=22, fontweight="bold")
    ax.set_ylabel("Bright threshold (intensity value)", fontsize=22, fontweight="bold")
    ax.set_title("Threshold values by percentile calculation method\nAveraged across all slices",
                 fontsize=24, fontweight="bold", pad=20)

    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)
    ax.legend(fontsize=20, frameon=True, loc="best")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


def plot_metrics_heatmap_comparison(results_df, out_path):
    """
    Heatmap comparando precision, recall, F1 entre os métodos
    """
    methods = sorted(results_df["method"].unique())
    metrics = ["precision", "recall", "f1"]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    method_labels = {
        "within_mask": "Within mask",
        "entire_slice": "Entire slice",
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for heatmap
        data = []
        for method in methods:
            df_method = results_df[results_df["method"] == method].copy()
            summary = df_method.groupby("percentile")[metric].mean().to_dict()
            row = [summary.get(p, np.nan) for p in PERCENTILES]
            data.append(row)
        
        data_array = np.array(data)
        
        sns.heatmap(
            data_array,
            ax=ax,
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            center=0.8,
            annot=True,
            fmt=".3f",
            linewidths=2,
            linecolor="white",
            cbar_kws={"label": metric.upper()},
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=[f"{p}%" for p in PERCENTILES],
            yticklabels=[method_labels.get(m, m) for m in methods],
        )
        
        ax.set_title(f"{metric.upper()}", fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("Percentile threshold", fontsize=16, fontweight="bold")
        ax.set_ylabel("Method", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


def create_summary_table(results_df, out_path):
    """
    Cria tabela comparativa com melhor config de cada método
    """
    summary_rows = []
    
    for method in results_df["method"].unique():
        df_method = results_df[results_df["method"] == method].copy()
        
        # Aggregate by percentile
        agg = df_method.groupby("percentile").agg({
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
        }).reset_index()
        
        # Find best by F1
        best = agg.nlargest(1, "f1").iloc[0]
        
        summary_rows.append({
            "method": method,
            "best_percentile": int(best["percentile"]),
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("f1", ascending=False)
    
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary table: {os.path.basename(out_path)}")
    print("\nBest configuration for each method:")
    print(summary_df.to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("DSRED PERCENTILE METHOD COMPARISON")
    print("=" * 80)
    print(f"Output: {OUT_DIR}")
    print(f"Comparing methods:")
    print("  1. Percentile calculated within mask only")
    print("  2. Percentile calculated across entire slice")

    # Find TIFFs
    tiff_paths = {}
    for sk in SLICES:
        try:
            tiff_paths[sk] = find_tiff_for_slice(BASE_RESULTS, sk)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return

    # Process all slices with both methods
    all_results = []
    for sk in SLICES:
        df = process_slice_both_methods(sk, tiff_paths[sk], GT_CSV_BY_SLICE[sk])
        all_results.append(df)

    if not all_results:
        print("No results!")
        return

    results_df = pd.concat(all_results, ignore_index=True)
    
    # Save all results
    out_csv = os.path.join(OUT_DIR, "all_results_both_methods.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)

    # 1. F1 comparison averaged
    out_path = os.path.join(FIG_DIR, "f1_comparison_methods.png")
    plot_f1_comparison_by_method(results_df, out_path)

    # 2. F1 by slice for each method
    out_path = os.path.join(FIG_DIR, "f1_by_slice_each_method.png")
    plot_f1_by_slice_for_each_method(results_df, out_path)

    # 3. Threshold values comparison
    out_path = os.path.join(FIG_DIR, "threshold_comparison.png")
    plot_threshold_comparison(results_df, out_path)

    # 4. Metrics heatmap
    out_path = os.path.join(FIG_DIR, "metrics_heatmap_comparison.png")
    plot_metrics_heatmap_comparison(results_df, out_path)

    # 5. Summary table
    out_path = os.path.join(OUT_DIR, "summary_best_per_method.csv")
    create_summary_table(results_df, out_path)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"All outputs in: {OUT_DIR}")
    print(f"Figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()