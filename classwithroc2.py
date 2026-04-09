#!/usr/bin/env python3
"""
In situ classification final complete version

Explores a wide grid of shell configurations
Exports only top 10 shells in heatmaps and tables
Optional outer assignment mode
boundary assigns by nearest nuclear pixel
centroid assigns by nearest centroid
"""

import os
import warnings

import numpy as np
import pandas as pd
import tifffile as tiff

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import seaborn as sns

from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

from sklearn.metrics import roc_curve, auc as calc_auc

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5


# =============================================================================
# CONFIG
# =============================================================================

BASE_RESULTS = r"C:\Users\OSVALDO\Downloads\results"
OUT_DIR = os.path.join(BASE_RESULTS, "insitu_FINAL_complete")
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

PERCENTILES = [50, 60, 65, 70, 75, 80, 85, 90]

FEATURES = ["pct_bright", "mean_intensity", "integrated_intensity"]

DISPLAY_P_LOW = 1.0
DISPLAY_P_HIGH = 99.0
DISPLAY_GAMMA = 0.7

FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ZOOM_CONTOUR_THICKNESS = 1.5
OVERVIEW_CONTOUR_THICKNESS = 1.5

ZOOM_PER_TEST = 3
ZOOM_MARGIN_PX = 80
ZOOM_MIN_SIZE = 360
ZOOM_PAD_PX = 60
OVERVIEW_CROP_MARGIN_PX = 120
MIN_ZOOM_SEPARATION_PX = 500

ZOOM_LABELS_BY_SLICE = {
    110: ["A", "B", "C"],
    300: ["D", "E", "F"],
    450: ["G", "H", "I"],
}

N_ZOOM_VARIANTS_SLICE300 = 8

TOP_N_SHELLS_HEATMAP = 10
TOP_N_SHELLS_TABLE = 10

# Shell search space
SHELL_START_MIN = -8
SHELL_START_MAX = 0
SHELL_END_MIN = -8
SHELL_END_MAX = 8
SHELL_STEP = 1

# Outer assignment mode
# "boundary" matches your current method
# "centroid" is Voronoi by centroids
SHELL_ASSIGNMENT_MODE = "boundary"

# Colors used in overlays
COLORS = {
    "TP": np.array([46, 204, 113], dtype=np.uint8),
    "FP": np.array([255, 0, 255], dtype=np.uint8),
    "FN": np.array([86, 180, 233], dtype=np.uint8),
    "TN": np.array([221, 221, 221], dtype=np.uint8),
}

ZOOM_BOX_COLORS = {
    "TP": "#2ECC71",
    "FP": "#FF00FF",
    "FN": "#56B4E9",
    "TN": "#DDDDDD",
}


# =============================================================================
# SHELL CONFIG GRID
# =============================================================================

def generate_shell_configs(start_min, start_max, end_min, end_max, step=1):
    starts = list(range(int(start_min), int(start_max) + 1, int(step)))
    ends = list(range(int(end_min), int(end_max) + 1, int(step)))
    out = []
    for s in starts:
        for e in ends:
            if s <= e:
                out.append((int(s), int(e)))
    return out


SHELL_CONFIGS = generate_shell_configs(
    SHELL_START_MIN, SHELL_START_MAX, SHELL_END_MIN, SHELL_END_MAX, SHELL_STEP
)


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
# SHELL HELPERS
# =============================================================================

def prepare_shell_helpers(label_slice, assignment_mode="boundary"):
    binary = label_slice > 0
    if not binary.any():
        return {
            "binary": binary,
            "dist_map": None,
            "idx": None,
            "tree": None,
            "centroid_labels": None,
            "centroids_rc": None,
            "assignment_mode": assignment_mode,
        }

    dist_out = distance_transform_edt(~binary)
    dist_in = distance_transform_edt(binary)
    dist_map = np.where(binary, -dist_in, dist_out)

    idx = None
    tree = None
    centroid_labels = None
    centroids_rc = None

    if assignment_mode == "boundary":
        _, idx = distance_transform_edt(~binary, return_indices=True)
    elif assignment_mode == "centroid":
        props = regionprops(label_slice.astype(np.int32))
        centroid_labels = []
        centroids_rc = []
        for p in props:
            centroid_labels.append(int(p.label))
            centroids_rc.append([float(p.centroid[0]), float(p.centroid[1])])
        centroid_labels = np.array(centroid_labels, dtype=np.int32)
        centroids_rc = np.array(centroids_rc, dtype=np.float32)
        if len(centroids_rc) > 0:
            tree = cKDTree(centroids_rc)
    else:
        raise ValueError("Unknown assignment_mode")

    return {
        "binary": binary,
        "dist_map": dist_map,
        "idx": idx,
        "tree": tree,
        "centroid_labels": centroid_labels,
        "centroids_rc": centroids_rc,
        "assignment_mode": assignment_mode,
    }

def create_shell_2d_from_helpers(label_slice, start_dist, end_dist, helpers):
    if helpers["dist_map"] is None:
        return np.zeros_like(label_slice)

    dist_map = helpers["dist_map"]
    shell_mask = (dist_map >= float(start_dist)) & (dist_map <= float(end_dist))
    shell = np.where(shell_mask, label_slice, 0).astype(label_slice.dtype)

    if end_dist > 0:
        outside = shell_mask & (dist_map > 0)
        if outside.any():
            mode = helpers["assignment_mode"]
            if mode == "boundary":
                idx = helpers["idx"]
                shell[outside] = label_slice[idx[0], idx[1]][outside]
            else:
                tree = helpers["tree"]
                centroid_labels = helpers["centroid_labels"]
                if tree is None or centroid_labels is None or len(centroid_labels) == 0:
                    return shell
                coords = np.column_stack(np.where(outside)).astype(np.float32)
                _, nn = tree.query(coords, k=1)
                shell[outside] = centroid_labels[nn].astype(label_slice.dtype)

    return shell


# =============================================================================
# FEATURES AND METRICS
# =============================================================================

def compute_shell_features(intensity, shell, bright_threshold):
    shell = shell.astype(np.int64)
    valid = shell > 0
    if not valid.any():
        return np.array([], dtype=np.int64), None

    labs = shell[valid]
    ivals = intensity[valid]
    bvals = (ivals > bright_threshold).astype(np.float32)

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    sum_int = np.bincount(labs, weights=ivals, minlength=maxlab + 1)
    sum_b = np.bincount(labs, weights=bvals, minlength=maxlab + 1)

    labels_present = np.nonzero(counts)[0]
    labels_present = labels_present[labels_present != 0]
    if labels_present.size == 0:
        return np.array([], dtype=np.int64), None

    n = counts[labels_present]
    pct_bright = (sum_b[labels_present] / np.maximum(n, 1.0) * 100.0)
    mean_int = sum_int[labels_present] / np.maximum(n, 1.0)
    integ_int = sum_int[labels_present]

    df = pd.DataFrame({
        "label": labels_present.astype(int),
        "n_shell_px": counts[labels_present].astype(int),
        "pct_bright": pct_bright,
        "mean_intensity": mean_int,
        "integrated_intensity": integ_int,
    })
    return labels_present, df

def classify_with_otsu(df_feat, feature_name):
    vals = df_feat[feature_name].values
    if vals.size < 2:
        return None
    try:
        thr = float(threshold_otsu(vals))
    except Exception:
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
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

def roc_auc(df_feat, gt_pos, feature_name):
    y_true = df_feat["label"].isin(gt_pos).astype(int).values
    scores = df_feat[feature_name].values
    if scores.size < 2 or np.all(scores == scores[0]):
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, float(calc_auc(fpr, tpr))


# =============================================================================
# DISPLAY AND OVERLAYS
# =============================================================================

def enhance_display(intensity, p_low=2.0, p_high=98.0, gamma=0.7):
    nz = intensity[intensity > 0]
    if nz.size == 0:
        return np.zeros_like(intensity, dtype=np.float32)

    lo, hi = np.percentile(nz, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0

    x = np.clip(intensity, lo, hi)
    y = (x - lo) / (hi - lo)
    y = np.power(y, gamma)
    return y

def gray_to_rgb(gray):
    gray_uint8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

def overlay_colored(rgb_base, colored_rgb, alpha=0.7):
    result = rgb_base.copy()
    mask = np.any(colored_rgb > 0, axis=-1)
    if mask.any():
        result[mask] = (alpha * colored_rgb[mask] + (1 - alpha) * rgb_base[mask]).astype(np.uint8)
    return result

def build_thick_contours(mask, labels_by_cat, thickness=1.5):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    b = find_boundaries(mask.astype(np.int32), mode="inner")

    for cat in ["TN", "FN", "FP", "TP"]:
        labs = labels_by_cat.get(cat, set())
        if not labs:
            continue
        sel = b & np.isin(mask, list(labs))
        if thickness > 1:
            sel = binary_dilation(sel, disk(1))
        out[sel] = COLORS[cat]

    return out

def build_bright_pixels_overlay(intensity, shell, bright_thr, labels_by_cat):
    out = np.zeros((shell.shape[0], shell.shape[1], 3), dtype=np.uint8)
    bright = (intensity > bright_thr) & (shell > 0)

    for cat in ["TN", "FN", "FP", "TP"]:
        labs = labels_by_cat.get(cat, set())
        if not labs:
            continue
        m = bright & np.isin(shell, list(labs))
        out[m] = COLORS[cat]

    return out

def create_legend_patches():
    return [
        mpatches.Patch(color=COLORS["TP"] / 255.0, label="True Positive"),
        mpatches.Patch(color=COLORS["FP"] / 255.0, label="False Positive"),
        mpatches.Patch(color=COLORS["FN"] / 255.0, label="False Negative"),
        mpatches.Patch(color=COLORS["TN"] / 255.0, label="True Negative"),
    ]

def label_props(mask):
    props = regionprops(mask.astype(np.int32))
    out = {}
    for p in props:
        per = float(p.perimeter) if getattr(p, "perimeter", 0) else 0.0
        area = float(p.area)
        circ = (4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
        out[int(p.label)] = {
            "bbox": p.bbox,
            "centroid": p.centroid,
            "circularity": float(circ),
            "area": float(area),
        }
    return out

def crop_bbox(bbox, H, W, margin, min_size):
    minr, minc, maxr, maxc = bbox
    minr = max(0, minr - margin)
    minc = max(0, minc - margin)
    maxr = min(H, maxr + margin)
    maxc = min(W, maxc + margin)

    h = maxr - minr
    w = maxc - minc
    size = max(h, w, min_size)

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

def crop_with_pad(r0, r1, c0, c1, H, W, pad):
    pr0 = max(0, r0 - pad)
    pr1 = min(H, r1 + pad)
    pc0 = max(0, c0 - pad)
    pc1 = min(W, c1 + pad)
    return pr0, pr1, pc0, pc1

def _far_enough(new_c, chosen_centroids):
    for ec in chosen_centroids:
        if float(np.hypot(new_c[0] - ec[0], new_c[1] - ec[1])) < MIN_ZOOM_SEPARATION_PX:
            return False
    return True

def _rank_candidates(df_labels, prop_map, cat, min_circularity=0.55):
    sub = df_labels[df_labels["category"] == cat].copy()
    if sub.empty:
        return []
    sub["circularity"] = sub["label"].map(lambda l: prop_map.get(int(l), {}).get("circularity", 0.0))
    sub = sub[sub["circularity"] >= float(min_circularity)]
    if sub.empty:
        return []
    sub = sub.sort_values(["circularity", "pct_bright"], ascending=[False, False])
    return [int(x) for x in sub["label"].tolist()]

def select_zoom_labels_forced(df_labels, prop_map, k=3, variant=0, min_circularity=0.55):
    desired = ["TP", "FP", "FN"]
    picks = []
    centroids = []
    used = set()

    for i, cat in enumerate(desired):
        cands = _rank_candidates(df_labels, prop_map, cat, min_circularity=min_circularity)
        if not cands:
            continue

        start = (variant + i) % len(cands)
        scan = cands[start:] + cands[:start]

        chosen = None
        for lab in scan:
            if lab in used:
                continue
            info = prop_map.get(int(lab))
            if info is None:
                continue
            c = info["centroid"]
            if _far_enough(c, centroids):
                chosen = int(lab)
                centroids.append(c)
                used.add(chosen)
                break

        if chosen is not None:
            picks.append(chosen)

        if len(picks) >= k:
            return picks[:k]

    all_labels = df_labels.copy()
    all_labels["circularity"] = all_labels["label"].map(lambda l: prop_map.get(int(l), {}).get("circularity", 0.0))
    all_labels = all_labels[all_labels["circularity"] >= float(min_circularity)]
    all_labels = all_labels.sort_values(["circularity", "pct_bright"], ascending=[False, False])

    rest = [int(x) for x in all_labels["label"].tolist() if int(x) not in used]
    if rest:
        start = variant % len(rest)
        rest = rest[start:] + rest[:start]

    for lab in rest:
        info = prop_map.get(int(lab))
        if info is None:
            continue
        c = info["centroid"]
        if _far_enough(c, centroids):
            picks.append(int(lab))
            centroids.append(c)
            used.add(int(lab))
        if len(picks) >= k:
            break

    return picks[:k]


# =============================================================================
# FIGURES
# =============================================================================

def create_complete_f1_heatmap(results_df, feature_name, out_path, top_n=10):
    df = results_df[results_df["feature"] == feature_name].copy()
    if df.empty:
        return

    pivot = df.groupby(["start", "end", "percentile"])["f1"].mean().reset_index()
    pivot["shell_label"] = pivot.apply(lambda x: f"[{int(x['start'])},{int(x['end'])}]", axis=1)
    pivot_table = pivot.pivot(index="shell_label", columns="percentile", values="f1")

    pivot_table["avg"] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values("avg", ascending=False)

    if top_n is not None and len(pivot_table) > int(top_n):
        pivot_table = pivot_table.iloc[:int(top_n)].copy()

    pivot_table = pivot_table.drop("avg", axis=1)

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_table) * 0.55)))
    sns.heatmap(
        pivot_table, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0.5, vmax=1.0, center=0.75,
        cbar_kws={"label": "F1 score"},
        linewidths=1, linecolor="white", ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_xlabel("Percentile threshold (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Shell configuration [inner, outer] (px)", fontsize=13, fontweight="bold")
    ax.set_title(f"Top {len(pivot_table)} shells by mean F1\nFeature {feature_name}",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")

def export_top_shells_table(results_df, feature="pct_bright", top_n=10, out_csv=None, out_tex=None):
    df = results_df[results_df["feature"] == feature].copy()
    if df.empty:
        return None

    agg = (
        df.groupby(["start", "end", "percentile"], as_index=False)[["f1", "precision", "recall", "accuracy"]]
          .mean()
    )
    idx = agg.groupby(["start", "end"])["f1"].idxmax()
    best_per_shell = agg.loc[idx].copy()
    best_per_shell = best_per_shell.sort_values("f1", ascending=False).head(int(top_n)).reset_index(drop=True)

    best_per_shell.rename(columns={
        "start": "inner_px",
        "end": "outer_px",
        "percentile": "best_percentile",
        "f1": "best_f1",
        "precision": "precision_mean",
        "recall": "recall_mean",
        "accuracy": "accuracy_mean",
    }, inplace=True)

    if out_csv is not None:
        best_per_shell.to_csv(out_csv, index=False)
        print(f"Saved {os.path.basename(out_csv)}")

    if out_tex is not None:
        tex_df = best_per_shell.copy()
        tex_df["shell"] = tex_df.apply(lambda r: f"[{int(r['inner_px'])},{int(r['outer_px'])}]", axis=1)
        tex_df = tex_df[["shell", "best_percentile", "best_f1", "precision_mean", "recall_mean", "accuracy_mean"]]
        tex_df.columns = ["Shell", "Best P", "F1", "Precision", "Recall", "Accuracy"]
        tex = tex_df.to_latex(index=False, float_format="%.3f")
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"Saved {os.path.basename(out_tex)}")

    return best_per_shell

def plot_metrics_vs_percentile_best_shell(results_df, feature="pct_bright", out_path=None):
    df = results_df[results_df["feature"] == feature].copy()
    if df.empty:
        print("Nao ha dados para essa feature.")
        return None

    # Choose best shell by peak mean F1 over percentiles
    mean_over_slices = df.groupby(["start", "end", "percentile"], as_index=False)["f1"].mean()
    peak_per_shell = mean_over_slices.groupby(["start", "end"], as_index=False)["f1"].max()
    best_shell_row = peak_per_shell.sort_values("f1", ascending=False).iloc[0]
    best_shell = (int(best_shell_row["start"]), int(best_shell_row["end"]))

    d = df[(df["start"] == best_shell[0]) & (df["end"] == best_shell[1])].copy()
    percentiles = sorted(d["percentile"].unique().tolist())

    means_p, means_r, means_f = [], [], []
    stds_p, stds_r, stds_f = [], [], []

    for p in percentiles:
        dp = d[d["percentile"] == p]
        per_slice = dp.groupby("slice", as_index=False)[["precision", "recall", "f1"]].mean()

        means_p.append(per_slice["precision"].mean())
        means_r.append(per_slice["recall"].mean())
        means_f.append(per_slice["f1"].mean())

        stds_p.append(per_slice["precision"].std(ddof=1) if len(per_slice) > 1 else 0.0)
        stds_r.append(per_slice["recall"].std(ddof=1) if len(per_slice) > 1 else 0.0)
        stds_f.append(per_slice["f1"].std(ddof=1) if len(per_slice) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(percentiles, means_p, yerr=stds_p, marker="s", linewidth=2.5, capsize=4, label="Precision")
    ax.errorbar(percentiles, means_r, yerr=stds_r, marker="^", linewidth=2.5, capsize=4, label="Recall")
    ax.errorbar(percentiles, means_f, yerr=stds_f, marker="o", linewidth=2.5, capsize=4, label="F1")

    ax.set_title(f"Metrics vs Percentile\nBest shell [{best_shell[0]},{best_shell[1]}] | Feature {feature}",
                 fontsize=20, fontweight="bold", pad=12)
    ax.set_xlabel("Percentile (%)", fontsize=20, fontweight="bold")
    ax.set_ylabel("Score", fontsize=20, fontweight="bold")

    ax.tick_params(axis="both", labelsize=18)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=1.0)
    ax.legend(fontsize=16, frameon=True)

    plt.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()

    return best_shell


# =============================================================================
# CONFUSION MATRICES
# =============================================================================

def _compute_prf(TP, FP, FN, TN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    return precision, recall, f1, acc

def _pick_best_shell_and_three_percentiles(summary_df, feature="pct_bright", preferred=(60, 70, 75)):
    sub = summary_df[summary_df["feature"] == feature].copy()
    if sub.empty:
        return None, None

    shell_rank = sub.groupby(["start", "end"])["f1"].max().reset_index()
    best_shell = shell_rank.sort_values("f1", ascending=False).iloc[0]
    best_start, best_end = int(best_shell["start"]), int(best_shell["end"])

    sub_shell = sub[(sub["start"] == best_start) & (sub["end"] == best_end)].copy()
    available = set(int(x) for x in sub_shell["percentile"].unique().tolist())

    chosen = [int(p) for p in preferred if int(p) in available]

    if len(chosen) < 3:
        rest = sub_shell[~sub_shell["percentile"].isin(chosen)].sort_values("f1", ascending=False)
        for p in rest["percentile"].tolist():
            p = int(p)
            if p not in chosen:
                chosen.append(p)
            if len(chosen) == 3:
                break

    chosen = chosen[:3]
    return (best_start, best_end), chosen

def _aggregate_confusion_from_all_results(all_results, start, end, percentile, feature="pct_bright"):
    df = all_results[
        (all_results["start"] == int(start)) &
        (all_results["end"] == int(end)) &
        (all_results["percentile"] == int(percentile)) &
        (all_results["feature"] == feature)
    ].copy()

    if df.empty:
        return None

    TP = int(df["TP"].sum())
    FP = int(df["FP"].sum())
    FN = int(df["FN"].sum())
    TN = int(df["TN"].sum())

    precision, recall, f1, acc = _compute_prf(TP, FP, FN, TN)
    cm = np.array([[TP, FP], [FN, TN]], dtype=int)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1, "acc": acc,
        "cm": cm,
    }

def plot_three_confusion_matrices_same_scale(all_results, summary_df, out_path,
                                             feature="pct_bright",
                                             preferred_percentiles=(60, 70, 75),
                                             show_cbar=True):
    shell, ps = _pick_best_shell_and_three_percentiles(
        summary_df, feature=feature, preferred=preferred_percentiles
    )
    if shell is None or ps is None or len(ps) < 3:
        print("Nao foi possivel selecionar 3 percentis para a melhor shell")
        return

    start, end = shell
    mats = []
    titles = []
    annots = []

    for p in ps:
        agg = _aggregate_confusion_from_all_results(all_results, start, end, p, feature=feature)
        if agg is None:
            continue

        TP, FP, FN, TN = agg["TP"], agg["FP"], agg["FN"], agg["TN"]
        cm = agg["cm"]
        total = max(1, TP + FP + FN + TN)

        annot = np.array([
            [f"TP\n{TP}\n({100*TP/total:.1f}%)", f"FP\n{FP}\n({100*FP/total:.1f}%)"],
            [f"FN\n{FN}\n({100*FN/total:.1f}%)", f"TN\n{TN}\n({100*TN/total:.1f}%)"],
        ])

        title = (
            "Confusion matrix\n"
            f"Shell [{start},{end}] px | P{int(p)} | Feature {feature}\n"
            f"F1 {agg['f1']:.3f} | Precision {agg['precision']:.3f} | Recall {agg['recall']:.3f}"
        )

        mats.append(cm)
        annots.append(annot)
        titles.append(title)

    if len(mats) != 3:
        print("Nao foi possivel gerar exatamente 3 matrizes")
        return

    vmax = int(max(m.max() for m in mats))
    vmin = 0

    fig, axes = plt.subplots(1, 3, figsize=(26, 8), constrained_layout=True)

    cmap = plt.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for i, ax in enumerate(axes):
        sns.heatmap(
            mats[i],
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
            linewidths=4,
            linecolor="black",
            annot=annots[i],
            fmt="",
            annot_kws={"size": 24, "weight": "bold"},
            xticklabels=["Predicted positive", "Predicted negative"],
            yticklabels=["Actual positive", "Actual negative"],
        )

        ax.set_title(titles[i], fontsize=22, fontweight="bold", pad=16)
        ax.set_xlabel("Predicted class", fontsize=28, fontweight="bold")
        ax.set_ylabel("True class", fontsize=28, fontweight="bold")

        ax.tick_params(axis="both", labelsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")

    if show_cbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("Count", fontsize=26, fontweight="bold")
        cbar.ax.tick_params(labelsize=24)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.basename(out_path)}")


# =============================================================================
# COMPUTE ONE CONFIG FOR OVERLAY
# =============================================================================

def compute_one_config_for_overlay(slice_key, shell_config, percentile, feature_name):
    tiff_path = TIFF_BY_SLICE[slice_key]
    csv_path = GT_CSV_BY_SLICE[slice_key]

    intensity, mask = load_tiff_channels(tiff_path)
    mask = filter_by_area(mask, AREA_MIN_PX, AREA_MAX_PX)

    reference_labels = set(np.unique(mask).astype(int)) - {0}
    gt_labels = load_positive_labels_from_csv(csv_path, mask)

    helpers = prepare_shell_helpers(mask, assignment_mode=SHELL_ASSIGNMENT_MODE)

    start_dist, end_dist = int(shell_config[0]), int(shell_config[1])
    shell = create_shell_2d_from_helpers(mask, start_dist, end_dist, helpers)

    intensity_in_shell = intensity[shell > 0]
    if intensity_in_shell.size == 0:
        return None

    bright_thr = float(np.percentile(intensity_in_shell, int(percentile)))
    labels_present, df_feat = compute_shell_features(intensity, shell, bright_thr)
    if df_feat is None or len(labels_present) < 2:
        return None

    df_feat = df_feat.sort_values("label").reset_index(drop=True)
    df_feat["gt_positive"] = df_feat["label"].isin(gt_labels)

    cls = classify_with_otsu(df_feat, feature_name)
    if cls is None:
        return None
    _, pred_pos = cls

    metrics = compute_metrics(pred_pos, gt_labels, reference_labels)

    df_labels = df_feat[["label", "pct_bright"]].copy()
    df_labels["pred_positive"] = df_labels["label"].isin(pred_pos)
    df_labels["category"] = "TN"
    df_labels.loc[df_labels["pred_positive"] & df_feat["gt_positive"], "category"] = "TP"
    df_labels.loc[df_labels["pred_positive"] & (~df_feat["gt_positive"]), "category"] = "FP"
    df_labels.loc[(~df_labels["pred_positive"]) & df_feat["gt_positive"], "category"] = "FN"

    labels_by_cat = {
        "TP": set(df_labels[df_labels["category"] == "TP"]["label"].astype(int)),
        "FP": set(df_labels[df_labels["category"] == "FP"]["label"].astype(int)),
        "FN": set(df_labels[df_labels["category"] == "FN"]["label"].astype(int)),
        "TN": set(df_labels[df_labels["category"] == "TN"]["label"].astype(int)),
    }

    prop_map = label_props(mask)

    return {
        "intensity": intensity,
        "mask": mask,
        "shell": shell,
        "bright_thr": bright_thr,
        "labels_by_cat": labels_by_cat,
        "df_labels": df_labels,
        "prop_map": prop_map,
        "metrics": metrics,
        "start": start_dist,
        "end": end_dist,
        "percentile": int(percentile),
        "feature": feature_name,
        "slice": int(slice_key),
    }


# =============================================================================
# PROCESS SLICE
# =============================================================================

def process_slice(slice_key):
    print(f"\n{'=' * 80}\nSLICE {slice_key}\n{'=' * 80}")

    tiff_path = TIFF_BY_SLICE[slice_key]
    csv_path = GT_CSV_BY_SLICE[slice_key]

    intensity, mask = load_tiff_channels(tiff_path)
    mask = filter_by_area(mask, AREA_MIN_PX, AREA_MAX_PX)

    reference_labels = set(np.unique(mask).astype(int)) - {0}
    gt_labels = load_positive_labels_from_csv(csv_path, mask)

    print(f"ROIs {len(reference_labels)}  Positive {len(gt_labels)}")
    print(f"Shells to test {len(SHELL_CONFIGS)}  Assignment {SHELL_ASSIGNMENT_MODE}")

    helpers = prepare_shell_helpers(mask, assignment_mode=SHELL_ASSIGNMENT_MODE)

    results_rows = []
    roc_data = []

    for (start_dist, end_dist) in SHELL_CONFIGS:
        shell = create_shell_2d_from_helpers(mask, start_dist, end_dist, helpers)

        for p in PERCENTILES:
            intensity_in_shell = intensity[shell > 0]
            if intensity_in_shell.size == 0:
                continue

            bright_thr = float(np.percentile(intensity_in_shell, p))

            labels_present, df_feat = compute_shell_features(intensity, shell, bright_thr)
            if df_feat is None or len(labels_present) < 2:
                continue

            df_feat = df_feat.sort_values("label").reset_index(drop=True)
            df_feat["gt_positive"] = df_feat["label"].isin(gt_labels)

            for feature_name in FEATURES:
                roc = roc_auc(df_feat, gt_labels, feature_name)
                if roc is not None:
                    fpr, tpr, aucv = roc
                    roc_data.append({
                        "slice": int(slice_key),
                        "shell": (int(start_dist), int(end_dist)),
                        "percentile": int(p),
                        "feature": feature_name,
                        "fpr": fpr,
                        "tpr": tpr,
                        "auc": float(aucv),
                    })

            for feature_name in FEATURES:
                cls = classify_with_otsu(df_feat, feature_name)
                if cls is None:
                    continue
                thr_otsu, pred_pos = cls
                metrics = compute_metrics(pred_pos, gt_labels, reference_labels)

                aucv = np.nan
                for c in reversed(roc_data):
                    if (c["slice"] == int(slice_key) and c["feature"] == feature_name
                        and c["shell"] == (int(start_dist), int(end_dist)) and c["percentile"] == int(p)):
                        aucv = float(c["auc"])
                        break

                results_rows.append({
                    "slice": int(slice_key),
                    "start": int(start_dist),
                    "end": int(end_dist),
                    "percentile": int(p),
                    "bright_threshold": float(bright_thr),
                    "feature": feature_name,
                    "otsu_threshold": float(thr_otsu),
                    **metrics,
                    "auc": float(aucv) if np.isfinite(aucv) else np.nan,
                })

    return pd.DataFrame(results_rows), roc_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("IN SITU FINAL COMPLETE VERSION")
    print("=" * 80)
    print(f"Output {OUT_DIR}")

    all_rows = []
    all_roc_data = []

    for sk in TIFF_BY_SLICE:
        df, roc_data = process_slice(sk)
        all_rows.append(df)
        all_roc_data.extend(roc_data)

    if not all_rows:
        print("No results")
        return

    all_results = pd.concat(all_rows, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "results_all_configurations.csv")
    all_results.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    out_path = os.path.join(FIG_DIR, "metrics_vs_percentile_best_shell_pct_bright.png")
    best_shell = plot_metrics_vs_percentile_best_shell(
        all_results,
        feature="pct_bright",
        out_path=out_path
    )
    print("Best shell usada no plot", best_shell)

    print("\n" + "=" * 80)
    print("TOP SHELL TABLES")
    print("=" * 80)
    for feature in FEATURES:
        out_top_csv = os.path.join(OUT_DIR, f"top{TOP_N_SHELLS_TABLE}_shells_{feature}.csv")
        out_top_tex = os.path.join(OUT_DIR, f"top{TOP_N_SHELLS_TABLE}_shells_{feature}.tex")
        export_top_shells_table(
            all_results,
            feature=feature,
            top_n=TOP_N_SHELLS_TABLE,
            out_csv=out_top_csv,
            out_tex=out_top_tex
        )

    print("\n" + "=" * 80)
    print("F1 HEATMAPS  TOP SHELLS ONLY")
    print("=" * 80)
    for feature in FEATURES:
        out_path = os.path.join(FIG_DIR, f"f1_heatmap_{feature}_top{TOP_N_SHELLS_HEATMAP}.png")
        create_complete_f1_heatmap(all_results, feature, out_path, top_n=TOP_N_SHELLS_HEATMAP)

    summary = all_results.groupby(["start", "end", "percentile", "feature"]).agg({
        "f1": "mean",
        "accuracy": "mean",
        "precision": "mean",
        "recall": "mean",
        "auc": "mean",
    }).reset_index().sort_values("f1", ascending=False)

    summary_path = os.path.join(OUT_DIR, "summary_ranked.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print("\n" + "=" * 80)
    print("CONFUSION MATRICES  SAME SHELL  THREE PERCENTILES")
    print("=" * 80)
    out_path = os.path.join(FIG_DIR, "confusion_best_shell_three_percentiles.png")
    plot_three_confusion_matrices_same_scale(
        all_results=all_results,
        summary_df=summary,
        out_path=out_path,
        feature="pct_bright",
        preferred_percentiles=(60, 70, 75),
        show_cbar=True,
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Outputs {OUT_DIR}")


if __name__ == "__main__":
    main()
