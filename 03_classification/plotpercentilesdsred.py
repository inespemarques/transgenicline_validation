#!/usr/bin/env python3
"""
DsRed classification figure with 3 outputs:

  Figure 1 (dsred_roc_and_sweep.png/pdf):
    Left:  ROC curves pooled across slices for 3 features
    Right: Precision/Recall/F1 vs percentile (per-slice Otsu)

  Figure 2 (dsred_distributions.png/pdf):
    3 columns (one per feature) x 3 rows (one per slice)
    Each panel: histogram of per-ROI feature values for GT+ and GT-
    with Otsu threshold marked. Shows that only bright-voxel fraction
    produces bimodal distributions suitable for unsupervised Otsu.

Usage:
    conda activate cellpose_ok_env
    python plot_dsred_roc_and_sweep.py
"""

import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, auc
from skimage.filters import threshold_otsu

# =============================================================================
# PATHS
# =============================================================================
RESULTS_DIR = r"C:\Users\OSVALDO\Downloads\results\03fev\overlays_realGT_otsu_hist_final"
CSV_PATH    = os.path.join(RESULTS_DIR, "confusion_summary_all.csv")
OUT_DIR     = RESULTS_DIR

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

SLICES = [110, 300, 450]

# =============================================================================
# CONFIG
# =============================================================================
AREA_MIN_PX = 400
AREA_MAX_PX = 6000
NEIGHBOR_RADIUS_PX = 1
BRIGHT_PERCENTILE = 75

# Right panel
METHOD = "otsu_per_slice"
PERCENTILES_TO_PLOT = [60, 65, 70, 75, 80, 85, 90]
SELECTED_PERCENTILE = 75
RECALL_CRITERION = 0.80


# =============================================================================
# HELPERS
# =============================================================================

def load_tiff_channels(tiff_path):
    img = tiff.imread(tiff_path)
    if img.ndim == 3 and img.shape[0] == 2:
        dsred, mask = img[0], img[1]
    elif img.ndim == 3 and img.shape[-1] == 2:
        dsred, mask = img[..., 0], img[..., 1]
    else:
        raise ValueError(f"Unexpected TIFF shape {img.shape}")
    return dsred.astype(np.float32), mask.astype(np.int32)


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


def load_gt_labels(gt_csv, mask):
    df = pd.read_csv(gt_csv)
    coords = df[["XM", "YM"]].to_numpy(dtype=float)
    H, W = mask.shape
    x, y = coords[:, 0], coords[:, 1]
    out0 = np.mean((x < 0) | (x >= W) | (y < 0) | (y >= H))
    out1 = np.mean((x-1 < 0) | (x-1 >= W) | (y-1 < 0) | (y-1 >= H))
    if out1 + 1e-6 < out0:
        coords[:, 0] -= 1
        coords[:, 1] -= 1
    labels = set()
    for xc, yc in coords:
        lab = _label_at_or_near(mask, xc, yc, r=NEIGHBOR_RADIUS_PX)
        if lab != 0:
            labels.add(lab)
    return labels


def compute_features_for_slice(dsred, mask, bright_percentile):
    """
    Per ROI in one slice, compute 3 INDEPENDENT features:

      1) Bright-voxel fraction [0-100%]:
           threshold = percentile of ALL non-zero DsRed in ENTIRE slice
           per ROI: (#pixels > threshold) / area * 100
           -> NORMALISED by per-slice percentile

      2) Mean intensity:
           per ROI: sum(DsRed in mask) / area
           -> RAW value, NO percentile

      3) Integrated density:
           per ROI: sum(DsRed in mask)
           -> RAW value, NO percentile
    """
    flat_m = mask.reshape(-1).astype(np.int64)
    flat_ds = dsred.reshape(-1).astype(np.float32)

    roi_mask = flat_m > 0
    labs = flat_m[roi_mask]
    ds_vals = flat_ds[roi_mask]

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1).astype(np.int64)
    present = np.nonzero(counts)[0]
    present = present[present != 0]
    area = counts[present]

    remap = -np.ones(maxlab + 1, dtype=np.int32)
    remap[present] = np.arange(len(present), dtype=np.int32)
    ridx = remap[labs]

    # Feature 1: bright-voxel fraction (per-slice percentile of ENTIRE slice)
    all_nonzero = dsred[dsred > 0].ravel()
    bright_thr = float(np.percentile(all_nonzero, float(bright_percentile)))
    bright_mask = (ds_vals > bright_thr).astype(np.float32)
    bright_per_roi = np.bincount(ridx, weights=bright_mask, minlength=len(present))
    f_bright = (bright_per_roi / np.maximum(area, 1) * 100.0).astype(np.float32)

    # Feature 2: mean intensity (NO percentile)
    sum_per_roi = np.bincount(ridx, weights=ds_vals.astype(np.float64),
                              minlength=len(present))
    mean_int = (sum_per_roi / np.maximum(area, 1)).astype(np.float32)

    # Feature 3: integrated density (NO percentile)
    integ_int = sum_per_roi.astype(np.float32)

    df = pd.DataFrame({
        "label": present.astype(np.int64),
        "area_px": area.astype(np.int64),
        "f_bright": f_bright,
        "mean_int": mean_int,
        "integ_int": integ_int,
    })

    return df, bright_thr


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" DsRed ROC + Sweep + Distributions")
    print("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: Load, compute features per slice
    # ──────────────────────────────────────────────────────────────────
    slice_data = {}   # sl -> DataFrame
    all_dfs = []

    for sl in SLICES:
        print(f"\n  Slice {sl}:")
        dsred, mask = load_tiff_channels(TIFF_BY_SLICE[sl])
        gt_labels = load_gt_labels(GT_CSV_BY_SLICE[sl], mask)

        feat_df, bright_thr = compute_features_for_slice(dsred, mask, BRIGHT_PERCENTILE)
        feat_df = feat_df[(feat_df["area_px"] >= AREA_MIN_PX) &
                          (feat_df["area_px"] <= AREA_MAX_PX)].copy()
        feat_df = feat_df.reset_index(drop=True)
        feat_df["gt_positive"] = feat_df["label"].isin(gt_labels).astype(int)
        feat_df["slice"] = sl

        slice_data[sl] = feat_df
        all_dfs.append(feat_df)

        n_pos = int(feat_df["gt_positive"].sum())
        print(f"    ROIs: {len(feat_df)}  GT+: {n_pos}  GT-: {len(feat_df)-n_pos}")
        print(f"    Bright thr (P{BRIGHT_PERCENTILE} entire slice): {bright_thr:.2f}")
        print(f"    f_bright  GT+: {feat_df.loc[feat_df.gt_positive==1,'f_bright'].mean():.2f}  "
              f"GT-: {feat_df.loc[feat_df.gt_positive==0,'f_bright'].mean():.2f}")
        print(f"    mean_int  GT+: {feat_df.loc[feat_df.gt_positive==1,'mean_int'].mean():.2f}  "
              f"GT-: {feat_df.loc[feat_df.gt_positive==0,'mean_int'].mean():.2f}")
        print(f"    integ_int GT+: {feat_df.loc[feat_df.gt_positive==1,'integ_int'].mean():.0f}  "
              f"GT-: {feat_df.loc[feat_df.gt_positive==0,'integ_int'].mean():.0f}")

    pooled = pd.concat(all_dfs, ignore_index=True)
    y_true = pooled["gt_positive"].to_numpy()

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: POOLED ROC
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(" POOLED ROC (all slices combined)")
    print("=" * 70)

    # Labels: NO percentile for mean/integrated
    features_roc = [
        (f"Bright pixels P{BRIGHT_PERCENTILE}%", "f_bright"),
        ("Mean intensity",                        "mean_int"),
        ("Integrated intensity",                  "integ_int"),
    ]

    roc_results = {}
    for label, col in features_roc:
        scores = pooled[col].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        roc_results[label] = (fpr, tpr, roc_auc)
        print(f"  {label:35s}  AUC = {roc_auc:.3f}")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: Right panel - sweep CSV
    # ──────────────────────────────────────────────────────────────────
    print(f"\n  Reading: {CSV_PATH}")
    sweep_df = pd.read_csv(CSV_PATH)
    sweep_df = sweep_df[sweep_df["method"] == METHOD].copy()
    sweep_df = sweep_df[sweep_df["bright_percentile"].isin(PERCENTILES_TO_PLOT)].copy()

    agg = sweep_df.groupby("bright_percentile").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    ).reset_index().fillna(0.0).sort_values("bright_percentile")

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 1: ROC + Sweep (2-panel)
    # ──────────────────────────────────────────────────────────────────
    fig1, (ax_roc, ax_sweep) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (label, (fpr, tpr, roc_auc)) in enumerate(roc_results.items()):
        ax_roc.plot(fpr, tpr, color=colors[idx], linewidth=2.0,
                    label=f"{label},  AUC {roc_auc:.3f}")

    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.0,
                label='Random', zorder=1)

    ax_roc.set_xlabel('False positive rate', fontsize=12, fontweight='medium')
    ax_roc.set_ylabel('True positive rate', fontsize=12, fontweight='medium')
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.02)
    ax_roc.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_roc.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_roc.tick_params(axis='both', labelsize=10)
    ax_roc.legend(loc='lower right', fontsize=9.5, frameon=True, framealpha=0.9,
                  edgecolor='#cccccc', fancybox=False)
    ax_roc.spines['top'].set_visible(False)
    ax_roc.spines['right'].set_visible(False)
    ax_roc.set_aspect('equal')

    # ── RIGHT: sweep ──
    pcts = agg["bright_percentile"].to_numpy()
    c_f1, c_prec, c_rec = "#1f77b4", "#ff7f0e", "#2ca02c"
    lw, ms = 2.2, 9
    cs, ct, ew = 4, 1.5, 1.5

    ax_sweep.errorbar(pcts, agg["f1_mean"], yerr=agg["f1_std"],
                      fmt='o-', color=c_f1, markerfacecolor=c_f1,
                      markersize=ms, linewidth=lw, capsize=cs,
                      capthick=ct, elinewidth=ew, label='F1', zorder=3)
    ax_sweep.errorbar(pcts, agg["precision_mean"], yerr=agg["precision_std"],
                      fmt='s-', color=c_prec, markerfacecolor=c_prec,
                      markersize=ms, linewidth=lw, capsize=cs,
                      capthick=ct, elinewidth=ew, label='Precision', zorder=3)
    ax_sweep.errorbar(pcts, agg["recall_mean"], yerr=agg["recall_std"],
                      fmt='^-', color=c_rec, markerfacecolor=c_rec,
                      markersize=ms, linewidth=lw, capsize=cs,
                      capthick=ct, elinewidth=ew, label='Recall', zorder=3)

    ax_sweep.axvline(x=SELECTED_PERCENTILE, color='gray', linestyle='--',
                     linewidth=1.2, alpha=0.7, zorder=1)
    ax_sweep.axhline(y=RECALL_CRITERION, color='gray', linestyle=':',
                     linewidth=1.0, alpha=0.5, zorder=1)

    ax_sweep.set_xlabel('Percentile threshold (%)', fontsize=12, fontweight='medium')
    ax_sweep.set_ylabel('Score', fontsize=12, fontweight='medium')
    ax_sweep.set_xlim(pcts.min() - 3, pcts.max() + 3)
    ax_sweep.set_ylim(0.15, 1.02)
    ax_sweep.xaxis.set_major_locator(MultipleLocator(5))
    ax_sweep.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_sweep.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_sweep.tick_params(axis='both', which='major', labelsize=10, length=5)
    ax_sweep.tick_params(axis='both', which='minor', length=3)
    ax_sweep.legend(loc='lower left', fontsize=10.5, frameon=True, framealpha=0.9,
                    edgecolor='#cccccc', fancybox=False)
    ax_sweep.spines['top'].set_visible(False)
    ax_sweep.spines['right'].set_visible(False)
    ax_sweep.grid(axis='y', alpha=0.25, linewidth=0.6)

    fig1.tight_layout(w_pad=3.0)

    out1_png = os.path.join(OUT_DIR, "dsred_roc_and_sweep.png")
    out1_pdf = os.path.join(OUT_DIR, "dsred_roc_and_sweep.pdf")
    fig1.savefig(out1_png, dpi=300, bbox_inches='tight')
    fig1.savefig(out1_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n  Saved: {out1_png}")

    # ──────────────────────────────────────────────────────────────────
    # FIGURE 2: Distributions (3 features × 3 slices = 9 panels)
    # Shows WHY only bright-fraction works with Otsu (bimodal)
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(" Distribution histograms with Otsu thresholds")
    print("=" * 70)

    feature_info = [
        ("f_bright",  f"Bright-voxel fraction (%) [P{BRIGHT_PERCENTILE}]",
         "Fraction of bright pixels (%)"),
        ("mean_int",  "Mean intensity (raw)",
         "Mean DsRed intensity (a.u.)"),
        ("integ_int", "Integrated density (raw)",
         "Integrated DsRed intensity (a.u.)"),
    ]

    fig2, axes = plt.subplots(3, 3, figsize=(16, 12))

    for col_idx, (feat_col, feat_title, feat_xlabel) in enumerate(feature_info):
        for row_idx, sl in enumerate(SLICES):
            ax = axes[row_idx, col_idx]
            df = slice_data[sl]

            vals_pos = df.loc[df["gt_positive"] == 1, feat_col].to_numpy()
            vals_neg = df.loc[df["gt_positive"] == 0, feat_col].to_numpy()
            all_vals = df[feat_col].to_numpy().astype(np.float32)

            # Compute Otsu on this slice's feature values
            try:
                otsu_thr = float(threshold_otsu(all_vals))
            except Exception:
                otsu_thr = float(np.median(all_vals))

            # Classify with Otsu
            pred_pos = all_vals >= otsu_thr
            gt = df["gt_positive"].to_numpy().astype(bool)
            tp = int(np.sum(pred_pos & gt))
            fp = int(np.sum(pred_pos & ~gt))
            fn = int(np.sum(~pred_pos & gt))
            tn = int(np.sum(~pred_pos & ~gt))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)

            print(f"  {feat_col:12s} slice {sl}: Otsu={otsu_thr:.2f}  "
                  f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
                  f"TP={tp} FP={fp} FN={fn} TN={tn}")

            # Histogram
            bins = 50
            # Compute shared range
            vmin = min(all_vals.min(), 0)
            vmax = all_vals.max() * 1.05

            ax.hist(vals_neg, bins=bins, range=(vmin, vmax),
                    alpha=0.6, color="#4a90d9", label=f"GT$-$ (n={len(vals_neg)})",
                    edgecolor='none')
            ax.hist(vals_pos, bins=bins, range=(vmin, vmax),
                    alpha=0.6, color="#e8833a", label=f"GT$+$ (n={len(vals_pos)})",
                    edgecolor='none')

            # Otsu threshold
            ax.axvline(otsu_thr, color='red', linestyle='--', linewidth=1.8,
                       label=f"Otsu = {otsu_thr:.1f}")

            # Metrics annotation
            ax.text(0.97, 0.95,
                    f"P={prec:.2f}  R={rec:.2f}\nF1={f1:.2f}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#aaa', alpha=0.85))

            # Labels
            if row_idx == 0:
                ax.set_title(feat_title, fontsize=11, fontweight='bold', pad=8)
            if row_idx == 2:
                ax.set_xlabel(feat_xlabel, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Slice {sl}\nCount", fontsize=10, fontweight='medium')

            ax.legend(fontsize=8, loc='upper right' if col_idx > 0 else 'upper left',
                      framealpha=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=9)

    fig2.suptitle(
        "Per-slice feature distributions with Otsu thresholds\n"
        "Only bright-voxel fraction produces bimodal distributions "
        "suitable for unsupervised Otsu classification",
        fontsize=12, fontweight='medium', y=1.02)

    fig2.tight_layout()

    out2_png = os.path.join(OUT_DIR, "dsred_distributions.png")
    out2_pdf = os.path.join(OUT_DIR, "dsred_distributions.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches='tight')
    fig2.savefig(out2_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"\n  Saved: {out2_png}")
    print(f"  Saved: {out2_pdf}")

    print("\nDone.")


if __name__ == "__main__":
    main()