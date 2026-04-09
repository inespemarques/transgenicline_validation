#!/usr/bin/env python3
"""
SLICE 90 — COMPLETE ANALYSIS PIPELINE v2
=========================================

Generates:
  1. Registration overlay: GCaMP functional vs GCaMP anatomical (magenta/green)
  2. Main thesis figure (2x2): GCaMP func | DsRed registered
                                ROI contours | DsRed classification
  3. Segmentation with multiple strategies to maximise ROI count
  4. Segmentation quality metrics

Strategies to capture more ROIs:
  - Mean / median / max projections
  - Lower cellprob (-5, -6, -8)
  - Smaller diameter (6, 7, 8)
  - With/without oneclick restore
"""

import numpy as np
from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
import pandas as pd
import json
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # === PATHS (from registration pipeline) ===
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    DSRED_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\dsred_registered_SyN.tif"
    GCAMP_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\gcamp_registered_SyN.tif"
    FUNC_TEMPLATE = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\functional_template.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\slice60_analysis_v2"

    # === SLICE ===
    PLANE_IDX = 60  # 0-based
    PLANE_PATTERN = "aligned_p{:d}_nan.tif"  # 1-based in filename

    # === CELLPOSE (base) ===
    CELLPOSE_MODEL = "cyto3"
    CELLPOSE_DIAMETER = 7
    CELLPOSE_CELLPROB = -5
    CELLPOSE_FLOW_THRESHOLD = 1.0

    # === STRATEGIES TO TEST (more ROIs) ===
    STRATEGIES = [
        # (name, projection, diameter, cellprob, use_restore)
        ("mean_d8_cp-5_restore",   "mean",   8, -5, True),
        ("mean_d7_cp-5_restore",   "mean",   7, -5, True),
        ("mean_d6_cp-5_restore",   "mean",   6, -5, True),
        ("mean_d8_cp-6_restore",   "mean",   8, -6, True),
        ("mean_d8_cp-8_restore",   "mean",   8, -8, True),
        ("median_d8_cp-5_restore", "median", 8, -5, True),
        ("median_d7_cp-5_restore", "median", 7, -5, True),
        ("max_d8_cp-5_restore",    "max",    8, -5, True),
        ("mean_d8_cp-5_norestore", "mean",   8, -5, False),
    ]

    # === DsRed CLASSIFICATION ===
    DSRED_PERCENTILE = 75

    # === FIGURE ===
    FONT_SIZE = 18
    FIGURE_DPI = 300


cfg = Config()


# =============================================================================
# LOAD DATA
# =============================================================================

def load_plane_frames(plane_idx):
    """Load all frames from a single plane TIF."""
    fname = cfg.PLANE_PATTERN.format(plane_idx + 1)  # 1-based filename
    fpath = Path(cfg.FUNC_TIFS_DIR) / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")

    print(f"  Loading plane {plane_idx} from: {fpath.name}")
    with TiffFile(str(fpath)) as tif:
        frames = []
        for page in tif.pages:
            frame = page.asarray().astype(np.float32)
            if not np.any(np.isnan(frame)):
                frames.append(frame)

    stack = np.array(frames)
    print(f"    {stack.shape[0]} valid frames, shape {stack.shape[1:]}")
    return stack


def compute_projections(stack):
    """Compute mean, median, max projections."""
    return {
        'mean': np.mean(stack, axis=0).astype(np.float32),
        'median': np.median(stack, axis=0).astype(np.float32),
        'max': np.max(stack, axis=0).astype(np.float32),
    }


def normalize_for_display(img, plow=0.5, phigh=99.7):
    v = img[img > 0]
    if v.size == 0:
        return np.zeros_like(img)
    vmin, vmax = np.percentile(v, [plow, phigh])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


# =============================================================================
# FIGURE 1: REGISTRATION OVERLAY (GCaMP functional vs anatomical)
# =============================================================================

def create_registration_overlay(func_slice, anat_slice, output_path):
    """
    Overlay: GCaMP functional (green) + GCaMP anatomical registered (magenta).
    Good alignment = white/grey overlap; misalignment = colored fringes.
    """
    fs = cfg.FONT_SIZE

    func_norm = normalize_for_display(func_slice)
    anat_norm = normalize_for_display(anat_slice)

    # RGB overlay: green = functional, magenta = anatomical
    rgb = np.zeros((*func_norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = anat_norm   # R (magenta)
    rgb[..., 1] = func_norm   # G (green)
    rgb[..., 2] = anat_norm   # B (magenta)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel A: Functional
    axes[0].imshow(func_norm, cmap='gray', interpolation='bilinear')
    axes[0].set_title('GCaMP functional', fontsize=fs, fontweight='bold')
    axes[0].axis('off')

    # Panel B: Anatomical registered
    axes[1].imshow(anat_norm, cmap='gray', interpolation='bilinear')
    axes[1].set_title('GCaMP anatomical (registered)', fontsize=fs, fontweight='bold')
    axes[1].axis('off')

    # Panel C: Overlay
    axes[2].imshow(np.clip(rgb, 0, 1), interpolation='bilinear')
    axes[2].set_title('Overlay', fontsize=fs, fontweight='bold')
    axes[2].axis('off')

    legend_elements = [
        Patch(facecolor='green', label='Functional'),
        Patch(facecolor='magenta', label='Anatomical'),
        Patch(facecolor='white', label='Overlap'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=fs - 2, frameon=True, bbox_to_anchor=(0.5, -0.02))

    for i, letter in enumerate(['A', 'B', 'C']):
        axes[i].text(0.02, 0.98, letter, transform=axes[i].transAxes,
                     fontsize=fs + 4, fontweight='bold', color='white', va='top',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {Path(output_path).name}")


# =============================================================================
# FIGURE 2: MAIN 2x2 THESIS FIGURE
# =============================================================================

def create_2x2_thesis_figure(func_img, dsred_img, masks, positive, negative,
                             projection_name, output_path):
    """
    2x2 layout:
      Top-left:     GCaMP functional
      Top-right:    DsRed registered
      Bottom-left:  ROI contours on GCaMP
      Bottom-right: DsRed classification (green=+, magenta=-)
    """
    fs = cfg.FONT_SIZE
    func_norm = normalize_for_display(func_img)
    dsred_norm = normalize_for_display(dsred_img)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # --- A: GCaMP functional ---
    axes[0, 0].imshow(func_norm, cmap='gray', interpolation='bilinear')
    axes[0, 0].set_title(f'GCaMP functional ({projection_name})', fontsize=fs, fontweight='bold')
    axes[0, 0].axis('off')

    # --- B: DsRed registered ---
    axes[0, 1].imshow(dsred_norm, cmap='gray', interpolation='bilinear')
    axes[0, 1].set_title('DsRed registered', fontsize=fs, fontweight='bold')
    axes[0, 1].axis('off')

    # --- C: ROI contours on functional ---
    boundaries = find_boundaries(masks, mode='thick')
    func_rgb = np.stack([func_norm] * 3, axis=-1).copy()
    func_rgb[boundaries] = [1, 1, 0]  # yellow
    n_total = len(positive) + len(negative)
    axes[1, 0].imshow(func_rgb, interpolation='bilinear')
    axes[1, 0].set_title(f'Segmentation ({n_total} ROIs)', fontsize=fs, fontweight='bold')
    axes[1, 0].axis('off')

    # --- D: Classification on DsRed ---
    dsred_rgb = np.stack([dsred_norm] * 3, axis=-1).copy()
    for lab in positive:
        roi = masks == lab
        b = find_boundaries(roi, mode='thick')
        dsred_rgb[b, 0] = 0; dsred_rgb[b, 1] = 1; dsred_rgb[b, 2] = 0
    for lab in negative:
        roi = masks == lab
        b = find_boundaries(roi, mode='thick')
        dsred_rgb[b, 0] = 1; dsred_rgb[b, 1] = 0; dsred_rgb[b, 2] = 1

    axes[1, 1].imshow(dsred_rgb, interpolation='bilinear')
    axes[1, 1].set_title('DsRed classification', fontsize=fs, fontweight='bold')
    axes[1, 1].axis('off')

    # Panel labels
    for i, (r, c, letter) in enumerate([(0, 0, 'A'), (0, 1, 'B'), (1, 0, 'C'), (1, 1, 'D')]):
        axes[r, c].text(0.02, 0.98, letter, transform=axes[r, c].transAxes,
                        fontsize=fs + 4, fontweight='bold', color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Legend
    fig.legend(
        handles=[
            Patch(facecolor='green', label=f'DsRed+ ({len(positive)})'),
            Patch(facecolor='magenta', label=f'DsRed− ({len(negative)})'),
            Patch(facecolor='yellow', label='ROI contours'),
        ],
        loc='lower center', ncol=3, fontsize=fs - 2, frameon=True,
        bbox_to_anchor=(0.5, -0.01)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {Path(output_path).name}")


# =============================================================================
# CELLPOSE SEGMENTATION (with/without restore)
# =============================================================================

def segment_with_restore(image, diameter, cellprob, use_restore=True):
    """Segment with optional oneclick restore."""
    if use_restore:
        from cellpose import denoise
        model = denoise.CellposeDenoiseModel(
            gpu=True,
            model_type=cfg.CELLPOSE_MODEL,
            restore_type=f"oneclick_{cfg.CELLPOSE_MODEL}",
        )
    else:
        from cellpose import models
        model = models.Cellpose(model_type=cfg.CELLPOSE_MODEL, gpu=True)

    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        cellprob_threshold=cellprob,
        flow_threshold=cfg.CELLPOSE_FLOW_THRESHOLD,
        channels=[0, 0],
    )
    return masks


# =============================================================================
# DsRed CLASSIFICATION
# =============================================================================

def classify_dsred(dsred_slice, masks, percentile=75):
    """Classify ROIs using bright-voxel fraction + Otsu."""
    ds = dsred_slice.astype(np.float32)
    m = masks.astype(np.int32)

    ds_positive = ds[ds > 0]
    if ds_positive.size == 0:
        ds_positive = ds.ravel()
    bright_thr = float(np.percentile(ds_positive, percentile))

    labels_unique = np.unique(m)
    labels_unique = labels_unique[labels_unique > 0]

    flat_m = m.ravel()
    flat_ds = ds.ravel()
    roi_mask = flat_m > 0
    labs = flat_m[roi_mask].astype(np.int64)
    ds_vals = flat_ds[roi_mask]

    maxlab = int(labs.max())
    counts = np.bincount(labs, minlength=maxlab + 1)
    bright_flags = (ds_vals > bright_thr).astype(np.float32)
    bright_counts = np.bincount(labs, weights=bright_flags, minlength=maxlab + 1)

    f_bright = {}
    for lab in labels_unique:
        area = counts[lab]
        if area > 10:  # minimal filter for 2D single slice
            f_bright[lab] = bright_counts[lab] / max(area, 1) * 100.0

    vals = np.array(list(f_bright.values()), dtype=np.float32)
    otsu_thr = threshold_otsu(vals)

    positive = {lab for lab, fbr in f_bright.items() if fbr >= otsu_thr}
    negative = {lab for lab, fbr in f_bright.items() if fbr < otsu_thr}

    return positive, negative, otsu_thr, f_bright


# =============================================================================
# SEGMENTATION QUALITY METRICS
# =============================================================================

def compute_segmentation_metrics(masks, func_img):
    """
    Compute metrics to assess segmentation quality.
    
    Without manual GT for this slice, we use proxy metrics:
      - n_cells: total ROIs detected
      - coverage: fraction of image area covered by masks
      - mean_area, std_area: size distribution (should be tight for nuclei)
      - circularity: nuclear masks should be roughly circular
      - mean_snr: signal-to-noise ratio (ROI intensity vs background)
    """
    props = regionprops(masks.astype(np.int32), intensity_image=func_img)

    if len(props) == 0:
        return {'n_cells': 0}

    areas = [p.area for p in props]
    circularities = [4 * np.pi * p.area / (p.perimeter ** 2 + 1e-8) for p in props]
    mean_intensities = [p.intensity_mean for p in props]

    # Background intensity
    bg_mask = masks == 0
    bg_mean = float(func_img[bg_mask].mean()) if bg_mask.any() else 0
    bg_std = float(func_img[bg_mask].std()) if bg_mask.any() else 1

    # SNR per ROI
    snrs = [(mi - bg_mean) / (bg_std + 1e-8) for mi in mean_intensities]

    # Coverage
    total_px = masks.shape[0] * masks.shape[1]
    mask_px = np.sum(masks > 0)

    metrics = {
        'n_cells': len(props),
        'coverage_pct': float(mask_px / total_px * 100),
        'mean_area_px': float(np.mean(areas)),
        'std_area_px': float(np.std(areas)),
        'median_area_px': float(np.median(areas)),
        'mean_circularity': float(np.mean(circularities)),
        'std_circularity': float(np.std(circularities)),
        'mean_snr': float(np.mean(snrs)),
        'median_snr': float(np.median(snrs)),
        'pct_area_lt50': float(np.mean([a < 50 for a in areas]) * 100),  # very small = debris?
        'pct_area_gt500': float(np.mean([a > 500 for a in areas]) * 100),  # very large = merged?
    }
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SLICE 90 — COMPLETE ANALYSIS v2")
    print("=" * 80)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ['figures', 'masks', 'projections', 'metrics']:
        (out_dir / d).mkdir(exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n--- Loading data ---")
    stack = load_plane_frames(cfg.PLANE_IDX)
    projections = compute_projections(stack)

    for name, proj in projections.items():
        imwrite(str(out_dir / 'projections' / f'plane90_{name}.tif'), proj)

    dsred_vol = imread(cfg.DSRED_REGISTERED).astype(np.float32)
    dsred_slice = dsred_vol[cfg.PLANE_IDX]
    print(f"  DsRed slice 90: {dsred_slice.shape}")

    # Load GCaMP anatomical registered + functional template
    gcamp_anat_vol = imread(cfg.GCAMP_REGISTERED).astype(np.float32)
    gcamp_anat_slice = gcamp_anat_vol[cfg.PLANE_IDX]

    func_template_vol = imread(cfg.FUNC_TEMPLATE).astype(np.float32)
    func_template_slice = func_template_vol[cfg.PLANE_IDX]
    print(f"  GCaMP anat slice 90: {gcamp_anat_slice.shape}")
    print(f"  Func template slice 90: {func_template_slice.shape}")

    # ── 2. Registration overlay ───────────────────────────────────────
    print("\n--- Registration overlay ---")
    create_registration_overlay(
        func_template_slice, gcamp_anat_slice,
        out_dir / 'figures' / 'registration_overlay_slice90.png'
    )

    # ── 3. Strategy sweep ─────────────────────────────────────────────
    print("\n--- Strategy sweep ---")
    all_results = []

    for name, proj_type, diam, cprob, use_restore in cfg.STRATEGIES:
        print(f"\n{'─'*60}")
        print(f"  Strategy: {name}")
        print(f"  Proj={proj_type}, diam={diam}, cellprob={cprob}, restore={use_restore}")
        print(f"{'─'*60}")

        img = projections[proj_type]

        try:
            masks = segment_with_restore(img, diam, cprob, use_restore)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        # Classify
        positive, negative, otsu_thr, f_bright = classify_dsred(
            dsred_slice, masks, cfg.DSRED_PERCENTILE
        )

        # Metrics
        seg_metrics = compute_segmentation_metrics(masks, img)

        result = {
            'strategy': name,
            'projection': proj_type,
            'diameter': diam,
            'cellprob': cprob,
            'restore': use_restore,
            'n_rois': seg_metrics['n_cells'],
            'n_dsred_pos': len(positive),
            'n_dsred_neg': len(negative),
            'otsu_threshold': float(otsu_thr),
            **seg_metrics,
        }
        all_results.append(result)

        print(f"  ROIs: {seg_metrics['n_cells']}, DsRed+: {len(positive)}, DsRed−: {len(negative)}")
        print(f"  Area: {seg_metrics['mean_area_px']:.0f} ± {seg_metrics['std_area_px']:.0f} px")
        print(f"  Circularity: {seg_metrics['mean_circularity']:.3f}")
        print(f"  SNR: {seg_metrics['mean_snr']:.1f}")

        # Save masks
        imwrite(str(out_dir / 'masks' / f'masks_{name}.tif'), masks.astype(np.uint16))

        # Generate 2x2 figure
        create_2x2_thesis_figure(
            img, dsred_slice, masks, positive, negative, proj_type,
            out_dir / 'figures' / f'{name}_2x2.png'
        )

    # ── 4. Save comparison table ──────────────────────────────────────
    print("\n--- Saving comparison ---")
    df = pd.DataFrame(all_results)
    df = df.sort_values('n_rois', ascending=False)
    df.to_csv(out_dir / 'metrics' / 'strategy_comparison.csv', index=False)

    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON (sorted by n_rois):")
    print(f"{'='*80}")
    print(df[['strategy', 'n_rois', 'n_dsred_pos', 'n_dsred_neg',
              'mean_area_px', 'mean_circularity', 'mean_snr']].to_string(index=False))

    # Best strategy
    best = df.iloc[0]
    print(f"\n→ BEST: {best['strategy']} with {best['n_rois']} ROIs")

    print(f"\n{'='*80}")
    print(f"✅ COMPLETE! Output: {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()