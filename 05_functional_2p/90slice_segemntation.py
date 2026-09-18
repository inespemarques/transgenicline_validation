#!/usr/bin/env python3
"""
SLICE 90 SEGMENTATION & DsRed CLASSIFICATION PIPELINE
=====================================================

1. Load functional plane 90 → mean, median, max projections
2. Segment each with Cellpose (cyto3, diameter 8, cellprob -5, flow 0.7)
3. Classify DsRed+/- using best parameters (P75, Entire slice, Per-block Otsu)
4. Generate thesis figures for each projection method

Output: 3 PNG figures (one per projection method), each showing:
  - Functional projection
  - ROI contours on functional
  - ROI contours on DsRed (green=positive, magenta=negative)
"""

import numpy as np
from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # === PATHS (from your registration pipeline) ===
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    DSRED_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\dsred_registered_SyN.tif"
    FUNC_TEMPLATE = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\functional_template.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\slice90_analysisoneclick"

    # === SLICE ===
    PLANE_IDX = 90  # 0-based
    PLANE_PATTERN = "aligned_p{:d}_nan.tif"  # 1-based in filename

    # === CELLPOSE ===
    CELLPOSE_MODEL = "cyto3"
    CELLPOSE_DIAMETER = 8
    CELLPOSE_CELLPROB = -5
    CELLPOSE_FLOW_THRESHOLD = 0.7

    # === DsRed CLASSIFICATION (best parameters) ===
    DSRED_PERCENTILE = 75
    AREA_MIN_PX = 0
    AREA_MAX_PX = 10000000

    # === FIGURE ===
    FONT_SIZE = 18
    FIGURE_DPI = 300


cfg = Config()

# =============================================================================
# STEP 1: LOAD FUNCTIONAL PLANE AND CREATE PROJECTIONS
# =============================================================================

def load_plane_frames(plane_idx):
    """Load all frames from a single plane TIF."""
    # Filename is 1-based
    fname = cfg.PLANE_PATTERN.format(plane_idx + 1)
    fpath = Path(cfg.FUNC_TIFS_DIR) / fname

    if not fpath.exists():
        raise FileNotFoundError(f"Plane file not found: {fpath}")

    print(f"Loading plane {plane_idx} from: {fpath.name}")

    with TiffFile(str(fpath)) as tif:
        frames = []
        for page in tif.pages:
            frame = page.asarray().astype(np.float32)
            if not np.any(np.isnan(frame)):
                frames.append(frame)

    stack = np.array(frames)
    print(f"  Loaded {stack.shape[0]} valid frames, shape {stack.shape[1:]}")
    return stack


def compute_projections(stack):
    """Compute mean, median, and max projections."""
    projections = {
        'mean': np.mean(stack, axis=0).astype(np.float32),
        'median': np.median(stack, axis=0).astype(np.float32),
        'max': np.max(stack, axis=0).astype(np.float32),
    }
    for name, proj in projections.items():
        print(f"  {name}: range [{proj.min():.1f}, {proj.max():.1f}]")
    return projections


# =============================================================================
# STEP 2: CELLPOSE SEGMENTATION
# =============================================================================

def segment_with_cellpose(image, label=""):
    """
    Segment using Cellpose3 with one-click image restoration.
    
    The CellposeDenoiseModel chains:
      1. Image restoration (oneclick_cyto3 = denoise + deblur + upsample)
      2. Segmentation (cyto3)
    This ensures the restored image is optimised for segmentation,
    not just for visual quality (Stringer & Pachitariu, Nat Methods 2025).
    """
    from cellpose import denoise

    print(f"\nSegmenting {label}...")
    print(f"  Restore: oneclick_{cfg.CELLPOSE_MODEL}")
    print(f"  Segment: {cfg.CELLPOSE_MODEL}, diameter: {cfg.CELLPOSE_DIAMETER}")
    print(f"  cellprob: {cfg.CELLPOSE_CELLPROB}, flow: {cfg.CELLPOSE_FLOW_THRESHOLD}")

    model = denoise.CellposeDenoiseModel(
        gpu=True,
        model_type=cfg.CELLPOSE_MODEL,
        restore_type=f"oneclick_{cfg.CELLPOSE_MODEL}",
    )

    masks, flows, styles, diams = model.eval(
        image,
        diameter=cfg.CELLPOSE_DIAMETER,
        cellprob_threshold=cfg.CELLPOSE_CELLPROB,
        flow_threshold=cfg.CELLPOSE_FLOW_THRESHOLD,
        channels=[0, 0],
    )

    n_cells = len(np.unique(masks)) - 1
    print(f"  ✓ Detected {n_cells} cells")
    return masks


# =============================================================================
# STEP 3: DsRed CLASSIFICATION (Entire slice + Per-block Otsu, P75)
# =============================================================================

def classify_dsred(dsred_slice, masks, percentile=75):
    """
    Classify ROIs as DsRed+/- using:
      - Bright-voxel fraction
      - Entire slice percentile (P75)
      - Otsu on the f_bright distribution
    
    Note: Per-block Otsu reduces to single-slice Otsu for a 2D slice.
    """
    ds = dsred_slice.astype(np.float32)
    m = masks.astype(np.int32)

    # Bright threshold from entire slice (pixels > 0)
    ds_positive = ds[ds > 0]
    if ds_positive.size == 0:
        ds_positive = ds.ravel()
    bright_thr = float(np.percentile(ds_positive, percentile))
    print(f"\n  DsRed classification:")
    print(f"    Bright threshold (P{percentile}): {bright_thr:.1f}")

    # Compute bright-voxel fraction per ROI
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

    # Filter by area
    f_bright = {}
    for lab in labels_unique:
        area = counts[lab]
        if cfg.AREA_MIN_PX <= area <= cfg.AREA_MAX_PX:
            f_bright[lab] = bright_counts[lab] / max(area, 1) * 100.0

    if len(f_bright) == 0:
        print("    ⚠️  No ROIs pass area filter, relaxing constraints...")
        for lab in labels_unique:
            area = counts[lab]
            if area > 10:  # minimal filter for 2D
                f_bright[lab] = bright_counts[lab] / max(area, 1) * 100.0

    # Otsu on f_bright distribution
    vals = np.array(list(f_bright.values()), dtype=np.float32)
    otsu_thr = threshold_otsu(vals)
    print(f"    Otsu threshold: {otsu_thr:.2f}%")

    # Classify
    positive_labels = set()
    negative_labels = set()
    for lab, fbr in f_bright.items():
        if fbr >= otsu_thr:
            positive_labels.add(lab)
        else:
            negative_labels.add(lab)

    print(f"    DsRed+: {len(positive_labels)}, DsRed-: {len(negative_labels)}")
    return positive_labels, negative_labels, otsu_thr


# =============================================================================
# STEP 4: THESIS FIGURES
# =============================================================================

def normalize_for_display(img, plow=0.5, phigh=99.7):
    """Robust normalization for display."""
    v = img[img > 0]
    if v.size == 0:
        return np.zeros_like(img)
    vmin, vmax = np.percentile(v, [plow, phigh])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def create_thesis_figure(func_img, dsred_img, masks, positive_labels, negative_labels,
                         projection_name, output_path):
    """
    Create 3-panel thesis figure:
      Panel A: Functional projection
      Panel B: ROI contours on functional
      Panel C: ROIs on DsRed colored by classification
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fs = cfg.FONT_SIZE

    func_norm = normalize_for_display(func_img)
    dsred_norm = normalize_for_display(dsred_img)

    # --- Panel A: Functional ---
    axes[0].imshow(func_norm, cmap='gray', interpolation='bilinear')
    axes[0].set_title(f'Functional ({projection_name})', fontsize=fs, fontweight='bold')
    axes[0].axis('off')

    # --- Panel B: ROI contours on functional ---
    boundaries = find_boundaries(masks, mode='thick')
    func_rgb = np.stack([func_norm]*3, axis=-1)
    func_rgb[boundaries] = [1, 1, 0]  # yellow contours
    axes[1].imshow(func_rgb, interpolation='bilinear')
    axes[1].set_title(f'Segmentation ({len(positive_labels)+len(negative_labels)} ROIs)',
                      fontsize=fs, fontweight='bold')
    axes[1].axis('off')

    # --- Panel C: Classification on DsRed ---
    dsred_rgb = np.stack([dsred_norm]*3, axis=-1).copy()

    # Color ROIs: green = positive, magenta = negative
    for lab in positive_labels:
        roi_mask = masks == lab
        boundary = find_boundaries(roi_mask, mode='thick')
        dsred_rgb[boundary, 0] = 0.0   # green
        dsred_rgb[boundary, 1] = 1.0
        dsred_rgb[boundary, 2] = 0.0

    for lab in negative_labels:
        roi_mask = masks == lab
        boundary = find_boundaries(roi_mask, mode='thick')
        dsred_rgb[boundary, 0] = 1.0   # magenta
        dsred_rgb[boundary, 1] = 0.0
        dsred_rgb[boundary, 2] = 1.0

    axes[2].imshow(dsred_rgb, interpolation='bilinear')
    axes[2].set_title('DsRed Classification', fontsize=fs, fontweight='bold')
    axes[2].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label=f'DsRed+ ({len(positive_labels)})'),
        Patch(facecolor='magenta', edgecolor='magenta', label=f'DsRed− ({len(negative_labels)})'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=fs-2, frameon=True, bbox_to_anchor=(0.5, -0.02))

    # Panel labels
    for i, letter in enumerate(['A', 'B', 'C']):
        axes[i].text(0.02, 0.98, letter, transform=axes[i].transAxes,
                     fontsize=fs+4, fontweight='bold', color='white',
                     va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {Path(output_path).name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SLICE 90: SEGMENTATION & DsRed CLASSIFICATION")
    print("=" * 80)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'masks').mkdir(exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)
    (out_dir / 'projections').mkdir(exist_ok=True)

    # ----- 1. Load plane frames and compute projections -----
    print("\n--- STEP 1: Projections ---")
    stack = load_plane_frames(cfg.PLANE_IDX)
    projections = compute_projections(stack)

    for name, proj in projections.items():
        imwrite(str(out_dir / 'projections' / f'plane90_{name}.tif'), proj)

    # ----- 2. Load registered DsRed slice 90 -----
    print("\n--- STEP 2: Load registered DsRed ---")
    dsred_vol = imread(cfg.DSRED_REGISTERED).astype(np.float32)
    print(f"  DsRed volume: {dsred_vol.shape}")

    if cfg.PLANE_IDX < dsred_vol.shape[0]:
        dsred_slice = dsred_vol[cfg.PLANE_IDX]
    else:
        raise ValueError(f"Plane {cfg.PLANE_IDX} out of range for DsRed ({dsred_vol.shape[0]})")
    print(f"  DsRed slice 90: {dsred_slice.shape}")

    # ----- 3. Segment each projection -----
    print("\n--- STEP 3: Cellpose Segmentation ---")
    masks_dict = {}
    for name, proj in projections.items():
        masks = segment_with_cellpose(proj, label=f"{name} projection")
        masks_dict[name] = masks
        imwrite(str(out_dir / 'masks' / f'masks_{name}.tif'), masks.astype(np.uint16))

    # ----- 4. Classify and generate figures -----
    print("\n--- STEP 4: Classification & Figures ---")
    for name in projections:
        print(f"\n{'='*60}")
        print(f"Processing: {name} projection")
        print(f"{'='*60}")

        positive, negative, otsu_thr = classify_dsred(
            dsred_slice, masks_dict[name], cfg.DSRED_PERCENTILE
        )

        create_thesis_figure(
            projections[name], dsred_slice, masks_dict[name],
            positive, negative, name,
            out_dir / 'figures' / f'slice90_{name}_classification.png'
        )

    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"Output: {out_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()