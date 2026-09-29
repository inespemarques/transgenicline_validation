#!/usr/bin/env python3
"""
SLICE 90 — COMPLETE ANALYSIS PIPELINE v3 (WITH ZOOM PANELS)
============================================================

NEW FEATURES:
  • Zoom panels showing ROI separation quality
  • DsRed integrity validation (before/after registration)
  • Spatial distribution maps
  • Enhanced metrics and visualizations

Generates:
  1. Registration overlay: GCaMP functional vs GCaMP anatomical (magenta/green)
  2. Main thesis figure (2×2): GCaMP func | DsRed registered
                                ROI contours | DsRed classification
  3. **NEW: Zoom panels** (6 regions showing individual ROIs)
  4. **NEW: DsRed integrity check** (before vs after registration)
  5. Segmentation with multiple strategies to maximise ROI count
  6. Spatial distribution heatmaps
  7. Comprehensive quality metrics
"""

import numpy as np
from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops, label
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
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
    DSRED_ORIGINAL = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"
    GCAMP_REGISTERED = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\gcamp_registered_SyN.tif"
    FUNC_TEMPLATE = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes\functional_template.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\slice60_analysis_v3"

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
        ("median_d8_cp-5_restore", "median", 8, -5, True),
    ]

    # === DsRed CLASSIFICATION ===
    DSRED_PERCENTILE = 75

    # === ZOOM PANELS ===
    N_ZOOM_REGIONS = 4  # Number of zoom regions to show
    ZOOM_SIZE = 120      # Size of zoom window (pixels) - larger for more context
    MIN_DSRED_POS = 3    # Minimum DsRed+ ROIs in a region to be selected

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
# NEW: PROFESSIONAL ZOOM PANELS WITH OVERVIEW + HIGH-RES INDIVIDUAL ZOOMS
# =============================================================================

def select_dsred_rich_regions(dsred_img, masks, positive, negative, n_regions=4, zoom_size=120, min_dsred_pos=3):
    """
    Select regions with high DsRed+ density for zoom visualization.
    Prioritizes areas with bright DsRed signal and multiple positive ROIs.
    
    Returns list of (y_center, x_center, n_dsred_pos, n_total_rois)
    """
    h, w = masks.shape
    props = regionprops(masks)
    
    if len(positive) < min_dsred_pos:
        print(f"  ⚠️  Only {len(positive)} DsRed+ ROIs found, need at least {min_dsred_pos}")
        return []
    
    # Get centroids of ALL ROIs
    all_centroids = np.array([p.centroid for p in props])
    all_labels = [p.label for p in props]
    
    # Grid search for DsRed+ rich regions
    candidates = []
    step = zoom_size // 3  # More overlap for better coverage
    
    for y in range(zoom_size//2, h - zoom_size//2, step):
        for x in range(zoom_size//2, w - zoom_size//2, step):
            # Find ROIs in window
            in_window_mask = (
                (all_centroids[:, 0] >= y - zoom_size//2) &
                (all_centroids[:, 0] < y + zoom_size//2) &
                (all_centroids[:, 1] >= x - zoom_size//2) &
                (all_centroids[:, 1] < x + zoom_size//2)
            )
            
            labels_in_window = [all_labels[i] for i in np.where(in_window_mask)[0]]
            n_dsred_pos = sum(1 for lab in labels_in_window if lab in positive)
            n_total = len(labels_in_window)
            
            # Calculate mean DsRed intensity in window (as a tiebreaker)
            y0, y1 = max(0, y - zoom_size//2), min(h, y + zoom_size//2)
            x0, x1 = max(0, x - zoom_size//2), min(w, x + zoom_size//2)
            mean_dsred = float(dsred_img[y0:y1, x0:x1].mean())
            
            if n_dsred_pos >= min_dsred_pos and n_total >= 5:
                # Score: prioritize DsRed+ count, then brightness, then total ROIs
                score = n_dsred_pos * 100 + mean_dsred * 10 + n_total
                candidates.append((y, x, n_dsred_pos, n_total, score))
    
    if len(candidates) == 0:
        print(f"  ⚠️  No regions found with ≥{min_dsred_pos} DsRed+ ROIs")
        # Fallback: pick regions around DsRed+ centroids
        pos_props = [p for p in props if p.label in positive]
        if len(pos_props) == 0:
            return []
        indices = np.random.choice(len(pos_props), min(n_regions, len(pos_props)), replace=False)
        return [(int(pos_props[i].centroid[0]), int(pos_props[i].centroid[1]), 1, 1) 
                for i in indices]
    
    # Sort by score (highest first)
    candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Select diverse regions (avoid clustering)
    selected = []
    for y, x, n_pos, n_tot, score in candidates:
        if len(selected) == 0:
            selected.append((y, x, n_pos, n_tot))
        else:
            # Check spatial separation
            dists = [np.sqrt((y - sy)**2 + (x - sx)**2) for sy, sx, _, _ in selected]
            if min(dists) > zoom_size * 0.7:  # 70% separation
                selected.append((y, x, n_pos, n_tot))
        
        if len(selected) >= n_regions:
            break
    
    print(f"  Selected {len(selected)} zoom regions with DsRed+ ROIs")
    return selected


def create_professional_zoom_panels(func_img, dsred_img, masks, positive, negative,
                                   output_dir, strategy_name, zoom_size=120):
    """
    Create publication-quality zoom panels:
    1. Overview image with boxes showing zoom locations
    2. Individual high-resolution zoom files
    
    Each zoom shows: GCaMP | DsRed | Classification (3 columns)
    """
    regions = select_dsred_rich_regions(dsred_img, masks, positive, negative, 
                                       n_regions=4, zoom_size=zoom_size, min_dsred_pos=3)
    
    if len(regions) == 0:
        print("  ⚠️  Could not find suitable zoom regions")
        return
    
    func_norm = normalize_for_display(func_img)
    dsred_norm = normalize_for_display(dsred_img)
    
    # ========== PART 1: OVERVIEW WITH BOXES ==========
    fig_overview, ax_overview = plt.subplots(1, 1, figsize=(16, 12))
    
    # Create RGB overview (DsRed classification overlay)
    overview_rgb = np.stack([dsred_norm] * 3, axis=-1).copy()
    
    # Add all ROI contours in faint yellow
    all_boundaries = find_boundaries(masks, mode='thick')
    overview_rgb[all_boundaries] = [0.4, 0.4, 0]  # dim yellow
    
    # Highlight DsRed+ in bright green
    for lab in positive:
        roi = masks == lab
        b = find_boundaries(roi, mode='thick')
        overview_rgb[b, 0] = 0; overview_rgb[b, 1] = 1; overview_rgb[b, 2] = 0
    
    ax_overview.imshow(overview_rgb, interpolation='bilinear')
    ax_overview.set_title(f'{strategy_name} — Zoom Locations', 
                         fontsize=20, fontweight='bold', pad=15)
    ax_overview.axis('off')
    
    # Draw zoom boxes
    colors = ['cyan', 'yellow', 'magenta', 'lime', 'orange', 'red']
    for idx, (cy, cx, n_pos, n_tot) in enumerate(regions):
        y0 = max(0, cy - zoom_size//2)
        y1 = min(masks.shape[0], cy + zoom_size//2)
        x0 = max(0, cx - zoom_size//2)
        x1 = min(masks.shape[1], cx + zoom_size//2)
        
        color = colors[idx % len(colors)]
        rect = Rectangle((x0, y0), x1-x0, y1-y0, 
                        linewidth=3, edgecolor=color, facecolor='none')
        ax_overview.add_patch(rect)
        
        # Add label
        ax_overview.text(x0 + 5, y0 + 15, f'Zoom {idx+1}',
                        fontsize=14, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7))
    
    # Legend
    legend_elements = [
        Patch(facecolor='green', edgecolor='green', label=f'DsRed+ ({len(positive)})'),
        Patch(facecolor='none', edgecolor='yellow', label=f'All ROIs ({len(positive) + len(negative)})'),
    ]
    ax_overview.legend(handles=legend_elements, loc='upper right', 
                      fontsize=14, frameon=True, fancybox=True)
    
    overview_path = output_dir / f'{strategy_name}_zoom_overview.png'
    plt.savefig(overview_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved overview: {overview_path.name}")
    
    # ========== PART 2: INDIVIDUAL HIGH-RES ZOOMS ==========
    for idx, (cy, cx, n_pos, n_tot) in enumerate(regions):
        # Extract zoom window
        y0 = max(0, cy - zoom_size//2)
        y1 = min(masks.shape[0], cy + zoom_size//2)
        x0 = max(0, cx - zoom_size//2)
        x1 = min(masks.shape[1], cx + zoom_size//2)
        
        func_crop = func_norm[y0:y1, x0:x1]
        dsred_crop = dsred_norm[y0:y1, x0:x1]
        masks_crop = masks[y0:y1, x0:x1]
        
        # Get ROIs in crop
        labels_in_crop = np.unique(masks_crop)
        labels_in_crop = labels_in_crop[labels_in_crop > 0]
        pos_in_crop = [lab for lab in labels_in_crop if lab in positive]
        neg_in_crop = [lab for lab in labels_in_crop if lab in negative]
        
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        color = colors[idx % len(colors)]
        
        # Panel 1: GCaMP + contours (NO NUMBERS - cleaner!)
        func_rgb = np.stack([func_crop] * 3, axis=-1).copy()
        for lab in labels_in_crop:
            roi = masks_crop == lab
            b = find_boundaries(roi, mode='thick')
            if lab in positive:
                func_rgb[b] = [0, 1, 0]  # green for DsRed+
            else:
                func_rgb[b] = [1, 1, 0]  # yellow for DsRed-
        
        axes[0].imshow(func_rgb, interpolation='nearest')
        axes[0].set_title('GCaMP + ROI contours', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # NO ROI NUMBERS on GCaMP - keep it clean!
        
        # Panel 2: DsRed + contours
        dsred_rgb = np.stack([dsred_crop] * 3, axis=-1).copy()
        for lab in labels_in_crop:
            roi = masks_crop == lab
            b = find_boundaries(roi, mode='thick')
            if lab in positive:
                dsred_rgb[b] = [0, 1, 0]
            else:
                dsred_rgb[b] = [1, 1, 0]
        
        axes[1].imshow(dsred_rgb, interpolation='nearest')
        axes[1].set_title('DsRed + contours', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Classification overlay
        class_rgb = np.stack([dsred_crop] * 3, axis=-1).copy()
        for lab in pos_in_crop:
            roi = masks_crop == lab
            b = find_boundaries(roi, mode='thick')
            class_rgb[b, 0] = 0; class_rgb[b, 1] = 1; class_rgb[b, 2] = 0
        for lab in neg_in_crop:
            roi = masks_crop == lab
            b = find_boundaries(roi, mode='thick')
            class_rgb[b, 0] = 1; class_rgb[b, 1] = 0; class_rgb[b, 2] = 1
        
        axes[2].imshow(class_rgb, interpolation='nearest')
        axes[2].set_title(f'Classification', fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        # Overall title
        fig.suptitle(f'Zoom {idx+1}: {n_pos} DsRed+ / {n_tot} total ROIs', 
                    fontsize=18, fontweight='bold', color=color)
        
        # Panel labels
        for i, letter in enumerate(['A', 'B', 'C']):
            axes[i].text(0.02, 0.98, letter, transform=axes[i].transAxes,
                        fontsize=18, fontweight='bold', color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='black', alpha=0.7))
        
        # Legend
        leg_elements = [
            Patch(facecolor='green', label=f'DsRed+ ({len(pos_in_crop)})'),
            Patch(facecolor='magenta', label=f'DsRed− ({len(neg_in_crop)})'),
        ]
        fig.legend(handles=leg_elements, loc='lower center', ncol=2,
                  fontsize=14, frameon=True, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        zoom_path = output_dir / f'{strategy_name}_zoom{idx+1}_detail.png'
        plt.savefig(zoom_path, dpi=400, bbox_inches='tight', facecolor='white')  # Higher DPI!
        plt.close()
        print(f"  ✓ Saved zoom {idx+1}: {zoom_path.name}")


# =============================================================================
# NEW: DsRed INTEGRITY CHECK (before vs after registration)
# =============================================================================

def create_dsred_integrity_check(dsred_original, dsred_registered, func_template, 
                                 plane_idx, output_path):
    """
    Compare DsRed original vs registered to check for distortion.
    
    Layout: 2×2
      Top-left: Original DsRed
      Top-right: Registered DsRed
      Bottom-left: Overlay (original = magenta, registered = cyan)
      Bottom-right: Difference map
    """
    # Extract slice from volumes (handling 3D or 4D)
    if dsred_original.ndim == 4:
        dsred_orig_slice = dsred_original[0, plane_idx] if dsred_original.shape[0] < 10 else dsred_original[plane_idx, :, :, 0]
    else:
        dsred_orig_slice = dsred_original[plane_idx]
    
    dsred_reg_slice = dsred_registered[plane_idx]
    func_slice = func_template[plane_idx]
    
    # Check if shapes match - if not, skip this comparison
    if dsred_orig_slice.shape != dsred_reg_slice.shape:
        print(f"  ⚠️  Shape mismatch: original {dsred_orig_slice.shape} vs registered {dsred_reg_slice.shape}")
        print(f"  Skipping DsRed integrity check (registration changed dimensions)")
        
        # Create a simpler figure showing just registered DsRed
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fs = 16
        
        orig_norm = normalize_for_display(dsred_orig_slice)
        reg_norm = normalize_for_display(dsred_reg_slice)
        
        axes[0].imshow(orig_norm, cmap='gray', interpolation='bilinear')
        axes[0].set_title('DsRed original (different dimensions)', fontsize=fs, fontweight='bold')
        axes[0].axis('off')
        axes[0].text(0.5, 0.05, f'Shape: {dsred_orig_slice.shape}', 
                    transform=axes[0].transAxes, fontsize=12, ha='center',
                    color='yellow', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        axes[1].imshow(reg_norm, cmap='gray', interpolation='bilinear')
        axes[1].set_title('DsRed registered (functional space)', fontsize=fs, fontweight='bold')
        axes[1].axis('off')
        axes[1].text(0.5, 0.05, f'Shape: {dsred_reg_slice.shape}', 
                    transform=axes[1].transAxes, fontsize=12, ha='center',
                    color='yellow', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        fig.text(0.5, 0.02, 
                'Note: Registration transformed dimensions — direct pixel-wise comparison not possible.\n'
                'This indicates the registration included resampling to match functional coordinates.',
                ha='center', fontsize=11, style='italic', color='gray')
        
        for i, letter in enumerate(['A', 'B']):
            axes[i].text(0.02, 0.98, letter, transform=axes[i].transAxes,
                        fontsize=fs + 4, fontweight='bold', color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Saved (simplified): {Path(output_path).name}")
        return
    
    # Normalize
    orig_norm = normalize_for_display(dsred_orig_slice)
    reg_norm = normalize_for_display(dsred_reg_slice)
    func_norm = normalize_for_display(func_slice)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fs = 16
    
    # A: Original DsRed
    axes[0, 0].imshow(orig_norm, cmap='gray', interpolation='bilinear')
    axes[0, 0].set_title('DsRed original (anatomical space)', fontsize=fs, fontweight='bold')
    axes[0, 0].axis('off')
    
    # B: Registered DsRed
    axes[0, 1].imshow(reg_norm, cmap='gray', interpolation='bilinear')
    axes[0, 1].set_title('DsRed registered (functional space)', fontsize=fs, fontweight='bold')
    axes[0, 1].axis('off')
    
    # C: Overlay (magenta = original, cyan = registered)
    overlay = np.zeros((*orig_norm.shape, 3), dtype=np.float32)
    overlay[..., 0] = orig_norm  # R
    overlay[..., 1] = reg_norm   # G
    overlay[..., 2] = orig_norm  # B
    # Result: original = magenta (R+B), registered = cyan (G+B overlap with original)
    # Where they overlap well = white/grey
    
    axes[1, 0].imshow(np.clip(overlay, 0, 1), interpolation='bilinear')
    axes[1, 0].set_title('Overlay (magenta=orig, cyan=reg)', fontsize=fs, fontweight='bold')
    axes[1, 0].axis('off')
    
    # D: Difference map (show where signal changed)
    diff = np.abs(reg_norm - orig_norm)
    axes[1, 1].imshow(diff, cmap='hot', interpolation='bilinear', vmin=0, vmax=0.3)
    axes[1, 1].set_title('Absolute difference', fontsize=fs, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Panel labels
    for i, (r, c, letter) in enumerate([(0, 0, 'A'), (0, 1, 'B'), (1, 0, 'C'), (1, 1, 'D')]):
        axes[r, c].text(0.02, 0.98, letter, transform=axes[r, c].transAxes,
                        fontsize=fs + 4, fontweight='bold', color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
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
    
    # Inter-ROI distances (nearest neighbor)
    centroids = np.array([p.centroid for p in props])
    if len(centroids) > 1:
        dists = cdist(centroids, centroids)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        mean_nn_dist = float(np.mean(min_dists))
        min_nn_dist = float(np.min(min_dists))
    else:
        mean_nn_dist = np.nan
        min_nn_dist = np.nan

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
        'mean_nn_distance': mean_nn_dist,
        'min_nn_distance': min_nn_dist,
        'pct_area_lt50': float(np.mean([a < 50 for a in areas]) * 100),
        'pct_area_gt500': float(np.mean([a > 500 for a in areas]) * 100),
    }
    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("SLICE 90 — COMPLETE ANALYSIS v3 (WITH ZOOM PANELS)")
    print("=" * 80)

    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in ['figures', 'masks', 'projections', 'metrics', 'zoom_panels']:
        (out_dir / d).mkdir(exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    print("\n--- Loading data ---")
    stack = load_plane_frames(cfg.PLANE_IDX)
    projections = compute_projections(stack)

    for name, proj in projections.items():
        imwrite(str(out_dir / 'projections' / f'plane90_{name}.tif'), proj)

    dsred_vol = imread(cfg.DSRED_REGISTERED).astype(np.float32)
    dsred_slice = dsred_vol[cfg.PLANE_IDX]
    print(f"  DsRed registered slice 90: {dsred_slice.shape}")
    
    # Load original DsRed for integrity check
    dsred_orig_vol = imread(cfg.DSRED_ORIGINAL).astype(np.float32)
    print(f"  DsRed original volume: {dsred_orig_vol.shape}")

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
    
    # ── 2b. DsRed integrity check ─────────────────────────────────────
    print("\n--- DsRed integrity check ---")
    create_dsred_integrity_check(
        dsred_orig_vol, dsred_vol, func_template_vol,
        cfg.PLANE_IDX,
        out_dir / 'figures' / 'dsred_integrity_check_slice90.png'
    )

    # ── 3. Strategy sweep ─────────────────────────────────────────────
    print("\n--- Strategy sweep ---")
    all_results = []
    best_strategy = None
    best_n_rois = 0

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
        if not np.isnan(seg_metrics.get('mean_nn_distance', np.nan)):
            print(f"  Mean NN dist: {seg_metrics['mean_nn_distance']:.1f} px")

        # Save masks
        imwrite(str(out_dir / 'masks' / f'masks_{name}.tif'), masks.astype(np.uint16))

        # Generate 2x2 figure
        create_2x2_thesis_figure(
            img, dsred_slice, masks, positive, negative, proj_type,
            out_dir / 'figures' / f'{name}_2x2.png'
        )
        
        # Track best strategy for zoom panels
        if seg_metrics['n_cells'] > best_n_rois:
            best_n_rois = seg_metrics['n_cells']
            best_strategy = (name, img, masks, positive, negative)

    # ── 4. Generate zoom panels for BEST strategy ────────────────────
    if best_strategy is not None:
        print("\n--- Generating zoom panels (best strategy) ---")
        name, img, masks, positive, negative = best_strategy
        print(f"  Using strategy: {name}")
        
        create_professional_zoom_panels(
            img, dsred_slice, masks, positive, negative,
            out_dir / 'zoom_panels',
            strategy_name=name,
            zoom_size=cfg.ZOOM_SIZE
        )

    # ── 5. Save comparison table ──────────────────────────────────────
    print("\n--- Saving comparison ---")
    df = pd.DataFrame(all_results)
    df = df.sort_values('n_rois', ascending=False)
    df.to_csv(out_dir / 'metrics' / 'strategy_comparison.csv', index=False)

    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON (sorted by n_rois):")
    print(f"{'='*80}")
    cols_to_show = ['strategy', 'n_rois', 'n_dsred_pos', 'n_dsred_neg',
                    'mean_area_px', 'mean_circularity', 'mean_snr']
    if 'mean_nn_distance' in df.columns:
        cols_to_show.append('mean_nn_distance')
    print(df[cols_to_show].to_string(index=False))

    # Best strategy
    best = df.iloc[0]
    print(f"\n→ BEST: {best['strategy']} with {best['n_rois']} ROIs")
    print(f"   DsRed+: {best['n_dsred_pos']}, DsRed−: {best['n_dsred_neg']}")

    print(f"\n{'='*80}")
    print(f"✅ COMPLETE! Output: {out_dir}")
    print(f"{'='*80}")
    print(f"\n📁 Generated files:")
    print(f"   • Registration overlay (GCaMP functional vs anatomical)")
    print(f"   • DsRed integrity check (before vs after registration)")
    print(f"   • 2×2 thesis figures for each strategy")
    print(f"   • Zoom panels showing ROI separation (best strategy)")
    print(f"   • Strategy comparison metrics (CSV)")


if __name__ == "__main__":
    main()