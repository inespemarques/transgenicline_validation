"""
GENERATE HIGH-RES OVERLAYS FROM ANTS SyN RESULTS
Saves individual before/after overlay PNGs for each plane (NO LEGEND)
"""

import numpy as np
from pathlib import Path
from tifffile import imread
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

class Config:
    # Input volumes (from ANTs SyN output)
    VOLUMES_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes"
    FUNC_TEMPLATE = "functional_template.tif"
    GCAMP_BEFORE = "gcamp_before.tif"
    GCAMP_AFTER = "gcamp_registered_SyN.tif"
    DSRED_BEFORE = "dsred_before.tif"
    DSRED_AFTER = "dsred_registered_SyN.tif"
    
    # Output
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\individual_overlays"
    
    # Planes to generate
    PLANES = [30, 60, 90, 120, 150]
    
    # Figure settings
    DPI = 300  # High resolution
    FIGSIZE = (12, 12)  # Large figure
    
    # Display enhancement
    PERCENTILE_LOW = 0.5
    PERCENTILE_HIGH = 99.7


cfg = Config()


# =============================================================================
# FUNCTIONS
# =============================================================================

def normalize_for_display(img, plow=0.5, phigh=99.7):
    """Normalize image for display using percentile clipping."""
    v = img[img > 0]
    if v.size == 0:
        return np.zeros_like(img)
    vmin, vmax = np.percentile(v, [plow, phigh])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def create_overlay_single(fixed_slice, moving_slice, output_path, title):
    """
    Create single overlay: green (fixed) + magenta (moving).
    NO LEGEND - clean for thesis.
    
    Args:
        fixed_slice: Functional reference (green channel)
        moving_slice: Anatomy (magenta channel)
        output_path: Where to save PNG
        title: Figure title
    """
    # Normalize
    f = normalize_for_display(fixed_slice, cfg.PERCENTILE_LOW, cfg.PERCENTILE_HIGH)
    m = normalize_for_display(moving_slice, cfg.PERCENTILE_LOW, cfg.PERCENTILE_HIGH)
    
    # Create RGB overlay
    rgb = np.zeros((*f.shape, 3), dtype=np.float32)
    rgb[..., 1] = f  # Green = functional
    rgb[..., 0] = m  # Red (magenta)
    rgb[..., 2] = m  # Blue (magenta)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=cfg.FIGSIZE)
    ax.imshow(rgb, interpolation='bilinear')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='white')
    ax.axis('off')
    
    # Save with black background (better for overlays)
    fig.patch.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.DPI, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close(fig)
    
    print(f"  ✓ Saved: {Path(output_path).name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("GENERATING HIGH-RES INDIVIDUAL OVERLAYS (NO LEGEND)")
    print("=" * 80)
    
    # Setup output directory
    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in ['gcamp_before', 'gcamp_after', 'dsred_before', 'dsred_after']:
        (out_dir / subdir).mkdir(exist_ok=True)
    
    # Load volumes
    print("\nLoading volumes...")
    vol_dir = Path(cfg.VOLUMES_DIR)
    
    func = imread(str(vol_dir / cfg.FUNC_TEMPLATE)).astype(np.float32)
    gcamp_before = imread(str(vol_dir / cfg.GCAMP_BEFORE)).astype(np.float32)
    gcamp_after = imread(str(vol_dir / cfg.GCAMP_AFTER)).astype(np.float32)
    dsred_before = imread(str(vol_dir / cfg.DSRED_BEFORE)).astype(np.float32)
    dsred_after = imread(str(vol_dir / cfg.DSRED_AFTER)).astype(np.float32)
    
    print(f"  Functional: {func.shape}")
    print(f"  GCaMP before: {gcamp_before.shape}")
    print(f"  GCaMP after: {gcamp_after.shape}")
    print(f"  DsRed before: {dsred_before.shape}")
    print(f"  DsRed after: {dsred_after.shape}")
    
    # Generate overlays for each plane
    print(f"\nGenerating overlays for {len(cfg.PLANES)} planes...")
    print("=" * 80)
    
    for z in cfg.PLANES:
        if z >= func.shape[0]:
            print(f"  ⚠️  Skipping plane {z} (out of bounds)")
            continue
        
        print(f"\nPlane z={z}:")
        
        # GCaMP BEFORE
        create_overlay_single(
            func[z], gcamp_before[z],
            out_dir / 'gcamp_before' / f'z{z:03d}_gcamp_before_registration.png',
            f'GCaMP BEFORE Registration | Plane {z}'
        )
        
        # GCaMP AFTER
        create_overlay_single(
            func[z], gcamp_after[z],
            out_dir / 'gcamp_after' / f'z{z:03d}_gcamp_after_registration.png',
            f'GCaMP AFTER SyN Registration | Plane {z}'
        )
        
        # DsRed BEFORE
        create_overlay_single(
            func[z], dsred_before[z],
            out_dir / 'dsred_before' / f'z{z:03d}_dsred_before_registration.png',
            f'DsRed BEFORE Registration | Plane {z}'
        )
        
        # DsRed AFTER
        create_overlay_single(
            func[z], dsred_after[z],
            out_dir / 'dsred_after' / f'z{z:03d}_dsred_after_registration.png',
            f'DsRed AFTER SyN Registration | Plane {z}'
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {out_dir}")
    print(f"\n📂 Structure:")
    print(f"   gcamp_before/  → {len(cfg.PLANES)} overlays BEFORE registration")
    print(f"   gcamp_after/   → {len(cfg.PLANES)} overlays AFTER registration")
    print(f"   dsred_before/  → {len(cfg.PLANES)} overlays BEFORE registration")
    print(f"   dsred_after/   → {len(cfg.PLANES)} overlays AFTER registration")
    print(f"\n📊 Total images: {len(cfg.PLANES) * 4}")
    print(f"🎨 Resolution: {cfg.DPI} DPI")
    print(f"💾 Format: PNG (black background, white title, NO LEGEND)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()