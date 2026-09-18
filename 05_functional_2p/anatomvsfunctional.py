"""
FUNCTIONAL vs ANATOMICAL - NOISE COMPARISON
Generate high-quality figures showing why registration is needed
and the difference in noise levels between modalities
"""

import numpy as np
from pathlib import Path
from tifffile import imread, TiffFile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import re

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    # Paths (update these to your actual paths)
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    GCAMP_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\Gcamp6s_averaged\alignment_drift\anatomy.tif"
    
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\functional_vs_anatomical"
    
    # Which planes to show
    EXAMPLE_PLANES = [60, 90, 120]  # Representative middle planes
    
    # Plane pattern for functional
    PLANE_PATTERN = r"aligned_p(\d+)_nan\.tif"
    
    # Figure settings
    FIGURE_DPI = 300
    
    # Processing
    FRAME_STRIDE = 2  # Use every 2nd frame for functional averaging
    PERCENTILE_LOW = 0.5
    PERCENTILE_HIGH = 99.5

cfg = Config()

# ============================================================================
# FUNCTIONS
# ============================================================================

def list_plane_files(tifs_dir, pattern_regex):
    """List all functional plane files"""
    tifs_path = Path(tifs_dir)
    pattern = re.compile(pattern_regex, re.IGNORECASE)
    plane_files = {}
    
    for f in tifs_path.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            plane_idx = int(m.group(1)) - 1
            plane_files[plane_idx] = f
    
    print(f"  Found {len(plane_files)} functional plane files")
    return plane_files


def load_functional_plane(path, frame_stride=2):
    """Load and average a functional plane (temporal mean projection)"""
    with TiffFile(str(path)) as tif:
        n_pages = len(tif.pages)
        indices = list(range(0, n_pages, frame_stride))
        
        frames = []
        for idx in indices:
            try:
                frame = tif.pages[idx].asarray().astype(np.float32)
                if not np.any(np.isnan(frame)):
                    frames.append(frame)
            except:
                continue
        
        if len(frames) == 0:
            raise RuntimeError(f"No valid frames in {path}")
        
        stack = np.array(frames)
        return np.mean(stack, axis=0).astype(np.float32)


def normalize_display(img, percentile_low=0.5, percentile_high=99.5):
    """Normalize image for display"""
    v = img[img > 0]
    if v.size < 100:
        return img
    vmin, vmax = np.percentile(v, [percentile_low, percentile_high])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def compute_noise_metrics(img):
    """
    Compute simple noise metrics
    - SNR estimate
    - Coefficient of variation
    """
    # Estimate noise from background regions (low intensity areas)
    background = img[img < np.percentile(img, 20)]
    if len(background) > 100:
        noise_std = np.std(background)
    else:
        noise_std = np.std(img)
    
    signal_mean = np.mean(img[img > np.percentile(img, 50)])
    
    snr = signal_mean / (noise_std + 1e-8)
    cv = noise_std / (signal_mean + 1e-8)  # Coefficient of variation
    
    return {
        'snr': float(snr),
        'cv': float(cv),
        'noise_std': float(noise_std),
        'signal_mean': float(signal_mean)
    }


def create_comparison_figure_single_plane(functional, anatomical, 
                                          plane_idx, output_path):
    """
    Create side-by-side comparison for a single plane
    Shows functional (noisy) vs anatomical (clean)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Normalize for display
    func_norm = normalize_display(functional)
    anat_norm = normalize_display(anatomical)
    
    # Compute noise metrics
    func_metrics = compute_noise_metrics(functional)
    anat_metrics = compute_noise_metrics(anatomical)
    
    # Plot functional
    ax = axes[0]
    im = ax.imshow(func_norm, cmap='gray', interpolation='bilinear')
    ax.set_title(f'Functional Template (Plane {plane_idx})\nTemporal mean of calcium activity', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # Add metrics text
    metrics_text = f"SNR: {func_metrics['snr']:.1f}\nCV: {func_metrics['cv']:.3f}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot anatomical
    ax = axes[1]
    im = ax.imshow(anat_norm, cmap='gray', interpolation='bilinear')
    ax.set_title(f'Anatomical Image (Plane {plane_idx})\nDedicated high-quality acquisition', 
                fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # Add metrics text
    metrics_text = f"SNR: {anat_metrics['snr']:.1f}\nCV: {anat_metrics['cv']:.3f}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Image Quality Comparison: Functional vs Anatomical', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add explanation at bottom
    explanation = ("The functional template (left) is derived from calcium activity recordings and inherently contains temporal noise.\n"
                  "The anatomical image (right) is a dedicated high-quality acquisition optimized for structural detail.")
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=11, 
            style='italic', wrap=True)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved: {Path(output_path).name}")
    
    return func_metrics, anat_metrics


def create_multi_plane_comparison(functional_vol, anatomical_vol, 
                                  planes, output_path):
    """
    Create multi-plane comparison showing multiple representative slices
    """
    n_planes = len(planes)
    fig, axes = plt.subplots(2, n_planes, figsize=(5*n_planes, 10))
    
    for col, plane_idx in enumerate(planes):
        if plane_idx >= functional_vol.shape[0] or plane_idx >= anatomical_vol.shape[0]:
            continue
        
        func_slice = normalize_display(functional_vol[plane_idx])
        anat_slice = normalize_display(anatomical_vol[plane_idx])
        
        # Functional
        axes[0, col].imshow(func_slice, cmap='gray', interpolation='bilinear')
        axes[0, col].set_title(f'Functional\nz={plane_idx}', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        
        # Anatomical
        axes[1, col].imshow(anat_slice, cmap='gray', interpolation='bilinear')
        axes[1, col].set_title(f'Anatomical\nz={plane_idx}', fontsize=12, fontweight='bold')
        axes[1, col].axis('off')
    
    # Row labels
    axes[0, 0].text(-0.15, 0.5, 'FUNCTIONAL\n(Noisy)', transform=axes[0, 0].transAxes,
                   fontsize=14, fontweight='bold', ha='right', va='center', rotation=90)
    axes[1, 0].text(-0.15, 0.5, 'ANATOMICAL\n(Clean)', transform=axes[1, 0].transAxes,
                   fontsize=14, fontweight='bold', ha='right', va='center', rotation=90)
    
    fig.suptitle('Image Quality Across Multiple Planes', 
                fontsize=16, fontweight='bold', y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved: {Path(output_path).name}")


def create_noise_analysis_figure(functional_vol, anatomical_vol, output_path):
    """
    Create detailed noise analysis figure with histograms and profiles
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Pick a representative plane
    mid_plane = functional_vol.shape[0] // 2
    func_slice = functional_vol[mid_plane]
    anat_slice = anatomical_vol[mid_plane]
    
    # 1. Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(normalize_display(func_slice), cmap='gray')
    ax1.set_title('Functional (Noisy)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(normalize_display(anat_slice), cmap='gray')
    ax2.set_title('Anatomical (Clean)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 2. Intensity histograms
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(func_slice.ravel(), bins=100, alpha=0.5, label='Functional', color='blue', density=True)
    ax3.hist(anat_slice.ravel(), bins=100, alpha=0.5, label='Anatomical', color='red', density=True)
    ax3.set_xlabel('Intensity', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Intensity Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3. Horizontal profile (middle row)
    ax4 = fig.add_subplot(gs[1, 0])
    mid_row = func_slice.shape[0] // 2
    ax4.plot(func_slice[mid_row, :], label='Functional', alpha=0.7, linewidth=1)
    ax4.plot(anat_slice[mid_row, :], label='Anatomical', alpha=0.7, linewidth=1)
    ax4.set_xlabel('X position (pixels)', fontsize=11)
    ax4.set_ylabel('Intensity', fontsize=11)
    ax4.set_title('Horizontal Profile (middle row)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 4. Vertical profile (middle column)
    ax5 = fig.add_subplot(gs[1, 1])
    mid_col = func_slice.shape[1] // 2
    ax5.plot(func_slice[:, mid_col], label='Functional', alpha=0.7, linewidth=1)
    ax5.plot(anat_slice[:, mid_col], label='Anatomical', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Y position (pixels)', fontsize=11)
    ax5.set_ylabel('Intensity', fontsize=11)
    ax5.set_title('Vertical Profile (middle column)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 5. Noise metrics comparison
    ax6 = fig.add_subplot(gs[1, 2])
    func_metrics = compute_noise_metrics(func_slice)
    anat_metrics = compute_noise_metrics(anat_slice)
    
    metrics = ['SNR', 'CV']
    func_vals = [func_metrics['snr'], func_metrics['cv']]
    anat_vals = [anat_metrics['snr'], anat_metrics['cv']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, func_vals, width, label='Functional', alpha=0.7, color='blue')
    bars2 = ax6.bar(x + width/2, anat_vals, width, label='Anatomical', alpha=0.7, color='red')
    
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('Noise Metrics Comparison', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Detailed Noise Analysis: Functional vs Anatomical', 
                fontsize=16, fontweight='bold')
    
    fig.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved: {Path(output_path).name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FUNCTIONAL vs ANATOMICAL - NOISE COMPARISON")
    print("=" * 80)
    
    out_dir = Path(cfg.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load anatomical volume
    print("\n[1] Loading anatomical volume...")
    anat_vol = imread(cfg.GCAMP_ANAT).astype(np.float32)
    
    if anat_vol.ndim == 4:
        anat_vol = anat_vol[0] if anat_vol.shape[0] < 10 else anat_vol[..., 0]
    
    print(f"  Anatomical shape: {anat_vol.shape}")
    
    # 2. Load functional planes
    print("\n[2] Loading functional planes...")
    plane_files = list_plane_files(cfg.FUNC_TIFS_DIR, cfg.PLANE_PATTERN)
    
    # Build functional volume for the example planes
    first_plane = load_functional_plane(plane_files[min(plane_files.keys())], cfg.FRAME_STRIDE)
    h, w = first_plane.shape
    
    # Load only the planes we need + some context
    planes_to_load = list(range(max(0, min(cfg.EXAMPLE_PLANES) - 10), 
                                min(max(cfg.EXAMPLE_PLANES) + 10, anat_vol.shape[0])))
    
    func_vol = np.zeros((len(planes_to_load), h, w), dtype=np.float32)
    
    for i, plane_idx in enumerate(planes_to_load):
        if plane_idx in plane_files:
            func_vol[i] = load_functional_plane(plane_files[plane_idx], cfg.FRAME_STRIDE)
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{len(planes_to_load)}")
    
    print(f"  Functional shape: {func_vol.shape}")
    
    # Adjust plane indices for the subset we loaded
    offset = planes_to_load[0]
    adjusted_planes = [p - offset for p in cfg.EXAMPLE_PLANES]
    
    # 3. Generate figures
    print("\n[3] Generating comparison figures...")
    
    # Single plane detailed comparison
    for plane_idx in cfg.EXAMPLE_PLANES:
        if plane_idx < func_vol.shape[0] and plane_idx < anat_vol.shape[0]:
            adj_idx = plane_idx - offset
            if 0 <= adj_idx < func_vol.shape[0]:
                func_metrics, anat_metrics = create_comparison_figure_single_plane(
                    func_vol[adj_idx], 
                    anat_vol[plane_idx],
                    plane_idx,
                    out_dir / f'comparison_plane_{plane_idx:03d}.png'
                )
    
    # Multi-plane overview
    create_multi_plane_comparison(
        func_vol,
        anat_vol[offset:offset+len(planes_to_load)],
        adjusted_planes,
        out_dir / 'multi_plane_comparison.png'
    )
    
    # Detailed noise analysis
    create_noise_analysis_figure(
        func_vol,
        anat_vol[offset:offset+len(planes_to_load)],
        out_dir / 'noise_analysis.png'
    )
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    print(f"\n📁 Output directory: {out_dir}")
    print("\nGenerated files:")
    print("  • comparison_plane_XXX.png - Single plane comparisons")
    print("  • multi_plane_comparison.png - Overview of multiple planes")
    print("  • noise_analysis.png - Detailed noise metrics")
    print("\n💡 Use these figures to show why registration is needed:")
    print("   - Functional images have inherent temporal noise")
    print("   - Anatomical images are cleaner but misaligned")
    print("   - Registration brings anatomical detail to functional space")
    print("=" * 80)


if __name__ == "__main__":
    main()