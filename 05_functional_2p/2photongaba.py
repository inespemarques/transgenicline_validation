"""
=============================================================================
GABA ANALYSIS - TRACES ONLY (Memory-Efficient)
=============================================================================

Otimizado para:
- 112,252 ROIs × 45,000 timepoints (18.8 GB)
- Processamento em batches
- Análises SEM regressores comportamentais
- Foco: classificação GABA, bleed-through, propriedades de traces

Características analisadas:
1. Activity metrics (mean, std, max, sparsity)
2. Temporal dynamics (peak frequency, duration)
3. Correlation structure
4. Population differences GABA vs non-GABA

Author: Inês Marques
Date: February 2025
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imread
from skimage.filters import threshold_otsu
from scipy import stats
from scipy.signal import find_peaks
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
sns.set_style('ticks')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY"
    REGISTRATION_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN\volumes"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS"
    
    # Classification
    DSRED_PERCENTILE = 80
    
    # Imaging parameters
    N_PLANES = 180
    FRAMES_PER_PLANE = 250
    FRAME_RATE = 2  # Hz
    
    # Memory management
    BATCH_SIZE = 5000  # Process 5000 ROIs at a time
    SUBSAMPLE_TRACES = 5  # Use every 5th frame for speed (9000 frames total)
    
    # Analysis parameters
    MIN_ACTIVE_FRAMES = 50  # Minimum valid frames to include ROI
    PEAK_THRESHOLD_SD = 2.0  # Threshold for peak detection (in SD)
    
    # Visualization
    N_EXAMPLE_TRACES = 10

cfg = Config()

# Create output directories
for subdir in ["figures", "results", "classification", "bleedthrough", "qa"]:
    Path(cfg.OUT_DIR, subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD DATA (Memory-Efficient)
# =============================================================================

def load_registered_volumes():
    """Load registered DsRed and functional volumes."""
    print("\n" + "="*70)
    print("LOADING REGISTERED VOLUMES")
    print("="*70)
    
    reg_dir = Path(cfg.REGISTRATION_DIR)
    
    dsred = imread(str(reg_dir / "dsred_registered_SyN.tif")).astype(np.float32)
    functional = imread(str(reg_dir / "functional_template.tif")).astype(np.float32)
    
    print(f"  ✓ DsRed: {dsred.shape}")
    print(f"  ✓ Functional: {functional.shape}")
    
    return dsred, functional


def load_roi_data():
    """Load ROI coordinates and metadata."""
    print("\n" + "="*70)
    print("LOADING ROI DATA")
    print("="*70)
    
    base = Path(cfg.ANALYSIS_DIR)
    
    # Coordinates
    coords = np.load(base / "rois_2d" / "roi_2d_coordinates.npz", allow_pickle=True)
    coords_dict = {
        'ypix': coords['ypix'],
        'xpix': coords['xpix'],
        'plane_idx': coords['plane_idx'],
        'centroid_y': coords['centroid_y'],
        'centroid_x': coords['centroid_x'],
    }
    
    # Metadata
    metadata = pd.read_csv(base / "rois_2d" / "roi_2d_metadata.csv")
    
    # Filtered indices
    filtered_path = base / "rois_2d" / "filtered_indices.npy"
    filtered_idx = np.load(filtered_path) if filtered_path.exists() else np.arange(len(metadata))
    
    print(f"  ✓ Total ROIs: {len(metadata):,}")
    print(f"  ✓ Filtered ROIs: {len(filtered_idx):,}")
    
    return coords_dict, metadata, filtered_idx


def load_traces_metadata():
    """Load only trace file metadata (not the actual data)."""
    print("\n" + "="*70)
    print("TRACE FILES AVAILABLE")
    print("="*70)
    
    traces_dir = Path(cfg.ANALYSIS_DIR) / "traces_2d"
    
    available_files = {
        'dff_centered': traces_dir / "traces_2d_dff_centered_WITH_NAN.npz",
        'dff': traces_dir / "traces_2d_dff_p20_WITH_NAN.npz",
        'F': traces_dir / "traces_2d_F_WITH_NAN.npz",
    }
    
    existing_files = {}
    for key, path in available_files.items():
        if path.exists():
            # Get file size
            size_mb = path.stat().st_size / (1024**2)
            existing_files[key] = str(path)
            print(f"  ✓ {key}: {path.name} ({size_mb:.1f} MB)")
    
    if not existing_files:
        raise FileNotFoundError(f"No trace files found in {traces_dir}")
    
    # Use dff_centered if available, otherwise dff
    if 'dff_centered' in existing_files:
        trace_file = existing_files['dff_centered']
        print(f"\n  → Using: dff_centered")
    elif 'dff' in existing_files:
        trace_file = existing_files['dff']
        print(f"\n  → Using: dff")
    else:
        trace_file = existing_files['F']
        print(f"\n  → Using: F")
    
    return trace_file


def load_traces_batch(trace_file, roi_indices, subsample=1):
    """
    Load traces for specific ROIs in memory-efficient way.
    
    Args:
        trace_file: path to npz file
        roi_indices: which ROIs to load
        subsample: load every Nth frame (default 1 = all frames)
    
    Returns:
        traces: (n_rois, n_frames) array
    """
    # Open npz file
    data = np.load(trace_file, allow_pickle=True)
    
    # Load only requested ROIs
    if 'traces' in data:
        # Full array shape is (n_rois, n_frames)
        # We need to load specific rows
        
        # Since we can't slice directly from compressed npz,
        # we'll load in chunks
        traces = data['traces'][roi_indices]
        
        # Subsample timepoints if requested
        if subsample > 1:
            traces = traces[:, ::subsample]
        
        return traces
    else:
        raise KeyError("'traces' not found in npz file")


# =============================================================================
# 2. GABA CLASSIFICATION
# =============================================================================

def compute_bright_mask(dsred, percentile=80):
    """Compute bright pixel mask per plane."""
    print(f"\n[COMPUTING BRIGHT MASK (P{percentile})]")
    
    n_z, H, W = dsred.shape
    bright_mask = np.zeros_like(dsred, dtype=bool)
    
    for z in range(n_z):
        plane = dsred[z]
        nonzero = plane[plane > 0]
        
        if len(nonzero) < 100:
            continue
        
        thr = np.percentile(nonzero, percentile)
        bright_mask[z] = plane > thr
    
    print(f"  ✓ Bright fraction: {bright_mask.mean():.2%}")
    
    return bright_mask


def classify_rois(coords, metadata, filtered_idx, bright_mask):
    """Classify ROIs as GABA+ or GABA-."""
    print("\n" + "="*70)
    print("CLASSIFYING ROIs")
    print("="*70)
    
    n_z, H, W = bright_mask.shape
    n_rois = len(metadata)
    
    bright_fractions = np.full(n_rois, np.nan)
    n_bright_pixels = np.zeros(n_rois, dtype=int)
    n_total_pixels = np.zeros(n_rois, dtype=int)
    
    print(f"  Processing {len(filtered_idx):,} ROIs...")
    
    for i, roi_idx in enumerate(filtered_idx):
        if i % 10000 == 0:
            print(f"    {i:,}/{len(filtered_idx):,}")
        
        z = coords['plane_idx'][roi_idx]
        ypix = coords['ypix'][roi_idx]
        xpix = coords['xpix'][roi_idx]
        
        if len(ypix) == 0 or z < 0 or z >= n_z:
            continue
        
        # Validate bounds
        valid = (ypix >= 0) & (ypix < H) & (xpix >= 0) & (xpix < W)
        ypix_v = ypix[valid]
        xpix_v = xpix[valid]
        
        if len(ypix_v) == 0:
            continue
        
        # Get bright values
        bright_vals = bright_mask[z, ypix_v, xpix_v]
        
        n_total_pixels[roi_idx] = len(bright_vals)
        n_bright_pixels[roi_idx] = bright_vals.sum()
        bright_fractions[roi_idx] = bright_vals.mean()
    
    # Otsu threshold
    valid_fracs = bright_fractions[filtered_idx]
    valid_fracs = valid_fracs[np.isfinite(valid_fracs)]
    
    otsu_thr = float(threshold_otsu(valid_fracs))
    print(f"\n  ✓ Otsu threshold: {otsu_thr:.4f}")
    
    # Create classification DataFrame
    df = metadata.copy()
    df['bright_fraction'] = bright_fractions
    df['n_bright_pixels'] = n_bright_pixels
    df['n_total_pixels'] = n_total_pixels
    df['is_filtered'] = False
    df.loc[filtered_idx, 'is_filtered'] = True
    df['is_gabaergic'] = (df['bright_fraction'] >= otsu_thr) & df['is_filtered']
    
    n_gaba = df['is_gabaergic'].sum()
    n_filtered = df['is_filtered'].sum()
    
    print(f"  ✓ GABAergic: {n_gaba:,}/{n_filtered:,} ({100*n_gaba/n_filtered:.1f}%)")
    
    return df, otsu_thr


# =============================================================================
# 3. BLEED-THROUGH ANALYSIS
# =============================================================================

def analyze_bleedthrough(dsred, functional):
    """Analyze channel correlation to detect bleed-through."""
    print("\n" + "="*70)
    print("BLEED-THROUGH ANALYSIS")
    print("="*70)
    
    results = {}
    
    # Global correlation (downsampled for memory)
    print("\n[1. Global Channel Correlation]")
    
    stride = 2  # Sample every 2nd pixel
    dsred_sub = dsred[:, ::stride, ::stride].flatten()
    func_sub = functional[:, ::stride, ::stride].flatten()
    
    mask = (dsred_sub > 0) & (func_sub > 0)
    
    r_global, p_global = stats.pearsonr(dsred_sub[mask], func_sub[mask])
    
    results['global'] = {
        'r': float(r_global),
        'p': float(p_global),
        'n_pixels': int(mask.sum())
    }
    
    print(f"  Global r = {r_global:.4f} (p = {p_global:.2e})")
    
    # Per-plane correlation
    print("\n[2. Per-Plane Correlation]")
    
    plane_r = np.zeros(dsred.shape[0])
    plane_p = np.zeros(dsred.shape[0])
    
    for z in range(dsred.shape[0]):
        dsred_z = dsred[z, ::stride, ::stride].flatten()
        func_z = functional[z, ::stride, ::stride].flatten()
        
        mask = (dsred_z > 0) & (func_z > 0)
        if mask.sum() < 100:
            plane_r[z] = np.nan
            plane_p[z] = np.nan
            continue
        
        r, p = stats.pearsonr(dsred_z[mask], func_z[mask])
        plane_r[z] = r
        plane_p[z] = p
    
    results['per_plane'] = {
        'r_values': plane_r,
        'p_values': plane_p,
        'mean_r': float(np.nanmean(plane_r)),
        'std_r': float(np.nanstd(plane_r))
    }
    
    print(f"  Mean r = {np.nanmean(plane_r):.4f} ± {np.nanstd(plane_r):.4f}")
    
    return results


def plot_bleedthrough(dsred, functional, bleed_results, save_dir):
    """Create bleed-through analysis figure."""
    print("\n[CREATING BLEED-THROUGH FIGURE]")
    
    fig = plt.figure(figsize=(16, 10))
    
    # MIPs
    ax1 = plt.subplot(2, 3, 1)
    dsred_mip = np.max(dsred, axis=0)
    im1 = ax1.imshow(dsred_mip, cmap='Reds', 
                     vmin=np.percentile(dsred_mip, 5),
                     vmax=np.percentile(dsred_mip, 99.5))
    ax1.set_title('A. DsRed (MIP)', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(2, 3, 2)
    func_mip = np.max(functional, axis=0)
    im2 = ax2.imshow(func_mip, cmap='Greens',
                     vmin=np.percentile(func_mip, 5),
                     vmax=np.percentile(func_mip, 99.5))
    ax2.set_title('B. GCaMP (MIP)', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(2, 3, 3)
    dsred_norm = (dsred_mip - dsred_mip.min()) / (dsred_mip.max() - dsred_mip.min())
    func_norm = (func_mip - func_mip.min()) / (func_mip.max() - func_mip.min())
    overlay = np.stack([dsred_norm, func_norm, np.zeros_like(dsred_norm)], axis=-1)
    ax3.imshow(overlay)
    ax3.set_title('C. Overlay', fontweight='bold')
    ax3.axis('off')
    
    # Correlation scatter (downsampled)
    ax4 = plt.subplot(2, 3, 4)
    stride = 5
    dsred_flat = dsred[::2, ::stride, ::stride].flatten()
    func_flat = functional[::2, ::stride, ::stride].flatten()
    mask = (dsred_flat > 0) & (func_flat > 0)
    
    n_plot = min(10000, mask.sum())
    idx = np.random.choice(np.where(mask)[0], n_plot, replace=False)
    
    ax4.hexbin(dsred_flat[idx], func_flat[idx], 
               gridsize=50, cmap='viridis', mincnt=1, bins='log')
    
    r = bleed_results['global']['r']
    p = bleed_results['global']['p']
    
    ax4.set_xlabel('DsRed Intensity')
    ax4.set_ylabel('GCaMP Intensity')
    ax4.set_title(f'D. Global Correlation\nr = {r:.4f}, p = {p:.2e}', fontweight='bold')
    
    # Per-plane correlation
    ax5 = plt.subplot(2, 3, 5)
    plane_r = bleed_results['per_plane']['r_values']
    valid = ~np.isnan(plane_r)
    
    ax5.plot(np.where(valid)[0], plane_r[valid], 'o-', alpha=0.6, markersize=3)
    ax5.axhline(0, color='k', linestyle='--', linewidth=1)
    ax5.axhline(plane_r[valid].mean(), color='r', linestyle='--', linewidth=1.5,
                label=f'Mean = {plane_r[valid].mean():.4f}')
    ax5.set_xlabel('Z Plane')
    ax5.set_ylabel('Pearson r')
    ax5.set_title('E. Per-Plane Correlation', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Interpretation
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    mean_r = plane_r[valid].mean()
    
    if abs(mean_r) < 0.3:
        interpretation = "✅ WEAK CORRELATION"
        color = '#2ECC71'
        conclusion = (
            f"Mean correlation r = {mean_r:.4f} indicates minimal channel coupling.\n\n"
            f"This supports the validity of DsRed-based GABAergic classification "
            f"independent of GCaMP functional activity.\n\n"
            f"No significant optical bleed-through detected."
        )
    elif abs(mean_r) < 0.5:
        interpretation = "⚠️ MODERATE CORRELATION"
        color = '#F39C12'
        conclusion = (
            f"Mean correlation r = {mean_r:.4f} suggests some channel coupling.\n\n"
            f"Further investigation recommended to distinguish between:\n"
            f"• Optical bleed-through\n"
            f"• Biological covariation (GABA+ neurons more active?)"
        )
    else:
        interpretation = "❌ STRONG CORRELATION"
        color = '#E74C3C'
        conclusion = (
            f"High correlation r = {mean_r:.4f} detected!\n\n"
            f"This may indicate:\n"
            f"• Significant optical bleed-through\n"
            f"• Tight biological coupling\n\n"
            f"GABA classification validity requires additional validation."
        )
    
    text = f"{interpretation}\n\n{conclusion}"
    
    ax6.text(0.5, 0.5, text, ha='center', va='center',
             fontsize=9, bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
             transform=ax6.transAxes, wrap=True)
    
    plt.suptitle('Channel Independence Analysis (Bleed-Through Assessment)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "bleedthrough" / "channel_independence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")
    
    return interpretation, mean_r


# =============================================================================
# 4. TRACE ANALYSIS (Batch Processing)
# =============================================================================

def extract_trace_features(trace):
    """
    Extract features from a single trace (only valid frames).
    
    Returns dict with metrics.
    """
    # Get valid frames (not NaN)
    valid = ~np.isnan(trace)
    
    if valid.sum() < cfg.MIN_ACTIVE_FRAMES:
        return None
    
    trace_valid = trace[valid]
    
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(trace_valid)
    features['std'] = np.std(trace_valid)
    features['median'] = np.median(trace_valid)
    features['max'] = np.max(trace_valid)
    features['min'] = np.min(trace_valid)
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / (abs(features['mean']) + 1e-10)  # Coefficient of variation
    
    # Activity metrics
    features['n_valid_frames'] = valid.sum()
    features['mean_abs'] = np.mean(np.abs(trace_valid))
    
    # Sparsity (fraction of time above mean + 1SD)
    threshold_1sd = features['mean'] + features['std']
    features['sparsity_1sd'] = np.sum(trace_valid > threshold_1sd) / len(trace_valid)
    
    # Sparsity (fraction above mean + 2SD)
    threshold_2sd = features['mean'] + 2 * features['std']
    features['sparsity_2sd'] = np.sum(trace_valid > threshold_2sd) / len(trace_valid)
    
    # Peak detection
    threshold_peaks = features['mean'] + cfg.PEAK_THRESHOLD_SD * features['std']
    peaks, properties = find_peaks(trace_valid, height=threshold_peaks, distance=5)
    
    features['n_peaks'] = len(peaks)
    features['peak_frequency'] = len(peaks) / (len(trace_valid) / cfg.FRAME_RATE)  # peaks per second
    
    if len(peaks) > 0:
        features['mean_peak_height'] = np.mean(properties['peak_heights'])
        features['max_peak_height'] = np.max(properties['peak_heights'])
    else:
        features['mean_peak_height'] = 0
        features['max_peak_height'] = 0
    
    return features


def analyze_traces_batch(trace_file, df_class):
    """Analyze traces in batches to avoid memory issues."""
    print("\n" + "="*70)
    print("ANALYZING TRACES (Batch Processing)")
    print("="*70)
    
    # Get GABA and non-GABA indices
    gaba_idx = df_class[df_class['is_gabaergic']].index.values
    non_gaba_idx = df_class[~df_class['is_gabaergic'] & df_class['is_filtered']].index.values
    
    print(f"\n  GABA neurons: {len(gaba_idx):,}")
    print(f"  Non-GABA neurons: {len(non_gaba_idx):,}")
    print(f"  Batch size: {cfg.BATCH_SIZE:,} ROIs")
    print(f"  Subsampling: every {cfg.SUBSAMPLE_TRACES} frames")
    
    # Process in batches
    all_features = {}
    
    # Combine indices for batch processing
    all_roi_idx = np.concatenate([gaba_idx, non_gaba_idx])
    is_gaba_label = np.concatenate([
        np.ones(len(gaba_idx), dtype=bool),
        np.zeros(len(non_gaba_idx), dtype=bool)
    ])
    
    n_batches = int(np.ceil(len(all_roi_idx) / cfg.BATCH_SIZE))
    
    print(f"\n  Total batches: {n_batches}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * cfg.BATCH_SIZE
        end_idx = min((batch_idx + 1) * cfg.BATCH_SIZE, len(all_roi_idx))
        
        batch_roi_idx = all_roi_idx[start_idx:end_idx]
        batch_is_gaba = is_gaba_label[start_idx:end_idx]
        
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: ROIs {start_idx:,}-{end_idx:,}")
        
        # Load traces for this batch
        try:
            traces = load_traces_batch(trace_file, batch_roi_idx, subsample=cfg.SUBSAMPLE_TRACES)
            print(f"    Loaded traces: {traces.shape}")
        except Exception as e:
            print(f"    ERROR loading batch: {e}")
            continue
        
        # Extract features
        for i, roi_idx in enumerate(batch_roi_idx):
            features = extract_trace_features(traces[i])
            
            if features is not None:
                features['roi_idx'] = roi_idx
                features['is_gaba'] = batch_is_gaba[i]
                all_features[roi_idx] = features
        
        print(f"    Processed {len(batch_roi_idx):,} ROIs, {len([f for f in all_features.values() if f['roi_idx'] in batch_roi_idx]):,} valid")
        
        # Clear memory
        del traces
    
    # Convert to DataFrame
    df_features = pd.DataFrame.from_dict(all_features, orient='index')
    
    print(f"\n  ✓ Total features extracted: {len(df_features):,} ROIs")
    print(f"    GABA: {df_features['is_gaba'].sum():,}")
    print(f"    Non-GABA: {(~df_features['is_gaba']).sum():,}")
    
    return df_features


def compare_populations(df_features):
    """Statistical comparison of GABA vs non-GABA populations."""
    print("\n" + "="*70)
    print("POPULATION COMPARISON")
    print("="*70)
    
    gaba = df_features[df_features['is_gaba']]
    non_gaba = df_features[~df_features['is_gaba']]
    
    metrics = ['mean', 'std', 'max', 'range', 'cv', 'sparsity_1sd', 'sparsity_2sd',
               'n_peaks', 'peak_frequency', 'mean_peak_height']
    
    results = {}
    
    print("\nMetric                    | GABA (mean±std)      | Non-GABA (mean±std)  | p-value    | Effect")
    print("-" * 100)
    
    for metric in metrics:
        gaba_vals = gaba[metric].dropna()
        non_gaba_vals = non_gaba[metric].dropna()
        
        if len(gaba_vals) < 10 or len(non_gaba_vals) < 10:
            continue
        
        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(gaba_vals, non_gaba_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((gaba_vals.std()**2 + non_gaba_vals.std()**2) / 2)
        cohens_d = (gaba_vals.mean() - non_gaba_vals.mean()) / pooled_std
        
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        
        results[metric] = {
            'gaba_mean': gaba_vals.mean(),
            'gaba_std': gaba_vals.std(),
            'non_gaba_mean': non_gaba_vals.mean(),
            'non_gaba_std': non_gaba_vals.std(),
            'p_value': p,
            'cohens_d': cohens_d,
            'significant': p < 0.05
        }
        
        print(f"{metric:25} | {gaba_vals.mean():8.4f}±{gaba_vals.std():7.4f} | "
              f"{non_gaba_vals.mean():8.4f}±{non_gaba_vals.std():7.4f} | "
              f"{p:10.2e} | {cohens_d:+6.3f} {sig}")
    
    return results


def plot_population_comparison(df_features, save_dir):
    """Create comprehensive population comparison figure."""
    print("\n[CREATING POPULATION COMPARISON FIGURE]")
    
    fig = plt.figure(figsize=(18, 12))
    
    gaba = df_features[df_features['is_gaba']]
    non_gaba = df_features[~df_features['is_gaba']]
    
    metrics = [
        ('mean', 'Mean Activity (ΔF/F₀)'),
        ('std', 'Activity Std (ΔF/F₀)'),
        ('max', 'Max Activity (ΔF/F₀)'),
        ('cv', 'Coefficient of Variation'),
        ('sparsity_2sd', 'Sparsity (>2σ)'),
        ('peak_frequency', 'Peak Frequency (Hz)'),
    ]
    
    for i, (metric, label) in enumerate(metrics):
        ax = plt.subplot(2, 3, i + 1)
        
        gaba_vals = gaba[metric].dropna()
        non_gaba_vals = non_gaba[metric].dropna()
        
        # Violin plot
        parts = ax.violinplot([gaba_vals, non_gaba_vals],
                              positions=[1, 2],
                              showmeans=True, showmedians=True)
        
        parts['bodies'][0].set_facecolor('red')
        parts['bodies'][1].set_facecolor('cyan')
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_alpha(0.7)
        
        # Statistical test
        stat, p = stats.mannwhitneyu(gaba_vals, non_gaba_vals)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['GABA', 'Non-GABA'])
        ax.set_ylabel(label)
        ax.set_title(f'{chr(65+i)}. {label}\np = {p:.2e} ({sig})', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Functional Properties: GABA vs Non-GABA Neurons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "figures" / "population_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def main():
    """Run complete analysis pipeline."""
    
    print("="*80)
    print("GABA ANALYSIS - TRACES ONLY (Memory-Efficient)")
    print("="*80)
    print(f"\nOutput: {cfg.OUT_DIR}")
    
    # 1. Load volumes
    dsred, functional = load_registered_volumes()
    
    # 2. Load ROI data
    coords, metadata, filtered_idx = load_roi_data()
    
    # 3. Classify GABA
    bright_mask = compute_bright_mask(dsred, cfg.DSRED_PERCENTILE)
    df_class, otsu_thr = classify_rois(coords, metadata, filtered_idx, bright_mask)
    
    # Save classification
    df_class.to_csv(Path(cfg.OUT_DIR) / "classification" / "gaba_classification.csv", index=False)
    print(f"\n✓ Saved classification")
    
    # 4. Bleed-through analysis
    bleed_results = analyze_bleedthrough(dsred, functional)
    interpretation, mean_r = plot_bleedthrough(dsred, functional, bleed_results, cfg.OUT_DIR)
    
    # Save bleed-through results
    with open(Path(cfg.OUT_DIR) / "bleedthrough" / "results.json", 'w') as f:
        save_results = {
            'interpretation': interpretation,
            'global': bleed_results['global'],
            'per_plane': {
                'mean_r': bleed_results['per_plane']['mean_r'],
                'std_r': bleed_results['per_plane']['std_r'],
            }
        }
        json.dump(save_results, f, indent=2)
    
    # 5. Trace analysis (batch processing)
    trace_file = load_traces_metadata()
    df_features = analyze_traces_batch(trace_file, df_class)
    
    # Save features
    df_features.to_csv(Path(cfg.OUT_DIR) / "results" / "trace_features.csv", index=False)
    print(f"\n✓ Saved trace features")
    
    # 6. Population comparison
    comparison_results = compare_populations(df_features)
    plot_population_comparison(df_features, cfg.OUT_DIR)
    
    # 7. Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'classification': {
            'total_rois': len(df_class),
            'filtered_rois': df_class['is_filtered'].sum(),
            'gaba_rois': df_class['is_gabaergic'].sum(),
            'pct_gaba': float(100 * df_class['is_gabaergic'].sum() / df_class['is_filtered'].sum()),
            'otsu_threshold': float(otsu_thr),
        },
        'bleedthrough': {
            'interpretation': interpretation,
            'mean_correlation': float(mean_r),
        },
        'trace_analysis': {
            'n_analyzed': len(df_features),
            'n_gaba': int(df_features['is_gaba'].sum()),
            'n_non_gaba': int((~df_features['is_gaba']).sum()),
        },
        'significant_differences': [
            metric for metric, res in comparison_results.items() if res['significant']
        ]
    }
    
    with open(Path(cfg.OUT_DIR) / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 SUMMARY:")
    print(f"  Classification:")
    print(f"    Total ROIs: {summary['classification']['total_rois']:,}")
    print(f"    GABA+: {summary['classification']['gaba_rois']:,} ({summary['classification']['pct_gaba']:.1f}%)")
    print(f"\n  Bleed-through: {interpretation} (r = {mean_r:.4f})")
    print(f"\n  Trace Analysis:")
    print(f"    Analyzed: {summary['trace_analysis']['n_analyzed']:,} ROIs")
    print(f"    Significant differences in {len(summary['significant_differences'])} metrics:")
    for metric in summary['significant_differences'][:5]:  # Show first 5
        print(f"      • {metric}")
    
    print(f"\n📁 Output: {cfg.OUT_DIR}")
    print("="*80)
    
    return df_class, bleed_results, df_features, summary


if __name__ == "__main__":
    df_class, bleed_results, df_features, summary = main()