"""
=============================================================================
PLANE-BY-PLANE TRACE ANALYSIS - GABA vs NON-GABA
=============================================================================

Comprehensive analysis of functional properties across imaging planes.

For each Z-plane (180 total):
1. Population statistics (GABA vs non-GABA)
2. Activity metrics (mean, std, sparsity, peaks)
3. Temporal dynamics
4. Cross-correlation within/between populations
5. Statistical comparisons

Designed for Master's thesis - publication-quality figures.

Author: Inês Marques
Date: February 2025
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, correlate
from scipy.ndimage import gaussian_filter1d
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# High-quality plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
sns.set_style('ticks')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY"
    CLASSIFICATION_FILE = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS\classification\gaba_classification.csv"
    ROI_COORDS_FILE = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY\rois_2d\roi_2d_coordinates.npz"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\2P_PLANE_ANALYSIS"
    
    # Imaging parameters
    N_PLANES = 180
    FRAMES_PER_PLANE = 250
    FRAME_RATE = 2  # Hz
    PLANE_SPACING = 1  # μm
    
    # Analysis parameters
    MIN_ROIS_PER_PLANE = 10  # Minimum ROIs to analyze a plane
    MIN_ACTIVE_FRAMES = 50
    PEAK_THRESHOLD_SD = 2.0
    
    # Subsampling for memory efficiency
    SUBSAMPLE_TRACES = 5  # Every 5th frame
    
    # Visualization
    N_EXAMPLE_TRACES = 5
    N_EXAMPLE_PLANES = 6  # Show detailed analysis for these planes
    
    # Colors
    COLOR_GABA = '#FF00FF'      # Magenta
    COLOR_NON_GABA = '#00FF00'  # Green
    COLOR_GABA_LIGHT = '#FFB3FF'
    COLOR_NON_GABA_LIGHT = '#B3FFB3'

cfg = Config()

# Create output directories
for subdir in ["figures", "results", "per_plane", "summary"]:
    Path(cfg.OUT_DIR, subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_classification():
    """Load GABA classification with plane information."""
    print("\n" + "="*70)
    print("LOADING CLASSIFICATION")
    print("="*70)
    
    df = pd.read_csv(cfg.CLASSIFICATION_FILE)
    
    # Check if plane column exists, if not load from coordinates
    if 'plane' not in df.columns:
        print("  Loading plane information from coordinates...")
        coords_data = np.load(cfg.ROI_COORDS_FILE, allow_pickle=True)
        df['plane'] = coords_data['plane_idx']
    
    # Filter to analyzed ROIs only
    df_filt = df[df['is_filtered']].copy()
    
    n_gaba = df_filt['is_gabaergic'].sum()
    n_non_gaba = (~df_filt['is_gabaergic']).sum()
    
    print(f"  ✓ Total analyzed ROIs: {len(df_filt):,}")
    print(f"    GABA+: {n_gaba:,} ({100*n_gaba/len(df_filt):.1f}%)")
    print(f"    Non-GABA: {n_non_gaba:,} ({100*n_non_gaba/len(df_filt):.1f}%)")
    
    # Distribution across planes
    print(f"\n  Planes with ROIs: {df_filt['plane'].nunique()}/{cfg.N_PLANES}")
    print(f"  ROIs per plane: {df_filt.groupby('plane').size().mean():.1f} ± "
          f"{df_filt.groupby('plane').size().std():.1f}")
    
    return df_filt


def find_trace_file():
    """Find best available trace file."""
    print("\n" + "="*70)
    print("LOCATING TRACE FILES")
    print("="*70)
    
    traces_dir = Path(cfg.ANALYSIS_DIR) / "traces_2d"
    
    # Priority order
    candidates = [
        "traces_2d_dff_centered_WITH_NAN.npz",
        "traces_2d_dff_p20_WITH_NAN.npz",
        "traces_2d_F_WITH_NAN.npz",
    ]
    
    for filename in candidates:
        path = traces_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"  ✓ Found: {filename} ({size_mb:.1f} MB)")
            return str(path)
    
    raise FileNotFoundError(f"No trace files found in {traces_dir}")


def load_traces_for_rois(trace_file, roi_indices, subsample=1):
    """Load traces for specific ROIs."""
    data = np.load(trace_file, allow_pickle=True)
    
    traces = data['traces'][roi_indices]
    
    if subsample > 1:
        traces = traces[:, ::subsample]
    
    return traces


# =============================================================================
# 2. TRACE FEATURE EXTRACTION
# =============================================================================

def extract_trace_features(trace, frame_rate=2.0, peak_threshold_sd=2.0):
    """
    Extract comprehensive features from a single trace.
    
    Returns dict with all metrics or None if insufficient data.
    """
    # Valid frames
    valid = ~np.isnan(trace)
    
    if valid.sum() < cfg.MIN_ACTIVE_FRAMES:
        return None
    
    trace_valid = trace[valid]
    
    features = {}
    
    # ===== BASIC STATISTICS =====
    features['mean'] = np.mean(trace_valid)
    features['std'] = np.std(trace_valid)
    features['median'] = np.median(trace_valid)
    features['max'] = np.max(trace_valid)
    features['min'] = np.min(trace_valid)
    features['range'] = features['max'] - features['min']
    
    # Coefficient of variation
    features['cv'] = features['std'] / (abs(features['mean']) + 1e-10)
    
    # Skewness and kurtosis (shape of distribution)
    features['skewness'] = stats.skew(trace_valid)
    features['kurtosis'] = stats.kurtosis(trace_valid)
    
    # ===== ACTIVITY METRICS =====
    features['n_valid_frames'] = int(valid.sum())
    features['fraction_valid'] = valid.mean()
    features['mean_abs'] = np.mean(np.abs(trace_valid))
    features['rms'] = np.sqrt(np.mean(trace_valid**2))
    
    # ===== SPARSITY =====
    # Fraction above mean + 1SD
    thr_1sd = features['mean'] + features['std']
    features['sparsity_1sd'] = np.sum(trace_valid > thr_1sd) / len(trace_valid)
    
    # Fraction above mean + 2SD
    thr_2sd = features['mean'] + 2 * features['std']
    features['sparsity_2sd'] = np.sum(trace_valid > thr_2sd) / len(trace_valid)
    
    # ===== PEAK DETECTION =====
    peak_threshold = features['mean'] + peak_threshold_sd * features['std']
    peaks, properties = find_peaks(
        trace_valid,
        height=peak_threshold,
        distance=5,  # Minimum 5 frames between peaks
        prominence=features['std']  # Require prominence
    )
    
    features['n_peaks'] = len(peaks)
    
    # Peak frequency (Hz)
    duration_sec = len(trace_valid) / frame_rate
    features['peak_frequency'] = len(peaks) / duration_sec
    
    if len(peaks) > 0:
        features['mean_peak_height'] = np.mean(properties['peak_heights'])
        features['max_peak_height'] = np.max(properties['peak_heights'])
        features['std_peak_height'] = np.std(properties['peak_heights'])
        
        # Inter-peak intervals
        if len(peaks) > 1:
            ipi = np.diff(peaks) / frame_rate  # in seconds
            features['mean_ipi'] = np.mean(ipi)
            features['std_ipi'] = np.std(ipi)
            features['cv_ipi'] = features['std_ipi'] / (features['mean_ipi'] + 1e-10)
        else:
            features['mean_ipi'] = np.nan
            features['std_ipi'] = np.nan
            features['cv_ipi'] = np.nan
    else:
        features['mean_peak_height'] = 0
        features['max_peak_height'] = 0
        features['std_peak_height'] = 0
        features['mean_ipi'] = np.nan
        features['std_ipi'] = np.nan
        features['cv_ipi'] = np.nan
    
    # ===== TEMPORAL DYNAMICS =====
    # Autocorrelation at lag 1
    if len(trace_valid) > 1:
        autocorr = np.corrcoef(trace_valid[:-1], trace_valid[1:])[0, 1]
        features['autocorr_lag1'] = autocorr if np.isfinite(autocorr) else 0
    else:
        features['autocorr_lag1'] = 0
    
    # Variance of first derivative (activity variability)
    if len(trace_valid) > 1:
        diff = np.diff(trace_valid)
        features['diff_variance'] = np.var(diff)
        features['mean_abs_diff'] = np.mean(np.abs(diff))
    else:
        features['diff_variance'] = 0
        features['mean_abs_diff'] = 0
    
    return features


# =============================================================================
# 3. PLANE-BY-PLANE ANALYSIS
# =============================================================================

def analyze_plane(plane_idx, df_class, trace_file):
    """
    Analyze a single imaging plane.
    
    Returns dict with all results for this plane.
    """
    # Get ROIs in this plane
    df_plane = df_class[df_class['plane'] == plane_idx].copy()
    
    if len(df_plane) < cfg.MIN_ROIS_PER_PLANE:
        return None
    
    # Separate GABA and non-GABA
    gaba_idx = df_plane[df_plane['is_gabaergic']].index.values
    non_gaba_idx = df_plane[~df_plane['is_gabaergic']].index.values
    
    n_gaba = len(gaba_idx)
    n_non_gaba = len(non_gaba_idx)
    
    if n_gaba < 5 or n_non_gaba < 5:
        return None
    
    # Load traces
    all_idx = np.concatenate([gaba_idx, non_gaba_idx])
    traces = load_traces_for_rois(trace_file, all_idx, subsample=cfg.SUBSAMPLE_TRACES)
    
    # Split back
    gaba_traces = traces[:n_gaba]
    non_gaba_traces = traces[n_gaba:]
    
    # Extract features
    gaba_features = []
    non_gaba_features = []
    
    for trace in gaba_traces:
        feat = extract_trace_features(trace, cfg.FRAME_RATE / cfg.SUBSAMPLE_TRACES, 
                                     cfg.PEAK_THRESHOLD_SD)
        if feat is not None:
            gaba_features.append(feat)
    
    for trace in non_gaba_traces:
        feat = extract_trace_features(trace, cfg.FRAME_RATE / cfg.SUBSAMPLE_TRACES,
                                     cfg.PEAK_THRESHOLD_SD)
        if feat is not None:
            non_gaba_features.append(feat)
    
    if len(gaba_features) < 5 or len(non_gaba_features) < 5:
        return None
    
    # Convert to DataFrames
    df_gaba = pd.DataFrame(gaba_features)
    df_non_gaba = pd.DataFrame(non_gaba_features)
    
    # Compute statistics and comparisons
    results = {
        'plane': plane_idx,
        'depth_um': plane_idx * cfg.PLANE_SPACING,
        'n_gaba': n_gaba,
        'n_non_gaba': n_non_gaba,
        'pct_gaba': 100 * n_gaba / (n_gaba + n_non_gaba),
        'gaba_features': df_gaba,
        'non_gaba_features': df_non_gaba,
        'comparisons': {},
        'example_traces': {
            'gaba': gaba_traces[:cfg.N_EXAMPLE_TRACES],
            'non_gaba': non_gaba_traces[:cfg.N_EXAMPLE_TRACES]
        }
    }
    
    # Statistical comparisons for each metric
    metrics = ['mean', 'std', 'max', 'range', 'cv', 'sparsity_1sd', 'sparsity_2sd',
               'n_peaks', 'peak_frequency', 'mean_peak_height', 'autocorr_lag1',
               'diff_variance']
    
    for metric in metrics:
        if metric not in df_gaba.columns or metric not in df_non_gaba.columns:
            continue
        
        gaba_vals = df_gaba[metric].dropna()
        non_gaba_vals = df_non_gaba[metric].dropna()
        
        if len(gaba_vals) < 3 or len(non_gaba_vals) < 3:
            continue
        
        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(gaba_vals, non_gaba_vals, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((gaba_vals.std()**2 + non_gaba_vals.std()**2) / 2)
        cohens_d = (gaba_vals.mean() - non_gaba_vals.mean()) / (pooled_std + 1e-10)
        
        results['comparisons'][metric] = {
            'gaba_mean': float(gaba_vals.mean()),
            'gaba_std': float(gaba_vals.std()),
            'gaba_median': float(gaba_vals.median()),
            'non_gaba_mean': float(non_gaba_vals.mean()),
            'non_gaba_std': float(non_gaba_vals.std()),
            'non_gaba_median': float(non_gaba_vals.median()),
            'p_value': float(p),
            'cohens_d': float(cohens_d),
            'significant': p < 0.05
        }
    
    return results


def analyze_all_planes(df_class, trace_file):
    """Analyze all imaging planes."""
    print("\n" + "="*70)
    print("PLANE-BY-PLANE ANALYSIS")
    print("="*70)
    
    results_by_plane = {}
    
    planes_with_data = df_class['plane'].unique()
    planes_with_data.sort()
    
    print(f"\n  Analyzing {len(planes_with_data)} planes...")
    
    for i, plane_idx in enumerate(planes_with_data):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(planes_with_data)}")
        
        try:
            result = analyze_plane(plane_idx, df_class, trace_file)
            if result is not None:
                results_by_plane[plane_idx] = result
        except Exception as e:
            print(f"    WARNING: Plane {plane_idx} failed: {e}")
            continue
    
    print(f"\n  ✓ Successfully analyzed {len(results_by_plane)} planes")
    
    return results_by_plane


# =============================================================================
# 4. SUMMARY STATISTICS
# =============================================================================

def create_summary_dataframe(results_by_plane):
    """Create summary DataFrame across all planes."""
    print("\n[CREATING SUMMARY DATAFRAME]")
    
    summary_rows = []
    
    for plane_idx, result in results_by_plane.items():
        row = {
            'plane': plane_idx,
            'depth_um': result['depth_um'],
            'n_gaba': result['n_gaba'],
            'n_non_gaba': result['n_non_gaba'],
            'pct_gaba': result['pct_gaba']
        }
        
        # Add mean values for key metrics
        for metric in ['mean', 'std', 'sparsity_2sd', 'peak_frequency']:
            if metric in result['comparisons']:
                comp = result['comparisons'][metric]
                row[f'{metric}_gaba'] = comp['gaba_mean']
                row[f'{metric}_non_gaba'] = comp['non_gaba_mean']
                row[f'{metric}_p'] = comp['p_value']
                row[f'{metric}_cohens_d'] = comp['cohens_d']
        
        summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values('plane')
    
    print(f"  ✓ Summary DataFrame: {len(df_summary)} planes")
    
    return df_summary


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

def plot_plane_summary(df_summary, save_dir):
    """Create overview figure across all planes."""
    print("\n[CREATING PLANE SUMMARY FIGURE]")
    
    fig = plt.figure(figsize=(18, 12))
    
    planes = df_summary['plane'].values
    
    # === Panel A: ROI counts ===
    ax1 = plt.subplot(3, 3, 1)
    
    ax1.bar(planes, df_summary['n_gaba'], width=0.8, alpha=0.7,
            color=cfg.COLOR_GABA, label='GABA')
    ax1.bar(planes, df_summary['n_non_gaba'], width=0.8, alpha=0.7,
            bottom=df_summary['n_gaba'], color=cfg.COLOR_NON_GABA, label='Non-GABA')
    
    ax1.set_xlabel('Z Plane')
    ax1.set_ylabel('Number of ROIs')
    ax1.set_title('A. ROI Distribution Across Planes', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # === Panel B: GABA percentage ===
    ax2 = plt.subplot(3, 3, 2)
    
    ax2.plot(planes, df_summary['pct_gaba'], 'o-', color=cfg.COLOR_GABA,
             markersize=3, linewidth=1, alpha=0.7)
    ax2.axhline(df_summary['pct_gaba'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean = {df_summary['pct_gaba'].mean():.1f}%")
    
    ax2.set_xlabel('Z Plane')
    ax2.set_ylabel('GABA %')
    ax2.set_title('B. GABA Proportion Across Depth', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # === Panel C: Mean activity ===
    ax3 = plt.subplot(3, 3, 3)
    
    if 'mean_gaba' in df_summary.columns:
        ax3.plot(planes, df_summary['mean_gaba'], 'o-', color=cfg.COLOR_GABA,
                label='GABA', markersize=4, linewidth=1.5, alpha=0.8)
        ax3.plot(planes, df_summary['mean_non_gaba'], 's-', color=cfg.COLOR_NON_GABA,
                label='Non-GABA', markersize=4, linewidth=1.5, alpha=0.8)
        
        ax3.set_xlabel('Z Plane')
        ax3.set_ylabel('Mean Activity (ΔF/F₀)')
        ax3.set_title('C. Mean Activity Across Depth', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # === Panel D: Activity std ===
    ax4 = plt.subplot(3, 3, 4)
    
    if 'std_gaba' in df_summary.columns:
        ax4.plot(planes, df_summary['std_gaba'], 'o-', color=cfg.COLOR_GABA,
                label='GABA', markersize=4, linewidth=1.5, alpha=0.8)
        ax4.plot(planes, df_summary['std_non_gaba'], 's-', color=cfg.COLOR_NON_GABA,
                label='Non-GABA', markersize=4, linewidth=1.5, alpha=0.8)
        
        ax4.set_xlabel('Z Plane')
        ax4.set_ylabel('Activity Std (ΔF/F₀)')
        ax4.set_title('D. Activity Variability', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    # === Panel E: Sparsity ===
    ax5 = plt.subplot(3, 3, 5)
    
    if 'sparsity_2sd_gaba' in df_summary.columns:
        ax5.plot(planes, df_summary['sparsity_2sd_gaba'], 'o-', color=cfg.COLOR_GABA,
                label='GABA', markersize=4, linewidth=1.5, alpha=0.8)
        ax5.plot(planes, df_summary['sparsity_2sd_non_gaba'], 's-', color=cfg.COLOR_NON_GABA,
                label='Non-GABA', markersize=4, linewidth=1.5, alpha=0.8)
        
        ax5.set_xlabel('Z Plane')
        ax5.set_ylabel('Sparsity (>2σ)')
        ax5.set_title('E. Activity Sparsity', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
    
    # === Panel F: Peak frequency ===
    ax6 = plt.subplot(3, 3, 6)
    
    if 'peak_frequency_gaba' in df_summary.columns:
        ax6.plot(planes, df_summary['peak_frequency_gaba'], 'o-', color=cfg.COLOR_GABA,
                label='GABA', markersize=4, linewidth=1.5, alpha=0.8)
        ax6.plot(planes, df_summary['peak_frequency_non_gaba'], 's-', 
                color=cfg.COLOR_NON_GABA, label='Non-GABA', markersize=4, 
                linewidth=1.5, alpha=0.8)
        
        ax6.set_xlabel('Z Plane')
        ax6.set_ylabel('Peak Frequency (Hz)')
        ax6.set_title('F. Transient Frequency', fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
    
    # === Panel G: P-values heatmap ===
    ax7 = plt.subplot(3, 3, 7)
    
    metrics_for_heatmap = ['mean_p', 'std_p', 'sparsity_2sd_p', 'peak_frequency_p']
    available_metrics = [m for m in metrics_for_heatmap if m in df_summary.columns]
    
    if len(available_metrics) > 0:
        p_matrix = df_summary[available_metrics].T.values
        
        im = ax7.imshow(p_matrix, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=0.05, interpolation='nearest')
        
        ax7.set_yticks(range(len(available_metrics)))
        ax7.set_yticklabels([m.replace('_p', '') for m in available_metrics])
        ax7.set_xlabel('Plane Index')
        ax7.set_title('G. Statistical Significance (p-values)', fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('p-value')
        
        # Add significance threshold line
        ax7.axhline(len(available_metrics) - 0.5, color='white', 
                   linestyle='--', linewidth=1)
    
    # === Panel H: Effect sizes ===
    ax8 = plt.subplot(3, 3, 8)
    
    metrics_for_effect = ['mean_cohens_d', 'std_cohens_d', 'sparsity_2sd_cohens_d', 
                         'peak_frequency_cohens_d']
    available_effect = [m for m in metrics_for_effect if m in df_summary.columns]
    
    if len(available_effect) > 0:
        for i, metric in enumerate(available_effect):
            label = metric.replace('_cohens_d', '')
            ax8.plot(planes, df_summary[metric], 'o-', label=label,
                    markersize=3, linewidth=1, alpha=0.7)
        
        ax8.axhline(0, color='black', linestyle='--', linewidth=1)
        ax8.axhline(0.2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax8.axhline(-0.2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax8.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax8.axhline(-0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        
        ax8.set_xlabel('Z Plane')
        ax8.set_ylabel("Cohen's d")
        ax8.set_title('H. Effect Sizes Across Depth', fontweight='bold')
        ax8.legend(fontsize=7, loc='best')
        ax8.grid(alpha=0.3)
        
        # Add effect size interpretation
        ax8.text(0.02, 0.98, 'Small: |d|=0.2\nMedium: |d|=0.5\nLarge: |d|=0.8',
                transform=ax8.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # === Panel I: Summary statistics ===
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Count significant differences
    n_sig_mean = (df_summary['mean_p'] < 0.05).sum() if 'mean_p' in df_summary.columns else 0
    n_sig_std = (df_summary['std_p'] < 0.05).sum() if 'std_p' in df_summary.columns else 0
    n_sig_sparsity = (df_summary['sparsity_2sd_p'] < 0.05).sum() if 'sparsity_2sd_p' in df_summary.columns else 0
    n_sig_freq = (df_summary['peak_frequency_p'] < 0.05).sum() if 'peak_frequency_p' in df_summary.columns else 0
    
    text = f"""
SUMMARY STATISTICS

Total Planes Analyzed: {len(df_summary)}

Mean GABA%: {df_summary['pct_gaba'].mean():.1f}%
  (Range: {df_summary['pct_gaba'].min():.1f}% - {df_summary['pct_gaba'].max():.1f}%)

Significant Differences:
  Mean Activity: {n_sig_mean}/{len(df_summary)} planes
  Activity Std: {n_sig_std}/{len(df_summary)} planes
  Sparsity: {n_sig_sparsity}/{len(df_summary)} planes
  Peak Frequency: {n_sig_freq}/{len(df_summary)} planes

{'GABA neurons show consistent' if n_sig_mean > len(df_summary)/2 else 'Mixed results for'}
functional differences across depth.
    """
    
    ax9.text(0.5, 0.5, text, ha='center', va='center',
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
             transform=ax9.transAxes)
    
    plt.suptitle('Plane-by-Plane Analysis: GABA vs Non-GABA Neurons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "figures" / "plane_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def plot_example_planes(results_by_plane, save_dir):
    """Plot detailed analysis for example planes."""
    print("\n[CREATING EXAMPLE PLANE FIGURES]")
    
    # Select evenly spaced planes
    all_planes = sorted(results_by_plane.keys())
    n_planes = len(all_planes)
    
    if n_planes < cfg.N_EXAMPLE_PLANES:
        example_planes = all_planes
    else:
        indices = np.linspace(0, n_planes-1, cfg.N_EXAMPLE_PLANES, dtype=int)
        example_planes = [all_planes[i] for i in indices]
    
    for plane_idx in example_planes:
        result = results_by_plane[plane_idx]
        
        fig = plt.figure(figsize=(16, 10))
        
        # === Panel A: Example traces (GABA) ===
        ax1 = plt.subplot(2, 3, 1)
        
        gaba_traces = result['example_traces']['gaba']
        n_frames = gaba_traces.shape[1]
        time = np.arange(n_frames) / (cfg.FRAME_RATE / cfg.SUBSAMPLE_TRACES)
        
        for i, trace in enumerate(gaba_traces):
            ax1.plot(time, trace + i*3, color=cfg.COLOR_GABA, 
                    linewidth=0.5, alpha=0.7)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('ΔF/F₀ (offset)')
        ax1.set_title(f'A. Example GABA Traces (n={len(gaba_traces)})', 
                     fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # === Panel B: Example traces (Non-GABA) ===
        ax2 = plt.subplot(2, 3, 2)
        
        non_gaba_traces = result['example_traces']['non_gaba']
        
        for i, trace in enumerate(non_gaba_traces):
            ax2.plot(time, trace + i*3, color=cfg.COLOR_NON_GABA,
                    linewidth=0.5, alpha=0.7)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('ΔF/F₀ (offset)')
        ax2.set_title(f'B. Example Non-GABA Traces (n={len(non_gaba_traces)})',
                     fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')
        
        # === Panel C: Activity distributions ===
        ax3 = plt.subplot(2, 3, 3)
        
        df_gaba = result['gaba_features']
        df_non_gaba = result['non_gaba_features']
        
        ax3.hist(df_non_gaba['mean'].dropna(), bins=30, alpha=0.6,
                color=cfg.COLOR_NON_GABA, label='Non-GABA', density=True)
        ax3.hist(df_gaba['mean'].dropna(), bins=30, alpha=0.6,
                color=cfg.COLOR_GABA, label='GABA', density=True)
        
        ax3.set_xlabel('Mean Activity (ΔF/F₀)')
        ax3.set_ylabel('Density')
        ax3.set_title('C. Activity Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # === Panel D: Violin plots ===
        ax4 = plt.subplot(2, 3, 4)
        
        metrics = ['mean', 'std', 'sparsity_2sd']
        positions_gaba = [1, 3, 5]
        positions_non_gaba = [2, 4, 6]
        
        for i, metric in enumerate(metrics):
            if metric in df_gaba.columns and metric in df_non_gaba.columns:
                parts_g = ax4.violinplot([df_gaba[metric].dropna()],
                                        positions=[positions_gaba[i]],
                                        showmeans=True, widths=0.7)
                parts_ng = ax4.violinplot([df_non_gaba[metric].dropna()],
                                         positions=[positions_non_gaba[i]],
                                         showmeans=True, widths=0.7)
                
                parts_g['bodies'][0].set_facecolor(cfg.COLOR_GABA)
                parts_g['bodies'][0].set_alpha(0.7)
                parts_ng['bodies'][0].set_facecolor(cfg.COLOR_NON_GABA)
                parts_ng['bodies'][0].set_alpha(0.7)
        
        ax4.set_xticks([1.5, 3.5, 5.5])
        ax4.set_xticklabels(['Mean', 'Std', 'Sparsity'])
        ax4.set_ylabel('Value')
        ax4.set_title('D. Metric Distributions', fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cfg.COLOR_GABA, alpha=0.7, label='GABA'),
            Patch(facecolor=cfg.COLOR_NON_GABA, alpha=0.7, label='Non-GABA')
        ]
        ax4.legend(handles=legend_elements)
        
        # === Panel E: Statistical comparisons ===
        ax5 = plt.subplot(2, 3, 5)
        
        comp_metrics = ['mean', 'std', 'sparsity_2sd', 'peak_frequency']
        y_pos = np.arange(len(comp_metrics))
        
        p_values = []
        effect_sizes = []
        labels = []
        
        for metric in comp_metrics:
            if metric in result['comparisons']:
                comp = result['comparisons'][metric]
                p_values.append(comp['p_value'])
                effect_sizes.append(comp['cohens_d'])
                labels.append(metric)
        
        if len(p_values) > 0:
            colors = [cfg.COLOR_GABA if p < 0.05 else 'gray' for p in p_values]
            
            ax5.barh(range(len(p_values)), [-np.log10(p) for p in p_values],
                    color=colors, alpha=0.7)
            ax5.axvline(-np.log10(0.05), color='red', linestyle='--',
                       linewidth=2, label='p=0.05')
            
            ax5.set_yticks(range(len(labels)))
            ax5.set_yticklabels(labels)
            ax5.set_xlabel('-log10(p-value)')
            ax5.set_title('E. Statistical Significance', fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3, axis='x')
        
        # === Panel F: Summary table ===
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
PLANE {plane_idx} SUMMARY
Depth: {result['depth_um']:.0f} μm

Population:
  GABA: {result['n_gaba']} neurons ({result['pct_gaba']:.1f}%)
  Non-GABA: {result['n_non_gaba']} neurons

Significant Differences:
"""
        
        for metric in comp_metrics:
            if metric in result['comparisons']:
                comp = result['comparisons'][metric]
                if comp['significant']:
                    sig_str = '***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*'
                    summary_text += f"\n  {metric}: p={comp['p_value']:.2e} {sig_str}"
                    summary_text += f"\n    (d={comp['cohens_d']:+.2f})"
        
        if len([c for c in result['comparisons'].values() if c['significant']]) == 0:
            summary_text += "\n  None detected"
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
                transform=ax6.transAxes)
        
        plt.suptitle(f'Detailed Analysis: Plane {plane_idx} '
                    f'(Depth: {result["depth_um"]:.0f} μm)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(save_dir) / "per_plane" / f"plane_{plane_idx:03d}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"  ✓ Saved {len(example_planes)} example plane figures")


# =============================================================================
# 6. EXPORT RESULTS
# =============================================================================

def export_results(results_by_plane, df_summary, save_dir):
    """Export all results to files."""
    print("\n[EXPORTING RESULTS]")
    
    # Save summary DataFrame
    summary_path = Path(save_dir) / "results" / "plane_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"  ✓ Saved summary: {summary_path}")
    
    # Save detailed results as JSON (without traces)
    detailed_results = {}
    for plane_idx, result in results_by_plane.items():
        detailed_results[int(plane_idx)] = {
            'plane': int(plane_idx),
            'depth_um': float(result['depth_um']),
            'n_gaba': int(result['n_gaba']),
            'n_non_gaba': int(result['n_non_gaba']),
            'pct_gaba': float(result['pct_gaba']),
            'comparisons': {
                k: {
                    'gaba_mean': float(v['gaba_mean']),
                    'gaba_std': float(v['gaba_std']),
                    'non_gaba_mean': float(v['non_gaba_mean']),
                    'non_gaba_std': float(v['non_gaba_std']),
                    'p_value': float(v['p_value']),
                    'cohens_d': float(v['cohens_d']),
                    'significant': bool(v['significant'])
                }
                for k, v in result['comparisons'].items()
            }
        }
    
    detailed_path = Path(save_dir) / "results" / "detailed_results.json"
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"  ✓ Saved detailed results: {detailed_path}")
    
    # Save per-plane feature CSVs
    for plane_idx, result in results_by_plane.items():
        df_gaba = result['gaba_features'].copy()
        df_gaba['population'] = 'GABA'
        
        df_non_gaba = result['non_gaba_features'].copy()
        df_non_gaba['population'] = 'Non-GABA'
        
        df_combined = pd.concat([df_gaba, df_non_gaba], ignore_index=True)
        df_combined['plane'] = plane_idx
        
        feature_path = Path(save_dir) / "per_plane" / f"features_plane_{plane_idx:03d}.csv"
        df_combined.to_csv(feature_path, index=False)
    
    print(f"  ✓ Saved {len(results_by_plane)} per-plane feature files")


# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def main():
    """Run complete plane-by-plane analysis."""
    
    print("="*80)
    print("PLANE-BY-PLANE TRACE ANALYSIS: GABA vs NON-GABA")
    print("="*80)
    print(f"\nOutput: {cfg.OUT_DIR}")
    print(f"Imaging: {cfg.N_PLANES} planes, {cfg.FRAMES_PER_PLANE} frames/plane")
    print(f"Subsampling: every {cfg.SUBSAMPLE_TRACES} frames")
    
    # Load data
    df_class = load_classification()
    trace_file = find_trace_file()
    
    # Analyze all planes
    results_by_plane = analyze_all_planes(df_class, trace_file)
    
    if len(results_by_plane) == 0:
        print("\n❌ ERROR: No planes successfully analyzed!")
        return
    
    # Create summary
    df_summary = create_summary_dataframe(results_by_plane)
    
    # Visualizations
    plot_plane_summary(df_summary, cfg.OUT_DIR)
    plot_example_planes(results_by_plane, cfg.OUT_DIR)
    
    # Export
    export_results(results_by_plane, df_summary, cfg.OUT_DIR)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    # Count significant differences
    sig_counts = {}
    for metric in ['mean', 'std', 'sparsity_2sd', 'peak_frequency']:
        col_name = f'{metric}_p'
        if col_name in df_summary.columns:
            sig_counts[metric] = (df_summary[col_name] < 0.05).sum()
    
    print(f"\n📊 SUMMARY:")
    print(f"  Planes analyzed: {len(results_by_plane)}/{cfg.N_PLANES}")
    print(f"  Mean GABA%: {df_summary['pct_gaba'].mean():.1f}%")
    print(f"\n  Planes with significant differences:")
    for metric, count in sig_counts.items():
        pct = 100 * count / len(df_summary)
        print(f"    {metric}: {count}/{len(df_summary)} ({pct:.1f}%)")
    
    print(f"\n📁 Output directory: {cfg.OUT_DIR}")
    print(f"  • figures/plane_summary.png - Overview across all planes")
    print(f"  • per_plane/*.png - Detailed analysis for example planes")
    print(f"  • results/plane_summary.csv - Summary statistics")
    print(f"  • results/detailed_results.json - Complete results")
    
    print("="*80)
    
    return results_by_plane, df_summary


if __name__ == "__main__":
    results_by_plane, df_summary = main()