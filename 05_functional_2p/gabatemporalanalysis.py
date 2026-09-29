"""
=============================================================================
GABA TEMPORAL ANALYSIS - FULL HINDBRAIN VERSION (Planes 50-100)
=============================================================================

Optimized for thesis conclusions:
- Analyzes complete hindbrain (planes 50-100)
- Regional analysis (anterior/mid/posterior)
- Comprehensive summary statistics
- Publication-quality figures
- Compact summary for 2-page conclusion

Author: Based on Inês Marques thesis requirements
Date: February 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, stats
import warnings
import traceback
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY"
    GABA_ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\GABA_TEMPORAL_FULL_HINDBRAIN"
    
    # FULL HINDBRAIN ANALYSIS
    PLANES_START = 10
    PLANES_END = 170
    
    # Regional divisions (for hindbrain anatomy)
    # Anterior: planes 50-66
    # Mid: planes 67-83
    # Posterior: planes 84-100
    REGION_BOUNDARIES = {
        'anterior': (50, 66),
        'mid': (67, 83),
        'posterior': (84, 100)
    }
    
    # Imaging parameters
    FRAME_RATE = 2  # Hz
    FRAMES_PER_PLANE = 250
    
    # Analysis parameters
    N_EXAMPLE_TRACES = 5
    XCORR_MAX_LAG_SEC = 15
    
    # Visualization
    COLOR_NON_GABA = '#D81B60'  # Magenta
    COLOR_GABA = '#00897B'      # Teal
    
    # For summary plots
    N_EXAMPLE_PLANES = 6  # Show 6 representative planes in summary

cfg = Config()

# Create output directories
Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
for subdir in ["individual_planes", "regional_analysis", "summary"]:
    Path(cfg.OUT_DIR, subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_gaba_classification():
    """Load GABA classification results."""
    print("\n" + "="*70)
    print("LOADING GABA CLASSIFICATION")
    print("="*70)
    
    class_file = Path(cfg.GABA_ANALYSIS_DIR) / "classification" / "gaba_classification.csv"
    
    if not class_file.exists():
        raise FileNotFoundError(f"Classification file not found: {class_file}")
    
    df = pd.read_csv(class_file)
    
    print(f"  ✓ Total ROIs: {len(df):,}")
    print(f"  ✓ GABA+: {df['is_gabaergic'].sum():,}")
    print(f"  ✓ Non-GABA: {(~df['is_gabaergic'] & df['is_filtered']).sum():,}")
    
    return df


def load_full_traces():
    """Load complete trace data."""
    print("\n" + "="*70)
    print("LOADING TRACE DATA")
    print("="*70)
    
    traces_dir = Path(cfg.ANALYSIS_DIR) / "traces_2d"
    
    trace_files = [
        traces_dir / "traces_2d_dff_p20_WITH_NAN.npz"
    ]
    
    trace_file = None
    for tf in trace_files:
        if tf.exists():
            trace_file = tf
            print(f"  ✓ Using: {tf.name}")
            break
    
    if trace_file is None:
        raise FileNotFoundError("No trace files found!")
    
    data = np.load(trace_file, allow_pickle=True)
    traces = data['traces']
    
    print(f"  ✓ Shape: {traces.shape}")
    print(f"  ✓ Size: {traces.nbytes / (1024**3):.2f} GB")
    
    return traces


def get_plane_frame_ranges(n_planes=180, frames_per_plane=250):
    """Calculate frame ranges for each plane."""
    frame_ranges = {}
    
    for plane in range(n_planes):
        start = plane * frames_per_plane
        end = start + frames_per_plane
        frame_ranges[plane] = (start, end)
    
    return frame_ranges


# =============================================================================
# CROSS-CORRELATION ANALYSIS
# =============================================================================

def compute_cross_correlation(trace1, trace2, max_lag_frames):
    """Compute normalized cross-correlation between two traces."""
    valid = ~(np.isnan(trace1) | np.isnan(trace2))
    
    if valid.sum() < 50:
        return None, None, None
    
    t1 = trace1[valid]
    t2 = trace2[valid]
    
    # Normalize
    t1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-10)
    t2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-10)
    
    # Cross-correlation
    xcorr = signal.correlate(t1, t2, mode='full')
    xcorr = xcorr / len(t1)
    
    lags = signal.correlation_lags(len(t1), len(t2), mode='full')
    
    # Restrict to max_lag
    mask = np.abs(lags) <= max_lag_frames
    lags = lags[mask]
    xcorr = xcorr[mask]
    
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    
    return lags, xcorr, peak_lag


def analyze_plane(plane_idx, df_class, traces, frame_ranges):
    """Analyze GABA vs Non-GABA dynamics for a single plane."""
    
    try:
        start_frame, end_frame = frame_ranges[plane_idx]
        
        # Get ROIs for this plane
        plane_rois = df_class[df_class['plane_idx'] == plane_idx].copy()
        
        gaba_rois = plane_rois[plane_rois['is_gabaergic']]
        non_gaba_rois = plane_rois[~plane_rois['is_gabaergic'] & plane_rois['is_filtered']]
        
        n_gaba = len(gaba_rois)
        n_non = len(non_gaba_rois)
        
        if n_gaba < 5 or n_non < 5:
            return None
        
        # Extract traces
        gaba_traces = traces[gaba_rois.index.values, start_frame:end_frame]
        non_gaba_traces = traces[non_gaba_rois.index.values, start_frame:end_frame]
        
        # Population means
        gaba_mean = np.nanmean(gaba_traces, axis=0)
        non_gaba_mean = np.nanmean(non_gaba_traces, axis=0)
        
        # Z-score
        gaba_mean_z = (gaba_mean - np.nanmean(gaba_mean)) / (np.nanstd(gaba_mean) + 1e-10)
        non_gaba_mean_z = (non_gaba_mean - np.nanmean(non_gaba_mean)) / (np.nanstd(non_gaba_mean) + 1e-10)
        
        # Cross-correlation
        max_lag_frames = int(cfg.XCORR_MAX_LAG_SEC * cfg.FRAME_RATE)
        lags, xcorr, peak_lag = compute_cross_correlation(
            gaba_mean_z, non_gaba_mean_z, max_lag_frames
        )
        
        if lags is None:
            return None
        
        peak_lag_sec = peak_lag / cfg.FRAME_RATE
        
        # Example traces
        def select_examples(traces_array, n_examples):
            valid_frac = 1 - np.isnan(traces_array).mean(axis=1)
            trace_std = np.nanstd(traces_array, axis=1)
            scores = trace_std * valid_frac
            top_idx = np.argsort(scores)[-n_examples:][::-1]
            return traces_array[top_idx]
        
        gaba_examples = select_examples(gaba_traces, cfg.N_EXAMPLE_TRACES)
        non_gaba_examples = select_examples(non_gaba_traces, cfg.N_EXAMPLE_TRACES)
        
        # Z-score examples
        gaba_examples_z = np.zeros_like(gaba_examples)
        for i in range(len(gaba_examples)):
            valid = ~np.isnan(gaba_examples[i])
            if valid.sum() > 10:
                gaba_examples_z[i, valid] = (gaba_examples[i, valid] - np.nanmean(gaba_examples[i])) / (np.nanstd(gaba_examples[i]) + 1e-10)
        
        non_gaba_examples_z = np.zeros_like(non_gaba_examples)
        for i in range(len(non_gaba_examples)):
            valid = ~np.isnan(non_gaba_examples[i])
            if valid.sum() > 10:
                non_gaba_examples_z[i, valid] = (non_gaba_examples[i, valid] - np.nanmean(non_gaba_examples[i])) / (np.nanstd(non_gaba_examples[i]) + 1e-10)
        
        results = {
            'plane_idx': plane_idx,
            'n_gaba': n_gaba,
            'n_non': n_non,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'gaba_mean_z': gaba_mean_z,
            'non_gaba_mean_z': non_gaba_mean_z,
            'lags': lags,
            'xcorr': xcorr,
            'peak_lag': peak_lag,
            'peak_lag_sec': peak_lag_sec,
            'max_xcorr': np.max(xcorr),
            'gaba_examples_z': gaba_examples_z,
            'non_gaba_examples_z': non_gaba_examples_z,
        }
        
        return results
        
    except Exception as e:
        print(f"  ⚠️ Error in plane {plane_idx}: {e}")
        return None


# =============================================================================
# REGIONAL ANALYSIS
# =============================================================================

def analyze_by_region(all_results):
    """Aggregate results by hindbrain region."""
    print("\n" + "="*70)
    print("REGIONAL ANALYSIS")
    print("="*70)
    
    regional_stats = {}
    
    for region_name, (start, end) in cfg.REGION_BOUNDARIES.items():
        # Filter results for this region
        region_results = [r for r in all_results if r is not None and start <= r['plane_idx'] <= end]
        
        if len(region_results) == 0:
            continue
        
        # Aggregate statistics
        lags = [r['peak_lag_sec'] for r in region_results]
        max_xcorrs = [r['max_xcorr'] for r in region_results]
        n_gaba_total = sum([r['n_gaba'] for r in region_results])
        n_non_total = sum([r['n_non'] for r in region_results])
        
        regional_stats[region_name] = {
            'n_planes': len(region_results),
            'mean_lag': np.mean(lags),
            'std_lag': np.std(lags),
            'median_lag': np.median(lags),
            'mean_xcorr': np.mean(max_xcorrs),
            'n_gaba': n_gaba_total,
            'n_non': n_non_total,
            'n_total': n_gaba_total + n_non_total,
            'gaba_fraction': n_gaba_total / (n_gaba_total + n_non_total) if (n_gaba_total + n_non_total) > 0 else 0,
        }
        
        print(f"\n{region_name.upper()} (planes {start}-{end}):")
        print(f"  Analyzed planes: {len(region_results)}")
        print(f"  Total neurons: {n_gaba_total + n_non_total:,} (GABA: {n_gaba_total:,}, Non: {n_non_total:,})")
        print(f"  GABA fraction: {regional_stats[region_name]['gaba_fraction']:.3f}")
        print(f"  Mean lag: {regional_stats[region_name]['mean_lag']:+.3f} ± {regional_stats[region_name]['std_lag']:.3f} s")
        print(f"  Mean max xcorr: {regional_stats[region_name]['mean_xcorr']:.3f}")
    
    return regional_stats


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_compact_summary_figure(all_results, regional_stats, save_dir):
    """Create compact summary figure for thesis."""
    print("\n[CREATING COMPACT SUMMARY FIGURE]")
    
    # Filter valid results
    valid_results = [r for r in all_results if r is not None]
    
    if len(valid_results) == 0:
        print("  No valid results to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # =========================================================================
    # A. All planes lag distribution
    # =========================================================================
    ax1 = plt.subplot(3, 3, 1)
    
    lags = [r['peak_lag_sec'] for r in valid_results]
    planes = [r['plane_idx'] for r in valid_results]
    
    scatter = ax1.scatter(planes, lags, c=lags, cmap='RdBu_r', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                         vmin=-2, vmax=2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Plane Index', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cross-correlation Lag (s)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Lag Across All Planes', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Lag (s)')
    
    # =========================================================================
    # B. Regional comparison
    # =========================================================================
    ax2 = plt.subplot(3, 3, 2)
    
    if regional_stats:
        regions = list(regional_stats.keys())
        mean_lags = [regional_stats[r]['mean_lag'] for r in regions]
        std_lags = [regional_stats[r]['std_lag'] for r in regions]
        
        colors_regional = ['#E57373', '#FFD54F', '#81C784']  # Red, Yellow, Green
        
        bars = ax2.bar(regions, mean_lags, yerr=std_lags, 
                      color=colors_regional, alpha=0.7, 
                      edgecolor='black', linewidth=1.5, capsize=5)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Mean Lag (s)', fontsize=11, fontweight='bold')
        ax2.set_title('B. Regional Comparison', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        # Add n values on bars
        for bar, region in zip(bars, regions):
            n = regional_stats[region]['n_planes']
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (std_lags[regions.index(region)] if height >= 0 else -std_lags[regions.index(region)]),
                    f'n={n}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    # =========================================================================
    # C. GABA fraction by region
    # =========================================================================
    ax3 = plt.subplot(3, 3, 3)
    
    if regional_stats:
        gaba_fracs = [regional_stats[r]['gaba_fraction'] for r in regions]
        
        bars = ax3.bar(regions, gaba_fracs, color=cfg.COLOR_GABA, 
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('GABA+ Fraction', fontsize=11, fontweight='bold')
        ax3.set_title('C. GABAergic Neuron Density', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 0.5])
        ax3.grid(alpha=0.3, axis='y')
        
        # Add percentages
        for bar, frac in zip(bars, gaba_fracs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{100*frac:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    
    # =========================================================================
    # D-I. Representative planes (6 examples)
    # =========================================================================
    # Select representative planes (evenly spaced)
    n_valid = len(valid_results)
    example_indices = np.linspace(0, n_valid-1, min(cfg.N_EXAMPLE_PLANES, n_valid), dtype=int)
    example_results = [valid_results[i] for i in example_indices]
    
    for idx, results in enumerate(example_results):
        ax = plt.subplot(3, 3, 4 + idx)
        
        lags_time = results['lags'] / cfg.FRAME_RATE
        plane = results['plane_idx']
        
        ax.plot(lags_time, results['xcorr'], 'k-', linewidth=1.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(results['peak_lag_sec'], color='red', linestyle='-', linewidth=2)
        ax.fill_between(lags_time, 0, results['xcorr'], alpha=0.3, color='black')
        
        ax.set_xlabel('Lag (s)', fontsize=9)
        ax.set_ylabel('Cross-corr', fontsize=9)
        ax.set_title(f'Plane {plane}: {results["peak_lag_sec"]:+.2f}s\n(n={results["n_gaba"]+results["n_non"]})',
                    fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    
    plt.suptitle('GABA vs Non-GABA Temporal Dynamics - Full Hindbrain Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    save_path = Path(save_dir) / "summary" / "full_hindbrain_summary.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    save_path_pdf = Path(save_dir) / "summary" / "full_hindbrain_summary.pdf"
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


def create_thesis_conclusion_stats(all_results, regional_stats, save_dir):
    """Generate statistics table for thesis conclusion."""
    print("\n[CREATING THESIS STATISTICS TABLE]")
    
    valid_results = [r for r in all_results if r is not None]
    
    # Overall statistics
    all_lags = [r['peak_lag_sec'] for r in valid_results]
    all_xcorrs = [r['max_xcorr'] for r in valid_results]
    total_gaba = sum([r['n_gaba'] for r in valid_results])
    total_non = sum([r['n_non'] for r in valid_results])
    
    stats_summary = {
        'Overall': {
            'n_planes': len(valid_results),
            'n_neurons_total': total_gaba + total_non,
            'n_gaba': total_gaba,
            'n_non_gaba': total_non,
            'gaba_fraction': total_gaba / (total_gaba + total_non),
            'mean_lag_sec': np.mean(all_lags),
            'std_lag_sec': np.std(all_lags),
            'median_lag_sec': np.median(all_lags),
            'mean_max_xcorr': np.mean(all_xcorrs),
            'fraction_synchronized': sum([1 for lag in all_lags if abs(lag) < 0.5]) / len(all_lags),
        }
    }
    
    # Add regional stats
    for region_name, stats in regional_stats.items():
        stats_summary[region_name.capitalize()] = stats
    
    # Create DataFrame
    df = pd.DataFrame(stats_summary).T
    
    # Save
    save_path = Path(save_dir) / "summary" / "thesis_statistics.csv"
    df.to_csv(save_path)
    
    # Print formatted table
    print("\n" + "="*80)
    print("THESIS CONCLUSION STATISTICS")
    print("="*80)
    print(df.to_string())
    print("="*80)
    
    # Save as formatted text
    with open(Path(save_dir) / "summary" / "thesis_statistics.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("THESIS CONCLUSION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY FINDINGS FOR CONCLUSION:\n")
        f.write("="*80 + "\n\n")
        
        overall = stats_summary['Overall']
        
        f.write(f"1. DATASET SIZE:\n")
        f.write(f"   - Analyzed {overall['n_planes']} planes across hindbrain (planes {cfg.PLANES_START}-{cfg.PLANES_END})\n")
        f.write(f"   - Total neurons: {overall['n_neurons_total']:,}\n")
        f.write(f"   - GABAergic: {overall['n_gaba']:,} ({100*overall['gaba_fraction']:.1f}%)\n")
        f.write(f"   - Non-GABAergic: {overall['n_non_gaba']:,} ({100*(1-overall['gaba_fraction']):.1f}%)\n\n")
        
        f.write(f"2. TEMPORAL DYNAMICS:\n")
        f.write(f"   - Mean cross-correlation lag: {overall['mean_lag_sec']:+.3f} ± {overall['std_lag_sec']:.3f} seconds\n")
        f.write(f"   - Median lag: {overall['median_lag_sec']:+.3f} seconds\n")
        f.write(f"   - Fraction synchronized (|lag| < 0.5s): {100*overall['fraction_synchronized']:.1f}%\n")
        f.write(f"   - Mean maximum cross-correlation: {overall['mean_max_xcorr']:.3f}\n\n")
        
        f.write(f"3. INTERPRETATION:\n")
        if abs(overall['mean_lag_sec']) < 0.5:
            f.write(f"   ✓ GABA and Non-GABA populations show SYNCHRONIZED activity\n")
            f.write(f"   ✓ No systematic temporal delay detected across hindbrain\n")
            f.write(f"   ✓ Suggests coordinated activation of inhibitory and excitatory networks\n")
        elif overall['mean_lag_sec'] > 0:
            f.write(f"   ✓ GABAergic neurons lead non-GABAergic by ~{overall['mean_lag_sec']:.2f}s\n")
            f.write(f"   ✓ Suggests feedforward inhibition architecture\n")
        else:
            f.write(f"   ✓ Non-GABAergic neurons lead GABAergic by ~{abs(overall['mean_lag_sec']):.2f}s\n")
            f.write(f"   ✓ Suggests feedback inhibition architecture\n")
        
        if regional_stats:
            f.write(f"\n4. REGIONAL DIFFERENCES:\n")
            for region in ['anterior', 'mid', 'posterior']:
                if region.capitalize() in stats_summary:
                    r_stats = stats_summary[region.capitalize()]
                    f.write(f"   {region.upper()}: lag = {r_stats['mean_lag']:+.3f}s, "
                           f"GABA fraction = {100*r_stats['gaba_fraction']:.1f}%\n")
    
    print(f"\n  ✓ Saved: {save_path}")
    print(f"  ✓ Saved: {save_path.parent / 'thesis_statistics.txt'}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run complete temporal analysis pipeline."""
    
    print("="*80)
    print("FULL HINDBRAIN TEMPORAL ANALYSIS")
    print("="*80)
    print(f"\nOutput: {cfg.OUT_DIR}")
    print(f"Planes: {cfg.PLANES_START}-{cfg.PLANES_END}")
    print(f"Regional divisions: {cfg.REGION_BOUNDARIES}")
    
    try:
        # 1. Load data
        df_class = load_gaba_classification()
        traces = load_full_traces()
        frame_ranges = get_plane_frame_ranges()
        
        # 2. Analyze all planes
        print("\n" + "="*70)
        print(f"ANALYZING PLANES {cfg.PLANES_START}-{cfg.PLANES_END}")
        print("="*70)
        
        all_results = []
        summary_stats = []
        
        planes_to_analyze = range(cfg.PLANES_START, cfg.PLANES_END + 1)
        
        for i, plane_idx in enumerate(planes_to_analyze):
            if (i + 1) % 10 == 0:
                print(f"\n  Progress: {i+1}/{len(planes_to_analyze)} planes analyzed")
            
            try:
                results = analyze_plane(plane_idx, df_class, traces, frame_ranges)
                
                if results is not None:
                    all_results.append(results)
                    
                    summary_stats.append({
                        'plane': plane_idx,
                        'n_gaba': results['n_gaba'],
                        'n_non': results['n_non'],
                        'peak_lag_frames': results['peak_lag'],
                        'peak_lag_sec': results['peak_lag_sec'],
                        'max_xcorr': results['max_xcorr'],
                    })
                    
                    print(f"  ✓ Plane {plane_idx}: lag={results['peak_lag_sec']:+.2f}s, n={results['n_gaba']+results['n_non']}")
                else:
                    print(f"  ⚠️ Plane {plane_idx}: insufficient data")
                    
            except Exception as e:
                print(f"  ❌ Plane {plane_idx}: {e}")
        
        print(f"\n  Total analyzed: {len(all_results)}/{len(planes_to_analyze)} planes")
        
        # 3. Regional analysis
        regional_stats = analyze_by_region(all_results)
        
        # 4. Save summary statistics
        if len(summary_stats) > 0:
            df_summary = pd.DataFrame(summary_stats)
            df_summary.to_csv(Path(cfg.OUT_DIR) / "summary" / "all_planes_statistics.csv", index=False)
            print(f"\n  ✓ Saved detailed statistics")
        
        # 5. Create visualizations
        create_compact_summary_figure(all_results, regional_stats, cfg.OUT_DIR)
        create_thesis_conclusion_stats(all_results, regional_stats, cfg.OUT_DIR)
        
        # 6. Final summary
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        print(f"📁 Results: {cfg.OUT_DIR}")
        print(f"📊 Analyzed planes: {len(all_results)}")
        print(f"📈 Regions analyzed: {len(regional_stats)}")
        print("\nKey outputs:")
        print(f"  - Summary figure: summary/full_hindbrain_summary.png")
        print(f"  - Statistics table: summary/thesis_statistics.csv")
        print(f"  - Formatted text: summary/thesis_statistics.txt")
        
        return all_results, regional_stats
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        return [], {}


if __name__ == "__main__":
    all_results, regional_stats = main()