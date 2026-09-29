"""
=============================================================================
ADVANCED PEAK TIMING ANALYSIS
=============================================================================

Análise avançada de timing de picos e sincronização entre GABA e Non-GABA:
- Peak detection individual
- Inter-peak intervals
- Synchrony analysis
- Leading/lagging behavior

Author: Inês Marques
Date: February 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY"
    GABA_ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\GABA_PEAK_TIMING_ANALYSIS"
    
    # Analysis parameters
    PLANES_TO_ANALYZE = [60, 70, 80, 90, 100]
    FRAME_RATE = 2  # Hz
    FRAMES_PER_PLANE = 250
    
    # Peak detection
    PEAK_THRESHOLD_SD = 2.0  # SD above mean
    MIN_PEAK_DISTANCE = 5  # frames (2.5s at 2Hz)
    
    # Synchrony
    SYNC_WINDOW_SEC = 2.5  # Window for considering peaks synchronized
    
    # Colors
    COLOR_NON_GABA = '#D81B60'
    COLOR_GABA = '#00897B'

cfg = Config()
Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)


# =============================================================================
# PEAK DETECTION
# =============================================================================

def detect_peaks_trace(trace, frame_rate=2):
    """
    Detect peaks in a single trace.
    
    Returns:
        peak_times: array of peak times in seconds
        peak_heights: array of peak heights
    """
    valid = ~np.isnan(trace)
    if valid.sum() < 50:
        return np.array([]), np.array([])
    
    trace_valid = trace[valid]
    valid_indices = np.where(valid)[0]
    
    # Calculate threshold
    mean_val = np.mean(trace_valid)
    std_val = np.std(trace_valid)
    threshold = mean_val + cfg.PEAK_THRESHOLD_SD * std_val
    
    # Find peaks
    peaks, properties = find_peaks(
        trace_valid, 
        height=threshold,
        distance=cfg.MIN_PEAK_DISTANCE
    )
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # Convert to original time indices
    peak_frames = valid_indices[peaks]
    peak_times = peak_frames / frame_rate
    peak_heights = properties['peak_heights']
    
    return peak_times, peak_heights


def analyze_population_peaks(traces, frame_rate=2):
    """
    Analyze peaks across a population of traces.
    
    Returns:
        all_peak_times: list of arrays (one per ROI)
        all_peak_heights: list of arrays
        summary_stats: dict with population statistics
    """
    n_rois = traces.shape[0]
    
    all_peak_times = []
    all_peak_heights = []
    n_peaks_per_roi = []
    
    for i in range(n_rois):
        peak_times, peak_heights = detect_peaks_trace(traces[i], frame_rate)
        all_peak_times.append(peak_times)
        all_peak_heights.append(peak_heights)
        n_peaks_per_roi.append(len(peak_times))
    
    # Summary statistics
    summary = {
        'n_rois': n_rois,
        'mean_peaks_per_roi': np.mean(n_peaks_per_roi),
        'std_peaks_per_roi': np.std(n_peaks_per_roi),
        'total_peaks': sum(n_peaks_per_roi),
        'rois_with_peaks': sum(1 for n in n_peaks_per_roi if n > 0),
    }
    
    return all_peak_times, all_peak_heights, summary


def compute_synchrony(peak_times_list, sync_window_sec=2.5):
    """
    Compute synchrony: fraction of peaks that occur within sync_window of each other.
    
    Args:
        peak_times_list: list of arrays, one per ROI
        sync_window_sec: time window for synchrony
    
    Returns:
        synchrony_score: 0-1, higher = more synchronous
    """
    # Pool all peak times
    all_peaks = np.concatenate([p for p in peak_times_list if len(p) > 0])
    
    if len(all_peaks) < 2:
        return 0.0
    
    # Sort
    all_peaks = np.sort(all_peaks)
    
    # Count peaks within sync_window
    n_synchronized = 0
    
    for i, peak in enumerate(all_peaks):
        # Count how many other peaks within window
        within_window = np.sum(np.abs(all_peaks - peak) <= sync_window_sec) - 1  # -1 to exclude itself
        if within_window > 0:
            n_synchronized += 1
    
    synchrony_score = n_synchronized / len(all_peaks)
    
    return synchrony_score


def compare_peak_timing(gaba_peak_times, non_gaba_peak_times, sync_window_sec=2.5):
    """
    Compare timing between GABA and non-GABA peaks.
    
    Returns:
        dict with comparison metrics
    """
    # Pool peaks
    gaba_all = np.concatenate([p for p in gaba_peak_times if len(p) > 0])
    non_all = np.concatenate([p for p in non_gaba_peak_times if len(p) > 0])
    
    if len(gaba_all) == 0 or len(non_all) == 0:
        return None
    
    # For each GABA peak, find closest non-GABA peak
    delays_gaba_to_non = []
    
    for gaba_peak in gaba_all:
        diffs = non_all - gaba_peak
        closest_idx = np.argmin(np.abs(diffs))
        delay = diffs[closest_idx]
        
        # Only consider if within reasonable window
        if abs(delay) <= sync_window_sec:
            delays_gaba_to_non.append(delay)
    
    # For each non-GABA peak, find closest GABA peak
    delays_non_to_gaba = []
    
    for non_peak in non_all:
        diffs = gaba_all - non_peak
        closest_idx = np.argmin(np.abs(diffs))
        delay = diffs[closest_idx]
        
        if abs(delay) <= sync_window_sec:
            delays_non_to_gaba.append(delay)
    
    if len(delays_gaba_to_non) == 0:
        return None
    
    results = {
        'mean_delay_gaba_to_non': np.mean(delays_gaba_to_non),
        'std_delay_gaba_to_non': np.std(delays_gaba_to_non),
        'median_delay_gaba_to_non': np.median(delays_gaba_to_non),
        'n_paired_peaks': len(delays_gaba_to_non),
        'fraction_gaba_leads': np.sum(np.array(delays_gaba_to_non) < 0) / len(delays_gaba_to_non),
        'fraction_non_leads': np.sum(np.array(delays_gaba_to_non) > 0) / len(delays_gaba_to_non),
    }
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_peak_timing_analysis(results, plane_idx, save_dir):
    """
    Create comprehensive peak timing figure.
    """
    print(f"\n[CREATING PEAK TIMING FIGURE FOR PLANE {plane_idx}]")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    gaba_stats = results['gaba_peak_stats']
    non_stats = results['non_peak_stats']
    timing_comp = results['timing_comparison']
    
    # =========================================================================
    # A. Peaks per ROI distribution
    # =========================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    gaba_n_peaks = [len(p) for p in results['gaba_peak_times']]
    non_n_peaks = [len(p) for p in results['non_peak_times']]
    
    bins = np.arange(0, max(max(gaba_n_peaks), max(non_n_peaks)) + 2)
    
    ax1.hist(non_n_peaks, bins=bins, alpha=0.6, color=cfg.COLOR_NON_GABA, 
            label=f'Non-GABA (mean={np.mean(non_n_peaks):.1f})', edgecolor='black')
    ax1.hist(gaba_n_peaks, bins=bins, alpha=0.6, color=cfg.COLOR_GABA,
            label=f'GABA (mean={np.mean(gaba_n_peaks):.1f})', edgecolor='black')
    
    ax1.set_xlabel('Peaks per ROI', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('A. Peak Count Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # =========================================================================
    # B. Peak timing histogram
    # =========================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    gaba_all_times = np.concatenate([p for p in results['gaba_peak_times'] if len(p) > 0])
    non_all_times = np.concatenate([p for p in results['non_peak_times'] if len(p) > 0])
    
    time_bins = np.linspace(0, cfg.FRAMES_PER_PLANE / cfg.FRAME_RATE, 30)
    
    ax2.hist(non_all_times, bins=time_bins, alpha=0.6, color=cfg.COLOR_NON_GABA,
            label=f'Non-GABA (n={len(non_all_times)})', edgecolor='black')
    ax2.hist(gaba_all_times, bins=time_bins, alpha=0.6, color=cfg.COLOR_GABA,
            label=f'GABA (n={len(gaba_all_times)})', edgecolor='black')
    
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Peak Count', fontsize=11, fontweight='bold')
    ax2.set_title('B. Peak Timing Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # =========================================================================
    # C. Synchrony scores
    # =========================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    sync_scores = [
        results['gaba_synchrony'],
        results['non_synchrony'],
    ]
    labels = ['GABA', 'Non-GABA']
    colors = [cfg.COLOR_GABA, cfg.COLOR_NON_GABA]
    
    bars = ax3.bar(labels, sync_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax3.set_ylabel('Synchrony Score', fontsize=11, fontweight='bold')
    ax3.set_title('C. Within-Population Synchrony', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(alpha=0.3, axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add values on bars
    for bar, score in zip(bars, sync_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # =========================================================================
    # D. Peak delay distribution
    # =========================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    if timing_comp is not None:
        # Get all delays
        gaba_all = np.concatenate([p for p in results['gaba_peak_times'] if len(p) > 0])
        non_all = np.concatenate([p for p in results['non_peak_times'] if len(p) > 0])
        
        delays = []
        for gaba_peak in gaba_all:
            diffs = non_all - gaba_peak
            closest_idx = np.argmin(np.abs(diffs))
            delay = diffs[closest_idx]
            if abs(delay) <= cfg.SYNC_WINDOW_SEC:
                delays.append(delay)
        
        delay_bins = np.linspace(-cfg.SYNC_WINDOW_SEC, cfg.SYNC_WINDOW_SEC, 30)
        
        ax4.hist(delays, bins=delay_bins, color='gray', alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='No delay')
        ax4.axvline(np.mean(delays), color='red', linestyle='-', linewidth=2.5,
                   label=f'Mean: {np.mean(delays):.2f}s')
        
        ax4.set_xlabel('Delay (s, positive = Non-GABA leads)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('D. Peak Delay: GABA to Nearest Non-GABA', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    
    # =========================================================================
    # E. Summary statistics table
    # =========================================================================
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    if timing_comp is not None:
        table_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Mean delay (GABA→Non)', f"{timing_comp['mean_delay_gaba_to_non']:.3f}s"],
            ['Median delay', f"{timing_comp['median_delay_gaba_to_non']:.3f}s"],
            ['Std delay', f"{timing_comp['std_delay_gaba_to_non']:.3f}s"],
            ['', ''],
            ['% GABA leads', f"{100*timing_comp['fraction_gaba_leads']:.1f}%"],
            ['% Non-GABA leads', f"{100*timing_comp['fraction_non_leads']:.1f}%"],
            ['', ''],
            ['Paired peaks', f"{timing_comp['n_paired_peaks']}"],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('E. Timing Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # =========================================================================
    # F. Interpretation
    # =========================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    if timing_comp is not None:
        mean_delay = timing_comp['mean_delay_gaba_to_non']
        
        if abs(mean_delay) < 0.25:
            interpretation = "✅ SYNCHRONIZED"
            conclusion = (
                f"Mean delay: {mean_delay:.3f}s\n\n"
                f"GABA and Non-GABA populations show\n"
                f"highly synchronized peak activity.\n\n"
                f"No consistent leading/lagging behavior."
            )
            color = '#4CAF50'
        elif mean_delay < 0:
            interpretation = "🔴 GABA LEADS"
            conclusion = (
                f"Mean delay: {mean_delay:.3f}s\n\n"
                f"GABA neurons tend to peak BEFORE\n"
                f"Non-GABA neurons by ~{abs(mean_delay):.2f}s.\n\n"
                f"Suggests feedforward inhibition or\n"
                f"GABAergic initiation of network events."
            )
            color = cfg.COLOR_GABA
        else:
            interpretation = "🔵 NON-GABA LEADS"
            conclusion = (
                f"Mean delay: {mean_delay:.3f}s\n\n"
                f"Non-GABA neurons tend to peak BEFORE\n"
                f"GABA neurons by ~{mean_delay:.2f}s.\n\n"
                f"Suggests feedback inhibition or\n"
                f"excitation-driven GABAergic activity."
            )
            color = cfg.COLOR_NON_GABA
        
        text = f"{interpretation}\n\n{conclusion}"
        
        ax6.text(0.5, 0.5, text, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
                transform=ax6.transAxes)
    
    ax6.set_title('F. Interpretation', fontsize=12, fontweight='bold', pad=20)
    
    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle(f'Peak Timing Analysis - Plane {plane_idx}\n'
                 f'GABA: {gaba_stats["n_rois"]} ROIs, {gaba_stats["total_peaks"]} peaks | '
                 f'Non-GABA: {non_stats["n_rois"]} ROIs, {non_stats["total_peaks"]} peaks',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_path = Path(save_dir) / f"peak_timing_plane{plane_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    save_path_pdf = Path(save_dir) / f"peak_timing_plane{plane_idx}.pdf"
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run peak timing analysis."""
    
    print("="*80)
    print("PEAK TIMING ANALYSIS")
    print("="*80)
    
    # Load data (reuse from previous script)
    sys.path.insert(0, str(Path(__file__).parent))
    from tracessomeplanes import (
        load_gaba_classification,
        load_full_traces,
        get_plane_frame_ranges
    )
    
    df_class = load_gaba_classification()
    traces = load_full_traces()
    frame_ranges = get_plane_frame_ranges()
    
    # Analyze each plane
    all_results = []
    
    for plane_idx in cfg.PLANES_TO_ANALYZE:
        print(f"\n{'='*70}")
        print(f"ANALYZING PLANE {plane_idx}")
        print(f"{'='*70}")
        
        # Get ROIs
        plane_rois = df_class[df_class['plane_idx'] == plane_idx]
        gaba_rois = plane_rois[plane_rois['is_gabaergic']]
        non_gaba_rois = plane_rois[~plane_rois['is_gabaergic'] & plane_rois['is_filtered']]
        
        if len(gaba_rois) < 5 or len(non_gaba_rois) < 5:
            print(f"  Insufficient neurons, skipping")
            continue
        
        # Extract traces
        start_frame, end_frame = frame_ranges[plane_idx]
        gaba_traces = traces[gaba_rois.index.values, start_frame:end_frame]
        non_gaba_traces = traces[non_gaba_rois.index.values, start_frame:end_frame]
        
        # Analyze peaks
        print(f"  Detecting GABA peaks...")
        gaba_peak_times, gaba_peak_heights, gaba_stats = analyze_population_peaks(gaba_traces, cfg.FRAME_RATE)
        
        print(f"  Detecting Non-GABA peaks...")
        non_peak_times, non_peak_heights, non_stats = analyze_population_peaks(non_gaba_traces, cfg.FRAME_RATE)
        
        # Compute synchrony
        print(f"  Computing synchrony...")
        gaba_sync = compute_synchrony(gaba_peak_times, cfg.SYNC_WINDOW_SEC)
        non_sync = compute_synchrony(non_peak_times, cfg.SYNC_WINDOW_SEC)
        
        # Compare timing
        print(f"  Comparing peak timing...")
        timing_comp = compare_peak_timing(gaba_peak_times, non_peak_times, cfg.SYNC_WINDOW_SEC)
        
        results = {
            'plane_idx': plane_idx,
            'gaba_peak_times': gaba_peak_times,
            'gaba_peak_heights': gaba_peak_heights,
            'gaba_peak_stats': gaba_stats,
            'non_peak_times': non_peak_times,
            'non_peak_heights': non_peak_heights,
            'non_peak_stats': non_stats,
            'gaba_synchrony': gaba_sync,
            'non_synchrony': non_sync,
            'timing_comparison': timing_comp,
        }
        
        all_results.append(results)
        
        # Plot
        plot_peak_timing_analysis(results, plane_idx, cfg.OUT_DIR)
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    GABA: {gaba_stats['total_peaks']} peaks, sync={gaba_sync:.3f}")
        print(f"    Non-GABA: {non_stats['total_peaks']} peaks, sync={non_sync:.3f}")
        if timing_comp:
            print(f"    Mean delay: {timing_comp['mean_delay_gaba_to_non']:.3f}s")
    
    print("\n" + "="*80)
    print("✅ PEAK TIMING ANALYSIS COMPLETE")
    print("="*80)
    print(f"📁 Output: {cfg.OUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    import sys
    all_results = main()