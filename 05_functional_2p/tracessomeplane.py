"""
GABA TEMPORAL ANALYSIS - VERSÃO CORRIGIDA E COM DEBUG
Fixes:
- Consistência nas chaves do dicionário (n_non vs n_non_gaba)
- Melhor error handling
- Debugging melhorado
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
    # Paths (do código anterior)
    ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY"
    GABA_ANALYSIS_DIR = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\GABA_TEMPORAL_ANALYSIS"
    
    # Planes to analyze
    PLANES_TO_ANALYZE = [50,70,75,77,78,79]
    
    # Imaging parameters
    FRAME_RATE = 2  # Hz
    FRAMES_PER_PLANE = 250  # Aproximadamente
    
    # Analysis parameters
    N_EXAMPLE_TRACES = 5  # Traces de exemplo por grupo
    XCORR_MAX_LAG_SEC = 15  # Maximum lag for cross-correlation (seconds)
    
    # Colors (colorblind-friendly)
    COLOR_NON_GABA = '#D81B60'  # Magenta vibrante
    COLOR_GABA = '#00897B'      # Verde teal

cfg = Config()

# Create output directories
Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
for plane in cfg.PLANES_TO_ANALYZE:
    Path(cfg.OUT_DIR, f"plane_{plane}").mkdir(parents=True, exist_ok=True)

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
    
    # Try different trace files
    trace_files = [
        traces_dir / "traces_2d_dff_p20_WITH_NAN.npz",
        traces_dir / "traces_2d_F_WITH_NAN.npz",
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
    # Remove NaNs
    valid = ~(np.isnan(trace1) | np.isnan(trace2))
    
    if valid.sum() < 50:
        return None, None, None
    
    t1 = trace1[valid]
    t2 = trace2[valid]
    
    # Normalize
    t1 = (t1 - np.mean(t1)) / (np.std(t1) + 1e-10)
    t2 = (t2 - np.mean(t2)) / (np.std(t2) + 1e-10)
    
    # Compute cross-correlation
    xcorr = signal.correlate(t1, t2, mode='full')
    xcorr = xcorr / len(t1)
    
    # Get lags
    lags = signal.correlation_lags(len(t1), len(t2), mode='full')
    
    # Restrict to max_lag
    mask = np.abs(lags) <= max_lag_frames
    lags = lags[mask]
    xcorr = xcorr[mask]
    
    # Find peak
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    
    return lags, xcorr, peak_lag


def analyze_plane(plane_idx, df_class, traces, frame_ranges):
    """Analyze GABA vs Non-GABA dynamics for a single plane."""
    print(f"\n{'='*70}")
    print(f"ANALYZING PLANE {plane_idx}")
    print(f"{'='*70}")
    
    try:
        # Get frame range for this plane
        start_frame, end_frame = frame_ranges[plane_idx]
        
        print(f"  Frame range: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")
        
        # Get ROIs for this plane
        plane_rois = df_class[df_class['plane_idx'] == plane_idx].copy()
        
        # Separate GABA and non-GABA
        gaba_rois = plane_rois[plane_rois['is_gabaergic']]
        non_gaba_rois = plane_rois[~plane_rois['is_gabaergic'] & plane_rois['is_filtered']]
        
        n_gaba = len(gaba_rois)
        n_non = len(non_gaba_rois)
        
        print(f"  GABA neurons: {n_gaba}")
        print(f"  Non-GABA neurons: {n_non}")
        
        if n_gaba < 5 or n_non < 5:
            print(f"  ⚠️ Insufficient neurons in plane {plane_idx}, skipping")
            return None
        
        # Extract traces for this plane's timepoints
        gaba_traces = traces[gaba_rois.index.values, start_frame:end_frame]
        non_gaba_traces = traces[non_gaba_rois.index.values, start_frame:end_frame]
        
        print(f"  GABA traces shape: {gaba_traces.shape}")
        print(f"  Non-GABA traces shape: {non_gaba_traces.shape}")
        
        # Compute population means
        gaba_mean = np.nanmean(gaba_traces, axis=0)
        non_gaba_mean = np.nanmean(non_gaba_traces, axis=0)
        
        # Z-score normalize
        gaba_mean_z = (gaba_mean - np.nanmean(gaba_mean)) / (np.nanstd(gaba_mean) + 1e-10)
        non_gaba_mean_z = (non_gaba_mean - np.nanmean(non_gaba_mean)) / (np.nanstd(non_gaba_mean) + 1e-10)
        
        # Cross-correlation
        max_lag_frames = int(cfg.XCORR_MAX_LAG_SEC * cfg.FRAME_RATE)
        lags, xcorr, peak_lag = compute_cross_correlation(
            gaba_mean_z, non_gaba_mean_z, max_lag_frames
        )
        
        if lags is None:
            print(f"  ⚠️ Cross-correlation failed for plane {plane_idx}")
            return None
        
        # Convert lag to time
        peak_lag_sec = peak_lag / cfg.FRAME_RATE
        
        print(f"  ✓ Cross-correlation peak at lag: {peak_lag} frames ({peak_lag_sec:.2f}s)")
        
        if peak_lag > 0:
            print(f"    → GABA leads Non-GABA by {peak_lag_sec:.2f}s")
        elif peak_lag < 0:
            print(f"    → Non-GABA leads GABA by {abs(peak_lag_sec):.2f}s")
        else:
            print(f"    → Synchronized (no lag)")
        
        # Select example traces
        def select_examples(traces_array, n_examples):
            valid_frac = 1 - np.isnan(traces_array).mean(axis=1)
            trace_std = np.nanstd(traces_array, axis=1)
            
            scores = trace_std * valid_frac
            top_idx = np.argsort(scores)[-n_examples:][::-1]
            
            return traces_array[top_idx]
        
        gaba_examples = select_examples(gaba_traces, cfg.N_EXAMPLE_TRACES)
        non_gaba_examples = select_examples(non_gaba_traces, cfg.N_EXAMPLE_TRACES)
        
        # Z-score normalize examples
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
            'n_non': n_non,  # FIXED: consistent naming
            'start_frame': start_frame,
            'end_frame': end_frame,
            'gaba_mean_z': gaba_mean_z,
            'non_gaba_mean_z': non_gaba_mean_z,
            'lags': lags,
            'xcorr': xcorr,
            'peak_lag': peak_lag,
            'peak_lag_sec': peak_lag_sec,
            'gaba_examples_z': gaba_examples_z,
            'non_gaba_examples_z': non_gaba_examples_z,
        }
        
        # DEBUG: Print keys
        print(f"  DEBUG: Results keys = {list(results.keys())}")
        
        return results
        
    except Exception as e:
        print(f"\n  ❌ ERROR in analyze_plane: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_plane_analysis(results, save_dir):
    """Create comprehensive figure."""
    plane_idx = results['plane_idx']
    n_gaba = results['n_gaba']
    n_non = results['n_non']  # FIXED: consistent naming
    
    print(f"\n[CREATING FIGURE FOR PLANE {plane_idx}]")
    
    try:
        # Time vectors
        n_frames = len(results['gaba_mean_z'])
        time = np.arange(n_frames) / cfg.FRAME_RATE
        
        lags_time = results['lags'] / cfg.FRAME_RATE
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        
        # A. Population Mean Traces
        ax1 = plt.subplot(3, 2, (1, 2))
        
        ax1.plot(time, results['non_gaba_mean_z'], 
                 color=cfg.COLOR_NON_GABA, linewidth=2, label='Non-GABA mean', alpha=0.9)
        ax1.plot(time, results['gaba_mean_z'], 
                 color=cfg.COLOR_GABA, linewidth=2, label='GABA mean', alpha=0.9)
        
        for t in np.arange(0, time[-1], 10):
            ax1.axvline(t, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        
        ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Z-scored dF/F', fontsize=12, fontweight='bold')
        ax1.set_title(f'A. Population Mean Traces (Plane {plane_idx})', 
                      fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax1.grid(alpha=0.3, axis='y')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        lag_text = f"Cross-corr lag: {results['peak_lag']:+d} frames ({results['peak_lag_sec']:+.2f}s)"
        ax1.text(0.02, 0.98, lag_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # B. Cross-correlation
        ax2 = plt.subplot(3, 2, 3)
        
        ax2.plot(lags_time, results['xcorr'], 'k-', linewidth=2)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(results['peak_lag_sec'], color='red', linestyle='-', linewidth=2.5, 
                    label=f'Peak: {results["peak_lag_sec"]:+.2f}s')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax2.fill_between(lags_time, 0, results['xcorr'], alpha=0.3, color='black')
        
        ax2.set_xlabel('Lag (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cross-correlation', fontsize=12, fontweight='bold')
        ax2.set_title('B. Cross-correlation (GABA vs Non)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # C. Example GABA traces
        ax3 = plt.subplot(3, 2, 4)
        
        for i, trace in enumerate(results['gaba_examples_z']):
            offset = i * 5
            valid = ~np.isnan(trace)
            ax3.plot(time[valid], trace[valid] + offset, 
                    color=cfg.COLOR_GABA, linewidth=1, alpha=0.7)
        
        ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Z-score (offset)', fontsize=12, fontweight='bold')
        ax3.set_title(f'C. Example GABA traces (n={len(results["gaba_examples_z"])})', 
                      fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # D. Example Non-GABA traces
        ax4 = plt.subplot(3, 2, 5)
        
        for i, trace in enumerate(results['non_gaba_examples_z']):
            offset = i * 5
            valid = ~np.isnan(trace)
            ax4.plot(time[valid], trace[valid] + offset, 
                    color=cfg.COLOR_NON_GABA, linewidth=1, alpha=0.7)
        
        ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Z-score (offset)', fontsize=12, fontweight='bold')
        ax4.set_title(f'D. Example Non-GABA traces (n={len(results["non_gaba_examples_z"])})', 
                      fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3, axis='x')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        # E. Zoom: First 25s
        ax5 = plt.subplot(3, 2, 6)
        
        zoom_end_sec = 25
        zoom_end_idx = int(zoom_end_sec * cfg.FRAME_RATE)
        zoom_end_idx = min(zoom_end_idx, len(time))
        
        time_zoom = time[:zoom_end_idx]
        
        ax5.plot(time_zoom, results['non_gaba_mean_z'][:zoom_end_idx], 
                 color=cfg.COLOR_NON_GABA, linewidth=2.5, label='Non-GABA', alpha=0.9)
        ax5.plot(time_zoom, results['gaba_mean_z'][:zoom_end_idx], 
                 color=cfg.COLOR_GABA, linewidth=2.5, label='GABA', alpha=0.9)
        
        for t in np.arange(0, zoom_end_sec, 5):
            ax5.axvline(t, color='green', linestyle='--', alpha=0.4, linewidth=1)
        
        ax5.axvspan(0, zoom_end_sec, alpha=0.05, color='blue')
        
        ax5.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Z-scored dF/F', fontsize=12, fontweight='bold')
        ax5.set_title(f'E. Zoom: First {zoom_end_sec}s', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(alpha=0.3, axis='y')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.set_xlim([0, zoom_end_sec])
        
        # Main title
        fig.suptitle(f'Plane {plane_idx}: n_GABA={n_gaba}, n_Non={n_non}',
                     fontsize=15, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save
        save_path = Path(save_dir) / f"plane_{plane_idx}" / f"temporal_analysis_plane{plane_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        save_path_pdf = Path(save_dir) / f"plane_{plane_idx}" / f"temporal_analysis_plane{plane_idx}.pdf"
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
        print(f"  ✓ Saved: {save_path_pdf}")
        
    except Exception as e:
        print(f"\n  ❌ ERROR in plot_plane_analysis: {e}")
        traceback.print_exc()


def create_summary_figure(all_results, save_dir):
    """Create summary figure comparing all planes."""
    print(f"\n[CREATING SUMMARY FIGURE]")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, results in enumerate(all_results):
            if results is None:
                continue
            
            ax = axes[i]
            plane = results['plane_idx']
            
            lags_time = results['lags'] / cfg.FRAME_RATE
            
            ax.plot(lags_time, results['xcorr'], 'k-', linewidth=2)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(results['peak_lag_sec'], color='red', linestyle='-', linewidth=2.5)
            ax.fill_between(lags_time, 0, results['xcorr'], alpha=0.3, color='black')
            
            ax.set_xlabel('Lag (s)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cross-correlation', fontsize=10, fontweight='bold')
            ax.set_title(f'Plane {plane}\nPeak: {results["peak_lag_sec"]:+.2f}s (n={results["n_gaba"]+results["n_non"]})',
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Hide unused subplots
        if len(all_results) < len(axes):
            for j in range(len(all_results), len(axes)):
                axes[j].axis('off')
        
        plt.suptitle('Cross-Correlation Summary Across Planes', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = Path(save_dir) / "summary_cross_correlation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        save_path_pdf = Path(save_dir) / "summary_cross_correlation.pdf"
        plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")
        print(f"  ✓ Saved: {save_path_pdf}")
        
    except Exception as e:
        print(f"\n  ❌ ERROR in create_summary_figure: {e}")
        traceback.print_exc()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run complete temporal analysis pipeline."""
    
    print("="*80)
    print("TEMPORAL ANALYSIS: GABA vs Non-GABA")
    print("="*80)
    print(f"\nOutput: {cfg.OUT_DIR}")
    print(f"Planes to analyze: {cfg.PLANES_TO_ANALYZE}")
    
    try:
        # 1. Load data
        df_class = load_gaba_classification()
        traces = load_full_traces()
        
        # 2. Get frame ranges
        frame_ranges = get_plane_frame_ranges()
        
        # 3. Analyze each plane
        all_results = []
        summary_stats = []
        
        for plane_idx in cfg.PLANES_TO_ANALYZE:
            try:
                results = analyze_plane(plane_idx, df_class, traces, frame_ranges)
                
                if results is not None:
                    all_results.append(results)
                    
                    # Plot individual plane
                    plot_plane_analysis(results, cfg.OUT_DIR)
                    
                    # Save stats
                    summary_stats.append({
                        'plane': plane_idx,
                        'n_gaba': results['n_gaba'],
                        'n_non': results['n_non'],  # FIXED: consistent naming
                        'peak_lag_frames': results['peak_lag'],
                        'peak_lag_sec': results['peak_lag_sec'],
                        'max_xcorr': results['xcorr'][np.argmax(results['xcorr'])],
                    })
            except Exception as e:
                print(f"\n❌ ERROR analyzing plane {plane_idx}: {e}")
                traceback.print_exc()
        
        # 4. Create summary figure
        if len(all_results) > 0:
            create_summary_figure(all_results, cfg.OUT_DIR)
            
            # Save summary stats
            if len(summary_stats) > 0:
                df_summary = pd.DataFrame(summary_stats)
                df_summary.to_csv(Path(cfg.OUT_DIR) / "summary_statistics.csv", index=False)
                
                print("\n" + "="*80)
                print("SUMMARY STATISTICS")
                print("="*80)
                print(df_summary.to_string(index=False))
                
                # Overall interpretation
                mean_lag = df_summary['peak_lag_sec'].mean()
                std_lag = df_summary['peak_lag_sec'].std()
            else:
                df_summary = None
                mean_lag = 0
                std_lag = 0
            
            print("\n" + "="*80)
            print("INTERPRETATION")
            print("="*80)
            
            if len(summary_stats) > 0:
                print(f"Mean cross-correlation lag: {mean_lag:+.2f} ± {std_lag:.2f} seconds")
                
                if abs(mean_lag) < 0.5:
                    print("  → GABA and Non-GABA populations are SYNCHRONIZED")
                    print("  → No consistent temporal delay detected")
                elif mean_lag > 0:
                    print(f"  → GABA neurons lead Non-GABA by ~{mean_lag:.2f}s")
                    print("  → Suggests GABAergic neurons activate BEFORE excitatory")
                else:
                    print(f"  → Non-GABA neurons lead GABA by ~{abs(mean_lag):.2f}s")
                    print("  → Suggests excitatory neurons activate BEFORE inhibitory")
            else:
                print("  No valid data for interpretation")
            
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE")
            print("="*80)
            print(f"📁 Results saved to: {cfg.OUT_DIR}")
            print(f"📊 Analyzed {len(all_results)} planes")
            print(f"📈 Figures generated: {len(all_results) + 1}")
        else:
            print("\n❌ No planes could be analyzed!")
            df_summary = None
        
        return all_results, df_summary
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR in main: {e}")
        traceback.print_exc()
        return [], None


if __name__ == "__main__":
    all_results, df_summary = main()