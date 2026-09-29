"""
=============================================================================
SPATIAL ANALYSIS - GABA DISTRIBUTION & LATERALITY
=============================================================================

Analyzes spatial patterns of GABA vs non-GABA neurons:
1. Left-Right distribution (laterality)
2. Rostro-caudal distribution (anterior-posterior axis)
3. Regional clustering
4. Anatomical hotspots

Uses:
- Detected midline (215.8)
- Cerebellum region (~Z=70, X=390-600)
- Green/Magenta color scheme

Author: Inês Marques
Date: February 2025
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Color scheme
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    CLASSIFICATION_FILE = r"C:\Users\OSVALDO\Downloads\2P_GABA_TRACES_ANALYSIS\classification\gaba_classification.csv"
    ROI_COORDS_FILE = r"C:\Users\OSVALDO\Downloads\ANALYSIS_READY\rois_2d\roi_2d_coordinates.npz"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\2P_SPATIAL_ANALYSIS"
    
    # Anatomical landmarks
    MIDLINE_X = 215.8  # Detected midline
    
    # Cerebellum region
    CEREBELLUM_Z_MIN = 0
    CEREBELLUM_Z_MAX = 70
    CEREBELLUM_X_MIN = 390
    CEREBELLUM_X_MAX = 600
    
    # Image dimensions
    IMAGE_HEIGHT = 464  # Y
    IMAGE_WIDTH = 720   # X
    N_PLANES = 180      # Z
    
    # Colors (Green/Magenta scheme)
    COLOR_GABA = '#FF00FF'      # Magenta
    COLOR_NON_GABA = '#00FF00'  # Green
    COLOR_GABA_ALT = '#D946EF'  # Lighter magenta
    COLOR_NON_GABA_ALT = '#10B981'  # Lighter green

cfg = Config()

# Create output directories
for subdir in ["figures", "results", "maps"]:
    Path(cfg.OUT_DIR, subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_classification_and_coords():
    """Load GABA classification and ROI coordinates."""
    print("\n" + "="*70)
    print("LOADING CLASSIFICATION & COORDINATES")
    print("="*70)
    
    # Classification
    df_class = pd.read_csv(cfg.CLASSIFICATION_FILE)
    print(f"  ✓ Loaded classification: {len(df_class):,} ROIs")
    print(f"    GABA+: {df_class['is_gabaergic'].sum():,}")
    print(f"    Non-GABA: {(~df_class['is_gabaergic'] & df_class['is_filtered']).sum():,}")
    
    # Coordinates
    coords_data = np.load(cfg.ROI_COORDS_FILE, allow_pickle=True)
    coords = {
        'plane_idx': coords_data['plane_idx'],
        'centroid_x': coords_data['centroid_x'],
        'centroid_y': coords_data['centroid_y'],
    }
    print(f"  ✓ Loaded coordinates: {len(coords['plane_idx']):,} ROIs")
    
    # Add coordinates to classification DataFrame
    df_class['plane'] = coords['plane_idx']
    df_class['centroid_x'] = coords['centroid_x']
    df_class['centroid_y'] = coords['centroid_y']
    
    # Calculate laterality (distance from midline)
    df_class['distance_from_midline'] = df_class['centroid_x'] - cfg.MIDLINE_X
    df_class['hemisphere'] = df_class['distance_from_midline'].apply(
        lambda x: 'LEFT' if x < 0 else 'RIGHT'
    )
    
    # Identify cerebellum region
    df_class['in_cerebellum'] = (
        (df_class['plane'] >= cfg.CEREBELLUM_Z_MIN) &
        (df_class['plane'] <= cfg.CEREBELLUM_Z_MAX) &
        (df_class['centroid_x'] >= cfg.CEREBELLUM_X_MIN) &
        (df_class['centroid_x'] <= cfg.CEREBELLUM_X_MAX)
    )
    
    print(f"\n  Cerebellum ROIs: {df_class['in_cerebellum'].sum():,}")
    print(f"  Left hemisphere: {(df_class['hemisphere'] == 'LEFT').sum():,}")
    print(f"  Right hemisphere: {(df_class['hemisphere'] == 'RIGHT').sum():,}")
    
    return df_class


# =============================================================================
# 2. LATERALITY ANALYSIS
# =============================================================================

def analyze_laterality(df):
    """Analyze left-right distribution of GABA vs non-GABA."""
    print("\n" + "="*70)
    print("LATERALITY ANALYSIS")
    print("="*70)
    
    # Filter for analyzed ROIs
    df_filt = df[df['is_filtered']].copy()
    
    gaba = df_filt[df_filt['is_gabaergic']]
    non_gaba = df_filt[~df_filt['is_gabaergic']]
    
    results = {}
    
    # Overall laterality
    print("\n[Overall Distribution]")
    
    for name, data in [('GABA', gaba), ('Non-GABA', non_gaba)]:
        left_count = (data['hemisphere'] == 'LEFT').sum()
        right_count = (data['hemisphere'] == 'RIGHT').sum()
        total = len(data)
        
        left_pct = 100 * left_count / total
        right_pct = 100 * right_count / total
        
        print(f"\n  {name}:")
        print(f"    LEFT:  {left_count:,} ({left_pct:.1f}%)")
        print(f"    RIGHT: {right_count:,} ({right_pct:.1f}%)")
        
        results[name.lower().replace('-', '_')] = {
            'left_count': left_count,
            'right_count': right_count,
            'left_pct': left_pct,
            'right_pct': right_pct,
        }
    
    # Chi-square test for laterality preference
    print("\n[Statistical Test]")
    
    contingency = np.array([
        [results['gaba']['left_count'], results['gaba']['right_count']],
        [results['non_gaba']['left_count'], results['non_gaba']['right_count']]
    ])
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"  Chi-square test:")
    print(f"    χ² = {chi2:.2f}, p = {p:.2e}")
    print(f"    {'Significant' if p < 0.05 else 'Not significant'} laterality difference")
    
    results['chi2_test'] = {
        'chi2': float(chi2),
        'p': float(p),
        'significant': p < 0.05
    }
    
    # Distance from midline
    print("\n[Distance from Midline]")
    
    gaba_dist = np.abs(gaba['distance_from_midline'])
    non_gaba_dist = np.abs(non_gaba['distance_from_midline'])
    
    print(f"  GABA:     {gaba_dist.mean():.1f} ± {gaba_dist.std():.1f} pixels")
    print(f"  Non-GABA: {non_gaba_dist.mean():.1f} ± {non_gaba_dist.std():.1f} pixels")
    
    stat, p_dist = stats.mannwhitneyu(gaba_dist, non_gaba_dist)
    print(f"  Mann-Whitney U = {stat:.1f}, p = {p_dist:.2e}")
    
    results['distance_from_midline'] = {
        'gaba_mean': float(gaba_dist.mean()),
        'gaba_std': float(gaba_dist.std()),
        'non_gaba_mean': float(non_gaba_dist.mean()),
        'non_gaba_std': float(non_gaba_dist.std()),
        'p_value': float(p_dist)
    }
    
    return results


def plot_laterality_analysis(df, laterality_results, save_dir):
    """Create laterality analysis figure."""
    print("\n[CREATING LATERALITY FIGURE]")
    
    df_filt = df[df['is_filtered']].copy()
    gaba = df_filt[df_filt['is_gabaergic']]
    non_gaba = df_filt[~df_filt['is_gabaergic']]
    
    fig = plt.figure(figsize=(16, 10))
    
    # === Panel A: Distribution across midline ===
    ax1 = plt.subplot(2, 3, 1)
    
    bins = np.linspace(-cfg.MIDLINE_X, cfg.IMAGE_WIDTH - cfg.MIDLINE_X, 60)
    
    ax1.hist(non_gaba['distance_from_midline'], bins=bins, alpha=0.6,
             color=cfg.COLOR_NON_GABA, label='Non-GABA', density=True)
    ax1.hist(gaba['distance_from_midline'], bins=bins, alpha=0.6,
             color=cfg.COLOR_GABA, label='GABA', density=True)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=2, label='Midline')
    ax1.set_xlabel('Distance from Midline (pixels)')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Distribution Across Midline', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add L/R labels
    ax1.text(-150, ax1.get_ylim()[1]*0.9, 'LEFT', fontsize=12, 
             fontweight='bold', ha='center')
    ax1.text(150, ax1.get_ylim()[1]*0.9, 'RIGHT', fontsize=12,
             fontweight='bold', ha='center')
    
    # === Panel B: Hemisphere counts ===
    ax2 = plt.subplot(2, 3, 2)
    
    x = np.arange(2)
    width = 0.35
    
    gaba_counts = [laterality_results['gaba']['left_count'],
                   laterality_results['gaba']['right_count']]
    non_gaba_counts = [laterality_results['non_gaba']['left_count'],
                       laterality_results['non_gaba']['right_count']]
    
    ax2.bar(x - width/2, gaba_counts, width, label='GABA',
            color=cfg.COLOR_GABA, alpha=0.8)
    ax2.bar(x + width/2, non_gaba_counts, width, label='Non-GABA',
            color=cfg.COLOR_NON_GABA, alpha=0.8)
    
    ax2.set_ylabel('Count')
    ax2.set_title('B. Hemisphere Distribution', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['LEFT', 'RIGHT'])
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    # Add percentages
    for i, (g, ng) in enumerate(zip(gaba_counts, non_gaba_counts)):
        g_pct = 100 * g / sum(gaba_counts)
        ng_pct = 100 * ng / sum(non_gaba_counts)
        ax2.text(i - width/2, g + 100, f'{g_pct:.1f}%', 
                ha='center', fontsize=8, fontweight='bold')
        ax2.text(i + width/2, ng + 100, f'{ng_pct:.1f}%',
                ha='center', fontsize=8, fontweight='bold')
    
    # === Panel C: Chi-square result ===
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    
    chi2_result = laterality_results['chi2_test']
    
    text = f"""
LATERALITY TEST

Chi-Square Test:
  χ² = {chi2_result['chi2']:.2f}
  p = {chi2_result['p']:.2e}
  
Result: {'SIGNIFICANT' if chi2_result['significant'] else 'NOT SIGNIFICANT'}

{'GABA neurons show laterality bias' if chi2_result['significant'] 
 else 'No significant laterality preference'}

Distance from Midline:
  GABA: {laterality_results['distance_from_midline']['gaba_mean']:.1f} px
  Non-GABA: {laterality_results['distance_from_midline']['non_gaba_mean']:.1f} px
  p = {laterality_results['distance_from_midline']['p_value']:.2e}
    """
    
    color = '#FF00FF' if chi2_result['significant'] else '#808080'
    
    ax3.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.2),
             transform=ax3.transAxes, family='monospace')
    
    # === Panel D: 2D spatial distribution (dorsal view) ===
    ax4 = plt.subplot(2, 3, 4)
    
    # Downsample for visualization
    n_plot = min(5000, len(non_gaba))
    non_gaba_sample = non_gaba.sample(n_plot)
    gaba_sample = gaba.sample(min(n_plot, len(gaba)))
    
    ax4.scatter(non_gaba_sample['centroid_x'], non_gaba_sample['centroid_y'],
                s=1, alpha=0.3, c=cfg.COLOR_NON_GABA, label='Non-GABA')
    ax4.scatter(gaba_sample['centroid_x'], gaba_sample['centroid_y'],
                s=1, alpha=0.5, c=cfg.COLOR_GABA, label='GABA')
    
    # Midline
    ax4.axvline(cfg.MIDLINE_X, color='white', linestyle='--', linewidth=2,
                label='Midline')
    
    # Cerebellum region
    from matplotlib.patches import Rectangle
    cerebellum_box = Rectangle(
        (cfg.CEREBELLUM_X_MIN, 0), 
        cfg.CEREBELLUM_X_MAX - cfg.CEREBELLUM_X_MIN,
        cfg.IMAGE_HEIGHT,
        linewidth=2, edgecolor='yellow', facecolor='none',
        linestyle='--', label='Cerebellum\n(approx)'
    )
    ax4.add_patch(cerebellum_box)
    
    ax4.set_xlabel('X Position (pixels)')
    ax4.set_ylabel('Y Position (pixels)')
    ax4.set_title('D. Spatial Distribution (Dorsal View, All Planes)', fontweight='bold')
    ax4.set_xlim(0, cfg.IMAGE_WIDTH)
    ax4.set_ylim(cfg.IMAGE_HEIGHT, 0)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_facecolor('black')
    
    # Add L/R labels
    ax4.text(cfg.MIDLINE_X - 100, 50, 'L', color='white', fontsize=20,
             fontweight='bold', ha='center')
    ax4.text(cfg.MIDLINE_X + 100, 50, 'R', color='white', fontsize=20,
             fontweight='bold', ha='center')
    
    # === Panel E: Absolute distance from midline ===
    ax5 = plt.subplot(2, 3, 5)
    
    gaba_abs_dist = np.abs(gaba['distance_from_midline'])
    non_gaba_abs_dist = np.abs(non_gaba['distance_from_midline'])
    
    parts = ax5.violinplot([gaba_abs_dist, non_gaba_abs_dist],
                           positions=[1, 2], showmeans=True, showmedians=True)
    
    parts['bodies'][0].set_facecolor(cfg.COLOR_GABA)
    parts['bodies'][1].set_facecolor(cfg.COLOR_NON_GABA)
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_alpha(0.7)
    
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels(['GABA', 'Non-GABA'])
    ax5.set_ylabel('Distance from Midline (pixels)')
    ax5.set_title('E. Proximity to Midline', fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    
    p_dist = laterality_results['distance_from_midline']['p_value']
    sig = '***' if p_dist < 0.001 else '**' if p_dist < 0.01 else '*' if p_dist < 0.05 else 'ns'
    ax5.text(1.5, ax5.get_ylim()[1]*0.95, f'p = {p_dist:.2e} ({sig})',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === Panel F: Per-plane laterality ===
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate laterality index per plane
    planes = np.arange(cfg.N_PLANES)
    
    gaba_laterality = []
    non_gaba_laterality = []
    
    for z in planes:
        gaba_z = gaba[gaba['plane'] == z]
        non_gaba_z = non_gaba[non_gaba['plane'] == z]
        
        if len(gaba_z) > 0:
            left = (gaba_z['hemisphere'] == 'LEFT').sum()
            right = (gaba_z['hemisphere'] == 'RIGHT').sum()
            laterality_index = (right - left) / (right + left) if (right + left) > 0 else 0
            gaba_laterality.append(laterality_index)
        else:
            gaba_laterality.append(np.nan)
        
        if len(non_gaba_z) > 0:
            left = (non_gaba_z['hemisphere'] == 'LEFT').sum()
            right = (non_gaba_z['hemisphere'] == 'RIGHT').sum()
            laterality_index = (right - left) / (right + left) if (right + left) > 0 else 0
            non_gaba_laterality.append(laterality_index)
        else:
            non_gaba_laterality.append(np.nan)
    
    # Smooth for visualization
    from scipy.ndimage import gaussian_filter1d
    gaba_smooth = gaussian_filter1d(np.nan_to_num(gaba_laterality), sigma=3)
    non_gaba_smooth = gaussian_filter1d(np.nan_to_num(non_gaba_laterality), sigma=3)
    
    ax6.plot(planes, gaba_smooth, color=cfg.COLOR_GABA, linewidth=2,
             label='GABA', alpha=0.8)
    ax6.plot(planes, non_gaba_smooth, color=cfg.COLOR_NON_GABA, linewidth=2,
             label='Non-GABA', alpha=0.8)
    
    ax6.axhline(0, color='black', linestyle='--', linewidth=1)
    ax6.fill_between(planes, -1, 0, alpha=0.1, color='blue', label='LEFT bias')
    ax6.fill_between(planes, 0, 1, alpha=0.1, color='red', label='RIGHT bias')
    
    # Cerebellum region
    ax6.axvspan(cfg.CEREBELLUM_Z_MIN, cfg.CEREBELLUM_Z_MAX, 
                alpha=0.2, color='yellow', label='Cerebellum')
    
    ax6.set_xlabel('Z Plane')
    ax6.set_ylabel('Laterality Index\n(+1=RIGHT, -1=LEFT)')
    ax6.set_title('F. Laterality Across Z-Planes', fontweight='bold')
    ax6.set_ylim(-1, 1)
    ax6.legend(fontsize=7, loc='upper right')
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Laterality Analysis: GABA vs Non-GABA Neurons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "figures" / "laterality_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


# =============================================================================
# 3. ROSTRO-CAUDAL ANALYSIS
# =============================================================================

def analyze_rostrocaudal(df):
    """Analyze anterior-posterior (rostro-caudal) distribution."""
    print("\n" + "="*70)
    print("ROSTRO-CAUDAL ANALYSIS")
    print("="*70)
    
    df_filt = df[df['is_filtered']].copy()
    gaba = df_filt[df_filt['is_gabaergic']]
    non_gaba = df_filt[~df_filt['is_gabaergic']]
    
    results = {}
    
    # Define regions (rough estimates)
    # More caudal (posterior) = lower X values
    # More rostral (anterior) = higher X values
    
    regions = {
        'Caudal (0-240)': (0, 240),
        'Mid-Caudal (240-390)': (240, 390),
        'Cerebellum (390-600)': (390, 600),
        'Rostral (600-720)': (600, 720)
    }
    
    print("\n[Distribution by Region]")
    print(f"{'Region':<25} | {'GABA':>12} | {'Non-GABA':>12} | {'GABA %':>8}")
    print("-" * 70)
    
    for region_name, (x_min, x_max) in regions.items():
        gaba_in_region = gaba[
            (gaba['centroid_x'] >= x_min) & (gaba['centroid_x'] < x_max)
        ]
        non_gaba_in_region = non_gaba[
            (non_gaba['centroid_x'] >= x_min) & (non_gaba['centroid_x'] < x_max)
        ]
        
        total_in_region = len(gaba_in_region) + len(non_gaba_in_region)
        gaba_pct = 100 * len(gaba_in_region) / total_in_region if total_in_region > 0 else 0
        
        print(f"{region_name:<25} | {len(gaba_in_region):>12,} | "
              f"{len(non_gaba_in_region):>12,} | {gaba_pct:>7.1f}%")
        
        results[region_name] = {
            'gaba_count': len(gaba_in_region),
            'non_gaba_count': len(non_gaba_in_region),
            'gaba_pct': gaba_pct
        }
    
    # Z-plane distribution
    print("\n[Z-Plane Distribution]")
    print(f"  GABA:     Z = {gaba['plane'].mean():.1f} ± {gaba['plane'].std():.1f}")
    print(f"  Non-GABA: Z = {non_gaba['plane'].mean():.1f} ± {non_gaba['plane'].std():.1f}")
    
    stat, p_z = stats.mannwhitneyu(gaba['plane'], non_gaba['plane'])
    print(f"  Mann-Whitney U = {stat:.1f}, p = {p_z:.2e}")
    
    results['z_distribution'] = {
        'gaba_mean': float(gaba['plane'].mean()),
        'gaba_std': float(gaba['plane'].std()),
        'non_gaba_mean': float(non_gaba['plane'].mean()),
        'non_gaba_std': float(non_gaba['plane'].std()),
        'p_value': float(p_z)
    }
    
    # X position distribution
    print("\n[X Position Distribution]")
    print(f"  GABA:     X = {gaba['centroid_x'].mean():.1f} ± {gaba['centroid_x'].std():.1f}")
    print(f"  Non-GABA: X = {non_gaba['centroid_x'].mean():.1f} ± {non_gaba['centroid_x'].std():.1f}")
    
    stat, p_x = stats.mannwhitneyu(gaba['centroid_x'], non_gaba['centroid_x'])
    print(f"  Mann-Whitney U = {stat:.1f}, p = {p_x:.2e}")
    
    results['x_distribution'] = {
        'gaba_mean': float(gaba['centroid_x'].mean()),
        'gaba_std': float(gaba['centroid_x'].std()),
        'non_gaba_mean': float(non_gaba['centroid_x'].mean()),
        'non_gaba_std': float(non_gaba['centroid_x'].std()),
        'p_value': float(p_x)
    }
    
    return results


def plot_rostrocaudal_analysis(df, rc_results, save_dir):
    """Create rostro-caudal analysis figure."""
    print("\n[CREATING ROSTRO-CAUDAL FIGURE]")
    
    df_filt = df[df['is_filtered']].copy()
    gaba = df_filt[df_filt['is_gabaergic']]
    non_gaba = df_filt[~df_filt['is_gabaergic']]
    
    fig = plt.figure(figsize=(16, 12))
    
    # === Panel A: X distribution ===
    ax1 = plt.subplot(3, 3, 1)
    
    bins = np.linspace(0, cfg.IMAGE_WIDTH, 60)
    
    ax1.hist(non_gaba['centroid_x'], bins=bins, alpha=0.6,
             color=cfg.COLOR_NON_GABA, label='Non-GABA', density=True)
    ax1.hist(gaba['centroid_x'], bins=bins, alpha=0.6,
             color=cfg.COLOR_GABA, label='GABA', density=True)
    
    # Mark cerebellum
    ax1.axvspan(cfg.CEREBELLUM_X_MIN, cfg.CEREBELLUM_X_MAX,
                alpha=0.2, color='yellow', label='Cerebellum')
    
    ax1.set_xlabel('X Position (pixels)')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Rostro-Caudal Distribution (X-axis)', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add anatomical labels
    ax1.text(120, ax1.get_ylim()[1]*0.9, 'Caudal', fontsize=10,
             ha='center', style='italic')
    ax1.text(600, ax1.get_ylim()[1]*0.9, 'Rostral', fontsize=10,
             ha='center', style='italic')
    
    # === Panel B: Z distribution ===
    ax2 = plt.subplot(3, 3, 2)
    
    bins_z = np.arange(0, cfg.N_PLANES + 5, 5)
    
    ax2.hist(non_gaba['plane'], bins=bins_z, alpha=0.6,
             color=cfg.COLOR_NON_GABA, label='Non-GABA', density=True)
    ax2.hist(gaba['plane'], bins=bins_z, alpha=0.6,
             color=cfg.COLOR_GABA, label='GABA', density=True)
    
    # Mark cerebellum Z-range
    ax2.axvspan(cfg.CEREBELLUM_Z_MIN, cfg.CEREBELLUM_Z_MAX,
                alpha=0.2, color='yellow', label='Cerebellum')
    
    ax2.set_xlabel('Z Plane')
    ax2.set_ylabel('Density')
    ax2.set_title('B. Dorso-Ventral Distribution (Z-axis)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # === Panel C: Statistical comparison ===
    ax3 = plt.subplot(3, 3, 3)
    ax3.axis('off')
    
    text = f"""
SPATIAL DISTRIBUTION TESTS

X Position (Rostro-Caudal):
  GABA: {rc_results['x_distribution']['gaba_mean']:.1f} ± {rc_results['x_distribution']['gaba_std']:.1f}
  Non-GABA: {rc_results['x_distribution']['non_gaba_mean']:.1f} ± {rc_results['x_distribution']['non_gaba_std']:.1f}
  p = {rc_results['x_distribution']['p_value']:.2e}
  
Z Plane (Dorso-Ventral):
  GABA: {rc_results['z_distribution']['gaba_mean']:.1f} ± {rc_results['z_distribution']['gaba_std']:.1f}
  Non-GABA: {rc_results['z_distribution']['non_gaba_mean']:.1f} ± {rc_results['z_distribution']['non_gaba_std']:.1f}
  p = {rc_results['z_distribution']['p_value']:.2e}

{'Significant spatial differences detected!' if rc_results['x_distribution']['p_value'] < 0.05 
 or rc_results['z_distribution']['p_value'] < 0.05 
 else 'No significant spatial bias'}
    """
    
    ax3.text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
             transform=ax3.transAxes, family='monospace')
    
    # === Panel D: 2D density map (GABA) ===
    ax4 = plt.subplot(3, 3, 4)
    
    # Create 2D histogram
    x_bins = np.linspace(0, cfg.IMAGE_WIDTH, 50)
    z_bins = np.linspace(0, cfg.N_PLANES, 50)
    
    H_gaba, xedges, zedges = np.histogram2d(
        gaba['centroid_x'], gaba['plane'],
        bins=[x_bins, z_bins]
    )
    
    # Smooth
    H_gaba_smooth = gaussian_filter(H_gaba.T, sigma=1.5)
    
    im = ax4.imshow(H_gaba_smooth, aspect='auto', cmap='magma',
                    extent=[0, cfg.IMAGE_WIDTH, cfg.N_PLANES, 0],
                    interpolation='bilinear')
    
    # Cerebellum box
    from matplotlib.patches import Rectangle
    cerebellum_box = Rectangle(
        (cfg.CEREBELLUM_X_MIN, cfg.CEREBELLUM_Z_MIN),
        cfg.CEREBELLUM_X_MAX - cfg.CEREBELLUM_X_MIN,
        cfg.CEREBELLUM_Z_MAX - cfg.CEREBELLUM_Z_MIN,
        linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
    )
    ax4.add_patch(cerebellum_box)
    
    ax4.set_xlabel('X Position (pixels)')
    ax4.set_ylabel('Z Plane')
    ax4.set_title('D. GABA Density Map', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Density')
    
    # === Panel E: 2D density map (Non-GABA) ===
    ax5 = plt.subplot(3, 3, 5)
    
    H_non_gaba, _, _ = np.histogram2d(
        non_gaba['centroid_x'], non_gaba['plane'],
        bins=[x_bins, z_bins]
    )
    
    H_non_gaba_smooth = gaussian_filter(H_non_gaba.T, sigma=1.5)
    
    im2 = ax5.imshow(H_non_gaba_smooth, aspect='auto', cmap='viridis',
                     extent=[0, cfg.IMAGE_WIDTH, cfg.N_PLANES, 0],
                     interpolation='bilinear')
    
    cerebellum_box2 = Rectangle(
        (cfg.CEREBELLUM_X_MIN, cfg.CEREBELLUM_Z_MIN),
        cfg.CEREBELLUM_X_MAX - cfg.CEREBELLUM_X_MIN,
        cfg.CEREBELLUM_Z_MAX - cfg.CEREBELLUM_Z_MIN,
        linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
    )
    ax5.add_patch(cerebellum_box2)
    
    ax5.set_xlabel('X Position (pixels)')
    ax5.set_ylabel('Z Plane')
    ax5.set_title('E. Non-GABA Density Map', fontweight='bold')
    plt.colorbar(im2, ax=ax5, label='Density')
    
    # === Panel F: Difference map ===
    ax6 = plt.subplot(3, 3, 6)
    
    # Normalize and compute difference
    H_gaba_norm = H_gaba_smooth / (H_gaba_smooth.max() + 1e-10)
    H_non_gaba_norm = H_non_gaba_smooth / (H_non_gaba_smooth.max() + 1e-10)
    
    difference = H_gaba_norm - H_non_gaba_norm
    
    im3 = ax6.imshow(difference, aspect='auto', cmap='RdBu_r',
                     extent=[0, cfg.IMAGE_WIDTH, cfg.N_PLANES, 0],
                     interpolation='bilinear', vmin=-0.5, vmax=0.5)
    
    cerebellum_box3 = Rectangle(
        (cfg.CEREBELLUM_X_MIN, cfg.CEREBELLUM_Z_MIN),
        cfg.CEREBELLUM_X_MAX - cfg.CEREBELLUM_X_MIN,
        cfg.CEREBELLUM_Z_MAX - cfg.CEREBELLUM_Z_MIN,
        linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--'
    )
    ax6.add_patch(cerebellum_box3)
    
    ax6.set_xlabel('X Position (pixels)')
    ax6.set_ylabel('Z Plane')
    ax6.set_title('F. Enrichment Map\n(Red=GABA, Blue=Non-GABA)', fontweight='bold')
    plt.colorbar(im3, ax=ax6, label='GABA - Non-GABA')
    
    # === Panels G-I: Regional breakdown ===
    regions_data = []
    for region_name in ['Caudal (0-240)', 'Mid-Caudal (240-390)', 
                       'Cerebellum (390-600)', 'Rostral (600-720)']:
        if region_name in rc_results:
            regions_data.append({
                'Region': region_name.split('(')[0].strip(),
                'GABA': rc_results[region_name]['gaba_count'],
                'Non-GABA': rc_results[region_name]['non_gaba_count'],
                'GABA%': rc_results[region_name]['gaba_pct']
            })
    
    df_regions = pd.DataFrame(regions_data)
    
    # Panel G: Stacked bar chart
    ax7 = plt.subplot(3, 3, 7)
    
    x_pos = np.arange(len(df_regions))
    
    ax7.bar(x_pos, df_regions['GABA'], label='GABA',
            color=cfg.COLOR_GABA, alpha=0.8)
    ax7.bar(x_pos, df_regions['Non-GABA'], bottom=df_regions['GABA'],
            label='Non-GABA', color=cfg.COLOR_NON_GABA, alpha=0.8)
    
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(df_regions['Region'], rotation=45, ha='right')
    ax7.set_ylabel('Count')
    ax7.set_title('G. Regional Distribution', fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3, axis='y')
    
    # Panel H: GABA percentage by region
    ax8 = plt.subplot(3, 3, 8)
    
    colors = [cfg.COLOR_GABA if pct > df_regions['GABA%'].mean() 
              else cfg.COLOR_GABA_ALT for pct in df_regions['GABA%']]
    
    ax8.bar(x_pos, df_regions['GABA%'], color=colors, alpha=0.8)
    ax8.axhline(df_regions['GABA%'].mean(), color='red', linestyle='--',
                linewidth=2, label=f"Mean ({df_regions['GABA%'].mean():.1f}%)")
    
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(df_regions['Region'], rotation=45, ha='right')
    ax8.set_ylabel('GABA %')
    ax8.set_title('H. GABA Enrichment by Region', fontweight='bold')
    ax8.legend()
    ax8.grid(alpha=0.3, axis='y')
    
    # Panel I: Summary table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create table
    table_data = []
    for _, row in df_regions.iterrows():
        table_data.append([
            row['Region'],
            f"{row['GABA']:,}",
            f"{row['Non-GABA']:,}",
            f"{row['GABA%']:.1f}%"
        ])
    
    table = ax9.table(cellText=table_data,
                      colLabels=['Region', 'GABA', 'Non-GABA', 'GABA%'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#CCCCCC')
        table[(0, i)].set_text_props(weight='bold')
    
    ax9.set_title('I. Regional Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Rostro-Caudal Distribution: GABA vs Non-GABA Neurons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(save_dir) / "figures" / "rostrocaudal_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path}")


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    """Run spatial analysis pipeline."""
    
    print("="*80)
    print("SPATIAL ANALYSIS - GABA DISTRIBUTION")
    print("="*80)
    print(f"\nOutput: {cfg.OUT_DIR}")
    print(f"Midline: X = {cfg.MIDLINE_X}")
    print(f"Cerebellum: Z={cfg.CEREBELLUM_Z_MIN}-{cfg.CEREBELLUM_Z_MAX}, "
          f"X={cfg.CEREBELLUM_X_MIN}-{cfg.CEREBELLUM_X_MAX}")
    
    # Load data
    df = load_classification_and_coords()
    
    # Laterality analysis
    laterality_results = analyze_laterality(df)
    plot_laterality_analysis(df, laterality_results, cfg.OUT_DIR)
    
    # Rostro-caudal analysis
    rc_results = analyze_rostrocaudal(df)
    plot_rostrocaudal_analysis(df, rc_results, cfg.OUT_DIR)
    
    # Save results
    import json
    
    all_results = {
        'laterality': laterality_results,
        'rostrocaudal': rc_results,
        'midline': cfg.MIDLINE_X,
        'cerebellum_region': {
            'z_min': cfg.CEREBELLUM_Z_MIN,
            'z_max': cfg.CEREBELLUM_Z_MAX,
            'x_min': cfg.CEREBELLUM_X_MIN,
            'x_max': cfg.CEREBELLUM_X_MAX
        }
    }
    
    # Convert numpy types to native Python for JSON
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    all_results = convert_to_native(all_results)
    
    with open(Path(cfg.OUT_DIR) / "results" / "spatial_analysis_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ SPATIAL ANALYSIS COMPLETE")
    print("="*80)
    
    print(f"\n📊 KEY FINDINGS:")
    
    # Laterality
    chi2_sig = laterality_results['chi2_test']['significant']
    print(f"\nLaterality:")
    print(f"  {'SIGNIFICANT' if chi2_sig else 'NO'} hemispheric bias detected")
    if chi2_sig:
        gaba_left = laterality_results['gaba']['left_pct']
        gaba_right = laterality_results['gaba']['right_pct']
        bias = 'RIGHT' if gaba_right > gaba_left else 'LEFT'
        print(f"  GABA neurons prefer {bias} hemisphere")
    
    # Rostro-caudal
    x_sig = rc_results['x_distribution']['p_value'] < 0.05
    if x_sig:
        gaba_x = rc_results['x_distribution']['gaba_mean']
        non_gaba_x = rc_results['x_distribution']['non_gaba_mean']
        bias = 'more rostral' if gaba_x > non_gaba_x else 'more caudal'
        print(f"\nRostro-caudal:")
        print(f"  GABA neurons are {bias} than non-GABA")
    
    print(f"\n📁 Output: {cfg.OUT_DIR}")
    print(f"  • laterality_analysis.png")
    print(f"  • rostrocaudal_analysis.png")
    print(f"  • spatial_analysis_results.json")
    print("="*80)
    
    return df, all_results


if __name__ == "__main__":
    df, results = main()