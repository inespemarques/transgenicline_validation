

import os, re, time, gc
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# GPU CHECK
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    GPU_AVAILABLE = True
    print(f"✓ GPU: {cp.cuda.Device()} ({cp.cuda.Device().mem_info[1]/1e9:.1f}GB)")
except:
    GPU_AVAILABLE = False
    print("✓ CPU only")

# ==================== CONFIG ====================

BASE_IMG = r"C:\Users\ABBE User\Downloads"
BASE_CSV = r"C:\Users\ABBE User\Downloads"
OUT = os.path.join(BASE_CSV, "shell_validation_FINAL2")

# Previous results locations
PREVIOUS_RESULTS = [
    os.path.join(BASE_CSV, "shell_validation_EXTREME", "csv", "results.csv"),
    os.path.join(BASE_CSV, "shell_validation_EXTREME2", "csv", "results.csv"),
]

MASK_INFO = {
    110: {"mask": os.path.join(BASE_IMG, "c2_110a240clahesegmented.tif"), "z": 110, "z_local": 0},
    300: {"mask": os.path.join(BASE_IMG, "c2_220a350clahesegmented.tif"), "z": 300, "z_local": 80},
    450: {"mask": os.path.join(BASE_IMG, "c2_330a460clahesegmented.tif"), "z": 450, "z_local": 120},
}

MIN_VOL, MAX_VOL = 3000, 70000
PERCENTILES = [60, 70, 75, 80, 85, 90, 95]

# Remaining shell configs between -5 and 5
SHELL_CONFIGS = [
    # Inner-biased
    (-5, 1), (-5, 3),
    (-3, 1), 
    # Outer-biased
     (-1, 3), (-1, 5),(-3, 5),
    # Edge cases
    (-5, 0),  (-3, 0), (-1, 0),
    (0, 1), (0, 3), (0, 5),
]

Z_RANGE_LOCAL = 10
N_WORKERS = min(len(SHELL_CONFIGS), cpu_count() - 2)
GPU_BATCH_SIZE = 5

# ==================== GLOBAL DATA ====================
GLOBAL_MASK3D = None
GLOBAL_FLUOR3D = None
GLOBAL_CELL_Z_SPANS = None
GLOBAL_PERCENTILE_THRESHOLDS = None
GLOBAL_POSITIVE_LABELS = None
GLOBAL_REFERENCE_LABELS = None
GLOBAL_SLICE_KEY = None

def init_worker(mask3d, fluor3d, cell_z_spans, percentile_thresholds, 
                positive_labels, reference_labels, slice_key):
    global GLOBAL_MASK3D, GLOBAL_FLUOR3D, GLOBAL_CELL_Z_SPANS
    global GLOBAL_PERCENTILE_THRESHOLDS, GLOBAL_POSITIVE_LABELS
    global GLOBAL_REFERENCE_LABELS, GLOBAL_SLICE_KEY
    
    GLOBAL_MASK3D = mask3d
    GLOBAL_FLUOR3D = fluor3d
    GLOBAL_CELL_Z_SPANS = cell_z_spans
    GLOBAL_PERCENTILE_THRESHOLDS = percentile_thresholds
    GLOBAL_POSITIVE_LABELS = positive_labels
    GLOBAL_REFERENCE_LABELS = reference_labels
    GLOBAL_SLICE_KEY = slice_key

# ==================== CORE FUNCTIONS (abbreviated for space) ====================

def parse_block_z(path):
    m = re.search(r'_(\d+)a(\d+)', os.path.basename(path))
    return int(m.group(1)), int(m.group(2))

def filter_by_volume(mask3d):
    labels, counts = np.unique(mask3d, return_counts=True)
    valid = (labels > 0) & (counts >= MIN_VOL) & (counts <= MAX_VOL)
    lut = np.zeros(labels.max() + 1, dtype=mask3d.dtype)
    lut[labels[valid]] = labels[valid]
    return lut[mask3d]

def create_shell_batch_gpu(mask_batch, start_dist, end_dist):
    if not GPU_AVAILABLE or mask_batch.shape[0] == 0:
        return np.array([create_shell_single_cpu(mask_batch[i], start_dist, end_dist) 
                        for i in range(mask_batch.shape[0])])
    try:
        mask_gpu = cp.asarray(mask_batch, dtype=cp.int32)
        results_gpu = []
        for i in range(mask_gpu.shape[0]):
            label_slice = mask_gpu[i]
            binary = label_slice > 0
            if not cp.any(binary):
                results_gpu.append(cp.zeros_like(label_slice))
                continue
            dist_out = cupy_ndimage.distance_transform_edt(~binary)
            dist_in = cupy_ndimage.distance_transform_edt(binary)
            dist_map = cp.where(binary, -dist_in, dist_out)
            mask = (dist_map >= start_dist) & (dist_map <= end_dist)
            shell = cp.where(mask, label_slice, 0)
            if end_dist > 0:
                outside = mask & (dist_map > 0)
                if cp.any(outside):
                    _, idx = cupy_ndimage.distance_transform_edt(~binary, return_indices=True)
                    shell[outside] = label_slice[idx[0], idx[1]][outside]
            results_gpu.append(shell)
        results = cp.asnumpy(cp.stack(results_gpu))
        del mask_gpu, results_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return results
    except:
        return np.array([create_shell_single_cpu(mask_batch[i], start_dist, end_dist) 
                        for i in range(mask_batch.shape[0])])

def create_shell_single_cpu(label_slice, start_dist, end_dist):
    binary = label_slice > 0
    if not binary.any():
        return np.zeros_like(label_slice)
    dist_out = distance_transform_edt(~binary)
    dist_in = distance_transform_edt(binary)
    dist_map = np.where(binary, -dist_in, dist_out)
    mask = (dist_map >= start_dist) & (dist_map <= end_dist)
    shell = np.where(mask, label_slice, 0)
    if end_dist > 0:
        outside = mask & (dist_map > 0)
        if outside.any():
            _, idx = distance_transform_edt(~binary, return_indices=True)
            shell[outside] = label_slice[idx[0], idx[1]][outside]
    return shell

def generate_shell_on_demand(mask3d, start_dist, end_dist):
    all_shells = []
    for batch_start in range(0, mask3d.shape[0], GPU_BATCH_SIZE):
        batch_end = min(batch_start + GPU_BATCH_SIZE, mask3d.shape[0])
        batch_shells = create_shell_batch_gpu(mask3d[batch_start:batch_end], start_dist, end_dist)
        all_shells.append(batch_shells)
    result = np.concatenate(all_shells, axis=0)
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    return result

def precompute_percentile_thresholds(fluor3d, percentiles):
    thresholds = np.zeros((fluor3d.shape[0], len(percentiles)), dtype=np.float32)
    for z in range(fluor3d.shape[0]):
        for p_idx, percentile in enumerate(percentiles):
            thresholds[z, p_idx] = np.percentile(fluor3d[z], percentile)
    return thresholds

def compute_brightness_VECTORIZED(fluor3d, cell_z_spans, shell3d, percentile_thresholds_all, percentile_idx):
    brightness_pct = {}
    for cell_label, z_list in cell_z_spans.items():
        total_pixels = 0
        bright_pixels = 0
        for z in z_list:
            cell_mask = (shell3d[z] == cell_label)
            if not cell_mask.any():
                continue
            threshold = percentile_thresholds_all[z, percentile_idx]
            shell_fluor = fluor3d[z][cell_mask]
            total_pixels += shell_fluor.size
            bright_pixels += np.sum(shell_fluor > threshold)
        if total_pixels > 0:
            brightness_pct[cell_label] = (bright_pixels / total_pixels) * 100.0
        else:
            brightness_pct[cell_label] = np.nan
    return brightness_pct

def load_csv(folder, sk):
    path = os.path.join(folder, f"Results_insituslice{sk}fish7.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    files = [f for f in os.listdir(folder) if str(sk) in f and 'insitu' in f.lower() and f.endswith('.csv')]
    return pd.read_csv(os.path.join(folder, files[0]))

def load_positive_labels(csv_df, label_slice):
    x_col = 'XM' if 'XM' in csv_df.columns else next((c for c in csv_df.columns if 'X' in c.upper()), None)
    y_col = 'YM' if 'YM' in csv_df.columns else next((c for c in csv_df.columns if 'Y' in c.upper()), None)
    if not x_col or not y_col:
        return set()
    xs = np.clip(csv_df[x_col].astype(float).round().astype(int).values, 0, label_slice.shape[1] - 1)
    ys = np.clip(csv_df[y_col].astype(float).round().astype(int).values, 0, label_slice.shape[0] - 1)
    return {int(label_slice[y, x]) for x, y in zip(xs, ys) if int(label_slice[y, x]) > 0}

def get_cell_z_spans(mask3d, reference_labels):
    cell_z_spans = {label: [] for label in reference_labels}
    for z in range(mask3d.shape[0]):
        for label in np.unique(mask3d[z]):
            if label in cell_z_spans:
                cell_z_spans[label].append(z)
    return {k: v for k, v in cell_z_spans.items() if v}

def compute_metrics(pred, gt, all_labels):
    TP, FP = len(pred & gt), len(pred - gt)
    FN, TN = len(gt - pred), len(all_labels - pred - gt)
    total = len(all_labels)
    if total == 0:
        return None
    acc = (TP + TN) / total
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    mcc_num = (TP * TN) - (FP * FN)
    mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "accuracy": acc, 
            "precision": prec, "recall": rec, "f1": f1, "mcc": mcc}

def process_shell_config_worker(shell_config):
    start_dist, end_dist = shell_config
    shell3d = generate_shell_on_demand(GLOBAL_MASK3D, start_dist, end_dist)
    results = []
    for p_idx, percentile in enumerate(PERCENTILES):
        brightness_dict = compute_brightness_VECTORIZED(
            GLOBAL_FLUOR3D, GLOBAL_CELL_Z_SPANS, shell3d, GLOBAL_PERCENTILE_THRESHOLDS, p_idx)
        valid_brightness = [v for v in brightness_dict.values() if np.isfinite(v)]
        if len(valid_brightness) < 2:
            continue
        try:
            threshold_otsu_global = threshold_otsu(np.array(valid_brightness))
            predicted_positive = {label for label, brightness in brightness_dict.items()
                                if np.isfinite(brightness) and brightness > threshold_otsu_global}
            metrics = compute_metrics(predicted_positive, GLOBAL_POSITIVE_LABELS, GLOBAL_REFERENCE_LABELS)
            if metrics:
                results.append({"slice": GLOBAL_SLICE_KEY, "start": start_dist, "end": end_dist,
                              "width": end_dist - start_dist, "inside": abs(min(start_dist, 0)),
                              "outside": max(end_dist, 0), "percentile": percentile,
                              "threshold_otsu": threshold_otsu_global, **metrics})
        except:
            pass
    del shell3d
    gc.collect()
    return results

def process_single_slice(slice_key, info):
    print(f"\n{'─'*80}\nSLICE {slice_key}\n{'─'*80}")
    z_start, z_end = parse_block_z(info["mask"])
    z_local = info['z_local']
    
    mask_full = filter_by_volume(tiff.imread(info["mask"]))
    fluor_all = tiff.imread(os.path.join(BASE_IMG, "3channelsdownscaledleft541slices_apenasinsitu.tif"))
    fluor_full = fluor_all[z_start:z_end+1]
    
    min_z = min(mask_full.shape[0], fluor_full.shape[0])
    z_local_start = max(0, z_local - Z_RANGE_LOCAL)
    z_local_end = min(min_z, z_local + Z_RANGE_LOCAL + 1)
    
    mask3d = mask_full[z_local_start:z_local_end]
    fluor3d = fluor_full[z_local_start:z_local_end]
    z_local_adjusted = z_local - z_local_start
    
    print(f"  ✓ Shape: {mask3d.shape} | RAM: {(mask3d.nbytes + fluor3d.nbytes)/1e9:.2f}GB")
    
    label_slice = mask3d[z_local_adjusted]
    reference_labels = set(np.unique(label_slice)) - {0}
    csv_df = load_csv(BASE_CSV, slice_key)
    positive_labels = load_positive_labels(csv_df, label_slice)
    
    print(f"  ✓ Labels: {len(reference_labels)} | Positive: {len(positive_labels)}")
    
    cell_z_spans = get_cell_z_spans(mask3d, reference_labels)
    percentile_thresholds = precompute_percentile_thresholds(fluor3d, PERCENTILES)
    
    print(f"  🚀 Processing {len(SHELL_CONFIGS)} configs...")
    
    with Pool(N_WORKERS, initializer=init_worker, 
              initargs=(mask3d, fluor3d, cell_z_spans, percentile_thresholds,
                       positive_labels, reference_labels, slice_key)) as pool:
        results_nested = list(tqdm(pool.imap(process_shell_config_worker, SHELL_CONFIGS),
                                   total=len(SHELL_CONFIGS), ncols=80))
    
    del mask_full, fluor_full, mask3d, fluor3d
    gc.collect()
    return [item for sublist in results_nested for item in sublist]

# ==================== VISUALIZATION ====================

def create_publication_figures(df_all, output_dir):
    print("\n" + "="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)
    
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.3)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    df_all['shell_label'] = df_all.apply(lambda r: f"[{int(r['start']):+d},{int(r['end']):+d}]", axis=1)
    
    # FIGURE 1: Heatmap
    print("📊 Figure 1: F1 Score Heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    pivot_f1 = df_all.pivot_table(values='f1', index='shell_label', columns='percentile', aggfunc='mean')
    pivot_f1 = pivot_f1.loc[pivot_f1.mean(axis=1).sort_values(ascending=False).index[:20]]
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', cbar_kws={'label': 'F1 Score'}, 
                linewidths=0.5, vmin=0.5, vmax=1.0, ax=ax)
    plt.title('Shell Configuration Performance (Top 20)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Percentile Threshold (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Shell [inner, outer] px', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig1_F1_Heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: Performance Metrics
    print("📊 Figure 2: Performance Metrics...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    top_configs = df_all.groupby('shell_label')['f1'].mean().nlargest(8).index
    df_top = df_all[df_all['shell_label'].isin(top_configs)]
    
    for ax, metric, title in zip(axes.flat, 
                                 ['f1', 'precision', 'recall', 'accuracy'],
                                 ['F1 Score', 'Precision', 'Recall', 'Accuracy']):
        sns.boxplot(data=df_top, x='shell_label', y=metric, ax=ax, palette='Set2')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Shell Configuration', fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Top 8 Configurations: Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig2_Metrics_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 3: Shell Geometry
    print("📊 Figure 3: Shell Geometry Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(df_all['width'], df_all['f1'], c=df_all['percentile'], 
                               cmap='viridis', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Shell Width (pixels)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Shell Width vs Performance', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Percentile (%)')
    
    scatter2 = axes[1].scatter(df_all['inside'], df_all['outside'], c=df_all['f1'], 
                               cmap='RdYlGn', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Inside (pixels)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Outside (pixels)', fontsize=12, fontweight='bold')
    axes[1].set_title('Inside vs Outside Extension', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='F1 Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig3_Shell_Geometry.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 4: Percentile Analysis
    print("📊 Figure 4: Percentile Impact...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    perc_stats = df_all.groupby('percentile').agg({'f1': ['mean', 'std'], 
                                                    'precision': 'mean', 'recall': 'mean'})
    perc_stats.columns = ['f1_mean', 'f1_std', 'precision', 'recall']
    perc_stats = perc_stats.reset_index()
    
    axes[0].errorbar(perc_stats['percentile'], perc_stats['f1_mean'], yerr=perc_stats['f1_std'],
                     marker='o', linewidth=2, markersize=10, capsize=5, label='F1')
    axes[0].plot(perc_stats['percentile'], perc_stats['precision'], 
                 marker='s', linewidth=2, markersize=8, label='Precision')
    axes[0].plot(perc_stats['percentile'], perc_stats['recall'], 
                 marker='^', linewidth=2, markersize=8, label='Recall')
    axes[0].set_xlabel('Percentile (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Metrics vs Percentile', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    sns.violinplot(data=df_all, x='percentile', y='f1', ax=axes[1], palette='muted')
    axes[1].set_xlabel('Percentile (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'Fig4_Percentile_Analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # FIGURE 5: Best Configuration
    print("📊 Figure 5: Optimal Configuration...")
    best = df_all.loc[df_all['f1'].idxmax()]
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('OPTIMAL CONFIGURATION FOR IN SITU CLASSIFICATION', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Info panel
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    info = f"""Shell: [{int(best['start']):+d}, {int(best['end']):+d}] px  |  Percentile: {int(best['percentile'])}%

F1: {best['f1']:.4f}  |  Accuracy: {best['accuracy']:.4f}  |  Precision: {best['precision']:.4f}  |  Recall: {best['recall']:.4f}  |  MCC: {best['mcc']:.4f}

TP: {int(best['TP'])}  |  FP: {int(best['FP'])}  |  FN: {int(best['FN'])}  |  TN: {int(best['TN'])}"""
    ax1.text(0.5, 0.5, info, ha='center', va='center', fontsize=13, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), family='monospace')
    
    # Comparison with other configs
    ax2 = fig.add_subplot(gs[1:, 0])
    best_shell = best['shell_label']
    df_comp = df_all[df_all['shell_label'].isin(
        df_all.groupby('shell_label')['f1'].mean().nlargest(5).index)]
    sns.barplot(data=df_comp, x='shell_label', y='f1', ax=ax2, palette='viridis')
    ax2.axhline(best['f1'], color='red', linestyle='--', linewidth=2, label='Best')
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('Top 5 Configurations', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Percentile sensitivity for best shell
    ax3 = fig.add_subplot(gs[1:, 1])
    df_best_shell = df_all[df_all['shell_label'] == best_shell]
    ax3.plot(df_best_shell['percentile'], df_best_shell['f1'], 
             marker='o', linewidth=3, markersize=10, color='darkgreen')
    ax3.axvline(best['percentile'], color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Percentile (%)', fontweight='bold')
    ax3.set_ylabel('F1 Score', fontweight='bold')
    ax3.set_title(f'Sensitivity: {best_shell}', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(fig_dir, 'Fig5_Optimal_Config.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All figures saved to: {fig_dir}")

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("PhD SHELL VALIDATION: COMPLETE ANALYSIS")
    print("="*80)
    
    os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
    
    # STEP 1: Run new configs
    print("\n[STEP 1] Running remaining configurations...")
    all_results = []
    for slice_key, info in MASK_INFO.items():
        slice_results = process_single_slice(slice_key, info)
        all_results.extend(slice_results)
        gc.collect()
    
    df_new = pd.DataFrame(all_results)
    df_new.to_csv(os.path.join(OUT, "csv", "results_new.csv"), index=False)
    print(f"\n✓ New results: {len(df_new)} tests")
    
    # STEP 2: Combine with previous results
    print("\n[STEP 2] Combining all results...")
    dfs = [df_new]
    for prev_path in PREVIOUS_RESULTS:
        if os.path.exists(prev_path):
            dfs.append(pd.read_csv(prev_path))
            print(f"  ✓ Loaded: {prev_path}")
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(os.path.join(OUT, "csv", "results_combined.csv"), index=False)
    print(f"\n✓ Combined: {len(df_all)} total tests")
    
    # STEP 3: Generate figures
    print("\n[STEP 3] Generating publication figures...")
    create_publication_figures(df_all, OUT)
    
    # Summary
    best = df_all.loc[df_all['f1'].idxmax()]
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total tests: {len(df_all)}")
    print(f"Configurations: {df_all['shell_label'].nunique()}")
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   Shell:      {best['shell_label']} px")
    print(f"   Percentile: {best['percentile']}%")
    print(f"   F1 Score:   {best['f1']:.4f}")
    print(f"   Precision:  {best['precision']:.4f}")
    print(f"   Recall:     {best['recall']:.4f}")
    print(f"\n📁 Results saved in: {OUT}")

# ==================== EXECUTION ====================
if __name__ == "__main__":
    main()