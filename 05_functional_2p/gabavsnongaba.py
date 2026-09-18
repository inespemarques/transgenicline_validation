import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Configuração para daltónicos (magenta para não-GABA, verde para GABA)
COLORS = {
    False: '#D81B60',  # Magenta vibrante (não-GABA)
    True: '#00897B'     # Verde teal (GABA)
}

# Carregar dados
df = pd.read_csv(r"C:\Users\OSVALDO\Downloads\RIGOROUS_TIMING_ANALYSIS\metrics_baseline_corrected.csv")

print(f"Total ROIs: {len(df)}")
print(f"GABAérgicos: {df['is_gaba'].sum()}")
print(f"Não-GABAérgicos: {(~df['is_gaba']).sum()}")

# ============================================
# ANÁLISE ESTATÍSTICA COMPLETA
# ============================================

metrics = ['baseline_F', 'amplitude_std', 'n_peaks', 'mean_peak_height', 
           'cv', 'peak_height_zscore', 'snr']

results = []

for metric in metrics:
    gaba = df[df['is_gaba'] == True][metric].dropna()
    non_gaba = df[df['is_gaba'] == False][metric].dropna()
    
    # Teste de normalidade
    _, p_norm_gaba = stats.shapiro(gaba.sample(min(5000, len(gaba))) if len(gaba) > 5000 else gaba)
    _, p_norm_non = stats.shapiro(non_gaba.sample(min(5000, len(non_gaba))) if len(non_gaba) > 5000 else non_gaba)
    
    # Escolher teste apropriado
    if p_norm_gaba > 0.05 and p_norm_non > 0.05:
        stat, p_val = ttest_ind(gaba, non_gaba)
        test_used = "t-test"
    else:
        stat, p_val = mannwhitneyu(gaba, non_gaba)
        test_used = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    cohens_d = (gaba.mean() - non_gaba.mean()) / np.sqrt(((len(gaba)-1)*gaba.std()**2 + (len(non_gaba)-1)*non_gaba.std()**2) / (len(gaba)+len(non_gaba)-2))
    
    results.append({
        'Metric': metric,
        'GABA_mean': gaba.mean(),
        'GABA_std': gaba.std(),
        'NonGABA_mean': non_gaba.mean(),
        'NonGABA_std': non_gaba.std(),
        'p_value': p_val,
        'Cohens_d': cohens_d,
        'Test': test_used,
        'Significant': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    })

stats_df = pd.DataFrame(results)
print("\n" + "="*80)
print("RESULTADOS ESTATÍSTICOS")
print("="*80)
print(stats_df.to_string(index=False))
stats_df.to_csv('statistical_results.csv', index=False)

# ============================================
# FIGURAS PARA A TESE
# ============================================

# Configuração geral
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5

# ============================================
# FIGURA 1: Distribuições comparativas (6 painéis)
# ============================================
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metrics_plot = ['baseline_F', 'amplitude_std', 'mean_peak_height', 
                'peak_height_zscore', 'snr', 'cv']
titles = ['Baseline Fluorescence', 'Amplitude Std', 'Mean Peak Height',
          'Peak Height Z-score', 'Signal-to-Noise Ratio', 'Coefficient of Variation']

for idx, (metric, title) in enumerate(zip(metrics_plot, titles)):
    ax = axes[idx]
    
    # Violin plots
    parts = ax.violinplot([df[df['is_gaba'] == False][metric].dropna(),
                           df[df['is_gaba'] == True][metric].dropna()],
                          positions=[0, 1], widths=0.7, showmeans=True, showmedians=True)
    
    # Colorir
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS[bool(i)])
        pc.set_alpha(0.7)
    
    # Estatísticas no gráfico
    row = stats_df[stats_df['Metric'] == metric].iloc[0]
    p_val = row['p_value']
    sig = row['Significant']
    
    y_max = df[metric].max()
    ax.plot([0, 1], [y_max*1.1, y_max*1.1], 'k-', linewidth=1.5)
    ax.text(0.5, y_max*1.15, f'p={p_val:.2e} {sig}', ha='center', fontsize=10, weight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-GABA', 'GABA'], fontsize=11)
    ax.set_ylabel(title, fontsize=11, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Figure1_Distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure1_Distributions.pdf', bbox_inches='tight')
print("\n✓ Figura 1 salva: Figure1_Distributions.png/pdf")

# ============================================
# FIGURA 2: Box plots com pontos individuais
# ============================================
fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (metric, title) in enumerate(zip(metrics_plot, titles)):
    ax = axes[idx]
    
    # Dados
    data_plot = df[['is_gaba', metric]].copy()
    data_plot['Cell Type'] = data_plot['is_gaba'].map({False: 'Non-GABA', True: 'GABA'})
    
    # Box plot
    bp = ax.boxplot([df[df['is_gaba'] == False][metric].dropna(),
                     df[df['is_gaba'] == True][metric].dropna()],
                    positions=[0, 1], widths=0.5, patch_artist=True,
                    boxprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2),
                    medianprops=dict(color='black', linewidth=2.5))
    
    # Colorir boxes
    for patch, is_gaba in zip(bp['boxes'], [False, True]):
        patch.set_facecolor(COLORS[is_gaba])
        patch.set_alpha(0.6)
    
    # Sample de pontos (max 1000 por grupo para não sobrecarregar)
    for is_gaba, pos in zip([False, True], [0, 1]):
        data_subset = df[df['is_gaba'] == is_gaba][metric].dropna()
        if len(data_subset) > 1000:
            data_subset = data_subset.sample(1000, random_state=42)
        
        x = np.random.normal(pos, 0.08, size=len(data_subset))
        ax.scatter(x, data_subset, alpha=0.15, s=8, color=COLORS[is_gaba], edgecolors='none')
    
    # Estatísticas
    row = stats_df[stats_df['Metric'] == metric].iloc[0]
    ax.text(0.5, ax.get_ylim()[1]*0.95, f"p={row['p_value']:.2e} {row['Significant']}", 
            ha='center', fontsize=10, weight='bold', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-GABA', 'GABA'], fontsize=11, weight='bold')
    ax.set_ylabel(title, fontsize=11, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Figure2_BoxPlots.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure2_BoxPlots.pdf', bbox_inches='tight')
print("✓ Figura 2 salva: Figure2_BoxPlots.png/pdf")

# ============================================
# FIGURA 3: Scatter plots com correlações
# ============================================
fig3, axes = plt.subplots(2, 2, figsize=(14, 12))

scatter_pairs = [
    ('snr', 'mean_peak_height', 'SNR', 'Mean Peak Height'),
    ('baseline_F', 'amplitude_std', 'Baseline F', 'Amplitude Std'),
    ('peak_height_zscore', 'cv', 'Peak Height Z-score', 'CV'),
    ('amplitude_std', 'mean_peak_height', 'Amplitude Std', 'Mean Peak Height')
]

for ax, (x_var, y_var, x_label, y_label) in zip(axes.flatten(), scatter_pairs):
    for is_gaba in [False, True]:
        data_subset = df[df['is_gaba'] == is_gaba][[x_var, y_var]].dropna()
        
        # Sample se muito grande
        if len(data_subset) > 5000:
            data_subset = data_subset.sample(5000, random_state=42)
        
        ax.scatter(data_subset[x_var], data_subset[y_var], 
                  alpha=0.3, s=15, color=COLORS[is_gaba],
                  label=f"{'GABA' if is_gaba else 'Non-GABA'} (n={len(data_subset)})",
                  edgecolors='none')
        
        # Linha de regressão
        z = np.polyfit(data_subset[x_var], data_subset[y_var], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data_subset[x_var].min(), data_subset[x_var].max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS[is_gaba], linewidth=2.5, alpha=0.8)
        
        # Correlação
        r, p_val = stats.pearsonr(data_subset[x_var], data_subset[y_var])
        ax.text(0.05, 0.95 - (0.08 * is_gaba), f"{'GABA' if is_gaba else 'Non-GABA'}: r={r:.3f}, p={p_val:.2e}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=COLORS[is_gaba], alpha=0.3))
    
    ax.set_xlabel(x_label, fontsize=11, weight='bold')
    ax.set_ylabel(y_label, fontsize=11, weight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Figure3_Correlations.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure3_Correlations.pdf', bbox_inches='tight')
print("✓ Figura 3 salva: Figure3_Correlations.png/pdf")

# ============================================
# FIGURA 4: Heatmap de correlação por tipo celular
# ============================================
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, is_gaba in enumerate([False, True]):
    data_corr = df[df['is_gaba'] == is_gaba][metrics].corr()
    
    im = axes[idx].imshow(data_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[idx].set_xticks(range(len(metrics)))
    axes[idx].set_yticks(range(len(metrics)))
    axes[idx].set_xticklabels(metrics, rotation=45, ha='right', fontsize=10)
    axes[idx].set_yticklabels(metrics, fontsize=10)
    axes[idx].set_title(f"{'GABA' if is_gaba else 'Non-GABA'} Neurons", 
                        fontsize=13, weight='bold', color=COLORS[is_gaba])
    
    # Valores na matriz
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text = axes[idx].text(j, i, f'{data_corr.iloc[i, j]:.2f}',
                                ha="center", va="center", color="black", fontsize=8)

# Colorbar compartilhada
fig4.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04, label='Correlation')
plt.tight_layout()
plt.savefig('Figure4_CorrelationHeatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure4_CorrelationHeatmap.pdf', bbox_inches='tight')
print("✓ Figura 4 salva: Figure4_CorrelationHeatmap.png/pdf")

# ============================================
# FIGURA 5: Distribuição de n_peaks
# ============================================
fig5, ax = plt.subplots(figsize=(10, 6))

peak_counts_gaba = df[df['is_gaba'] == True]['n_peaks'].value_counts().sort_index()
peak_counts_non = df[df['is_gaba'] == False]['n_peaks'].value_counts().sort_index()

x = np.arange(len(peak_counts_non))
width = 0.35

ax.bar(x - width/2, peak_counts_non.values, width, label='Non-GABA', 
       color=COLORS[False], alpha=0.8, edgecolor='black', linewidth=1.5)
ax.bar(x + width/2, peak_counts_gaba.values, width, label='GABA', 
       color=COLORS[True], alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Number of Peaks', fontsize=13, weight='bold')
ax.set_ylabel('Count', fontsize=13, weight='bold')
ax.set_title('Peak Count Distribution by Cell Type', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(peak_counts_non.index)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Figure5_PeakDistribution.png', dpi=300, bbox_inches='tight')
plt.savefig('Figure5_PeakDistribution.pdf', bbox_inches='tight')
print("✓ Figura 5 salva: Figure5_PeakDistribution.png/pdf")

print("\n" + "="*80)
print("ANÁLISE COMPLETA!")
print("="*80)
print(f"Total de figuras geradas: 5 (PNG + PDF)")
print(f"Arquivo estatístico: statistical_results.csv")
print("\nFiguras recomendadas para a tese:")
print("  - Figura 1 ou 2: Comparações principais (escolhe a que preferires)")
print("  - Figura 3: Relações entre métricas")
print("  - Figura 4: Correlações intra-tipo celular")
print("  - Figura 5: Distribuição de eventos")

plt.show()