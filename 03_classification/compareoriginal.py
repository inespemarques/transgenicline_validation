#!/usr/bin/env python3
"""
Search through all percentiles to find which one gives F1=0.849
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\OSVALDO\Downloads\results\03fev\overlays_realGT_otsu_hist_final")

print("="*70)
print("SEARCHING FOR F1=0.849 ACROSS ALL PERCENTILES")
print("="*70)

percentiles = [60, 65, 70, 75, 80, 85, 90, 95, 99]

results = []

for p in percentiles:
    pct_dir = BASE_DIR / f"bright_p{p}"
    csv_path = pct_dir / "confusion_summary.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Check both methods
        for method in ['otsu_global', 'otsu_per_slice']:
            subset = df[df['method'] == method]
            
            if len(subset) > 0:
                # Pool metrics
                tp = subset['TP'].sum()
                fp = subset['FP'].sum()
                fn = subset['FN'].sum()
                tn = subset['TN'].sum()
                
                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f1 = 2 * prec * rec / (prec + rec + 1e-12)
                
                results.append({
                    'percentile': p,
                    'method': method,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'TN': tn
                })
                
                # Check if close to target
                if abs(f1 - 0.849) < 0.01:
                    print(f"\n🎯 FOUND MATCH!")
                    print(f"   Percentile: {p}")
                    print(f"   Method: {method}")
                    print(f"   Precision: {prec:.3f}")
                    print(f"   Recall: {rec:.3f}")
                    print(f"   F1: {f1:.3f}")

# Create summary table
if results:
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("COMPLETE RESULTS TABLE")
    print("="*70)
    print("\nGLOBAL OTSU:")
    global_df = df_results[df_results['method'] == 'otsu_global'].sort_values('f1', ascending=False)
    print(global_df[['percentile', 'precision', 'recall', 'f1']].to_string(index=False))
    
    print("\n\nPER-SLICE OTSU:")
    slice_df = df_results[df_results['method'] == 'otsu_per_slice'].sort_values('f1', ascending=False)
    print(slice_df[['percentile', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # Find best F1
    best = df_results.loc[df_results['f1'].idxmax()]
    print("\n" + "="*70)
    print("BEST F1 SCORE:")
    print("="*70)
    print(f"   Percentile: {best['percentile']}")
    print(f"   Method: {best['method']}")
    print(f"   Precision: {best['precision']:.3f}")
    print(f"   Recall: {best['recall']:.3f}")
    print(f"   F1: {best['f1']:.3f}")
    
    # Save to CSV
    output_csv = BASE_DIR / "all_percentiles_summary.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n✓ Saved summary to: {output_csv}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print(f"""
The values you mentioned (Precision=0.853, Recall=0.856, F1=0.849) 
don't appear in the P85 results.

Possible explanations:
1. You used a DIFFERENT percentile
2. You used DIFFERENT area filters
3. You used DIFFERENT ground truth CSVs
4. You AVERAGED metrics across slices instead of pooling

Please check where those values came from!
    """)