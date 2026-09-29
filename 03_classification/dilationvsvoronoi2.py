#!/usr/bin/env python3
"""Fix: gera figuras a partir do CSV já guardado, sem recalcular."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 20

BASE = r"C:\Users\OSVALDO\Downloads\results\insitu_FOCUSED_COMPARISON"
CSV_PATH = os.path.join(BASE, "focused_comparison_all_results.csv")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")
print(f"Shells available: {sorted(df['end'].unique())}")

# ── FIX: usar as shells que realmente existem ──
SELECTED_EXTERIOR = sorted(df["end"].unique())  # [1, 3, 7, 10]

def _metrics(TP, FP, FN, TN):
    p = TP / max(TP + FP, 1)
    r = TP / max(TP + FN, 1)
    f = 2*p*r / max(p + r, 1e-8)
    return p, r, f

# Build summary
rows = []
for method in ["voronoi", "dilation"]:
    for ext in SELECTED_EXTERIOR:
        sub = df[(df["method"] == method) & (df["start"] == 0) & (df["end"] == ext)]
        if sub.empty:
            continue
        tp, fp, fn, tn = sub["TP"].sum(), sub["FP"].sum(), sub["FN"].sum(), sub["TN"].sum()
        p, r, f = _metrics(tp, fp, fn, tn)
        rows.append({"method": method.upper(), "ext": ext, "precision": p, "recall": r, "f1": f})

summary = pd.DataFrame(rows)

# ── Bar plot ──
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
metrics = ["precision", "recall", "f1"]
titles = ["Precision", "Recall", "F1 Score"]
x = np.arange(len(SELECTED_EXTERIOR))
width = 0.35

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    v = summary[summary["method"] == "VORONOI"].sort_values("ext")[metric].values
    d = summary[summary["method"] == "DILATION"].sort_values("ext")[metric].values
    
    ax.bar(x - width/2, v, width, label="Voronoi", color="#0173B2", edgecolor="black", linewidth=2)
    ax.bar(x + width/2, d, width, label="Dilation", color="#DE8F05", edgecolor="black", linewidth=2)
    
    for i in range(len(SELECTED_EXTERIOR)):
        ax.text(i - width/2, v[i] + 0.015, f'{v[i]:.3f}', ha='center', fontsize=16, fontweight='bold')
        ax.text(i + width/2, d[i] + 0.015, f'{d[i]:.3f}', ha='center', fontsize=16, fontweight='bold')
        diff = v[i] - d[i]
        ax.text(i, max(v[i], d[i]) + 0.06, f'Δ={diff:+.3f}', ha='center', fontsize=14,
                fontweight='bold', color='green' if diff > 0 else 'red')
    
    ax.set_ylabel(title, fontsize=24, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e} px" for e in SELECTED_EXTERIOR], fontsize=20, fontweight="bold")
    ax.set_xlabel("Exterior distance", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=18, frameon=True)
    ax.tick_params(labelsize=18)

fig.suptitle("Voronoi vs Dilation: Pure Exterior Shells (Slice 110)",
             fontsize=28, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(BASE, "bar_comparison_fixed.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✓ bar_comparison_fixed.png saved")