import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imread
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment

# ============================================================
# 1. PATHS DEFINIDOS POR TI
# ============================================================

DOWNLOADS = r"C:\Users\ABBE User\Downloads"

RAW_PATH  = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif",
)

GT_PATH   = os.path.join(DOWNLOADS, "manual_anot.tif")
SWIN_PATH = os.path.join(DOWNLOADS, "swincell_output", "masks_prediction.tif")
STAR_PATH = os.path.join(DOWNLOADS, "prediction_labels.tif")

NUC_PATH  = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_nuclei_masks.tif",
)

CYTO3_PRE_PATH = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_cyto3_masks.tif",
)

CYTO3_FT_PATH = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_cyto3finetunedmodel.tif",
)

OUTPUT_DIR = "report_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD RAW + MASKS
# ============================================================

print("\n=== Loading RAW and segmentation masks ===")

raw      = imread(RAW_PATH)
gt       = imread(GT_PATH)
swin_pred = imread(SWIN_PATH)
star_pred = imread(STAR_PATH)
nuc_pred  = imread(NUC_PATH)
cyto3_pre = imread(CYTO3_PRE_PATH)
cyto3_ft  = imread(CYTO3_FT_PATH)

print("Loaded all inputs successfully.")


# ============================================================
# 3. FILTER OUT HUGE INSTANCES (>200k voxels)
# ============================================================

def filter_large_instances(mask, max_size=200000):
    new_mask = np.zeros_like(mask)
    for region in regionprops(mask):
        if region.area <= max_size:
            new_mask[mask == region.label] = region.label
    return new_mask

gt       = filter_large_instances(gt)
swin_pred = filter_large_instances(swin_pred)
star_pred = filter_large_instances(star_pred)
nuc_pred  = filter_large_instances(nuc_pred)
cyto3_pre = filter_large_instances(cyto3_pre)
cyto3_ft  = filter_large_instances(cyto3_ft)

print("Filtered out giant incorrect objects.")


# ============================================================
# 4. INSTANCE IoU MATRIX
# ============================================================

def compute_instance_iou(mask_gt, mask_pred):
    gt_ids = np.unique(mask_gt)[1:]
    pred_ids = np.unique(mask_pred)[1:]

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gid in enumerate(gt_ids):
        g = (mask_gt == gid)
        g_area = g.sum()
        for j, pid in enumerate(pred_ids):
            p = (mask_pred == pid)
            p_area = p.sum()
            inter = np.logical_and(g, p).sum()
            union = g_area + p_area - inter
            if union > 0:
                iou_mat[i, j] = inter / union

    return iou_mat, gt_ids, pred_ids


# ============================================================
# 5. COMPUTE AP AND F1
# ============================================================

def compute_AP_F1(mask_gt, mask_pred, thresholds):
    iou_mat, gt_ids, pred_ids = compute_instance_iou(mask_gt, mask_pred)

    APs, F1s = {}, {}

    for thr in thresholds:
        cost = -(iou_mat >= thr).astype(int)
        row_ind, col_ind = linear_sum_assignment(cost)

        TP = sum(iou_mat[row_ind[k], col_ind[k]] >= thr for k in range(len(row_ind)))
        FP = len(pred_ids) - TP
        FN = len(gt_ids) - TP

        APs[thr] = TP / (TP + FP + FN + 1e-8)
        F1s[thr] = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return APs, F1s


# ============================================================
# 6. RUN METRICS FOR ALL MODELS
# ============================================================

models = {
    "SwinCell": swin_pred,
    "StarDist3D": star_pred,
    "Cellpose nuclei ft": nuc_pred,
    "Cellpose cyto3": cyto3_pre,
    "Cellpose cyto3 ft": cyto3_ft
}

thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))

results_AP = {}
results_F1 = {}

print("\n=== Computing metrics for all models ===")

for name, pred in models.items():
    print(f" → {name}")
    APs, F1s = compute_AP_F1(gt, pred, thresholds)
    results_AP[name] = APs
    results_F1[name] = F1s


# ============================================================
# 7. BUILD METRIC TABLE
# ============================================================

rows = []
for name in models.keys():
    AP05 = results_AP[name][0.5]
    AP075 = results_AP[name][0.75]
    AP_range = np.mean([results_AP[name][t] for t in thresholds])
    F1_obj = np.mean([results_F1[name][t] for t in thresholds])

    rows.append([name, AP05, AP075, AP_range, F1_obj])

df = pd.DataFrame(rows, columns=["Model","AP@0.5","AP@0.75","AP@[0.5:0.95]","F1-object"])
df.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)

with open(f"{OUTPUT_DIR}/metrics.tex", "w") as f:
    f.write(df.to_latex(index=False, float_format="%.4f"))

print("Saved CSV + LaTeX metrics.")


# ============================================================
# 8. COCO CURVE
# ============================================================

plt.figure(figsize=(8,5))
for name in models.keys():
    curve = [results_AP[name][t] for t in thresholds]
    plt.plot(thresholds, curve, marker="o", label=name)

plt.xlabel("IoU threshold")
plt.ylabel("AP")
plt.title("COCO-style AP Curve (3D Instance Segmentation)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/AP_curve.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/AP_curve.pdf")
plt.close()

print("Saved AP curve.")


# ============================================================
# 9. HEATMAP
# ============================================================

plt.figure(figsize=(6,4))
sns.heatmap(df.set_index("Model"), annot=True, cmap="viridis", fmt=".3f")
plt.title("Instance Segmentation Metrics Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap.png", dpi=300)
plt.savefig(f"{OUTPUT_DIR}/heatmap.pdf")
plt.close()

print("Saved heatmap.")


print("\n=== ALL DONE ===")
print(f"Results saved in: {OUTPUT_DIR}")
