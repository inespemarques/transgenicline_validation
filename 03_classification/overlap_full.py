# ================================================================
# FishX — Block-wise DsRed + InSitu classifier (GPU-safe, metrics)
#
#  - DsRed: inside cell mask, P85 slice-wise + %bright + Otsu
#  - InSitu: Voronoi shell (5 px), P90 slice-wise + %bright + Otsu
#
#  For each cell we save:
#    - pct_bright_dsred, pct_bright_insitu
#    - DsRed_positive, InSitu_positive
#    - mean_dsred, integrated_dsred, area_dsred
#    - mean_insitu_shell, integrated_insitu_shell, area_insitu_shell
#
#  Design:
#    * GPU used slice-by-slice (CuPy) → safe for 11 GB VRAM
#    * No overlays, one CSV per block
# ================================================================

import os
import re
import numpy as np
import tifffile as tiff
import pandas as pd
from skimage.filters import threshold_otsu

# ---------------- GPU CHECK ----------------
try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as cp_distance_transform_edt
    GPU = True
    print("✅ Using GPU (CuPy) slice-wise")
except ImportError:
    from scipy.ndimage import distance_transform_edt
    GPU = False
    print("⚠️ CuPy not found → using CPU only")


# ---------------- CONFIG ----------------
BASE = r"D:\User data\InesMarques\202501024\fish7\cleaned_masks"

INSITU_PATH = os.path.join(BASE, "3channelsdownscaledleft541slices_apenasinsitu.tif")
DSRED_PATH  = os.path.join(BASE, "3channelsdownscaledleft541slices_apenasdsred.tif")

OUT_DIR  = os.path.join(BASE, "classification_blocks_full_cleaned")
os.makedirs(OUT_DIR, exist_ok=True)

MASK_FILES = sorted([f for f in os.listdir(BASE) if f.endswith("_cleaned.tif")])

P_DSRED_BRIGHT  = 85
P_INSITU_BRIGHT = 90
RING_PX         = 5


# ---------------- Helpers ----------------
def ensure_3d(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[None, ...]
    return a

def parse_block_z(fname):
    m = re.search(r'_(\d+)a(\d+)', fname)
    if m is None:
        raise ValueError(f"Cannot find '_ZaZ' pattern in {fname}")
    return int(m.group(1)), int(m.group(2))

# ---------------- Voronoi shell (2D) ----------------
def voronoi_shell_2d(labels_2d, radius):
    """
    Create non-overlapping cytoplasmic shell around each labelled object.
    Pixels 0<dist<=radius from the nearest cell get that cell's label.
    """
    if GPU:
        lab = cp.asarray(labels_2d.astype(np.int32))
        dist, inds = cp_distance_transform_edt(lab == 0, return_indices=True)
        lab_nn = lab[inds[0], inds[1]]
        shell_gpu = cp.where((dist > 0) & (dist <= radius) & (lab_nn > 0), lab_nn, 0)
        shell = cp.asnumpy(shell_gpu).astype(np.int32)
        del lab, dist, inds, lab_nn, shell_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return shell
    else:
        from scipy.ndimage import distance_transform_edt
        lab = labels_2d.astype(np.int32)
        dist, inds = distance_transform_edt(lab == 0, return_indices=True)
        lab_nn = lab[inds[0], inds[1]]
        shell = np.where((dist > 0) & (dist <= radius) & (lab_nn > 0), lab_nn, 0)
        return shell.astype(np.int32)


# ---------------- % bright DsRed (slice-wise GPU) ----------------
def compute_pct_bright_dsred(ds_block, lab_block, percentile):
    """
    Returns:
        ids, pct_bright_per_id, mean_intensity_per_id,
        integrated_intensity_per_id, area_per_id
    """
    Z, Y, X = ds_block.shape
    labels = lab_block
    ids = np.unique(labels)
    ids = ids[ids != 0]
    max_label = int(labels.max()) + 1

    total_counts = np.zeros(max_label, dtype=np.float64)
    total_bright = np.zeros(max_label, dtype=np.float64)
    sum_intensity = np.zeros(max_label, dtype=np.float64)

    for z in range(Z):
        img = ds_block[z]
        lab = labels[z]

        if GPU:
            img_gpu = cp.asarray(img, dtype=cp.float32)
            thr = float(cp.percentile(img_gpu, percentile))
            bright_gpu = (img_gpu > thr).astype(cp.float32)
            lab_gpu = cp.asarray(lab, dtype=cp.int32)

            counts_gpu = cp.bincount(lab_gpu.ravel(), minlength=max_label)
            bright_sum_gpu = cp.bincount(
                lab_gpu.ravel(), weights=bright_gpu.ravel(), minlength=max_label
            )
            sum_int_gpu = cp.bincount(
                lab_gpu.ravel(), weights=img_gpu.ravel(), minlength=max_label
            )

            counts = cp.asnumpy(counts_gpu)
            bright_sum = cp.asnumpy(bright_sum_gpu)
            sum_int = cp.asnumpy(sum_int_gpu)

            del img_gpu, bright_gpu, lab_gpu, counts_gpu, bright_sum_gpu, sum_int_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            thr = np.percentile(img, percentile)
            bright = (img > thr).astype(np.float32)

            counts = np.bincount(lab.ravel(), minlength=max_label)
            bright_sum = np.bincount(
                lab.ravel(), weights=bright.ravel(), minlength=max_label
            )
            sum_int = np.bincount(
                lab.ravel(), weights=img.ravel(), minlength=max_label
            )

        total_counts += counts
        total_bright += bright_sum
        sum_intensity += sum_int

    frac_all = total_bright / np.maximum(total_counts, 1.0)
    frac_all[total_counts == 0] = np.nan
    pct = frac_all[ids] * 100.0

    mean_int = sum_intensity / np.maximum(total_counts, 1.0)
    mean_int_ids = mean_int[ids]
    integ_ids = sum_intensity[ids]
    area_ids = total_counts[ids]

    return ids, pct, mean_int_ids, integ_ids, area_ids


# ---------------- % bright InSitu in Voronoi shell ----------------
def compute_pct_bright_insitu(ins_block, lab_block, percentile, ring_px):
    """
    Same as above but using Voronoi shells instead of the original labels.
    """
    Z, Y, X = ins_block.shape
    labels = lab_block
    ids = np.unique(labels)
    ids = ids[ids != 0]
    max_label = int(labels.max()) + 1

    total_counts = np.zeros(max_label, dtype=np.float64)
    total_bright = np.zeros(max_label, dtype=np.float64)
    sum_intensity = np.zeros(max_label, dtype=np.float64)

    for z in range(Z):
        img = ins_block[z]
        lab_z = labels[z]

        shell_z = voronoi_shell_2d(lab_z, ring_px)

        if GPU:
            img_gpu = cp.asarray(img, dtype=cp.float32)
            thr = float(cp.percentile(img_gpu, percentile))
            bright_gpu = (img_gpu > thr).astype(cp.float32)
            shell_gpu = cp.asarray(shell_z, dtype=cp.int32)

            counts_gpu = cp.bincount(shell_gpu.ravel(), minlength=max_label)
            bright_sum_gpu = cp.bincount(
                shell_gpu.ravel(), weights=bright_gpu.ravel(), minlength=max_label
            )
            sum_int_gpu = cp.bincount(
                shell_gpu.ravel(), weights=img_gpu.ravel(), minlength=max_label
            )

            counts = cp.asnumpy(counts_gpu)
            bright_sum = cp.asnumpy(bright_sum_gpu)
            sum_int = cp.asnumpy(sum_int_gpu)

            del img_gpu, bright_gpu, shell_gpu, counts_gpu, bright_sum_gpu, sum_int_gpu
            cp.get_default_memory_pool().free_all_blocks()
        else:
            thr = np.percentile(img, percentile)
            bright = (img > thr).astype(np.float32)

            counts = np.bincount(shell_z.ravel(), minlength=max_label)
            bright_sum = np.bincount(
                shell_z.ravel(), weights=bright.ravel(), minlength=max_label
            )
            sum_int = np.bincount(
                shell_z.ravel(), weights=img.ravel(), minlength=max_label
            )

        total_counts += counts
        total_bright += bright_sum
        sum_intensity += sum_int

    frac_all = total_bright / np.maximum(total_counts, 1.0)
    frac_all[total_counts == 0] = np.nan
    pct = frac_all[ids] * 100.0

    mean_int = sum_intensity / np.maximum(total_counts, 1.0)
    mean_int_ids = mean_int[ids]
    integ_ids = sum_intensity[ids]
    area_ids = total_counts[ids]

    return ids, pct, mean_int_ids, integ_ids, area_ids


# =====================================================================
# MAIN
# =====================================================================
print("\n📂 Loading global InSitu and DsRed TIFFs...")
insitu_full = ensure_3d(tiff.imread(INSITU_PATH))
dsred_full  = ensure_3d(tiff.imread(DSRED_PATH))
print("  InSitu full shape:", insitu_full.shape)
print("  DsRed  full shape:", dsred_full.shape)

from skimage.filters import threshold_otsu

for mask_fname in MASK_FILES:
    print("\n" + "="*70)
    print(f"🧱 Processing block: {mask_fname}")
    mask_path = os.path.join(BASE, mask_fname)

    z0, z1 = parse_block_z(mask_fname)
    print(f"   Z range (global) = {z0} .. {z1}")

    labels_block = ensure_3d(tiff.imread(mask_path).astype(np.int32))
    Zb = labels_block.shape[0]
    print(f"   Labels block shape: {labels_block.shape}")

    dsred_block  = dsred_full[z0:z1+1]
    insitu_block = insitu_full[z0:z1+1]

    Zmin = min(Zb, dsred_block.shape[0], insitu_block.shape[0])
    labels_block = labels_block[:Zmin]
    dsred_block  = dsred_block[:Zmin]
    insitu_block = insitu_block[:Zmin]

    print(f"   Block shapes aligned: DsRed {dsred_block.shape}, InSitu {insitu_block.shape}")

    ids_block = np.unique(labels_block)
    ids_block = ids_block[ids_block != 0]
    print(f"   {len(ids_block)} labels in this block")

    if len(ids_block) == 0:
        print("   ⚠️ No labels in block, skipping.")
        continue

    # ---------------- DsRed metrics ----------------
    print("   🔴 Computing DsRed metrics (P85, %bright, mean, integrated)...")
    ids_ds, pct_ds, mean_ds, integ_ds, area_ds = compute_pct_bright_dsred(
        dsred_block, labels_block, P_DSRED_BRIGHT
    )
    assert np.array_equal(np.sort(ids_ds), np.sort(ids_block))

    valid_ds = ~np.isnan(pct_ds)
    thr_ds = threshold_otsu(pct_ds[valid_ds]) if valid_ds.sum() > 1 else np.nanmean(pct_ds[valid_ds])
    dsred_pos = pct_ds > thr_ds
    print(f"      Otsu DsRed = {thr_ds:.3f}  |  positives = {dsred_pos.sum()} / {len(dsred_pos)}")

    # ---------------- InSitu metrics ----------------
    print("   🟣 Computing InSitu metrics (Voronoi shell, P90)...")
    ids_in, pct_in, mean_in, integ_in, area_in = compute_pct_bright_insitu(
        insitu_block, labels_block, P_INSITU_BRIGHT, RING_PX
    )
    assert np.array_equal(np.sort(ids_in), np.sort(ids_block))

    valid_in = ~np.isnan(pct_in)
    thr_in = threshold_otsu(pct_in[valid_in]) if valid_in.sum() > 1 else np.nanmean(pct_in[valid_in])
    insitu_pos = pct_in > thr_in
    print(f"      Otsu InSitu = {thr_in:.3f}  |  positives = {insitu_pos.sum()} / {len(insitu_pos)}")

    # ---------------- Save CSV ----------------
    df = pd.DataFrame({
        "label": ids_block.astype(int),
        "pct_bright_dsred": pct_ds,
        "pct_bright_insitu": pct_in,
        "DsRed_positive": dsred_pos,
        "InSitu_positive": insitu_pos,
        "mean_dsred": mean_ds,
        "integrated_dsred": integ_ds,
        "area_dsred": area_ds,
        "mean_insitu_shell": mean_in,
        "integrated_insitu_shell": integ_in,
        "area_insitu_shell": area_in,
    })

    block_tag = f"{z0}a{z1}"
    csv_name = f"fish_block_{block_tag}_classification_FULL_cleaned.csv"
    csv_path = os.path.join(OUT_DIR, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"   💾 Saved CSV: {csv_path}")

print("\n✅ ALL DONE — all blocks processed (GPU slice-wise, full metrics)")
