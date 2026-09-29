import os
import sys
import numpy as np
import cupy as cp
import tifffile as tiff
from cupyx.scipy.ndimage import binary_dilation, binary_erosion


# ================================================================
# LOGS IMEDIATOS
# ================================================================
sys.stdout.reconfigure(line_buffering=True)

print("🔧 Inicializando CUDA…", flush=True)
_ = cp.zeros((1,), dtype=cp.uint8)
cp.cuda.Stream.null.synchronize()
print("✔ CUDA pronto.\n", flush=True)


# ================================================================
# PASTAS A PROCESSAR
# ================================================================
ROOT = r"D:\User data\InesMarques\202501024"

FISH_FOLDERS = [
    os.path.join(ROOT, "fish3"),
]

print("🐟 Folders a processar:")
for f in FISH_FOLDERS:
    print("  -", f)
print("\n")


# ================================================================
# FUNÇÃO TURBO PARA PROCESSAR UM TIFF
# ================================================================
def process_tiff(path_in, path_out):

    print(f"\n📥 A carregar {os.path.basename(path_in)} …")
    vol_np = tiff.imread(path_in)
    print(f"   Shape: {vol_np.shape}, dtype: {vol_np.dtype}\n")

    # ------------------------------------------------------------
    # GPU VOLUME FILTER
    # ------------------------------------------------------------
    vol_cp = cp.asarray(vol_np)
    flat = vol_cp.ravel()
    counts = cp.bincount(flat)

    labels = counts.nonzero()[0]
    labels = labels[labels != 0]
    volumes = counts[labels]

    MIN_VOL = 3000
    MAX_VOL = 70000

    valid_mask = (volumes >= MIN_VOL) & (volumes <= MAX_VOL)
    valid_labels_cpu = cp.asnumpy(labels[valid_mask])

    print(f"   Labels totais: {len(labels)}")
    print(f"   Labels mantidos: {len(valid_labels_cpu)}\n")

    # ------------------------------------------------------------
    # PRECOMPUTAR COORDENADAS TURBO
    # ------------------------------------------------------------
    print("🚀 A preparar coordenadas TURBO…")

    z_all, y_all, x_all = np.where(vol_np != 0)
    lbl_all = vol_np[vol_np != 0]

    coords_by_label = {}
    for z, y, x, lbl in zip(z_all, y_all, x_all, lbl_all):
        if lbl not in coords_by_label:
            coords_by_label[lbl] = []
        coords_by_label[lbl].append((z, y, x))

    for lbl in coords_by_label:
        coords_by_label[lbl] = np.array(coords_by_label[lbl], dtype=np.int32)

    print(f"✔ Coordenadas preparadas: {len(coords_by_label)} labels.\n")

    # ------------------------------------------------------------
    # PROCESSAMENTO TURBO (CLOSING 2D + Z SMOOTH)
    # ------------------------------------------------------------
    print("🚀 A iniciar processamento TURBO…\n")

    final = cp.zeros_like(vol_cp)

    struct = cp.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=cp.uint8)

    for lbl in valid_labels_cpu:

        print(f"Label {lbl} processed.")

        coords = coords_by_label.get(lbl)
        if coords is None:
            continue

        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1

        sub = (vol_cp[z0:z1, y0:y1, x0:x1] == lbl)

        Z = sub.shape[0]
        compact = cp.zeros_like(sub)

        # closing 2D slice-by-slice
        for zz in range(Z):
            slc = sub[zz]
            dil = binary_dilation(slc, structure=struct, iterations=1)
            er  = binary_erosion(dil, structure=struct, iterations=1)
            compact[zz] = er

        # smooth Z
        compact[:-1] |= compact[1:]
        compact[1:]  |= compact[:-1]

        final[z0:z1, y0:y1, x0:x1][compact] = lbl

    cp.cuda.Stream.null.synchronize()

    # ------------------------------------------------------------
    # GUARDAR TIFF COM LZW
    # ------------------------------------------------------------
    print("💾 A guardar com compressão LZW…")

    tiff.imwrite(
        path_out,
        cp.asnumpy(final),
        compression="lzw",
        predictor=True,
    )

    print(f"✔ Guardado: {path_out}\n")


# ================================================================
# LOOP GLOBAL — PROCESSAR FISH 5, 6, 7
# ================================================================
for FOLDER in FISH_FOLDERS:

    print("\n=====================================================")
    print(f"▶ A processar pasta: {FOLDER}")
    print("=====================================================\n")

    files = sorted([f for f in os.listdir(FOLDER) if f.endswith("_segmented.tif")])
    if len(files) == 0:
        print("⚠ Nenhum TIFF encontrado.\n")
        continue

    print("📂 TIFFs encontrados:")
    for f in files:
        print("   ", f)
    print()

    # pasta cleaned_masks dentro de cada fish
    OUT = os.path.join(FOLDER, "cleaned_masks")
    os.makedirs(OUT, exist_ok=True)

    # processar cada segmented
    for fname in files:
        path_in = os.path.join(FOLDER, fname)
        base = fname.replace(".tif", "")
        path_out = os.path.join(OUT, f"{base}_cleaned.tif")

        process_tiff(path_in, path_out)

print("\n✨ TODAS AS PASTAS (fish5, fish6, fish7) FORAM PROCESSADAS! ✨")


























