import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ====== CONFIGURAÇÃO ======
raw_path  = r"D:\User data\InesMarques\SwinCell-main\data_root2\train\images\raw.tif"
mask_path = r"D:\User data\InesMarques\SwinCell-main\output20oct\pred_mask.tif"

# Alterna para True se quiseres abrir também no napari (se instalado)
USE_NAPARI = False  # True/False

# ====== LER IMAGEM E MÁSCARA ======
img  = tifffile.imread(raw_path)           # esperado (Z, Y, X)
mask = tifffile.imread(mask_path)          # (Z, Y, X) com labels inteiros

if img.ndim != 3 or mask.ndim != 3:
    raise ValueError(f"Esperava 3D (Z,Y,X). Shapes: img={img.shape}, mask={mask.shape}")

# normalizar imagem para 0–1 (apenas para visualização)
img = img.astype(np.float32)
imin, imax = float(img.min()), float(img.max())
img = (img - imin) / (imax - imin) if imax > imin else np.zeros_like(img, dtype=np.float32)

# alinhar shapes se necessário (por ex. máscara com padding diferente)
minZ = min(img.shape[0], mask.shape[0])
minY = min(img.shape[1], mask.shape[1])
minX = min(img.shape[2], mask.shape[2])
img  = img[:minZ, :minY, :minX]
mask = mask[:minZ, :minY, :minX]

Z, Y, X = img.shape
zmid, ymid, xmid = Z // 2, Y // 2, X // 2

# ====== CORTES ORTOGONAIS ======
# XY @ zmid -> (Y,X)
img_xy  = img[zmid, :, :]
mask_xy = mask[zmid, :, :]

# XZ @ ymid -> (Z,X); para mostrar com X horizontal e Z vertical, transpomos para (Z,X) -> (Z,X) ok
# para imshow com o “X” no eixo horizontal e “Z” no vertical, usamos (Z,X) e depois .T se quiseres rodar
img_xz  = img[:, ymid, :]           # (Z, X)
mask_xz = mask[:, ymid, :]          # (Z, X)

# YZ @ xmid -> (Z,Y)
img_yz  = img[:, :, xmid]           # (Z, Y)
mask_yz = mask[:, :, xmid]          # (Z, Y)

# ====== FIGURA MATPLOTLIB (central slices) ======
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
# 1) XY
axes[0,0].imshow(img_xy, cmap="gray")
axes[0,0].set_title(f"Raw XY @ z={zmid}")
axes[1,0].imshow(mask_xy, cmap="gray")  # se preferires cores por label, muda cmap para 'nipy_spectral'
axes[1,0].set_title(f"Mask XY @ z={zmid}")

# 2) XZ (mostramos Z para cima; se quiseres “deitado”, usa .T)
axes[0,1].imshow(img_xz, cmap="gray", aspect=Z/X)
axes[0,1].set_title(f"Raw XZ @ y={ymid}")
axes[1,1].imshow(mask_xz, cmap="nipy_spectral", aspect=Z/X)
axes[1,1].set_title(f"Mask XZ @ y={ymid}")

# 3) YZ
axes[0,2].imshow(img_yz, cmap="gray", aspect=Z/Y)
axes[0,2].set_title(f"Raw YZ @ x={xmid}")
axes[1,2].imshow(mask_yz, cmap="nipy_spectral", aspect=Z/Y)
axes[1,2].set_title(f"Mask YZ @ x={xmid}")

for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# ====== OPCIONAL: VISUALIZAÇÃO INTERATIVA NO NAPARI ======
if USE_NAPARI:
    try:
        import napari  # pip install napari
        # napari: cada layer pode ser 3D (Z,Y,X) e dá para “scrollar” no Z
        viewer = napari.Viewer()
        viewer.add_image(img, name="raw", contrast_limits=[0, 1])
        # mostra máscara como labels (com colormap de labels)
        viewer.add_labels(mask.astype(np.int32), name="mask")
        napari.run()
    except Exception as e:
        print("[WARN] Napari não disponível ou falhou ao iniciar:", e)
