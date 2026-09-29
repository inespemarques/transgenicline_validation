# data_augmentation_swincell_3d.py
# Augmentation 3D tipo Cellpose, adaptado ao diretório correto:
# C:\Users\ABBE User\Downloads\data_root

import os
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates, gaussian_filter, convolve
import random
import matplotlib.pyplot as plt

# ==========================================================
# --------------------- CONFIG -----------------------------
# ==========================================================

# ❗ Camino fix: usar raw string r"..."
BASE_DIR = r"C:\Users\ABBE User\Downloads\data_root"

RAW_PATH = os.path.join(BASE_DIR, "images", "fish_raw.tif")
LAB_PATH = os.path.join(BASE_DIR, "labels", "fish_inst.tif")

# validação automática
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"ERRO: Imagem não encontrada em:\n{RAW_PATH}")

if not os.path.exists(LAB_PATH):
    raise FileNotFoundError(f"ERRO: Label de instância não encontrada em:\n{LAB_PATH}")

print("✓ Caminhos validados:")
print("  IMG:", RAW_PATH)
print("  LAB:", LAB_PATH)

ANISO_FACTOR = 2.73
NUM_AUG = 30
VAL_FRAC = 0.2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INTENSITY_PROB = 0.4
GEOM_PROB = 0.6
HEAVY_GEOM_PROB = 0.2

# augmentations ativas
USE_MOTION_BLUR = True
USE_ELASTIC = True
USE_PERSPECTIVE = True
USE_PINCUSHION = True
USE_EMBOSS = False      # pedido: NÃO usar emboss

ROT_MAX_DEG = 30
SCALE_MIN = 0.9
SCALE_MAX = 1.1
ELASTIC_ALPHA = 15.0
ELASTIC_SIGMA = 4.0
PERSPECTIVE_MAX_SHIFT = 0.08
PINCUSHION_STRENGTH = 0.00005
MOTION_BLUR_KERNEL = 11

# ==========================================================
# ------------------ DIRECTÓRIOS DE OUTPUT -----------------
# ==========================================================

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LAB_DIR = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "images")
VAL_LAB_DIR = os.path.join(BASE_DIR, "val", "labels")

for d in [TRAIN_IMG_DIR, TRAIN_LAB_DIR, VAL_IMG_DIR, VAL_LAB_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================================
# ------------------------ LOAD ----------------------------
# ==========================================================

img_vol = tifffile.imread(RAW_PATH).astype(np.float32)
lab_vol = tifffile.imread(LAB_PATH).astype(np.uint16)

vmin, vmax = img_vol.min(), img_vol.max()
img_vol = (img_vol - vmin) / (vmax - vmin + 1e-8)

print("\n=== Dataset carregado ===")
print("Imagem:", img_vol.shape)
print("Label:", lab_vol.shape)
print("Instâncias (IDs únicos):", np.unique(lab_vol)[:20], "...")
print("===========================")


# ==========================================================
# ------------- FUNÇÕES DE AUGMENTATION 3D -----------------
# ==========================================================

def random_scale_rotate_3d(img, lab, aniso=ANISO_FACTOR):
    Z, Y, X = img.shape
    angle = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
    scale = random.uniform(SCALE_MIN, SCALE_MAX)

    cz, cy, cx = Z/2, Y/2, X/2
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad)*scale, np.sin(rad)*scale

    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )
    y_c = y_idx - cy
    x_c = x_idx - cx

    y_rot = cos_a*y_c - sin_a*x_c + cy
    x_rot = sin_a*y_c + cos_a*x_c + cx
    z_rot = z_idx.astype(np.float32)

    img2 = map_coordinates(img, [z_rot, y_rot, x_rot], order=1, mode='reflect')
    lab2 = map_coordinates(lab, [z_rot, y_rot, x_rot], order=0, mode='reflect')
    return img2, lab2.astype(np.uint16)


def random_flip_3d(img, lab):
    if random.random() < 0.5: img, lab = np.flip(img,0), np.flip(lab,0)
    if random.random() < 0.5: img, lab = np.flip(img,1), np.flip(lab,1)
    if random.random() < 0.5: img, lab = np.flip(img,2), np.flip(lab,2)
    return img.copy(), lab.copy()


def elastic_transform_3d(img, lab, alpha, sigma):
    Z, Y, X = img.shape
    dz = gaussian_filter((np.random.rand(Z,Y,X)*2-1), sigma)*alpha
    dy = gaussian_filter((np.random.rand(Z,Y,X)*2-1), sigma)*alpha
    dx = gaussian_filter((np.random.rand(Z,Y,X)*2-1), sigma)*alpha

    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(Z),np.arange(Y),np.arange(X),indexing='ij'
    )
    map_z = (z_idx + dz).astype(np.float32)
    map_y = (y_idx + dy).astype(np.float32)
    map_x = (x_idx + dx).astype(np.float32)

    img2 = map_coordinates(img, [map_z,map_y,map_x], order=1, mode='reflect')
    lab2 = map_coordinates(lab, [map_z,map_y,map_x], order=0, mode='reflect')
    return img2, lab2.astype(np.uint16)


def perspective_3d(img, lab, max_shift):
    Z,Y,X = img.shape
    shift = np.random.uniform(-max_shift, max_shift)

    z_idx, y_idx, x_idx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
    )

    map_z = z_idx
    map_y = y_idx + shift * (y_idx - Y/2)
    map_x = x_idx + shift * (x_idx - X/2)

    img2 = map_coordinates(img, [map_z,map_y,map_x], order=1, mode='reflect')
    lab2 = map_coordinates(lab, [map_z,map_y,map_x], order=0, mode='reflect')
    return img2, lab2.astype(np.uint16)


def pincushion_distort_3d(img, lab, strength, aniso):
    Z,Y,X = img.shape
    cz,cy,cx = Z/2, Y/2, X/2

    z_idx,y_idx,x_idx = np.meshgrid(
        np.arange(Z),np.arange(Y),np.arange(X),indexing='ij'
    )
    z_n = (z_idx - cz)/aniso
    y_n = y_idx - cy
    x_n = x_idx - cx

    r2 = z_n**2 + y_n**2 + x_n**2
    k = 1 + strength*r2

    map_z = (z_n*k*aniso + cz).astype(np.float32)
    map_y = (y_n*k + cy).astype(np.float32)
    map_x = (x_n*k + cx).astype(np.float32)

    img2 = map_coordinates(img,[map_z,map_y,map_x],order=1,mode='reflect')
    lab2 = map_coordinates(lab,[map_z,map_y,map_x],order=0,mode='reflect')
    return img2, lab2.astype(np.uint16)


def motion_blur_3d(img, k=MOTION_BLUR_KERNEL):
    kernel = np.ones((k,1,1), np.float32) / k
    return convolve(img, kernel, mode="reflect")


# ==========================================================
#                   PIPELINE DE AUGMENTAÇÃO
# ==========================================================

n_val = int(NUM_AUG * VAL_FRAC)
print(f"\nGerando {NUM_AUG} volumes (train={NUM_AUG-n_val}, val={n_val})...\n")

img_aug_vis = None
lab_aug_vis = None

for i in range(NUM_AUG):
    img_aug = img_vol.copy()
    lab_aug = lab_vol.copy()

    if random.random() < GEOM_PROB:
        img_aug, lab_aug = random_scale_rotate_3d(img_aug, lab_aug)
        img_aug, lab_aug = random_flip_3d(img_aug, lab_aug)

    if random.random() < HEAVY_GEOM_PROB:
        if USE_ELASTIC:
            img_aug, lab_aug = elastic_transform_3d(img_aug, lab_aug, ELASTIC_ALPHA, ELASTIC_SIGMA)
        if USE_PERSPECTIVE:
            img_aug, lab_aug = perspective_3d(img_aug, lab_aug, PERSPECTIVE_MAX_SHIFT)
        if USE_PINCUSHION:
            img_aug, lab_aug = pincushion_distort_3d(img_aug, lab_aug, PINCUSHION_STRENGTH, ANISO_FACTOR)

    if random.random() < INTENSITY_PROB:
        img_aug = gaussian_filter(img_aug, sigma=0.8)
    if random.random() < INTENSITY_PROB:
        img_aug = img_aug + np.random.normal(0, 0.02, img_aug.shape)
    if USE_MOTION_BLUR and random.random() < INTENSITY_PROB:
        img_aug = motion_blur_3d(img_aug)

    # guardar primeira augment para visualização
    if i == 0:
        img_aug_vis = img_aug.copy()
        lab_aug_vis = lab_aug.copy()

    # escolher diretório de saída
    if i < n_val:
        img_out = os.path.join(VAL_IMG_DIR, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(VAL_LAB_DIR, f"fish_aug{i:02d}.tif")
    else:
        img_out = os.path.join(TRAIN_IMG_DIR, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(TRAIN_LAB_DIR, f"fish_aug{i:02d}.tif")

    tifffile.imwrite(img_out, img_aug.astype(np.float32))
    tifffile.imwrite(lab_out, lab_aug.astype(np.uint16))

print("\n✓ Augmentations criadas com sucesso!")
print("Pasta train/ e val/ prontas.")


# ==========================================================
#                   VISUALIZAÇÃO ANTES / DEPOIS
# ==========================================================

mid = img_vol.shape[0] // 2

plt.figure(figsize=(12,8))
plt.suptitle(f"AUGMENTAÇÃO 3D | Slice Z={mid}", fontsize=16)

plt.subplot(2,2,1)
plt.title("Imagem ORIGINAL")
plt.imshow(img_vol[mid], cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Label ORIGINAL")
plt.imshow(lab_vol[mid], cmap='nipy_spectral')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Imagem AUGMENTADA")
plt.imshow(img_aug_vis[mid], cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Label AUGMENTADA")
plt.imshow(lab_aug_vis[mid], cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

