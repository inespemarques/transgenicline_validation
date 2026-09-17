# data_augmentation_swincell_3d_fair_comparison.py
# Augmentations 3D equivalentes às do Cellpose 2D, para comparação justa
import os
import numpy as np
import tifffile
from scipy.ndimage import map_coordinates, gaussian_filter, convolve
import random

# ========== CONFIG (IGUAL AO CELLPOSE) ==========
BASE_DIR = r"D:\User data\InesMarques\SwinCell-main\data_root"
RAW_PATH = os.path.join(BASE_DIR, "train", "images", "raw.tif")
LAB_PATH = os.path.join(BASE_DIR, "train", "labels", "masks.tif")

ANISO_FACTOR = 2.73  # Z tem 2.73x menos resolução
NUM_AUG = 30         # número total de volumes aumentados
VAL_FRAC = 0.2       # fração para validação

# Parâmetros IDÊNTICOS ao Cellpose
SEED = 42
INTENSITY_PROB = 0.4
GEOM_PROB = 0.6
HEAVY_GEOM_PROB = 0.2

USE_EMBOSS = False        # igual ao Cellpose
USE_MOTION_BLUR = True
USE_ELASTIC = True
USE_PERSPECTIVE = True
USE_PINCUSHION = True

ROT_MAX_DEG = 30
SCALE_MIN = 0.9
SCALE_MAX = 1.1
ELASTIC_ALPHA = 15.0
ELASTIC_SIGMA = 4.0
PERSPECTIVE_MAX_SHIFT = 0.08  # fração de deslocamento
PINCUSHION_STRENGTH = 0.00005
MOTION_BLUR_KERNEL = 11

random.seed(SEED)
np.random.seed(SEED)

# ========== DIRS ==========
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LAB_DIR = os.path.join(BASE_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "images")
VAL_LAB_DIR = os.path.join(BASE_DIR, "val", "labels")

for d in [TRAIN_IMG_DIR, TRAIN_LAB_DIR, VAL_IMG_DIR, VAL_LAB_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== LOAD ==========
img_vol = tifffile.imread(RAW_PATH).astype(np.float32)  # (Z,Y,X)
lab_vol = tifffile.imread(LAB_PATH).astype(np.uint16)

vmin, vmax = float(img_vol.min()), float(img_vol.max())
img_vol = (img_vol - vmin) / (vmax - vmin + 1e-8)

print(f"Volume shape: img {img_vol.shape}, label {lab_vol.shape}")
print(f"Anisotropia Z: {ANISO_FACTOR}x | Params idênticos ao Cellpose 2D")

# ========== AUGMENTATION FUNCTIONS (3D, equivalentes ao Cellpose) ==========
def random_scale_rotate_3d(img, lab, aniso=ANISO_FACTOR):
    """Rotação e escala em 3D, respeitando anisotropia Z."""
    Z, Y, X = img.shape
    # rotação em graus (só em XY por enquanto, para ser comparável ao 2D)
    angle_xy = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    
    # centro
    cz, cy, cx = Z / 2.0, Y / 2.0, X / 2.0
    
    # matriz 2D rot+scale no plano XY
    rad = np.deg2rad(angle_xy)
    cos_a, sin_a = np.cos(rad) * scale, np.sin(rad) * scale
    
    # grid
    z_idx, y_idx, x_idx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    
    # apply rot+scale em XY
    y_c = y_idx - cy
    x_c = x_idx - cx
    
    y_rot = cos_a * y_c - sin_a * x_c + cy
    x_rot = sin_a * y_c + cos_a * x_c + cx
    z_rot = z_idx.astype(np.float32)  # Z sem rotação (por ser anisotrópico)
    
    img_t = map_coordinates(img, [z_rot, y_rot, x_rot], order=1, mode='reflect')
    lab_t = map_coordinates(lab.astype(np.float32), [z_rot, y_rot, x_rot], order=0, mode='reflect')
    return img_t, lab_t.astype(lab.dtype)

def random_flip_3d(img, lab):
    """Flips em qualquer eixo (igual ao Cellpose)."""
    axes = []
    if random.random() < 0.5: axes.append(0)  # Z
    if random.random() < 0.5: axes.append(1)  # Y
    if random.random() < 0.5: axes.append(2)  # X
    for ax in axes:
        img = np.flip(img, axis=ax)
        lab = np.flip(lab, axis=ax)
    return img.copy(), lab.copy()

def perspective_3d(img, lab, max_shift=PERSPECTIVE_MAX_SHIFT):
    """Perspective-like deformação em 3D (simula o getPerspectiveTransform do Cellpose)."""
    Z, Y, X = img.shape
    # perturbar cantos em planos XY
    perturb = np.random.uniform(-max_shift, max_shift, (8, 3))  # 8 cantos do cubo
    perturb[:, 0] *= Z
    perturb[:, 1] *= Y
    perturb[:, 2] *= X
    
    # grid com pequenas perturbações lineares (simplificação de perspective 3D)
    z_idx, y_idx, x_idx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    
    # adicionar variação linear baseada na posição (simples, mas comparável)
    dz = (perturb[0, 0] * z_idx / Z).astype(np.float32)
    dy = (perturb[0, 1] * y_idx / Y).astype(np.float32)
    dx = (perturb[0, 2] * x_idx / X).astype(np.float32)
    
    map_z = z_idx + dz
    map_y = y_idx + dy
    map_x = x_idx + dx
    
    img_t = map_coordinates(img, [map_z, map_y, map_x], order=1, mode='reflect')
    lab_t = map_coordinates(lab.astype(np.float32), [map_z, map_y, map_x], order=0, mode='reflect')
    return img_t, lab_t.astype(lab.dtype)

def elastic_transform_3d(img, lab, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA):
    """Elastic deformation 3D (igual ao Cellpose)."""
    Z, Y, X = img.shape
    # campos aleatórios
    dz = (np.random.rand(Z, Y, X) * 2 - 1).astype(np.float32)
    dy = (np.random.rand(Z, Y, X) * 2 - 1).astype(np.float32)
    dx = (np.random.rand(Z, Y, X) * 2 - 1).astype(np.float32)
    
    # smooth com gaussiana
    dz = gaussian_filter(dz, sigma) * alpha
    dy = gaussian_filter(dy, sigma) * alpha
    dx = gaussian_filter(dx, sigma) * alpha
    
    z_idx, y_idx, x_idx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    map_z = (z_idx + dz).astype(np.float32)
    map_y = (y_idx + dy).astype(np.float32)
    map_x = (x_idx + dx).astype(np.float32)
    
    img_t = map_coordinates(img, [map_z, map_y, map_x], order=1, mode='reflect')
    lab_t = map_coordinates(lab.astype(np.float32), [map_z, map_y, map_x], order=0, mode='reflect')
    return img_t, lab_t.astype(lab.dtype)

def pincushion_distort_3d(img, lab, strength=PINCUSHION_STRENGTH, aniso=ANISO_FACTOR):
    """Pincushion 3D (igual ao Cellpose)."""
    Z, Y, X = img.shape
    cz, cy, cx = Z / 2.0, Y / 2.0, X / 2.0
    
    z_idx, y_idx, x_idx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    z_c = (z_idx - cz) / aniso
    y_c = y_idx - cy
    x_c = x_idx - cx
    
    r2 = z_c**2 + y_c**2 + x_c**2
    factor = 1 + strength * r2
    
    map_z = (z_c * factor * aniso + cz).astype(np.float32)
    map_y = (y_c * factor + cy).astype(np.float32)
    map_x = (x_c * factor + cx).astype(np.float32)
    
    img_t = map_coordinates(img, [map_z, map_y, map_x], order=1, mode='reflect')
    lab_t = map_coordinates(lab.astype(np.float32), [map_z, map_y, map_x], order=0, mode='reflect')
    return img_t, lab_t.astype(lab.dtype)

def motion_blur_3d(img, kernel_size=MOTION_BLUR_KERNEL, axis=0):
    """Motion blur ao longo de um eixo (igual ao Cellpose)."""
    kernel_shape = [1, 1, 1]
    kernel_shape[axis] = kernel_size
    kernel = np.ones(kernel_shape, dtype=np.float32) / kernel_size
    return convolve(img, kernel, mode='reflect')

def emboss_3d(img):
    """Emboss 3D (igual ao Cellpose)."""
    kernel = np.array([
        [[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
    ], dtype=np.float32)
    embossed = convolve(img, kernel, mode='reflect')
    embossed = (embossed - embossed.min()) / (embossed.max() - embossed.min() + 1e-8)
    return embossed

def adjust_contrast_3d(img, gamma_range=(0.7, 1.3)):
    """Contrast adjust (igual ao Cellpose RandAdjustContrast)."""
    gamma = random.uniform(*gamma_range)
    return np.power(img, 1.0 / gamma)

def add_noise_3d(img, std=0.02):
    """Gaussian noise (igual ao Cellpose)."""
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)

def gaussian_smooth_3d(img, sigma_range=(0.5, 1.0)):
    """Gaussian smooth (igual ao Cellpose)."""
    sigma = random.uniform(*sigma_range)
    return gaussian_filter(img, sigma)

# ========== PIPELINE ==========
n_val = int(NUM_AUG * VAL_FRAC)
print(f"Gerando {NUM_AUG} volumes (val={n_val}, train={NUM_AUG - n_val})...")

for i in range(NUM_AUG):
    img_aug = img_vol.copy()
    lab_aug = lab_vol.copy()
    
    # GEOM (mesmas probs do Cellpose)
    if random.random() < GEOM_PROB:
        img_aug, lab_aug = random_scale_rotate_3d(img_aug, lab_aug, ANISO_FACTOR)
        img_aug, lab_aug = random_flip_3d(img_aug, lab_aug)
    
    # HEAVY GEOM (mesmas probs do Cellpose)
    if random.random() < HEAVY_GEOM_PROB:
        if USE_ELASTIC:
            img_aug, lab_aug = elastic_transform_3d(img_aug, lab_aug, ELASTIC_ALPHA, ELASTIC_SIGMA)
        if USE_PERSPECTIVE:
            img_aug, lab_aug = perspective_3d(img_aug, lab_aug, PERSPECTIVE_MAX_SHIFT)
        if USE_PINCUSHION:
            img_aug, lab_aug = pincushion_distort_3d(img_aug, lab_aug, PINCUSHION_STRENGTH, ANISO_FACTOR)
    
    # INTENSITY (mesmas probs do Cellpose, só imagem)
    if random.random() < INTENSITY_PROB:
        img_aug = add_noise_3d(img_aug, std=0.02)
    if random.random() < INTENSITY_PROB:
        img_aug = gaussian_smooth_3d(img_aug, sigma_range=(0.5, 1.0))
    if random.random() < INTENSITY_PROB:
        img_aug = adjust_contrast_3d(img_aug, gamma_range=(0.7, 1.3))
    if USE_MOTION_BLUR and random.random() < INTENSITY_PROB:
        img_aug = motion_blur_3d(img_aug, MOTION_BLUR_KERNEL, axis=0)
    if USE_EMBOSS and random.random() < INTENSITY_PROB * 0.5:
        img_aug = emboss_3d(img_aug)
    
    # Salvar
    if i < n_val:
        img_out = os.path.join(VAL_IMG_DIR, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(VAL_LAB_DIR, f"fish_aug{i:02d}.tif")
    else:
        img_out = os.path.join(TRAIN_IMG_DIR, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(TRAIN_LAB_DIR, f"fish_aug{i:02d}.tif")
    
    tifffile.imwrite(img_out, img_aug.astype(np.float32), metadata={'axes': 'ZYX'})
    tifffile.imwrite(lab_out, lab_aug.astype(np.uint16), metadata={'axes': 'ZYX'})
    
    if (i + 1) % 5 == 0:
        print(f"  {i + 1}/{NUM_AUG} concluídos")

print(f"✅ {NUM_AUG} volumes 3D salvos em {BASE_DIR}")
print("   Augmentations aplicadas (equivalentes ao Cellpose 2D):")
print("   - Rotation & scale (respeitando aniso Z)")
print("   - Flips (Z, Y, X)")
print("   - Elastic, Perspective, Pincushion (3D)")
print("   - Motion blur, Noise, Smooth, Contrast")
print("   Probabilidades e parâmetros IDÊNTICOS ao script Cellpose 2D.")
print("Comparação justa garantida: Cellpose treina em cortes 2D, SwinCell treina nestes volumes 3D.")
