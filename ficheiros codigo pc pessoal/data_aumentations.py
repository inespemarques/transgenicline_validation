import os
import numpy as np
import tifffile as tiff
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    EnsureTyped
)
from tqdm import tqdm

# === Caminhos ===
base_dir = r"D:\User data\InesMarques\SwinCell-main\swincell\data_root"

# Caminhos das imagens originais
raw_path = r"D:\User data\InesMarques\SwinCell-main\swincell\data_root\train\images\fish_raw.tif"
lab_path = r"D:\User data\InesMarques\SwinCell-main\swincell\data_root\train\labels\fish_label.tif"

# Criar estrutura de saída
train_dir_img = os.path.join(base_dir, "train", "images")
train_dir_lab = os.path.join(base_dir, "train", "labels")
val_dir_img = os.path.join(base_dir, "val", "images")
val_dir_lab = os.path.join(base_dir, "val", "labels")

for d in [train_dir_img, train_dir_lab, val_dir_img, val_dir_lab]:
    os.makedirs(d, exist_ok=True)

# === Carregar imagem e máscara ===
img = tiff.imread(raw_path).astype(np.float32)
lab = tiff.imread(lab_path).astype(np.uint8)

# Normalizar a imagem
img = (img - img.min()) / (img.max() - img.min() + 1e-8)

# Adicionar eixo de canal (C, Z, Y, X)
img = img[None, ...]
lab = lab[None, ...]

# === Definir transformações ===
augment = Compose([
    EnsureTyped(keys=["image", "label"]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # flip Z
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # flip Y
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),  # flip X
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
    RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
    RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
])

# === Gerar augmentations ===
num_aug = 10   # número total
val_frac = 0.2
n_val = int(num_aug * val_frac)

print(f"Gerando {num_aug} volumes aumentados...")

for i in tqdm(range(num_aug)):
    data = {"image": img.copy(), "label": lab.copy()}
    aug_data = augment(data)

    img_aug = aug_data["image"].squeeze()  # remove eixo de canal
    lab_aug = aug_data["label"].squeeze()

    # Decide se vai pra treino ou validação
    if i < n_val:
        img_out = os.path.join(val_dir_img, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(val_dir_lab, f"fish_aug{i:02d}.tif")
    else:
        img_out = os.path.join(train_dir_img, f"fish_aug{i:02d}.tif")
        lab_out = os.path.join(train_dir_lab, f"fish_aug{i:02d}.tif")

    tiff.imwrite(img_out, img_aug.astype(np.float32))
    tiff.imwrite(lab_out, lab_aug.astype(np.uint8))

print(f"✅ Aumentação concluída. Dados salvos em {base_dir}")
print("Pronto para treino com SwinCell!")
