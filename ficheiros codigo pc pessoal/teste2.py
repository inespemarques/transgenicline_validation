import os

data_dir = r"D:\User data\InesMarques\Swincell-main\data_root"
images_dir = os.path.join(data_dir, "images")
masks_dir = os.path.join(data_dir, "labels")

images = sorted(os.listdir(images_dir))
masks = sorted(os.listdir(masks_dir))

for img in images:
    if img not in masks:
        print(f"Imagem sem máscara correspondente: {img}")

for msk in masks:
    if msk not in images:
        print(f"Máscara sem imagem correspondente: {msk}")
