# salvar como prepare_data.py e rodar: python prepare_data.py
import tifffile as tiff
import numpy as np
import os

img_path = r"C:\Users\ABBE User\Downloads\fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"
lab_path = r"C:\Users\ABBE User\Downloads\manual_anot.tif"
out_img = r"C:\Users\ABBE User\Downloads\data_root\images\fish_raw.tif"
out_lab = r"C:\Users\ABBE User\Downloads\data_root\labels\fish_label.tif"

os.makedirs(os.path.dirname(out_img), exist_ok=True)
os.makedirs(os.path.dirname(out_lab), exist_ok=True)

img = tiff.imread(img_path)  # shape (Z, Y, X) ou (C, Z, Y, X) — adapte se necessário
lab = tiff.imread(lab_path)

print("img dtype, shape:", img.dtype, img.shape)
print("lab dtype, shape:", lab.dtype, lab.shape)

# If labels are instance masks (non-binary), convert to binary
lab_bin = (lab > 0).astype(np.uint8)

tiff.imwrite(out_img, img.astype(img.dtype))
tiff.imwrite(out_lab, lab_bin)
print("Arquivos salvos em:", out_img, out_lab)
