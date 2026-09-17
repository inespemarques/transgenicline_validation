# salvar como prepare_data_swin.py e rodar:
# python prepare_data_swin.py

import tifffile as tiff
import numpy as np
import os
import shutil

# --------------------------------------------------------------
# 1. PATHS DE ENTRADA
# --------------------------------------------------------------
img_path = r"C:\Users\ABBE User\Downloads\fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"
lab_path = r"C:\Users\ABBE User\Downloads\manual_anot.tif"

# --------------------------------------------------------------
# 2. PATHS DE SAÍDA (ESTRUTURA SWINCELL)
# --------------------------------------------------------------
out_img = r"C:\Users\ABBE User\Downloads\data_root\images\fish_raw.tif"
out_lab = r"C:\Users\ABBE User\Downloads\data_root\labels\fish_inst.tif"  # instância verdadeira

# Criar pastas se não existirem
os.makedirs(os.path.dirname(out_img), exist_ok=True)
os.makedirs(os.path.dirname(out_lab), exist_ok=True)

# --------------------------------------------------------------
# 3. LER TIFFS
# --------------------------------------------------------------
img = tiff.imread(img_path)
lab = tiff.imread(lab_path)

print("img dtype, shape:", img.dtype, img.shape)
print("lab dtype, shape:", lab.dtype, lab.shape)

# --------------------------------------------------------------
# 4. VERIFICAR SE O LABEL É BINÁRIO OU INSTÂNCIA
# --------------------------------------------------------------
unique_vals = np.unique(lab)
print("\nValores únicos no label:", unique_vals)

if len(unique_vals) == 2 and set(unique_vals) == {0, 1}:
    print("⚠️ O label é BINÁRIO — isto NÃO serve para SwinCell!")
    print("Pare e corrige o label antes de continuar.")
else:
    print("✔ O label contém várias instâncias — formato CORRETO para SwinCell.")

# --------------------------------------------------------------
# 5. COPIAR IMAGEM E LABEL PARA A ESTRUTURA DO DATASET
# --------------------------------------------------------------
print("\nCopiando imagem e label para a pasta do dataset...")

# Copiar imagem RAW
shutil.copy(img_path, out_img)

# Copiar rótulo INSTÂNCIA sem modificar
shutil.copy(lab_path, out_lab)

print("✔ Imagem salva em:", out_img)
print("✔ Label de instância salvo em:", out_lab)

print("\nTudo pronto para treinar no SwinCell! 🤖")
