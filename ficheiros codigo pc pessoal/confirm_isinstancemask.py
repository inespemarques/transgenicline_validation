import os
import glob
import tifffile
import numpy as np
from natsort import natsorted

# Caminho para as tuas máscaras
DATA_DIR = r"D:\User data\InesMarques\Swincell-main\data_root2\train\labels"

mask_files = natsorted(glob.glob(os.path.join(DATA_DIR, "*.tif")))

if not mask_files:
    raise FileNotFoundError(f"Nenhum ficheiro .tif encontrado em {DATA_DIR}")

binary_count = 0
instance_count = 0
details = []

print(f"[INFO] A verificar {len(mask_files)} máscaras...\n")

for i, path in enumerate(mask_files):
    mask = tifffile.imread(path)
    unique_vals = np.unique(mask)
    
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
        binary_count += 1
        status = "BINÁRIA (0/1)"
    elif len(unique_vals) > 2:
        instance_count += 1
        status = f"INSTANCIADA ({len(unique_vals)-1} células)"
    else:
        status = f"VALORES INVÁLIDOS: {unique_vals}"

    details.append((os.path.basename(path), status))
    print(f"[{i+1:03d}] {os.path.basename(path)} → {status}")

print("\n==============================")
print(f"Total: {len(mask_files)} máscaras")
print(f" - Binárias:   {binary_count}")
print(f" - Instanciadas: {instance_count}")
print("==============================")

# Mostra 5 exemplos aleatórios
print("\nExemplos:")
for name, status in details[:5]:
    print(f"  {name}: {status}")
