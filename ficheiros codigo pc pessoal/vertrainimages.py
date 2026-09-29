# ===========================================================
# read_tif_metadata.py — Leitura de metadados TIFF do treino
# ===========================================================

import os
import glob
import tifffile
import pandas as pd

# Caminho da tua pasta de treino
DATA_DIR = r"D:\User data\InesMarques\Swincell-main\data_root2\train"

# Subpastas padrão
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")

def read_tif_metadata(path):
    """Lê metadados básicos e OME (se existirem) de um TIFF."""
    info = {"filename": os.path.basename(path)}
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            info["shape"] = page.shape
            info["dtype"] = page.dtype
            info["n_pages"] = len(tif.pages)

            # tentar ler voxel size (se for OME-TIFF)
            if tif.ome_metadata:
                from xml.etree import ElementTree as ET
                ome = ET.fromstring(tif.ome_metadata)
                px = ome.find(".//{*}Pixels")
                if px is not None:
                    info["SizeX"] = px.attrib.get("SizeX")
                    info["SizeY"] = px.attrib.get("SizeY")
                    info["SizeZ"] = px.attrib.get("SizeZ")
                    info["PhysicalSizeX"] = px.attrib.get("PhysicalSizeX")
                    info["PhysicalSizeY"] = px.attrib.get("PhysicalSizeY")
                    info["PhysicalSizeZ"] = px.attrib.get("PhysicalSizeZ")
                    info["Unit"] = px.attrib.get("PhysicalSizeXUnit")
            else:
                info["ome_metadata"] = False

    except Exception as e:
        info["error"] = str(e)
    return info


# -----------------------------------------------------------
# Leitura de todas as imagens
# -----------------------------------------------------------
all_tifs = glob.glob(os.path.join(IMG_DIR, "*.tif")) + glob.glob(os.path.join(LBL_DIR, "*.tif"))
print(f"[INFO] Encontrados {len(all_tifs)} ficheiros TIFF.")

metadata_list = [read_tif_metadata(f) for f in all_tifs]
df = pd.DataFrame(metadata_list)

# Mostra resumo
print(df.head())

# Guarda para Excel ou CSV
out_path = os.path.join(DATA_DIR, "tif_metadata_summary.csv")
df.to_csv(out_path, index=False)
print(f"[SAVE] Metadados guardados em: {out_path}")
