#!/usr/bin/env python3
# train_stardist_fish_gpu.py
#
# Script pronto para Windows (path com espaços tratado) — treina StarDist3D usando os teus ficheiros em:
# C:\Users\ABBE User\Downloads
#
# Requisitos:
#   conda env com tensorflow (GPU build), stardist, csbdeep, tifffile, scikit-image, scipy, sklearn, numpy
# Instalação rápida (exemplo):
#   conda create -n stardist3d python=3.10 -y
#   conda activate stardist3d
#   pip install tensorflow  # GPU build if your env has CUDA/cuDNN set up
#   pip install stardist csbdeep tifffile scikit-image scipy sklearn numpy tqdm
#
# Uso (exemplo):
#   python train_stardist_fish_gpu.py
#
# O script usa por defeito:
#   raw = "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"
#   masks = "Cellpose Masks update5.tif"
#   pasta = r"C:\Users\ABBE User\Downloads"
#
# Ajusta parâmetros (epochs, mixed_precision, train_patch_size) nas variáveis abaixo ou via args.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage import measure
from scipy import ndimage
from sklearn.model_selection import train_test_split

# Force TF/GPU imports after potential env var modifications
import tensorflow as tf
from stardist.models import Config3D, StarDist3D
from csbdeep.utils import normalize, Path as CSPPath

def configure_gpus(enable_mixed_precision=False, gpu_ids=None):
    # Optional: limitar GPUs visíveis (gpu_ids: list of ints or None)
    if gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in gpu_ids)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[WARNING] Nenhuma GPU detectada. O treino correrá em CPU.")
        return False
    print("[INFO] GPUs detectadas:", gpus)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] Memory growth ativada para GPUs.")
    except Exception as e:
        print("[WARNING] Não foi possível ajustar memory growth:", e)
    if enable_mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("[INFO] Mixed precision ativada (mixed_float16).")
        except Exception as e:
            print("[WARNING] Não foi possível ativar mixed precision:", e)
    return True

def load_volume(path):
    print(f"[LOAD] {path}")
    vol = imread(str(path))
    vol = np.asarray(vol)
    # Garantir (z,y,x) ordering: tifffile normalmente devolve (z,y,x) se stack 3D
    if vol.ndim == 2:
        raise RuntimeError(f"Ficheiro {path} parece 2D (esperava 3D).")
    if vol.ndim == 4:
        # Possível caso: (channels,z,y,x) ou (z,y,x,channels) — tentar descobrir
        print("[WARN] Volume tem 4 dims. Tentando reduzir canais...")
        # heurística: se shape[0] é channels (small), ou last is channels
        if vol.shape[0] <= 4:
            vol = vol[0]  # assume canais primeiro
            print("[INFO] Assumi channel-first e usei o canal 0.")
        elif vol.shape[-1] <= 4:
            vol = vol[...,0]
            print("[INFO] Assumi channel-last e usei o canal 0.")
        else:
            raise RuntimeError("Volume 4D com dimensão de canais inesperada. Inspeciona manualmente.")
    return vol

def masks_to_labels_if_needed(masks):
    """
    Se 'masks' for binária (0/1) converte em rótulos com connected components;
    Se já tiver rótulos (valores inteiros >1), só garante dtype int32.
    """
    # detect binary-ish: valores únicos pequenos e incluem 0/1
    unique = np.unique(masks)
    # Note: cellpose pode produzir máscaras com int ids >0 already.
    if set(unique.tolist()).issubset({0,1}):
        print("[INFO] Masks binárias detectadas — convertendo para labels com connected-components (3D).")
        labeled, n = ndimage.label(masks.astype(np.uint8))
        print(f"[INFO] Encontrados {n} objetos após connected-components.")
        return labeled.astype(np.int32)
    else:
        # if values look like labels but maybe floats
        if not np.issubdtype(masks.dtype, np.integer):
            print("[INFO] Masks contém valores não-inteiros -> convertendo para int32 (round).")
            masks = np.rint(masks).astype(np.int32)
        # ensure background is 0
        if masks.min() < 0:
            masks = masks - masks.min()
        print("[INFO] Masks parecem já rotuladas (skipping connected-components). Unique count:", len(unique))
        return masks.astype(np.int32)

def prepare_and_train(raw_path, masks_path, out_model_dir,
                      epochs=200, n_rays=32, grid=(1,2,2), train_patch_size=(32,128,128),
                      batch_size=2, mixed_precision=False, gpu_ids=None):
    # configure GPU
    gpu_available = configure_gpus(enable_mixed_precision=mixed_precision, gpu_ids=gpu_ids)

    # load
    raw = load_volume(raw_path)
    masks = load_volume(masks_path)

    # sometimes raw and masks are swapped in orientation - quick shape check
    print("[INFO] raw.shape =", raw.shape, "masks.shape =", masks.shape)
    if raw.shape != masks.shape:
        # attempt simple transpositions if likely mismatch (e.g., z,y,x vs y,x,z) - be conservative
        print("[WARN] Shapes não coincidem. Vou tentar transpor máscaras para ver se coincide.")
        if masks.T.shape == raw.shape:
            print("[INFO] Encontrada correspondência com masks.T — transpondo masks.")
            masks = masks.T
        else:
            raise RuntimeError(f"Shapes diferentes e não foi encontrada transposição simples. raw {raw.shape}, masks {masks.shape}")

    # Convert masks to labels if necessary
    labels = masks_to_labels_if_needed(masks)

    # Save converted labels for record (in out_model_dir/prep_labels.tif)
    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    prep_labels_path = out_model_dir / "prep_labels_converted.tif"
    print(f"[SAVE] Guardando labels convertidos em {prep_labels_path}")
    imwrite(str(prep_labels_path), labels.astype(np.int32))


    # Normalize raw for training (csbdeep normalize uses percentiles)
    raw_norm = raw.astype(np.float32)
    raw_norm = normalize(raw_norm, 1, 99.8)

    # Wrap into lists (API StarDist aceita listas de volumes)
    X = [raw_norm]
    Y = [labels]

    # If you have only one volume, the model.train still works but will sample patches.
    print("[INFO] Preparando split treino/val (com apenas um volume, val será 0 items — StarDist pode usar validação por patches).")
    # If multiple volumes available you can split; here we do simple split: if >1 volumes, split, else keep as-is.
    if len(X) > 1:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    else:
        X_train, X_val, Y_train, Y_val = X, [], Y, []

    print(f"[INFO] Train volumes: {len(X_train)}, Val volumes: {len(X_val)}")

    # Build config
    cfg = Config3D(
        n_rays = n_rays,
        grid = tuple(grid),
        anisotropy = None,
        train_patch_size = tuple(train_patch_size),
        train_batch_size = batch_size,
        train_steps_per_epoch = 200 if gpu_available else 100,
        train_sample_cache = True
    )

    # instantiate model
    model_name = "stardist_fish_custom"
    basedir = CSPPath(str(out_model_dir))
    model = StarDist3D(cfg, name=model_name, basedir=basedir)

    print("[TRAIN] Iniciando treino — epochs:", epochs, "batch:", batch_size, "patch:", train_patch_size)
    # train; if no validation volumes provided, pass validation_data=None (StarDist will handle patch val)
    validation = (X_val, Y_val) if len(X_val)>0 else None
    model.train(X_train, Y_train, validation_data=validation, epochs=epochs)

    # save final weights
    final_weights = out_model_dir / model_name / "final_weights.h5"
    model.model.save_weights(str(final_weights))
    print("[DONE] Treino terminado. Pesos salvos em:", final_weights)
    print("[DONE] Pasta do modelo:", out_model_dir / model_name)

if __name__ == "__main__":
    # Defaults tuned to the filenames and path you deu
    default_dir = r"C:\Users\ABBE User\Downloads"
    default_raw = "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"
    default_masks = "Cellpose Masks update5.tif"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=default_dir, help="pasta com raw e masks")
    parser.add_argument("--raw", default=default_raw, help="nome do raw .tif")
    parser.add_argument("--masks", default=default_masks, help="nome do masks .tif")
    parser.add_argument("--out", default="./stardist_fish_model", help="onde guardar o modelo e labels convertidos")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_rays", type=int, default=32)
    parser.add_argument("--grid", nargs=3, type=int, default=(1,2,2))
    parser.add_argument("--train_patch_size", nargs=3, type=int, default=(32,128,128))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default=None, help="ex: '0' ou '0,1' (ids CUDA)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    raw_path = data_dir / args.raw
    masks_path = data_dir / args.masks
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()!='']

    if not raw_path.exists():
        print("[ERROR] Raw não encontrado em:", raw_path)
        sys.exit(1)
    if not masks_path.exists():
        print("[ERROR] Masks não encontrado em:", masks_path)
        sys.exit(1)

    prepare_and_train(raw_path, masks_path, args.out,
                      epochs=args.epochs,
                      n_rays=args.n_rays,
                      grid=tuple(args.grid),
                      train_patch_size=tuple(args.train_patch_size),
                      batch_size=args.batch_size,
                      mixed_precision=args.mixed_precision,
                      gpu_ids=gpu_ids)
