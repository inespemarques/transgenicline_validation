import os
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt

from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks


# ===========================================================
# 1) Caminhos
# ===========================================================
MODEL_PATH = r"D:\User data\InesMarques\Swincell\model_epoch_049.pt"
INPUT_TIF  = r"C:\Users\ABBE User\Downloads\fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"
OUTPUT_DIR = r"C:\Users\ABBE User\Downloads\swincell_output25625664_ov0.75"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# 2) Carregar imagem sem MONAI
# ===========================================================
raw_img = tifffile.imread(INPUT_TIF)   # shape (Z,Y,X)
assert raw_img.ndim == 3
Z, Y, X = raw_img.shape
print("[INFO] Loaded TIFF:", raw_img.shape)


# ===========================================================
# 3) Preparar modelo SwinCell
# ===========================================================
args = load_default_config()
args.roi_x = 256
args.roi_y = 256
args.roi_z = 64

model = load_model(args).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()


# ===========================================================
# 4) Converter imagem manualmente em tensor MONAI
# ===========================================================
# Normalizar para [0,1] como no treino
img_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())

# Colocar no formato (1,1,Z,Y,X)
img_tensor = torch.tensor(img_norm, dtype=torch.float32)[None, None].to(device)


# ===========================================================
# 5) Sliding window inference sem MONAI transforms
# ===========================================================
model_inferer = partial(
    sliding_window_inference,
    roi_size = (256, 256, 64),
    sw_batch_size=2,
    predictor=model,
    overlap= 0.75,
    mode="gaussian"
)

post_sig = Activations(sigmoid=True)
post_thr = AsDiscrete(threshold=0.5)

with torch.no_grad():
    logits = model_inferer(img_tensor)            # (1,4,Z,Y,X)
    logits_np = np.squeeze(logits.cpu().numpy())  # (4,Z,Y,X)

    cellprob = post_thr(post_sig(logits_np[0]))
    flows    = logits_np[1:4]


# Guardar outputs
tifffile.imwrite(os.path.join(OUTPUT_DIR, "cellprob.tif"), cellprob.astype(np.float32))
tifffile.imwrite(os.path.join(OUTPUT_DIR, "flow_x.tif"), flows[0].astype(np.float32))
tifffile.imwrite(os.path.join(OUTPUT_DIR, "flow_y.tif"), flows[1].astype(np.float32))
tifffile.imwrite(os.path.join(OUTPUT_DIR, "flow_z.tif"), flows[2].astype(np.float32))


# ===========================================================
# 6) compute_masks
# ===========================================================
logits_T = np.transpose(logits_np, (0,3,2,1))  # (4,X,Y,Z)
flows_xyz = logits_T[[3,2,1]]
cellprob_xyz = logits_T[0]

masks, _ = compute_masks(
    flows_xyz, cellprob_xyz,
    cellprob_threshold=-5,
    flow_threshold=0.4,
    do_3D=True,
    min_size=2500,
    use_gpu=True
)

masks = np.squeeze(masks)
masks_zyx = np.transpose(masks, (2,1,0))

tifffile.imwrite(os.path.join(OUTPUT_DIR, "masks_prediction.tif"), masks_zyx.astype(np.uint16))


# ===========================================================
# 7) Mostrar patches segmentados
# ===========================================================
slice_id = Z // 2

plt.figure(figsize=(12, 4))

plt.subplot(1,3,1)
plt.imshow(raw_img[slice_id], cmap="gray")
plt.title("Raw")

plt.subplot(1,3,2)
plt.imshow(cellprob[slice_id], cmap="viridis")
plt.title("Cell Probability")

plt.subplot(1,3,3)
plt.imshow(masks_zyx[slice_id], cmap="nipy_spectral")
plt.title("Mask")

plt.show()



