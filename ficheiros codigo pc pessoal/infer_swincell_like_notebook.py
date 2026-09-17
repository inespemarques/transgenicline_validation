# ===========================================================
# infer_confocal.py — Inference for confocal microscopy model
# ===========================================================

import os
import glob
import torch
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from functools import partial
from monai import data, transforms
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks
from swincell.utils.utils import get_random_cmap

# ===========================================================
# CONFIGURAÇÃO
# ===========================================================
DATA_DIR = r"D:\User data\InesMarques\Swincell-main\data_root2"
MODEL_PATH = r"D:\User data\InesMarques\Swincell\model_epoch_199.pt"
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# ===========================================================
# PARÂMETROS DO MODELO
# ===========================================================
args = load_default_config()
args.dataset = "custom"
args.roi_x, args.roi_y, args.roi_z = 128, 128, 64
args.a_min, args.a_max = 0, 255
args.b_min, args.b_max = 0, 1
args.downsample_factor = 1
args.save_logits = True

# ===========================================================
# CARREGAR MODELO TREINADO
# ===========================================================
model = load_model(args).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"[INFO] Modelo carregado de {MODEL_PATH}")

# ===========================================================
# DEFINIR ROI PARA SLIDING WINDOW
# ===========================================================
infer_ROI = (256, 256, 32)

# ===========================================================
# TRANSFORMAÇÕES DE TESTE
# ===========================================================
test_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image"]),
    transforms.EnsureChannelFirstd(keys=["image"]),
    transforms.ScaleIntensityRanged(
        keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0, b_max=1, clip=True
    ),
    transforms.ToTensord(keys=["image"]),
])

# ===========================================================
# LISTAR IMAGENS DE TESTE
# ===========================================================
test_images = glob.glob(os.path.join(DATA_DIR, "test", "images", "*.tif"))
if not test_images:
    raise FileNotFoundError(f"Nenhum TIFF encontrado em {DATA_DIR}\\test\\images")

test_datalist = [{"image": p} for p in test_images]
test_ds = data.Dataset(data=test_datalist, transform=test_transform)
test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False)

# ===========================================================
# SLIDING WINDOW INFERENCE
# ===========================================================
model_inferer = partial(
    sliding_window_inference,
    roi_size=infer_ROI,
    sw_batch_size=1,
    predictor=model,
    overlap=0.5,
    mode="gaussian"
)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)

# ===========================================================
# LOOP DE INFERÊNCIA
# ===========================================================
with torch.no_grad():
    for idx, batch_data in enumerate(test_loader):
        img_tensor = batch_data["image"].to(device)
        img_name = os.path.basename(test_datalist[idx]["image"]).replace(".tif", "")
        print(f"[INFO] Inferindo {img_name}...")

        # inferência
        logits = model_inferer(img_tensor)
        logits_np = np.squeeze(logits.detach().cpu().numpy())

        # pós-processamento
        logits_np[0] = post_pred(post_sigmoid(logits_np[0]))
        logits_np_transposed = np.transpose(logits_np, (0, 3, 2, 1))  # (C, Z, Y, X)

        # reconstrução de máscaras
        masks_recon, _ = compute_masks(
            logits_np_transposed[[3, 2, 1], :, :, :],
            logits_np_transposed[0, :, :, :],
            cellprob_threshold=-1,
            flow_threshold=0.6,
            do_3D=True,
            min_size=2500 // args.downsample_factor // args.downsample_factor,
            use_gpu=torch.cuda.is_available()
        )

        # guardar resultados
        output_path = os.path.join(OUTPUT_DIR, f"{img_name}_pred.tif")
        tifffile.imwrite(output_path, masks_recon.astype(np.uint16))
        print(f"[SAVE] {output_path}")

        # visualização rápida
        slice2view = masks_recon.shape[0] // 2
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        img = np.squeeze(img_tensor.detach().cpu().numpy())[0]
        flow = logits_np[1:4]

        if img.ndim == 3:
           axes[0, 0].imshow(img[:, :, slice2view], cmap="gray")
        else:
           axes[0, 0].imshow(img, cmap="gray")

        axes[0, 0].set_title("Raw image")

        axes[0, 1].imshow(logits_np[0, :, :, slice2view])
        axes[0, 1].set_title("Cell probability")

        axes[1, 0].imshow(flow[:, :, :, slice2view].transpose(1, 2, 0))
        axes[1, 0].set_title("Predicted flows")

        axes[1, 1].imshow(masks_recon[slice2view].T, cmap=get_random_cmap(30))
        axes[1, 1].set_title("Predicted masks")

        for a in axes.ravel(): a.axis("off")
        plt.tight_layout(); plt.show()

print("\n✅ Inferência concluída com sucesso!")
