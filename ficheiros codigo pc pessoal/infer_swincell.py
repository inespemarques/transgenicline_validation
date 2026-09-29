import os, gc, math, warnings
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt

from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

from swincell.utils.utils import load_default_config, load_model, get_random_cmap
from swincell.cellpose_dynamics import compute_masks

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
# usa GPU se existir (como no notebook)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}  (cuda: {torch.cuda.is_available()})")

# --- caminhos (ajusta aos teus) ---
data_dir       = r"D:\User data\InesMarques\Swincell-main\data_root2\train\images"
raw_path       = os.path.join(data_dir, "raw.tif")         # <- volume único que queres segmentar
checkpoint_path= r"D:\User data\InesMarques\SwinCell-main\runs\cpu_train_128\checkpoint_epoch30.pt"
output_dir     = r"D:\User data\InesMarques\SwinCell-main\output20oct"
os.makedirs(output_dir, exist_ok=True)

# ROI como no notebook (podes pôr 128x128x32 se preferires)
infer_ROI      = (256, 256, 32)
overlap        = 0.5
sw_batch_size  = 2

# pós-processamento cellpose-like
cellprob_threshold = 0.4
flow_threshold     = 0.4
min_size_pixels    = 2500    # como no notebook; ajusta ao teu downsample se existir
use_gpu_for_masks  = torch.cuda.is_available()  # notebook usa GPU

# Opcional: modo seguro (processa por blocos no eixo Z no compute_masks)
SAFE_Z_CHUNKS = False   # mete True se a RAM subir demasiado

# FP16 para reduzir memória (opcional, só em GPU moderno)
USE_HALF = torch.cuda.is_available()

# =========================
# 1) Modelo (como no notebook)
# =========================
cfg = load_default_config()
model = load_model(cfg)
ckpt = torch.load(checkpoint_path, map_location="cpu")
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(device).eval()
if USE_HALF:
    model.half()
print("[INFO] modelo carregado ✓")

# =========================
# 2) Ler TIFF directo (evita ITK/LoadImaged a rebentar)
# =========================
try:
    img_np = tifffile.imread(raw_path)
except Exception as e:
    raise RuntimeError(f"Falha a ler {raw_path} com tifffile: {e}")

# img_np shape esperado: (Z, Y, X) ou (Y, X, Z). Napari costuma guardar (Z, Y, X).
if img_np.ndim != 3:
    raise ValueError(f"Esperava volume 3D, mas obtive shape={img_np.shape}")

# NORMALIZAR para 0–1 em float32
img_np = img_np.astype(np.float32)
imin, imax = float(img_np.min()), float(img_np.max())
if imax > imin:
    img_np = (img_np - imin) / (imax - imin)
else:
    img_np = np.zeros_like(img_np, dtype=np.float32)

# tensor [1, 1, Z, Y, X]
img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
if USE_HALF:
    img_t = img_t.half()
img_t = img_t.to(device, non_blocking=True)
print(f"[INFO] tensor entrada: {tuple(img_t.shape)}")

# =========================
# 3) Inferência deslizante (igual notebook)
# =========================
model_inferer = partial(
    sliding_window_inference,
    roi_size=infer_ROI,
    sw_batch_size=sw_batch_size,
    predictor=model,
    overlap=overlap,
    mode="gaussian"
)

post_sigmoid = Activations(sigmoid=True)
post_pred    = AsDiscrete(argmax=False, threshold=0.5)

with torch.no_grad():
    if USE_HALF:
        # autocast reduz memória de ativação
        with torch.cuda.amp.autocast():
            logits = model_inferer(img_t)
    else:
        logits = model_inferer(img_t)

# logits: [1, 4, Z, Y, X]
logits_np = np.squeeze(logits.detach().float().cpu().numpy())  # (4, Z, Y, X)
del logits; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# igual ao notebook: sigmoid + threshold no canal 0
logits_np[0] = post_pred(post_sigmoid(torch.from_numpy(logits_np[0])).float()).numpy()

# Transpor p/ compute_masks: (C, X, Y, Z)
# Notebook faz: logits_out_transposed = np.transpose(logits_out, (0,3,2,1))
# partindo de (C, Z, Y, X) → (C, X, Y, Z)
logits_T = np.transpose(logits_np, (0, 3, 2, 1))  # (4, X, Y, Z)
flows_T  = logits_T[1:4, :, :, :]                 # (3, X, Y, Z)
cell_T   = logits_T[0, :, :, :]                   # (X, Y, Z)

print(f"[INFO] logits shape (C,Z,Y,X)={tuple(logits_np.shape)}  → transposto (C,X,Y,Z)={tuple(logits_T.shape)}")

# =========================
# 4) compute_masks (igual notebook) com opção de segurança em blocos Z
# =========================
def compute_masks_chunked(flows_xyz, cellprob_xyz, z_chunk=64, **kw):
    """
    Divide o volume no eixo Z em blocos e recombina (bordas com overlap 4 slices).
    Saída final é id-consistente por concatenação (não funde IDs entre blocos).
    """
    X, Y, Z = cellprob_xyz.shape
    out = np.zeros((Z, Y, X), dtype=np.uint16)
    start = 0
    overlap = 4
    next_id = 1
    while start < Z:
        end = min(Z, start + z_chunk)
        s0 = max(0, start - overlap)
        e0 = min(Z, end + overlap)
        # corta bloco expandido
        cprob_blk = cellprob_xyz[:, :, s0:e0]
        flows_blk = flows_xyz[:, :, :, s0:e0]  # (3, X, Y, z_blk)

        masks_blk, _ = compute_masks(
            flows_blk[[2, 1, 0], :, :, :],  # notebook usa [3,2,1]; aqui flows_xyz é (X,Y,Z) por canal
            cprob_blk,
            cellprob_threshold=kw.get("cellprob_threshold", 0.4),
            flow_threshold=kw.get("flow_threshold", 0.4),
            do_3D=True,
            min_size=kw.get("min_size", 2500),
            use_gpu=kw.get("use_gpu", True),
        )

        # remover as bordas do bloco expandido
        inner_start = start - s0
        inner_end   = inner_start + (end - start)
        masks_inner = masks_blk[inner_start:inner_end].copy()  # (z_inner, Y, X)

        # relabel para IDs não colidirem
        if masks_inner.max() > 0:
            masks_inner[masks_inner > 0] += (next_id - 1)
            next_id = int(masks_inner.max()) + 1

        out[start:end] = masks_inner
        print(f"[INFO] chunk Z [{start}:{end}] → labels até {next_id-1}")
        start = end

    return out  # (Z, Y, X)

# ——— Caminho 1 (igual notebook): volume inteiro
if not SAFE_Z_CHUNKS:
    masks_recon, _ = compute_masks(
        flows_T[[2, 1, 0], :, :, :],  # (= [3,2,1] em índice 1-based)
        cell_T,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        do_3D=True,
        min_size=min_size_pixels,
        use_gpu=use_gpu_for_masks
    )  # (Z, Y, X)

# ——— Caminho 2 (seguro em memória): por blocos de Z
else:
    # define chunk Z ~64/96 conforme memória
    z_chunk = 64 if logits_np.shape[1] >= 96 else 48
    masks_recon = compute_masks_chunked(
        flows_xyz=flows_T, cellprob_xyz=cell_T,
        z_chunk=z_chunk,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size_pixels,
        use_gpu=use_gpu_for_masks
    )

# salvar
out_mask_path = os.path.join(output_dir, "pred_mask.tif")
tifffile.imwrite(out_mask_path, masks_recon.astype(np.uint16))
print(f"[INFO] máscara salva: {out_mask_path}")

# =========================
# 5) Visualização (igual notebook)
# =========================
cmap = get_random_cmap(30)
Z = masks_recon.shape[0]
zmid = Z // 2

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img_np[zmid], cmap="gray");                 axes[0, 0].set_title("Raw")
axes[0, 1].imshow(logits_np[0, zmid], cmap="inferno");        axes[0, 1].set_title("Cell prob (sigmoid+thr)")
flow_rgb = np.moveaxis(logits_np[1:4, :, :, zmid], 0, -1)     # (Y,X,3)
axes[1, 0].imshow(flow_rgb);                                  axes[1, 0].set_title("Predicted flows")
axes[1, 1].imshow(masks_recon[zmid].T, cmap=cmap);            axes[1, 1].set_title("Predicted Masks")
for a in axes.ravel(): a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_visualization.png"), dpi=200)
plt.close()

print("[INFO] Inferência finalizada ✓")
