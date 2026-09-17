# swin_inference_correct_orientation.py — Visualização no plano XY (slice em Z)
import os, argparse, numpy as np, torch, tifffile, matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer
from monai.transforms import Activations, AsDiscrete
from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks

def get_random_cmap(n, seed=0):
    import matplotlib.colors as mcolors
    rng = np.random.RandomState(seed)
    colors = rng.rand(n, 3)
    return mcolors.ListedColormap(np.vstack([[0,0,0], colors]))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--in_tif", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--infer_roi", type=int, nargs=3, default=[256,256,48])
ap.add_argument("--overlap", type=float, default=0.5)
ap.add_argument("--sw_batch_size", type=int, default=2)
ap.add_argument("--prob_ch", type=int, default=0)
ap.add_argument("--cellprob_thr", type=float, default=0.4)
ap.add_argument("--flow_thr", type=float, default=0.4)
ap.add_argument("--min_size", type=int, default=2500)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# 1) Modelo (CPU)
cfg = load_default_config(); cfg.model="swin"
model = load_model(cfg).to("cpu"); model.eval()
ckpt = torch.load(os.path.abspath(args.ckpt), map_location="cpu")
state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
model.load_state_dict(state)
print(f"[Modelo] Carregado de {args.ckpt}", flush=True)

# 2) Ler TIFF 3D (assumindo formato Z,Y,X padrão de microscopia)
vol = tifffile.imread(os.path.abspath(args.in_tif)).astype("float32")
print(f"[Input] Shape original: {vol.shape} (assumindo Z,Y,X)", flush=True)

vmin, vmax = float(vol.min()), float(vol.max())
vol = (vol - vmin) / max(1e-8, (vmax - vmin))
x = torch.from_numpy(vol[None, None, ...])  # (1,1,Z,Y,X)

# 3) Inferência com barra de progresso
inferer = SlidingWindowInferer(
    roi_size=tuple(args.infer_roi),
    overlap=args.overlap,
    sw_batch_size=args.sw_batch_size,
    mode="gaussian",
    progress=True,
)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)

print("[Inferência] A processar...", flush=True)
with torch.no_grad():
    logits = inferer(x, model)  # (1,C,Z,Y,X)

# 4) Pós-processamento
logits_np = np.squeeze(logits.detach().cpu().numpy())  # (C,Z,Y,X)
logits_np[args.prob_ch] = post_pred(post_sigmoid(torch.from_numpy(logits_np[args.prob_ch]))).numpy()

# Para compute_masks, precisa de (C,X,Y,Z) segundo o código original
logits_T = np.transpose(logits_np, (0,3,2,1))  # (C,X,Y,Z)

print("[Cellpose] A computar máscaras...", flush=True)
masks, _ = compute_masks(
    logits_T[[3,2,1]],          # dZ,dY,dX (flows)
    logits_T[args.prob_ch],     # cellprob
    cellprob_threshold=args.cellprob_thr,
    flow_threshold=args.flow_thr,
    do_3D=True,
    min_size=args.min_size,
    use_gpu=False,
)
# masks sai em (X,Y,Z), então transpor de volta para (Z,Y,X)
masks_zyx = np.transpose(masks, (2,1,0))

# 5) Guardar TIFFs em orientação microscópica (Z,Y,X)
prob_zyx = 1.0 / (1.0 + np.exp(-logits_np[args.prob_ch]))  # sigmoid, já em (Z,Y,X)
tifffile.imwrite(os.path.join(args.out_dir, "prob.tif"), prob_zyx.astype("float32"), metadata={'axes': 'ZYX'})
tifffile.imwrite(os.path.join(args.out_dir, "labels_cellpose.tif"), masks_zyx.astype("uint32"), metadata={'axes': 'ZYX'})
print(f"[Output] prob.tif e labels_cellpose.tif salvos (Z,Y,X)", flush=True)

# 6) Visualização: SLICE NO MEIO DE Z para ver plano XY
Z_dim = vol.shape[0]
slice_z = Z_dim // 2  # meio da stack Z

# Extrair slices XY (no meio de Z)
img_slice_xy = vol[slice_z, :, :]                  # (Y, X)
prob_slice_xy = prob_zyx[slice_z, :, :]            # (Y, X)
masks_slice_xy = masks_zyx[slice_z, :, :]          # (Y, X)

# Flows no slice Z médio (em Z,Y,X)
flow_dz = logits_np[3, slice_z, :, :]  # (Y, X)
flow_dy = logits_np[2, slice_z, :, :]
flow_dx = logits_np[1, slice_z, :, :]
flow_rgb = np.stack([flow_dx, flow_dy, flow_dz], axis=-1)  # (Y, X, 3)
# Normalizar flows para visualização RGB
flow_rgb = (flow_rgb - flow_rgb.min()) / (flow_rgb.max() - flow_rgb.min() + 1e-8)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].imshow(img_slice_xy, cmap="gray")
axes[0,0].set_title(f"Raw (Z={slice_z})")
axes[0,1].imshow(prob_slice_xy, cmap="gray")
axes[0,1].set_title(f"Cell prob (Z={slice_z})")
axes[1,0].imshow(flow_rgb)
axes[1,0].set_title(f"Predicted flows (Z={slice_z})")
axes[1,1].imshow(masks_slice_xy, cmap=get_random_cmap(int(masks_slice_xy.max()+1)))
axes[1,1].set_title(f"Predicted Masks (Z={slice_z})")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
viz_path = os.path.join(args.out_dir, "prediction_viz_XY.png")
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[Viz] Slice XY (Z={slice_z}) salvo em {viz_path}", flush=True)

# 7) Opcional: MIP (Maximum Intensity Projection) ao longo de Z
mip_xy = np.max(vol, axis=0)            # (Y, X)
mip_prob = np.max(prob_zyx, axis=0)
mip_masks = np.max(masks_zyx, axis=0)   # mostra células de todos os Z sobrepostos

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
axes2[0].imshow(mip_xy, cmap="gray")
axes2[0].set_title("MIP Raw (max Z)")
axes2[1].imshow(mip_prob, cmap="gray")
axes2[1].set_title("MIP Cell prob")
axes2[2].imshow(mip_masks, cmap=get_random_cmap(int(mip_masks.max()+1)))
axes2[2].set_title("MIP Masks")

for ax in axes2:
    ax.axis("off")

plt.tight_layout()
mip_path = os.path.join(args.out_dir, "MIP_XY.png")
plt.savefig(mip_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"[MIP] Projeção máxima XY salva em {mip_path}", flush=True)

print(f"\n[OK] Inferência completa! Outputs:")
print(f"  - prob.tif (Z,Y,X): probabilidade de célula")
print(f"  - labels_cellpose.tif (Z,Y,X): máscaras de instâncias")
print(f"  - prediction_viz_XY.png: slice Z={slice_z} no plano XY")
print(f"  - MIP_XY.png: projeção máxima ao longo de Z\n")
