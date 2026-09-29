# swin_notebook_single_image_viz.py — inferência estilo notebook + visualização (CPU)
import os, argparse, numpy as np, torch, tifffile, matplotlib.pyplot as plt
from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_tif", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--infer_roi", type=int, nargs=3, default=[256,256,32])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--sw_batch_size", type=int, default=2)
    ap.add_argument("--prob_ch", type=int, default=0)
    ap.add_argument("--cellprob_thr", type=float, default=0.4)
    ap.add_argument("--flow_thr", type=float, default=0.4)
    ap.add_argument("--min_size", type=int, default=2500)
    return ap.parse_args()

def get_random_cmap(n, seed=0):
    import matplotlib.colors as mcolors
    rng = np.random.RandomState(seed)
    colors = rng.rand(n, 3)
    return mcolors.ListedColormap(np.vstack([[0,0,0], colors]))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Modelo (CPU)
    cfg = load_default_config(); cfg.model="swin"
    model = load_model(cfg).to("cpu"); model.eval()
    ckpt = torch.load(os.path.abspath(args.ckpt), map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    model.load_state_dict(state)

    # 2) Ler TIFF 3D
    vol = tifffile.imread(os.path.abspath(args.in_tif)).astype("float32")
    vmin, vmax = float(vol.min()), float(vol.max())
    vol = (vol - vmin) / max(1e-8, (vmax - vmin))
    x = torch.from_numpy(vol[None, None, ...])  # (1,1,Z,Y,X)

    # 3) Inferência como no notebook
    model_inferer = partial(
        sliding_window_inference,
        roi_size=tuple(args.infer_roi),
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.overlap,
        mode="gaussian",
    )
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    with torch.no_grad():
        logits = model_inferer(x)  # (1,C,Z,Y,X)

    # 4) Pós-processamento (igual notebook)
    logits_out = np.squeeze(logits.detach().cpu().numpy())  # (C,Z,Y,X)
    # prob bin para viz
    logits_out[args.prob_ch] = post_pred(post_sigmoid(torch.from_numpy(logits_out[args.prob_ch])).numpy())
    # transpor p/ (C,X,Y,Z) antes de compute_masks
    logits_out_T = np.transpose(logits_out, (0,3,2,1))      # (C,X,Y,Z)

    # 5) Reconstrução de máscaras 3D
    masks, _ = compute_masks(
        logits_out_T[[3,2,1]],                               # dZ,dY,dX
        logits_out_T[args.prob_ch],                          # cellprob
        cellprob_threshold=args.cellprob_thr,
        flow_threshold=args.flow_thr,
        do_3D=True,
        min_size=args.min_size,
        use_gpu=False,
    )

    # 6) Guardar TIFFs
    prob = (1.0/(1.0+np.exp(-np.transpose(logits_out_T[args.prob_ch], (2,1,0))))).astype("float32")
    tifffile.imwrite(os.path.join(args.out_dir, "prob.tif"), prob)
    tifffile.imwrite(os.path.join(args.out_dir, "labels_cellpose.tif"), masks.astype("uint32"))

    # 7) Visualização estilo notebook
    # slice central ao longo de X (como no caderno: slice2view = img_shape[-1]//2)
    img_shape = logits.shape  # (1,C,Z,Y,X)
    slice2view = int(img_shape[-1] // 2)
    img = np.squeeze(x.numpy())                               # (Z,Y,X)
    flow = logits_out[1:4]                                    # (3,Z,Y,X)
    flow_slice = flow[:, :, :, slice2view].transpose(1, 2, 0) # (Y,Z?) -> segue notebook

    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(12,10))
    axes[0,0].imshow(img[:, :, slice2view], cmap="gray");            axes[0,0].set_title("Raw")
    axes[0,1].imshow(logits_out[args.prob_ch, :, :, slice2view], cmap="gray"); axes[0,1].set_title("Cell prob")
    axes[1,0].imshow(flow_slice);                                   axes[1,0].set_title("Predicted flows")
    axes[1,1].imshow(masks[slice2view].T, cmap=get_random_cmap(30)); axes[1,1].set_title("Predicted Masks")
    for i in range(2):
        for j in range(2):
            axes[i,j].axis("off")
    plt.tight_layout()
    viz_path = os.path.join(args.out_dir, "prediction_viz.png")
    plt.savefig(viz_path, dpi=150)
    plt.close(fig)
    print("[OK] Guardado:", viz_path)

if __name__ == "__main__":
    main()

