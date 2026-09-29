# segment_with_model.py — Inferência CPU com SlidingWindowInferer (MONAI) e logs detalhados
import os, argparse, warnings, traceback
import numpy as np
import torch, torch.nn.functional as F
import tifffile
from monai.inferers import SlidingWindowInferer
from swincell.utils.utils import load_default_config, load_model

warnings.filterwarnings("ignore")

def ensure_min_size_zyx(t: torch.Tensor, min_sz):
    z, y, x = t.shape[-3], t.shape[-2], t.shape[-1]
    dz = max(0, min_sz[0] - z)
    dy = max(0, min_sz[1] - y)
    dx = max(0, min_sz[2] - x)
    if dz or dy or dx:
        t = F.pad(t, (0, dx, 0, dy, 0, dz))
    return t

def pad_to_multiple_zyx(t: torch.Tensor, multiple=32):
    z, y, x = t.shape[-3], t.shape[-2], t.shape[-1]
    pad_z = (multiple - z % multiple) % multiple
    pad_y = (multiple - y % multiple) % multiple
    pad_x = (multiple - x % multiple) % multiple
    t = F.pad(t, (0, pad_x, 0, pad_y, 0, pad_z))
    return t, (pad_z, pad_y, pad_x)

def unpad_zyx(t: torch.Tensor, pads):
    pad_z, pad_y, pad_x = pads
    Z, Y, X = t.shape[-3], t.shape[-2], t.shape[-1]
    return t[..., :Z - pad_z, :Y - pad_y, :X - pad_x]

def reorder_zyx_if_needed(vol3d: np.ndarray) -> np.ndarray:
    assert vol3d.ndim == 3
    zyx = list(vol3d.shape)
    min_idx = int(np.argmin(zyx))
    if min_idx != 0:
        order = [min_idx] + [i for i in range(3) if i != min_idx]
        vol3d = np.transpose(vol3d, axes=order)
    return vol3d

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Caminho para best.pth")
    ap.add_argument("--in_tif", required=True, help="Imagem TIFF 3D de entrada")
    ap.add_argument("--out_dir", required=True, help="Diretório para salvar saídas")
    ap.add_argument("--prob_ch", type=int, default=0, help="Índice do canal de probabilidade")
    ap.add_argument("--threshold", type=float, default=0.5, help="Limiar para binarização")
    ap.add_argument("--roi_x", type=int, default=192)
    ap.add_argument("--roi_y", type=int, default=192)
    ap.add_argument("--roi_z", type=int, default=64)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--sw_batch_size", type=int, default=1)
    return ap.parse_args()

def main():
    args = parse_args()

    try:
        print("[INFO] Início da inferência", flush=True)
        out_dir_abs = os.path.abspath(args.out_dir)
        in_tif_abs = os.path.abspath(args.in_tif)
        ckpt_abs = os.path.abspath(args.ckpt)
        print(f"[INFO] in_tif = {in_tif_abs}", flush=True)
        print(f"[INFO] ckpt   = {ckpt_abs}", flush=True)
        print(f"[INFO] out_dir= {out_dir_abs}", flush=True)
        os.makedirs(out_dir_abs, exist_ok=True)
        print("[INFO] Diretório de saída ok", flush=True)

        # 1) Modelo igual ao treino
        cfg = load_default_config()
        cfg.model = "swin"
        cfg.roi_x, cfg.roi_y, cfg.roi_z = args.roi_x, args.roi_y, args.roi_z
        model = load_model(cfg).to("cpu")
        model.eval()
        print("[INFO] Modelo criado e em modo eval()", flush=True)

        # 2) Checkpoint no CPU
        ckpt = torch.load(ckpt_abs, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print("[INFO] Checkpoint carregado", flush=True)

        # 3) Ler TIFF e normalizar
        print("[INFO] A ler TIFF ...", flush=True)
        vol = tifffile.imread(in_tif_abs).astype("float32")
        print(f"[INFO] TIFF lido, shape={vol.shape}", flush=True)
        if vol.ndim != 3:
            raise ValueError(f"A imagem deve ser 3D; ndim={vol.ndim}")
        vol = reorder_zyx_if_needed(vol)
        vmin, vmax = float(vol.min()), float(vol.max())
        scale = max(1e-8, (vmax - vmin))
        vol = (vol - vmin) / scale
        print("[INFO] Normalização concluída", flush=True)

        # 4) Preparar tensor (N,C,Z,Y,X) e padding
        x = torch.from_numpy(vol[None, None, ...])
        x = ensure_min_size_zyx(x, (cfg.roi_z, cfg.roi_y, cfg.roi_x))
        x, pads = pad_to_multiple_zyx(x, 32)
        print(f"[INFO] Tensor preparado: shape={tuple(x.shape)} pads={pads}", flush=True)

        # 5) Sliding window inferer (usa ROI do treino)
        inferer = SlidingWindowInferer(
            roi_size=(cfg.roi_z, cfg.roi_y, cfg.roi_x),
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            mode="gaussian",
            progress=True,
        )
        print("[INFO] A inferir com SlidingWindowInferer ...", flush=True)

        # 6) Inferência sem gradientes
        with torch.inference_mode():
            logits = inferer(x, model)  # (1, C_out, Z, Y, X)
            y = torch.sigmoid(logits[:, args.prob_ch:args.prob_ch+1, ...])
        print(f"[INFO] Inferência ok: out shape={tuple(y.shape)}", flush=True)

        # 7) Remover padding e gravar
        y = unpad_zyx(y, pads)
        y_np = y.squeeze(0).squeeze(0).cpu().numpy()  # (Z,Y,X)
        out_prob = os.path.join(out_dir_abs, "prob.tif")
        out_mask = os.path.join(out_dir_abs, "mask.tif")
        print(f"[INFO] A gravar:\n  prob={out_prob}\n  mask={out_mask}", flush=True)
        tifffile.imwrite(out_prob, y_np.astype("float32"))
        tifffile.imwrite(out_mask, (y_np >= args.threshold).astype("uint8") * 255)
        print("[OK] Guardado com sucesso.", flush=True)

    except Exception:
        print("[ERRO] Ocorreu uma exceção durante a inferência:", flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

