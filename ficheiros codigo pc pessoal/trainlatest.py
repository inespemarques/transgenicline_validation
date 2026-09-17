# ===========================================================
# trainlatest.py — SwinCell adapted for confocal microscopy
# ===========================================================

import os
import glob
import time
import warnings
import argparse
import csv
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch

from natsort import natsorted
from monai import data, transforms
from torch.nn import MSELoss
from monai.losses import DiceLoss

from swincell.utils.utils import load_default_config, load_model
from swincell.utils.data_utils import split_dataset_folder, flow_generationd
from swincell.trainer import save_checkpoint


# ===========================================================
# Dataset customizado — evita patches sem células
# ===========================================================
class NonEmptyPatchDataset(data.Dataset):
    def __init__(self, datalist, transform=None, min_mask_fraction=0.001):
        self.datalist = datalist
        self.transform = transform
        self.min_mask_fraction = min_mask_fraction

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = self.datalist[idx]

        # Aplica transformações com segurança
        try:
            data_i = self.transform(sample) if self.transform else sample
        except MemoryError:
            print("[WARN] MemoryError em transform — retornando patch vazio.")
            dummy = torch.zeros((4, 64, 64, 64), dtype=torch.float32)
            return {"image": dummy[:1], "label": dummy}

        # 🔧 se o transform retornar uma lista (ex: RandSpatialCropSamplesd)
        if isinstance(data_i, list):
            data_i = data_i[0]

        lbl = data_i["label"].numpy()
        mask = lbl[0]  # canal 0 é a máscara binária
        frac = np.mean(mask > 0)

        # rejeita patches sem células
        if frac < self.min_mask_fraction:
            print("got empty masks, returning zeros")
            lbl[:] = 0.0
            data_i["label"] = torch.from_numpy(lbl)

        return data_i


# ===========================================================
# Função principal
# ===========================================================
def main():
    warnings.filterwarnings("ignore")

    # -------------------------------
    # ARGUMENTOS DE LINHA DE COMANDO
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None, help="Caminho para retomar treino de checkpoint anterior.")
    cmd_args = parser.parse_args()
    resume_path = cmd_args.resume_from

    # -------------------------------
    # CONFIGURAÇÕES
    # -------------------------------
    torch.set_num_threads(os.cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU ativa: {torch.cuda.get_device_name(0)}")

    DATA_DIR = r"D:\User data\InesMarques\Swincell-main\data_root2"
    LOGDIR   = r"D:\User data\InesMarques\Swincell-main\runs\confocal_train"
    os.makedirs(LOGDIR, exist_ok=True)

    args = load_default_config()
    args.data_dir = DATA_DIR
    args.dataset  = "custom"
    args.roi_x, args.roi_y, args.roi_z = 128, 128, 64
    args.max_epochs = 300
    args.a_min, args.a_max = 0, 255
    args.save_logits = False
    args.downsample_factor = 1
    if not hasattr(args, "optim_lr"):
        args.optim_lr = 1e-4

    # -------------------------------
    # LISTAR FICHEIROS
    # -------------------------------
    image_files = natsorted(glob.glob(os.path.join(DATA_DIR, "train", "images", "*.tif")))
    mask_files  = natsorted(glob.glob(os.path.join(DATA_DIR, "train", "labels", "*.tif")))

    if not image_files or not mask_files:
        raise FileNotFoundError(f"Nenhum ficheiro .tif encontrado em {DATA_DIR}")

    train_list, val_list = split_dataset_folder(image_files, mask_files, split_ratios=[0.8, 0.2])
    print(f"[INFO] train pares={len(train_list)} | val pares={len(val_list)}")

    # -------------------------------
    # DETECTAR SHAPE AUTOMATICAMENTE
    # -------------------------------
    sample_img = tifffile.imread(image_files[0])
    if sample_img.ndim != 3:
        raise ValueError(f"Esperava TIFF 3D, obtive shape={sample_img.shape}")
    img_shape = tuple(int(s) for s in sample_img.shape)
    img_reshape = tuple(int(s // args.downsample_factor) for s in img_shape)
    print(f"[INFO] img_shape={img_shape}  → img_reshape={img_reshape}")

    # -------------------------------
    # TRANSFORMS
    # -------------------------------
    base_train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Resized(keys=["image", "label"], spatial_size=img_reshape),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0, b_max=1, clip=True
        ),
        transforms.RandSpatialCropSamplesd(
            keys=["image", "label"],
            roi_size=[args.roi_x, args.roi_y, args.roi_z],
            num_samples=2,
            random_center=True,
            random_size=False,
        ),
        flow_generationd(keys=["label"]),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    val_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Resized(keys=["image", "label"], spatial_size=img_reshape),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0, b_max=1, clip=True
        ),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    # -------------------------------
    # DATASETS
    # -------------------------------
    num_workers = max(0, os.cpu_count() - 4)
    train_ds = NonEmptyPatchDataset(train_list, transform=base_train_transform)
    val_ds   = NonEmptyPatchDataset(val_list,   transform=val_transform)

    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader   = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

    # -------------------------------
    # MODELO
    # -------------------------------
    model = load_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=1e-5)

    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] A retomar treino a partir de checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[INFO] Retomado na epoch {start_epoch}")
    else:
        print("[INFO] Treino novo iniciado (sem checkpoint).")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[INFO] Modelo: {type(model).__name__} | Params={n_params:.2f}M")

    loss_mse_fn  = MSELoss(reduction="mean")
    loss_dice_fn = DiceLoss(to_onehot_y=False, sigmoid=True)

    # -------------------------------
    # CSV PARA GUARDAR AS LOSSES
    # -------------------------------
    csv_path = os.path.join(LOGDIR, "train_losses.csv")
    if start_epoch == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "time_sec"])

    # -------------------------------
    # LOOP DE TREINO
    # -------------------------------
    print("\n[INFO] Início do treino...\n")

    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        t0 = time.time()
        tsum = 0.0
        nb = 0

        for idx, batch_data in enumerate(train_loader):
            img = batch_data["image"].to(device)
            lbl = batch_data["label"].to(device)

            pred = model(img)

            tgt_mask, tgt_flow = lbl[:, 0:1], lbl[:, 1:4]
            pred_mask, pred_flow = pred[:, 0:1], pred[:, 1:4]

            loss_dice = loss_dice_fn(pred_mask, tgt_mask)
            loss_mse = loss_mse_fn(pred_flow, tgt_flow)
            loss = loss_dice + loss_mse

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            tsum += float(loss.item())
            nb += 1

            print(f"Epoch {epoch}/{args.max_epochs-1} {idx}/{len(train_loader)} "
                  f"loss: {loss.item():.4f} time {time.time()-t0:.2f}s")

        avg_loss = tsum / nb
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.max_epochs-1} | loss={avg_loss:.4f} | time={dt:.2f}s")

        # Salva loss no CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, round(dt, 2)])

        # -------------------------------
        # SALVAR CHECKPOINTS
        # -------------------------------
        if not hasattr(args, "distributed"):
            args.distributed = False

        if epoch == 0 or (epoch + 1) % 10 == 0:
            save_model_name = f"model_epoch_{epoch:03d}.pt"
            save_checkpoint(model, epoch, args, save_model_name, best_acc=0, optimizer=optimizer)
            print(f"[CKPT] Modelo salvo no epoch {epoch+1}")

    print("\n[INFO] Treino concluído com sucesso 🚀")


# ===========================================================
# Execução principal (necessário no Windows)
# ===========================================================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.freeze_support()
    main()
