# ===========================================================
# train_fish_swin3d_val.py — SwinCell 3D no Windows (com val)
# ===========================================================

import os
import glob
import time
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

from natsort import natsorted
from monai import data, transforms
from torch.nn import MSELoss
from monai.losses import DiceLoss

from swincell.utils.utils import load_default_config, load_model
from swincell.utils.data_utils import flow_generationd
from swincell.trainer import save_checkpoint


def main():
    # =======================================================
    # 1. Diretórios
    # =======================================================
    DATA_DIR = r"C:\Users\ABBE User\Downloads\data_root"
    LOGDIR   = os.path.join(DATA_DIR, "runs")
    os.makedirs(LOGDIR, exist_ok=True)

    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
    TRAIN_LAB_DIR = os.path.join(DATA_DIR, "train", "labels")
    VAL_IMG_DIR   = os.path.join(DATA_DIR, "val", "images")
    VAL_LAB_DIR   = os.path.join(DATA_DIR, "val", "labels")

    # =======================================================
    # 2. CUDA + flags
    # =======================================================
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    # =======================================================
    # 3. Ficheiros
    # =======================================================
    train_imgs = natsorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.tif")))
    train_labs = natsorted(glob.glob(os.path.join(TRAIN_LAB_DIR, "*.tif")))
    val_imgs   = natsorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.tif")))
    val_labs   = natsorted(glob.glob(os.path.join(VAL_LAB_DIR, "*.tif")))

    if not train_imgs:
        raise RuntimeError("Nenhum ficheiro em train/images — corre o script de augmentations primeiro.")

    train_list = [{"image": i, "label": j} for i, j in zip(train_imgs, train_labs)]
    val_list   = [{"image": i, "label": j} for i, j in zip(val_imgs,   val_labs)]

    print(f"[INFO] Nº volumes treino: {len(train_list)}")
    print(f"[INFO] Nº volumes val:    {len(val_list)}")

    # =======================================================
    # 4. Shape da imagem
    # =======================================================
    sample_img = tifffile.imread(train_imgs[0])
    assert sample_img.ndim == 3, "TIFF deve ser 3D (Z,Y,X)."
    img_shape = sample_img.shape

    img_reshape = img_shape  # sem downsample

    # =======================================================
    # 5. Config SwinCell
    # =======================================================
    args = load_default_config()
    args.data_dir = DATA_DIR
    args.dataset = "custom"

    args.roi_x = 128
    args.roi_y = 128
    args.roi_z = 32

    args.a_min = 0
    args.a_max = 1

    args.max_epochs = 300
    args.downsample_factor = 1
    args.save_logits = False

    if not hasattr(args, "optim_lr"):
        args.optim_lr = 1e-4

    # =======================================================
    # 6. Transforms
    # =======================================================
    common_resize = transforms.Resized(keys=["image", "label"], spatial_size=img_reshape)

    train_transform = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        common_resize,
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1),
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
        common_resize,
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1, b_min=0, b_max=1),
        flow_generationd(keys=["label"]),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    # =======================================================
    # 7. Loaders sem multiprocessing (Windows-safe)
    # =======================================================
    train_ds = data.Dataset(train_list, transform=train_transform)
    val_ds   = data.Dataset(val_list,   transform=val_transform)

    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader   = data.DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0)

    # =======================================================
    # 8. Modelo + losses
    # =======================================================
    model = load_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=1e-5)

    loss_mse  = MSELoss()
    loss_dice = DiceLoss(to_onehot_y=False, sigmoid=True)

    best_val = 1e9  # para guardar melhor modelo

    # =======================================================
    # 9. Loop de treino + validação
    # =======================================================
    print("\n===============================")
    print("   TREINO SWINCELL INICIADO")
    print("===============================\n")

    for epoch in range(args.max_epochs):
        # TRAIN
        model.train()
        train_loss = 0
        t0 = time.time()

        for batch in train_loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)

            pred = model(img)

            tgt_mask = lbl[:, 0:1]
            tgt_flow = lbl[:, 1:4]
            pred_mask = pred[:, 0:1]
            pred_flow = pred[:, 1:4]

            l1 = loss_dice(pred_mask, tgt_mask)
            l2 = loss_mse(pred_flow, tgt_flow)
            loss = l1 + l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)

                pred = model(img)
                tgt_mask = lbl[:, 0:1]
                tgt_flow = lbl[:, 1:4]
                pred_mask = pred[:, 0:1]
                pred_flow = pred[:, 1:4]

                l1 = loss_dice(pred_mask, tgt_mask)
                l2 = loss_mse(pred_flow, tgt_flow)

                val_loss += (l1 + l2).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | val={val_loss:.4f} | time={time.time()-t0:.2f}s")

        # MELHOR MODELO
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, epoch, args, "best_model.pt", best_acc=0, optimizer=optimizer)
            print(f"[BEST] Novo melhor modelo salvo (val={val_loss:.4f})")

        # CHECKPOINTS NORMAIS
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, epoch, args, f"model_epoch_{epoch:03d}.pt", best_acc=0, optimizer=optimizer)

    print("\nTreino concluído com sucesso! 🚀")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
