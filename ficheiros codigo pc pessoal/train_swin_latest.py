# train_swincell_fixed_binary_masks.py — Correção: máscaras binárias para Dice Loss
import os, sys, glob, gc, math, time, argparse, warnings, traceback, faulthandler
import numpy as np
faulthandler.enable(all_threads=True)
warnings.filterwarnings("ignore")

os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)

import torch, tifffile, torch.nn.functional as F
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from monai.losses import DiceLoss
from swincell.utils.utils import load_default_config, load_model
from tqdm import tqdm
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=r"D:\User data\InesMarques\SwinCell-main\data_root")
    p.add_argument("--logdir",   type=str, default=r"D:\User data\InesMarques\SwinCell-main\checkpoints19oct_fast")
    p.add_argument("--epochs",   type=int, default=50)
    p.add_argument("--roi_x",    type=int, default=128)
    p.add_argument("--roi_y",    type=int, default=128)
    p.add_argument("--roi_z",    type=int, default=32)
    p.add_argument("--lr",       type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_every",    type=int, default=2)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--threads",      type=int, default=0)
    p.add_argument("--gc_every",     type=int, default=32)
    p.add_argument("--prob_ch",      type=int, default=0)
    p.add_argument("--pos_ratio",    type=float, default=0.8)
    p.add_argument("--border_margin", type=int, default=16)
    p.add_argument("--log_patches_first_epoch", action="store_true")
    return p.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def pad_to_multiple(x: torch.Tensor, multiple=32):
    _, z, y, x_ = x.shape
    pad_z = (multiple - z % multiple) % multiple
    pad_y = (multiple - y % multiple) % multiple
    pad_x = (multiple - x_ % multiple) % multiple
    return F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z))

def ensure_min_size(t: torch.Tensor, min_sz):
    _, z, y, x = t.shape
    dz = max(0, min_sz[0]-z); dy = max(0, min_sz[1]-y); dx = max(0, min_sz[2]-x)
    if dz or dy or dx:
        t = F.pad(t, (0, dx, 0, dy, 0, dz))
    return t

class ForegroundNoBorderDataset(Dataset):
    def __init__(self, image_files, mask_files, roi_zyx, pos_ratio=0.8, border_margin=16, log_patches=False):
        assert len(image_files) == len(mask_files) and len(image_files) > 0
        self.image_files = image_files
        self.mask_files  = mask_files
        self.roi_z, self.roi_y, self.roi_x = roi_zyx
        self.pos_ratio = pos_ratio
        self.border_margin = border_margin
        self.log_patches = log_patches
        self.patch_log = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img = tifffile.imread(self.image_files[idx]).astype("float32")
        msk = tifffile.imread(self.mask_files[idx]).astype("float32")
        orig_shape = img.shape

        # >>> BINARIZAR MÁSCARA: converter IDs de instâncias para 0/1 <<<
        msk = (msk > 0).astype("float32")  # 0 = background, 1 = células

        # Normalizar imagem
        imin, imax = float(img.min()), float(img.max())
        img = (img - imin) / ((imax - imin) + 1e-8)

        if img.ndim == 3: img = img[None, ...]
        if msk.ndim == 3: msk = msk[None, ...]

        img_t = torch.from_numpy(img)
        msk_t = torch.from_numpy(msk)

        img_t = ensure_min_size(img_t, (self.roi_z, self.roi_y, self.roi_x))
        msk_t = ensure_min_size(msk_t, (self.roi_z, self.roi_y, self.roi_x))
        img_t = pad_to_multiple(img_t, 32)
        msk_t = pad_to_multiple(msk_t, 32)

        _, Z_padded, Y_padded, X_padded = img_t.shape
        Z_orig, Y_orig, X_orig = orig_shape

        z_min = max(self.border_margin, self.roi_z // 2)
        z_max = min(Z_orig - self.border_margin, Z_padded - self.roi_z // 2)
        y_min = max(self.border_margin, self.roi_y // 2)
        y_max = min(Y_orig - self.border_margin, Y_padded - self.roi_y // 2)
        x_min = max(self.border_margin, self.roi_x // 2)
        x_max = min(X_orig - self.border_margin, X_padded - self.roi_x // 2)

        if z_max <= z_min: z_min, z_max = 0, Z_orig
        if y_max <= y_min: y_min, y_max = 0, Y_orig
        if x_max <= x_min: x_min, x_max = 0, X_orig

        sample_foreground = (np.random.rand() < self.pos_ratio)

        if sample_foreground:
            fg_coords = torch.nonzero(msk_t[0] > 0, as_tuple=False)
            valid_fg = fg_coords[
                (fg_coords[:, 0] >= z_min) & (fg_coords[:, 0] < z_max) &
                (fg_coords[:, 1] >= y_min) & (fg_coords[:, 1] < y_max) &
                (fg_coords[:, 2] >= x_min) & (fg_coords[:, 2] < x_max)
            ]
            if valid_fg.numel() > 0:
                idx_c = np.random.randint(0, valid_fg.shape[0])
                zc, yc, xc = valid_fg[idx_c].tolist()
            else:
                zc = np.random.randint(z_min, max(z_min+1, z_max))
                yc = np.random.randint(y_min, max(y_min+1, y_max))
                xc = np.random.randint(x_min, max(x_min+1, x_max))
            
            z0 = max(0, min(Z_padded - self.roi_z, zc - self.roi_z // 2))
            y0 = max(0, min(Y_padded - self.roi_y, yc - self.roi_y // 2))
            x0 = max(0, min(X_padded - self.roi_x, xc - self.roi_x // 2))
        else:
            zc = np.random.randint(z_min, max(z_min+1, z_max))
            yc = np.random.randint(y_min, max(y_min+1, y_max))
            xc = np.random.randint(x_min, max(x_min+1, x_max))
            z0 = max(0, min(Z_padded - self.roi_z, zc - self.roi_z // 2))
            y0 = max(0, min(Y_padded - self.roi_y, yc - self.roi_y // 2))
            x0 = max(0, min(X_padded - self.roi_x, xc - self.roi_x // 2))

        if self.log_patches:
            self.patch_log.append((idx, z0, y0, x0, zc, yc, xc))

        img_roi = img_t[:, z0:z0+self.roi_z, y0:y0+self.roi_y, x0:x0+self.roi_x]
        msk_roi = msk_t[:, z0:z0+self.roi_z, y0:y0+self.roi_y, x0:x0+self.roi_x]
        return img_roi, msk_roi

    def get_patch_log(self):
        return self.patch_log

def build_dice_loss():
    try:
        return DiceLoss(to_onehot_y=False, sigmoid=True)  # sigmoid=True porque modelo dá logits
    except TypeError:
        return DiceLoss(to_onehot_y=False, do_sigmoid=True)

def visualize_patches(dataset, logdir, epoch, max_samples=5):
    patch_dir = os.path.join(logdir, "patch_logs")
    os.makedirs(patch_dir, exist_ok=True)
    
    patch_log = dataset.get_patch_log()
    if not patch_log:
        return
    
    by_file = {}
    for (idx, z0, y0, x0, zc, yc, xc) in patch_log:
        if idx not in by_file:
            by_file[idx] = []
        by_file[idx].append((z0, y0, x0, zc, yc, xc))
    
    for i, (idx, patches) in enumerate(list(by_file.items())[:max_samples]):
        try:
            img = tifffile.imread(dataset.image_files[idx]).astype("float32")
            mip_xy = np.max(img, axis=0)
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(mip_xy, cmap='gray', vmin=np.percentile(mip_xy, 1), vmax=np.percentile(mip_xy, 99))
            ax.set_title(f"Patches Ep{epoch+1} - {os.path.basename(dataset.image_files[idx])}")
            
            for (z0, y0, x0, zc, yc, xc) in patches:
                rect = plt.Rectangle((x0, y0), dataset.roi_x, dataset.roi_y, 
                                      edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                ax.plot(xc, yc, 'r+', markersize=15, markeredgewidth=3)
            
            ax.axis('off')
            plt.tight_layout()
            out_path = os.path.join(patch_dir, f"patches_ep{epoch+1:02d}_file{i}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cpu")

    if args.threads <= 0:
        num_threads = os.cpu_count() or 16
    else:
        num_threads = args.threads
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 2))

    train_img_dir = os.path.join(args.data_dir, "train", "images")
    train_lab_dir = os.path.join(args.data_dir, "train", "labels")
    val_img_dir   = os.path.join(args.data_dir, "val", "images")
    val_lab_dir   = os.path.join(args.data_dir, "val", "labels")

    train_images = natsorted(glob.glob(os.path.join(train_img_dir, "*.tif")))
    train_labels = natsorted(glob.glob(os.path.join(train_lab_dir, "*.tif")))
    val_images   = natsorted(glob.glob(os.path.join(val_img_dir, "*.tif")))
    val_labels   = natsorted(glob.glob(os.path.join(val_lab_dir, "*.tif")))

    assert len(train_images) > 0 and len(val_images) > 0

    roi_zyx = (args.roi_z, args.roi_y, args.roi_x)
    
    train_ds = ForegroundNoBorderDataset(train_images, train_labels, roi_zyx, 
                                          args.pos_ratio, args.border_margin, log_patches=False)
    val_ds   = ForegroundNoBorderDataset(val_images, val_labels, roi_zyx, 
                                          args.pos_ratio, args.border_margin, log_patches=False)

    def make_train_loader(use_workers=True):
        nw = args.num_workers if use_workers else 0
        kwargs = {"batch_size": args.batch_size, "shuffle": True, "num_workers": nw,
                  "persistent_workers": False, "pin_memory": False, "timeout": 0}
        if nw > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(train_ds, **kwargs)
    
    val_kwargs = {"batch_size": args.batch_size, "shuffle": False,
                  "num_workers": max(1, args.num_workers // 2),
                  "persistent_workers": True if args.num_workers > 0 else False,
                  "pin_memory": False, "timeout": 0}
    if args.num_workers > 0:
        val_kwargs["prefetch_factor"] = 4
    val_loader = DataLoader(val_ds, **val_kwargs)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    cfg = load_default_config()
    cfg.data_dir = args.data_dir; cfg.model = 'swin'
    cfg.roi_x, cfg.roi_y, cfg.roi_z = args.roi_x, args.roi_y, args.roi_z
    cfg.max_epochs = args.epochs; cfg.optim_lr = args.lr; cfg.weight_decay = args.weight_decay

    model = load_model(cfg).to(device)
    dice_loss = build_dice_loss()
    optimizer = Adam(model.parameters(), lr=cfg.optim_lr, weight_decay=cfg.weight_decay)

    os.makedirs(args.logdir, exist_ok=True)

    losses_csv = os.path.join(args.logdir, "losses.csv")
    if not os.path.exists(losses_csv):
        with open(losses_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    ckpt_last = os.path.join(args.logdir, "last_checkpoint.pth")
    start_epoch, train_losses, val_losses = 0, [], []
    if os.path.exists(ckpt_last):
        ckpt = torch.load(ckpt_last, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        start_epoch = ckpt.get("epoch", -1) + 1

    def save_ckpt(epoch, path):
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_losses": train_losses, "val_losses": val_losses}, path)

    @torch.no_grad()
    def validate():
        model.eval(); val_loss, batches = 0.0, 0
        for img, msk in tqdm(val_loader, desc="Val", leave=False):
            pred = model(img)
            pred_prob = pred[:, args.prob_ch:args.prob_ch+1, ...]
            loss_d = dice_loss(pred_prob, msk)
            val_loss += float(loss_d.item()); batches += 1
            del pred, pred_prob, loss_d, img, msk
        gc.collect()
        return val_loss / max(1, batches)

    best_val = math.inf
    
    try:
        for epoch in range(start_epoch, cfg.max_epochs):
            t0 = time.time(); model.train(); epoch_loss = 0.0
            
            if epoch == 0 and args.log_patches_first_epoch:
                train_ds.log_patches = True
                train_ds.patch_log = []
                train_loader_log = make_train_loader(use_workers=False)
                
                for bidx, (img, msk) in enumerate(tqdm(train_loader_log, desc=f"Ep {epoch+1} [LOG]", leave=True)):
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(img)
                    pred_prob = pred[:, args.prob_ch:args.prob_ch+1, ...]
                    loss_d = dice_loss(pred_prob, msk)
                    
                    # >>> DEBUG NO 1º BATCH <<<
                    if bidx == 0:
                        print(f"\n[DEBUG] pred shape={pred_prob.shape}, min={pred_prob.min():.3f}, max={pred_prob.max():.3f}")
                        print(f"[DEBUG] msk shape={msk.shape}, min={msk.min():.3f}, max={msk.max():.3f}, unique={torch.unique(msk)}")
                        print(f"[DEBUG] loss={loss_d.item():.4f}\n", flush=True)
                    
                    loss_d.backward(); optimizer.step()
                    epoch_loss += float(loss_d.item())
                    del pred, pred_prob, loss_d, img, msk
                    if (bidx+1) % args.gc_every == 0: gc.collect()
                
                visualize_patches(train_ds, args.logdir, epoch, max_samples=5)
                train_ds.log_patches = False
            else:
                train_loader_normal = make_train_loader(use_workers=True)
                for bidx, (img, msk) in enumerate(tqdm(train_loader_normal, desc=f"Ep {epoch+1}", leave=True)):
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(img)
                    pred_prob = pred[:, args.prob_ch:args.prob_ch+1, ...]
                    loss_d = dice_loss(pred_prob, msk)
                    loss_d.backward(); optimizer.step()
                    epoch_loss += float(loss_d.item())
                    del pred, pred_prob, loss_d, img, msk
                    if (bidx+1) % args.gc_every == 0: gc.collect()

            avg_loss = epoch_loss / max(1, len(train_ds) // args.batch_size + 1)
            train_losses.append(avg_loss)

            if (epoch+1) % args.val_every == 0:
                val_loss = validate()
                print(f"[Ep {epoch+1:03d}] train={avg_loss:.4f} val={val_loss:.4f} {time.time()-t0:.1f}s", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(epoch, os.path.join(args.logdir, "best.pth"))
            else:
                val_loss = float("nan")
                print(f"[Ep {epoch+1:03d}] train={avg_loss:.4f} {time.time()-t0:.1f}s", flush=True)

            val_losses.append(val_loss)
            with open(losses_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch+1, avg_loss, val_loss, round(time.time()-t0,3)])

            if (epoch+1) % 5 == 0 or (epoch+1) == cfg.max_epochs:
                save_ckpt(epoch, os.path.join(args.logdir, f"ckpt_ep{epoch+1:02d}.pth"))
                save_ckpt(epoch, ckpt_last)

        with open(os.path.join(args.logdir, "losses.json"), "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
        print("[OK]!", flush=True)
    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support(); main()


