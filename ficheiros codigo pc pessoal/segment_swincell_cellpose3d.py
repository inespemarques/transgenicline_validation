# infer_from_notebook_style.py — replica o bloco do training_prediction_pipeline.ipynb
import os, glob, numpy as np, torch, matplotlib.pyplot as plt, tifffile
from functools import partial
from monai import transforms, data
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks
from swincell.utils.plot_utils import get_random_cmap

def main():
    # --------- config (ajuste estes caminhos/args conforme o seu setup) ---------
    data_dir = r"D:\User data\InesMarques\SwinCell-main\data_root\\"
    output_folder = os.path.join(data_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    infer_ROI = (256, 256, 32)   # tal como no notebook
    img_reshape = infer_ROI      # notebook usa a mesma shape para Resized
    device = torch.device("cpu") # notebook usou cuda(device); aqui CPU por segurança
    save_logits = True           # corresponde a args.save_logits no notebook
    a_min, a_max, b_min, b_max = 0.0, 1.0, 0.0, 1.0
    cellprob_threshold = 0.4     # iguais aos do notebook
    flow_threshold = 0.4
    downsample_factor = 1        # use o mesmo args.downsample_factor do seu treino

    # listas de teste no estilo do notebook (test_datalist)
    # aqui supondo um par imagem/label por TIFF (labels opcionais para esta inferência)
    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.tif")))
    test_datalist = [{"image": p, "label": p.replace("\\images\\", "\\labels\\")} for p in image_paths]

    # --------- modelo SwinCell (igual ao notebook) ---------
    cfg = load_default_config()
    cfg.model = "swin"
    model = load_model(cfg).to(device)
    model.eval()

    # se tiver um checkpoint:
    # ckpt = torch.load(r"D:\...\best.pth", map_location=device)
    # state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    # model.load_state_dict(state)

    # --------- transforms MONAI (iguais ao notebook) ---------
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # flow_reshaped(keys=["label"]),  # está comentado no notebook
            transforms.Resized(keys=["image", "label"], spatial_size=img_reshape),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_ds = data.Dataset(data=test_datalist, transform=test_transform)
    test_loader = data.DataLoader(test_ds, batch_size=1, sampler=None, drop_last=False)

    # --------- inferer como no notebook (partial do sliding_window_inference) ---------
    model_inferer = partial(
        sliding_window_inference,
        roi_size=infer_ROI,
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
        mode="gaussian",
    )
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            out_filename = os.path.basename(test_datalist[idx]["image"]).split(".")[0] + "_pred.tiff"
            data_test = batch_data["image"].to(device)

            # logits: (1, C, Z, Y, X)
            logits = model_inferer(data_test)

            # Para operações numpy/matplotlib no estilo do notebook:
            logits_out = np.squeeze(logits.detach().cpu().numpy())  # (C, Z, Y, X)

            # aplicar sigmoid+threshold só no canal 0, como no caderno
            logits_out[0] = post_pred(post_sigmoid(torch.from_numpy(logits_out[0])).numpy())

            # transpor para (C, X, Y, Z) como no notebook
            logits_out_transposed = np.transpose(logits_out, (0, 3, 2, 1))  # (C, X, Y, Z)
            flows = logits_out[1:4, :, :, :]  # (3, Z, Y, X) ainda em ordem original

            if save_logits:
                tifffile.imwrite(os.path.join(output_folder, "pred_logits_transposed," + out_filename), logits_out_transposed.astype("float32"))

            # compute_masks no estilo notebook:
            # eles passam [3,2,1] para reordenar eixos para (dZ,dY,dX)
            masks_recon, p = compute_masks(
                logits_out_transposed[[3, 2, 1], :, :, :],        # (dZ,dY,dX) em (X,Y,Z)
                logits_out_transposed[0, :, :, :],                # cellprob em (X,Y,Z)
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                do_3D=True,
                min_size=2500 // downsample_factor // downsample_factor,
                use_gpu=False,   # True se quiser GPU e tiver CUDA
            )
            print("masks shape:", masks_recon.shape)

            # --------- visualização como no notebook ---------
            # usar o tensor original 'logits' para obter shape e slice central
            img_shape = logits.shape  # (1, C, Z, Y, X)
            slice2view = int(img_shape[-1] // 2)  # eixo X no caderno

            img = np.squeeze(data_test.detach().cpu().numpy())  # (Z,Y,X)
            flow = logits_out[1:4]  # (3, Z, Y, X)
            flow_slice = flow[:, :, :, slice2view].transpose(1, 2, 0)  # (Y, Z?) -> segue exactly caderno

            n_row = 2
            fig, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(12, 10))

            axes[0, 0].imshow(img[:, :, slice2view], cmap="gray")
            axes[0, 0].set_title("Raw")

            axes[0, 1].imshow(logits_out[0, :, :, slice2view], cmap="gray")
            axes[0, 1].set_title("Cell prob")

            axes[1, 0].imshow(flow_slice)
            axes[1, 0].set_title("Predicted flows")

            axes[1, 1].imshow(masks_recon[slice2view].T, cmap=get_random_cmap(30))
            axes[1, 1].set_title("Predicted Masks")

            for i in range(2):
                for j in range(2):
                    axes[i, j].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, os.path.basename(out_filename).replace("_pred.tiff", "_viz.png")), dpi=150)
            plt.close(fig)

            # gravar também as saídas principais se quiser
            tifffile.imwrite(os.path.join(output_folder, os.path.basename(out_filename).replace("_pred.tiff", "_prob.tif")),
                             logits_out[0].astype("float32"))
            tifffile.imwrite(os.path.join(output_folder, os.path.basename(out_filename).replace("_pred.tiff", "_labels.tif")),
                             masks_recon.astype("uint32"))

if __name__ == "__main__":
    main()

