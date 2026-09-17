import os
import torch
import tifffile
import numpy as np

from monai import transforms, data
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from functools import partial

from swincell.utils.utils import load_default_config, load_model
from swincell.cellpose_dynamics import compute_masks


# ===========================================================
# 1) Caminhos atualizados pelo utilizador
# ===========================================================
MODEL_PATH = r"D:\User data\InesMarques\Swincell\model_epoch_049.pt"
INPUT_TIF  = r"D:\User data\InesMarques\Swincell\my_image.tif"   # <-- substituir pelo teu TIFF
OUTPUT_DIR = r"D:\User data\InesMarques\Swincell\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)


# ===========================================================
# 2) Carregar modelo treinado
# ===========================================================
args = load_default_config()
args.roi_x = 128
args.roi_y = 128
args.roi_z = 32
args.a_min = 0
args.a_max = 1

model = load_model(args).to(device)

print("[INFO] Loading checkpoint:", MODEL_PATH)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state["state_dict"])
model.eval()


# ===========================================================
# 3) Preparar TIFF
# ===========================================================
img = tifffile.imread(INPUT_TIF)  
assert img.ndim == 3, "A imagem deve ser 3D (Z,Y,X)."

img_reshape = img.shape

test_list = [{"image": INPUT_TIF, "label": INPUT_TIF}]  # label é dummy, não usada


# ===========================================================
# 4) Transforms iguais ao treino
# ===========================================================
test_transform = transforms.Compose([
    transforms.LoadImaged(keys=["image", "label"]),
    transforms.EnsureChannelFirstd(keys=["image", "label"]),
    transforms.Resized(keys=["image", "label"], spatial_size=img_reshape),
    transforms.ScaleIntensityRanged(keys=["image"], 
                                    a_min=0, a_max=1,
                                    b_min=0, b_max=1,
                                    clip=True),
    transforms.ToTensord(keys=["image", "label"]),
])

test_ds = data.Dataset(test_list, test_transform)
test_loader = data.DataLoader(test_ds, batch_size=1)


# ===========================================================
# 5) Inferência com sliding window
# ===========================================================
model_inferer = partial(
    sliding_window_inference,
    roi_size=(128,128,32),
    sw_batch_size=2,
    predictor=model,
    overlap=0.5,
    mode="gaussian"
)

post_sigmoid = Activations(sigmoid=True)
post_threshold = AsDiscrete(threshold=0.5)


with torch.no_grad():
    batch = next(iter(test_loader))
    x = batch["image"].to(device)

    logits = model_inferer(x)
    logits_np = np.squeeze(logits.detach().cpu().numpy())  # (4, Z, Y, X)

    cellprob = post_threshold(post_sigmoid(logits_np[0]))
    flows = logits_np[1:4]

    # Transpor para (C, X, Y, Z) para o compute_masks
    logits_T = np.transpose(logits_np, (0,3,2,1))

    masks, flow = compute_masks(
        logits_T[[3,2,1]],    # flows invertidos (Z,Y,X → X,Y,Z)
        logits_T[0],
        cellprob_threshold=0.4,
        flow_threshold=0.4,
        do_3D=True,
        min_size=2500,
        use_gpu=True
    )


    # =======================================================
    # Guardar resultado
    # =======================================================
    out_path = os.path.join(OUTPUT_DIR, "prediction_masks.tif")
    tifffile.imwrite(out_path, masks.astype(np.uint16))

    print("[INFO] Máscara 3D salva em:", out_path)
