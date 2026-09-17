import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tifffile import imread
from scipy.stats import ttest_ind

# ======================================================================
#  CONFIGURAÇÃO DE DIRETÓRIOS E MODELOS
# ======================================================================

DOWNLOADS = r"D:\User data\InesMarques\Swincell"

GT_PATH   = os.path.join(DOWNLOADS, "manual_anot.tif")
SWIN_PATH = os.path.join(DOWNLOADS, "swincell_output", "masks_prediction.tif")
STAR_PATH = os.path.join(DOWNLOADS, "prediction_labels.tif")
NUC_PATH  = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_nuclei_masks.tif",
)
CYTO3_PRE_PATH = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_cyto3_masks.tif",
)
CYTO3_FT_PATH  = os.path.join(
    DOWNLOADS,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_cyto3finetunedmodel.tif",
)

OUTPUT_DIR = "report_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# thresholds para AP (COCO-style)
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# cores consistentes (Nature-ish)
COLOR_MAP = {
    "GT":                "#000000",
    "SwinCell":          "#3b4cc0",
    "StarDist3D":        "#f4a259",
    "Cellpose nuclei ft":"#8a508f",
    "Cellpose cyto3":    "#d1495b",
    "Cellpose cyto3 ft": "#5aa469",
}

plt.rcParams["font.size"] = 20


# ======================================================================
#  FUNÇÕES AUXILIARES
# ======================================================================

def load_labels():
    print("[INFO] Loading label volumes...")
    gt      = imread(GT_PATH)
    swin    = imread(SWIN_PATH)
    stard   = imread(STAR_PATH)
    nuclei  = imread(NUC_PATH)
    cyto3   = imread(CYTO3_PRE_PATH)
    cyto3ft = imread(CYTO3_FT_PATH)
    print("[INFO] Loaded all volumes.")
    models = {
        "GT": gt,
        "SwinCell": swin,
        "StarDist3D": stard,
        "Cellpose nuclei ft": nuclei,
        "Cellpose cyto3": cyto3,
        "Cellpose cyto3 ft": cyto3ft,
    }
    return models


def extract_volumes(label_vol, min_size=2500, max_size=100000):
    """Devolve array com volumes (número de voxels) de cada objeto filtrado."""
    ids = np.unique(label_vol)
    ids = ids[ids > 0]  # remover background
    vols = []
    for lab in ids:
        v = np.count_nonzero(label_vol == lab)
        if min_size <= v <= max_size:
            vols.append(v)
    return np.array(vols)


def voxelwise_metrics(gt, pred):
    """Métricas voxelwise (binário) entre GT e pred."""
    g = gt > 0
    p = pred > 0
    tp = np.logical_and(g, p).sum()
    fp = np.logical_and(~g, p).sum()
    fn = np.logical_and(g, ~p).sum()

    if tp + fp + fn == 0:
        iou = 0.0
    else:
        iou = tp / (tp + fp + fn)

    if (2*tp + fp + fn) == 0:
        dice = 0.0
    else:
        dice = 2 * tp / (2*tp + fp + fn)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    return dice, iou, prec, rec, f1


def compute_instance_iou_matrix(gt, pred):
    """
    Constrói matriz IoU entre labels de GT e de Pred.
    Usa histograma conjunto (interseções) + volumes marginais.
    """
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)

    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]
    pr_ids = np.unique(pred)
    pr_ids = pr_ids[pr_ids > 0]

    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return gt_ids, pr_ids, np.zeros((len(gt_ids), len(pr_ids)), dtype=float)

    gt_flat = gt.ravel()
    pr_flat = pred.ravel()

    # só voxels onde há algo em GT ou Pred
    valid = np.logical_or(gt_flat > 0, pr_flat > 0)
    gt_flat = gt_flat[valid]
    pr_flat = pr_flat[valid]

    pairs = np.stack([gt_flat, pr_flat], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)

    # mapear ids para índices
    gt_id_to_idx = {lab: i for i, lab in enumerate(gt_ids)}
    pr_id_to_idx = {lab: i for i, lab in enumerate(pr_ids)}

    inter = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
    for (g_lab, p_lab), c in zip(uniq, counts):
        if g_lab == 0 or p_lab == 0:
            continue
        gi = gt_id_to_idx[g_lab]
        pi = pr_id_to_idx[p_lab]
        inter[gi, pi] += c

    # áreas marginais
    gt_counts = np.bincount(gt_flat[gt_flat > 0])
    # gt_flat >0 pode não começar em 1, então compute properly:
    gt_vols = {lab: np.count_nonzero(gt == lab) for lab in gt_ids}
    pr_vols = {lab: np.count_nonzero(pred == lab) for lab in pr_ids}

    iou_mat = np.zeros_like(inter, dtype=float)
    for i, g_lab in enumerate(gt_ids):
        for j, p_lab in enumerate(pr_ids):
            inter_ij = inter[i, j]
            if inter_ij == 0:
                continue
            union_ij = gt_vols[g_lab] + pr_vols[p_lab] - inter_ij
            if union_ij > 0:
                iou_mat[i, j] = inter_ij / union_ij

    return gt_ids, pr_ids, iou_mat


def compute_AP_curve(gt, pred, thresholds):
    """
    Para cada IoU threshold, faz matching greedy de instâncias
    e calcula AP_t = TP / (TP + FP + FN) (tipo Jaccard de instâncias).
    Devolve dict threshold -> AP.
    """
    gt_ids, pr_ids, iou_mat = compute_instance_iou_matrix(gt, pred)

    n_gt = len(gt_ids)
    n_pr = len(pr_ids)
    AP = {}

    if n_gt == 0 and n_pr == 0:
        return {t: 0.0 for t in thresholds}
    if n_gt == 0:
        return {t: 0.0 for t in thresholds}
    if n_pr == 0:
        return {t: 0.0 for t in thresholds}

    # lista de pares (i,j,IoU) ordenada por IoU decrescente
    pairs = []
    for i in range(n_gt):
        for j in range(n_pr):
            if iou_mat[i, j] > 0:
                pairs.append((i, j, iou_mat[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for thr in thresholds:
        matched_gt = set()
        matched_pr = set()
        tp = 0

        for i, j, v in pairs:
            if v < thr:
                break
            if i in matched_gt or j in matched_pr:
                continue
            matched_gt.add(i)
            matched_pr.add(j)
            tp += 1

        fp = n_pr - tp
        fn = n_gt - tp

        denom = tp + fp + fn
        ap_t = tp / denom if denom > 0 else 0.0
        AP[thr] = ap_t

    return AP


# ======================================================================
#  PIPELINE PRINCIPAL
# ======================================================================

def main():
    models = load_labels()

    # ------------------------------------------------------------------
    # 1) Volumes por modelo (para violin/boxplots)
    # ------------------------------------------------------------------
    print("[INFO] Extracting object volumes...")
    vol_dict = {}
    rows = []
    for name, lab in models.items():
        if name == "GT":
            vols = extract_volumes(lab)
        else:
            vols = extract_volumes(lab)
        vol_dict[name] = vols
        for v in vols:
            rows.append([name, v])

    df_vol = pd.DataFrame(rows, columns=["Model", "Volume"])

    # ------------------------------------------------------------------
    # 2) Métricas voxelwise + AP multi-threshold por modelo
    # ------------------------------------------------------------------
    gt = models["GT"]
    metrics_rows = []
    AP_results = {}

    print("[INFO] Computing metrics and AP curves...")
    for name, lab in models.items():
        if name == "GT":
            continue
        dice, iou, prec, rec, f1 = voxelwise_metrics(gt, lab)
        ap_curve = compute_AP_curve(gt, lab, IOU_THRESHOLDS)
        AP_results[name] = ap_curve

        ap50  = ap_curve.get(0.50, 0.0)
        ap75  = ap_curve.get(0.75, 0.0)
        map5095 = np.mean(list(ap_curve.values())) if len(ap_curve) > 0 else 0.0

        metrics_rows.append({
            "Model": name,
            "Dice": dice,
            "IoU": iou,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AP@0.5": ap50,
            "AP@0.75": ap75,
            "AP@[0.5:0.95]": map5095,
        })

    df_metrics = pd.DataFrame(metrics_rows).set_index("Model")
    print("\n=== METRICS (voxelwise + AP) ===")
    print(df_metrics)

    # ==================================================================
    # 3) VIOLIN PLOTS (LIN + LOG) COM P-VALUES AO LADO
    # ==================================================================
    models_order = ["GT"] + [m for m in df_metrics.index]
    palette = [COLOR_MAP[m] for m in models_order]

    # GT como referência
    gt_vol = vol_dict["GT"]
    y_max = df_vol["Volume"].max()

    # --- Linear ---
    plt.figure(figsize=(22, 10))
    ax = sns.violinplot(
        data=df_vol, x="Model", y="Volume",
        order=models_order,
        palette=palette,
        inner="quartile", cut=0, linewidth=1.5
    )

    for i, model in enumerate(models_order):
        if model == "GT":
            label = "ref."
        else:
            v = vol_dict[model]
            p = ttest_ind(gt_vol, v, equal_var=False).pvalue
            label = f"p = {p:.1e}"
        ax.text(
            i + 0.18,
            y_max * 0.95,
            label,
            ha="left", va="top",
            fontsize=18,
            color="black"
        )

    plt.title("Distribution of Object Volumes per Model (filtered)", fontsize=22)
    plt.ylabel("Volume (voxels)")
    plt.xlabel("")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "volumes_violin_lin.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # --- Log10 ---
    plt.figure(figsize=(22, 10))
    ax = sns.violinplot(
        data=df_vol, x="Model", y="Volume",
        order=models_order,
        palette=palette,
        inner="quartile", cut=0, linewidth=1.5
    )
    ax.set_yscale("log")
    # usar mesma posição para o texto (em y linear, matplotlib converte para log)
    for i, model in enumerate(models_order):
        if model == "GT":
            label = "ref."
        else:
            v = vol_dict[model]
            p = ttest_ind(gt_vol, v, equal_var=False).pvalue
            label = f"p = {p:.1e}"
        ax.text(
            i + 0.18,
            y_max * 0.9,
            label,
            ha="left", va="top",
            fontsize=18,
            color="black"
        )

    plt.title("Distribution of Object Volumes per Model (log scale)", fontsize=22)
    plt.ylabel("Volume (voxels, log scale)")
    plt.xlabel("")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "volumes_violin_log.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # ==================================================================
    # 4) BOXPLOTS (LIN + LOG)
    # ==================================================================
    # Linear
    plt.figure(figsize=(22, 10))
    sns.boxplot(
        data=df_vol, x="Model", y="Volume",
        order=models_order,
        palette=palette,
        showfliers=False
    )
    plt.title("Object Volume Distribution per Model (boxplot)", fontsize=22)
    plt.ylabel("Volume (voxels)")
    plt.xlabel("")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "volumes_boxplot_lin.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # Log
    plt.figure(figsize=(22, 10))
    ax = sns.boxplot(
        data=df_vol, x="Model", y="Volume",
        order=models_order,
        palette=palette,
        showfliers=False
    )
    ax.set_yscale("log")
    plt.title("Object Volume Distribution per Model (boxplot, log scale)", fontsize=22)
    plt.ylabel("Volume (voxels, log scale)")
    plt.xlabel("")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "volumes_boxplot_log.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # ==================================================================
    # 5) HEATMAP DE MÉTRICAS (Dice, IoU, Precision, Recall, F1)
    # ==================================================================
    metrics_for_heatmap = df_metrics[["Dice", "IoU", "Precision", "Recall", "F1"]]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        metrics_for_heatmap,
        annot=True, fmt=".3f",
        cmap="viridis",
        vmin=0.0, vmax=1.0,
        cbar_kws={"label": "Metric value"}
    )
    plt.title("Segmentation metrics (3D voxelwise)", fontsize=22)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "metrics_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # ==================================================================
    # 6) BARPLOT DAS MÉTRICAS (opcional mas útil)
    # ==================================================================
    df_melt = df_metrics.reset_index().melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Value"
    )

    plt.figure(figsize=(18, 8))
    sns.barplot(
        data=df_melt,
        x="Metric",
        y="Value",
        hue="Model",
        palette=[COLOR_MAP[m] for m in df_metrics.index]
    )
    plt.title("Segmentation metrics per model", fontsize=22)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "metrics_barplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # ==================================================================
    # 7) mAP vs IoU threshold (Nature-style)
    # ==================================================================
    plt.figure(figsize=(8, 6))

    # escolher alguns modelos para a curva (tu podes ajustar)
    pretrained = ["SwinCell", "Cellpose cyto3"]
    finetuned  = ["Cellpose cyto3 ft", "StarDist3D", "Cellpose nuclei ft"]

    for name in pretrained:
        if name not in AP_results:
            continue
        d = AP_results[name]
        thr = sorted(d.keys())
        vals = [d[t] for t in thr]
        plt.plot(
            thr, vals,
            "--",
            linewidth=2.5,
            color=COLOR_MAP[name],
            label=f"{name} (pretrained)"
        )

    for name in finetuned:
        if name not in AP_results:
            continue
        d = AP_results[name]
        thr = sorted(d.keys())
        vals = [d[t] for t in thr]
        plt.plot(
            thr, vals,
            "-",
            linewidth=2.5,
            color=COLOR_MAP[name],
            label=name
        )

    plt.xlabel("IoU threshold", fontsize=20)
    plt.ylabel("mAP", fontsize=20)
    plt.ylim(0, 1.02)
    plt.xlim(0.5, 0.95)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, fontsize=14)
    plt.title("mAP vs IoU threshold", fontsize=22)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "map_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("[SAVED]", out_path)

    # ==================================================================
    # 8) TABELA LaTeX COM MÉTRICAS + APs
    # ==================================================================
    latex_table = df_metrics.to_latex(
        float_format="%.4f",
        column_format="l" + "c" * df_metrics.shape[1],
        caption="3D segmentation metrics for all models (voxelwise and instance-wise AP).",
        label="tab:segmentation_metrics"
    )
    tex_path = os.path.join(OUTPUT_DIR, "metrics_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print("[SAVED]", tex_path)
    # ======================================================================
# 9) ORTOGONAIS E PATCHES PARA TESE 3 EXEMPLOS
# ======================================================================

from matplotlib.colors import ListedColormap

RAW_PATH = None
# Exemplo
# RAW_PATH = os.path.join(DOWNLOADS, "raw_image.tif")

PIXEL_SIZE_UM = 0.1
SCALEBAR_UM = 1.0
PATCH_SIZE = 256
N_EXAMPLES = 3
PATCH_SEED = 7
MIN_DIST_PX = 180

def load_raw_volume():
    if RAW_PATH is None:
        return None
    if not os.path.exists(RAW_PATH):
        print("[WARN] RAW_PATH definido mas ficheiro nao existe", RAW_PATH)
        return None
    return imread(RAW_PATH)

def make_overlap_map(g, p):
    g = g.astype(bool)
    p = p.astype(bool)
    out = np.zeros(g.shape, dtype=np.uint8)
    out[np.logical_and(g, np.logical_not(p))] = 1
    out[np.logical_and(np.logical_not(g), p)] = 2
    out[np.logical_and(g, p)] = 3
    return out

def safe_crop(img2d, y0, x0, h, w):
    H, W = img2d.shape
    y0 = int(max(0, y0))
    x0 = int(max(0, x0))
    y1 = int(min(H, y0 + h))
    x1 = int(min(W, x0 + w))
    return img2d[y0:y1, x0:x1], y0, x0

def draw_scalebar(ax, patch_h, pixel_size_um, scalebar_um):
    if pixel_size_um is None or pixel_size_um <= 0:
        return
    bar_px = int(round(float(scalebar_um) / float(pixel_size_um)))
    bar_px = int(max(5, min(bar_px, PATCH_SIZE - 20)))
    x0 = 10
    y0 = patch_h - 12
    ax.plot([x0, x0 + bar_px], [y0, y0], linewidth=6, color="white", solid_capstyle="butt")
    ax.text(x0, y0 - 8, f"{scalebar_um:.0f} µm", color="white", fontsize=12, ha="left", va="bottom")

def pick_centers_from_union(union3d, n, rng, min_dist_px=180, max_tries=200000):
    Z, H, W = union3d.shape
    coords = np.argwhere(union3d)
    if coords.size == 0:
        return []

    picked = []
    tries = 0
    while len(picked) < n and tries < max_tries:
        tries += 1
        z, y, x = coords[int(rng.integers(0, len(coords)))]

        y0 = int(y - PATCH_SIZE // 2)
        x0 = int(x - PATCH_SIZE // 2)
        if y0 < 0 or x0 < 0 or (y0 + PATCH_SIZE) > H or (x0 + PATCH_SIZE) > W:
            continue

        ok = True
        for (pz, py, px) in picked:
            if pz != z:
                continue
            dy = float(py - y)
            dx = float(px - x)
            if (dy * dy + dx * dx) < float(min_dist_px * min_dist_px):
                ok = False
                break
        if not ok:
            continue

        picked.append((int(z), int(y), int(x)))

    return picked

def norm_to_uint8(img):
    x = img.astype(np.float32)
    x = x - np.percentile(x, 1)
    p = np.percentile(x, 99)
    if p <= 0:
        p = x.max() if x.max() > 0 else 1.0
    x = np.clip(x / p, 0, 1)
    return (255 * x).astype(np.uint8)

def plot_raw_or_blank(ax, raw2d):
    ax.set_xticks([])
    ax.set_yticks([])
    if raw2d is None:
        ax.imshow(np.zeros((10, 10), dtype=np.uint8), cmap="gray", interpolation="nearest")
        return
    ax.imshow(norm_to_uint8(raw2d), cmap="gray", interpolation="nearest")

def plot_overlap(ax, raw2d, overlap2d, title, add_scalebar=False):
    ax.set_xticks([])
    ax.set_yticks([])
    if raw2d is not None:
        ax.imshow(norm_to_uint8(raw2d), cmap="gray", interpolation="nearest")
    cmap = ListedColormap([
        (0, 0, 0, 0.0),
        (1.0, 0.0, 1.0, 0.85),
        (1.0, 1.0, 0.0, 0.85),
        (0.0, 1.0, 0.0, 0.70),
    ])
    ax.imshow(overlap2d, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    if add_scalebar:
        draw_scalebar(ax, overlap2d.shape[0], PIXEL_SIZE_UM, SCALEBAR_UM)

def make_orthogonal_and_patch_figure(raw, gt, pred_dict, center, out_png):
    zc, yc, xc = center
    Z, H, W = gt.shape

    y0 = int(yc - PATCH_SIZE // 2)
    x0 = int(xc - PATCH_SIZE // 2)

    gt_bin = gt > 0

    columns = ["Raw", "Ground truth"] + list(pred_dict.keys())
    ncol = len(columns)
    nrow = 4

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow))

    def get_views(vol):
        xy = vol[zc]
        xz = vol[:, yc, :]
        yz = vol[:, :, xc]
        xz = np.asarray(xz)
        yz = np.asarray(yz)
        return xy, xz, yz

    raw_xy = raw_xz = raw_yz = None
    if raw is not None:
        raw_xy, raw_xz, raw_yz = get_views(raw)

    gt_xy, gt_xz, gt_yz = get_views(gt_bin)

    raw_xy_patch = None
    if raw is not None:
        raw_xy_patch, _, _ = safe_crop(raw_xy, y0, x0, PATCH_SIZE, PATCH_SIZE)

    gt_xy_patch, yy, xx = safe_crop(gt_xy, y0, x0, PATCH_SIZE, PATCH_SIZE)

    for r in range(nrow):
        for c in range(ncol):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    axes[0, 0].set_title("Raw", fontsize=14)
    axes[0, 1].set_title("Ground truth", fontsize=14)
    for i, name in enumerate(pred_dict.keys(), start=2):
        axes[0, i].set_title(name, fontsize=14)

    plot_raw_or_blank(axes[0, 0], raw_xy)
    axes[0, 1].imshow(gt_xy, cmap="gray", interpolation="nearest")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    plot_raw_or_blank(axes[1, 0], raw_xz)
    axes[1, 1].imshow(gt_xz, cmap="gray", interpolation="nearest")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    plot_raw_or_blank(axes[2, 0], raw_yz)
    axes[2, 1].imshow(gt_yz, cmap="gray", interpolation="nearest")
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])

    for j, (name, pred) in enumerate(pred_dict.items(), start=2):
        pbin = pred > 0
        pr_xy, pr_xz, pr_yz = get_views(pbin)

        axes[0, j].imshow(pr_xy, cmap="gray", interpolation="nearest")
        axes[1, j].imshow(pr_xz, cmap="gray", interpolation="nearest")
        axes[2, j].imshow(pr_yz, cmap="gray", interpolation="nearest")

        pr_xy_patch, _, _ = safe_crop(pr_xy, y0, x0, PATCH_SIZE, PATCH_SIZE)
        overlap_patch = make_overlap_map(gt_xy_patch, pr_xy_patch)
        plot_overlap(
            axes[3, j],
            raw_xy_patch,
            overlap_patch,
            "Patch XY overlap",
            add_scalebar=(j == 2)
        )

    plot_raw_or_blank(axes[3, 0], raw_xy_patch)
    axes[3, 0].set_title("Patch XY raw", fontsize=14)

    axes[3, 1].imshow(gt_xy_patch, cmap="gray", interpolation="nearest")
    axes[3, 1].set_title("Patch XY ground truth", fontsize=14)

    axes[0, 0].set_ylabel("XY", fontsize=14)
    axes[1, 0].set_ylabel("XZ", fontsize=14)
    axes[2, 0].set_ylabel("YZ", fontsize=14)
    axes[3, 0].set_ylabel("Patch", fontsize=14)

    fig.suptitle(f"z {zc}  y {yc}  x {xc}", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print("[SAVED]", out_png)

def run_patch_examples(models):
    out_dir = os.path.join(OUTPUT_DIR, "orthogonal_patches")
    os.makedirs(out_dir, exist_ok=True)

    raw = load_raw_volume()
    gt = models["GT"]

    pred_dict = {k: v for k, v in models.items() if k != "GT"}

    union = np.zeros_like(gt, dtype=bool)
    union |= (gt > 0)
    for v in pred_dict.values():
        union |= (v > 0)

    rng = np.random.default_rng(PATCH_SEED)
    centers = pick_centers_from_union(union, N_EXAMPLES, rng, min_dist_px=MIN_DIST_PX)

    if len(centers) == 0:
        print("[WARN] Nao foi possivel encontrar patches validos")
        return

    meta = []
    for i, c in enumerate(centers):
        zc, yc, xc = c
        out_png = os.path.join(out_dir, f"example_{i+1:02d}_z{zc}_y{yc}_x{xc}.png")
        make_orthogonal_and_patch_figure(raw, gt, pred_dict, c, out_png)
        meta.append({"example": i + 1, "z": zc, "y": yc, "x": xc, "patch_size": PATCH_SIZE})

    meta_csv = os.path.join(out_dir, "examples_meta.csv")
    pd.DataFrame(meta).to_csv(meta_csv, index=False)
    print("[SAVED]", meta_csv)



if __name__ == "__main__":
    main()
