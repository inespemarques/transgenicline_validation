import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage import exposure
from skimage.morphology import disk
from cellpose import models as cp_models

# ============================================================
# CONFIGURACAO
# ============================================================

SWINCELL_DIR = r"D:\User data\InesMarques\Swincell"

RAW_PATH = os.path.join(
    SWINCELL_DIR,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif",
)

GT_PATH = os.path.join(SWINCELL_DIR, "manual_anot.tif")
SWIN_PATH = os.path.join(SWINCELL_DIR, "swincell_output", "masks_prediction.tif")
STAR_PATH = os.path.join(SWINCELL_DIR, "prediction_labels.tif")
CELLPOSE_NUCLEI_FT_PATH = os.path.join(
    SWINCELL_DIR,
    "fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices_nuclei_masks.tif",
)

CELLPOSE_FT_MODEL_PATH = r"D:\User data\InesMarques\20250920\train_hindbrain\data\train_crops\preprocessimage100radiuscubic\alltrain\augmented_cellpose_split\train\models\hindbrain_cyto3_finetune2dezembro"

# Pasta de saída (Atualizada para V3 para não misturar)
OUT_ROOT = os.path.join(SWINCELL_DIR, "tese_patches_fixed_GT_preprocessing_optimizedcflows")
os.makedirs(OUT_ROOT, exist_ok=True)

PATCH_META_CSV = os.path.join(OUT_ROOT, "patches_meta.csv")

# Caminhos para máscaras E flows 3D completas do Cellpose
CELLPOSE_CYTO3_FULL_PATH = os.path.join(OUT_ROOT, "cellpose_cyto3_masks_full.tif")
CELLPOSE_CYTO3_FT_FULL_PATH = os.path.join(OUT_ROOT, "cellpose_cyto3_finetuned_masks_full.tif")
CELLPOSE_CYTO3_FLOWS_PATH = os.path.join(OUT_ROOT, "cellpose_cyto3_flows_full.pkl")
CELLPOSE_CYTO3_FT_FLOWS_PATH = os.path.join(OUT_ROOT, "cellpose_cyto3_finetuned_flows_full.pkl")

PIXEL_SIZE_UM = 0.1
SCALEBAR_UM = 1.0

PATCH_SIZE = 256
N_EXAMPLES = 50
PATCH_SEED = 7       # Seed fixa
MIN_DIST_PX = 180

CELLPOSE_GPU = True
CELLPOSE_CHANNELS = [0, 0]
CELLPOSE_DIAMETER = None
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_CELLPROB_THRESHOLD = -5
CELLPOSE_DO_3D = True
CELLPOSE_ANISOTROPY = 2.7
CELLPOSE_STITCH_THRESHOLD = 0.0

FORCE_RESEGMENT = False

# --- PARAMETROS DO PREPROCESSING ---
ROLLING_BALL_RADIUS = 30  # Raio otimizado
MEDIAN_FILTER_RADIUS = 1
CLAHE_BLOCKSIZE = 127
CLAHE_BINS = 256
CLAHE_SLOPE = 3.0

# ============================================================
# CORES DA CLASSIFICACAO
# ============================================================

CLASS_CMAP = ListedColormap([
    (0, 0, 0, 0.0),        # 0: fundo transparente
    (1.0, 0.0, 0.0, 0.8),  # 1: FN vermelho
    (1.0, 1.0, 0.0, 0.8),  # 2: FP amarelo
    (0.0, 1.0, 0.0, 0.7),  # 3: TP verde
])

# ============================================================
# UTILITARIOS
# ============================================================

def ensure_zyx_single_channel(raw):
    a = np.asarray(raw)
    if a.ndim == 2: return a[None, ...]
    if a.ndim == 3: return a
    if a.ndim == 4:
        if a.shape[-1] in [1, 2, 3, 4]: return a[..., 0]
        if a.shape[1] in [1, 2, 3, 4]: return a[:, 0, ...]
    raise ValueError(f"Formato inesperado para RAW, shape {a.shape}")

def norm_to_uint8(img):
    x = img.astype(np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    denom = max(p99 - p1, 1e-8)
    x = (x - p1) / denom
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)

def class_map(gt_bin, pr_bin):
    g = gt_bin.astype(bool)
    p = pr_bin.astype(bool)
    out = np.zeros(g.shape, dtype=np.uint8)
    out[g & ~p] = 1; out[~g & p] = 2; out[g & p] = 3
    return out

def compute_iou(gt_bin, pr_bin):
    g = gt_bin.astype(bool); p = pr_bin.astype(bool)
    intersection = (g & p).sum(); union = (g | p).sum()
    return float(intersection) / float(union) if union > 0 else 0.0

def draw_scalebar(ax, h, w):
    bar_px = int(round(SCALEBAR_UM / PIXEL_SIZE_UM))
    bar_px = max(5, min(bar_px, w - 20))
    x0, y0 = 10, h - 15
    ax.plot([x0, x0 + bar_px], [y0, y0], linewidth=5, color="white", solid_capstyle="butt")
    ax.text(x0 + bar_px/2, y0 - 10, f"{SCALEBAR_UM:.0f} µm",
            color="white", fontsize=10, ha="center", va="bottom", weight='bold')

def pick_centers_from_union(union3d, n, rng, patch_size, min_dist_px=180):
    coords = np.argwhere(union3d)
    if coords.size == 0: return []
    Z, H, W = union3d.shape
    picked = []
    tries = 0
    while len(picked) < n and tries < 200000:
        tries += 1
        z, y, x = coords[rng.integers(0, len(coords))]
        y0 = int(y - patch_size // 2); x0 = int(x - patch_size // 2)
        if y0 < 0 or x0 < 0 or (y0 + patch_size) > H or (x0 + patch_size) > W: continue
        ok = True
        for pz, py, px in picked:
            if pz == z and (py - y)**2 + (px - x)**2 < min_dist_px**2:
                ok = False; break
        if ok: picked.append((int(z), int(y), int(x)))
    return picked

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def bleach_correction_histogram_matching(volume):
    print("  → Bleach correction (histogram matching)...")
    corrected = np.zeros_like(volume, dtype=np.float32)
    reference = volume[0].astype(np.float32)
    corrected[0] = reference
    for z in range(1, volume.shape[0]):
        matched = exposure.match_histograms(volume[z], reference, channel_axis=None)
        corrected[z] = matched.astype(np.float32)
    return corrected

def rolling_ball_background_subtraction_3d(volume, radius):
    print(f"  → Background subtraction (rolling ball radius={radius})...")
    struct = disk(radius)
    corrected = np.zeros_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        slice_img = volume[z].astype(np.float32)
        background = ndimage.grey_opening(slice_img, structure=struct)
        diff = slice_img - background
        corrected[z] = np.clip(diff, 0, None)
    return corrected

def median_filter_3d(volume, radius=1):
    print(f"  → 3D Median filtering (radius={radius})...")
    kernel_size = 2 * radius + 1
    return ndimage.median_filter(volume, size=kernel_size)

def clahe_3d_slicewise(volume, blocksize=127, bins=256, clip_limit=3.0):
    print(f"  → CLAHE (blocksize={blocksize}, slope={clip_limit})...")
    enhanced = np.zeros_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        slice_img = volume[z]
        slice_min, slice_max = slice_img.min(), slice_img.max()
        if slice_max > slice_min:
            slice_norm = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_norm = np.zeros_like(slice_img, dtype=np.uint8)

        kernel_size = blocksize if blocksize % 2 == 1 else blocksize + 1
        clahe_result = exposure.equalize_adapthist(
            slice_norm, kernel_size=kernel_size, clip_limit=clip_limit/100.0, nbins=bins
        )
        enhanced[z] = clahe_result.astype(np.float32) * 255.0
    return enhanced

def intensity_normalization(volume):
    print("  → Intensity normalization...")
    p1, p99 = np.percentile(volume, [0.1, 99.9])
    normalized = (volume - p1) / max(p99 - p1, 1e-8)
    return np.clip(normalized, 0, 1) * 255.0

def apply_full_preprocessing_pipeline(volume):
    print("\n🔧 Aplicando pipeline de preprocessing (ORDEM OTIMIZADA)...")
    stages = {}
    stages['raw'] = volume.copy()

    # 1. Background subtraction (Primeiro, para limpar o offset)
    bg_subtracted = rolling_ball_background_subtraction_3d(volume, ROLLING_BALL_RADIUS)
    stages['background_subtracted'] = bg_subtracted

    # Debug print
    diff = volume - bg_subtracted
    print(f"     [DEBUG] Background removido (Avg): {diff.mean():.2f}, Max: {diff.max():.2f}")

    # 2. Bleach correction (Segundo, no sinal limpo)
    bleach_corrected = bleach_correction_histogram_matching(bg_subtracted)
    stages['bleach_corrected'] = bleach_corrected

    # 3. Median filter
    median_filtered = median_filter_3d(bleach_corrected, MEDIAN_FILTER_RADIUS)
    stages['median_filtered'] = median_filtered

    # 4. CLAHE
    clahe_enhanced = clahe_3d_slicewise(median_filtered, CLAHE_BLOCKSIZE, CLAHE_BINS, CLAHE_SLOPE)
    stages['clahe'] = clahe_enhanced

    # 5. Norm
    normalized = intensity_normalization(clahe_enhanced)
    stages['normalized'] = normalized.astype(np.float32)

    print("✓ Pipeline completo aplicado!\n")
    return stages

# ============================================================
# FIGURA DE DIAGNÓSTICO (CORRIGIDA)
# ============================================================

def save_diagnostic_figure(stages_dict, patch_row, out_png):
    z = int(patch_row["z"]); y0 = int(patch_row["y0"]); x0 = int(patch_row["x0"])
    ps = int(patch_row["patch_size"]); yc = y0 + ps // 2

    # Dados
    raw = stages_dict['raw']
    bg_sub = stages_dict['background_subtracted']
    bleach = stages_dict['bleach_corrected']
    median = stages_dict['median_filtered']
    final = stages_dict['normalized']

    fig = plt.figure(figsize=(22, 11))

    # 1. RAW
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(norm_to_uint8(raw[z, y0:y0+ps, x0:x0+ps]), cmap='gray')
    ax1.set_title("1. Raw Image (XY)", fontweight='bold')
    ax1.axis('off'); draw_scalebar(ax1, ps, ps)

    # 2. BLEACH CORRECTION (XZ View)
    # Comparar Antes (BG Subtracted) vs Depois (Bleach Corrected)
    before_bleach_xz = bg_sub[:, yc, x0:x0+ps]
    after_bleach_xz = bleach[:, yc, x0:x0+ps]

    ax2 = fig.add_subplot(2, 4, 2)
    combined_xz = np.vstack([norm_to_uint8(before_bleach_xz), norm_to_uint8(after_bleach_xz)])
    ax2.imshow(combined_xz, cmap='gray', aspect='auto')
    ax2.set_title("2. Bleach Correction (XZ)\nTop: Before | Bottom: After", fontweight='bold')
    ax2.axhline(before_bleach_xz.shape[0], color='red', linestyle='--', linewidth=1)
    ax2.axis('off')

    # 3. BACKGROUND (Raw - BgSubtracted)
    diff_bg = raw[z, y0:y0+ps, x0:x0+ps] - bg_sub[z, y0:y0+ps, x0:x0+ps]
    ax3 = fig.add_subplot(2, 4, 3)
    # Vmax robusto
    vmax_bg = np.percentile(diff_bg, 99.5) if diff_bg.max() > 0 else 1
    im3 = ax3.imshow(diff_bg, cmap='magma', vmin=0, vmax=vmax_bg)
    ax3.set_title(f"3. Background Removed\n(Raw - BG_Sub)", fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    # 4. NOISE REMOVED (Bleach - Median)
    # Comparar entrada do median com saída do median
    diff_noise = bleach[z, y0:y0+ps, x0:x0+ps] - median[z, y0:y0+ps, x0:x0+ps]
    ax4 = fig.add_subplot(2, 4, 4)
    limit_noise = max(abs(diff_noise.min()), abs(diff_noise.max()), 1e-5)
    limit_noise = max(limit_noise, 5.0)
    im4 = ax4.imshow(diff_noise, cmap='seismic', vmin=-limit_noise, vmax=limit_noise)
    ax4.set_title("4. Noise Removed\n(Bleach - Median)", fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    # 5 & 6. Zoom details
    zoom_s = ps // 2; mid = ps // 4
    crop_before = median[z, y0+mid:y0+mid+zoom_s, x0+mid:x0+mid+zoom_s]
    crop_after = final[z, y0+mid:y0+mid+zoom_s, x0+mid:x0+mid+zoom_s]

    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(norm_to_uint8(crop_before), cmap='gray')
    ax5.set_title("5. Before CLAHE (Zoom)", fontweight='bold'); ax5.axis('off')

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(norm_to_uint8(crop_after), cmap='gray')
    ax6.set_title("6. Final (CLAHE+Norm)", fontweight='bold'); ax6.axis('off')

    # 7. HISTOGRAM (Normalized for visualization)
    ax7 = fig.add_subplot(2, 4, (7, 8))
    h_raw = raw[z, y0:y0+ps, x0:x0+ps].flatten()
    h_final = final[z, y0:y0+ps, x0:x0+ps].flatten()

    # Normalizar Raw para 0-255 para o gráfico ficar bonito
    h_raw_display = (h_raw - h_raw.min()) / (h_raw.max() - h_raw.min()) * 255

    ax7.hist(h_raw_display, bins=100, color='gray', alpha=0.5, label='Raw Input (Scaled 0-255)', density=True)
    ax7.hist(h_final, bins=100, color='green', alpha=0.5, label='Final Output', density=True)
    ax7.set_title("Intensity Distribution (Aligned Ranges)", fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Diag: {os.path.basename(out_png)}")

# ============================================================
# SEGMENTATION & FIGURES MAIN
# ============================================================

def segment_cellpose_full_volume(raw_zyx, model_type, save_path_masks, save_path_flows):
    if not FORCE_RESEGMENT and os.path.exists(save_path_masks) and os.path.exists(save_path_flows):
        print(f"✓ Carregando {model_type}: {os.path.basename(save_path_masks)}")
        import pickle
        with open(save_path_flows, 'rb') as f: flows = pickle.load(f)
        return imread(save_path_masks), flows

    print(f"⚙ Segmentando {model_type}...")
    from cellpose.denoise import DenoiseModel
    dn = DenoiseModel(model_type="denoise_cyto3", gpu=CELLPOSE_GPU)
    imgs_denoised = dn.eval(raw_zyx, diameter=CELLPOSE_DIAMETER, channels=CELLPOSE_CHANNELS, z_axis=0)

    if model_type == "finetuned":
        model = cp_models.CellposeModel(gpu=CELLPOSE_GPU, pretrained_model=CELLPOSE_FT_MODEL_PATH)
    else:
        model = cp_models.Cellpose(gpu=CELLPOSE_GPU, model_type='cyto3')

    masks, flows, _ = model.eval(
        imgs_denoised, channels=CELLPOSE_CHANNELS, diameter=CELLPOSE_DIAMETER,
        flow_threshold=CELLPOSE_FLOW_THRESHOLD, cellprob_threshold=CELLPOSE_CELLPROB_THRESHOLD,
        do_3D=CELLPOSE_DO_3D, anisotropy=CELLPOSE_ANISOTROPY, z_axis=0
    )[0:3]

    mask_stack = masks.astype(np.uint16)
    imwrite(save_path_masks, mask_stack, compression='zlib')
    import pickle
    with open(save_path_flows, 'wb') as f: pickle.dump(flows, f)
    return mask_stack, flows

def save_figure_thesis_style(raw_zyx, models, patch_row, out_png):
    z = int(patch_row["z"]); y0 = int(patch_row["y0"]); x0 = int(patch_row["x0"])
    ps = int(patch_row["patch_size"]); yc = y0 + ps // 2; xc = x0 + ps // 2

    raw_xy = raw_zyx[z, y0:y0+ps, x0:x0+ps]
    raw_xz = raw_zyx[:, yc, x0:x0+ps]; raw_yz = raw_zyx[:, y0:y0+ps, xc]
    gt3 = models["GT"] > 0
    gt_xy = gt3[z, y0:y0+ps, x0:x0+ps]; gt_xz = gt3[:, yc, x0:x0+ps]; gt_yz = gt3[:, y0:y0+ps, xc]

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.25, left=0.04, right=0.98, top=0.94, bottom=0.04)

    def add_panel(ax, xy, xz, yz, title, overlay_xy=None, overlay_xz=None, overlay_yz=None, bar=False):
        ax.imshow(norm_to_uint8(xy), cmap='gray', interpolation='nearest')
        if overlay_xy is not None: ax.imshow(overlay_xy, cmap=CLASS_CMAP, vmin=0, vmax=3, interpolation='nearest')
        ax.set_title(title, fontsize=13, weight='bold', pad=8); ax.axis('off')

        ax_xz = ax.inset_axes([1.02, 0, 0.18, 1])
        ax_xz.imshow(norm_to_uint8(xz), cmap='gray', aspect='auto', interpolation='nearest')
        if overlay_xz is not None: ax_xz.imshow(overlay_xz, cmap=CLASS_CMAP, vmin=0, vmax=3, aspect='auto', interpolation='nearest')
        ax_xz.axis('off')

        ax_yz = ax.inset_axes([0, -0.20, 1, 0.18])
        ax_yz.imshow(norm_to_uint8(yz), cmap='gray', aspect='auto', interpolation='nearest')
        if overlay_yz is not None: ax_yz.imshow(overlay_yz, cmap=CLASS_CMAP, vmin=0, vmax=3, aspect='auto', interpolation='nearest')
        ax_yz.axis('off')
        if bar: draw_scalebar(ax, ps, ps)

    ax_raw = fig.add_subplot(gs[0, 0])
    add_panel(ax_raw, raw_xy, raw_xz, raw_yz, 'Preprocessed Input', bar=True)
    ax_gt = fig.add_subplot(gs[0, 1])
    add_panel(ax_gt, gt_xy, gt_xz, gt_yz, 'Ground Truth')

    if "Cellpose cyto3 ft" in models:
        pr3 = models["Cellpose cyto3 ft"] > 0
        pr_xy = pr3[z, y0:y0+ps, x0:x0+ps]
        pr_xz = pr3[:, yc, x0:x0+ps]
        pr_yz = pr3[:, y0:y0+ps, xc]
        ax = fig.add_subplot(gs[0, 2])
        add_panel(ax, raw_xy, raw_xz, raw_yz, f'Cellpose cyto3 ft (IoU={compute_iou(gt_xy, pr_xy):.2f})',
                  class_map(gt_xy, pr_xy), class_map(gt_xz, pr_xz), class_map(gt_yz, pr_yz))

    methods_row2 = ["StarDist3D", "SwinCell", "Cellpose cyto3", "Cellpose nuclei ft"]
    for col, m_name in enumerate(methods_row2):
        if m_name not in models: continue
        pr3 = models[m_name] > 0
        pr_xy = pr3[z, y0:y0+ps, x0:x0+ps]
        pr_xz = pr3[:, yc, x0:x0+ps]
        pr_yz = pr3[:, y0:y0+ps, xc]
        ax = fig.add_subplot(gs[1, col])
        add_panel(ax, raw_xy, raw_xz, raw_yz, f'{m_name} (IoU={compute_iou(gt_xy, pr_xy):.2f})',
                  class_map(gt_xy, pr_xy), class_map(gt_xz, pr_xz), class_map(gt_yz, pr_yz))

    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Fig: {os.path.basename(out_png)}")


# ============================================================
# ACRESCENTO PARA GERAR O PAINEL DOS FLOWS POR PATCH
# ============================================================

def _panel_norm01(img2d):
    x = img2d.astype(np.float32)
    p1, p99 = np.percentile(x, [1, 99])
    den = max(p99 - p1, 1e-8)
    x = (x - p1) / den
    return np.clip(x, 0, 1)

def _mask_to_rgb(mask2d, seed=7):
    m = mask2d.astype(np.int32)
    h, w = m.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    labels = np.unique(m)
    labels = labels[labels != 0]
    for lab in labels:
        rng = np.random.default_rng(seed + int(lab))
        col = 0.2 + 0.8 * rng.random(3)
        rgb[m == lab] = col
    return rgb

def _outlines_on_processed(processed2d, mask2d):
    base = np.dstack([processed2d, processed2d, processed2d])
    b = find_boundaries(mask2d > 0, mode="outer")
    base[b] = (1.0, 0.0, 0.0)
    return base

def _extract_dP_xy_patch_from_full_flows(flows_full, z, y0, x0, ps):
    f = flows_full[0] if isinstance(flows_full, (list, tuple)) else flows_full
    dP = np.asarray(f)

    if dP.ndim == 4 and dP.shape[0] >= 2:
        return dP[:2, z, y0:y0+ps, x0:x0+ps]

    if dP.ndim == 4 and dP.shape[-1] >= 2:
        dP_hw2 = dP[z, y0:y0+ps, x0:x0+ps, :2]
        return np.moveaxis(dP_hw2, -1, 0)

    if dP.ndim == 3 and dP.shape[0] >= 2:
        return dP[:2, y0:y0+ps, x0:x0+ps]

    raise ValueError(f"Formato inesperado de dP. Shape {dP.shape}")

def save_cellpose_flow_panels_for_patch(vol_pre, masks_cyto3, flows_cyto3, masks_ft, flows_ft, patch_row, out_png):
    from cellpose import plot as cp_plot

    z = int(patch_row["z"]); y0 = int(patch_row["y0"]); x0 = int(patch_row["x0"])
    ps = int(patch_row["patch_size"])

    proc = _panel_norm01(vol_pre[z, y0:y0+ps, x0:x0+ps])

    m_cyto3 = masks_cyto3[z, y0:y0+ps, x0:x0+ps]
    m_ft = masks_ft[z, y0:y0+ps, x0:x0+ps]

    dP_cyto3 = _extract_dP_xy_patch_from_full_flows(flows_cyto3, z, y0, x0, ps)
    dP_ft = _extract_dP_xy_patch_from_full_flows(flows_ft, z, y0, x0, ps)

    flow_rgb_cyto3 = cp_plot.dx_to_circ(dP_cyto3)
    flow_rgb_ft = cp_plot.dx_to_circ(dP_ft)

    flow_rgb_cyto3 = np.asarray(flow_rgb_cyto3)
    flow_rgb_ft = np.asarray(flow_rgb_ft)
    if flow_rgb_cyto3.dtype != np.float32 and flow_rgb_cyto3.dtype != np.float64:
        flow_rgb_cyto3 = flow_rgb_cyto3.astype(np.float32) / 255.0
    if flow_rgb_ft.dtype != np.float32 and flow_rgb_ft.dtype != np.float64:
        flow_rgb_ft = flow_rgb_ft.astype(np.float32) / 255.0

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)

    titles = ["Processed input", "Predicted outlines", "Predicted masks", "Predicted flows"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=12, fontweight="bold")

    rows = [
        ("cyto3", m_cyto3, flow_rgb_cyto3),
        ("cyto3 fine tuned", m_ft, flow_rgb_ft),
    ]

    for i, (rlabel, m2d, frgb) in enumerate(rows):
        axes[i, 0].imshow(proc, cmap="gray")
        axes[i, 1].imshow(_outlines_on_processed(proc, m2d))
        axes[i, 2].imshow(_mask_to_rgb(m2d, seed=PATCH_SEED))
        axes[i, 3].imshow(np.clip(frgb, 0, 1))

        axes[i, 0].set_ylabel(rlabel, fontsize=12, fontweight="bold")
        for j in range(4):
            axes[i, j].axis("off")

    draw_scalebar(axes[0, 0], ps, ps)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Flow panel: {os.path.basename(out_png)}")


def main():
    print("="*60 + "\nPROCESSAMENTO COMPLETO + DIAGNÓSTICO (PATCHES FIXOS)\n" + "="*60)
    raw = ensure_zyx_single_channel(imread(RAW_PATH).astype(np.float32))
    print(f"📂 Raw shape: {raw.shape}")

    # 1. Pipeline
    stages = apply_full_preprocessing_pipeline(raw)
    vol_pre = stages['normalized']

    # 2. Models
    print("🔬 Carregando modelos...")
    models = {"GT": imread(GT_PATH)}
    if os.path.exists(STAR_PATH): models["StarDist3D"] = imread(STAR_PATH)
    if os.path.exists(SWIN_PATH): models["SwinCell"] = imread(SWIN_PATH)
    if os.path.exists(CELLPOSE_NUCLEI_FT_PATH): models["Cellpose nuclei ft"] = imread(CELLPOSE_NUCLEI_FT_PATH)

    print("🧪 Segmentando...")
    models["Cellpose cyto3 ft"], _ = segment_cellpose_full_volume(
        vol_pre, "finetuned", CELLPOSE_CYTO3_FT_FULL_PATH, CELLPOSE_CYTO3_FT_FLOWS_PATH)
    models["Cellpose cyto3"], _ = segment_cellpose_full_volume(
        vol_pre, "cyto3", CELLPOSE_CYTO3_FULL_PATH, CELLPOSE_CYTO3_FLOWS_PATH)

    # 3. Patches (ALTERADO: USA APENAS GT PARA MANTER POSIÇÕES FIXAS)
    print("📍 Patches (baseados apenas no GT para consistência)...")
    union = (models["GT"] > 0) # <--- MUDANÇA IMPORTANTE AQUI

    rng = np.random.default_rng(PATCH_SEED)
    centers = pick_centers_from_union(union, N_EXAMPLES, rng, PATCH_SIZE, MIN_DIST_PX)
    rows = [{"example": i, "z": z, "y0": int(y - PATCH_SIZE//2), "x0": int(x - PATCH_SIZE//2), "patch_size": PATCH_SIZE}
            for i, (z, y, x) in enumerate(centers, 1)]
    df = pd.DataFrame(rows)
    df.to_csv(PATCH_META_CSV, index=False)

    # 4. Figures
    print("🎨 Gerando DIAGNÓSTICOS...")
    diag_dir = os.path.join(OUT_ROOT, "figures_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    for _, row in df.iterrows():
        save_diagnostic_figure(stages, row, os.path.join(diag_dir, f"diag_ex{row['example']:02d}.png"))

    print("🎨 Gerando FIGURES FINAIS...")
    fig_dir = os.path.join(OUT_ROOT, "figures_thesis_style")
    os.makedirs(fig_dir, exist_ok=True)
    for _, row in df.iterrows():
        save_figure_thesis_style(vol_pre, models, row, os.path.join(fig_dir, f"fig_ex{row['example']:02d}.png"))

    # 5. Acrescento, painéis de flows no mesmo conjunto de patches
    print("🎨 Gerando PAINÉIS DOS FLOWS (cyto3 e cyto3 fine tuned)...")
    flow_dir = os.path.join(OUT_ROOT, "figures_cellpose_flow_panels")
    os.makedirs(flow_dir, exist_ok=True)

    import pickle
    with open(CELLPOSE_CYTO3_FLOWS_PATH, "rb") as f:
        flows_cyto3_full = pickle.load(f)
    with open(CELLPOSE_CYTO3_FT_FLOWS_PATH, "rb") as f:
        flows_cyto3_ft_full = pickle.load(f)

    for _, row in df.iterrows():
        out_png = os.path.join(flow_dir, f"flowpanel_ex{row['example']:02d}.png")
        save_cellpose_flow_panels_for_patch(
            vol_pre,
            models["Cellpose cyto3"],
            flows_cyto3_full,
            models["Cellpose cyto3 ft"],
            flows_cyto3_ft_full,
            row,
            out_png
        )

    print("\n✅ FEITO!")

if __name__ == "__main__":
    main()
