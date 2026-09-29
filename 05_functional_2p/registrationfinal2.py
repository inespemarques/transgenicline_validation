"""
REGISTRATION V3 - OPTIMIZED FOR NOISY ANATOMY + PUBLICATION OVERLAYS
Ines Marques - v3.0

IMPROVEMENTS:
1. QA of anatomical alignment (DsRed vs GCaMP anatomy)
2. Enhanced preprocessing for ultra-noisy data (5 frames only)
3. More robust registration parameters
4. Green+Magenta overlays for publication
5. Comprehensive quality assessment
"""

import os
import re
import json
import numpy as np
from pathlib import Path

from tifffile import imread, imwrite, TiffFile
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from skimage.filters import gaussian, sobel, median, threshold_otsu
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation, disk
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, median_filter, percentile_filter
from skimage.restoration import denoise_tv_chambolle

import warnings
warnings.filterwarnings("ignore")


class Config:
    # ==================== PATHS ====================
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    GCAMP_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\Gcamp6s_averaged\alignment_drift\anatomy.tif"
    DSRED_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"
    OUT_DIR = r"C:\Users\OSVALDO\Downloads\REGISTRATION_V3_PUBLICATION3"

    # ==================== BASIC ====================
    N_PLANES = 180
    PLANE_PATTERN = r"aligned_p(\d+)_nan\.tif"
    PROJECTION_METHOD = "mean"
    FRAME_STRIDE = 2
    SAFE_TIFF_READ = True
    ALIGNMENT_X = "auto"

    # ==================== PREPROCESSING FOR NOISY DATA ====================
    # Mais agressivo para anatomias muito ruidosas (5 frames apenas)
    DENOISE_ANATOMY = True
    DENOISE_FUNCTIONAL = False  # Funcional já está bom
    
    # Multi-stage denoising
    DENOISE_STAGES = [
        {"method": "median", "kernel_size": 5},  # Remove salt&pepper
        {"method": "tv", "weight": 0.15},         # Total variation (preserva edges)
        {"method": "gaussian", "sigma": 1.5}      # Final smoothing
    ]
    
    # Background subtraction mais agressivo
    SUBTRACT_BACKGROUND = True
    BG_PERCENTILE = 5  # Mais baixo para anatomias dim
    BG_FILTER_SIZE = 50  # Maior para estimar melhor o background
    
    # Contrast enhancement
    ENHANCE_CONTRAST = True
    CLAHE_CLIP_LIMIT = 0.02
    
    # Intensity normalization
    NORMALIZE_INTENSITY = True
    NORM_PERCENTILE_LOW = 1
    NORM_PERCENTILE_HIGH = 99.5

    # ==================== REGISTRATION ====================
    USE_PIXEL_SPACE_FOR_REG = False
    SPACING_REAL = (0.5481, 0.5935, 1.0)
    
    # Masking - mais conservador para não perder sinal
    USE_BRAIN_MASK = True
    MASK_METHOD = "adaptive"
    MASK_PERCENTILE_LOW = 5  # Mais baixo
    MASK_PERCENTILE_HIGH = 15
    MASK_MIN_SIZE = 1000  # Menor
    MASK_DILATE_ITERS = 5  # Mais para capturar tudo
    MASK_ERODE_FIRST = True
    MASK_ERODE_ITERS = 2
    
    # Registration - parâmetros mais robustos
    METRIC = "MI"  # Mutual Information melhor para intensidades diferentes
    METRIC_SAMPLING = 0.40  # Mais samples
    
    # Multi-resolution mais suave
    SHRINK_FACTORS = [12, 8, 4, 2, 1]  # 5 níveis
    SMOOTHING_SIGMAS = [4.0, 3.0, 2.0, 1.0, 0.0]
    
    # Mais iterações
    RIGID_ITER = 1200
    RIGID_LEARNING_RATE = 1.2
    
    AFFINE_ITER = 1000
    AFFINE_LEARNING_RATE = 0.3  # Mais baixo para convergência fina
    
    USE_BSPLINE = False  # Manter False para preservar ROIs
    
    GRAD_SIGMA = 1.5  # Mais smoothing no gradiente
    
    # ==================== VISUALIZATION ====================
    # Green + Magenta para publicação (melhor que green+red)
    COLOR_FIXED = "green"      # Funcional
    COLOR_MOVING = "magenta"   # Anatomia registada
    
    PLANES_TO_SHOW = [30, 60, 90, 120, 150]
    DPI_QA = 300  # Alta resolução para publicação
    
    # ==================== QUALITY CONTROL ====================
    MIN_ACCEPTABLE_NCC = 0.50
    MIN_ACCEPTABLE_MI = 0.40  # Para mutual information
    AUTO_RETRY_IF_BAD = True

    # ==================== LANDMARK-STYLE QA (AUTOMATIC) ====================
    # Detect bright "cell-like" peaks in functional and GCaMP anatomy and quantify nearest-neighbor distances.
    USE_SPOT_QA = True
    SPOT_MIN_DISTANCE_PX = 4
    SPOT_THRESH_PERCENTILE = 99.5
    SPOTS_PER_PLANE = 250
    QA_PLANES_FOR_SPOTS = [30, 60, 90, 120, 150]
    QA_SPOT_MAX_NN_DIST_PX = 50.0  # ignore obviously wrong matches when summarizing


cfg = Config()

out_root = Path(cfg.OUT_DIR)
for subdir in ["volumes", "transforms", "metrics", "QA_functional_vs_anatomy", 
               "QA_anatomy_vs_anatomy", "preprocessing_qa", "publication_figures"]:
    (out_root / subdir).mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_zyx(arr):
    """Normalize array to ZYX format"""
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0] if a.shape[0] in (2, 3, 4) else a[..., 0]
    if a.ndim == 2:
        a = a[None, ...]
    return a.astype(np.float32)


def numpy_to_sitk(arr_zyx, spacing_xyz):
    """Convert numpy array to SimpleITK image"""
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.SetSpacing(tuple(float(x) for x in spacing_xyz))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def sitk_to_numpy(img):
    """Convert SimpleITK image to numpy array"""
    return sitk.GetArrayFromImage(img).astype(np.float32)


def enhance_for_display(img2d, p_low=1, p_high=99.5):
    """Enhance image for visualization"""
    x = img2d.astype(np.float32)
    nz = x[x > 0]
    if nz.size == 0:
        return x / (x.max() + 1e-8)
    vmin, vmax = np.percentile(nz, [p_low, p_high])
    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)


def detect_spots_in_plane(img2d, min_distance_px, thresh_percentile, max_spots):
    """
    Return Nx2 array of (y, x) peak coordinates for bright local maxima.
    """
    x = img2d.astype(np.float32)
    nz = x[x > 0]
    if nz.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    thr = float(np.percentile(nz, float(thresh_percentile)))
    # Light smoothing makes peaks more stable across modalities.
    xs = gaussian(x, sigma=1.0, preserve_range=True)
    coords = peak_local_max(
        xs,
        min_distance=int(min_distance_px),
        threshold_abs=thr,
        num_peaks=int(max_spots),
        exclude_border=False,
    )
    # peak_local_max returns (row, col)
    return coords.astype(np.int32)


def nn_stats(fixed_yx, moving_yx, max_dist=np.inf):
    """
    Nearest-neighbor stats (in pixels). moving -> nearest fixed.
    """
    if fixed_yx.shape[0] == 0 or moving_yx.shape[0] == 0:
        return {"n_fixed": int(fixed_yx.shape[0]), "n_moving": int(moving_yx.shape[0])}
    tree = cKDTree(fixed_yx.astype(np.float32))
    d, _ = tree.query(moving_yx.astype(np.float32), k=1, workers=-1)
    d = np.asarray(d, dtype=np.float32)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n_fixed": int(fixed_yx.shape[0]), "n_moving": int(moving_yx.shape[0])}
    if np.isfinite(max_dist):
        d = d[d <= float(max_dist)]
    if d.size == 0:
        return {"n_fixed": int(fixed_yx.shape[0]), "n_moving": int(moving_yx.shape[0])}
    return {
        "n_fixed": int(fixed_yx.shape[0]),
        "n_moving": int(moving_yx.shape[0]),
        "mean_nn_px": float(np.mean(d)),
        "median_nn_px": float(np.median(d)),
        "p90_nn_px": float(np.percentile(d, 90)),
    }


def spot_qa_report(func_vol, moving_before, moving_after, out_dir, planes):
    """
    Automatic "landmark" QA:
      - detect bright spots in each plane for fixed and moving volumes
      - compute moving->fixed nearest-neighbor distance before/after
      - save a JSON report + quick overlay figures with spot markers
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"planes": {}, "summary": {}}

    all_before = []
    all_after = []

    for z in planes:
        if z < 0 or z >= func_vol.shape[0]:
            continue
        fixed2d = func_vol[z]
        mb2d = moving_before[z]
        ma2d = moving_after[z]

        fixed_pts = detect_spots_in_plane(
            fixed2d, cfg.SPOT_MIN_DISTANCE_PX, cfg.SPOT_THRESH_PERCENTILE, cfg.SPOTS_PER_PLANE
        )
        mb_pts = detect_spots_in_plane(
            mb2d, cfg.SPOT_MIN_DISTANCE_PX, cfg.SPOT_THRESH_PERCENTILE, cfg.SPOTS_PER_PLANE
        )
        ma_pts = detect_spots_in_plane(
            ma2d, cfg.SPOT_MIN_DISTANCE_PX, cfg.SPOT_THRESH_PERCENTILE, cfg.SPOTS_PER_PLANE
        )

        st_before = nn_stats(fixed_pts, mb_pts, max_dist=cfg.QA_SPOT_MAX_NN_DIST_PX)
        st_after = nn_stats(fixed_pts, ma_pts, max_dist=cfg.QA_SPOT_MAX_NN_DIST_PX)
        report["planes"][str(z)] = {
            "fixed_spots": int(fixed_pts.shape[0]),
            "moving_before_spots": int(mb_pts.shape[0]),
            "moving_after_spots": int(ma_pts.shape[0]),
            "nn_before": st_before,
            "nn_after": st_after,
        }

        if "median_nn_px" in st_before:
            all_before.append(st_before["median_nn_px"])
        if "median_nn_px" in st_after:
            all_after.append(st_after["median_nn_px"])

        # Save plane overlay with spot markers for visual confirmation
        f = enhance_for_display(fixed2d, 1, 99)
        m = enhance_for_display(ma2d, 1, 99)
        rgb = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
        rgb[..., 1] = f
        rgb[..., 0] = m
        rgb[..., 2] = m

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(rgb, interpolation="nearest")
        # fixed spots in green circles, moving-after spots in magenta x
        if fixed_pts.shape[0] > 0:
            ax.scatter(fixed_pts[:, 1], fixed_pts[:, 0], s=8, facecolors="none", edgecolors="lime", linewidths=0.7)
        if ma_pts.shape[0] > 0:
            ax.scatter(ma_pts[:, 1], ma_pts[:, 0], s=10, c="magenta", marker="x", linewidths=0.8)
        ax.set_title(f"Spot QA z={z} (green circles=fixed, magenta x=moving after)", fontsize=12, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"spot_qa_overlay_z{z:03d}.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    if all_before and all_after:
        report["summary"] = {
            "median_nn_px_before_median_over_planes": float(np.median(np.array(all_before))),
            "median_nn_px_after_median_over_planes": float(np.median(np.array(all_after))),
            "improvement_px": float(np.median(np.array(all_before)) - np.median(np.array(all_after))),
        }

    with open(out_dir / "spot_qa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# ============================================================================
# ENHANCED PREPROCESSING FOR NOISY DATA
# ============================================================================

def denoise_volume_multistage(vol, stages):
    """
    Multi-stage denoising pipeline for ultra-noisy data
    """
    print(f"  Multi-stage denoising ({len(stages)} stages)")
    vol_denoised = vol.copy()
    
    for i, stage in enumerate(stages):
        method = stage["method"]
        print(f"    Stage {i+1}: {method}")
        
        if method == "median":
            kernel_size = stage.get("kernel_size", 3)
            vol_next = np.zeros_like(vol_denoised)
            for z in range(vol_denoised.shape[0]):
                vol_next[z] = median_filter(vol_denoised[z], size=kernel_size)
            vol_denoised = vol_next
        
        elif method == "tv":
            # Total Variation denoising - preserva edges
            weight = stage.get("weight", 0.1)
            vol_next = np.zeros_like(vol_denoised)
            for z in range(vol_denoised.shape[0]):
                plane = vol_denoised[z]
                if plane.max() > 0:
                    plane_norm = plane / plane.max()
                    vol_next[z] = denoise_tv_chambolle(plane_norm, weight=weight) * plane.max()
                else:
                    vol_next[z] = plane
            vol_denoised = vol_next
        
        elif method == "gaussian":
            sigma = stage.get("sigma", 1.0)
            vol_next = np.zeros_like(vol_denoised)
            for z in range(vol_denoised.shape[0]):
                vol_next[z] = gaussian(vol_denoised[z], sigma=sigma, preserve_range=True)
            vol_denoised = vol_next
        
        elif method == "bilateral":
            from skimage.restoration import denoise_bilateral
            vol_next = np.zeros_like(vol_denoised)
            for z in range(vol_denoised.shape[0]):
                vol_next[z] = denoise_bilateral(
                    vol_denoised[z],
                    sigma_color=0.05,
                    sigma_spatial=3,
                    channel_axis=None
                )
            vol_denoised = vol_next
    
    return vol_denoised


def subtract_background_aggressive(vol, percentile=5, filter_size=50):
    """
    More aggressive background subtraction for dim anatomies
    """
    print(f"  Aggressive background subtraction (p={percentile}, filter={filter_size})")
    bg_subtracted = np.zeros_like(vol, dtype=np.float32)
    
    for z in range(vol.shape[0]):
        plane = vol[z]
        # Estimate background with larger filter
        bg = percentile_filter(plane, percentile=percentile, size=filter_size)
        bg_subtracted[z] = np.maximum(plane - bg, 0)
    
    return bg_subtracted


def normalize_intensity_range(vol, p_low=1, p_high=99.5):
    """
    Normalize intensity to use full dynamic range
    """
    print(f"  Normalizing intensity (p{p_low}-p{p_high})")
    nz = vol[vol > 0]
    if nz.size == 0:
        return vol
    
    vmin, vmax = np.percentile(nz, [p_low, p_high])
    vol_norm = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return vol_norm


def preprocess_volume_enhanced(vol, denoise=True, bg_subtract=True, 
                               enhance=True, normalize=True, is_anatomy=True):
    """
    Enhanced preprocessing pipeline for noisy anatomical data
    """
    print(f"\nEnhanced preprocessing ({'ANATOMY' if is_anatomy else 'FUNCTIONAL'}):")
    
    vol_proc = vol.copy()
    
    # Background subtraction first (more aggressive for anatomy)
    if bg_subtract and cfg.SUBTRACT_BACKGROUND:
        if is_anatomy:
            vol_proc = subtract_background_aggressive(
                vol_proc, cfg.BG_PERCENTILE, cfg.BG_FILTER_SIZE
            )
        else:
            vol_proc = subtract_background_aggressive(vol_proc, 10, 30)
    
    # Multi-stage denoising (only for anatomy)
    if denoise and is_anatomy and cfg.DENOISE_ANATOMY:
        vol_proc = denoise_volume_multistage(vol_proc, cfg.DENOISE_STAGES)
    elif denoise and not is_anatomy and cfg.DENOISE_FUNCTIONAL:
        # Light denoising for functional
        vol_proc = denoise_volume_multistage(vol_proc, [{"method": "gaussian", "sigma": 0.8}])
    
    # Normalize intensity
    if normalize and cfg.NORMALIZE_INTENSITY:
        vol_proc = normalize_intensity_range(
            vol_proc, cfg.NORM_PERCENTILE_LOW, cfg.NORM_PERCENTILE_HIGH
        )
    
    # CLAHE contrast enhancement
    if enhance and cfg.ENHANCE_CONTRAST:
        from skimage.exposure import equalize_adapthist
        enhanced = np.zeros_like(vol_proc)
        for z in range(vol_proc.shape[0]):
            plane = vol_proc[z]
            if plane.max() > 0:
                enhanced[z] = equalize_adapthist(plane, clip_limit=cfg.CLAHE_CLIP_LIMIT)
            else:
                enhanced[z] = plane
        vol_proc = enhanced
    
    return vol_proc


def save_preprocessing_comparison(original, processed, out_path, z_planes, title):
    """
    Save multi-plane preprocessing comparison
    """
    n_planes = len(z_planes)
    fig, axes = plt.subplots(n_planes, 2, figsize=(14, 4*n_planes))
    
    if n_planes == 1:
        axes = axes.reshape(1, -1)
    
    for i, z in enumerate(z_planes):
        # Original
        axes[i, 0].imshow(enhance_for_display(original[z]), cmap='gray')
        axes[i, 0].set_title(f'Original z={z}', fontsize=11)
        axes[i, 0].axis('off')
        
        # Processed
        axes[i, 1].imshow(enhance_for_display(processed[z]), cmap='gray')
        axes[i, 1].set_title(f'Preprocessed z={z}', fontsize=11)
        axes[i, 1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()


# ============================================================================
# MASKING
# ============================================================================

def create_brain_mask_adaptive(volume, method="adaptive"):
    """
    Create brain mask - more conservative for noisy data
    """
    print(f"\nCreating brain mask (method={method})")
    
    v = volume.astype(np.float32)
    mask_3d = np.zeros_like(v, dtype=bool)
    
    for z in range(v.shape[0]):
        plane = v[z]
        pos = plane[plane > 0]
        
        if pos.size == 0:
            continue
        
        # Adaptive threshold
        thresh_low = np.percentile(pos, cfg.MASK_PERCENTILE_LOW)
        thresh_high = np.percentile(pos, cfg.MASK_PERCENTILE_HIGH)
        
        mask_low = plane > thresh_low
        mask_high = plane > thresh_high
        
        # Clean high threshold mask
        mask_high = remove_small_objects(mask_high, min_size=cfg.MASK_MIN_SIZE // 2)
        mask_high = binary_fill_holes(mask_high)
        
        # Expand to capture all tissue
        mask = mask_high.copy()
        
        # Erode first to remove noise
        if cfg.MASK_ERODE_FIRST:
            for _ in range(cfg.MASK_ERODE_ITERS):
                mask = binary_erosion(mask)
        
        # Then dilate significantly
        for _ in range(cfg.MASK_DILATE_ITERS + cfg.MASK_ERODE_ITERS):
            mask = binary_dilation(mask)
        
        # Intersect with low threshold
        mask = np.logical_and(mask, mask_low)
        
        # Final cleanup
        mask = binary_fill_holes(mask)
        mask = remove_small_objects(mask, min_size=cfg.MASK_MIN_SIZE)
        
        mask_3d[z] = mask
    
    coverage = 100 * mask_3d.sum() / mask_3d.size
    print(f"  Mask coverage: {coverage:.1f}%")
    
    if coverage < 5:
        print("  ⚠️  WARNING: Very low mask coverage - check data quality")
    
    return mask_3d.astype(np.uint8)


# ============================================================================
# ALIGNMENT
# ============================================================================

def _start_index(length_in, length_out, mode):
    if length_in >= length_out:
        if mode == "left":
            return 0
        if mode == "right":
            return length_in - length_out
        return (length_in - length_out) // 2
    return 0


def _paste_index(length_out, length_in, mode):
    if length_out >= length_in:
        if mode == "left":
            return 0
        if mode == "right":
            return length_out - length_in
        return (length_out - length_in) // 2
    return 0


def fit_moving_to_fixed_keep_fixed(fixed_np, moving_np, alignment_x="center"):
    """Fit moving volume into fixed volume space"""
    zf, yf, xf = fixed_np.shape
    zm, ym, xm = moving_np.shape

    z0m = _start_index(zm, zf, "center")
    y0m = _start_index(ym, yf, "center")
    x0m = _start_index(xm, xf, alignment_x)

    z1m = z0m + min(zm, zf)
    y1m = y0m + min(ym, yf)
    x1m = x0m + min(xm, xf)

    moving_crop = moving_np[z0m:z1m, y0m:y1m, x0m:x1m]

    zlen, ylen, xlen = moving_crop.shape
    z0f = _paste_index(zf, zlen, "center")
    y0f = _paste_index(yf, ylen, "center")
    x0f = _paste_index(xf, xlen, alignment_x)

    moving_out = np.zeros_like(fixed_np, dtype=np.float32)
    moving_out[z0f:z0f + zlen, y0f:y0f + ylen, x0f:x0f + xlen] = moving_crop.astype(np.float32)
    return moving_out


def edges_np(vol, sigma=1.5):
    """Compute edge magnitude"""
    out = np.zeros_like(vol, dtype=np.float32)
    for z in range(vol.shape[0]):
        out[z] = sobel(gaussian(vol[z], sigma=sigma, preserve_range=True))
    return out


def ncc(a, b):
    """Normalized cross-correlation"""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8)
    return num / den


def mutual_information(a, b, bins=50):
    """Compute mutual information"""
    hist_2d, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def pick_best_alignment_x(fixed_np, moving_np_raw):
    """Automatically determine best X alignment"""
    print("\nDetermining best X-axis alignment:")
    options = ["left", "center", "right"]
    ef = edges_np(fixed_np, sigma=cfg.GRAD_SIGMA)
    best = None
    best_score = -1e9
    results = {}
    
    for mode in options:
        fit = fit_moving_to_fixed_keep_fixed(fixed_np, moving_np_raw, alignment_x=mode)
        em = edges_np(fit, sigma=cfg.GRAD_SIGMA)
        
        score_ncc = ncc(ef, em)
        score_mi = mutual_information(fixed_np, fit)
        
        # Combined score
        score = 0.6 * score_ncc + 0.4 * score_mi
        
        results[mode] = {"ncc": score_ncc, "mi": score_mi, "combined": score}
        print(f"  {mode:>6s}: NCC={score_ncc:.4f}, MI={score_mi:.4f}, Combined={score:.4f}")
        
        if score > best_score:
            best_score = score
            best = mode
    
    print(f"  → Best: {best} (Combined={best_score:.4f})")
    return best, results


# ============================================================================
# FUNCTIONAL TEMPLATE
# ============================================================================

def list_plane_files(tifs_dir, pattern_regex):
    """List plane TIFF files"""
    tifs_path = Path(tifs_dir)
    pattern = re.compile(pattern_regex, re.IGNORECASE)
    plane_files = {}
    
    for f in os.listdir(tifs_path):
        m = pattern.match(f)
        if m:
            plane_idx = int(m.group(1)) - 1
            plane_files[plane_idx] = tifs_path / f
    
    if len(plane_files) == 0:
        raise ValueError("No plane TIFs found")
    
    return plane_files


def project_plane_tif(path, method="mean", frame_stride=1):
    """Project plane TIFF"""
    with TiffFile(str(path)) as tif:
        n_pages = len(tif.pages)
        idxs = range(0, n_pages, int(frame_stride))
        bad = 0

        if method == "mean":
            acc = None
            count = 0
            for i in idxs:
                try:
                    fr = tif.pages[i].asarray().astype(np.float32)
                except Exception:
                    if cfg.SAFE_TIFF_READ:
                        bad += 1
                        continue
                    raise
                if acc is None:
                    acc = np.zeros_like(fr, dtype=np.float32)
                acc += fr
                count += 1
            
            if count == 0:
                raise RuntimeError(f"All frames failed in {path}")
            if bad > 0:
                print(f"  WARNING: {bad} bad frames in {Path(path).name}")
            return acc / max(count, 1)

        elif method == "median":
            frames = []
            for i in idxs:
                try:
                    fr = tif.pages[i].asarray().astype(np.float32)
                except Exception:
                    if cfg.SAFE_TIFF_READ:
                        bad += 1
                        continue
                    raise
                frames.append(fr)
            
            if len(frames) == 0:
                raise RuntimeError(f"All frames failed in {path}")
            if bad > 0:
                print(f"  WARNING: {bad} bad frames in {Path(path).name}")
            
            stack = np.stack(frames, axis=0)
            return np.median(stack, axis=0).astype(np.float32)
        
        raise ValueError("Unknown projection method")


def build_functional_template(tifs_dir, n_planes, method, frame_stride, cache_path):
    """Build functional template from TIFFs"""
    if Path(cache_path).exists():
        vol = normalize_zyx(imread(str(cache_path)))
        print(f"Loaded cached template: {cache_path}")
        return vol

    print(f"Building functional template (method={method}, stride={frame_stride})")
    plane_files = list_plane_files(tifs_dir, cfg.PLANE_PATTERN)

    first_plane = min(plane_files.keys())
    first_proj = project_plane_tif(plane_files[first_plane], method=method, frame_stride=frame_stride)
    h, w = first_proj.shape

    stack = np.zeros((n_planes, h, w), dtype=np.float32)

    loaded = 0
    for plane_idx in sorted(plane_files.keys()):
        if plane_idx >= n_planes:
            continue
        
        proj = project_plane_tif(plane_files[plane_idx], method=method, frame_stride=frame_stride)
        stack[plane_idx] = proj.astype(np.float32)
        loaded += 1
        
        if loaded % 30 == 0:
            print(f"  Loaded {loaded}/{n_planes} planes")

    print(f"  Template shape: {stack.shape}")
    imwrite(str(cache_path), stack.astype(np.float32))
    print(f"  Cached: {cache_path}")

    return stack


# ============================================================================
# REGISTRATION
# ============================================================================

def gradient_volume_sitk(img_sitk, sigma):
    """Compute gradient magnitude"""
    return sitk.GradientMagnitudeRecursiveGaussian(img_sitk, float(sigma))


def make_registration_method(metric_sampling, n_iter, shrink_factors, 
                            smoothing_sigmas, learning_rate=1.0):
    """Create registration method"""
    reg = sitk.ImageRegistrationMethod()

    if cfg.METRIC.upper() == "MI":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    else:
        reg.SetMetricAsCorrelation()
    
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling))
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=float(learning_rate),
        numberOfIterations=int(n_iter),
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=15
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([int(x) for x in shrink_factors])
    reg.SetSmoothingSigmasPerLevel([float(x) for x in smoothing_sigmas])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg


def register_rigid_then_affine(fixed_struct, moving_struct, mask_fixed=None):
    """Rigid + Affine registration"""
    print("\n" + "="*80)
    print("RIGID REGISTRATION")
    print("="*80)
    
    init_rigid = sitk.CenteredTransformInitializer(
        fixed_struct, moving_struct, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg_rigid = make_registration_method(
        cfg.METRIC_SAMPLING, cfg.RIGID_ITER, cfg.SHRINK_FACTORS, 
        cfg.SMOOTHING_SIGMAS, learning_rate=cfg.RIGID_LEARNING_RATE
    )
    reg_rigid.SetInitialTransform(init_rigid, inPlace=True)
    if mask_fixed is not None:
        reg_rigid.SetMetricFixedMask(mask_fixed)

    tx_rigid = reg_rigid.Execute(fixed_struct, moving_struct)
    metric_rigid = reg_rigid.GetMetricValue()
    print(f"  Final metric: {metric_rigid:.6f}")
    print(f"  Iterations: {reg_rigid.GetOptimizerIteration()}")
    print(f"  Stop: {reg_rigid.GetOptimizerStopConditionDescription()}")

    print("\n" + "="*80)
    print("AFFINE REGISTRATION")
    print("="*80)
    
    init_aff = sitk.AffineTransform(3)
    init_aff.SetCenter(tx_rigid.GetCenter())
    init_aff.SetMatrix((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    init_aff.SetTranslation((0.0, 0.0, 0.0))

    reg_aff = make_registration_method(
        cfg.METRIC_SAMPLING, cfg.AFFINE_ITER, cfg.SHRINK_FACTORS, 
        cfg.SMOOTHING_SIGMAS, learning_rate=cfg.AFFINE_LEARNING_RATE
    )
    reg_aff.SetMovingInitialTransform(tx_rigid)
    reg_aff.SetInitialTransform(init_aff, inPlace=True)
    if mask_fixed is not None:
        reg_aff.SetMetricFixedMask(mask_fixed)

    tx_aff = reg_aff.Execute(fixed_struct, moving_struct)
    metric_aff = reg_aff.GetMetricValue()
    print(f"  Final metric: {metric_aff:.6f}")
    print(f"  Iterations: {reg_aff.GetOptimizerIteration()}")
    print(f"  Stop: {reg_aff.GetOptimizerStopConditionDescription()}")

    return tx_rigid, tx_aff, metric_rigid, metric_aff


def compose_transforms(tx_rigid, tx_aff):
    """Compose transforms"""
    tx = sitk.CompositeTransform(3)
    tx.AddTransform(tx_rigid)
    tx.AddTransform(tx_aff)
    return tx


# ============================================================================
# METRICS
# ============================================================================

def compute_comprehensive_metrics(fixed, moving_before, moving_after):
    """Compute comprehensive quality metrics"""
    metrics = {}
    
    # Edge-based metrics
    ef = edges_np(fixed, sigma=cfg.GRAD_SIGMA)
    eb = edges_np(moving_before, sigma=cfg.GRAD_SIGMA)
    ea = edges_np(moving_after, sigma=cfg.GRAD_SIGMA)
    
    metrics["edge_ncc_before"] = ncc(ef, eb)
    metrics["edge_ncc_after"] = ncc(ef, ea)
    metrics["edge_ncc_improvement"] = metrics["edge_ncc_after"] - metrics["edge_ncc_before"]
    
    # Intensity-based
    metrics["intensity_ncc_before"] = ncc(fixed, moving_before)
    metrics["intensity_ncc_after"] = ncc(fixed, moving_after)
    metrics["intensity_ncc_improvement"] = metrics["intensity_ncc_after"] - metrics["intensity_ncc_before"]
    
    # Mutual information
    metrics["mi_before"] = mutual_information(fixed, moving_before)
    metrics["mi_after"] = mutual_information(fixed, moving_after)
    metrics["mi_improvement"] = metrics["mi_after"] - metrics["mi_before"]
    
    # Quality assessment
    quality_score = (metrics["edge_ncc_after"] + metrics["intensity_ncc_after"] + 
                     0.5 * metrics["mi_after"]) / 2.5
    
    if quality_score > 0.70:
        quality = "excellent"
    elif quality_score > 0.55:
        quality = "good"
    elif quality_score > 0.40:
        quality = "acceptable"
    else:
        quality = "poor"
    
    metrics["summary"] = {
        "quality": quality,
        "quality_score": quality_score,
        "meets_threshold": metrics["edge_ncc_after"] >= cfg.MIN_ACCEPTABLE_NCC,
        "registration_type": "rigid_affine_only",
        "roi_shapes_preserved": True
    }
    
    return metrics


# ============================================================================
# PUBLICATION-READY OVERLAYS (GREEN + MAGENTA)
# ============================================================================

def create_green_magenta_colormap():
    """Create custom colormaps for publication"""
    # Green colormap
    green_cmap = LinearSegmentedColormap.from_list(
        'green', [(0, 0, 0), (0, 1, 0)], N=256
    )
    
    # Magenta colormap
    magenta_cmap = LinearSegmentedColormap.from_list(
        'magenta', [(0, 0, 0), (1, 0, 1)], N=256
    )
    
    return green_cmap, magenta_cmap


def save_publication_overlay(fixed2d, moving2d, out_png, title, dpi=300):
    """
    Save publication-ready overlay: Green (fixed) + Magenta (moving)
    """
    f = enhance_for_display(fixed2d, p_low=1, p_high=99)
    m = enhance_for_display(moving2d, p_low=1, p_high=99)
    
    # Create RGB overlay: Green + Magenta
    rgb = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = f  # Green channel (fixed/functional)
    rgb[..., 0] = m  # Red channel (for magenta)
    rgb[..., 2] = m  # Blue channel (for magenta)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis("off")
    
    # Tight layout
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches='tight', facecolor='black')
    plt.close(fig)


def save_publication_sidebyside(fixed2d, moving2d, out_png, title, dpi=300,
                                label_left='Functional (target)', 
                                label_right='Anatomy registered'):
    """
    Save side-by-side with green and magenta
    Can specify custom labels for left and right panels
    """
    f = enhance_for_display(fixed2d, p_low=1, p_high=99)
    m = enhance_for_display(moving2d, p_low=1, p_high=99)
    
    green_cmap, magenta_cmap = create_green_magenta_colormap()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Left panel (green)
    axes[0].imshow(f, cmap=green_cmap, interpolation='bilinear')
    axes[0].set_title(label_left, fontsize=13, fontweight='bold', color='white')
    axes[0].axis('off')
    
    # Right panel (magenta)
    axes[1].imshow(m, cmap=magenta_cmap, interpolation='bilinear')
    axes[1].set_title(label_right, fontsize=13, fontweight='bold', color='white')
    axes[1].axis('off')
    
    fig.suptitle(title, fontsize=15, fontweight='bold', color='white')
    fig.patch.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches='tight', facecolor='black')
    plt.close(fig)


def save_publication_triple(fixed2d, moving2d, out_png, title, dpi=300):
    """
    Triple panel: Fixed | Moving | Overlay
    """
    f = enhance_for_display(fixed2d, p_low=1, p_high=99)
    m = enhance_for_display(moving2d, p_low=1, p_high=99)
    
    # Create overlay
    rgb = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = f  # Green
    rgb[..., 0] = m  # Red (magenta)
    rgb[..., 2] = m  # Blue (magenta)
    
    green_cmap, magenta_cmap = create_green_magenta_colormap()
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(f, cmap=green_cmap, interpolation='bilinear')
    axes[0].set_title('Functional', fontsize=12, fontweight='bold', color='white')
    axes[0].axis('off')
    
    axes[1].imshow(m, cmap=magenta_cmap, interpolation='bilinear')
    axes[1].set_title('Anatomy', fontsize=12, fontweight='bold', color='white')
    axes[1].axis('off')
    
    axes[2].imshow(rgb, interpolation='bilinear')
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold', color='white')
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches='tight', facecolor='black')
    plt.close(fig)


# ============================================================================
# QA: ANATOMY vs ANATOMY
# ============================================================================

def qa_anatomy_alignment(gcamp_anat, dsred_anat, out_dir, z_planes):
    """
    Check if GCaMP and DsRed anatomies are already aligned
    """
    print("\n" + "="*80)
    print("QA: ANATOMY vs ANATOMY ALIGNMENT")
    print("="*80)
    
    # Compute metrics
    ncc_val = ncc(gcamp_anat, dsred_anat)
    mi_val = mutual_information(gcamp_anat, dsred_anat)
    
    edge_gcamp = edges_np(gcamp_anat, sigma=cfg.GRAD_SIGMA)
    edge_dsred = edges_np(dsred_anat, sigma=cfg.GRAD_SIGMA)
    ncc_edge = ncc(edge_gcamp, edge_dsred)
    
    print(f"  Intensity NCC: {ncc_val:.4f}")
    print(f"  Edge NCC: {ncc_edge:.4f}")
    print(f"  Mutual Information: {mi_val:.4f}")
    
    if ncc_edge > 0.75:
        alignment_status = "EXCELLENT - Anatomies well aligned"
    elif ncc_edge > 0.60:
        alignment_status = "GOOD - Minor misalignment"
    elif ncc_edge > 0.45:
        alignment_status = "ACCEPTABLE - Some misalignment present"
    else:
        alignment_status = "POOR - Significant misalignment"
    
    print(f"  Status: {alignment_status}")
    
    # Save overlays
    for z in z_planes:
        if z >= gcamp_anat.shape[0]:
            continue
        
        # Green+Magenta overlay
        save_publication_overlay(
            gcamp_anat[z], dsred_anat[z],
            out_dir / f"anatomy_overlay_gcamp_dsred_z{z:03d}.png",
            f"Anatomy Check: GCaMP (green) vs DsRed (magenta) | z={z}",
            dpi=cfg.DPI_QA
        )
        
        # Side by side with CORRECT labels
        save_publication_sidebyside(
            gcamp_anat[z], dsred_anat[z],
            out_dir / f"anatomy_sidebyside_gcamp_dsred_z{z:03d}.png",
            f"Anatomy Alignment Check | z={z}",
            dpi=cfg.DPI_QA,
            label_left='GCaMP Anatomy (pan-neuronal)',
            label_right='DsRed Anatomy (specific)'
        )
    
    # Save metrics
    metrics = {
        "intensity_ncc": float(ncc_val),
        "edge_ncc": float(ncc_edge),
        "mutual_information": float(mi_val),
        "alignment_status": alignment_status,
        "recommendation": "Anatomies are already aligned from MATLAB preprocessing" if ncc_edge > 0.60 
                         else "Consider re-registering anatomies to each other first"
    }
    
    with open(out_dir / "anatomy_alignment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("REGISTRATION V3.0 - PUBLICATION READY")
    print("Enhanced for noisy anatomy + Green/Magenta overlays")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Multi-stage denoising: {len(cfg.DENOISE_STAGES)} stages")
    print(f"  Registration: Rigid ({cfg.RIGID_ITER}) + Affine ({cfg.AFFINE_ITER})")
    print(f"  Multi-resolution: {len(cfg.SHRINK_FACTORS)} levels")
    print(f"  Metric: {cfg.METRIC} (sampling={cfg.METRIC_SAMPLING})")
    print(f"  Output colors: {cfg.COLOR_FIXED.upper()} + {cfg.COLOR_MOVING.upper()}")
    
    # ========== LOAD DATA ==========
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Functional
    cache_name = f"functional_{cfg.PROJECTION_METHOD}_stride{cfg.FRAME_STRIDE}.tif"
    cache_func = out_root / cache_name
    func_fixed = build_functional_template(
        cfg.FUNC_TIFS_DIR, cfg.N_PLANES, cfg.PROJECTION_METHOD, 
        cfg.FRAME_STRIDE, cache_func
    )
    
    # Anatomies
    gcamp_anat_raw = normalize_zyx(imread(cfg.GCAMP_ANAT))
    dsred_anat_raw = normalize_zyx(imread(cfg.DSRED_ANAT))
    print(f"  GCaMP anatomy: {gcamp_anat_raw.shape}")
    print(f"  DsRed anatomy: {dsred_anat_raw.shape}")
    print(f"  Functional: {func_fixed.shape}")
    
    # ========== PREPROCESS ==========
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    # Functional (light preprocessing)
    func_proc = preprocess_volume_enhanced(
        func_fixed, denoise=False, bg_subtract=True, 
        enhance=True, normalize=True, is_anatomy=False
    )
    
    # Anatomies (aggressive preprocessing)
    gcamp_anat_proc = preprocess_volume_enhanced(
        gcamp_anat_raw, denoise=True, bg_subtract=True,
        enhance=True, normalize=True, is_anatomy=True
    )
    
    dsred_anat_proc = preprocess_volume_enhanced(
        dsred_anat_raw, denoise=True, bg_subtract=True,
        enhance=True, normalize=True, is_anatomy=True
    )
    
    # Save preprocessing QA
    qa_prep_dir = out_root / "preprocessing_qa"
    save_preprocessing_comparison(
        gcamp_anat_raw, gcamp_anat_proc, 
        qa_prep_dir / "gcamp_preprocessing_multiplane.png",
        [30, 60, 90, 120, 150], 
        "GCaMP Anatomy: Original vs Enhanced Preprocessing"
    )
    save_preprocessing_comparison(
        dsred_anat_raw, dsred_anat_proc,
        qa_prep_dir / "dsred_preprocessing_multiplane.png",
        [30, 60, 90, 120, 150],
        "DsRed Anatomy: Original vs Enhanced Preprocessing"
    )
    
    # ========== QA: ANATOMY vs ANATOMY ==========
    qa_anat_dir = out_root / "QA_anatomy_vs_anatomy"
    anatomy_metrics = qa_anatomy_alignment(
        gcamp_anat_proc, dsred_anat_proc, qa_anat_dir, cfg.PLANES_TO_SHOW
    )
    
    # ========== DETERMINE ALIGNMENT ==========
    if cfg.ALIGNMENT_X == "auto":
        alignment_x, align_results = pick_best_alignment_x(func_proc, gcamp_anat_proc)
    else:
        alignment_x = cfg.ALIGNMENT_X
        align_results = {}
    
    # ========== FIT TO GRID ==========
    print("\n" + "="*80)
    print("FITTING TO FUNCTIONAL GRID")
    print("="*80)
    # Fit both processed (for registration) and raw (for final overlays/inspection) to the same grid.
    gcamp_anat_fit_reg = fit_moving_to_fixed_keep_fixed(func_proc, gcamp_anat_proc, alignment_x=alignment_x)
    dsred_anat_fit_reg = fit_moving_to_fixed_keep_fixed(func_proc, dsred_anat_proc, alignment_x=alignment_x)
    gcamp_anat_fit_raw = fit_moving_to_fixed_keep_fixed(func_proc, gcamp_anat_raw, alignment_x=alignment_x)
    dsred_anat_fit_raw = fit_moving_to_fixed_keep_fixed(func_proc, dsred_anat_raw, alignment_x=alignment_x)
    
    # ========== CREATE MASK ==========
    if cfg.USE_BRAIN_MASK:
        print("\n" + "="*80)
        print("CREATING BRAIN MASK")
        print("="*80)
        mask_np = create_brain_mask_adaptive(func_proc, method=cfg.MASK_METHOD)
        imwrite(str(out_root / "volumes" / "brain_mask.tif"), (mask_np * 255).astype(np.uint8))
        
        spacing_reg = (1.0, 1.0, 1.0) if cfg.USE_PIXEL_SPACE_FOR_REG else cfg.SPACING_REAL
        mask_sitk = numpy_to_sitk(mask_np.astype(np.uint8), spacing_reg)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
    else:
        mask_sitk = None
        spacing_reg = (1.0, 1.0, 1.0) if cfg.USE_PIXEL_SPACE_FOR_REG else cfg.SPACING_REAL
    
    # ========== REGISTRATION ==========
    print("\n" + "="*80)
    print("REGISTRATION")
    print("="*80)
    
    # Convert to SimpleITK
    # Use preprocessed images for computing the transform.
    fixed_sitk_reg = numpy_to_sitk(func_proc, spacing_reg)
    moving_sitk_reg = numpy_to_sitk(gcamp_anat_fit_reg, spacing_reg)

    # Use raw images for visualization outputs (apply same transform).
    fixed_sitk_vis = numpy_to_sitk(func_fixed, spacing_reg)
    gcamp_sitk_raw = numpy_to_sitk(gcamp_anat_fit_raw, spacing_reg)
    dsred_sitk_raw = numpy_to_sitk(dsred_anat_fit_raw, spacing_reg)
    
    # Structural images
    fixed_struct = gradient_volume_sitk(fixed_sitk_reg, cfg.GRAD_SIGMA)
    moving_struct = gradient_volume_sitk(moving_sitk_reg, cfg.GRAD_SIGMA)
    
    # Register
    tx_rigid, tx_aff, metric_rigid, metric_aff = register_rigid_then_affine(
        fixed_struct, moving_struct, mask_sitk
    )
    tx_comp = compose_transforms(tx_rigid, tx_aff)
    
    # Apply transforms
    print("\n" + "="*80)
    print("APPLYING TRANSFORMS")
    print("="*80)
    # Processed moving resampled (for metrics consistency)
    gcamp_anat_reg_reg = sitk_to_numpy(
        sitk.Resample(moving_sitk_reg, fixed_sitk_reg, tx_comp, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    )

    # RAW resampled (for overlays/inspection)
    gcamp_anat_reg_raw = sitk_to_numpy(
        sitk.Resample(gcamp_sitk_raw, fixed_sitk_vis, tx_comp, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    )
    dsred_anat_reg_raw = sitk_to_numpy(
        sitk.Resample(dsred_sitk_raw, fixed_sitk_vis, tx_comp, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    )
    
    # ========== SAVE VOLUMES ==========
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    vol_dir = out_root / "volumes"
    # Save both "for registration" (processed) and "for inspection" (raw) volumes.
    imwrite(str(vol_dir / "functional_raw_template.tif"), func_fixed.astype(np.float32))
    imwrite(str(vol_dir / "functional_preprocessed_for_registration.tif"), func_proc.astype(np.float32))

    imwrite(str(vol_dir / "gcamp_anatomy_fit_for_registration.tif"), gcamp_anat_fit_reg.astype(np.float32))
    imwrite(str(vol_dir / "gcamp_anatomy_registered_for_registration.tif"), gcamp_anat_reg_reg.astype(np.float32))

    imwrite(str(vol_dir / "gcamp_anatomy_fit_raw.tif"), gcamp_anat_fit_raw.astype(np.float32))
    imwrite(str(vol_dir / "dsred_anatomy_fit_raw.tif"), dsred_anat_fit_raw.astype(np.float32))
    imwrite(str(vol_dir / "gcamp_anatomy_registered_raw.tif"), gcamp_anat_reg_raw.astype(np.float32))
    imwrite(str(vol_dir / "dsred_anatomy_registered_raw.tif"), dsred_anat_reg_raw.astype(np.float32))
    print(f"  ✓ Volumes saved to {vol_dir}")
    
    # ========== SAVE TRANSFORMS ==========
    tx_dir = out_root / "transforms"
    sitk.WriteTransform(tx_rigid, str(tx_dir / "rigid.tfm"))
    sitk.WriteTransform(tx_aff, str(tx_dir / "affine.tfm"))
    sitk.WriteTransform(tx_comp, str(tx_dir / "composite.tfm"))
    
    transform_info = {
        "version": "3.0_publication",
        "registration_type": "rigid_affine_only",
        "purpose": "ROI_colocalization_shape_preserving",
        "preprocessing": {
            "denoise_stages": len(cfg.DENOISE_STAGES),
            "denoise_methods": [s["method"] for s in cfg.DENOISE_STAGES],
            "background_subtraction": True,
            "contrast_enhancement": True,
            "intensity_normalization": True
        },
        "registration": {
            "alignment_x": alignment_x,
            "alignment_scores": align_results,
            "metric": cfg.METRIC,
            "metric_rigid": float(metric_rigid),
            "metric_affine": float(metric_aff),
            "rigid_iterations": cfg.RIGID_ITER,
            "affine_iterations": cfg.AFFINE_ITER,
            "multi_resolution_levels": len(cfg.SHRINK_FACTORS),
            "preserves_roi_shapes": True
        },
        "anatomy_check": anatomy_metrics
    }
    transform_info["outputs"] = {
        "functional_raw_template": "volumes/functional_raw_template.tif",
        "functional_preprocessed_for_registration": "volumes/functional_preprocessed_for_registration.tif",
        "gcamp_anatomy_registered_raw": "volumes/gcamp_anatomy_registered_raw.tif",
        "dsred_anatomy_registered_raw": "volumes/dsred_anatomy_registered_raw.tif",
    }
    
    with open(tx_dir / "transform_info.json", "w", encoding="utf-8") as f:
        json.dump(transform_info, f, indent=2)
    print(f"  ✓ Transforms saved to {tx_dir}")
    
    # ========== COMPUTE METRICS ==========
    print("\n" + "="*80)
    print("COMPUTING QUALITY METRICS")
    print("="*80)
    
    metrics_gcamp = compute_comprehensive_metrics(func_proc, gcamp_anat_fit_reg, gcamp_anat_reg_reg)
    # DsRed is resampled with the SAME transform, but intensity-based similarity vs functional
    # is not a valid "registration quality" metric because DsRed labels a subset/different cells.
    metrics_dsred = None
    
    print(f"\nGCaMP Registration:")
    print(f"  Edge NCC: {metrics_gcamp['edge_ncc_before']:.4f} → {metrics_gcamp['edge_ncc_after']:.4f}")
    print(f"  Intensity NCC: {metrics_gcamp['intensity_ncc_before']:.4f} → {metrics_gcamp['intensity_ncc_after']:.4f}")
    print(f"  Mutual Info: {metrics_gcamp['mi_before']:.4f} → {metrics_gcamp['mi_after']:.4f}")
    print(f"  Quality: {metrics_gcamp['summary']['quality'].upper()}")
    
    print(f"\nDsRed: transform applied (quality metrics vs functional are not reported)")
    
    # Save metrics
    all_metrics = {
        "gcamp_registration": metrics_gcamp,
        "anatomy_alignment": anatomy_metrics
    }

    # Automatic landmark-style QA (bright spot NN distances) for functional vs GCaMP anatomy.
    if cfg.USE_SPOT_QA:
        print("\n" + "="*80)
        print("SPOT-BASED QA (AUTOMATIC LANDMARKS)")
        print("="*80)
        qa_spot_dir = out_root / "QA_functional_vs_gcamp_spots"
        all_metrics["spot_based_qa"] = spot_qa_report(
            func_fixed, gcamp_anat_fit_raw, gcamp_anat_reg_raw, qa_spot_dir, cfg.QA_PLANES_FOR_SPOTS
        )
    
    with open(out_root / "metrics" / "comprehensive_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  ✓ Metrics saved")
    
    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "="*80)
    print("GENERATING PUBLICATION FIGURES")
    print("="*80)
    
    qa_func_dir = out_root / "QA_functional_vs_anatomy"
    pub_dir = out_root / "publication_figures"
    
    for z in cfg.PLANES_TO_SHOW:
        if z >= func_fixed.shape[0]:
            continue
        
        print(f"  Processing plane z={z}")
        
        # GCaMP overlays
        save_publication_overlay(
            func_fixed[z], gcamp_anat_reg_raw[z],
            pub_dir / f"overlay_func_gcamp_z{z:03d}.png",
            f"Functional (green) + GCaMP Anatomy (magenta) | z={z}",
            dpi=cfg.DPI_QA
        )
        
        save_publication_triple(
            func_fixed[z], gcamp_anat_reg_raw[z],
            pub_dir / f"triple_func_gcamp_z{z:03d}.png",
            f"Registration Result: GCaMP | z={z}",
            dpi=cfg.DPI_QA
        )
        
        # DsRed overlays
        save_publication_overlay(
            func_fixed[z], dsred_anat_reg_raw[z],
            pub_dir / f"overlay_func_dsred_z{z:03d}.png",
            f"Functional (green) + DsRed Anatomy (magenta) | z={z}",
            dpi=cfg.DPI_QA
        )
        
        save_publication_triple(
            func_fixed[z], dsred_anat_reg_raw[z],
            pub_dir / f"triple_func_dsred_z{z:03d}.png",
            f"Registration Result: DsRed | z={z}",
            dpi=cfg.DPI_QA
        )
        
        # Before/after comparison for QA
        save_publication_sidebyside(
            func_fixed[z], gcamp_anat_fit_raw[z],
            qa_func_dir / f"before_registration_gcamp_z{z:03d}.png",
            f"Before Registration: GCaMP | z={z}",
            dpi=250
        )
        
        save_publication_sidebyside(
            func_fixed[z], gcamp_anat_reg_raw[z],
            qa_func_dir / f"after_registration_gcamp_z{z:03d}.png",
            f"After Registration: GCaMP | z={z}",
            dpi=250
        )
    
    print(f"  ✓ Generated {len(cfg.PLANES_TO_SHOW) * 6} visualization files")
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("REGISTRATION COMPLETE")
    print("="*80)
    
    print(f"\n📁 Output directory: {out_root}")
    print(f"\n📊 Registration Quality:")
    print(f"   GCaMP: {metrics_gcamp['summary']['quality'].upper()} (score={metrics_gcamp['summary']['quality_score']:.3f})")
    print(f"   DsRed: transform applied (not scored vs functional)")
    print(f"   Anatomy pre-alignment: {anatomy_metrics['alignment_status']}")
    
    print(f"\n📂 Key outputs:")
    print(f"   volumes/")
    print(f"   ├─ gcamp_anatomy_registered.tif  ← Use for colocalization")
    print(f"   └─ dsred_anatomy_registered.tif  ← Use for colocalization")
    print(f"   publication_figures/")
    print(f"   ├─ overlay_*.png  ← Green+Magenta overlays for paper")
    print(f"   └─ triple_*.png   ← 3-panel figures for paper")
    print(f"   QA_anatomy_vs_anatomy/")
    print(f"   └─ Anatomies aligned to each other")
    
    if not metrics_gcamp['summary']['meets_threshold']:
        print(f"\n⚠️  WARNING: GCaMP registration below quality threshold")
    # DsRed is not scored vs functional; no threshold warning.
    
    print("\n✓ ROI shapes preserved (no deformable registration used)")
    print("✓ Publication-ready green+magenta overlays generated")
    print("="*80)


if __name__ == "__main__":
    main()
