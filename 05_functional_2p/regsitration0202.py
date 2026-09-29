"""
REGISTRATION FAST AND ROBUST
GCaMP anatomy and DsRed anatomy to functional space

Fixed space
Functional template

Moving
GCaMP anatomy

Apply same transform
DsRed anatomy

Registration strategy
Register on structural images using gradient magnitude and correlation
Apply transforms to original intensity volumes

Outputs
Volumes in functional space
Rigid and affine transforms saved separately

Ines Marques
"""

import os
import re
import json
import time
import numpy as np
from pathlib import Path

from tifffile import imread, imwrite, TiffFile
import SimpleITK as sitk
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, binary_dilation

import warnings
warnings.filterwarnings("ignore")


class Config:
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"

    GCAMP_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\Gcamp6s_averaged\alignment_drift\anatomy.tif"
    DSRED_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"

    OUT_DIR = r"C:\Users\OSVALDO\Downloads\REGISTRATION_ANAT_TO_FUNCTIONAL_bspline"

    N_PLANES = 180

    PLANE_PATTERN = r"aligned_p(\d+)_nan\.tif"

    PROJECTION_METHOD = "mean"   # mean or median
    FRAME_STRIDE = 2             # 1 uses all frames, 2 uses every other frame, 3 uses every third frame
    SAFE_TIFF_READ = True        # skip bad frames when reading tiffs
    MAX_BAD_FRAMES_PER_PLANE = 10

    ALIGNMENT_X = "auto"         # auto or center or left or right

    USE_PIXEL_SPACE_FOR_REG = False
    SPACING_REAL = (0.5481, 0.5935, 1.0)

    USE_BRAIN_MASK = True
    MASK_PERCENTILE = 15
    MASK_MIN_SIZE = 1500
    MASK_DILATE_ITERS = 2

    METRIC_SAMPLING = 0.25
    METRIC = "MI"                # MI or CC
    RIGID_ITER = 500
    AFFINE_ITER = 800
    USE_BSPLINE = False           # deformable refinement (best possible without ANTs)
    BSPLINE_PRE_SHRINK = (4, 4, 1)
    BSPLINE_GRID_PHYS = 28.0
    BSPLINE_ITER = 60
    BSPLINE_SAMPLING = 0.05
    # extra downsample before B-spline to reduce RAM (x,y,z in SimpleITK)
    BSPLINE_RETRY_ON_BAD_ALLOC = True
    BSPLINE_RETRY_GRID_MULT = 2.0
    BSPLINE_RETRY_SAMPLING = 0.10
    BSPLINE_RETRY_ITER = 120

    SHRINK_FACTORS = [6, 3, 1]
    SMOOTHING_SIGMAS = [2.0, 1.0, 0.0]

    GRAD_SIGMA = 1.0

    PLANES_TO_SHOW = [60, 90, 120]
    DPI_QA = 250


cfg = Config()

out_root = Path(cfg.OUT_DIR)
for subdir in ["volumes", "transforms", "metrics", "QA"]:
    (out_root / subdir).mkdir(parents=True, exist_ok=True)


def normalize_zyx(arr):
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0] if a.shape[0] in (2, 3, 4) else a[..., 0]
    if a.ndim == 2:
        a = a[None, ...]
    return a.astype(np.float32)


def numpy_to_sitk(arr_zyx, spacing_xyz):
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.SetSpacing(tuple(float(x) for x in spacing_xyz))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0))
    return img


def sitk_to_numpy(img):
    return sitk.GetArrayFromImage(img).astype(np.float32)


def enhance_for_display(img2d, p_low=2, p_high=98):
    x = img2d.astype(np.float32)
    nz = x[x > 0]
    if nz.size == 0:
        m = float(x.max())
        return x / (m + 1e-8)
    vmin, vmax = np.percentile(nz, [p_low, p_high])
    return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)


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


def create_brain_mask(volume, threshold_percentile, min_size, dilate_iters):
    v = volume.astype(np.float32)
    pos = v[v > 0]
    if pos.size == 0:
        return np.ones_like(v, dtype=np.uint8)

    thresh = np.percentile(pos, threshold_percentile)
    mask = (v > thresh)

    mask_clean = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        sl = mask[z]
        sl = remove_small_objects(sl, min_size=int(min_size))
        sl = binary_fill_holes(sl)
        sl = binary_dilation(sl, iterations=int(dilate_iters))
        mask_clean[z] = sl

    return mask_clean.astype(np.uint8)


def list_plane_files(tifs_dir, pattern_regex):
    tifs_path = Path(tifs_dir)
    pattern = re.compile(pattern_regex, re.IGNORECASE)

    plane_files = {}
    for f in os.listdir(tifs_path):
        m = pattern.match(f)
        if m:
            plane_idx = int(m.group(1)) - 1
            plane_files[plane_idx] = tifs_path / f

    if len(plane_files) == 0:
        raise ValueError("No plane TIFs found with the expected pattern")

    return plane_files


def project_plane_tif(path, method="mean", frame_stride=1):
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
                print(f"  WARNING: {bad} bad frames skipped in {Path(path).name}")
            if bad > cfg.MAX_BAD_FRAMES_PER_PLANE:
                print(f"  WARNING: too many bad frames in {Path(path).name}")
            return acc / max(count, 1)

        if method == "median":
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
                print(f"  WARNING: {bad} bad frames skipped in {Path(path).name}")
            if bad > cfg.MAX_BAD_FRAMES_PER_PLANE:
                print(f"  WARNING: too many bad frames in {Path(path).name}")
            stack = np.stack(frames, axis=0)
            return np.median(stack, axis=0).astype(np.float32)

        raise ValueError("Unknown projection method")


def build_functional_template_from_tifs(tifs_dir, n_planes, method, frame_stride, cache_path):
    if Path(cache_path).exists():
        vol = normalize_zyx(imread(str(cache_path)))
        print(f"Loaded cached template {cache_path}")
        return vol

    plane_files = list_plane_files(tifs_dir, cfg.PLANE_PATTERN)

    first_plane = min(plane_files.keys())
    first_proj = project_plane_tif(plane_files[first_plane], method=method, frame_stride=frame_stride)
    h, w = first_proj.shape

    stack = np.zeros((n_planes, h, w), dtype=np.float32)

    frames_per_plane = []
    loaded = 0
    for plane_idx in sorted(plane_files.keys()):
        if plane_idx >= n_planes:
            continue

        with TiffFile(str(plane_files[plane_idx])) as tif:
            frames_per_plane.append(len(tif.pages))

        proj = project_plane_tif(plane_files[plane_idx], method=method, frame_stride=frame_stride)
        stack[plane_idx] = proj.astype(np.float32)
        loaded += 1
        if loaded % 30 == 0:
            print(f"  Plane {loaded}/{n_planes} loaded")

    print(f"Functional template shape {stack.shape}")
    print(f"Frames per plane min {min(frames_per_plane)} max {max(frames_per_plane)} mean {np.mean(frames_per_plane):.1f}")

    imwrite(str(cache_path), stack.astype(np.float32))
    print(f"Cached template {cache_path}")

    return stack


def edges_np(vol):
    out = np.zeros_like(vol, dtype=np.float32)
    for z in range(vol.shape[0]):
        out[z] = sobel(gaussian(vol[z], sigma=1.0, preserve_range=True))
    return out


def ncc(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    num = float(np.sum(a * b))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-8)
    return num / den


def pick_best_alignment_x(fixed_np, moving_np_raw):
    options = ["left", "center", "right"]
    ef = edges_np(fixed_np)
    best = None
    best_score = -1e9
    for mode in options:
        fit = fit_moving_to_fixed_keep_fixed(fixed_np, moving_np_raw, alignment_x=mode)
        em = edges_np(fit)
        score = ncc(ef, em)
        if score > best_score:
            best_score = score
            best = mode
    print(f"Best alignment_x {best} with edge NCC {best_score:.4f}")
    return best


def gradient_volume_sitk(img_sitk, sigma):
    return sitk.GradientMagnitudeRecursiveGaussian(img_sitk, float(sigma))


def make_registration_method(metric_sampling, n_iter, shrink_factors, smoothing_sigmas):
    reg = sitk.ImageRegistrationMethod()

    if cfg.METRIC.upper() == "MI":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(metric_sampling))

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0,
        numberOfIterations=int(n_iter),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=12
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel([int(x) for x in shrink_factors])
    reg.SetSmoothingSigmasPerLevel([float(x) for x in smoothing_sigmas])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    return reg


def register_rigid_then_affine(fixed_struct, moving_struct, mask_fixed=None):
    print("  Rigid registration")
    init_rigid = sitk.CenteredTransformInitializer(
        fixed_struct, moving_struct, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg_rigid = make_registration_method(
        cfg.METRIC_SAMPLING, cfg.RIGID_ITER, cfg.SHRINK_FACTORS, cfg.SMOOTHING_SIGMAS
    )
    reg_rigid.SetInitialTransform(init_rigid, inPlace=True)
    if mask_fixed is not None:
        reg_rigid.SetMetricFixedMask(mask_fixed)

    tx_rigid = reg_rigid.Execute(fixed_struct, moving_struct)

    print("  Affine registration")
    init_aff = sitk.AffineTransform(3)
    init_aff.SetCenter(tx_rigid.GetCenter())
    init_aff.SetMatrix((1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0))
    init_aff.SetTranslation((0.0, 0.0, 0.0))

    reg_aff = make_registration_method(
        cfg.METRIC_SAMPLING, cfg.AFFINE_ITER, cfg.SHRINK_FACTORS, cfg.SMOOTHING_SIGMAS
    )
    reg_aff.SetMovingInitialTransform(tx_rigid)
    reg_aff.SetInitialTransform(init_aff, inPlace=True)
    if mask_fixed is not None:
        reg_aff.SetMetricFixedMask(mask_fixed)

    tx_aff = reg_aff.Execute(fixed_struct, moving_struct)

    return tx_rigid, tx_aff


def register_bspline(fixed_struct, moving_struct, initial_transform, mask_fixed=None):
    # B-spline can be RAM-hungry on Windows. We downsample once up-front and use LBFGSB
    # (limited-memory optimizer) to avoid "bad allocation" crashes.
    shrink = tuple(int(x) for x in cfg.BSPLINE_PRE_SHRINK)
    if any(s > 1 for s in shrink):
        fixed_use = sitk.Shrink(fixed_struct, shrink)
        moving_use = sitk.Shrink(moving_struct, shrink)
        mask_use = sitk.Shrink(mask_fixed, shrink) if mask_fixed is not None else None
    else:
        fixed_use = fixed_struct
        moving_use = moving_struct
        mask_use = mask_fixed

    grid_physical = float(cfg.BSPLINE_GRID_PHYS)
    size = fixed_use.GetSize()
    spacing = fixed_use.GetSpacing()
    # Physical size should use (size-1)*spacing, not size*spacing.
    image_physical_size = [(max(1, s - 1) * sp) for s, sp in zip(size, spacing)]
    mesh_size = [max(1, int(ps / grid_physical + 0.5)) for ps in image_physical_size]

    bspline = sitk.BSplineTransformInitializer(fixed_use, mesh_size, order=3)

    reg = sitk.ImageRegistrationMethod()
    if cfg.METRIC.upper() == "MI":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        reg.SetMetricAsCorrelation()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(cfg.BSPLINE_SAMPLING))
    reg.SetInterpolator(sitk.sitkLinear)

    start_t = time.time()
    state = {"last_print": -1}

    def _iter_cmd():
        # Print progress occasionally so long B-spline runs don't look "stuck".
        it = int(reg.GetOptimizerIteration())
        if it % 10 != 0 or it == state["last_print"]:
            return
        state["last_print"] = it
        try:
            mv = float(reg.GetMetricValue())
        except Exception:
            mv = float("nan")
        elapsed_min = (time.time() - start_t) / 60.0
        print(f"  B-spline iter {it:04d} metric {mv:.6g} elapsed {elapsed_min:.1f} min")

    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=int(cfg.BSPLINE_ITER),
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=2000,
        costFunctionConvergenceFactor=1e7
    )
    # LBFGSB in ITK ignores optimizer scales; leaving them unset avoids warnings.

    # Keep B-spline multires modest; we already shrank once.
    reg.SetShrinkFactorsPerLevel([3, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetMovingInitialTransform(initial_transform)
    reg.SetInitialTransform(bspline, inPlace=True)
    if mask_use is not None:
        reg.SetMetricFixedMask(sitk.Cast(mask_use, sitk.sitkUInt8))

    reg.AddCommand(sitk.sitkIterationEvent, _iter_cmd)
    tx_bspline = reg.Execute(fixed_use, moving_use)
    return tx_bspline


def try_register_bspline(fixed_struct, moving_struct, initial_transform, mask_fixed=None):
    try:
        return register_bspline(fixed_struct, moving_struct, initial_transform, mask_fixed)
    except RuntimeError as e:
        msg = str(e)
        if ("bad allocation" not in msg.lower()) or (not cfg.BSPLINE_RETRY_ON_BAD_ALLOC):
            raise

        print("  WARNING: B-spline failed with bad allocation; retrying with coarser grid / fewer iters")
        old_grid = cfg.BSPLINE_GRID_PHYS
        old_iter = cfg.BSPLINE_ITER
        old_samp = cfg.BSPLINE_SAMPLING
        try:
            cfg.BSPLINE_GRID_PHYS = float(old_grid) * float(cfg.BSPLINE_RETRY_GRID_MULT)
            cfg.BSPLINE_ITER = int(cfg.BSPLINE_RETRY_ITER)
            cfg.BSPLINE_SAMPLING = float(cfg.BSPLINE_RETRY_SAMPLING)
            return register_bspline(fixed_struct, moving_struct, initial_transform, mask_fixed)
        finally:
            cfg.BSPLINE_GRID_PHYS = old_grid
            cfg.BSPLINE_ITER = old_iter
            cfg.BSPLINE_SAMPLING = old_samp


def compose_transforms(tx_rigid, tx_aff):
    tx = sitk.CompositeTransform(3)
    tx.AddTransform(tx_rigid)
    tx.AddTransform(tx_aff)
    return tx


def dice_binary(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter) / (denom + 1e-8))


def edge_metrics(fixed, moving_before, moving_after):
    ef = edges_np(fixed)
    eb = edges_np(moving_before)
    ea = edges_np(moving_after)

    m = {}
    m["edge_ncc_before"] = ncc(ef, eb)
    m["edge_ncc_after"] = ncc(ef, ea)
    m["edge_ncc_improvement"] = m["edge_ncc_after"] - m["edge_ncc_before"]

    def thresh(vol, p=90):
        v = vol[vol > 0]
        if v.size == 0:
            return np.zeros_like(vol, dtype=bool)
        t = np.percentile(v, p)
        return vol > t

    bf = thresh(ef, 90)
    bb = thresh(eb, 90)
    ba = thresh(ea, 90)

    m["edge_dice_before"] = dice_binary(bf, bb)
    m["edge_dice_after"] = dice_binary(bf, ba)
    m["edge_dice_improvement"] = m["edge_dice_after"] - m["edge_dice_before"]
    return m


def save_overlay_png(fixed2d, moving2d, out_png, title, dpi):
    f = enhance_for_display(fixed2d)
    m = enhance_for_display(moving2d)
    rgb = np.zeros((f.shape[0], f.shape[1], 3), dtype=np.float32)
    rgb[..., 1] = f
    rgb[..., 0] = m

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb, interpolation="bilinear")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading functional template")
    cache_name = f"functional_{cfg.PROJECTION_METHOD}_stride{cfg.FRAME_STRIDE}_from_tifs.tif"
    cache_func = out_root / cache_name
    func_fixed = build_functional_template_from_tifs(
        cfg.FUNC_TIFS_DIR, cfg.N_PLANES, cfg.PROJECTION_METHOD, cfg.FRAME_STRIDE, cache_func
    )

    print("Loading anatomy volumes")
    gcamp_anat_raw = normalize_zyx(imread(cfg.GCAMP_ANAT))
    dsred_anat_raw = normalize_zyx(imread(cfg.DSRED_ANAT))

    if cfg.ALIGNMENT_X == "auto":
        alignment_x = pick_best_alignment_x(func_fixed, gcamp_anat_raw)
    else:
        alignment_x = cfg.ALIGNMENT_X

    print("Fitting anatomy to functional grid")
    gcamp_anat_fit = fit_moving_to_fixed_keep_fixed(func_fixed, gcamp_anat_raw, alignment_x=alignment_x)
    dsred_anat_fit = fit_moving_to_fixed_keep_fixed(func_fixed, dsred_anat_raw, alignment_x=alignment_x)

    if cfg.USE_BRAIN_MASK:
        print("Creating mask from functional")
        mask_np = create_brain_mask(func_fixed, cfg.MASK_PERCENTILE, cfg.MASK_MIN_SIZE, cfg.MASK_DILATE_ITERS)
        imwrite(str(out_root / "volumes" / "brain_mask.tif"), (mask_np.astype(np.uint8) * 255))
        mask_sitk = numpy_to_sitk(mask_np.astype(np.uint8), (1.0, 1.0, 1.0) if cfg.USE_PIXEL_SPACE_FOR_REG else cfg.SPACING_REAL)
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
    else:
        mask_sitk = None

    spacing_reg = (1.0, 1.0, 1.0) if cfg.USE_PIXEL_SPACE_FOR_REG else cfg.SPACING_REAL

    fixed_sitk = numpy_to_sitk(func_fixed, spacing_reg)
    moving_sitk = numpy_to_sitk(gcamp_anat_fit, spacing_reg)
    dsred_sitk = numpy_to_sitk(dsred_anat_fit, spacing_reg)

    print("Building structural images for registration")
    fixed_struct = gradient_volume_sitk(fixed_sitk, cfg.GRAD_SIGMA)
    moving_struct = gradient_volume_sitk(moving_sitk, cfg.GRAD_SIGMA)

    print("Running rigid and affine registration on structure")
    tx_rigid, tx_aff = register_rigid_then_affine(fixed_struct, moving_struct, mask_sitk)
    tx_comp = compose_transforms(tx_rigid, tx_aff)

    if cfg.USE_BSPLINE:
        print("Running B-spline refinement (deformable)")
        tx_bspline = try_register_bspline(fixed_struct, moving_struct, tx_comp, mask_sitk)
        tx_comp.AddTransform(tx_bspline)
    else:
        tx_bspline = None

    print("Applying transform to original intensity volumes")
    gcamp_anat_reg = sitk_to_numpy(
        sitk.Resample(moving_sitk, fixed_sitk, tx_comp, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    )
    dsred_anat_reg = sitk_to_numpy(
        sitk.Resample(dsred_sitk, fixed_sitk, tx_comp, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    )

    print("Saving volumes")
    vol_dir = out_root / "volumes"
    imwrite(str(vol_dir / "functional_fixed.tif"), func_fixed.astype(np.float32))
    imwrite(str(vol_dir / "gcamp_anatomy_fit_before.tif"), gcamp_anat_fit.astype(np.float32))
    imwrite(str(vol_dir / "dsred_anatomy_fit_before.tif"), dsred_anat_fit.astype(np.float32))
    imwrite(str(vol_dir / "gcamp_anatomy_registered_to_functional.tif"), gcamp_anat_reg.astype(np.float32))
    imwrite(str(vol_dir / "dsred_anatomy_registered_to_functional.tif"), dsred_anat_reg.astype(np.float32))

    print("Saving transforms")
    tx_dir = out_root / "transforms"
    sitk.WriteTransform(tx_rigid, str(tx_dir / "rigid.tfm"))
    sitk.WriteTransform(tx_aff, str(tx_dir / "affine.tfm"))
    if tx_bspline is not None:
        sitk.WriteTransform(tx_bspline, str(tx_dir / "bspline.tfm"))

    info = {
        "alignment_x_used": alignment_x,
        "use_pixel_space_for_reg": bool(cfg.USE_PIXEL_SPACE_FOR_REG),
        "spacing_reg": spacing_reg,
        "spacing_real": cfg.SPACING_REAL,
        "projection_method": cfg.PROJECTION_METHOD,
        "frame_stride": int(cfg.FRAME_STRIDE),
        "rigid_iter": int(cfg.RIGID_ITER),
        "affine_iter": int(cfg.AFFINE_ITER),
        "bspline": bool(cfg.USE_BSPLINE),
        "bspline_grid_phys": float(cfg.BSPLINE_GRID_PHYS),
        "bspline_iter": int(cfg.BSPLINE_ITER),
        "metric": str(cfg.METRIC),
        "bspline_sampling": float(cfg.BSPLINE_SAMPLING),
        "metric_sampling": float(cfg.METRIC_SAMPLING),
    }
    with open(out_root / "transforms" / "transform_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("Computing edge based metrics")
    metrics = edge_metrics(func_fixed, gcamp_anat_fit, gcamp_anat_reg)
    metrics["summary"] = {
        "edge_quality": "excellent" if metrics["edge_ncc_after"] > 0.75 else "good" if metrics["edge_ncc_after"] > 0.6 else "acceptable"
    }

    metrics_path = out_root / "metrics" / "metrics_edge_based.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics")
    print(f"Edge NCC {metrics['edge_ncc_before']:.4f} to {metrics['edge_ncc_after']:.4f}")
    print(f"Edge Dice {metrics['edge_dice_before']:.4f} to {metrics['edge_dice_after']:.4f}")
    print(f"Edge quality {metrics['summary']['edge_quality']}")

    print("Saving QA overlays")
    qa_dir = out_root / "QA"
    for z in cfg.PLANES_TO_SHOW:
        if z >= func_fixed.shape[0]:
            continue
        save_overlay_png(
            func_fixed[z],
            gcamp_anat_reg[z],
            qa_dir / f"overlay_func_vs_gcampReg_z{z:03d}.png",
            f"Functional green and GCaMP registered red z {z}",
            cfg.DPI_QA
        )
        save_overlay_png(
            func_fixed[z],
            dsred_anat_reg[z],
            qa_dir / f"overlay_func_vs_dsredReg_z{z:03d}.png",
            f"Functional green and DsRed registered red z {z}",
            cfg.DPI_QA
        )

    print("Done")
    print(f"Output directory {out_root}")
    print(f"Key output {vol_dir / 'dsred_anatomy_registered_to_functional.tif'}")
    print(f"Transforms saved as {tx_dir / 'rigid.tfm'} and {tx_dir / 'affine.tfm'}")


if __name__ == "__main__":
    main()
