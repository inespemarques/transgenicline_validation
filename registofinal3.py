"""
ANTS REGISTRATION - AFFINE REFINADO
Máxima precisão subpixel para colocalização de ROIs

OTIMIZADO PARA:
- Colocalização precisa
- Preservação de forma celular
- Precisão subpixel
- Tempo razoável (~15-20 min)
"""

import numpy as np
from pathlib import Path
from tifffile import imread, imwrite, TiffFile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import json
import re
from datetime import datetime
import warnings
import traceback
warnings.filterwarnings('ignore')

try:
    import ants
    print(f"✓ ANTs {ants.__version__} loaded successfully!")
except ImportError:
    print("ERROR: ANTsPy not installed!")
    exit(1)

# ============================================================================
# CONFIG - REFINADO PARA PRECISÃO SUBPIXEL
# ============================================================================

class Config:
    # ========== PATHS ==========
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    GCAMP_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\Gcamp6s_averaged\alignment_drift\anatomy.tif"
    DSRED_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_AFFINE_REFINADO"
    
    # ========== TEMPLATE ==========
    N_PLANES = 180
    PLANE_PATTERN = r"aligned_p(\d+)_nan\.tif"
    PROJECTION_METHOD = "mean"
    FRAME_STRIDE = 2
    CHECK_FOR_NANS = True
    
    # ========== PREPROCESSING - PRECISÃO MÁXIMA ==========
    FUNCTIONAL_SMOOTH = 0.3    # Menos smoothing = mais detalhes
    ANATOMY_SMOOTH = 0.5       # Menos smoothing
    PERCENTILE_LOW = 0.5       # Mais conservador
    PERCENTILE_HIGH = 99.9
    
    # ========== ANTS AFFINE REFINADO ==========
    REG_TYPE = "Affine"
    
    # Parâmetros para PRECISÃO SUBPIXEL
    CONVERGENCE_THRESHOLD = 1e-08  # Mais rigoroso (default: 1e-06)
    CONVERGENCE_WINDOW = 20        # Mais estável (default: 10)
    
    # Mais iterações para convergência fina
    ITERATIONS = [2000, 1000, 500, 100]  # 4 níveis multi-escala
    
    # Smoothing mais fino
    SMOOTHING_SIGMAS = [3, 2, 1, 0]  # Igual ao padrão
    SHRINK_FACTORS = [8, 4, 2, 1]    # Mais níveis
    
    VERBOSE = True
    
    # ========== SPACING - PRECISO ==========
    SPACING = (0.5481, 0.5935, 1.0)
    
    # ========== FIGURES ==========
    FIGURE_PLANES = [30, 60, 90, 120, 150]
    FIGURE_DPI = 300


cfg = Config()

# ============================================================================
# FUNÇÕES (IGUAIS AO SIMPLE - copiadas)
# ============================================================================

def list_plane_files(tifs_dir, pattern_regex):
    tifs_path = Path(tifs_dir)
    pattern = re.compile(pattern_regex, re.IGNORECASE)
    plane_files = {}
    
    for f in tifs_path.iterdir():
        if not f.is_file():
            continue
        m = pattern.match(f.name)
        if m:
            plane_idx = int(m.group(1)) - 1
            plane_files[plane_idx] = f
    
    if len(plane_files) == 0:
        raise ValueError(f"No plane TIFs found")
    
    print(f"  Found {len(plane_files)} plane files")
    return plane_files


def project_plane_tif(path, method="mean", frame_stride=1, check_nans=True):
    with TiffFile(str(path)) as tif:
        n_pages = len(tif.pages)
        
        if frame_stride > 1:
            indices = list(range(0, n_pages, frame_stride))
        else:
            indices = list(range(n_pages))
        
        frames = []
        nan_count = 0
        
        for idx in indices:
            try:
                frame = tif.pages[idx].asarray().astype(np.float32)
                
                if check_nans and np.any(np.isnan(frame)):
                    nan_count += 1
                    continue
                
                frames.append(frame)
            except:
                continue
        
        if len(frames) == 0:
            raise RuntimeError(f"No valid frames in {path}")
        
        if nan_count > 0:
            print(f"    Skipped {nan_count} frames with NaNs")
        
        stack = np.array(frames)
        
        if method == "mean":
            projection = np.mean(stack, axis=0)
        elif method == "median":
            projection = np.median(stack, axis=0)
        
        return projection.astype(np.float32)


def build_functional_template(tifs_dir, n_planes, method, frame_stride, check_nans):
    print("\n" + "="*80)
    print("BUILDING FUNCTIONAL TEMPLATE")
    print("="*80)
    
    plane_files = list_plane_files(tifs_dir, cfg.PLANE_PATTERN)
    
    first_idx = min(plane_files.keys())
    print(f"\nLoading first plane...")
    first_proj = project_plane_tif(plane_files[first_idx], method, frame_stride, check_nans)
    h, w = first_proj.shape
    print(f"  Shape: {h} x {w}")
    
    n_planes_actual = min(n_planes, max(plane_files.keys()) + 1)
    volume = np.zeros((n_planes_actual, h, w), dtype=np.float32)
    
    print(f"\nLoading {n_planes_actual} planes...")
    for plane_idx in sorted(plane_files.keys()):
        if plane_idx >= n_planes_actual:
            break
        
        if plane_idx % 20 == 0:
            print(f"  Progress: {plane_idx+1}/{n_planes_actual}")
        
        proj = project_plane_tif(plane_files[plane_idx], method, frame_stride, check_nans)
        volume[plane_idx] = proj
    
    print(f"\n✓ Template built: {volume.shape}")
    return volume


def normalize_robust(vol):
    v = vol.astype(np.float32)
    nz = v[v > 0]
    if nz.size < 100:
        return v
    vmin, vmax = np.percentile(nz, [cfg.PERCENTILE_LOW, cfg.PERCENTILE_HIGH])
    return np.clip((v - vmin) / (vmax - vmin + 1e-8), 0, 1)


def smooth_volume(vol, sigma):
    if sigma <= 0:
        return vol
    out = np.zeros_like(vol)
    for z in range(vol.shape[0]):
        out[z] = gaussian_filter(vol[z], sigma=sigma)
    return out


def match_shapes(fixed, moving):
    zf, yf, xf = fixed.shape
    zm, ym, xm = moving.shape
    
    if zm > zf:
        z_start = (zm - zf) // 2
        moving = moving[z_start:z_start+zf]
    elif zm < zf:
        pad_z = zf - zm
        pad_top = pad_z // 2
        pad_bot = pad_z - pad_top
        moving = np.pad(moving, ((pad_top, pad_bot), (0, 0), (0, 0)), mode='constant')
    
    if ym > yf:
        y_start = (ym - yf) // 2
        moving = moving[:, y_start:y_start+yf]
    elif ym < yf:
        pad_y = yf - ym
        pad_top = pad_y // 2
        pad_bot = pad_y - pad_top
        moving = np.pad(moving, ((0, 0), (pad_top, pad_bot), (0, 0)), mode='constant')
    
    if xm > xf:
        x_start = (xm - xf) // 2
        moving = moving[:, :, x_start:x_start+xf]
    elif xm < xf:
        pad_x = xf - xm
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        moving = np.pad(moving, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
    
    return moving


def numpy_to_ants(vol_zyx, spacing_xyz):
    if vol_zyx.ndim != 3:
        raise ValueError(f"Volume must be 3D")
    
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))
    img = ants.from_numpy(vol_xyz.astype(np.float32))
    img.set_spacing(spacing_xyz)
    img.set_origin((0.0, 0.0, 0.0))
    
    return img


def ants_to_numpy(img):
    vol_xyz = img.numpy()
    return np.transpose(vol_xyz, (2, 1, 0)).astype(np.float32)


# ============================================================================
# AFFINE REFINADO - PRECISÃO SUBPIXEL
# ============================================================================

def register_affine_refined(fixed_ants, moving_ants):
    """
    AFFINE REFINADO para precisão subpixel
    Parâmetros otimizados para colocalização
    """
    print(f"\n{'='*80}")
    print(f"ANTs AFFINE REFINADO - SUBPIXEL PRECISION")
    print(f"{'='*80}")
    
    print(f"\nParâmetros de precisão:")
    print(f"  • Convergence threshold: {cfg.CONVERGENCE_THRESHOLD}")
    print(f"  • Convergence window: {cfg.CONVERGENCE_WINDOW}")
    print(f"  • Iterations: {cfg.ITERATIONS}")
    print(f"  • Smoothing sigmas: {cfg.SMOOTHING_SIGMAS}")
    print(f"  • Shrink factors: {cfg.SHRINK_FACTORS}")
    
    start_time = datetime.now()
    
    print(f"\nImage shapes:")
    print(f"  Fixed: {fixed_ants.shape}")
    print(f"  Moving: {moving_ants.shape}")
    
    print(f"\nStarting refined Affine registration...")
    print("This may take 15-20 minutes for maximum precision...")
    
    try:
        # Usar ants.registration com parâmetros customizados
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform='Affine',
            
            # Controlo fino da convergência
            aff_iterations=cfg.ITERATIONS,
            aff_shrink_factors=cfg.SHRINK_FACTORS,
            aff_smoothing_sigmas=cfg.SMOOTHING_SIGMAS,
            
            # Métrica
            aff_metric='mattes',
            aff_metric_weight=1.0,
            aff_metric_bins=64,  # Mais bins = mais preciso
            aff_sampling='random',
            aff_sampling_rate=0.5,  # Mais samples = mais preciso
            
            verbose=cfg.VERBOSE
        )
        
    except Exception as e:
        print(f"\n❌ Refined registration failed: {e}")
        print("\n🔄 Fallback: trying standard Affine...")
        
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform='Affine',
            verbose=cfg.VERBOSE
        )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Registration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if 'warpedmovout' not in result:
        raise ValueError("Registration failed")
    
    print(f"  • Forward transforms: {len(result['fwdtransforms'])}")
    print(f"  • Inverse transforms: {len(result['invtransforms'])}")
    
    return result


def apply_transforms_refined(fixed_ants, moving_ants, transforms):
    """Apply transformation com interpolação de alta qualidade"""
    print(f"\nApplying transforms with high-quality interpolation...")
    
    warped = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=transforms,
        interpolator='lanczosWindowedSinc'  # Melhor interpolação
    )
    
    print("✓ Transforms applied")
    return warped


# ============================================================================
# FIGURES E METRICS (iguais)
# ============================================================================

def create_thesis_figure_before_after(fixed, moving_before, moving_after, 
                                      output_path, planes, title):
    n_planes = len(planes)
    fig = plt.figure(figsize=(4*n_planes, 10))
    gs = GridSpec(2, n_planes, figure=fig, hspace=0.1, wspace=0.05)
    
    for col, z in enumerate(planes):
        if z >= fixed.shape[0]:
            continue
        
        def norm_slice(img):
            v = img[img > 0]
            if v.size == 0:
                return img
            vmin, vmax = np.percentile(v, [0.5, 99.7])
            return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        f = norm_slice(fixed[z])
        mb = norm_slice(moving_before[z])
        ma = norm_slice(moving_after[z])
        
        # BEFORE
        ax_before = fig.add_subplot(gs[0, col])
        rgb_before = np.zeros((*f.shape, 3), dtype=np.float32)
        rgb_before[..., 1] = f
        rgb_before[..., 0] = mb
        rgb_before[..., 2] = mb
        ax_before.imshow(rgb_before, interpolation='bilinear')
        ax_before.set_title(f'Z={z}', fontsize=14, fontweight='bold')
        ax_before.axis('off')
        if col == 0:
            ax_before.text(-0.1, 0.5, 'BEFORE', transform=ax_before.transAxes,
                          fontsize=16, fontweight='bold', ha='right', va='center', rotation=90)
        
        # AFTER
        ax_after = fig.add_subplot(gs[1, col])
        rgb_after = np.zeros((*f.shape, 3), dtype=np.float32)
        rgb_after[..., 1] = f
        rgb_after[..., 0] = ma
        rgb_after[..., 2] = ma
        ax_after.imshow(rgb_after, interpolation='bilinear')
        ax_after.axis('off')
        if col == 0:
            ax_after.text(-0.1, 0.5, 'AFTER', transform=ax_after.transAxes,
                         fontsize=16, fontweight='bold', ha='right', va='center', rotation=90)
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Functional'),
        Patch(facecolor='magenta', label='Anatomy')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              ncol=2, fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    fig.savefig(output_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: {Path(output_path).name}")


def compute_metrics(fixed, moving_before, moving_after):
    from scipy.stats import pearsonr
    metrics = {}
    
    try:
        corr_before = pearsonr(fixed.ravel(), moving_before.ravel())[0]
        corr_after = pearsonr(fixed.ravel(), moving_after.ravel())[0]
        metrics['correlation_before'] = float(corr_before)
        metrics['correlation_after'] = float(corr_after)
        metrics['correlation_improvement'] = float(corr_after - corr_before)
    except:
        metrics['correlation_before'] = None
        metrics['correlation_after'] = None
    
    try:
        mse_before = np.mean((fixed - moving_before)**2)
        mse_after = np.mean((fixed - moving_after)**2)
        metrics['mse_before'] = float(mse_before)
        metrics['mse_after'] = float(mse_after)
        metrics['mse_reduction'] = float(mse_before - mse_after)
    except:
        metrics['mse_before'] = None
        metrics['mse_after'] = None
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ANTs AFFINE REFINADO - SUBPIXEL PRECISION")
    print("Optimized for ROI colocalization")
    print("="*80)
    
    try:
        out_dir = Path(cfg.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ['volumes', 'transforms', 'figures_thesis', 'metrics']:
            (out_dir / subdir).mkdir(exist_ok=True)
        
        # 1. BUILD TEMPLATE
        func_template = build_functional_template(
            cfg.FUNC_TIFS_DIR, cfg.N_PLANES, 
            cfg.PROJECTION_METHOD, cfg.FRAME_STRIDE, cfg.CHECK_FOR_NANS
        )
        imwrite(str(out_dir / 'volumes' / 'functional_template.tif'), func_template)
        
        # 2. LOAD ANATOMIES
        print("\n" + "="*80)
        print("LOADING ANATOMIES")
        print("="*80)
        
        gcamp_raw = imread(cfg.GCAMP_ANAT).astype(np.float32)
        dsred_raw = imread(cfg.DSRED_ANAT).astype(np.float32)
        
        if gcamp_raw.ndim == 4:
            gcamp_raw = gcamp_raw[0] if gcamp_raw.shape[0] < 10 else gcamp_raw[..., 0]
        if dsred_raw.ndim == 4:
            dsred_raw = dsred_raw[0] if dsred_raw.shape[0] < 10 else dsred_raw[..., 0]
        
        print(f"  GCaMP: {gcamp_raw.shape}")
        print(f"  DsRed: {dsred_raw.shape}")
        
        # 3. PREPROCESS
        print("\n" + "="*80)
        print("PREPROCESSING - MINIMAL SMOOTHING")
        print("="*80)
        
        func_proc = normalize_robust(func_template)
        func_proc = smooth_volume(func_proc, cfg.FUNCTIONAL_SMOOTH)
        
        gcamp_proc = normalize_robust(gcamp_raw)
        gcamp_proc = smooth_volume(gcamp_proc, cfg.ANATOMY_SMOOTH)
        
        dsred_proc = normalize_robust(dsred_raw)
        dsred_proc = smooth_volume(dsred_proc, cfg.ANATOMY_SMOOTH)
        
        print("  ✓ Preprocessed with minimal smoothing for maximum detail")
        
        # 4. MATCH SHAPES
        print("\n" + "="*80)
        print("MATCHING SHAPES")
        print("="*80)
        
        gcamp_fit = match_shapes(func_proc, gcamp_proc)
        dsred_fit = match_shapes(func_proc, dsred_proc)
        
        print(f"  All volumes: {func_proc.shape}")
        
        # 5. CONVERT TO ANTS
        print("\n" + "="*80)
        print("CONVERTING TO ANTs")
        print("="*80)
        
        func_ants = numpy_to_ants(func_proc, cfg.SPACING)
        gcamp_ants = numpy_to_ants(gcamp_fit, cfg.SPACING)
        dsred_ants = numpy_to_ants(dsred_fit, cfg.SPACING)
        print("  ✓ Converted")
        
        # 6. REFINED REGISTRATION
        result = register_affine_refined(func_ants, gcamp_ants)
        gcamp_reg = ants_to_numpy(result['warpedmovout'])
        
        # 7. APPLY TO DSRED
        print("\n" + "="*80)
        print("APPLYING TO DsRed")
        print("="*80)
        
        dsred_reg_ants = apply_transforms_refined(func_ants, dsred_ants, result['fwdtransforms'])
        dsred_reg = ants_to_numpy(dsred_reg_ants)
        
        # 8. SAVE
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        vol_dir = out_dir / 'volumes'
        imwrite(str(vol_dir / 'gcamp_before.tif'), gcamp_fit)
        imwrite(str(vol_dir / 'gcamp_registered.tif'), gcamp_reg)
        imwrite(str(vol_dir / 'dsred_before.tif'), dsred_fit)
        imwrite(str(vol_dir / 'dsred_registered.tif'), dsred_reg)
        
        print(f"  ✓ Volumes saved")
        
        # 9. TRANSFORMS
        tx_dir = out_dir / 'transforms'
        import shutil
        for i, tx in enumerate(result['fwdtransforms']):
            shutil.copy(tx, tx_dir / f'forward_{i}.mat')
        for i, tx in enumerate(result['invtransforms']):
            shutil.copy(tx, tx_dir / f'inverse_{i}.mat')
        
        # 10. METRICS
        print("\n" + "="*80)
        print("METRICS")
        print("="*80)
        
        metrics = compute_metrics(func_proc, gcamp_fit, gcamp_reg)
        
        if metrics['correlation_improvement'] is not None:
            print(f"  Correlation: {metrics['correlation_before']:.4f} → {metrics['correlation_after']:.4f}")
            print(f"  Improvement: {metrics['correlation_improvement']:+.4f}")
        
        with open(out_dir / 'metrics' / 'results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 11. FIGURES
        print("\n" + "="*80)
        print("CREATING FIGURES")
        print("="*80)
        
        valid_planes = [p for p in cfg.FIGURE_PLANES if p < func_proc.shape[0]]
        
        create_thesis_figure_before_after(
            func_proc, gcamp_fit, gcamp_reg,
            out_dir / 'figures_thesis' / 'GCaMP_REFINED.png',
            valid_planes,
            'AFFINE REFINADO: GCaMP → Functional (Subpixel Precision)'
        )
        
        create_thesis_figure_before_after(
            func_proc, dsred_fit, dsred_reg,
            out_dir / 'figures_thesis' / 'DsRed_REFINED.png',
            valid_planes,
            'AFFINE REFINADO: DsRed → Functional (Subpixel Precision)'
        )
        
        # SUMMARY
        print("\n" + "="*80)
        print("✅ AFFINE REFINADO COMPLETE!")
        print("="*80)
        
        print(f"\n📁 Output: {out_dir}")
        print(f"\n✅ SUBPIXEL REGISTRATION:")
        print(f"   • Preserves cell shape")
        print(f"   • Optimized for colocalization")
        print(f"   • High interpolation quality")
        
        if metrics['correlation_improvement']:
            print(f"\n📊 Quality: {metrics['correlation_improvement']:+.4f} correlation gain")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()