
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
# CONFIG - SyN OTIMIZADO
# ============================================================================

class Config:
    # ========== PATHS ==========
    FUNC_TIFS_DIR = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3\final_semnan"
    GCAMP_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\Gcamp6s_averaged\alignment_drift\anatomy.tif"
    DSRED_ANAT = r"D:\Dados 2photon\20251104gad1bdsred_hucH2BGCaMP6s\20251104gad1bdsred_hucH2BGCaMP6s_anatomy\dsred_averaged\reapplied_alignment\20251104gad1bdsred_hucH2BGCaMP6s_anatomy_.000000.000000.1_realigned.tif"
    OUTPUT_DIR = r"C:\Users\OSVALDO\Downloads\ANTS_SyN"
    
    # ========== TEMPLATE ==========
    N_PLANES = 180
    PLANE_PATTERN = r"aligned_p(\d+)_nan\.tif"
    PROJECTION_METHOD = "mean"
    FRAME_STRIDE = 2
    CHECK_FOR_NANS = True
    
    # ========== PREPROCESSING ==========
    FUNCTIONAL_SMOOTH = 0.5
    ANATOMY_SMOOTH = 1.0
    PERCENTILE_LOW = 1.0
    PERCENTILE_HIGH = 99.5
    
    # ========== SyN OTIMIZADO (mais rápido) ==========
    REG_TYPE = "SyN"  # SyNRA = SyN mais rápido
    
    VERBOSE = True
    
    # ========== SPACING ==========
    SPACING = (0.5481, 0.5935, 1.0)
    
    # ========== FIGURES ==========
    FIGURE_PLANES = [30, 60, 90, 120, 150]
    FIGURE_DPI = 300


cfg = Config()

# ============================================================================
# FUNÇÕES (IGUAIS)
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
        indices = list(range(0, n_pages, frame_stride)) if frame_stride > 1 else list(range(n_pages))
        
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
            raise RuntimeError(f"No valid frames")
        
        if nan_count > 0:
            print(f"    Skipped {nan_count} NaN frames")
        
        stack = np.array(frames)
        return np.mean(stack, axis=0).astype(np.float32) if method == "mean" else np.median(stack, axis=0).astype(np.float32)


def build_functional_template(tifs_dir, n_planes, method, frame_stride, check_nans):
    print("\n" + "="*80)
    print("BUILDING FUNCTIONAL TEMPLATE")
    print("="*80)
    
    plane_files = list_plane_files(tifs_dir, cfg.PLANE_PATTERN)
    first_idx = min(plane_files.keys())
    first_proj = project_plane_tif(plane_files[first_idx], method, frame_stride, check_nans)
    h, w = first_proj.shape
    
    n_planes_actual = min(n_planes, max(plane_files.keys()) + 1)
    volume = np.zeros((n_planes_actual, h, w), dtype=np.float32)
    
    print(f"Loading {n_planes_actual} planes...")
    for plane_idx in sorted(plane_files.keys()):
        if plane_idx >= n_planes_actual:
            break
        if plane_idx % 20 == 0:
            print(f"  Progress: {plane_idx+1}/{n_planes_actual}")
        volume[plane_idx] = project_plane_tif(plane_files[plane_idx], method, frame_stride, check_nans)
    
    print(f"✓ Template: {volume.shape}")
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
        moving = moving[(zm-zf)//2:(zm-zf)//2+zf]
    elif zm < zf:
        pad_z = zf - zm
        moving = np.pad(moving, ((pad_z//2, pad_z-pad_z//2), (0, 0), (0, 0)), mode='constant')
    
    if ym > yf:
        moving = moving[:, (ym-yf)//2:(ym-yf)//2+yf]
    elif ym < yf:
        pad_y = yf - ym
        moving = np.pad(moving, ((0, 0), (pad_y//2, pad_y-pad_y//2), (0, 0)), mode='constant')
    
    if xm > xf:
        moving = moving[:, :, (xm-xf)//2:(xm-xf)//2+xf]
    elif xm < xf:
        pad_x = xf - xm
        moving = np.pad(moving, ((0, 0), (0, 0), (pad_x//2, pad_x-pad_x//2)), mode='constant')
    
    return moving


def numpy_to_ants(vol_zyx, spacing_xyz):
    if vol_zyx.ndim != 3:
        raise ValueError(f"Must be 3D")
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))
    img = ants.from_numpy(vol_xyz.astype(np.float32))
    img.set_spacing(spacing_xyz)
    img.set_origin((0.0, 0.0, 0.0))
    return img


def ants_to_numpy(img):
    return np.transpose(img.numpy(), (2, 1, 0)).astype(np.float32)


# ============================================================================
# SyN REGISTRATION
# ============================================================================

def register_syn_optimized(fixed_ants, moving_ants):
    """
    SyN OTIMIZADO - mais rápido que SyN padrão
    ⚠️ ATENÇÃO: Pode distorcer células!
    """
    print(f"\n{'='*80}")
    print(f"ANTs SyNRA REGISTRATION (OPTIMIZED DEFORMABLE)")
    print(f"{'='*80}")
    
    print(f"\n⚠️  WARNING:")
    print(f"  • Deformable registration may distort cell shapes")
    print(f"  • Use for visual comparison only")
    print(f"  • For ROI colocalization → prefer AFFINE!")
    
    start_time = datetime.now()
    
    print(f"\nShapes:")
    print(f"  Fixed: {fixed_ants.shape}")
    print(f"  Moving: {moving_ants.shape}")
    
    print(f"\nStarting SyNRA registration...")
    print("Expected time: 30-40 minutes")
    print("(SyNRA is faster than full SyN)")
    
    try:
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform='SyNRA',  # Rapid SyN
            verbose=cfg.VERBOSE
        )
        
    except Exception as e:
        print(f"\n❌ SyNRA failed: {e}")
        print("\n🔄 Fallback: trying SyNOnly (slower)...")
        
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform='SyNOnly',
            verbose=cfg.VERBOSE
        )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Registration complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if 'warpedmovout' not in result:
        raise ValueError("Registration failed")
    
    print(f"  • Transforms: {len(result['fwdtransforms'])}")
    
    return result


def apply_transforms_syn(fixed_ants, moving_ants, transforms):
    """Apply SyN transforms"""
    print(f"\nApplying deformable transforms...")
    
    warped = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=transforms,
        interpolator='linear'
    )
    
    print("✓ Applied")
    return warped


# ============================================================================
# FIGURES
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
        Patch(facecolor='magenta', label='Anatomy (SyN - may be distorted!)')
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
        metrics['correlation_before'] = float(pearsonr(fixed.ravel(), moving_before.ravel())[0])
        metrics['correlation_after'] = float(pearsonr(fixed.ravel(), moving_after.ravel())[0])
        metrics['correlation_improvement'] = metrics['correlation_after'] - metrics['correlation_before']
    except:
        metrics['correlation_before'] = None
        metrics['correlation_after'] = None
    
    try:
        metrics['mse_before'] = float(np.mean((fixed - moving_before)**2))
        metrics['mse_after'] = float(np.mean((fixed - moving_after)**2))
        metrics['mse_reduction'] = metrics['mse_before'] - metrics['mse_after']
    except:
        metrics['mse_before'] = None
        metrics['mse_after'] = None
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ANTs SyN - DEFORMABLE REGISTRATION (FOR COMPARISON)")
    print("="*80)
    print("\n⚠️  WARNING: This is for visual comparison only!")
    print("For ROI colocalization, use AFFINE instead!")
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
        
        # 2. LOAD
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
        print("PREPROCESSING")
        print("="*80)
        
        func_proc = normalize_robust(func_template)
        func_proc = smooth_volume(func_proc, cfg.FUNCTIONAL_SMOOTH)
        gcamp_proc = normalize_robust(gcamp_raw)
        gcamp_proc = smooth_volume(gcamp_proc, cfg.ANATOMY_SMOOTH)
        dsred_proc = normalize_robust(dsred_raw)
        dsred_proc = smooth_volume(dsred_proc, cfg.ANATOMY_SMOOTH)
        print("  ✓ Done")
        
        # 4. MATCH
        print("\n" + "="*80)
        print("MATCHING SHAPES")
        print("="*80)
        
        gcamp_fit = match_shapes(func_proc, gcamp_proc)
        dsred_fit = match_shapes(func_proc, dsred_proc)
        print(f"  All: {func_proc.shape}")
        
        # 5. CONVERT
        print("\n" + "="*80)
        print("CONVERTING TO ANTs")
        print("="*80)
        
        func_ants = numpy_to_ants(func_proc, cfg.SPACING)
        gcamp_ants = numpy_to_ants(gcamp_fit, cfg.SPACING)
        dsred_ants = numpy_to_ants(dsred_fit, cfg.SPACING)
        print("  ✓ Done")
        
        # 6. SyN REGISTRATION
        result = register_syn_optimized(func_ants, gcamp_ants)
        gcamp_reg = ants_to_numpy(result['warpedmovout'])
        
        # 7. APPLY TO DSRED
        print("\n" + "="*80)
        print("APPLYING TO DsRed")
        print("="*80)
        
        dsred_reg_ants = apply_transforms_syn(func_ants, dsred_ants, result['fwdtransforms'])
        dsred_reg = ants_to_numpy(dsred_reg_ants)
        
        # 8. SAVE
        print("\n" + "="*80)
        print("SAVING")
        print("="*80)
        
        vol_dir = out_dir / 'volumes'
        imwrite(str(vol_dir / 'gcamp_before.tif'), gcamp_fit)
        imwrite(str(vol_dir / 'gcamp_registered_SyN.tif'), gcamp_reg)
        imwrite(str(vol_dir / 'dsred_before.tif'), dsred_fit)
        imwrite(str(vol_dir / 'dsred_registered_SyN.tif'), dsred_reg)
        print("  ✓ Volumes")
        
        # 9. TRANSFORMS
        tx_dir = out_dir / 'transforms'
        import shutil
        for i, tx in enumerate(result['fwdtransforms']):
            shutil.copy(tx, tx_dir / f'forward_{i}.mat')
        for i, tx in enumerate(result['invtransforms']):
            shutil.copy(tx, tx_dir / f'inverse_{i}.mat')
        print("  ✓ Transforms")
        
        # 10. METRICS
        print("\n" + "="*80)
        print("METRICS")
        print("="*80)
        
        metrics = compute_metrics(func_proc, gcamp_fit, gcamp_reg)
        
        if metrics['correlation_improvement']:
            print(f"  Correlation: {metrics['correlation_before']:.4f} → {metrics['correlation_after']:.4f}")
            print(f"  Improvement: {metrics['correlation_improvement']:+.4f}")
        
        with open(out_dir / 'metrics' / 'results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 11. FIGURES
        print("\n" + "="*80)
        print("FIGURES")
        print("="*80)
        
        valid_planes = [p for p in cfg.FIGURE_PLANES if p < func_proc.shape[0]]
        
        create_thesis_figure_before_after(
            func_proc, gcamp_fit, gcamp_reg,
            out_dir / 'figures_thesis' / 'GCaMP_SyN.png',
            valid_planes,
            'SyN DEFORMABLE: GCaMP (⚠️ cells may be distorted)'
        )
        
        create_thesis_figure_before_after(
            func_proc, dsred_fit, dsred_reg,
            out_dir / 'figures_thesis' / 'DsRed_SyN.png',
            valid_planes,
            'SyN DEFORMABLE: DsRed (⚠️ cells may be distorted)'
        )
        
        # SUMMARY
        print("\n" + "="*80)
        print("✅ SyN COMPLETE!")
        print("="*80)
        
        print(f"\n📁 Output: {out_dir}")
        print(f"\n⚠️  REMEMBER:")
        print(f"   • SyN gives better visual alignment")
        print(f"   • BUT may distort individual cells")
        print(f"   • For ROI colocalization → USE AFFINE!")
        
        if metrics['correlation_improvement']:
            print(f"\n📊 Quality: {metrics['correlation_improvement']:+.4f} correlation gain")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()