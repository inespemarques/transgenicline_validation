#!/usr/bin/env python
"""
PIPELINE COMPLETO DE MOTION CORRECTION - 2-PHOTON ZEBRAFISH
============================================================
Parte dos TIFFs originais (250 frames cada) e faz:
1. Within-plane motion correction (Suite2p)
2. Cálculo de offsets REAIS (original vs registado)
3. Between-plane alignment
4. Bad frame detection em 2 PASSOS (usando SHIFTS, não correlação!)
5. TIFFs finais com NaNs nas bad frames

NOVO: Detecção de bad frames melhorada com 3 critérios:
  - Outliers extremos (>50px) - artefactos severos como riscos
  - Outliers estatísticos (>mean + 1.5 SD) - movimento anormal
  - Outliers moderados (>10px) - movimento suspeito

IMPORTANTE: Offsets são calculados comparando TIFFs ORIGINAIS vs frames registadas
           para ter os shifts REAIS aplicados pelo Suite2p
"""

import os
import sys
from suite2p import default_ops

# Importa as funções do teu módulo
from motionCorrection_suite2p_manel_UPDATED import (
    withinPlane_correction,
    betweenPlanes_correction,
    find_badFrames,
    write_clean_tiffs,
    convert_badframes_to_bool_array,
)

# TAMBÉM vamos precisar de calcular os offsets APÓS o within-plane
import numpy as np
from natsort import natsorted
from skimage.registration import phase_cross_correlation
from suite2p.io import BinaryFile
import tifffile

# =========================
# CONFIGURAÇÃO - EDITA AQUI
# =========================

# Pasta com os TIFFs ORIGINAIS (p01.tif, p02.tif, ... com 250 frames cada)
ORIGINAL_TIFF_DIR = r"D:\Dados 2photon\20251030gad1bdsred_hucH2BGCaMP6s\tiffs_concatenados"
OUTPUT_ROOT = r"D:\Dados 2photon\20251030gad1bdsred_hucH2BGCaMP6s\suite2p_NOVOthr3"  # ← MUDEI para thr3!

# Info do dataset
DATA_INFO = {
    "Ly": 464,
    "Lx": 720,
    "frames_per_plane": 250,
    "total_planes": 180,
}

# Planos a processar
START_PLANE = 1
END_PLANE = 171

# Parâmetros de registo
PAD_PIXELS = 0          # Sem padding (adequado para zebrafish)
X_CROP = 100            # Crop para between-planes (ajusta conforme ROI)
Y_CROP = 100
SMOOTH = 0              # Sem smoothing (movimentos são graduais)

# ═══════════════════════════════════════════════════════════
# PARÂMETROS DE DETECÇÃO DE BAD FRAMES - 2-STEP METHOD
# ═══════════════════════════════════════════════════════════
BAD_FRAME_THRESHOLD_SD = 1.5      # Threshold estatístico (desvio padrão)
BAD_FRAME_ABSOLUTE = 50           # Qualquer shift >50px é automático bad frame
BAD_FRAME_MODERATE = 10           # Threshold moderado (ajusta conforme necessário)
                                  # Valores sugeridos para zebrafish 2-photon:
                                  # - 5-10px: movimento normal
                                  # - 10-20px: suspeito mas pode ser OK
                                  # - >20px: provavelmente mau
                                  # - >50px: definitivamente artefacto

# =========================
# FUNÇÕES AUXILIARES
# =========================

def calculate_offsets_from_original_tiffs(
    original_tiff_dir,
    registered_bin_dir,
    output_dir,
    data_info,
    start_plane,
    end_plane
):
    """
    Calcula os offsets VERDADEIROS comparando:
    - TIFFs originais (antes de qualquer registo)
    - Bins registados (depois do within-plane)
    
    Isto dá os shifts REAIS que foram aplicados pelo Suite2p!
    
    IMPORTANTE: Esta é a métrica correcta para detectar bad frames porque:
      - Independente da actividade neural
      - Mede movimento real, não diferenças de imagem
      - Artefactos (riscos, sujidade) causam shifts enormes
    """
    Ly = data_info['Ly']
    Lx = data_info['Lx']
    frames_per_plane = data_info['frames_per_plane']
    
    print("\n" + "="*70)
    print("CALCULANDO OFFSETS: TIFFs ORIGINAIS vs BINS REGISTADOS")
    print("="*70)
    print(f"Original TIFFs: {original_tiff_dir}")
    print(f"Registered bins: {registered_bin_dir}")
    print(f"Output: {output_dir}")
    print("="*70)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_processed = 0
    
    for p in range(start_plane, end_plane + 1):
        try:
            # Paths
            tiff_path = os.path.join(original_tiff_dir, f"p{p:02d}.tif")
            reg_bin_path = os.path.join(registered_bin_dir, f"p{p:02d}_reg.bin")
            
            if not os.path.exists(tiff_path):
                print(f"  Plane {p:3d}: ✗ TIFF original não encontrado")
                continue
            
            if not os.path.exists(reg_bin_path):
                print(f"  Plane {p:3d}: ✗ Bin registado não encontrado")
                continue
            
            # 1. Carrega TIFF original
            original_stack = tifffile.imread(tiff_path)  # shape: (frames, Ly, Lx)
            
            if original_stack.shape[0] != frames_per_plane:
                print(f"  Plane {p:3d}: ✗ TIFF tem {original_stack.shape[0]} frames, esperava {frames_per_plane}")
                continue
            
            # 2. Carrega bin registado e calcula referência
            f_reg = BinaryFile(Ly=Ly, Lx=Lx, filename=reg_bin_path, n_frames=frames_per_plane)
            reference = np.mean([f_reg[i] for i in range(frames_per_plane)], axis=0)
            f_reg.close()
            
            # 3. Calcula shift de cada frame original vs referência registada
            yoff = np.zeros(frames_per_plane, dtype=np.float32)
            xoff = np.zeros(frames_per_plane, dtype=np.float32)
            
            for i in range(frames_per_plane):
                frame_orig = original_stack[i].astype(np.float32)
                
                # Phase cross-correlation com upsampling para sub-pixel precision
                shift, error, diffphase = phase_cross_correlation(
                    reference, frame_orig, upsample_factor=10
                )
                
                yoff[i] = shift[0]
                xoff[i] = shift[1]
            
            # 4. Guarda offsets
            output_file = os.path.join(output_dir, f"p{p:02d}_offsets.npy")
            np.save(output_file, {
                'yoff': yoff,
                'xoff': xoff,
            })
            
            # 5. Estatísticas
            shift_mag = np.sqrt(yoff**2 + xoff**2)
            
            print(f"  Plane {p:3d}: ✓ {frames_per_plane} frames | "
                  f"shifts: mean={shift_mag.mean():.2f}, max={shift_mag.max():.2f} px")
            
            n_processed += 1
            
        except Exception as e:
            print(f"  Plane {p:3d}: ✗ Erro - {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print(f"✓ Processados {n_processed}/{end_plane - start_plane + 1} planos")
    print(f"✓ Offsets guardados em: {output_dir}")
    print()
    
    return n_processed

# =========================
# PIPELINE PRINCIPAL
# =========================

def main():
    # Verifica se TIFFs originais existem
    if not os.path.isdir(ORIGINAL_TIFF_DIR):
        print(f"❌ ERRO: Pasta de TIFFs originais não existe: {ORIGINAL_TIFF_DIR}")
        sys.exit(1)
    
    tiff_files = [f for f in os.listdir(ORIGINAL_TIFF_DIR) if f.lower().endswith('.tif')]
    if len(tiff_files) == 0:
        print(f"❌ ERRO: Nenhum TIFF encontrado em: {ORIGINAL_TIFF_DIR}")
        sys.exit(1)
    
    print(f"✓ Encontrados {len(tiff_files)} TIFFs em {ORIGINAL_TIFF_DIR}")
    
    # Cria estrutura de pastas
    withinPlane_dir = os.path.join(OUTPUT_ROOT, "withinPlane")
    betweenPlane_dir = os.path.join(OUTPUT_ROOT, "betweenPlane")
    final_dir = os.path.join(OUTPUT_ROOT, "final")
    
    os.makedirs(withinPlane_dir, exist_ok=True)
    os.makedirs(betweenPlane_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    # Ops do Suite2p - otimizado para zebrafish 2-photon
    ops = default_ops()
    ops.update({
        "nonrigid": True,              # Essencial para zebrafish (curvatura)
        "two_step_registration": True, # Melhor precisão
        "maxregshift": 0.1,            # 10% do FOV (adequado para larvas)
        "maxregshiftNR": 5,            # Não-rígido: 5px (movimentos locais)
        "nimg_init": 100,              # Frames para referência inicial
        "do_bidiphase": True,          # Corrige artefacto bidirecional
        "bidiphase": 0,                # Auto-detecta
        "bidi_corrected": False,
        "th_badframes": 0.7,           # Threshold conservador (não usado pelo nosso método)
    })
    
    print("\n" + "="*70)
    print("PIPELINE DE MOTION CORRECTION - ZEBRAFISH 2-PHOTON")
    print("="*70)
    print(f"Input:  {ORIGINAL_TIFF_DIR}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Planos: {START_PLANE} a {END_PLANE}")
    print(f"\nPARÂMETROS DE DETECÇÃO DE BAD FRAMES:")
    print(f"  - Statistical threshold: {BAD_FRAME_THRESHOLD_SD} SD")
    print(f"  - Absolute threshold: {BAD_FRAME_ABSOLUTE} px")
    print(f"  - Moderate threshold: {BAD_FRAME_MODERATE} px")
    print("="*70)
    
    # =========================
    # 1. WITHIN-PLANE CORRECTION
    # =========================
    print("\n" + "="*70)
    print("PASSO 1: WITHIN-PLANE MOTION CORRECTION")
    print("="*70)
    print("O que faz:")
    print("  - Corrige movimento dentro de cada plano (XY)")
    print("  - Usa Suite2p com registo não-rígido")
    print("  - Essencial para compensar curvatura do peixe")
    print("="*70)
    
    withinPlane_correction(
        ORIGINAL_TIFF_DIR,
        DATA_INFO,
        ops,
        withinPlane_dir,
        START_PLANE,
        END_PLANE,
        PAD_PIXELS
    )
    
    print("\n✓ Within-plane correction completo!")
    print(f"  Output: {withinPlane_dir}")
    
    # =========================
    # 2. CALCULAR OFFSETS VERDADEIROS
    # =========================
    print("\n" + "="*70)
    print("PASSO 2: CALCULAR OFFSETS (ORIGINAIS vs REGISTADOS)")
    print("="*70)
    print("O que faz:")
    print("  - Compara frames originais com frames registados")
    print("  - Calcula os shifts REAIS aplicados pelo Suite2p")
    print("  - Estes shifts são a métrica CORRECTA para bad frames")
    print("="*70)
    
    n_offsets = calculate_offsets_from_original_tiffs(
        ORIGINAL_TIFF_DIR,
        withinPlane_dir,
        withinPlane_dir,  # Guarda offsets na mesma pasta
        DATA_INFO,
        START_PLANE,
        END_PLANE
    )
    
    if n_offsets == 0:
        print("❌ ERRO: Nenhum offset foi calculado!")
        sys.exit(1)
    
    print("✓ Offsets calculados com sucesso!")
    
    # =========================
    # 3. BETWEEN-PLANE CORRECTION
    # =========================
    print("\n" + "="*70)
    print("PASSO 3: BETWEEN-PLANE ALIGNMENT")
    print("="*70)
    print("O que faz:")
    print("  - Alinha planos consecutivos (eixo Z)")
    print("  - Cada plano é alinhado ao anterior")
    print("  - Corrige drift lento ao longo do volume")
    print("="*70)
    
    betweenPlanes_correction(
        withinPlane_dir,
        DATA_INFO,
        ops,
        betweenPlane_dir,
        X_CROP,
        Y_CROP,
        START_PLANE,
        END_PLANE,
        SMOOTH
    )
    
    reg_dir = os.path.join(betweenPlane_dir, "registeredBin")
    
    print("\n✓ Between-plane correction completo!")
    print(f"  Output: {reg_dir}")
    
    # =========================
    # 4. DETECTAR BAD FRAMES (MÉTODO EM 2 PASSOS!)
    # =========================
    print("\n" + "="*70)
    print("PASSO 4: DETECTAR BAD FRAMES (2-step method)")
    print("="*70)
    print("MÉTODO MELHORADO:")
    print(f"  1. Remove outliers extremos (>{BAD_FRAME_ABSOLUTE}px)")
    print(f"  2. Recalcula estatísticas sem outliers")
    print(f"  3. Aplica threshold relativo ({BAD_FRAME_THRESHOLD_SD} SD)")
    print(f"  4. Marca também shifts >{BAD_FRAME_MODERATE}px")
    print("\nVANTAGENS:")
    print("  - Independente da actividade neural")
    print("  - Detecta artefactos severos (riscos, sujidade)")
    print("  - Detecta movimento anormal sem falsos positivos")
    print("="*70)
    
    find_badFrames(
        reg_dir,
        DATA_INFO,
        reg_dir,
        threshold_sd=BAD_FRAME_THRESHOLD_SD,
        absolute_threshold=BAD_FRAME_ABSOLUTE,
        moderate_threshold=BAD_FRAME_MODERATE,
        offsets_dir=withinPlane_dir  # ← Usa os offsets que calculámos!
    )
    
    print("\n✓ Bad frames detectados!")
    
    # =========================
    # 5. CONVERTER PARA BOOLEAN ARRAY
    # =========================
    print("\n" + "="*70)
    print("PASSO 5: CONVERTER BAD FRAMES PARA BOOLEAN ARRAY")
    print("="*70)
    print("O que faz:")
    print("  - Converte dicionário para array booleano (180 x 250)")
    print("  - Formato compatível com pipelines de análise")
    print("  - True = bad frame, False = good frame")
    print("="*70)
    
    convert_badframes_to_bool_array(reg_dir, DATA_INFO, reg_dir)
    
    print("\n✓ Boolean array criado!")
    
    # =========================
    # 6. ESCREVER TIFFS LIMPOS
    # =========================
    print("\n" + "="*70)
    print("PASSO 6: ESCREVER TIFFs FINAIS (com NaN nas bad frames)")
    print("="*70)
    print("O que faz:")
    print("  - Cria TIFFs finais por plano")
    print("  - Bad frames são substituídos por NaN")
    print("  - Prontos para segmentação (Suite2p, CaImAn, etc.)")
    print("="*70)
    
    write_clean_tiffs(reg_dir, DATA_INFO, final_dir)
    
    print("\n✓ TIFFs finais escritos!")
    
    # =========================
    # RESUMO FINAL
    # =========================
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETO!")
    print("="*70)
    print(f"📁 Within-plane bins:  {withinPlane_dir}")
    print(f"📁 Between-plane bins: {reg_dir}")
    print(f"📁 TIFFs finais:       {final_dir}")
    print(f"📁 Bad frames dict:    {os.path.join(reg_dir, 'badFrames_corr_all_planes.npy')}")
    print(f"📁 Bad frames bool:    {os.path.join(reg_dir, 'bad_masks_bool.npy')}")
    print(f"📁 Shift statistics:   {os.path.join(reg_dir, 'shift_stats_all_planes.npy')}")
    print("="*70)
    
    # Mostra estatísticas de bad frames
    try:
        stats = np.load(os.path.join(reg_dir, 'shift_stats_all_planes.npy'), allow_pickle=True).item()
        total_bad = sum(s['n_bad'] for s in stats.values())
        total_frames = len(stats) * DATA_INFO['frames_per_plane']
        total_extreme = sum(s['n_extreme'] for s in stats.values())
        total_statistical = sum(s['n_statistical'] for s in stats.values())
        total_moderate = sum(s['n_moderate'] for s in stats.values())
        
        print(f"\n📊 ESTATÍSTICAS DE BAD FRAMES:")
        print(f"   Total bad frames: {total_bad}/{total_frames} ({100*total_bad/total_frames:.2f}%)")
        print(f"\n   Breakdown por critério:")
        print(f"   - Extreme outliers (>{BAD_FRAME_ABSOLUTE}px): {total_extreme}")
        print(f"   - Statistical outliers ({BAD_FRAME_THRESHOLD_SD} SD): {total_statistical}")
        print(f"   - Moderate outliers (>{BAD_FRAME_MODERATE}px): {total_moderate}")
        
        # Mostra alguns exemplos
        print(f"\n   Exemplos de planos:")
        for i, (key, s) in enumerate(list(stats.items())[:5]):
            print(f"   - {key}: {s['n_bad']:3d} bad frames | "
                  f"mean shift={s['mean_shift']:.2f}, max={s['max_shift']:.2f} px | "
                  f"[extreme:{s['n_extreme']}, stat:{s['n_statistical']}, mod:{s['n_moderate']}]")
    except Exception as e:
        print(f"\n⚠️  Não foi possível carregar estatísticas: {e}")
    
    print("\n" + "="*70)
    print("🎉 PRONTO PARA SEGMENTAÇÃO!")
    print("="*70)
    print(f"📂 Usa os TIFFs em: {final_dir}")
    print(f"📊 Bad frames boolean: {os.path.join(reg_dir, 'bad_masks_bool.npy')}")
    print("\nPRÓXIMOS PASSOS:")
    print("  1. Valida visualmente alguns planos em ImageJ/Fiji")
    print("  2. Verifica se bad frames detetados fazem sentido")
    print("  3. Ajusta thresholds se necessário e re-corre só o PASSO 4-6")
    print("  4. Prossegue para segmentação (Suite2p, CaImAn, etc.)")
    print("="*70)

if __name__ == "__main__":
    main()