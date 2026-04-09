# run_from_existing_withinplane_bins.py
# Continua o pipeline SEM voltar a correr withinPlane_correction.
# Usa diretamente os *_reg.bin já existentes em withinPlane_dir.

import os
import sys
from glob import glob
from suite2p import default_ops

# Ajusta este import ao teu ficheiro real:
# (é o mesmo import que tens no teu runner antigo) :contentReference[oaicite:2]{index=2}
from motionCorrection_suite2p_manel import (
    betweenPlanes_correction,
    find_badFrames,
    write_clean_tiffs,
)

def main():
    # =========================
    # EDITA AQUI (paths e params)
    # =========================
    input_folder = r"D:\User data\InesMarques\2photon\20251104gad1bdsred_hucH2BGCaMP6s"
    output_root  = os.path.join(input_folder, "suite2p_1812")

    # Pasta onde JÁ tens os pXX_reg.bin (e possivelmente pXX.bin)
    withinPlane_dir = os.path.join(output_root, "withinPlane")

    # Onde o between-planes vai escrever
    betweenPlane_dir = os.path.join(output_root, "betweenPlane")

    # Onde vais guardar os TIFFs limpos finais
    final_dir = os.path.join(output_root, "motionCorrected")

    # Info do dataset
    nplanes = 180
    og_data_info = {
        "Ly": 464,
        "Lx": 720,
        "frames_per_plane": 250,
        "total_planes": nplanes,
    }

    # Intervalo de planos (mantém igual ao teu setup)
    startP = 1
    endP   = 180

    # Crops para between-planes (iguais ao teu runner antigo) :contentReference[oaicite:3]{index=3}
    x_crop = 100
    y_crop = 100

    # Threshold badframes
    threshold = 2

    # Parâmetro smooth do between-planes (tu tinhas 0 no runner) :contentReference[oaicite:4]{index=4}
    smooth = 0

    # =========================
    # OPS (podes manter simples)
    # =========================
    ops = default_ops()
    ops.update({
        "nonrigid": True,
        "two_step_registration": True,
        "maxregshift": 0.1,
        "maxregshiftNR": 5,
        "nimg_init": 100,
        "do_bidiphase": True,
        "bidiphase": 0,
        "bidi_corrected": False,
        "th_badframes": 0.7,
    })

    # =========================
    # CHECKS
    # =========================
    if not os.path.isdir(withinPlane_dir):
        print(f"[ERROR] withinPlane_dir não existe: {withinPlane_dir}")
        sys.exit(1)

    reg_bins = sorted(glob(os.path.join(withinPlane_dir, "*_reg.bin")))
    if len(reg_bins) == 0:
        print(f"[ERROR] Não encontrei *_reg.bin em: {withinPlane_dir}")
        print("Confere se os teus ficheiros se chamam tipo p01_reg.bin, p02_reg.bin, ...")
        sys.exit(1)

    os.makedirs(betweenPlane_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    print(f"[OK] Encontrei {len(reg_bins)} ficheiros *_reg.bin em withinPlane_dir.")
    print("[INFO] Vou saltar within-plane e continuar a partir do between-planes.")

    # =========================
    # 1) Between-planes correction
    # =========================
    print("\n=== Running between-planes correction (from existing withinPlane bins) ===")
    betweenPlanes_correction(
        withinPlane_dir,  # <-- usa DIRETAMENTE os bins que já tens aqui :contentReference[oaicite:5]{index=5}
        og_data_info,
        ops,
        betweenPlane_dir,
        x_crop,
        y_crop,
        startP,
        endP,
        smooth,
    )

    # betweenPlanes_correction cria output_dir/registeredBin :contentReference[oaicite:6]{index=6}
    reg_dir = os.path.join(betweenPlane_dir, "registeredBin")
    if not os.path.isdir(reg_dir):
        print(f"\n[ERROR] Não encontrei {reg_dir}")
        print("Confere onde o betweenPlanes_correction está a escrever os bins (nome da pasta).")
        sys.exit(1)

    # =========================
    # 2) Find bad frames (USING SHIFTS, NOT CORRELATION!)
    # =========================
    print("\n=== Finding bad frames ===")
    # offsets_dir aponta para withinPlane onde os offsets foram guardados
    # Isto usa os SHIFTS do registo (não correlação) para detetar bad frames
    # É independente da atividade neural - não remove frames com respostas!
    find_badFrames(reg_dir, og_data_info, reg_dir, threshold, offsets_dir=withinPlane_dir)

    # =========================
    # 3) Write clean TIFFs
    # =========================
    print("\n=== Writing clean TIFFs ===")
    write_clean_tiffs(reg_dir, og_data_info, final_dir)

    print("\nDONE ✅")
    print("Between-planes bins:", reg_dir)
    print("Final motion-corrected TIFFs:", final_dir)

if __name__ == "__main__":
    main()