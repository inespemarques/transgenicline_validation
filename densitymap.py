#!/usr/bin/env python3
"""
GABAergic density maps — ALL FISH combined + individual.

Main figure:  Combined 4-fish density (DsRed, InSitu, Coexpression)
Appendix:     Per-fish density panels

Input: mask TIFFs + classification CSVs per fish per block.
"""

import os, sys, warnings, glob
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = r"C:\Users\OSVALDO\Downloads\results\density_maps_all_fish"
os.makedirs(OUT_DIR, exist_ok=True)

PIXEL_SIZE_XY = 0.1        # µm/pixel
DENSITY_SIGMA  = 30         # Gaussian smoothing σ (pixels)
AREA_MIN_PX    = 200        # min label area

# ── Fish definitions ────────────────────────────────────────────
# Each fish: mask_dir, csv_dir
FISH = {
    "Fish 3": {
        "mask_dir": r"D:\fish3_masks",
        "csv_dir":  r"C:\Users\OSVALDO\Downloads\fish3class",
    },
    "Fish 5": {
        "mask_dir": r"D:\fish5_masks",
        "csv_dir":  r"C:\Users\OSVALDO\Downloads\fish5class",
    },
    "Fish 6": {
        "mask_dir": r"D:\fish6_masks",
        "csv_dir":  r"C:\Users\OSVALDO\Downloads\drive-download-20260207T144317Z-1-001",
    },
    "Fish 7": {
        "mask_dir": r"D:\fish7_masks",
        "csv_dir":  r"C:\Users\OSVALDO\Downloads\drive-download-20260207T140751Z-1-001",
    },
}


# =============================================================================
# AUTO-DISCOVERY: pair mask TIFFs with classification CSVs
# =============================================================================

def extract_block_key(filename):
    """Extract numeric block identifier like '0a130', '110a240', etc."""
    import re
    # Match patterns like 0a130, 110a240, 220a350, etc.
    m = re.search(r'(\d+a\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def discover_pairs(mask_dir, csv_dir):
    """Auto-discover matching mask TIFF and CSV pairs by block key."""
    pairs = []

    # Find all TIF files in mask dir
    mask_files = {}
    if os.path.exists(mask_dir):
        for f in os.listdir(mask_dir):
            if f.lower().endswith(('.tif', '.tiff')):
                key = extract_block_key(f)
                if key:
                    mask_files[key] = os.path.join(mask_dir, f)

    # Find all CSV files in csv dir
    csv_files = {}
    if os.path.exists(csv_dir):
        for f in os.listdir(csv_dir):
            if f.lower().endswith('.csv'):
                key = extract_block_key(f)
                if key:
                    csv_files[key] = os.path.join(csv_dir, f)

    # Match by block key
    for key in sorted(set(mask_files.keys()) & set(csv_files.keys())):
        pairs.append({
            "block": key,
            "mask": mask_files[key],
            "csv": csv_files[key],
        })

    return pairs, mask_files, csv_files


# =============================================================================
# CENTROID EXTRACTION (fast, vectorised)
# =============================================================================

def extract_centroids(mask_path):
    """Extract 2D centroids (y, x) from 3D label volume."""
    mask = tiff.imread(mask_path)
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]

    nz, ny, nx = mask.shape
    flat = mask.ravel()

    # Coordinate arrays (only Y and X for XY projection)
    yy = np.tile(np.arange(ny).reshape(-1, 1), (nz, 1, nx)).ravel().astype(np.float64)
    xx = np.tile(np.arange(nx).reshape(1, -1), (nz, ny, 1)).ravel().astype(np.float64)

    max_label = int(flat.max())
    if max_label == 0:
        return {}

    counts = np.bincount(flat, minlength=max_label + 1)
    sum_y  = np.bincount(flat, weights=yy, minlength=max_label + 1)
    sum_x  = np.bincount(flat, weights=xx, minlength=max_label + 1)

    centroids = {}
    labels = np.where(counts >= AREA_MIN_PX)[0]
    labels = labels[labels > 0]
    for lbl in labels:
        centroids[int(lbl)] = (sum_y[lbl] / counts[lbl],
                               sum_x[lbl] / counts[lbl])
    return centroids


# =============================================================================
# LOAD + MATCH
# =============================================================================

def load_and_match(csv_path, centroids):
    df = pd.read_csv(csv_path)
    for col in ["DsRed_positive", "InSitu_positive"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().isin(["true", "1"])

    rows = []
    for _, row in df.iterrows():
        lbl = int(row["label"])
        if lbl in centroids:
            cy, cx = centroids[lbl]
            rows.append({
                "label": lbl, "y": cy, "x": cx,
                "DsRed_positive":  bool(row.get("DsRed_positive", False)),
                "InSitu_positive": bool(row.get("InSitu_positive", False)),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# PROCESS ONE FISH
# =============================================================================

def process_fish(fish_name, mask_dir, csv_dir):
    print(f"\n{'─' * 60}")
    print(f"  {fish_name}")
    print(f"{'─' * 60}")

    pairs, mask_files, csv_files = discover_pairs(mask_dir, csv_dir)

    if not pairs:
        print(f"  ⚠ No matching pairs found!")
        print(f"    Mask keys: {sorted(mask_files.keys())}")
        print(f"    CSV keys:  {sorted(csv_files.keys())}")
        return pd.DataFrame()

    print(f"  Found {len(pairs)} block pairs: {[p['block'] for p in pairs]}")

    all_dfs = []
    for p in pairs:
        print(f"    Block {p['block']}...", end=" ")
        centroids = extract_centroids(p["mask"])
        df = load_and_match(p["csv"], centroids)
        if len(df) > 0:
            df["block"] = p["block"]
            all_dfs.append(df)
            print(f"{len(df)} cells matched")
        else:
            print("0 cells (check files)")

    if not all_dfs:
        return pd.DataFrame()

    df_fish = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate overlapping blocks
    df_fish["yr"] = df_fish["y"].round(0).astype(int)
    df_fish["xr"] = df_fish["x"].round(0).astype(int)
    n0 = len(df_fish)
    df_fish = df_fish.drop_duplicates(subset=["yr", "xr"], keep="first")
    df_fish = df_fish.drop(columns=["yr", "xr"])
    if len(df_fish) < n0:
        print(f"  Deduplicated: {n0} → {len(df_fish)}")

    n = len(df_fish)
    nd = df_fish["DsRed_positive"].sum()
    ni = df_fish["InSitu_positive"].sum()
    nb = (df_fish["DsRed_positive"] & df_fish["InSitu_positive"]).sum()
    print(f"  Total: {n} cells | DsRed+: {nd} | InSitu+: {ni} | Both: {nb}")

    return df_fish


# =============================================================================
# DENSITY HELPERS
# =============================================================================

def make_density(ys, xs, shape, sigma=DENSITY_SIGMA):
    img = np.zeros(shape, dtype=np.float64)
    yi = np.clip(np.round(ys).astype(int), 0, shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, shape[1] - 1)
    np.add.at(img, (yi, xi), 1)
    return gaussian_filter(img, sigma=sigma)


def add_colorbar(ax, im, label="", size="4%", pad=0.08):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    return cb


# =============================================================================
# FIGURE: COMBINED 4-FISH (main body)
# =============================================================================

def plot_combined(all_fish_data, out_dir):
    """
    Main thesis figure: 3 rows (DsRed frac, InSitu frac, Coexpr frac) × 4 cols (fish).
    """
    fish_names = sorted(all_fish_data.keys())
    n_fish = len(fish_names)

    # Compute per-fish densities
    fish_densities = {}
    for fname in fish_names:
        df = all_fish_data[fname]
        ymax = int(df["y"].max()) + 50
        xmax = int(df["x"].max()) + 50
        shape = (ymax, xmax)

        d_all    = make_density(df["y"].values, df["x"].values, shape)
        d_dsred  = make_density(df.loc[df["DsRed_positive"], "y"].values,
                                df.loc[df["DsRed_positive"], "x"].values, shape)
        d_insitu = make_density(df.loc[df["InSitu_positive"], "y"].values,
                                df.loc[df["InSitu_positive"], "x"].values, shape)
        d_both   = make_density(
            df.loc[df["DsRed_positive"] & df["InSitu_positive"], "y"].values,
            df.loc[df["DsRed_positive"] & df["InSitu_positive"], "x"].values, shape)

        eps = 1e-6
        thr = d_all.max() * 0.015
        mask_low = d_all < thr

        f_dsred  = np.where(mask_low, np.nan, d_dsred  / (d_all + eps))
        f_insitu = np.where(mask_low, np.nan, d_insitu / (d_all + eps))
        f_both   = np.where(mask_low, np.nan, d_both   / (d_all + eps))

        extent = [0, xmax * PIXEL_SIZE_XY, ymax * PIXEL_SIZE_XY, 0]

        fish_densities[fname] = {
            "f_dsred": f_dsred, "f_insitu": f_insitu, "f_both": f_both,
            "d_all": d_all, "d_dsred": d_dsred, "d_insitu": d_insitu,
            "extent": extent, "shape": shape,
        }

    # ── Main figure: fraction maps ───────────────────────────────
    # Custom colormaps
    cmap_dsred  = "YlOrRd"
    cmap_insitu = "YlGn"
    cmap_both   = "YlGnBu"

    row_labels = [
        "DsRed$^{+}$ fraction",
        r"$\it{In\ situ}$$^{+}$ fraction",
        "Coexpression fraction",
    ]
    row_keys   = ["f_dsred", "f_insitu", "f_both"]
    row_cmaps  = [cmap_dsred, cmap_insitu, cmap_both]

    fig, axes = plt.subplots(3, n_fish, figsize=(4.5 * n_fish, 13),
                              gridspec_kw={"hspace": 0.25, "wspace": 0.12})
    if n_fish == 1:
        axes = axes.reshape(-1, 1)

    for col, fname in enumerate(fish_names):
        d = fish_densities[fname]
        for row in range(3):
            ax = axes[row, col]
            data = d[row_keys[row]]
            im = ax.imshow(data, cmap=row_cmaps[row], extent=d["extent"],
                           aspect="equal", vmin=0, vmax=1,
                           interpolation="bilinear")

            # Contour lines at 0.3 and 0.6
            clean = np.nan_to_num(data, nan=0)
            try:
                ax.contour(clean, levels=[0.3, 0.6], colors="white",
                           linewidths=0.8, alpha=0.6,
                           extent=d["extent"], origin="upper")
            except Exception:
                pass

            # Labels
            if row == 0:
                ax.set_title(fname, fontsize=16, fontweight="bold", pad=8)
            if col == 0:
                ax.set_ylabel(row_labels[row] + "\nY (µm)",
                              fontsize=13, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("X (µm)", fontsize=12)
            else:
                ax.set_xticklabels([])

            ax.tick_params(labelsize=10)

            # Colorbar only on rightmost column
            if col == n_fish - 1:
                add_colorbar(ax, im, label="Fraction")

    # Panel letters
    for i, (row, label) in enumerate(zip(range(3), "ABC")):
        axes[row, 0].text(-0.22, 1.02, label, transform=axes[row, 0].transAxes,
                          fontsize=20, fontweight="bold", va="top")

    out_path = os.path.join(out_dir, "MAIN_fraction_maps_all_fish.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ MAIN FIGURE: {out_path}")

    # ── RGB overlay figure (1 row × 4 fish) ─────────────────────
    fig2, axes2 = plt.subplots(1, n_fish, figsize=(4.5 * n_fish, 8))
    if n_fish == 1:
        axes2 = [axes2]

    for col, fname in enumerate(fish_names):
        ax = axes2[col]
        d = fish_densities[fname]
        eps = 1e-6

        r = d["d_dsred"]  / (d["d_dsred"].max()  + eps)
        g = d["d_insitu"] / (d["d_insitu"].max() + eps)

        rgb = np.zeros((*d["shape"], 3))
        rgb[..., 0] = r
        rgb[..., 1] = g
        rgb = np.clip(rgb ** 0.55, 0, 1)  # gamma

        # Black background where no cells
        no_cells = d["d_all"] < d["d_all"].max() * 0.005
        rgb[no_cells] = 0

        ax.imshow(rgb, extent=d["extent"], aspect="equal", interpolation="bilinear")
        ax.set_title(fname, fontsize=16, fontweight="bold")
        ax.set_xlabel("X (µm)", fontsize=12)
        if col == 0:
            ax.set_ylabel("Y (µm)", fontsize=13)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=10)

    # Legend
    axes2[-1].legend(handles=[
        Patch(facecolor='red',    label='DsRed$^{+}$ dominant'),
        Patch(facecolor='green',  label='InSitu$^{+}$ dominant'),
        Patch(facecolor='yellow', label='Colocalized'),
    ], loc='lower right', fontsize=11, framealpha=0.9)

    fig2.suptitle("DsRed (red) vs In situ (green) density overlay",
                  fontsize=18, fontweight="bold", y=1.01)
    out_path2 = os.path.join(out_dir, "MAIN_RGB_overlay_all_fish.png")
    fig2.savefig(out_path2, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig2)
    print(f"✓ RGB OVERLAY: {out_path2}")

    return fish_densities


# =============================================================================
# FIGURE: PER-FISH (appendix)
# =============================================================================

def plot_individual(fish_name, df, out_dir):
    """4-panel figure for one fish (appendix)."""
    if df.empty:
        return

    ymax = int(df["y"].max()) + 50
    xmax = int(df["x"].max()) + 50
    shape = (ymax, xmax)
    extent = [0, xmax * PIXEL_SIZE_XY, ymax * PIXEL_SIZE_XY, 0]

    d_all    = make_density(df["y"].values, df["x"].values, shape)
    d_dsred  = make_density(df.loc[df["DsRed_positive"], "y"].values,
                            df.loc[df["DsRed_positive"], "x"].values, shape)
    d_insitu = make_density(df.loc[df["InSitu_positive"], "y"].values,
                            df.loc[df["InSitu_positive"], "x"].values, shape)
    d_both   = make_density(
        df.loc[df["DsRed_positive"] & df["InSitu_positive"], "y"].values,
        df.loc[df["DsRed_positive"] & df["InSitu_positive"], "x"].values, shape)

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))

    panels = [
        (d_all,    "All nuclei",              "inferno"),
        (d_dsred,  "DsRed$^{+}$",            "Reds"),
        (d_insitu, "InSitu$^{+}$",           "Greens"),
        (d_both,   "DsRed$^{+}$ ∩ InSitu$^{+}$", "Purples"),
    ]

    for ax, (data, title, cmap), letter in zip(axes, panels, "ABCD"):
        im = ax.imshow(data, cmap=cmap, extent=extent,
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("X (µm)", fontsize=12)
        ax.tick_params(labelsize=10)
        add_colorbar(ax, im, label="Density")
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top")

    axes[0].set_ylabel("Y (µm)", fontsize=13)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    n = len(df)
    nd = df["DsRed_positive"].sum()
    ni = df["InSitu_positive"].sum()
    fig.suptitle(f"{fish_name} — Spatial density (n = {n} cells, "
                 f"DsRed$^+$ = {nd}, InSitu$^+$ = {ni})",
                 fontsize=17, fontweight="bold", y=1.02)

    safe_name = fish_name.replace(" ", "_").lower()
    out_path = os.path.join(out_dir, f"APPENDIX_density_{safe_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Appendix: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  DENSITY MAPS — ALL FISH")
    print("=" * 70)

    all_fish_data = {}

    for fish_name, info in FISH.items():
        df = process_fish(fish_name, info["mask_dir"], info["csv_dir"])
        if not df.empty:
            all_fish_data[fish_name] = df

    if not all_fish_data:
        print("\n❌ No fish processed. Check paths.")
        sys.exit(1)

    # Save combined CSV
    df_all = pd.concat(
        [d.assign(fish=fn) for fn, d in all_fish_data.items()],
        ignore_index=True
    )
    df_all.to_csv(os.path.join(OUT_DIR, "all_fish_cells_coordinates.csv"),
                  index=False, float_format="%.2f")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for fn, df in sorted(all_fish_data.items()):
        n = len(df)
        nd = df["DsRed_positive"].sum()
        ni = df["InSitu_positive"].sum()
        nb = (df["DsRed_positive"] & df["InSitu_positive"]).sum()
        print(f"  {fn:10s}: {n:6d} cells | DsRed+ {nd:5d} | "
              f"InSitu+ {ni:5d} | Both {nb:5d}")
    print(f"  {'TOTAL':10s}: {len(df_all):6d}")

    # ── Generate figures ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  GENERATING FIGURES")
    print(f"{'=' * 70}")

    # Main body: combined fraction maps
    plot_combined(all_fish_data, OUT_DIR)

    # Appendix: per-fish density
    for fish_name, df in sorted(all_fish_data.items()):
        plot_individual(fish_name, df, OUT_DIR)

    print(f"\n{'=' * 70}")
    print(f"  DONE — all outputs in {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()