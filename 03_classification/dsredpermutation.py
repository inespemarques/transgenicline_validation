# dsred_perm_roc_article_EN.py
# DsRed validation (paper-style figures):
#  - Primary classifier: P85 (% bright pixels) + Otsu threshold
#  - Comparators (continuous scores): mean intensity, integrated density
#  - ROC curves: per-slice + global (pooled cells)
#  - Permutation test: label-shuffle within slice (preserve prevalence), pooled null (F1 only)
#  - Method performance plot (F1): bars show MEAN with MIN-MAX error bars (so they reach points)

import os
import re
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve, auc


# =========================
# PATHS
# =========================

BASE_IMG = r"D:\User data\InesMarques\202501024\fish7\cleaned_masks"
BASE_CSV = r"D:\User data\InesMarques\202501024\validation_overlap"

OUT = os.path.join(BASE_CSV, "validation_output_dsred_perm_roc4")
SUBFOLDERS = ["csv", "plots", "roc_curves", "perm_hists"]
for sub in SUBFOLDERS:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

# Cleaned mask stacks + validated z-slice index (1-based)
MASK_INFO = {
    110: {"mask": os.path.join(BASE_IMG, "c2_110a240clahe_segmented_cleaned.tif"), "z": 1},
    300: {"mask": os.path.join(BASE_IMG, "c2_220a350clahe_segmented_cleaned.tif"), "z": 80},
    450: {"mask": os.path.join(BASE_IMG, "c2_330a460clahe_segmented_cleaned.tif"), "z": 120},
}

# Full DsRed TIFF (global Z)
DS_PATH = os.path.join(BASE_IMG, "3channelsdownscaledleft541slices_apenasdsred.tif")
DS = tiff.imread(DS_PATH)


# =========================
# PARAMETERS
# =========================

P_DS = 85
N_PERM = 5000
SEED = 123

MAKE_PER_SLICE_ROC_PLOTS = True
MAKE_PER_SLICE_PERM_PLOTS = False  # set True if you also want per-slice permutation hists


# =========================
# STYLE (paper-like)
# =========================

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False


# =========================
# HELPERS
# =========================

def parse_block_z(mask_path: str):
    fname = os.path.basename(mask_path)
    m = re.search(r"_(\d+)a(\d+)", fname)
    if m is None:
        raise ValueError(f"Cannot parse Z range from: {fname}")
    return int(m.group(1)), int(m.group(2))


def load_csv_smart(folder: str, sl: int, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    sl_str = str(sl)

    cand = []
    for f in os.listdir(folder):
        fl = f.lower()
        if not fl.endswith(".csv"):
            continue
        if mode not in fl:
            continue
        if sl_str in fl:
            cand.append(f)

    if not cand:
        for f in os.listdir(folder):
            fl = f.lower()
            if not fl.endswith(".csv"):
                continue
            if mode not in fl:
                continue
            nums = re.findall(r"\d+", fl)
            if sl_str in nums:
                cand.append(f)

    if not cand:
        raise FileNotFoundError(f"No CSV found for slice={sl}, mode={mode} in {folder}")

    cand = sorted(cand)[0]
    print(f"CSV ({mode}, slice {sl}) -> {cand}")
    return pd.read_csv(os.path.join(folder, cand))


def safe_otsu(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.nan
    vals = x[m]
    if np.all(vals == vals[0]):
        return np.nan
    return float(threshold_otsu(vals))


def pctbright_3d_dsred(img3d: np.ndarray, lab3d: np.ndarray, perc: int):
    ids = np.unique(lab3d)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return {}

    mx = int(ids.max()) + 1
    tot = np.zeros(mx, dtype=np.float64)
    bri = np.zeros(mx, dtype=np.float64)

    for z in range(img3d.shape[0]):
        img = img3d[z]
        L = lab3d[z]
        thr = np.percentile(img, perc)
        b = (img > thr).astype(np.float32)

        tot += np.bincount(L.ravel(), minlength=mx)
        bri += np.bincount(L.ravel(), weights=b.ravel(), minlength=mx)

    pct = (bri / np.maximum(tot, 1.0)) * 100.0
    pct[tot == 0] = np.nan
    return {int(i): float(pct[int(i)]) for i in ids}


def compute_mean_and_intden_3d(img3d: np.ndarray, lab3d: np.ndarray):
    ids = np.unique(lab3d)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return None, None

    mx = int(ids.max()) + 1
    counts = np.zeros(mx, dtype=np.float64)
    sums = np.zeros(mx, dtype=np.float64)

    for z in range(img3d.shape[0]):
        img = img3d[z].astype(np.float64, copy=False)
        L = lab3d[z]
        counts += np.bincount(L.ravel(), minlength=mx)
        sums += np.bincount(L.ravel(), weights=img.ravel(), minlength=mx)

    mean = sums / np.maximum(counts, 1.0)
    mean[counts == 0] = np.nan
    intden = sums
    intden[counts == 0] = np.nan
    return mean, intden


def binarize_by_otsu(df_scores: pd.DataFrame, feature: str):
    thr = safe_otsu(df_scores[feature].to_numpy(dtype=float))
    if not np.isfinite(thr):
        return thr, set()
    auto_set = set(df_scores.loc[df_scores[feature] > thr, "label"].astype(int).tolist())
    return thr, auto_set


def confusion_counts(auto_set, manual_set, ids_all):
    auto_set = set(auto_set)
    manual_set = set(manual_set)
    ids_all = set(ids_all)

    TP = len(auto_set & manual_set)
    FP = len(auto_set - manual_set)
    FN = len(manual_set - auto_set)
    TN = len(ids_all - auto_set - manual_set)
    return TP, FP, FN, TN


def f1_from_counts(TP, FP, FN):
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    return float(2 * prec * rec / max(prec + rec, 1e-12))


def operating_point_fpr_tpr(auto_set, manual_set, ids_all):
    TP, FP, FN, TN = confusion_counts(auto_set, manual_set, ids_all)
    tpr = TP / max(TP + FN, 1)
    fpr = FP / max(FP + TN, 1)
    return float(fpr), float(tpr)


def permutation_test_f1(ids_all, manual_set, auto_set, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    ids_all = np.array(sorted(list(set(ids_all))), dtype=int)
    k = len(manual_set)

    TP, FP, FN, TN = confusion_counts(auto_set, manual_set, ids_all)
    f1_obs = f1_from_counts(TP, FP, FN)

    f1_null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm_pos = set(rng.choice(ids_all, size=k, replace=False))
        TPp, FPp, FNp, TNp = confusion_counts(auto_set, perm_pos, ids_all)
        f1_null[i] = f1_from_counts(TPp, FPp, FNp)

    p_f1 = (1 + np.sum(f1_null >= f1_obs)) / (n_perm + 1)
    stats = {
        "f1_obs": float(f1_obs),
        "p_f1": float(p_f1),
        "f1_null_mean": float(np.mean(f1_null)),
        "f1_null_sd": float(np.std(f1_null)),
    }
    return stats, f1_null


def plot_perm_hist_f1(null_vals, obs, title, outpath):
    plt.figure(figsize=(5.0, 3.6))
    plt.hist(null_vals, bins=40, density=True)
    plt.axvline(obs, linewidth=2, color="tab:red")
    plt.title(title)
    plt.xlabel("F1-score")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_roc_compare(df_scores: pd.DataFrame, title: str, outpath: str,
                     manual_set=None, ids_all=None, auto_set=None):
    plt.figure(figsize=(4.2, 4.2))

    feat_list = [
        ("pct85", "P85 (% bright)", "tab:blue"),
        ("mean3d", "Mean intensity", "tab:orange"),
        ("intden3d", "Integrated density", "tab:green"),
    ]

    for feat, label, color in feat_list:
        x = df_scores[feat].to_numpy(dtype=float)
        y = df_scores["y"].to_numpy(dtype=int)
        m = np.isfinite(x)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], x[m])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, color=color, label=f"{label} (AUC = {auc_val:.3f})")

    if (manual_set is not None) and (ids_all is not None) and (auto_set is not None):
        fpr_pt, tpr_pt = operating_point_fpr_tpr(auto_set, manual_set, ids_all)
        plt.scatter([fpr_pt], [tpr_pt], s=70, color="tab:blue", edgecolor="black",
                    linewidth=0.4, label="P85 + Otsu (operating point)")

    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5, color="grey")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def extract_dsred_scores_for_slice(sl: int, info: dict, perc: int = 85):
    mask_path = info["mask"]
    mask_stack = tiff.imread(mask_path)
    zloc = info["z"] - 1  # 1-based to 0-based

    lab2d = mask_stack[zloc]
    ids2d = np.unique(lab2d)
    ids2d = ids2d[ids2d != 0]
    ids_all = set(ids2d.tolist())

    man_ds = load_csv_smart(BASE_CSV, sl, "dsred")
    xs = (man_ds["XM"] - 1).astype(int).clip(0, lab2d.shape[1] - 1)
    ys = (man_ds["YM"] - 1).astype(int).clip(0, lab2d.shape[0] - 1)
    manual_set = set(lab2d[ys, xs])
    manual_set.discard(0)

    z0, z1 = parse_block_z(mask_path)
    ds_block = DS[z0:z1 + 1]
    lab_block = mask_stack

    Z = min(ds_block.shape[0], lab_block.shape[0])
    ds_block = ds_block[:Z]
    lab_block = lab_block[:Z]

    pctdict = pctbright_3d_dsred(ds_block, lab_block, perc)
    mean_arr, intden_arr = compute_mean_and_intden_3d(ds_block, lab_block)

    rows = []
    for lab in ids2d:
        lab = int(lab)
        y = 1 if lab in manual_set else 0
        pct85 = float(pctdict.get(lab, np.nan))
        mean3d = float(mean_arr[lab]) if (mean_arr is not None and mean_arr.size > lab) else np.nan
        intden3d = float(intden_arr[lab]) if (intden_arr is not None and intden_arr.size > lab) else np.nan
        rows.append({
            "slice": sl,
            "label": lab,
            "y": y,
            "pct85": pct85,
            "mean3d": mean3d,
            "intden3d": intden3d
        })

    return pd.DataFrame(rows), manual_set, ids_all


# =========================
# ARTICLE-STYLE FIGURES
# =========================

def plot_roc_global_article(df_scores_all: pd.DataFrame, outpath: str):
    plt.figure(figsize=(4.2, 4.2))

    feat_list = [
        ("pct85", "P85 (% bright)", "tab:blue"),
        ("mean3d", "Mean intensity", "tab:orange"),
        ("intden3d", "Integrated density", "tab:green"),
    ]

    for feat, label, color in feat_list:
        x = df_scores_all[feat].to_numpy(dtype=float)
        y = df_scores_all["y"].to_numpy(dtype=int)
        m = np.isfinite(x)
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], x[m])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, color=color, label=f"{label} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.5, color="grey")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Global ROC (DsRed)")
    plt.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_f1_summary_article_minmax(df_methods: pd.DataFrame, outpath: str):
    """
    Bars show MEAN. Error bars show MIN-MAX across slices (so they reach points).
    Points show per-slice F1 values, aligned to the bar center.
    """
    methods_order = ["p85_otsu", "mean_otsu", "intden_otsu"]
    labels = {
        "p85_otsu": "P85 + Otsu",
        "mean_otsu": "Mean + Otsu",
        "intden_otsu": "IntDen + Otsu",
    }
    colors = {
        "p85_otsu": "tab:blue",
        "mean_otsu": "tab:orange",
        "intden_otsu": "tab:green",
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    x = np.arange(len(methods_order))

    means, mins, maxs = [], [], []
    per_method_vals = {}

    for mname in methods_order:
        vals = df_methods.loc[df_methods["method"] == mname, "f1"].to_numpy(dtype=float)
        per_method_vals[mname] = vals
        means.append(np.nanmean(vals))
        mins.append(np.nanmin(vals))
        maxs.append(np.nanmax(vals))

    means = np.array(means, dtype=float)
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)

    # asymmetric yerr: [mean - min, max - mean]
    yerr = np.vstack([means - mins, maxs - means])

    ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=5,
        color=[colors[m] for m in methods_order],
        edgecolor="black",
        linewidth=0.8,
    )

    for i, mname in enumerate(methods_order):
        ys = per_method_vals[mname]
        xs = np.full(len(ys), x[i], dtype=float)  # aligned
        ax.scatter(xs, ys, s=28, color=colors[mname], edgecolor="black",
                   linewidth=0.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods_order], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Method performance (F1)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_perm_f1_pooled_article(df_perm: pd.DataFrame, f1_null_all, outpath_f1: str, n_perm_per_slice: int):
    f1_null = np.concatenate(f1_null_all) if len(f1_null_all) else np.array([])
    f1_obs_mean = float(np.mean(df_perm["f1_obs"].to_numpy(dtype=float)))
    n_slices = int(df_perm.shape[0])

    if f1_null.size:
        p_emp = (1 + np.sum(f1_null >= f1_obs_mean)) / (len(f1_null) + 1)
    else:
        p_emp = np.nan

    # Extra top space for header text
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=False)
    fig.subplots_adjust(top=0.78)  # <- reserves space above axes

    # Plot
    if f1_null.size:
        ax.hist(f1_null, bins=45, density=True, edgecolor="none")

    ax.axvline(f1_obs_mean, linewidth=2.2, color="tab:red")

    ax.set_xlabel("F1-score")
    ax.set_ylabel("Density")
    ax.set_title("Permutation test (F1)", pad=6)

    # Top “legend” text (outside axes, like paper)
    fig.text(
        0.125, 0.94,
        f"Pooled null across slices (N = {n_slices}; {n_perm_per_slice} permutations per slice)",
        ha="left", va="top", fontsize=9
    )
    fig.text(
        0.125, 0.90,
        f"Observed mean F1 = {f1_obs_mean:.2f}  |  one-sided permutation p = {p_emp:.2g}",
        ha="left", va="top", fontsize=9, color="tab:red"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(outpath_f1, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


# =========================
# RUN
# =========================

print("Running DsRed analysis: P85 + Otsu, permutation test (F1), and ROC comparisons")
print("Output folder:", OUT)
print("Permutations per slice:", N_PERM, "minimum resolvable p approx", 1.0 / (N_PERM + 1))

all_scores = []
rows_perm = []
rows_methods = []
f1_null_all = []

for sl, info in MASK_INFO.items():
    print("\nSlice", sl)

    df_sc, manual_set, ids_all = extract_dsred_scores_for_slice(sl, info, perc=P_DS)
    all_scores.append(df_sc)

    # Otsu binarization for each score (method comparison)
    thr_pct, auto_pct = binarize_by_otsu(df_sc, "pct85")
    thr_mean, auto_mean = binarize_by_otsu(df_sc, "mean3d")
    thr_int, auto_int = binarize_by_otsu(df_sc, "intden3d")

    # Save per-slice method F1
    for method_name, thr, auto_set in [
        ("p85_otsu", thr_pct, auto_pct),
        ("mean_otsu", thr_mean, auto_mean),
        ("intden_otsu", thr_int, auto_int),
    ]:
        TP, FP, FN, TN = confusion_counts(auto_set, manual_set, ids_all)
        f1 = f1_from_counts(TP, FP, FN)
        rows_methods.append({
            "slice": sl,
            "method": method_name,
            "otsu_thr": thr,
            "n_cells": len(ids_all),
            "n_pos_manual": len(manual_set),
            "n_pos_auto": len(auto_set),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "f1": float(f1),
        })

    # Permutation test only for primary method (P85 + Otsu)
    stats, f1_null = permutation_test_f1(
        ids_all=ids_all,
        manual_set=manual_set,
        auto_set=auto_pct,
        n_perm=N_PERM,
        seed=SEED + int(sl),
    )
    rows_perm.append({
        "slice": sl,
        "percentile": P_DS,
        "otsu_thr_p85": thr_pct,
        "n_cells": len(ids_all),
        "n_pos_manual": len(manual_set),
        "n_pos_auto": len(auto_pct),
        **stats
    })
    f1_null_all.append(f1_null)

    if MAKE_PER_SLICE_PERM_PLOTS:
        plot_perm_hist_f1(
            f1_null, stats["f1_obs"],
            title=f"Permutation test (F1) - slice {sl} - N={N_PERM}",
            outpath=os.path.join(OUT, "perm_hists", f"perm_f1_slice{sl}.png"),
        )

    if MAKE_PER_SLICE_ROC_PLOTS:
        plot_roc_compare(
            df_sc,
            title=f"ROC (DsRed) - slice {sl}",
            outpath=os.path.join(OUT, "roc_curves", f"roc_compare_slice{sl}.png"),
            manual_set=manual_set,
            ids_all=ids_all,
            auto_set=auto_pct
        )

# Save CSVs
df_scores_all = pd.concat(all_scores, ignore_index=True)
df_perm = pd.DataFrame(rows_perm)
df_methods = pd.DataFrame(rows_methods)

df_scores_all.to_csv(os.path.join(OUT, "csv", f"dsred_P{P_DS}_scores_long.csv"), index=False)
df_perm.to_csv(os.path.join(OUT, "csv", f"dsred_P{P_DS}_permutation_test_F1.csv"), index=False)
df_methods.to_csv(os.path.join(OUT, "csv", f"dsred_P{P_DS}_methods_F1.csv"), index=False)

print("\nSaved CSVs to:", os.path.join(OUT, "csv"))

# Article-style summary figures
plot_roc_global_article(
    df_scores_all,
    outpath=os.path.join(OUT, "plots", "article_global_ROC_dsred.png"),
)

plot_f1_summary_article_minmax(
    df_methods,
    outpath=os.path.join(OUT, "plots", "article_method_F1_summary_minmax.png"),
)

plot_perm_f1_pooled_article(
    df_perm,
    f1_null_all=f1_null_all,
    outpath_f1=os.path.join(OUT, "plots", "article_perm_f1_pooled.png"),
    n_perm_per_slice=N_PERM,
)

print("\nDone. Figures saved to:", os.path.join(OUT, "plots"))
