# Transgenic Reporter Validation Pipeline

A 3D image analysis pipeline that measures, cell by cell, how faithfully a transgenic
reporter line labels the population it is meant to label. It compares the reporter
against an independent molecular ground truth (RNA *in situ* hybridisation) in
whole-mount larval zebrafish brains.

It was developed to validate the presumptively GABAergic line
**Tg(gad1b:DsRed); Tg(elavl3:H2B-GCaMP6s)** against HCR v3.0 *in situ* hybridisation
for *gad1b* and *gad2*. The steps are not specific to this line: any reporter / probe
pair can be used, as long as the same volume also has a nuclear channel to segment.

> **Status.** This repository contains the analysis code as it was used for the study.
> The scripts are not yet a packaged tool: most of them have input paths set at the
> top of the file and must be edited before running (see [Known gaps](#known-gaps)).

---

## Main results

Numbers are from the thesis (see [Citation](#citation)).

**Segmentation** (finetuned Cellpose `cyto3`, held-out benchmark volume of
430 × 562 × 140 voxels with 1,144 manually annotated nuclei):

| Dice | IoU | AP@[0.5:0.95] |
|---|---|---|
| 0.879 | 0.784 | 0.380 |

About 50,000 nuclei per brain hemisphere were segmented.

**Reporter fidelity** (DsRed vs. *gad1b* ∪ *gad2* in situ, four larvae, mean):

| Specificity | Precision | Sensitivity | F1 | Odds ratio |
|---|---|---|---|---|
| 0.925 | 0.808 | 0.558 | 0.656 | 16.86 |

Fisher's exact test gave p < 10⁻³⁰⁰ in every fish. Sensitivity is moderate partly by
design: the in situ reference is the union of *gad1b* and *gad2*, while the reporter is
driven by *gad1b* only.

---

## Who this is for

Anyone who needs to answer: **"does my transgenic reporter actually label the cell
population I think it does?"** You need a transgenic line, an independent molecular
marker for the population of interest, and a pan-neuronal (or pan-cellular) nuclear
label imaged in the same volume.

### Requirements for your data

| Input | Requirement |
|---|---|
| Nuclear channel | Nuclear label with enough contrast for 3D instance segmentation (we used H2B-GCaMP6s) |
| Reporter channel | The transgenic reporter, imaged in the same volume (we used DsRed) |
| Reference channel | An independent marker for the population (we used HCR v3.0 in situ, Alexa Fluor 647) |
| Imaging | Multi-channel 3D stack, channels acquired in register on the same voxel grid |

The reporter and the reference are quantified in different compartments depending on
where the signal sits:

- **Signal that fills the nucleus** (DsRed in our case) is measured inside the nuclear mask.
- **Punctate perinuclear / cytoplasmic signal** (in situ mRNA) is measured in a
  Voronoi-constrained shell around each nucleus, so neighbouring cells never share voxels.

### Imaging used in the study

Zeiss LSM 980 with Airyscan 2 (Multiplex 4Y), 40×/1.2 NA water objective, Standard
Airyscan Processing in ZEN. Channels were acquired sequentially frame by frame
(488 / 561 / 633 nm). The acquired voxel size was 0.05 × 0.05 × 0.25 µm; after
downscaling, all analysis runs at **0.1 × 0.1 × 0.25 µm**. Ten tiles per brain were
stitched into one stack. Only the left hemisphere was segmented.

---

## Pipeline

| # | Stage | Thesis section | Code | Notes |
|---|---|---|---|---|
| 1 | Preprocessing of the nuclear channel | 3.5, App. A.5 | [`01_preprocessing/`](01_preprocessing/README.md) | Full description, parameters and run order are in the [preprocessing README](01_preprocessing/README.md). DsRed and in situ channels are only resampled; their raw intensities are kept. |
| 2a | Training data and augmentation | 3.6 | `02_segmentation/partiremdiferentesperspetivas.ipynb` (orthogonal views for Cellpose), `02_segmentation/dataaumentation/dataumentations3Dlatest.py` (3D augmentation) | Cellpose was finetuned in the Cellpose GUI (human-in-the-loop), not by a script. The 2D augmentation used for Cellpose is not in the repo. |
| 2b | Segmentation benchmark | 3.6, 4.1 | `train_stardist3D3dez.py`, `predict_stardist3d3dez.py`, `trainswincell2dezcomval.py`, `inferswincell3dez.py`, `comparisonmodels.py` (all in `02_segmentation/`) | Compares Cellpose, StarDist3D and SwinCell against the manual annotation. |
| 2c | Full-brain segmentation | 3.7 | `02_segmentation/trained_models/hindbrain_cyto3_finetune` (model) | Run with Cellpose 3 on overlapping z-slabs. No script for the slab run or for merging slabs is in the repo. |
| 2d | Postprocessing | 3.7.1 | `02_segmentation/postprocessing/postprocessingfill.py` | Volume filter, slice-wise closing, z-continuity. Needs a CUDA GPU (CuPy). |
| 3 | Coexpression classification | 3.8.2 | `03_classification/overlap_full.py` | Per cell: bright-voxel fraction for DsRed (inside nucleus) and in situ (Voronoi shell), then Otsu. One CSV per z-block. |
| 4a | Parameter optimisation | 3.8.3, 4.3 | `03_classification/classificationdsred.py` (DsRed percentile sweep), `03_classification/3jan_shellchoose.py` (in situ shell × percentile grid search), `03_classification/dsredpermutation.py` (ROC and permutation test) | Needs the manual point annotations (`Results coexpression/Results_*.csv`). |
| 4b | Fidelity metrics and maps | 3.8.4, 4.4 | `03_classification/densitymap.py` | Density maps. The script that computes Table 4.3 (sensitivity, specificity, odds ratio, Fisher test) is not clearly identified yet. |
| 5 | Two-photon functional imaging (optional) | 3.9, 4.5 | `05_functional_2p/` | Only needed to link reporter identity to calcium imaging. See below. |

### Stage 5: two-photon functional imaging

This is a proof of concept on one larva. The steps and the files that correspond to them:

- **Motion correction** (Suite2p): `motioncorrectionfull.py`,
  `Motioncorrection suite2p manel updated.py`, `Run motion correction1812 updated.py`
- **3D ROI stitching** of Suite2p 2D ROIs across planes: `02_segmentation/3. 3Dstiching.ipynb`
  (stored in the segmentation folder but belongs to this stage)
- **Registration** of the anatomical GCaMP6s / DsRed volume into functional space:
  `registofinal4deform.py` (ANTs SyN, the method selected in the thesis) and
  `registrationmetrics.py`. `registrationfinal2.py`, `registofinal3.py` and
  `regsitration0202.py` are earlier versions.
- **DsRed classification of functional ROIs** (P80 + Otsu): `dsredclassification4fev.py`,
  `colocalization.py`, `colocalizationv2.py`
- **Activity and spatial analysis**: `gabavsnongaba.py`, `gabapeakanalyis.py`,
  `gabatemporalanalysis.py`, `spatialanalysis2.py`, `spatialanalysisgaba.py`,
  `tracesbyplane.py`, `tracessomeplane.py`

---

## Parameters used in the study

These were tuned for our line, probes and microscope. For a new reporter / marker
pair, treat them as starting points and re-tune them against a small manually
annotated set, as described in thesis Section 3.8.3.

| Step | Parameter | Value (thesis) |
|---|---|---|
| Preprocessing | see [`01_preprocessing/README.md`](01_preprocessing/README.md) | rolling ball 60 px, 3D median r = 1, CLAHE 127 / 256 / 3 |
| Cellpose inference | model / diameter / cellprob / flow3D smooth | finetuned `cyto3` / ≈45 px / −4 / 3, with `denoise` restoration |
| Postprocessing | kept object volume | 5,000–70,000 voxels |
| Postprocessing | slice-wise closing | disk, radius 3 px; then 1 px z-dilation ∩ original mask |
| DsRed | feature / percentile / threshold | bright-voxel fraction inside the nucleus / **P75** of the whole slice / Otsu per block |
| In situ | shell / percentile / threshold | Voronoi shell **[−5, +3] px** from the nuclear edge / **P85** / global Otsu |
| 2P DsRed | percentile | P80 (not optimised) |

**Validation set:** three annotated slices of one fish (z = 110, 300, 450; 5,502 nuclei
in total), stored as point annotations in `Results coexpression/`.

At these settings the classifier matched the manual labels with precision 0.857 and
recall 0.806 for DsRed. For the in situ features inside the selected shell, the
bright-voxel fraction had the highest AUC (0.946).

---

## Installation

The code was run on Windows with Python and an NVIDIA GPU (the classification was
designed for an RTX 3060 with 11 GB VRAM). Fiji is needed for the preprocessing macros.

```bash
git clone https://github.com/inespemarques/transgenicline_validation
cd transgenicline_validation
pip install -r requirements.txt
```

`requirements.txt` does not pin versions yet, and it does not list every package the
scripts import (for example `tifffile`, `matplotlib`, `seaborn`, `scikit-learn`,
`tqdm`, `tensorflow` / `csbdeep` for StarDist, `torch` / `monai` for SwinCell, and
`antspyx` for stage 5). In the study, Cellpose, StarDist and ANTs were used in
separate environments.

## Running on your own data

1. Preprocess the nuclear channel as described in
   [`01_preprocessing/README.md`](01_preprocessing/README.md).
2. Segment nuclei with Cellpose 3 in 3D, using `02_segmentation/trained_models/hindbrain_cyto3_finetune`
   or a model finetuned on your own nuclei (Cellpose GUI, orthogonal XY / XZ / YZ crops).
3. Clean the masks with `02_segmentation/postprocessing/postprocessingfill.py`
   (set `ROOT` and `FISH_FOLDERS` at the top of the file).
4. Classify each cell with `03_classification/overlap_full.py`
   (set `BASE`, `INSITU_PATH`, `DSRED_PATH` and the parameters at the top of the file;
   see the note on percentiles under [Known gaps](#known-gaps)).
5. Annotate a few slices by hand in Fiji (Multipoint tool, one point per positive
   nucleus) and use the scripts in stage 4a to re-tune the percentile and shell for your
   markers.

---

## Repository structure

```
transgenicline_validation/
├── 01_preprocessing/          Fiji macros, Z-correction script, README, DISCREPANCIES.md
├── 02_segmentation/
│   ├── dataaumentation/       3D augmentation for StarDist3D / SwinCell
│   ├── postprocessing/        mask cleaning
│   ├── trained_models/        finetuned Cellpose models
│   ├── benchmarking/          benchmark volume, manual annotation, model outputs
│   └── *.py, *.ipynb          training, inference and model comparison
├── 03_classification/         coexpression classifier, parameter optimisation, validation
├── 04_results/                density maps
├── 05_functional_2p/          two-photon motion correction, registration, analysis
├── Figures/                   preprocessing and benchmarking example figures
├── Results coexpression/      manual annotations and per-block classification output
├── ficheiros codigo pc pessoal/   drafts, not part of the pipeline
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Known gaps

These are open before publication. They are listed so users know what to expect.

- **Classifier defaults differ from the thesis.** `overlap_full.py` is set to DsRed P85,
  in situ P90 and a 5 px Voronoi ring. The thesis reports DsRed P75 with per-block Otsu
  and in situ P85 with a [−5, +3] px shell. Section 3.9.2 of the thesis also refers to
  "the P85 used in the confocal pipeline". Which values produced Table 4.3 still needs
  to be confirmed.
- **Postprocessing volume filter.** `postprocessingfill.py` keeps objects from 3,000
  voxels; the thesis says 5,000–70,000.
- **Missing code:** running Cellpose on z-slabs and merging labels across slab
  overlaps (Section 3.7); the 2D augmentation used to finetune Cellpose (Table 3.1);
  the script that computes the fidelity table (Table 4.3).
- **Preprocessing:** see [`01_preprocessing/DISCREPANCIES.md`](01_preprocessing/DISCREPANCIES.md).
- **Hardcoded paths:** most scripts point to local Windows folders and have no
  command-line arguments.
- **Several versions of the same step** are kept side by side (for example the
  `insitu*.py`, `classinsituwithroc.py` and `*dilationvsvoronoi*.py` scripts in
  `03_classification/` are exploratory versions of the shell optimisation that
  `3jan_shellchoose.py` replaced).
- **No small demo dataset** with expected output yet.

---

## Data availability

Raw confocal stacks, segmentation masks and full-resolution TIFFs will be deposited at
the BioImage Archive (accession to be added). This repository is meant to hold code,
processed CSV outputs and example figures.

## Limitations

- Segmentation quality depends on the SNR of the nuclear channel; dim or touching
  nuclei may require finetuning the model on your data.
- The segmentation model was trained on one annotated volume from one animal, with
  augmentation standing in for variability between animals.
- The classifier assumes the per-cell bright-voxel fraction is roughly bimodal so that
  Otsu can separate positive and negative cells. Sparse or very low-prevalence markers
  may need a manually checked threshold.
- False positives cluster near bright neuropil, where halo fluorescence inflates the
  bright-voxel fraction; false negatives are dim cells near the detection limit.

---

## Citation

This pipeline was developed and validated in:

> Marques, I. P. *Quantitative Characterization of Zebrafish Reporter Lines Targeting
> Different Neurotransmitter Populations.* MSc thesis in Biomedical Engineering,
> Instituto Superior Técnico, Universidade de Lisboa, 2026. Work carried out at
> Champalimaud Research, Lisbon.

A journal article is in preparation; this section will be updated with its DOI.
See also [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
