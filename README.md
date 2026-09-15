# Transgenic Reporter Validation Pipeline

A 3D image analysis pipeline for quantitatively validating **any transgenic reporter line** against a
molecular ground truth (e.g. RNA *in situ* hybridisation), at single-cell resolution, in whole-mount
larval zebrafish brains.

**What this does:** given (1) a nuclear reference channel, (2) a reporter fluorescence channel, and
(3) an independent molecular marker (e.g. HCR *in situ*), the pipeline segments individual nuclei in
3D, quantifies both signals per cell, and reports how faithfully the reporter recapitulates the
molecular ground truth (precision, recall, specificity, odds ratio).

Originally developed and validated for Tg(gad1b:DsRed); Tg(elavl3:H2B-GCaMP6s) against gad1b/gad2
*in situ* signal, but the pipeline is marker-agnostic: the same steps apply to any reporter/probe
combination, provided a nuclear channel is available for segmentation. See [Citation](#citation) for
the full validation study.

---

## Who this is for

Anyone who needs to answer: **"does my transgenic reporter actually label the cell population I think
it does?"** Typical use case: you have a new or existing transgenic line, an independent molecular
marker for the population of interest, and want a quantitative, per-cell fidelity estimate instead of
visual impression.

---

## Requirements for your data

| Input | Requirement |
|---|---|
| Nuclear channel | Panneuronal (or pancellular) nuclear label with enough contrast for instance segmentation (e.g. H2B-GCaMP, H2B-RFP, DAPI) |
| Reporter channel | Your transgenic line's fluorescent reporter, imaged in the same volume |
| Reference channel | An independent marker for the population of interest (e.g. *in situ* hybridisation, antibody staining) |
| Imaging | Multi-channel 3D confocal stack, channels co-registered (same voxel grid) |

If your reporter/marker signal is **nuclear** (fills the nucleus), quantify it directly within the nuclear
mask (Section 3). If it is **perinuclear/cytoplasmic** (e.g. punctate mRNA), use the Voronoi shell
approach (Section 3) instead — this is set per channel, not fixed.

---

## Quickstart

```bash
# 1. Preprocess the nuclear channel
python 01_preprocessing/zattenuation_correction.py --input raw_stack.tif --output corrected.tif
# then apply FIJI macros in 01_preprocessing/fiji_macros/ (see that folder's README for parameters)

# 2. Segment nuclei in 3D
python 02_segmentation/inference/segment_bricks.py --input corrected.tif --model cyto3_finetuned --output masks.tif

# 3. Classify reporter / reference coexpression per cell
python 03_classification/densitymap.py \
    --masks masks.tif \
    --reporter_channel reporter.tif --reporter_mode nuclear \
    --reference_channel reference.tif --reference_mode perinuclear \
    --output results.csv
```

`results.csv` contains, per segmented cell: cell ID, 3D centroid, reporter⁺/⁻ label, reference⁺/⁻
label, and the underlying intensity features — ready for the precision/recall/specificity/odds-ratio
computation described in `04_results/`.

---

## Pipeline stages

| # | Stage | What you can/should adapt for your line |
|---|-------|---|
| 1 | Preprocessing | Rolling-ball radius, CLAHE settings — tune to your channel's noise/contrast, don't assume our values |
| 2 | 3D nuclear segmentation | Retrain/finetune Cellpose on your own nuclear marker if morphology differs; diameter and probability threshold are dataset-specific |
| 3 | Coexpression classification | **Percentile threshold and Voronoi shell radius are the two parameters most likely to need re-optimisation** for a new marker — see `03_classification/README.md` for the grid-search procedure used to pick ours |
| 4 | Validation | Requires a small set of manually annotated cells (we used ~1,000–2,000 per slice) as ground truth to optimise/report performance |
| 5 | Functional registration (optional) | Only relevant if you also plan to link reporter identity to calcium imaging data |

Default parameter values shipped in this repo (rolling-ball = 60 px, CLAHE blocksize = 127, percentile
= P85, shell radius = [−5, +3] px, etc.) were optimised for gad1b:DsRed / gad1b+gad2 *in situ* on our
imaging setup. **Treat them as starting points, not universal defaults** — Section 3.8.3-equivalent
optimisation (sweeping percentile and shell geometry against a small manually annotated set) is
recommended for any new reporter/marker pair.

---

## Repository structure

```
transgenicline_validation/
├── README.md
├── 01_preprocessing/
│   ├── zattenuation_correction.py
│   └── fiji_macros/
├── 02_segmentation/
│   ├── training/
│   └── inference/
├── 03_classification/
│   └── densitymap.py
├── 04_results/
│   ├── dsred/
│   └── insitu/
├── 05_functional_2p/
└── figures/
```

---

## Dependencies

```
cellpose
stardist
cupy
scikit-image
scipy
suite2p          # only needed for stage 5 (functional imaging)
SimpleITK / ants # only needed for stage 5
numpy
pandas
```

---

## Limitations

- Segmentation quality depends on nuclear channel SNR; very dim or overlapping nuclei may require
  retraining the segmentation model.
- The classification pipeline assumes reporter and reference signals occupy distinguishable
  subcellular compartments (nuclear vs. perinuclear); heavily overlapping distributions may need a
  different ROI strategy.
- Otsu thresholding assumes a roughly bimodal distribution of the per-cell feature; sparse or very
  low-prevalence markers may need manual threshold validation.

---

## Citation

This pipeline was developed and validated in:

> [Author]. *Quantitative Characterization of Zebrafish Reporter Lines Targeting Different
> Neurotransmitter Populations.* [Thesis/Journal, Year]. [DOI/link]

If you use or adapt this pipeline, please cite the above.

## License

[Add license, e.g. MIT]
