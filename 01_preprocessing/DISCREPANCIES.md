# Open points: code vs. thesis

The files in this folder follow the order and parameters written in the thesis
(Section 3.5). The points below are where the existing code disagrees with the thesis,
or where the thesis does not give a value. Resolve these before publication.

## 1. Z-correction method

- **Thesis:** slice-wise histogram matching to the middle slice.
- **`bleachcorrection.ipynb`:** ratio correction. Each slice is multiplied by
  `median(profile) / profile[z]`, where `profile` is the per-slice mean smoothed over
  5 slices. Appendix A.5 reports this method as the worse of the two.
- **Now in the repo:** `zattenuation_correction.py` implements histogram matching,
  written from the thesis description. It is not the original code used for the paper.
- **To do:** find the original histogram-matching code, or confirm which method
  produced the published results. Decide whether to keep `bleachcorrection.ipynb`.

## 2. Step order

- **Thesis:** downscaling → Z correction → background subtraction → median → CLAHE → normalisation.
- **`bleachcorrection.ipynb`:** reads `c2downscaledbacksubtraction.tif`, so background
  subtraction was already done before the Z correction.
- **Now in the repo:** thesis order.
- **To do:** confirm the order used for the published data.

## 3. Fiji macros are reconstructed

The original Fiji macro is not in the repository. `fiji_macros/*.ijm` were written from
the parameters in the thesis and have not been run against the original data.

- **To do:** replace with the original macro if it exists, or run these on one stack and
  compare with the published preprocessed output.

## 4. Values not stated in the thesis

| Item | Now in the repo | To do |
|---|---|---|
| Fiji / ImageJ version | not stated | add the version used |
| Normalisation saturation (%) | 0.35 (Fiji default) | confirm the value used |
| Z downscaling | none (Z unchanged) | confirm Z was not resampled |
| CLAHE "fast" mode | off (full precision) | confirm |
