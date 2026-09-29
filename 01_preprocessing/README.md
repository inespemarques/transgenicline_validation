# 1. Preprocessing

Only the nuclear **H2B-GCaMP6s** channel goes through the full preprocessing, because it
is the input for segmentation. The DsRed and in situ (HCR) channels are only resampled
to the same voxel grid, and their raw intensities are kept for quantification.

All images are 16-bit. Steps 1 and 3–6 run in Fiji; step 2 runs in Python. The order
and parameters follow the thesis (Section 3.5, Appendix A.5). Where the older code in
this folder disagrees with the thesis, see [DISCREPANCIES.md](DISCREPANCIES.md).

| # | Step | Tool | File | Parameters |
|---|---|---|---|---|
| 0 | Split channels | Fiji | – | *Image ▸ Color ▸ Split Channels*, keep the GCaMP6s channel |
| 1 | XY downscaling | Fiji | `fiji_macros/step1_downscale.ijm` | 0.5× in XY, Z unchanged, bicubic interpolation (→ 0.1 µm/px) |
| 2 | Z-attenuation (bleach) correction | Python | `zattenuation_correction.py` | slice-wise histogram matching to the middle Z slice |
| 3 | Background subtraction | Fiji | `fiji_macros/step3_background_median_clahe_normalise.ijm` | rolling ball, radius 60 px, whole stack |
| 4 | 3D median filter | Fiji | 〃 | radius 1 voxel in x, y, z |
| 5 | CLAHE | Fiji | 〃 | slice-wise; block size 127, histogram bins 256, maximum slope 3 |
| 6 | Intensity normalisation | Fiji | 〃 | linear histogram stretch to a common range |

The output of step 6 is the input to `02_segmentation`.

## How to run

1. In Fiji, open the raw stack, split channels and keep the GCaMP6s channel.
   Run `fiji_macros/step1_downscale.ijm` (*Plugins ▸ Macros ▸ Run…*) and save the result
   as `downscaled.tif`.
2. `python 01_preprocessing/zattenuation_correction.py --input downscaled.tif --output zcorrected.tif`
3. Open `zcorrected.tif` in Fiji, run `fiji_macros/step3_background_median_clahe_normalise.ijm`
   and save the result as the segmentation input.

## How the parameters were chosen

Parameters were chosen one step at a time on a fixed mid-depth GCaMP6s subvolume,
segmented with the pretrained Cellpose `denoise_cyto3` model (cellprob = −4,
flow3d_smooth = 3), keeping the variant with the most nuclei and the most complete
boundaries at each step:

- **Downscaling:** bicubic beat bilinear and *Bin (Average)*.
- **Z correction:** histogram matching recovered more deep nuclei than ratio correction.
- **Rolling ball:** 40 px over-subtracted nuclear signal; radii below 60 px left halos that fragmented masks.
- **Median filter:** 3×3×3 median and Gaussian kernels merged touching nuclei; radius 1 kept boundaries.
- **CLAHE:** recovered dim nuclei missing without it.

These values were tuned for our imaging setup. For a new line or microscope, treat them
as starting points and re-tune them the same way.
