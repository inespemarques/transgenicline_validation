"""Step 2: Z-attenuation (bleach) correction by slice-wise histogram matching.

Each Z slice is histogram-matched to the middle slice of the stack, as
described in the thesis (Section 3.5). Input is the downscaled nuclear
channel from fiji_macros/step1_downscale.ijm.

Usage:
    python zattenuation_correction.py --input downscaled.tif --output zcorrected.tif
"""
import argparse

import numpy as np
import tifffile as tiff
from skimage.exposure import match_histograms


def correct_z_attenuation(stack):
    """Match the histogram of every Z slice to the middle slice."""
    reference = stack[stack.shape[0] // 2]
    max_value = np.iinfo(stack.dtype).max
    corrected = np.empty_like(stack)
    for z in range(stack.shape[0]):
        matched = match_histograms(stack[z], reference)
        corrected[z] = np.clip(matched, 0, max_value).astype(stack.dtype)
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="3D stack (Z, Y, X), integer dtype")
    parser.add_argument("--output", required=True, help="output TIFF path")
    args = parser.parse_args()

    img = tiff.imread(args.input)
    if img.ndim != 3:
        raise ValueError("Expected a 3D stack (Z, Y, X).")
    tiff.imwrite(args.output, correct_z_attenuation(img), imagej=True)


if __name__ == "__main__":
    main()
