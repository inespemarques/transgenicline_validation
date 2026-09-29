// Steps 3-6: background subtraction, 3D median, CLAHE, normalisation (Fiji)
// Parameters from thesis Section 3.5. Reconstructed, not the original macro; see ../DISCREPANCIES.md.
// Input: open 16-bit stack after Z-attenuation correction (step 2, ../zattenuation_correction.py).

// 3. Rolling-ball background subtraction, radius 60 px
run("Subtract Background...", "rolling=60 stack");

// 4. 3D median filter, radius 1 voxel in x, y, z
run("Median 3D...", "x=1 y=1 z=1");

// 5. CLAHE, applied slice by slice
blocksize = 127;
histogram_bins = 256;
max_slope = 3;
for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    run("Enhance Local Contrast (CLAHE)", "blocksize=" + blocksize + " histogram=" + histogram_bins +
        " maximum=" + max_slope + " mask=*None*");
}

// 6. Intensity normalisation (linear histogram stretch over the whole stack)
// Saturation value not stated in the thesis; 0.35 is the Fiji default (see ../DISCREPANCIES.md)
run("Enhance Contrast...", "saturated=0.35 normalize process_all use");
