// Step 1: XY downscaling (Fiji)
// Parameters from thesis Section 3.5. Reconstructed, not the original macro; see ../DISCREPANCIES.md.
// Input: open 16-bit stack of the H2B-GCaMP6s channel (after Split Channels).
// Downscales XY by 0.5 with bicubic interpolation; Z is unchanged.

title = getTitle();
getDimensions(width, height, channels, slices, frames);
run("Scale...", "x=0.5 y=0.5 z=1.0 width=" + round(width/2) + " height=" + round(height/2) +
    " depth=" + slices + " interpolation=Bicubic average process create title=" + title + "_downscaled");
