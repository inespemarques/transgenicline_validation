
import time
import os
import tifffile
from natsort import natsorted
from suite2p import io, registration
from suite2p.registration.register import register_frames
from suite2p.registration.register import compute_reference
from suite2p.registration.rigid import phasecorr
from suite2p import default_ops
from suite2p.run_s2p import run_s2p
from suite2p.io.binary import BinaryFile
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
import numpy as np
import h5py
import pandas as pd



# Within plane motion correction 

def withinPlane_correction(input_dir, data_info, ops, output_dir, startP, endP, pad_pixels):
    #Performs motion correction of all the frames of a given plane using suite2p's main registration function
    #Inputs: 
    #
    #  - input_dir: a folder where the data is. The format of the data should be 1 file per plane with all the frames of that plane, in hdf5 format
    #  - data_info: a dictionary containing the information of the data - data_info={'Ly':464,'Lx':720,'frames_per_plane':250, 'total_planes':180}
    #  - ops: The standard ops dictionary for suite2p, where the registration parameters are defined 
    #  - output_dir: a folder where the outputs will be saved
    #  - StartP: Plane where the alignment should begin 
    #  - EndP :Plane where the alignment should end
    #  - pad_pixels: (experimental) Since Suite2P uses np.roll to applly the shifts, if there are large enoughshifts the images will 'roll' over to the opposite      #                size. To prevent this we add a padding of pad_pixels. Currently not using this feature, best if set to 0
    #Outputs:
    #
    #  - binary files (_reg.bin files) corresponding to each aligned plane 
    #  - one large TIFF file corresponding to the whole stack OR one TIFF of the whole stack with each plane averaged (see code)
    #  

    #Get information about the images 
    Lx = data_info['Lx']
    Ly = data_info['Ly']
    frames_per_plane = data_info['frames_per_plane']
    total_planes = data_info['total_planes']

    #paths 
    output_tif = os.path.join(output_dir, 'motion_corrected_all_planes_within.tif')
    
    with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:

        for p in range(1, total_planes + 1):   # ✅ PROCESS ALL PLANES
            print(f"\nProcessing plane {p} ...")
    
            # --------------------------------------------------------
            # STEP 1 — Load raw TIFF plane
            # --------------------------------------------------------

            tiff_file = os.path.join(input_dir, f"p{p:02d}.tif")
            raw_stack = tifffile.imread(tiff_file)  # shape (frames, Ly, Lx)

            frames = raw_stack.shape[0]
            
            '''
            This section was meant to read .hdf5 files instead of TIFFs. I no longer think that's a good idea. 
            #original snippet:
            infile = os.path.join(input_dir, f"p{p:02d}.h5")
            with h5py.File(infile, "r") as f:
                raw_stack = f["data"][:]   # shape (frames, Ly, Lx)

            #This was the alternative snippet made to handle the erros (unsucessfully):  
            infile = os.path.join(input_dir, f"p{p:02d}.h5")
            with h5py.File(infile, "r") as f:
                dset = f["data"]
                shape = dset.shape
                raw_stack = np.empty(shape, dtype=dset.dtype)
            
                for i in range(shape[0]):  # frame-by-frame read
                    success = False
                    for attempt in range(3):  # retry up to 3 times
                        try:
                            raw_stack[i] = dset[i]
                            success = True
                            break
                        except OSError as e:
                            print(f"WARNING: failed to read frame {i} in plane {p} (attempt {attempt+1}/3): {e}")
                            import time
                            time.sleep(0.1)  # short delay before retry
            
                    if not success:
                        # Stop immediately if a frame cannot be read after retries
                        raise RuntimeError(f"ERROR: Could not read frame {i} in plane {p} after 3 attempts. Aborting.")
            
            frames = raw_stack.shape[0]
            '''
            # --------------------------------------------------------
            # STEP 2 — Padding (same for all planes)
            # --------------------------------------------------------
            if pad_pixels > 0:
                mean_value = int(np.round(np.mean(raw_stack)))
                raw_stack = np.pad(
                    raw_stack,
                    pad_width=((0, 0),
                               (pad_pixels, pad_pixels),
                               (pad_pixels, pad_pixels)),
                    mode="constant",
                    constant_values=mean_value
                )
                Ly_padded = raw_stack.shape[1]
                Lx_padded = raw_stack.shape[2]
            else:
                Ly_padded = Ly
                Lx_padded = Lx
    
            # --------------------------------------------------------
            # STEP 3 — Write padded RAW to .bin for Suite2p
            # --------------------------------------------------------
            raw_bin = os.path.join(output_dir, f"p{p:02d}.bin")
            raw_stack.astype("int16").tofile(raw_bin)
    
            # --------------------------------------------------------
            # CASE A: plane inside [startP, endP] → do motion correction
            # --------------------------------------------------------
            if startP <= p <= endP:
    
                print(f"  -> Performing motion correction")
    
                f_raw = io.BinaryFile(Ly=Ly_padded, Lx=Lx_padded, filename=raw_bin, n_frames=frames)
    
                # Create binary for registered output
                reg_filename = os.path.join(output_dir, f"p{p:02d}_reg.bin")
                f_reg = io.BinaryFile(Ly=Ly_padded, Lx=Lx_padded, filename=reg_filename, n_frames=frames_per_plane)
    
                # Run Suite2p rigid/nonrigid alignment
                refImg, rmin, rmax, meanImg, rigid_offsets, \
                nonrigid_offsets, zest, meanImg_chan2, badframes, \
                yrange, xrange = registration.registration_wrapper(
                    f_reg,
                    f_raw=f_raw,
                    f_reg_chan2=None,
                    f_raw_chan2=None,
                    refImg=None,
                    align_by_chan2=False,
                    ops=ops
                )
    
                f_reg.file.flush()
    
                # Save badframes - Suite2P's bad frames isn't very useful so no point in saving it
                #badframes_filename = os.path.join(output_dir, f"p{p:02d}_badframes.npy")
                #np.save(badframes_filename, badframes)

                # =====================================================
                # SAVE REGISTRATION OFFSETS FOR BAD FRAME DETECTION
                # These shifts indicate how much each frame moved - 
                # large shifts = motion artifacts (independent of neural activity!)
                # =====================================================
                offsets_filename = os.path.join(output_dir, f"p{p:02d}_offsets.npy")
                np.save(offsets_filename, {
                    'yoff': rigid_offsets[0],  # Y shifts for each frame
                    'xoff': rigid_offsets[1],  # X shifts for each frame
                })
                print(f"  -> Saved registration offsets to {offsets_filename}")
                
                # Write registered frames to BigTIFF
                # If you want the TIFF to contain every frame uncomment this: 
                #for i in range(frames_per_plane):
                #    tif_writer.write(f_reg[i], compression="DEFLATE")
                # Otherwise, this writes a stack with the average of each plane: 
                tif_writer.write(meanImg.astype(raw_stack.dtype), compression="DEFLATE")
    
            # --------------------------------------------------------
            # CASE B: plane outside correction range → write raw padded
            # --------------------------------------------------------
            else:
                print(f"  -> Outside correction range; writing raw frames")
    
                # Save raw padded .bin ALSO with _reg filename for consistency
                reg_filename = os.path.join(output_dir, f"p{p:02d}_reg.bin")
                raw_stack.astype("int16").tofile(reg_filename)

                
                # Write raw padded frames to BigTIFF
                # If you want the TIFF to contain every frame uncomment this:
                #for i in range(frames):
                #    tif_writer.write(raw_stack[i], compression="DEFLATE")
                # Otherwise, this writes a stack with the average of each plane: 
                tif_writer.write(np.mean(raw_stack, axis=0).astype(raw_stack.dtype), compression="DEFLATE")




# Between-planes motion correction 


def betweenPlanes_correction(reg_dir, data_info, ops, output_dir, x_crop, y_crop, startP, endP, smooth):
    #Aligns every frame of one plane to the previous plane
    #Inputs: 
    #
    #  - input_dir: a folder where the registered binaries from the within plane correction are saved. It will look for files ending in _reg.bin
    #  - data_info: a dictionary containing the information of the data - data_info={'Ly':464,'Lx':720,'frames_per_plane':250}
    #  - ops: The standard ops dictionary for suite2p, where the registration parameters are defined 
    #  - output_dir: a folder where the outputs will be saved
    #  - x_crop and y_crop: Defines the area of interest for computing the shift
    #  - StartP and endP: first and last planes - should match what comes from motion correction. 
    #Outputs:
    #
    #  - binary files (_reg.bin files) corresponding to each aligned plane 
    #  - one large TIFF file corresponding to the whole stack OR one TIFF of the whole stack with each plane averaged (see code) 
    #  - prints a vector of x_shifts and y_shifts 

    # Basic parameters
    Lx = data_info['Lx']
    Ly = data_info['Ly']
    frames_per_plane = data_info['frames_per_plane']
    ops.update({'nonrigid': False, 'do_bidiphase': False, 'bidiphase': 0, 'bidi_corrected': True, 'maxregshift':0.01})

    # Paths
    final_reg_dir = os.path.join(output_dir, "registeredBin")
    os.makedirs(final_reg_dir, exist_ok=True)
    output_tif = os.path.join(output_dir, 'motion_corrected_all_planes_between.tif')

    # Collect registered binaries
    reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

    # Open BigTIFF writer
    with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:
    
        aligned_templates = []      # store mean images for sequential reference
        x_shifts = []
        y_shifts = []
    
        for i, fname in enumerate(reg_files):
            plane_number = i + 1
            print(f"\n🔹 Processing plane {plane_number} — file {fname}")
    
            # ------------------------------------------------------------
            # LOAD RAW REGISTERED BINARY
            # ------------------------------------------------------------
            path_in = os.path.join(reg_dir, fname)
            path_out = os.path.join(final_reg_dir, f"aligned_{fname}")
            # Input suite2p binary 
            f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)
            raw_stack=f_raw[:]
            # Create output binary
            f_aligned = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_out, n_frames=frames_per_plane)
    
            # ------------------------------------------------------------
            # CASE A — plane inside correction range → align 
            # ------------------------------------------------------------
            if startP <= plane_number <= endP:
    
                print("Performing inter-plane alignment")
    
                # Determine reference template
                if len(aligned_templates) == 0:
                    # First corrected plane
                    ref_frames = raw_stack
                    #refImg = compute_reference(ref_frames, ops)
                    refImg = np.mean(raw_stack, axis=0)
                else:
                    refImg = aligned_templates[-1]

                # Crop images (we want to determine the shift based on a smaller field of view)
                #current_plane = compute_reference(raw_stack, ops)
                current_plane = np.mean(raw_stack, axis=0)
                #current_plane_cropped = current_plane[np.newaxis, y_crop:Ly-y_crop, x_crop:Lx-x_crop]
                current_plane_cropped = current_plane[y_crop:Ly-y_crop, x_crop:Lx-x_crop]
                refImg_cropped = refImg[y_crop:Ly-y_crop, x_crop:Lx-x_crop]
                
                # Determine shift 
                shifts, error, phase_diff = phase_cross_correlation(refImg_cropped, current_plane_cropped, upsample_factor=1, space='real',   disambiguate=False, reference_mask=None, moving_mask=None, overlap_ratio=0.3, normalization='phase')
                #dy, dx, cmax = phasecorr(current_plane_cropped, refImg_cropped, ops['maxregshift'], 0)
                dy=shifts[0]
                dx=shifts[1]

                if smooth>0 and i>1:
                    if abs(dy-y_shifts[i-1])>smooth:
                        dy=y_shifts[i-1]
                    if abs(dx-x_shifts[i-1])>smooth:
                        dx=x_shifts[i-1]
                        
                    
                x_shifts.append(dx)
                y_shifts.append(dy)
                        
                
                #Apply correction 
                aligned_stack = shift(
                    raw_stack, 
                    shift=[0, float(dy), float(dx)], 
                    order=0, 
                    mode='constant', 
                    cval=float(np.round(np.mean(raw_stack)))
                )      
    
                # Build next-plane reference
                ref_frames = aligned_stack
                #next_template = compute_reference(ref_frames, ops)
                next_template=np.mean(aligned_stack, axis=0)
                aligned_templates.append(next_template)
    
            # ------------------------------------------------------------
            # CASE B — plane *outside* correction range → copy raw
            # ------------------------------------------------------------
            else:
                print(" Outside correction range — copying frames without alignment")

                aligned_stack = raw_stack 
            # ------------------------------------------------------------
            # STEP — save .bin + BigTIFF
            # ------------------------------------------------------------
    
            f_aligned[:]=aligned_stack.astype(np.uint16)
            tif_writer.write(np.mean(aligned_stack, axis=0).astype(aligned_stack.dtype), compression="DEFLATE")

        print(x_shifts)
        print(y_shifts)
    
        print("\n✅ ✅ Finished: all planes processed and saved.")



def find_badFrames(input_dir, data_info, output_dir, threshold, offsets_dir=None):
    #Finds bad frames through REGISTRATION SHIFTS (not correlation!)
    #
    #IMPORTANT: This version uses the registration shifts computed by Suite2p
    #during motion correction. This is much better for calcium imaging because:
    #  - Neural activity (responses to stimuli) does NOT affect this metric
    #  - Only actual MOTION artifacts are detected
    #  - Large shifts = the frame had significant movement = potentially bad
    #
    #The old correlation-based method was problematic because frames with 
    #strong neural responses look different from the mean image and could
    #be incorrectly flagged as "bad" - removing the most interesting data!
    #
    #Inputs: 
    #
    #  - input_dir: folder with *_reg.bin files (for fallback correlation method)
    #  - data_info: dictionary with 'Ly', 'Lx', 'frames_per_plane'
    #  - output_dir: folder where the outputs will be saved
    #  - threshold: number of standard-deviations ABOVE the mean shift magnitude
    #  - offsets_dir: folder with *_offsets.npy files (default: same as input_dir)
    #                 Use this if offsets are in a different folder (e.g., withinPlane)
    #
    #Method:
    #  For each frame, compute shift magnitude: sqrt(yoff² + xoff²)
    #  Flag as bad if: magnitude > mean + threshold * std
    #
    #Outputs:
    #
    #  - badFrames_corr_all_planes.npy: dictionary with bad frame indices per plane
    #  - shift_stats_all_planes.npy: statistics for QC
    #

    frames_per_plane = data_info['frames_per_plane']
    
    # If offsets_dir not specified, look in input_dir
    if offsets_dir is None:
        offsets_dir = input_dir
    
    # Paths
    badframes_outfile = os.path.join(output_dir, "badFrames_corr_all_planes.npy")  # keep same name for compatibility
    stats_outfile = os.path.join(output_dir, "shift_stats_all_planes.npy")
    
    # Find offset files (created by withinPlane_correction)
    offset_files = natsorted([f for f in os.listdir(offsets_dir) if f.endswith("_offsets.npy")])
    
    if len(offset_files) == 0:
        print(f"[WARNING] No *_offsets.npy files found in {offsets_dir}")
        print("Make sure withinPlane_correction saved the registration offsets!")
        print("Falling back to the old correlation-based method...")
        # Fallback to old method if no offsets found
        return find_badFrames_correlation(input_dir, data_info, output_dir, threshold)
    
    badframes_dict = {}
    stats_dict = {}
    
    print(f"\n{'='*60}")
    print("BAD FRAME DETECTION BASED ON REGISTRATION SHIFTS")
    print(f"{'='*60}")
    print(f"Offsets loaded from: {offsets_dir}")
    print(f"Threshold: {threshold} SD above mean shift magnitude")
    print(f"This method is INDEPENDENT of neural activity!\n")
    
    for fname in offset_files:
        # Load offsets
        offsets = np.load(os.path.join(offsets_dir, fname), allow_pickle=True).item()
        yoff = offsets['yoff']
        xoff = offsets['xoff']
        
        # Compute shift magnitude for each frame: sqrt(y² + x²)
        shift_magnitude = np.sqrt(yoff**2 + xoff**2)
        
        # Statistics
        mean_shift = np.mean(shift_magnitude)
        std_shift = np.std(shift_magnitude)
        max_shift = np.max(shift_magnitude)
        
        # Threshold: mean + threshold * SD
        thresh = mean_shift + threshold * std_shift
        
        # Find bad frames (shifts larger than threshold)
        bad_indices = np.where(shift_magnitude > thresh)[0]
        
        # Store results - figure out the corresponding reg.bin key
        # Offsets are named like: p01_offsets.npy
        # Reg bins could be named: p01_reg.bin OR aligned_p01_reg.bin
        plane_base = fname.replace("_offsets.npy", "")  # e.g., "p01"
        
        # Try to find matching reg.bin in input_dir (where the bins are)
        reg_files_in_input = [f for f in os.listdir(input_dir) if f.endswith("_reg.bin")]
        matching_reg = None
        for rf in reg_files_in_input:
            if plane_base in rf:
                matching_reg = rf
                break
        
        if matching_reg is None:
            # Fallback: use the standard naming
            matching_reg = f"{plane_base}_reg.bin"
        
        badframes_dict[matching_reg] = np.array(bad_indices)
        stats_dict[matching_reg] = {
            'mean_shift': mean_shift,
            'std_shift': std_shift,
            'max_shift': max_shift,
            'threshold_used': thresh,
            'n_bad': len(bad_indices)
        }
        
        # Report
        print(f"Plane {plane_base}: {len(bad_indices):3d}/{frames_per_plane} bad frames | "
              f"shifts: mean={mean_shift:.2f}, max={max_shift:.2f} px | thresh={thresh:.2f}")
    
    # Save results
    np.save(badframes_outfile, badframes_dict)
    np.save(stats_outfile, stats_dict)
    
    # Summary
    total_bad = sum(len(v) for v in badframes_dict.values())
    total_frames = len(badframes_dict) * frames_per_plane
    
    print(f"\n{'='*60}")
    print(f"💾 Saved bad frames to: {badframes_outfile}")
    print(f"💾 Saved shift statistics to: {stats_outfile}")
    print(f"\n📊 SUMMARY: {total_bad}/{total_frames} total bad frames "
          f"({100*total_bad/total_frames:.2f}%)")


def find_badFrames_correlation(input_dir, data_info, output_dir, threshold):
    #OLD METHOD - Finds bad frames through correlation (NOT RECOMMENDED)
    #
    #WARNING: This method can incorrectly flag frames with strong neural 
    #responses as "bad" because they look different from the mean image.
    #Use find_badFrames (shift-based) instead!
    #
    #This function is kept as a fallback if offset files are not available.
    #

    Lx = data_info['Lx']
    Ly = data_info['Ly']
    frames_per_plane = data_info['frames_per_plane']
    # Paths
    badframes_outfile = os.path.join(output_dir, "badFrames_corr_all_planes.npy")  # final output
    # Get all registered binaries and sort naturally
    #reg_dir = os.path.join(input_dir, "registeredBin")
    reg_dir = input_dir
    reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

    # Dictionary to store bad frames per plane
    badframes_dict = {}
    corr_dict = {}
    
    print("\n⚠️  WARNING: Using correlation-based bad frame detection!")
    print("This may incorrectly flag frames with neural responses as bad.\n")
    
    # Loop over planes
    for i, fname in enumerate(reg_files):
        path_in = os.path.join(reg_dir, fname)
        f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)

        # Compute plane average
        avg_img = np.mean([f_raw[j] for j in range(frames_per_plane)], axis=0)

        # Compute correlation of each frame with average
        corr_list=[]
        badframes_plane = []
        for j in range(frames_per_plane):
            frame = f_raw[j]
            corr = np.corrcoef(frame.flatten(), avg_img.flatten())[0, 1]
            corr_list.append(corr)

        for k in range(frames_per_plane):
            corr = corr_list[k]
            mean_corr = np.mean(corr_list)
            sd_corr = np.std(corr_list)
            if corr < mean_corr-threshold*sd_corr:
                badframes_plane.append(k)

        badframes_dict[fname] = np.array(badframes_plane)
        corr_dict[fname] = np.array(corr_list)
        print(f"Plane {fname}: {len(badframes_plane)}/{frames_per_plane} bad frames detected")

    # Save dictionary
    np.save(badframes_outfile, badframes_dict)
    print(f"\n💾 Saved correlation-based bad frames for all planes to {badframes_outfile}")


def write_clean_tiffs(input_dir, data_info, output_dir):
    #Save per-plane TIFFs where bad frames are replaced with NaNs 
    #Inputs: 
    #
    #  - input_dir: a folder where the data is. It expects the registered binaries that suite2p produces
    #  - data_info: a dictionary containing the information of the data - data_info={'Ly':464,'Lx':720,'frames_per_plane':250}
    #  - output_dir: a folder where the outputs will be saved
    #  - threshold: number of standard-deviations below the mean correlation of all frames 
    #
    
    #Outputs:
    #
    #  - one TIFF per plane with bad frames replaced by NAN's 
    #


    Lx = data_info['Lx']
    Ly = data_info['Ly']
    frames_per_plane = data_info['frames_per_plane']

    # Paths
    #final_reg_dir = os.path.join(input_dir, "registeredBin")
    final_reg_dir = input_dir
    #total_badframes_path = os.path.join(input_dir, 'badFrames_all_planes.npy')
    corr_badframes_path = os.path.join(input_dir, 'badFrames_corr_all_planes.npy')

    # Load both bad frame dictionaries
    #total_badframes_dict = np.load(total_badframes_path, allow_pickle=True).item()
    corr_badframes_dict = np.load(corr_badframes_path, allow_pickle=True).item()

    # Get registered binary files in correct numeric order
    reg_files = natsorted([f for f in os.listdir(final_reg_dir) if f.endswith("_reg.bin")])

    print(f"Found {len(reg_files)} registered planes.")
    print(f"Saving cleaned TIFFs (NaN in bad frames) to: {output_dir}")

    for i, fname in enumerate(reg_files):
        path_in = os.path.join(final_reg_dir, fname)
        print(f"\nProcessing plane {i+1}/{len(reg_files)}: {fname}")

        # Load registered binary
        f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)

        # --- Combine bad frames from both sources ---
        plane_key = f"plane_{i+1:02d}"
        #merged_bad = total_badframes_dict.get(plane_key, total_badframes_dict.get(fname, np.array([], dtype=int)))
        merged_bad = corr_badframes_dict.get(fname, np.array([], dtype=int))
        #merged_bad = np.unique(np.concatenate([bad1, bad2])) if bad1.size or bad2.size else np.array([], dtype=int)

        # --- Save this plane's TIFF ---
        output_tif_path = os.path.join(output_dir, fname.replace("_reg.bin", "_nan.tif"))
        with tifffile.TiffWriter(output_tif_path, bigtiff=True) as tif_writer:
            for j in range(frames_per_plane):
                if j in merged_bad:
                    # Replace bad frames with NaNs (same shape)
                    frame = np.full((Ly, Lx), np.nan, dtype=np.float32)
                else:
                    frame = f_raw[j].astype(np.float32)
                tif_writer.write(frame, compression="DEFLATE")

        print(f"✅ Plane {i+1}: wrote {frames_per_plane} frames ({len(merged_bad)} replaced with NaN) → {os.path.basename(output_tif_path)}")

    print("\n💾 All per-plane TIFFs saved with NaN-filled bad frames.")
    print(f"Output folder: {output_dir}")





'''
Old functions / versions

def anatomicalStack_correction(input_dir, x_crop, y_crop, startP, endP):
    #DOESN'T WORK
    #Aligns every each plane to the previous plane
    #Inputs: 
    #
    #  - input_dir: a folder where there are two folders, 'green_planes' and 'red_planes' which contain the tiff images recorded in the green and red channel
    #  - x_crop and y_crop: Defines the area of interest for computing the shift
    #  - StartP and endP: first and last planes. 
    #Outputs:
    # 
    #  - the aligned anatomical stacks for the green and red channel (shifts are computed in the green and are then applied to the red, so they are always aligned)
    #  - a csv file with the applied x_shifts and y_shifts 

    # Paths
    green_dir = os.path.join(input_dir, "green_planes")
    red_dir = os.path.join(input_dir, "red_planes")
    green_aligned_dir = os.path.join(green_dir, "motionCorrected")
    os.makedirs(green_aligned_dir, exist_ok=True)
    red_aligned_dir = os.path.join(red_dir, "motionCorrected")
    os.makedirs(red_aligned_dir, exist_ok=True)
    output_green_tif = os.path.join(green_aligned_dir, 'green_anatomicalStack.tif')
    output_red_tif = os.path.join(red_aligned_dir, 'red_anatomicalStack.tif')

    # Collect registered binaries
    green_planes = natsorted([f for f in os.listdir(green_dir) if f.endswith(".tif")])
    red_planes = natsorted([f for f in os.listdir(red_dir) if f.endswith(".tif")])

    assert len(green_planes)==len(red_planes), "number of green and red planes don't match!"
    
    # Open BigTIFF writer
    with tifffile.TiffWriter(output_green_tif, bigtiff=True) as tif_green_writer:
        with tifffile.TiffWriter(output_red_tif, bigtiff=True) as tif_red_writer:
            
            aligned_templates = []      # store mean images for sequential reference
            x_shifts = []
            y_shifts = []
    
            i=1
            for green_plane, red_plane in zip(green_planes, red_planes):
    
                print(f"\n🔹 Processing plane number {i}: {green_plane}")
        
                # ------------------------------------------------------------
                # Load current plane
                # ------------------------------------------------------------
                current_green_plane = tifffile.imread(os.path.join(green_dir, green_plane))  # shape (Ly, Lx)
                Ly, Lx = current_green_plane.shape
                current_red_plane = tifffile.imread(os.path.join(red_dir, red_plane))
        
                # ------------------------------------------------------------
                # CASE A — plane inside correction range → align 
                # ------------------------------------------------------------
                if startP <= i <= endP:
        
                    print("Performing inter-plane alignment")
        
                    # Determine reference template
                    if len(aligned_templates) == 0:
                        # First corrected plane
                        refImg = current_green_plane
                    else:
                        refImg = aligned_templates[-1]
    
                    # Crop images (we want to determine the shift based on a smaller field of view)
                    current_green_plane_cropped = current_green_plane[y_crop:Ly-y_crop, x_crop:Lx-x_crop]
                    refImg_cropped = refImg[y_crop:Ly-y_crop, x_crop:Lx-x_crop]
                    
                    # Determine shift 
                    shifts, error, phase_diff = phase_cross_correlation(refImg_cropped, current_green_plane_cropped, upsample_factor=1, space='real',   disambiguate=False, reference_mask=None, moving_mask=None, overlap_ratio=0.3, normalization='phase')
                    dy=shifts[0]
                    dx=shifts[1]
    
                    x_shifts.append(dx)
                    y_shifts.append(dy)
                    
                    #Apply correction 
                    aligned_green_plane = shift(
                        current_green_plane, 
                        shift=[float(dy), float(dx)], 
                        order=0, 
                        mode='constant', 
                        cval=float(np.round(np.mean(current_green_plane)))
                    )     
    
                    #Apply correction to red channel using the same shifts
                    aligned_red_plane = shift(
                        current_red_plane, 
                        shift=[float(dy), float(dx)], 
                        order=0, 
                        mode='constant', 
                        cval=float(np.round(np.mean(current_red_plane)))
                    )  
        
                    # Build next-plane reference
                    next_template=aligned_green_plane
                    aligned_templates.append(next_template)
        
                # ------------------------------------------------------------
                # CASE B — plane *outside* correction range → copy raw
                # ------------------------------------------------------------
                else:
                    print(" Outside correction range — copying frames without alignment")
    
                    aligned_green_plane = current_green_plane
                    aligned_red_plane = current_red_plane 
                    
                # ------------------------------------------------------------
                # STEP — save TIFF
                # ------------------------------------------------------------
        
                tif_green_writer.write(aligned_green_plane, compression="DEFLATE")
                tif_red_writer.write(aligned_red_plane, compression="DEFLATE")
                i+=1

        shifts_df = pd.DataFrame({"x_shifts": x_shifts, "y_shifts": y_shifts})
        shifts_df.to_csv(os.path.join(input_dir, "shifts.csv"))
    
        print("\n✅ ✅ Finished: all planes processed and saved.")

def betweenPlanes_correction(reg_dir, data_info, ops, output_dir, pad_pixels, startP, endP):
    #Aligns every frame of one plane to the previous plane
    #Inputs: 
    #
    #  - input_dir: a folder where the registered binaries from the within plane correction are saved. It will look for files ending in _reg.bin
    #  - data_info: a dictionary containing the information of the data - data_info={'Ly':464,'Lx':720,'frames_per_plane':250}
    #  - ops: The standard ops dictionary for suite2p, where the registration parameters are defined 
    #  - output_dir: a folder where the outputs will be saved
    #  - pad_pixels: size of padding so that it can crop the output to its original size 
    #
    #Outputs:
    #
    #  - binary files (_reg.bin files) corresponding to each aligned plane 
    #  - one large TIFF file corresponding to the whole stack
    #

    Lx = data_info['Lx']
    Ly = data_info['Ly']
    frames_per_plane = data_info['frames_per_plane']

    # Paths
    final_reg_dir = os.path.join(output_dir, "registeredBin")
    os.makedirs(final_reg_dir, exist_ok=True)
    output_tif = os.path.join(output_dir, 'motion_corrected_all_planes_between.tif')
    badframes_all_planes_file = os.path.join(output_dir, 'badFrames_all_planes.npy')

    # Collect registered binaries
    reg_files = natsorted([f for f in os.listdir(reg_dir) if f.endswith("_reg.bin")])

    aligned_templates = []           # store mean images for sequential reference
    total_badframes_all_planes = []  # list of arrays for each plane

    #Where to crop the resulting image so that it stays the same size
    crop_y_start = pad_pixels
    crop_y_end = Ly - pad_pixels 
    crop_x_start = pad_pixels
    crop_x_end = Lx - pad_pixels 

    ops.update({'nonrigid': False, 'do_bidiphase': False, 'bidiphase': 0, 'bidi_corrected': True})

    # Open BigTIFF writer
    with tifffile.TiffWriter(output_tif, bigtiff=True) as tif_writer:
    
        aligned_templates = []
        total_badframes_all_planes = []
    
        for i, fname in enumerate(reg_files):
            plane_number = i + 1
            print(f"\n🔹 Processing plane {plane_number} — file {fname}")
    
            # ------------------------------------------------------------
            # LOAD RAW REGISTERED BINARY
            # ------------------------------------------------------------
            path_in = os.path.join(reg_dir, fname)
            path_temp_padded = os.path.join(final_reg_dir, f"temp_padded_{fname}")
            path_out_cropped = os.path.join(final_reg_dir, f"final_cropped_{fname}")
    
            # Input suite2p binary (with padding from the previous code or no padding)
            f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=path_in, n_frames=frames_per_plane)

            # --------------------------------------------------------
            # Padding (same for all planes)
            # --------------------------------------------------------
            raw_stack = f_raw[:]
            
            if pad_pixels > 0:
                mean_value = int(np.round(np.mean(raw_stack)))
                raw_stack = np.pad(
                    raw_stack,
                    pad_width=((0, 0),
                               (pad_pixels, pad_pixels),
                               (pad_pixels, pad_pixels)),
                    mode="constant",
                    constant_values=mean_value
                )
                Ly_padded = raw_stack.shape[1]
                Lx_padded = raw_stack.shape[2]
            else:
                Ly_padded = Ly
                Lx_padded = Lx
    
            # --------------------------------------------------------
            # STEP 3 — Write padded RAW to .bin for Suite2p
            # --------------------------------------------------------
            raw_bin = os.path.join(output_dir, f"p{p:02d}.bin")
            raw_stack.astype("int16").tofile(raw_bin)

            # Input suite2p binary (padded)
            f_raw = io.BinaryFile(Ly=Ly_padded, Lx=Lx_padded, filename=raw_bin, n_frames=frames_per_plane)
    
            # Output temporary padded file
            f_aligned = io.BinaryFile(Ly=Ly_padded, Lx=Lx_padded, filename=path_temp_padded, n_frames=frames_per_plane)
    
            # ------------------------------------------------------------
            # LOAD BAD FRAMES FROM CELL 5
            # ------------------------------------------------------------
            badframes_cell5_file = os.path.join(reg_dir, f"p{plane_number:02d}_badframes.npy")
            badframes_cell5 = (
                np.load(badframes_cell5_file)
                if os.path.exists(badframes_cell5_file)
                else np.array([], dtype=int)
            )
    
            # ------------------------------------------------------------
            # CASE A — plane inside correction range → align using cell6
            # ------------------------------------------------------------
            if startP <= plane_number <= endP:
    
                print("   ✅ Performing inter-plane alignment")
    
                # Determine reference template
                if len(aligned_templates) == 0:
                    # First corrected plane: mean of good frames
                    good_idx = np.setdiff1d(np.arange(frames_per_plane), badframes_cell5)
                    ref_frames = np.array([f_raw[j] for j in good_idx])
                    refImg = compute_reference(ref_frames, ops)
                else:
                    refImg = aligned_templates[-1]
    
                # Run Suite2p alignment
                refImg, rmin, rmax, meanImg, rigid_offsets, \
                nonrigid_offsets, zest, meanImg_chan2, badframes_cell6, \
                yrange, xrange = registration.registration_wrapper(
                    f_aligned,
                    f_raw=f_raw,
                    f_reg_chan2=None,
                    f_raw_chan2=None,
                    refImg=refImg,
                    align_by_chan2=False,
                    ops=ops
                )
    
                # Combine bad frames
                total_bad = np.union1d(badframes_cell5, np.where(badframes_cell6)[0])
    
                # Build next-plane reference
                good_idx = np.setdiff1d(np.arange(frames_per_plane), total_bad)
                if len(good_idx) > 0:
                    ref_frames = np.array([f_aligned[j] for j in good_idx])
                    next_template = compute_reference(ref_frames, ops)
                else:
                    next_template = refImg  # fallback
    
                aligned_templates.append(next_template)
                total_badframes_all_planes.append(total_bad)
    
            # ------------------------------------------------------------
            # CASE B — plane *outside* correction range → copy raw
            # ------------------------------------------------------------
            else:
                print("   ⚠️ Outside correction range — copying frames without alignment")
    
                for j in range(frames_per_plane):
                    f_aligned.write(f_raw[j])
    
                f_aligned.file.flush()
    
                # No cell6 badframes for this plane
                total_badframes_all_planes.append(badframes_cell5)
    
            # ------------------------------------------------------------
            # STEP — Crop padded → original size, save .bin + BigTIFF
            # ------------------------------------------------------------
            print("   ✂️ Cropping & writing output files")
    
            with open(path_out_cropped, "wb") as f_bin_cropped:
                for j in range(frames_per_plane):
                    padded_frame = f_aligned[j]
                    cropped_frame = padded_frame[
                        crop_y_start:crop_y_end,
                        crop_x_start:crop_x_end
                    ]
    
                    # Save to final cropped .bin
                    cropped_frame.astype(np.uint16).tofile(f_bin_cropped)
    
                    # Save to BigTIFF
                    tif_writer.write(cropped_frame, compression="DEFLATE")
            
            os.remove(path_temp_padded)
        # ------------------------------------------------------------
        # SAVE PER-PLANE BADFRAME DICTIONARY
        # ------------------------------------------------------------
        badframes_dict = {
            f"plane_{k+1:02d}": total_badframes_all_planes[k]
            for k in range(len(total_badframes_all_planes))
        }
        np.save(badframes_all_planes_file, badframes_dict, allow_pickle=True)
    
        print("\n✅ ✅ Finished: all planes processed and saved.")

'''