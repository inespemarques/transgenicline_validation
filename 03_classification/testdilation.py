#!/usr/bin/env python3
"""
Fast dilation WITH PIXEL OVERLAP - pixels can belong to multiple nuclei

This is the KEY difference from Voronoi:
- Voronoi: each exterior pixel belongs to EXACTLY ONE nucleus
- Dilation: each exterior pixel can belong to MULTIPLE nuclei
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

def create_shell_dilation_with_overlap(label_slice, start_dist, end_dist):
    """
    Create shell allowing pixel overlap between nuclei.
    
    Key idea: For each nucleus, we track which pixels fall in its shell.
    The SAME exterior pixel can be in multiple nuclei's shells!
    
    This is FAST because we use vectorized distance calculations.
    """
    
    # Get all unique labels
    all_labels = np.unique(label_slice)
    all_labels = all_labels[all_labels > 0]
    
    if len(all_labels) == 0:
        return {}
    
    # For each nucleus, compute which pixels belong to its shell
    # Store as dictionary: {label: pixel_indices}
    nucleus_shells = {}
    
    for lbl in all_labels:
        # Create mask for this specific nucleus
        nucleus_mask = (label_slice == lbl)
        
        # Compute distance FROM this nucleus
        dist_inside = distance_transform_edt(nucleus_mask)
        dist_outside = distance_transform_edt(~nucleus_mask)
        
        # Signed distance for THIS nucleus
        signed_dist = np.where(nucleus_mask, -dist_inside, dist_outside)
        
        # Shell region for THIS nucleus
        shell_mask = (signed_dist >= float(start_dist)) & (signed_dist <= float(end_dist))
        
        # Store pixel coordinates in this nucleus's shell
        shell_coords = np.argwhere(shell_mask)
        
        if len(shell_coords) > 0:
            nucleus_shells[int(lbl)] = shell_coords
    
    return nucleus_shells


def compute_features_with_overlap(intensity, nucleus_shells, bright_threshold):
    """
    Compute features allowing pixel overlap.
    
    CRITICAL: The same pixel can contribute to MULTIPLE nuclei!
    This inflates pct_bright for nuclei in crowded regions.
    """
    import pandas as pd
    
    results = []
    
    for lbl, coords in nucleus_shells.items():
        if len(coords) == 0:
            continue
        
        # Extract intensity values for THIS nucleus's shell
        # Note: pixels may also be in OTHER nuclei's shells!
        ys, xs = coords[:, 0], coords[:, 1]
        intensities = intensity[ys, xs]
        
        # Calculate features
        n_pixels = len(intensities)
        bright_pixels = np.sum(intensities > float(bright_threshold))
        pct_bright = (bright_pixels / n_pixels * 100.0) if n_pixels > 0 else 0.0
        
        results.append({
            "label": int(lbl),
            "n_shell_px": int(n_pixels),
            "pct_bright": float(pct_bright),
        })
    
    return pd.DataFrame(results)


# Example usage comparison:
if __name__ == "__main__":
    # Create test image
    label_img = np.zeros((100, 100), dtype=np.int32)
    
    # Two close nuclei
    label_img[20:30, 20:30] = 1  # Nucleus 1 (label=1)
    label_img[25:35, 35:45] = 2  # Nucleus 2 (label=2) - CLOSE to nucleus 1!
    
    # Create intensity with bright region between nuclei
    intensity = np.random.rand(100, 100) * 100
    intensity[27:28, 30:36] = 200  # Bright strip between nuclei
    
    # Compute shells with overlap
    shells = create_shell_dilation_with_overlap(label_img, start_dist=0, end_dist=5)
    
    print("Dilation with overlap:")
    for lbl, coords in shells.items():
        print(f"  Nucleus {lbl}: {len(coords)} pixels in shell")
    
    # Check for overlapping pixels
    all_pixels = {}
    for lbl, coords in shells.items():
        for coord in coords:
            key = tuple(coord)
            if key not in all_pixels:
                all_pixels[key] = []
            all_pixels[key].append(lbl)
    
    overlapping = {k: v for k, v in all_pixels.items() if len(v) > 1}
    print(f"\n{len(overlapping)} pixels belong to multiple nuclei!")
    print(f"Example overlapping pixels: {list(overlapping.items())[:5]}")
    
    # Compute features
    features = compute_features_with_overlap(intensity, shells, bright_threshold=150)
    print("\nFeatures:")
    print(features)
    
    print("\n" + "="*60)
    print("KEY POINT:")
    print("The bright strip at y=27, x=30-36 contributes to BOTH nuclei!")
    print("This INFLATES pct_bright for both, showing why Voronoi is better.")
    print("="*60)