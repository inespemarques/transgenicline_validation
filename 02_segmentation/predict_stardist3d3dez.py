###############################################################
# StarDist3D Inference + XY/XZ Overlays + Comparison with Manual Annotation
###############################################################

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist.models import StarDist3D
from stardist.matching import matching
from stardist import random_label_cmap


###############################################################
# USER INPUTS
###############################################################

MODEL_NAME = "stardist3d_mydata"
MODEL_BASEDIR = "models"

# Your test image
TEST_IMAGE_PATH = r"C:\Users\ABBE User\Downloads\fish3lente40x_0_3z-_onlyraw_smallregionAiryscanProcessingstandard_downsamples0_5_140slices.tif"

# Your manual ground truth mask
MANUAL_MASK_PATH = r"C:\Users\ABBE User\Downloads\manual_anot.tif"

# Output TIFF for predicted labels
OUTPUT_TIF = r"prediction_labels.tif"


###############################################################
# LOAD MODEL
###############################################################

print("\n=== Loading trained StarDist3D model ===")
model = StarDist3D(None, name=MODEL_NAME, basedir=MODEL_BASEDIR)
print("Model loaded successfully.")
print("Thresholds:", model.thresholds)


###############################################################
# LOAD AND NORMALIZE TEST IMAGE
###############################################################

print("\n=== Loading test image ===")
img = imread(TEST_IMAGE_PATH)
print("Image shape:", img.shape)

img_norm = normalize(img, 1, 99.8)


###############################################################
# PREDICTION
###############################################################

print("\n=== Running StarDist3D prediction ===")
labels_pred, details = model.predict_instances(
    img_norm,
    n_tiles = (2, 2, 2)   # tenta isto primeiro
)
print("Prediction done. Predicted objects:", labels_pred.max())



###############################################################
# VISUALISATION — XY AND XZ OVERLAY
###############################################################

print("\n=== Showing XY/XZ overlays ===")

lbl_cmap = random_label_cmap()

z = img.shape[0] // 2      # middle slice
y = img.shape[1] // 2      # middle Y for XZ

img_show = img_norm

plt.figure(figsize=(15,10))

# XY raw
plt.subplot(221)
plt.title("XY Raw Image")
plt.imshow(img_show[z], cmap="gray", clim=(0,1))
plt.axis("off")

# XY overlay
plt.subplot(222)
plt.title("XY Prediction Overlay")
plt.imshow(img_show[z], cmap="gray", clim=(0,1))
plt.imshow(labels_pred[z], cmap=lbl_cmap, alpha=0.4)
plt.axis("off")

# XZ raw
plt.subplot(223)
plt.title("XZ Raw Image")
plt.imshow(img_show[:, y], cmap="gray", clim=(0,1))
plt.axis("off")

# XZ overlay
plt.subplot(224)
plt.title("XZ Prediction Overlay")
plt.imshow(img_show[:, y], cmap="gray", clim=(0,1))
plt.imshow(labels_pred[:, y], cmap=lbl_cmap, alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.show()


###############################################################
# SAVE PREDICTED LABEL TIFF
###############################################################

print("\n=== Saving predicted labels TIFF ===")
save_tiff_imagej_compatible(OUTPUT_TIF, labels_pred.astype(np.uint16), axes="ZYX")
print(f"Saved prediction to: {OUTPUT_TIF}")

print("\n=== DONE ===")
