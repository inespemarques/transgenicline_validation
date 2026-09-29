###############################################################
# OFFICIAL STARdist3D TRAINING PIPELINE (TensorFlow/Keras)
# Compatible with older StarDist3D versions expecting:
# StarDistData3D(X, Y, batch_size, patch_size, rays, length)
###############################################################

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
from glob import glob
from tifffile import imread
from tqdm import tqdm

from csbdeep.utils import Path, normalize
from stardist import calculate_extents, Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D, StarDistData3D


###############################################################
# USER PATHS
###############################################################

DATA_ROOT = r"C:\Users\ABBE User\Downloads\data_root"

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATA_ROOT, "train", "labels_fixed")  # using fixed labels

X_paths = sorted(glob(os.path.join(TRAIN_IMG_DIR, "*.tif")))
Y_paths = sorted(glob(os.path.join(TRAIN_LBL_DIR, "*.tif")))

assert len(X_paths) == len(Y_paths), "ERROR: number of images != number of masks"
assert len(X_paths) > 1, "ERROR: Not enough 3D volumes to train."


###############################################################
# STEP 1 — LOAD VOLUMES (NO FILL-HOLES)
###############################################################

print("\n=== Loading 3D volumes ===")

X = [imread(p) for p in tqdm(X_paths)]
Y = [imread(p) for p in tqdm(Y_paths)]

print(f"Loaded {len(X)} volumes.")


###############################################################
# STEP 2 — NORMALIZE IMAGES
###############################################################

print("\n=== Normalizing (1–99.8 percentile) ===")
X = [normalize(x, 1, 99.8) for x in tqdm(X)]


###############################################################
# STEP 3 — TRAIN/VALIDATION SPLIT (20%)
###############################################################

print("\n=== Splitting into train/validation ===")
rng = np.random.RandomState(42)
inds = rng.permutation(len(X))

n_val = max(1, int(len(inds) * 0.20))
ind_train, ind_val = inds[:-n_val], inds[-n_val:]

X_train = [X[i] for i in ind_train]
Y_train = [Y[i] for i in ind_train]
X_val   = [X[i] for i in ind_val]
Y_val   = [Y[i] for i in ind_val]

print(f"Total volumes:      {len(X)}")
print(f"Training volumes:   {len(X_train)}")
print(f"Validation volumes: {len(X_val)}")


###############################################################
# STEP 4 — CONFIGURE MODEL
###############################################################

print("\n=== Computing anisotropy ===")
extents = calculate_extents(Y)
emp_aniso = tuple(np.max(extents) / extents)
print("Empirical anisotropy:", emp_aniso)

ANISO = (2.7, 1, 1)  # Your anisotropy
print("Using anisotropy:", ANISO)

n_rays = 96
rays = Rays_GoldenSpiral(n_rays, anisotropy=ANISO)
grid = tuple(1 if a > 1.5 else 2 for a in ANISO)

conf = Config3D(
    rays=rays,
    grid=grid,
    anisotropy=ANISO,
    n_channel_in=1,
    train_patch_size=(32,128,128),
    train_batch_size=2,
    train_foreground_only=0.9,
    train_epochs=400,
    train_steps_per_epoch=100,
    train_learning_rate=0.0003,
    train_loss_weights=(1,0.2),
    train_dist_loss="mae",
    use_gpu=False
)

print("\n=== Model Config ===")
print(conf)


###############################################################
# STEP 5 — CREATE OFFICIAL DATA GENERATORS
###############################################################

print("\n=== Creating StarDistData3D generators ===")

# Number of patches per epoch = steps_per_epoch × batch_size
length_train = conf.train_steps_per_epoch * conf.train_batch_size
length_val   = max(1, int(0.20 * length_train))

data_train = StarDistData3D(
    X_train,
    Y_train,
    batch_size=conf.train_batch_size,
    patch_size=conf.train_patch_size,
    rays=rays,
    length=length_train
)

data_val = StarDistData3D(
    X_val,
    Y_val,
    batch_size=conf.train_batch_size,
    patch_size=conf.train_patch_size,
    rays=rays,
    length=length_val
)

print("Generators created.")


###############################################################
###############################################################
# STEP 6 — TRAIN MODEL  (YOUR VERSION OF STARdist3D)
###############################################################

print("\n=== Initialising model ===")
model = StarDist3D(conf, name="stardist3d_mydata", basedir="models")

print("\n=== Training started ===")

model.train(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    augmenter=None,
    epochs=conf.train_epochs,
    steps_per_epoch=conf.train_steps_per_epoch,
)

print("\n=== Training complete ===")


###############################################################
# STEP 7 — OPTIMISE THRESHOLDS
###############################################################

print("\n=== Optimising NMS + probability thresholds ===")
model.optimize_thresholds(X_val, Y_val)

print("\n=== DONE. MODEL TRAINED SUCCESSFULLY ===\n")



