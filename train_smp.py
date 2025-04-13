import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from my_functions import dice_coefficient, calculate_iou, basic_transform, dice_bce_loss
import segmentation_models_pytorch as smp
from new_dataset import WSITileDatasetBalanced

# trénovací data
wsi_paths_train = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_001.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_002.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_003.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_004.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_005.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_006.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_007.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_008.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_009.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_010.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_011.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_012.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_013.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_014.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_015.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_016.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_017.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_018.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_019.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_020.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_021.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_022.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_023.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_024.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_025.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_026.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_027.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_028.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_029.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_030.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_031.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_032.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_033.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_034.tif",
    # r"C:\Users\USER\Desktop\wsi_dir\tumor_035.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_036.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_037.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_038.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_039.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_040.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_041.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_042.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_043.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_044.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_045.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_046.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_047.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_048.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_049.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_050.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif",
]

tissue_mask_paths_train = [
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_001.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_002.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_003.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_004.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_005.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_006.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_007.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_008.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_009.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_010.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_011.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_012.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_013.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_014.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_015.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_016.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_017.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_018.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_019.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_020.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_021.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_022.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_023.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_024.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_025.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_026.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_027.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_028.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_029.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_030.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_031.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_032.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_033.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_034.npy",
    # r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_035.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_036.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_037.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_038.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_039.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_040.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_041.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_042.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_043.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_044.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_045.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_046.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_047.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_048.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_049.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_050.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_089.npy",
]

mask_paths_train = [
    r"C:\Users\USER\Desktop\wsi_dir\mask_001.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_002.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_003.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_004.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_005.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_006.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_007.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_008.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_009.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_010.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_011.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_012.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_013.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_014.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_015.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_016.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_017.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_018.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_019.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_020.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_021.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_022.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_023.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_024.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_025.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_026.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_027.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_028.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_029.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_030.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_031.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_032.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_033.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_034.tif",
    # r"C:\Users\USER\Desktop\wsi_dir\mask_035.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_036.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_037.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_038.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_039.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_040.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_041.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_042.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_043.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_044.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_045.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_046.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_047.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_048.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_049.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_050.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_089.tif",
]

gt_lowres_mask_paths_train = [
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_001_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_002_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_003_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_004_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_005_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_006_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_007_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_008_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_009_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_010_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_011_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_012_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_013_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_014_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_015_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_016_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_017_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_018_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_019_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_020_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_021_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_022_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_023_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_024_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_025_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_026_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_027_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_028_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_029_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_030_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_031_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_032_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_033_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_034_cancer.npy",
    # r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_035_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_036_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_037_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_038_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_039_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_040_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_041_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_042_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_043_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_044_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_045_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_046_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_047_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_048_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_049_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_050_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_089_cancer.npy",
]
# validační data
wsi_paths_val = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_051.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_052.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_053.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_054.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_055.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_056.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_057.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_058.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_059.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_060.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_061.tif",
]

tissue_mask_paths_val = [
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_051.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_052.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_053.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_054.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_055.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_056.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_057.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_058.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_059.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_060.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_061.npy",
]

mask_paths_val = [
    r"C:\Users\USER\Desktop\wsi_dir\mask_051.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_052.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_053.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_054.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_055.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_056.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_057.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_058.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_059.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_060.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_061.tif",
]

gt_lowres_mask_paths_val = [
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_051_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_052_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_053_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_054_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_055_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_056_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_057_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_058_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_059_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_060_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_061_cancer.npy",
]

# testovací data
wsi_paths_test = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_062.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_063.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_064.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_065.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_066.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_067.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_068.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_069.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_070.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_084.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_091.tif",
]

tissue_mask_paths_test = [
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_062.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_063.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_064.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_065.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_066.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_067.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_068.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_069.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_070.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_084.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_091.npy",
]

mask_paths_test = [
    r"C:\Users\USER\Desktop\wsi_dir\mask_062.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_063.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_064.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_065.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_066.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_067.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_068.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_069.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_070.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_084.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_091.tif",
]

gt_lowres_mask_paths_test = [
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_062_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_063_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_064_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_065_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_066_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_067_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_068_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_069_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_070_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_084_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_091_cancer.npy",
]

if __name__ == "__main__":

# Začátek trénovacího skriptu
    # color_jitter_params = {
    #     "brightness": 0.2,
    #     "contrast": 0.2,
    #     "saturation": 0.2,
    #     "hue": 0.1
    # }

    # augmentations = MyAugmentations(
    #     p_flip=0.5,
    #     color_jitter_params=color_jitter_params,
    #     mean=(0.485, 0.456, 0.406),
    #     std=(0.229, 0.224, 0.225)
    # )
    
    start = time.time()
    epochs = 51
    batch = 32
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
                                           mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
                                           tile_size=256, wanted_level=2, positive_sampling_prob=0.6,
                                           min_cancer_ratio_in_tile=0.05, augmentations=basic_transform)
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)

    val_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
                                         mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
                                         tile_size=256, wanted_level=2, positive_sampling_prob=0.6,
                                         min_cancer_ratio_in_tile=0.05, augmentations=basic_transform)
    
    validloader= DataLoader(val_dataset,batch_size=batch, num_workers=4, shuffle=True)

    test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
                                          mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
                                          tile_size=256, wanted_level=2, positive_sampling_prob=0.6,
                                          min_cancer_ratio_in_tile=0.05, augmentations=basic_transform)
    
    testloader = DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=True)

    # net = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainloader), epochs=epochs, max_lr=0.001, pct_start=0.3, div_factor=25, final_div_factor=1000)

    train_loss = []
    valid_loss = []
    train_iou = []
    valid_iou = []
    train_dice = []
    valid_dice = []

    batch_load_time = [] 

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs-1}")
        if epoch >=10  and epoch % 10 == 0:
            print(f"IoU: train: {np.mean(train_iou)}, val: {np.mean(valid_iou)}")
            print(f"Loss: train: {np.mean(train_loss)}, val: {np.mean(valid_loss)}")
            print(f"Dice: train: {np.mean(train_dice)}, val: {np.mean(valid_dice)}")
        
        iou_tmp = []
        loss_tmp = []
        dice_tmp = []

        start_time = time.time()
        for k,(data,lbl) in enumerate(trainloader):
            # mask_range_values = lbl.max(), lbl.min()
            # print(f"Mask range values: {mask_range_values}")
            # print(k)
            batch_time = time.time() - start_time
            batch_load_time.append(batch_time)

            data = data.to(device)
            lbl = lbl.to(device)

            net.train()
            output = net(data)
            output = torch.sigmoid(output)

            loss = dice_bce_loss(output, lbl)
            dice = dice_coefficient(output, lbl)
            iou = calculate_iou(output=output, lbl=lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # scheduler.step()
            
            iou_tmp.append(iou)
            loss_tmp.append(loss.cpu().detach().numpy())
            dice_tmp.append(dice.cpu().detach().numpy())

            start_time = time.time()

        train_loss.append(np.mean(loss_tmp))
        train_iou.append(np.mean(iou_tmp))
        train_dice.append(np.mean(dice_tmp))


        iou_tmp = []
        loss_tmp = []
        dice_tmp = []

        # val loop
        for kk,(data, lbl) in enumerate(validloader):
            with torch.inference_mode():

                data = data.to(device)
                lbl = lbl.to(device)

                net.eval()
                output = net(data)

                output = torch.sigmoid(output)
                loss = dice_bce_loss(output, lbl)
                dice = dice_coefficient(output, lbl)

                # Výpočet metrik
                iou = calculate_iou(output=output, lbl=lbl)

                iou_tmp.append(iou)
                loss_tmp.append(loss.cpu().detach().numpy())
                dice_tmp.append(dice.cpu().detach().numpy())

        valid_loss.append(np.mean(loss_tmp))
        valid_iou.append(np.mean(iou_tmp))
        valid_dice.append(np.mean(dice_tmp))
        
        scheduler.step()


    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\Users\USER\Desktop\unet_smp_e50_11200len.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    test_dice_scores = []
    test_iou_scores = []

    # Test loop
    for kk, (data, lbl) in enumerate(testloader):
        # if kk > 3:
        #     break
        with torch.inference_mode():
            data = data.to(device)
            lbl = lbl.to(device)

            # Předpověď sítě
            net.eval()
            output = net(data)
            output = torch.sigmoid(output)

            # Výpočet metrik
            dice_score = dice_coefficient(output, lbl)  # Dice koeficient
            iou = calculate_iou(lbl=lbl, output=output)

            test_dice_scores.append(dice_score.cpu().item())
            test_iou_scores.append(iou)

            # Vizualizace
            if kk < 3:
                plt.figure(figsize=(15, 10))
                plt.subplot(131)
                plt.title("Input")
                # plt.imshow(data[0, 1, :, :].detach().cpu().numpy())
                plt.imshow(data[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.subplot(132)
                plt.title("Prediction")
                plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
                plt.subplot(133)
                plt.title("Ground Truth")
                plt.imshow(lbl[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
                plt.show(block=False)
                plt.pause(5)
                plt.close()
    # Výpočet průměrných metrik na testovacích datech
    average_test_dice = np.mean(test_dice_scores)
    average_test_iou = np.mean(test_iou_scores)

    print(f"Average Dice coefficient on test data: {average_test_dice:.4f}")
    print(f"Average IoU on test data: {average_test_iou:.4f}")
    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s")
    print("Proběhl celý skript.")

    model_save_path = r"C:\Users\USER\Desktop\weights\unet_smp_e50_11200len.pth"

    # Uložení váh modelu
    torch.save(net.state_dict(), model_save_path)
    print(f"Model byl uložen do {model_save_path}")
    end = time.time()
    print(f"Trénování trvalo {end - start:.2f} s")
