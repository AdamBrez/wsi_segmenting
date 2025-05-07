import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from my_functions import basic_transform
import segmentation_models_pytorch as smp
from multi_res import WSITileDatasetBalanced
from my_augmentation import MyAugmentations
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
from segmentation_models_pytorch.losses import DiceLoss



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
    r"C:\Users\USER\Desktop\wsi_dir\tumor_071.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_072.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_073.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_074.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_075.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_076.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_077.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_078.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_079.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_080.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_081.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_082.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_083.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_085.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_086.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_087.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_088.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_090.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_092.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_093.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_094.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_095.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_096.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_097.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_098.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_099.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_100.tif",
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
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_071.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_072.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_073.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_074.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_075.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_076.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_077.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_078.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_079.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_080.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_081.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_082.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_083.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_085.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_086.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_087.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_088.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_089.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_090.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_092.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_093.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_094.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_095.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_096.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_097.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_098.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_099.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_100.npy",
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
    r"C:\Users\USER\Desktop\wsi_dir\mask_071.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_072.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_073.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_074.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_075.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_076.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_077.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_078.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_079.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_080.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_081.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_082.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_083.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_085.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_086.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_087.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_088.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_089.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_090.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_092.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_093.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_094.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_095.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_096.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_097.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_098.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_099.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_100.tif",
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
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_071_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_072_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_073_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_074_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_075_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_076_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_077_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_078_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_079_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_080_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_081_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_082_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_083_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_085_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_086_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_087_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_088_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_089_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_090_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_092_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_093_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_094_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_095_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_096_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_097_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_098_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_099_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_100_cancer.npy",
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
    color_jitter_params = {
        "brightness": 0.25,
        "contrast": 0.25,
        "saturation": 0.20,
        "hue": 0.1
    }

    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        p_color=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    start = time.time()
    epochs = 51
    batch = 32
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
                                           mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
                                           tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                           min_cancer_ratio_in_tile=0.05, augmentations=augmentations,
                                           dataset_len=22400, context_level=3, context_size=256)
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
                                         mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
                                         tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                         min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                         dataset_len=5600, context_level=3, context_size=256)
    
    validloader= DataLoader(val_dataset,batch_size=batch, num_workers=4, shuffle=False, pin_memory=True)

    test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
                                          mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
                                          tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                          min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                          dataset_len=5600, context_level=3, context_size=256)
    
    testloader = DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=False)

    # net = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=6, classes=1, activation=None)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45], gamma=0.1)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainloader), epochs=epochs, max_lr=0.001, pct_start=0.3, div_factor=25, final_div_factor=1000)
    # early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    dice_loss = DiceLoss(mode="binary", from_logits=True, smooth=1e-6)
    train_loss = []
    valid_loss = []

    train_iou = []
    valid_iou = []

    train_dice = []
    valid_dice = []

    smp_f1_train = []
    smp_f1_val = []

    smp_iou_train = []
    smp_iou_val = []

    smp_recall_train = []
    smp_recall_val = []
    
    smp_precision_train = []
    smp_precision_val = []

    train_loss_hist = []
    valid_loss_hist = []

    epoch_train_tp, epoch_train_fp, epoch_train_fn, epoch_train_tn = 0, 0, 0, 0

    batch_load_time = [] 

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs-1}")
        if epoch >= 10 and epoch % 10 == 0:
            # print(f"IoU: train: {np.mean(train_iou)}, val: {np.mean(valid_iou)}")
            # print(f"Loss: train: {np.mean(train_loss)}, val: {np.mean(valid_loss)}")
            # print(f"Dice: train: {np.mean(train_dice)}, val: {np.mean(valid_dice)}")
            print(f"F1: train: {np.mean(smp_f1_train)}, val: {np.mean(smp_f1_val)}")
            print(f"Recall: train: {np.mean(smp_recall_train)}, val: {np.mean(smp_recall_val)}")
            print(f"Precision: train: {np.mean(smp_precision_train)}, val: {np.mean(smp_precision_val)}")
            print(f"IoU (smp): train: {np.mean(smp_iou_train)}, val: {np.mean(smp_iou_val)}")
            print(f"Loss (gemini): train: {np.mean(train_loss_hist)}, val: {np.mean(valid_loss_hist)}")
        
        # iou_tmp = []
        # loss_tmp = []
        # dice_tmp = []

        epoch_train_loss = 0.0

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
            loss = dice_loss(output, lbl)
            output = torch.sigmoid(output)

            # loss = dice_bce_loss(output, lbl)
            # dice = dice_coefficient(output, lbl)
            # iou = calculate_iou(output=output, lbl=lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            with torch.inference_mode():
                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5) 
                epoch_train_fp += fp.sum()
                epoch_train_fn += fn.sum()
                epoch_train_tp += tp.sum()
                epoch_train_tn += tn.sum()

            # scheduler.step()
            
            # iou_tmp.append(iou)
            # loss_tmp.append(loss.cpu().detach().numpy())
            # dice_tmp.append(dice.cpu().detach().numpy())

            start_time = time.time()

        avg_train_loss = epoch_train_loss / len(trainloader)
        epoch_train_precision = precision(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_recall = recall(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_dice = f1_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_iou = iou_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()

        train_loss_hist.append(avg_train_loss)
        smp_precision_train.append(epoch_train_precision)
        smp_recall_train.append(epoch_train_recall)
        smp_f1_train.append(epoch_train_dice)
        smp_iou_train.append(epoch_train_iou)

        # train_loss.append(np.mean(loss_tmp))
        # train_iou.append(np.mean(iou_tmp))
        # train_dice.append(np.mean(dice_tmp))
        print(f"Epoch {epoch}- Precision: {epoch_train_precision:.4f}, Recall: {epoch_train_recall:.4f}, F1: {epoch_train_dice:.4f}, IoU (smp): {epoch_train_iou:.4f}, Loss (gemini): {avg_train_loss:.4f}")
        print("-"*30)
        # iou_tmp = []
        # loss_tmp = []
        # dice_tmp = []

        # val loop
        net.eval()
        epoch_val_tp, epoch_val_fp, epoch_val_fn, epoch_val_tn = 0, 0, 0, 0
        epoch_val_loss = 0.0
        with torch.inference_mode():
            for kk,(data, lbl) in enumerate(validloader):

                data = data.to(device)
                lbl = lbl.to(device)

                output = net(data)
                loss = dice_loss(output, lbl)
                output = torch.sigmoid(output)

                # loss = dice_bce_loss(output, lbl)
                # dice = dice_coefficient(output, lbl)
                # iou = calculate_iou(output=output, lbl=lbl)

                epoch_val_loss += loss.item()

                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5)
                epoch_val_fp += fp.sum()
                epoch_val_fn += fn.sum()
                epoch_val_tp += tp.sum()
                epoch_val_tn += tn.sum()

                # iou_tmp.append(iou)
                # loss_tmp.append(loss.cpu().detach().numpy())
                # dice_tmp.append(dice.cpu().detach().numpy())

        avg_valid_loss = epoch_val_loss / len(validloader)
        epoch_val_precision = precision(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_recall = recall(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_dice = f1_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_iou = iou_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()

        valid_loss_hist.append(avg_valid_loss)
        smp_precision_val.append(epoch_val_precision)
        smp_recall_val.append(epoch_val_recall)
        smp_f1_val.append(epoch_val_dice)
        smp_iou_val.append(epoch_val_iou)
        # valid_loss.append(np.mean(loss_tmp))
        # valid_iou.append(np.mean(iou_tmp))
        # valid_dice.append(np.mean(dice_tmp))
        
        print(f"Epoch {epoch}- Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}, F1: {epoch_val_dice:.4f}, IoU (smp): {epoch_val_iou:.4f}, Loss (gemini): {avg_valid_loss:.4f}")
        print("-"*30)

        scheduler.step()


    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\Users\USER\Desktop\multi_res_dice_loss_e50_11200len.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    model_save_path = r"C:\Users\USER\Desktop\weights\multi_res_dice_loss_e50_11200len.pth"

    # Uložení váh modelu
    torch.save(net.state_dict(), model_save_path)
    print(f"Model byl uložen do {model_save_path}")

    test_dice_scores = []
    test_iou_scores = []
    test_recall_scores = []
    test_precision_scores = []
    epoch_test_tp, epoch_test_fp, epoch_test_fn, epoch_test_tn = 0, 0, 0, 0
    # Test loop
    with torch.inference_mode():
        for kk, (data, lbl) in enumerate(testloader):
        # if kk > 3:
        #     break
            data = data.to(device)
            lbl = lbl.to(device)

            # Předpověď sítě
            net.eval()
            output = net(data)
            output = torch.sigmoid(output)

            lbl_long = lbl.long()
            tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5)
            epoch_test_fp += fp.sum()
            epoch_test_fn += fn.sum()
            epoch_test_tp += tp.sum()
            epoch_test_tn += tn.sum()

            # Výpočet metrik
            # dice_score = dice_coefficient(output, lbl)  # Dice koeficient
            # iou = calculate_iou(lbl=lbl, output=output)

            # test_dice_scores.append(dice_score.cpu().item())
            # test_iou_scores.append(iou)

            # Vizualizace
            # if kk < 3:
            #     plt.figure(figsize=(15, 10))
            #     plt.subplot(131)
            #     plt.title("Input")
            #     # plt.imshow(data[0, 1, :, :].detach().cpu().numpy())
            #     plt.imshow(data[0].permute(1, 2, 0).detach().cpu().numpy())
            #     plt.subplot(132)
            #     plt.title("Prediction")
            #     plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
            #     plt.subplot(133)
            #     plt.title("Ground Truth")
            #     plt.imshow(lbl[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
            #     plt.show(block=False)
            #     plt.pause(5)
            #     plt.close()
    # Výpočet průměrných metrik na testovacích datech
    # average_test_dice = np.mean(test_dice_scores)
    # average_test_iou = np.mean(test_iou_scores)
    epoch_test_precision = precision(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_recall = recall(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_dice = f1_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_iou = iou_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()

    test_dice_scores.append(epoch_test_dice)
    test_iou_scores.append(epoch_test_iou)
    test_recall_scores.append(epoch_test_recall)
    test_precision_scores.append(epoch_test_precision)

    print(f"Average Dice coefficient on test data: {np.mean(test_dice_scores):.4f}")
    print(f"Average IoU on test data: {np.mean(test_iou_scores):.4f}")
    print(f"Average recall on test data: {np.mean(test_recall_scores):.4f}")
    print(f"Average precision on test data: {np.mean(test_precision_scores):.4f}")
    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s")
    print("Proběhl celý skript.")

    end = time.time()
    print(f"Trénování trvalo {end - start:.2f} s")
