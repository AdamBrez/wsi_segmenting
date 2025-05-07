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
from new_dataset import WSITileDatasetBalanced
from my_augmentation import MyAugmentations, AlbumentationsAug
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
from segmentation_models_pytorch.losses import DiceLoss
from my_model import Unet2D
import datetime


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
    r"C:\Users\USER\Desktop\wsi_dir\tumor_035.tif",
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
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_035.npy",
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
    r"C:\Users\USER\Desktop\wsi_dir\mask_035.tif",
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
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_035_cancer.npy",
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
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.15,
        "hue": 0.05
    }

    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        p_color=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    alubmentations_aug = AlbumentationsAug()
    
    start = time.time()
    epochs = 61
    batch = 32
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
                                           mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
                                           tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                           min_cancer_ratio_in_tile=0.05, augmentations=augmentations,
                                           dataset_len=11200)
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
                                         mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
                                         tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                         min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                         dataset_len=5600)
    
    validloader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=False, pin_memory=True)

    test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
                                          mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
                                          tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                          min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                          dataset_len=22400)
    
    testloader = DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=False)

    # net = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = smp.Unet(encoder_name=None, encoder_weights=None, in_channels=3, classes=1, activation=None)
    # net = Unet2D(in_size=3, out_size=1)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.01) #weight_decay=1e-5
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=7, threshold=0.001)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainloader), epochs=epochs, max_lr=0.001, pct_start=0.3, div_factor=25, final_div_factor=1000)
    # early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    dice_loss = DiceLoss(mode="binary", from_logits=True, smooth=1e-6)

    best_val_dice = float("-inf")
    weights_patience = 7
    min_impovement = 0.001

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
    lr_change_epochs = []

    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs-1}\n")
        if epoch >= 10 and epoch % 10 == 0:
            print(f"F1: train: {np.mean(smp_f1_train)}, val: {np.mean(smp_f1_val)}")
            print(f"Recall: train: {np.mean(smp_recall_train)}, val: {np.mean(smp_recall_val)}")
            print(f"Precision: train: {np.mean(smp_precision_train)}, val: {np.mean(smp_precision_val)}")
            print(f"IoU (smp): train: {np.mean(smp_iou_train)}, val: {np.mean(smp_iou_val)}")
            print(f"Loss (gemini): train: {np.mean(train_loss_hist)}, val: {np.mean(valid_loss_hist)}")
            print("-"*30)

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

        print(f"Epoch {epoch}- Train stats- Precision: {epoch_train_precision:.4f}, Recall: {epoch_train_recall:.4f}, F1: {epoch_train_dice:.4f}, IoU (smp): {epoch_train_iou:.4f}, Loss (gemini): {avg_train_loss:.4f}")
        print("-"*30)

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

                epoch_val_loss += loss.item()

                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5)
                epoch_val_fp += fp.sum()
                epoch_val_fn += fn.sum()
                epoch_val_tp += tp.sum()
                epoch_val_tn += tn.sum()

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
        
        print(f"Epoch {epoch}- Val stats- Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}, F1: {epoch_val_dice:.4f}, IoU (smp): {epoch_val_iou:.4f}, Loss (gemini): {avg_valid_loss:.4f}")
        print("-"*30)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        print("-"*30)
        if epoch > 10 and epoch < epochs - 10:
            if epoch_val_dice > (best_val_dice + min_impovement):
                best_val_dice = epoch_val_dice
                print(f"Uložení váh modelu z epochy {epoch} s F1: {epoch_val_dice:.4f}")
                print("-"*30)
                torch.save(net.state_dict(), r"C:\Users\USER\Desktop\weights\best_weights.pth")
                weights_patience = 0
            else:
                weights_patience += 1


        scheduler.step(epoch_val_dice)  # pro ReduceLROnPlateau
        if current_lr > optimizer.param_groups[0]['lr']:
            print(f"Změna LR: {optimizer.param_groups[0]['lr']} -> {current_lr:.6f} | Epoch: {epoch}")
            lr_change_epochs.append(epoch)
        # scheduler.step()  <-- pro multi step LR



    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss_hist, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\Users\USER\Desktop\loss_plot.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    test_dice_scores = []
    test_iou_scores = []
    test_recall_scores = []
    test_precision_scores = []
    epoch_test_tp, epoch_test_fp, epoch_test_fn, epoch_test_tn = 0, 0, 0, 0
    # Test loop
    with torch.inference_mode():
        for kk, (data, lbl) in enumerate(testloader):
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

    print(len(test_dice_scores), len(test_iou_scores), len(test_recall_scores), len(test_precision_scores))

    end = time.time()

    model_save_path = r"C:\Users\USER\Desktop\weights\final_weights.pth"
    
    model_and_metadata = {
        "model": type(net).__name__,
        "encoder": "resnet34",
        "epochs": epochs,
        "lr_start": 0.001, 
        "lr_end": optimizer.param_groups[0]['lr'],
        scheduler: type(scheduler).__name__,
        "optimizer": type(optimizer).__name__,
        "batch_size": batch,
        "train_dataset_len": vars(train_dataset)["dataset_len"],
        "train_dice": smp_f1_train,
        "train_iou": smp_iou_train,
        "train_precision": smp_precision_train,
        "train_recall": smp_recall_train,
        "val_dice": smp_f1_val,
        "val_iou": smp_iou_val,
        "val_precision": smp_precision_val,
        "val_recall": smp_recall_val,
        "test_dice": epoch_test_dice,
        "test_iou": epoch_test_iou,
        "test_precision": epoch_test_precision,
        "test_recall": epoch_test_recall,
        "loss_function": type(dice_loss).__name__,
        "augmentation": type(augmentations).__name__,
        "wanted_level": vars(test_dataset)["wanted_level"],
        "model_state_dict": net.state_dict(),
        "train_loss_hist": train_loss_hist,
        "valid_loss_hist": valid_loss_hist,
        "runtime": end - start
    }
    # Uložení váh modelu
    torch.save(model_and_metadata, model_save_path)
    print(f"Model byl uložen do {model_save_path}")

    print(f"Trénování trvalo {end - start:.2f} s")
