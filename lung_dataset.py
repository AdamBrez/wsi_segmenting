import os
import numpy as np

os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide
from openslide.deepzoom import DeepZoomGenerator
from matplotlib import pyplot as plt
import torch
mask = r"F:\wsi_dir_test\mask_003.tif"
slide = r"F:\wsi_dir_test\test_003.tif"
mask = openslide.OpenSlide(mask)
slide = openslide.OpenSlide(slide)
print(mask.level_dimensions)
print(slide.level_dimensions)
mask = DeepZoomGenerator(mask, tile_size=256, overlap=0, limit_bounds=True)
slide = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=True)

print(mask.level_count)
# Kontrola, zda mají maska a WSI stejný počet úrovní
if len(slide.level_dimensions) != len(mask.level_dimensions):
    print("Maska a WSI mají jiný počet úrovní.")
wsi_dir = r"F:\wsi_dir_test"

# for file in os.listdir(wsi_dir):
#     if file.startswith("test_"):
#         mask = file.replace("test_", "mask_")
#         print(f"Zpracovávám soubor: {file} a masku: {mask}")
#         mask_path = os.path.join(wsi_dir, mask)
#         wsi_path = os.path.join(wsi_dir, file)
#         slide = openslide.OpenSlide(wsi_path)
#         mask_slide = openslide.OpenSlide(mask_path)
#         dz = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=True)
#         dz_mask = DeepZoomGenerator(mask_slide, tile_size=256, overlap=0, limit_bounds=True)
#         if len(dz.level_dimensions) != len(dz_mask.level_dimensions):
#             print(f"Soubor {file} a maska {mask} mají jiný počet úrovní.")
