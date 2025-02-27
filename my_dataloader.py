import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

from torch.utils.data import Dataset
from random import randint
import numpy as np
from openslide import OpenSlide
from torchvision.transforms import functional as TF

class WSITileDataset(Dataset):
    def __init__(self, wsi_paths, tissue_mask_paths, mask_paths, tile_size=256, augmentations=None, min_foreground_ratio=0.05):
        self.wsi_paths = wsi_paths
        self.tissue_mask_paths = tissue_mask_paths  # Cesty k maskám tkáně (.npy)
        self.mask_paths = mask_paths
        self.tile_size = tile_size
        self.augmentations = augmentations
        self.min_foreground_ratio = min_foreground_ratio  # Minimální podíl tkáně

        # Načteme masky tkáně
        self.tissue_masks = [np.load(path) for path in self.tissue_mask_paths]

    def __len__(self):
        return 10000  # Počet dlaždic na epochu

    def __getitem__(self, idx):
        while True:
            # Náhodný výběr WSI a jeho masky tkáně
            wsi_idx = randint(0, len(self.wsi_paths) - 1)
            # print(self.wsi_paths[wsi_idx], self.mask_paths[wsi_idx], self.tissue_mask_paths[wsi_idx])
            wsi = OpenSlide(self.wsi_paths[wsi_idx])
            mask = OpenSlide(self.mask_paths[wsi_idx])
            tissue_mask = self.tissue_masks[wsi_idx]

            # Rozměry WSI a masky tkáně
            wsi_width, wsi_height = wsi.dimensions
            tissue_mask_height, tissue_mask_width = tissue_mask.shape

            # Přepočítání měřítka mezi WSI a maskou tkáně
            scale_x = wsi_width / tissue_mask_width
            scale_y = wsi_height / tissue_mask_height

            # Náhodné souřadnice v masce tkáně
            tissue_y, tissue_x = np.where(tissue_mask > 0)  # Najdeme oblasti tkáně (1 = tkáň)
            rand_idx = randint(0, len(tissue_x) - 1)

            # Převedeme souřadnice zpět do originálního rozlišení WSI
            x = int(tissue_x[rand_idx] * scale_x)
            y = int(tissue_y[rand_idx] * scale_y)

            # Ověříme, že dlaždice je v rámci WSI
            if x + self.tile_size <= wsi_width and y + self.tile_size <= wsi_height:
                # Načtení dlaždice a odpovídající masky
                tile = wsi.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert("RGB")
                mask_tile = mask.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert("L")

                # Augmentace
                if self.augmentations:
                    tile, mask_tile = self.augmentations(tile, mask_tile)
                else:
                    tile = TF.to_tensor(tile)
                    mask_tile = TF.to_tensor(mask_tile)

                return tile, mask_tile