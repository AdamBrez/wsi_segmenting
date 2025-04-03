import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

from PIL import Image
from torch.utils.data import Dataset
from random import randint
import numpy as np
from openslide import OpenSlide
from torchvision.transforms import functional as TF
import time
"""
    V tomto skriptu mám vytvořený custom dataset, který používám pro načítání dlaždic do sítě.
"""

class WSITileDataset(Dataset):
    def __init__(self, wsi_paths, tissue_mask_paths, mask_paths, tile_size=256, augmentations=None, min_foreground_ratio=0.05):
        self.wsi_paths = wsi_paths
        self.tissue_mask_paths = tissue_mask_paths  # Cesty k maskám tkáně (.npy)
        self.mask_paths = mask_paths
        self.tile_size = tile_size
        self.augmentations = augmentations
        self.min_foreground_ratio = min_foreground_ratio  # Minimální podíl tkáně

        # Načteme masky tkáně
        # self.tissue_masks = [np.load(path) for path in self.tissue_mask_paths]

    def __len__(self):
        return 320  # Počet dlaždic na epochu

    def __getitem__(self, idx):
        while True:
            # Náhodný výběr WSI a jeho masky tkáně
            wsi_idx = randint(0, len(self.wsi_paths) - 1)
            wanted_level = 2
            # print(self.wsi_paths[wsi_idx], self.mask_paths[wsi_idx], self.tissue_mask_paths[wsi_idx])
            wsi = OpenSlide(self.wsi_paths[wsi_idx])
            mask = OpenSlide(self.mask_paths[wsi_idx])
            tissue_mask = self.tissue_mask_paths[wsi_idx]
            tissue_mask = np.load(tissue_mask)
            print(f"Načítám dlaždici z {self.wsi_paths[wsi_idx].split("\\")[-1]} a masku z {self.mask_paths[wsi_idx].split("\\")[-1]} s pomocí {self.tissue_mask_paths[wsi_idx].split("\\")[-1]}")

            # Rozměry WSI a masky tkáně
            wsi_width, wsi_height = wsi.level_dimensions[wanted_level]  #<-- změna dimenze (rozlišení) 
            native_width, native_height = wsi.level_dimensions[0]
            tissue_mask_height, tissue_mask_width = tissue_mask.shape
            # Přepočítání měřítka mezi WSI a maskou tkáně
            scale_x = wsi_width / tissue_mask_width
            scale_y = wsi_height / tissue_mask_height
            native_to_wanted_x = native_width / wsi_width
            native_to_wanted_y = native_height / wsi_height

            # Náhodné souřadnice v masce tkáně
            tissue_y, tissue_x = np.where(tissue_mask > 0)  # Vrací indexy z numpy pole kde je hodnota > 0 (tkáň)
            rand_idx = randint(0, len(tissue_x) - 1)
            # Převedeme souřadnice zpět do originálního rozlišení WSI
            x = int(tissue_x[rand_idx] * scale_x)
            y = int(tissue_y[rand_idx] * scale_y)

            # Ověříme, že dlaždice je v rámci WSI
            if x + self.tile_size <= wsi_width and y + self.tile_size <= wsi_height:
                # Načtení dlaždice a odpovídající masky
                tile = wsi.read_region((x*int(native_to_wanted_x), y*int(native_to_wanted_y)), wanted_level, (self.tile_size, self.tile_size)).convert("RGB")
                mask_tile = mask.read_region((x*int(native_to_wanted_x), y*int(native_to_wanted_y)), wanted_level, (self.tile_size, self.tile_size)).convert("L")
                mask_tile = np.array(mask_tile) > 128
                mask_tile = Image.fromarray(mask_tile.astype(np.uint8) * 255, mode="L")  # mode "L" nechá masku jako 8bitvou grayscale není binární

                # Augmentace
                if self.augmentations:
                    tile, mask_tile = self.augmentations(tile, mask_tile)
                else:
                    tile = TF.to_tensor(tile)
                    mask_tile = TF.to_tensor(mask_tile)

                return tile, mask_tile
            
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    from my_augmentation import MyAugmentations

    wsi_paths_train = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_001.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_002.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_003.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_089.tif",
    ]

    mask_paths_train = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_001.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_002.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_003.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_089.tif"
    ]

    tissue_mask_paths_train = [
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_001.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_002.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_003.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_089.npy",
    ]
    color_jitter_params = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }

    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

    dataset = WSITileDataset(wsi_paths_train, tissue_mask_paths_train, mask_paths_train, augmentations=None)
    trainloader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(trainloader))
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        img = images[i].permute(1, 2, 0).numpy()
        # img  = (img + 1) /2.0
        mask = labels[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[1, i].imshow(mask, cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.show()
        
