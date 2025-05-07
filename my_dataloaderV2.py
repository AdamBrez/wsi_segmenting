import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

from PIL import Image
from torch.utils.data import Dataset
from random import randint, random
import numpy as np
from openslide import OpenSlide
from torchvision.transforms import functional as TF

"""
    V tomto skriptu mám vytvořený custom dataset, který používám pro načítání dlaždic do sítě.
    Vytváří bounding box okolo tkáně a poté náhodně vybírá dlaždice uvnitř této oblasti.
    Tím může dojít k vybrání dlaždice z ajkéhokoliv místa v rámci tkáně.
    Při aplikaci u učení došlo k tomu, že se model naučil segmentovat zdravou tkáň místo nádorové tkáně.
    Také trénink trval dost dlouho, průměrný batch se načítal 2.4s.
"""

class WSITileDataset(Dataset):
    def __init__(self, wsi_paths, tissue_mask_paths, mask_paths, lowres_gt_masks, tile_size=256, augmentations=None, min_foreground_ratio=0.05):
        self.wsi_paths = wsi_paths
        self.tissue_mask_paths = tissue_mask_paths  # Cesty k maskám tkáně (.npy)
        self.mask_paths = mask_paths
        self.lowres_gt_masks = lowres_gt_masks
        self.tile_size = tile_size
        self.augmentations = augmentations
        self.min_foreground_ratio = min_foreground_ratio  # Minimální podíl tkáně


        # Načteme masky tkáně
        # self.tissue_masks = [np.load(path) for path in self.tissue_mask_paths]

    def __len__(self):
        return 9920  # Počet dlaždic na epochu

    def __getitem__(self, idx):
        while True:
            choice = random()
            # Náhodný výběr WSI a jeho masky tkáně
            wsi_idx = randint(0, len(self.wsi_paths) - 1)
            wanted_level = 2
            
            wsi = OpenSlide(self.wsi_paths[wsi_idx])
            mask = OpenSlide(self.mask_paths[wsi_idx])
            if choice < 0.4:
                tissue_mask = np.load(self.tissue_mask_paths[wsi_idx])
            else:
                # mask_np = np.array(mask.read_region(location=(0, 0), level=6, size=mask.level_dimensions[6]).convert("L"))
                # tissue_mask = (mask_np > 128).astype(np.uint8) * 255  # binarizace masky
                tissue_mask = np.load(self.lowres_gt_masks[wsi_idx])
            # Rozměry WSI a masky tkáně
            wsi_width, wsi_height = wsi.level_dimensions[wanted_level]
            native_width, native_height = wsi.level_dimensions[0]
            tissue_mask_height, tissue_mask_width = tissue_mask.shape
            
            # Přepočítání měřítka mezi WSI a maskou tkáně
            scale_x = wsi_width / tissue_mask_width
            scale_y = wsi_height / tissue_mask_height
            native_to_wanted_x = native_width / wsi_width
            native_to_wanted_y = native_height / wsi_height
            
            # Místo výběru diskrétního pixelu vybereme celou oblast tkáně
            tissue_y, tissue_x = np.where(tissue_mask > 0)
            
            # Vytvoříme obálku (bounding box) tkáně
            min_x, max_x = np.min(tissue_x), np.max(tissue_x)
            min_y, max_y = np.min(tissue_y), np.max(tissue_y)
            
            # Přepočítáme na úroveň WSI
            min_x_wsi = int(min_x * scale_x)
            max_x_wsi = int(max_x * scale_x)
            min_y_wsi = int(min_y * scale_y)
            max_y_wsi = int(max_y * scale_y)
            
            # Nyní vybereme náhodnou pozici uvnitř této oblasti s libovolným pixelovým posunem
            width_range = max_x_wsi - min_x_wsi - self.tile_size
            height_range = max_y_wsi - min_y_wsi - self.tile_size
            
            if width_range <= 0 or height_range <= 0:
                continue  # WSI je příliš malé
            
            # Náhodné pozice v rámci celé oblasti tkáně
            x_offset = randint(0, width_range)
            y_offset = randint(0, height_range)
            
            x = min_x_wsi + x_offset
            y = min_y_wsi + y_offset
            
            # Ověříme, že jsme v masce tkáně
            tile_mask_x = int(x / scale_x)
            tile_mask_y = int(y / scale_y)
            
            # Kontrola, zda jsme stále v oblasti tkáně (volitelné, ale užitečné)
            is_tissue = False
            for check_y in range(tile_mask_y, min(tile_mask_y + int(self.tile_size / scale_y), tissue_mask_height)):
                for check_x in range(tile_mask_x, min(tile_mask_x + int(self.tile_size / scale_x), tissue_mask_width)):
                    if tissue_mask[check_y, check_x] > 0:
                        is_tissue = True
                        break
                if is_tissue:
                    break
            
            if not is_tissue:
                continue  # zkusíme jiný výběr
            
            # Ověříme, že dlaždice je v rámci WSI
            if x + self.tile_size <= wsi_width and y + self.tile_size <= wsi_height:
                # Načtení dlaždice a odpovídající masky
                print(f"Pro wsi: {self.wsi_paths[wsi_idx].split("\\")[-1]}: x: {x*int(native_to_wanted_x)}, y: {y*int(native_to_wanted_y)}")
                tile = wsi.read_region((x*int(native_to_wanted_x), y*int(native_to_wanted_y)), wanted_level, (self.tile_size, self.tile_size)).convert("RGB")
                mask_tile = mask.read_region((x*int(native_to_wanted_x), y*int(native_to_wanted_y)), wanted_level, (self.tile_size, self.tile_size)).convert("L")
                mask_tile = np.array(mask_tile) > 128
                mask_tile = Image.fromarray(mask_tile.astype(np.uint8) * 255, mode="L")
                #musím to předělávat na PIL image????
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
    r"C:\Users\USER\Desktop\wsi_dir\tumor_001.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_002.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_003.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_017.tif",
    ]

    mask_paths_train = [
    r"C:\Users\USER\Desktop\wsi_dir\mask_001.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_002.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_003.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_089.tif",
    r"C:\Users\USER\Desktop\wsi_dir\mask_017.tif",
    ]

    tissue_mask_paths_train = [
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_001.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_002.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_003.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_089.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_017.npy",
    ]

    lowres_gt_masks_train = [
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_001_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_002_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_003_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_089_cancer.npy",
    r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_017_cancer.npy",
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

    dataset = WSITileDataset(wsi_paths_train, tissue_mask_paths_train, mask_paths_train, lowres_gt_masks_train, augmentations=None)
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
        
