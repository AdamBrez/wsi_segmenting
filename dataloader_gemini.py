import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

from PIL import Image
from torch.utils.data import Dataset
from random import randint
import numpy as np
from openslide import OpenSlide
from torchvision.transforms import functional as TF
import random
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
        return 19840  # Počet dlaždic na epochu

    def __getitem__(self, idx):
        while True:
            # Náhodný výběr WSI a jeho masky tkáně
            wsi_idx = random.randint(0, len(self.wsi_paths) - 1) # Použít random.randint
            wanted_level = 2

            try: # Přidat try-except pro robustnější načítání
                wsi = OpenSlide(self.wsi_paths[wsi_idx])
                mask = OpenSlide(self.mask_paths[wsi_idx])
                tissue_mask_path = self.tissue_mask_paths[wsi_idx]
                tissue_mask = np.load(tissue_mask_path)

                # Najít souřadnice tkáně v nízkorozlišovací masce
                tissue_y_low, tissue_x_low = np.where(tissue_mask > 0)
                if len(tissue_x_low) == 0:
                    # print(f"Varování: V masce tkáně {tissue_mask_path} nebyla nalezena žádná tkáň > 0. Zkouším jiný WSI.")
                    continue # Pokud v masce nic není, zkusit jiný WSI

                # Náhodný výběr indexu pixelu tkáně z nízkorozlišovací masky
                rand_idx = random.randint(0, len(tissue_x_low) - 1)
                tx = tissue_x_low[rand_idx]
                ty = tissue_y_low[rand_idx]

                # Rozměry WSI na cílovém levelu a nativním levelu
                wsi_width, wsi_height = wsi.level_dimensions[wanted_level]
                native_width, native_height = wsi.level_dimensions[0]
                tissue_mask_height, tissue_mask_width = tissue_mask.shape

                # Přepočítání měřítka
                # scale_x/y: Kolik pixelů na wanted_level odpovídá jednomu pixelu na tissue_mask levelu
                if tissue_mask_width <= 0 or tissue_mask_height <= 0:
                    print(f"Varování: Neplatné rozměry tissue mask {tissue_mask_path}: {tissue_mask.shape}. Přeskakuji.")
                    continue
                scale_x = wsi_width / tissue_mask_width
                scale_y = wsi_height / tissue_mask_height
                # native_to_wanted_x/y: Faktor pro převod souřadnic z wanted_level na level 0
                if wsi_width <= 0 or wsi_height <= 0:
                     print(f"Varování: Neplatné rozměry WSI na levelu {wanted_level} pro {self.wsi_paths[wsi_idx]}. Přeskakuji.")
                     continue
                native_to_wanted_x = native_width / wsi_width
                native_to_wanted_y = native_height / wsi_height


                # ----- ZAČÁTEK ÚPRAVY -----

                # 1. Vypočítat oblast ve wanted_level odpovídající (tx, ty)
                x_start_high = tx * scale_x
                y_start_high = ty * scale_y
                # Exkluzivní konec - oblast [start, end)
                x_end_high = (tx + 1) * scale_x
                y_end_high = (ty + 1) * scale_y

                # 2. Vybrat náhodný *střed* dlaždice v této oblasti
                # Zajistíme, aby x_end > x_start (podobně pro y) pro uniform
                if x_end_high <= x_start_high: x_end_high = x_start_high + 1e-6 # Přidat malou hodnotu
                if y_end_high <= y_start_high: y_end_high = y_start_high + 1e-6

                center_x = random.uniform(x_start_high, x_end_high)
                center_y = random.uniform(y_start_high, y_end_high)

                # 3. Vypočítat levý horní roh (x, y) z tohoto středu
                x_float = center_x - self.tile_size / 2.0
                y_float = center_y - self.tile_size / 2.0

                # 4. Zaokrouhlit a oříznout na platné souřadnice ve wanted_level
                x = int(round(x_float))
                y = int(round(y_float))

                x = max(0, min(x, wsi_width - self.tile_size))
                y = max(0, min(y, wsi_height - self.tile_size))

                # ----- KONEC ÚPRAVY -----


                # Souřadnice pro čtení musí být na levelu 0
                read_x = int(round(x * native_to_wanted_x))
                read_y = int(round(y * native_to_wanted_y))

                # Ověříme, že i na levelu 0 jsme v mezích (mělo by být díky oříznutí x, y)
                # Ale OpenSlide může mít mírné nepřesnosti v level_dimensions vs downsamples
                if read_x < 0 or read_y < 0 or \
                   read_x + int(round(self.tile_size * native_to_wanted_x)) > native_width or \
                   read_y + int(round(self.tile_size * native_to_wanted_y)) > native_height:
                   # print(f"Varování: Vypočtené souřadnice pro čtení {read_x, read_y} jsou mimo meze levelu 0 {native_width, native_height}. Zkouším znovu.")
                   wsi.close()
                   mask.close()
                   continue # Neplatné souřadnice pro čtení, zkusit znovu

                # Načtení dlaždice a odpovídající masky
                tile = wsi.read_region((read_x, read_y), wanted_level, (self.tile_size, self.tile_size)).convert("RGB")
                mask_tile_pil = mask.read_region((read_x, read_y), wanted_level, (self.tile_size, self.tile_size)).convert("L")

                wsi.close() # Zavřít soubory co nejdříve
                mask.close()

                # Zpracování masky (binarizace)
                mask_tile_np = np.array(mask_tile_pil) > 128 # True/False pole
                mask_tile_img = Image.fromarray(mask_tile_np.astype(np.uint8) * 255, mode="L")

                # Zde by se případně mohl přidat i check min_foreground_ratio pro načtenou dlaždici,
                # ale spoléhání na tissue_mask by mělo být většinou dostatečné a je rychlejší.
                # např.: tile_gray = tile.convert('L')
                #       tile_np = np.array(tile_gray)
                #       is_foreground = tile_np < threshold # Např. Otsu nebo fixní threshold
                #       foreground_ratio = np.sum(is_foreground) / (self.tile_size * self.tile_size)
                #       if foreground_ratio < self.min_foreground_ratio:
                #           continue # Příliš málo tkáně v načtené dlaždici, zkusit znovu


                # Augmentace
                if self.augmentations:
                    tile, mask_tile_img = self.augmentations(tile, mask_tile_img)
                else:
                    tile = TF.to_tensor(tile)
                    mask_tile_img = TF.to_tensor(mask_tile_img) # TF.to_tensor převede [0, 255] na [0, 1]

                return tile, mask_tile_img

            except Exception as e:
                print(f"Chyba při zpracování WSI indexu {wsi_idx} ({self.wsi_paths[wsi_idx]}): {e}")
                # Pokud se soubor nepodaří otevřít nebo nastane jiná chyba, zkusíme další iteraci
                if 'wsi' in locals() and wsi: wsi.close()
                if 'mask' in locals() and mask: mask.close()
                continue # Zkusit další náhodný výběr

            
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
        
