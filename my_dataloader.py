import os
import numpy as np
from PIL import Image # Potřebné pro práci s PIL obrázky, pokud MyAugmentations očekává PIL
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as TF

# Přidání cesty k OpenSlide DLL (upravte podle potřeby)
try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if hasattr(os, 'add_dll_directory') and os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    elif 'OPENSLIDE_PATH' not in os.environ and os.path.exists(os.path.join(openslide_dll_path, 'openslide-0.dll')): # Fallback pro starší Python/OpenSlide
         os.environ['PATH'] = openslide_dll_path + os.pathsep + os.environ['PATH']
except Exception as e:
    print(f"Chyba při nastavování OpenSlide DLL: {e}")

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from my_augmentation import AlbumentationsAug # Použijeme AlbumentationsAug

# --- Konfigurace ---
WSI_PATH = r"C:\Users\USER\Desktop\wsi_dir\tumor_068.tif"
MASK_PATH = r"C:\Users\USER\Desktop\wsi_dir\mask_068.tif"
TARGET_LEVEL_OFFSET = 2 # Kolik úrovní od nejvyššího rozlišení (0) chceme použít.
TILE_COORDS = (32, 120) # Souřadnice dlaždice (sloupec, řádek) na vybrané úrovni
TILE_SIZE = 256 # Velikost dlaždice načítané z WSI
CROP_SIZE = (181, 181) # Cílová velikost po center_crop

# Parametry pro normalizaci (měly by odpovídat těm v AlbumentationsAug)
AUG_MEAN = (0.702, 0.546, 0.696)
AUG_STD = (0.239, 0.282, 0.216)

# --- Hlavní skript ---
try:
    wsi_slide = OpenSlide(WSI_PATH)
    mask_slide = OpenSlide(MASK_PATH)

    wsi_dz = DeepZoomGenerator(wsi_slide, tile_size=TILE_SIZE, overlap=0, limit_bounds=True)
    mask_dz = DeepZoomGenerator(mask_slide, tile_size=TILE_SIZE, overlap=0, limit_bounds=True)

    if wsi_dz.level_count > TARGET_LEVEL_OFFSET:
        target_dz_level = wsi_dz.level_count - 1 - TARGET_LEVEL_OFFSET
    else:
        target_dz_level = wsi_dz.level_count -1
    
    print(f"Používám DeepZoom úroveň: {target_dz_level} (z {wsi_dz.level_count} úrovní)")
    print(f"Odpovídající OpenSlide úroveň přibližně: {wsi_dz.level_count - 1 - target_dz_level}")
    print(f"Rozměry na této DZ úrovni: {wsi_dz.level_dimensions[target_dz_level]}")

    original_wsi_tile_pil = wsi_dz.get_tile(target_dz_level, TILE_COORDS)
    original_mask_tile_pil = mask_dz.get_tile(target_dz_level, TILE_COORDS).convert("L")
    
    print(f"Načtená dlaždice WSI - tvar: {np.array(original_wsi_tile_pil).shape}, typ: {type(original_wsi_tile_pil)}")
    print(f"Načtená dlaždice masky - tvar: {np.array(original_mask_tile_pil).shape}, typ: {type(original_mask_tile_pil)}")

    # Inicializace augmentační třídy AlbumentationsAug
    augmentations = AlbumentationsAug(
        mean=AUG_MEAN, # Tyto hodnoty by měly odpovídat AUG_MEAN, AUG_STD pro správnou denormalizaci
        std=AUG_STD,
        p_flip=0.1,
        p_rotate90=0.5,
        p_shiftscalerotate=0.9,
        p_elastic=0.2, # Můžete upravit pravděpodobnosti
        p_color=0.0,
        p_hestain=0.5,
        p_noise=0.2,
        p_blur=0.1
    )
    
    mean_tensor = torch.tensor(AUG_MEAN, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(AUG_STD, dtype=torch.float32).view(3, 1, 1)

    augmented_outputs = []
    for _ in range(3):
        aug_img_tensor, aug_mask_tensor = augmentations(original_wsi_tile_pil, original_mask_tile_pil)
        augmented_outputs.append((aug_img_tensor, aug_mask_tensor))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Oříznutí originálního snímku a masky na stejnou velikost jako augmentované verze
    original_wsi_tile_pil_cropped = TF.center_crop(original_wsi_tile_pil, CROP_SIZE)
    original_mask_tile_pil_cropped = TF.center_crop(original_mask_tile_pil, CROP_SIZE)

    # Vykreslení oříznutého originálního snímku
    axes[0, 0].imshow(original_wsi_tile_pil_cropped)
    axes[0, 0].set_title("Originální snímek")
    axes[0, 0].axis("off")

    # Vykreslení oříznuté originální masky
    original_mask_display = np.array(original_mask_tile_pil_cropped)
    axes[1, 0].imshow(original_mask_display, cmap='gray')
    axes[1, 0].set_title("Originální maska")
    axes[1, 0].axis("off")

    for i, (aug_img, aug_mask) in enumerate(augmented_outputs):
        # Center crop po augmentaci
        aug_img = TF.center_crop(aug_img, CROP_SIZE)
        aug_mask = TF.center_crop(aug_mask, CROP_SIZE)

        # Denormalizace obrazu (AlbumentationsAug již provedla normalizaci)
        img_display = aug_img.clone() * std_tensor + mean_tensor
        img_display = img_display.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        
        axes[0, i + 1].imshow(img_display)
        axes[0, i + 1].set_title(f"Augmentovaný snímek {i+1}")
        axes[0, i + 1].axis("off")

        # Příprava masky pro zobrazení
        # AlbumentationsAug by měla vracet masku jako [1, H, W] float tensor.
        # Po center_crop bude [1, CROP_SIZE[0], CROP_SIZE[1]]
        if aug_mask.ndim == 4: # Pro případ, že by maska měla neočekávaný tvar [B, C, H, W] nebo [B, H, W, C]
            if aug_mask.shape[0] == 1 and aug_mask.shape[1] == 1 : # [1,1,H,W]
                 mask_display = aug_mask.squeeze(0).squeeze(0).cpu().numpy().clip(0,1)
            elif aug_mask.shape[0] == 1 and aug_mask.shape[3] == 1: # [1, H, W, 1]
                mask_display = aug_mask.squeeze(0).squeeze(-1).cpu().numpy().clip(0, 1)
            elif aug_mask.shape[0] == 1 and aug_mask.shape[3] == 3: # [1, H, W, 3]
                print(f"Varování: Aug. maska {i+1} má tvar [1,H,W,3]. Vykresluji první kanál.")
                mask_display = aug_mask.squeeze(0)[:, :, 0].cpu().numpy().clip(0, 1)
            else:
                raise ValueError(f"Neočekávaný tvar 4D aug. masky: {aug_mask.shape}")
        elif aug_mask.ndim == 3:
            if aug_mask.shape[0] == 1: # Očekávaný tvar [1, H, W]
                mask_display = aug_mask.squeeze(0).cpu().numpy().clip(0, 1)
            elif aug_mask.shape[0] == 3: # Tvar [3, H, W]
                print(f"Varování: Aug. maska {i+1} má 3 kanály. Vykresluji první kanál.")
                mask_display = aug_mask[0, :, :].cpu().numpy().clip(0, 1)
            else:
                raise ValueError(f"Neočekávaný tvar 3D aug. masky: {aug_mask.shape}")
        elif aug_mask.ndim == 2: # Tvar [H, W]
            mask_display = aug_mask.cpu().numpy().clip(0, 1)
        else:
            raise ValueError(f"Neočekávaný počet dimenzí ({aug_mask.ndim}) pro aug. masku: {aug_mask.shape}")

        axes[1, i + 1].imshow(mask_display, cmap='gray')
        axes[1, i + 1].set_title(f"Augmentovaná maska {i+1}")
        axes[1, i + 1].axis("off")

    plt.subplots_adjust(hspace=0.4)
    plt.show()

except FileNotFoundError as e:
    print(f"Chyba: Soubor nebyl nalezen - {e}")
except ImportError as e:
    print(f"Chyba importu: {e}. Ujistěte se, že máte nainstalované všechny potřebné knihovny a soubor my_augmentation.py je dostupný.")
except Exception as e:
    print(f"Nastala neočekávaná chyba: {e}")
    import traceback
    traceback.print_exc()