# --- START OF FILE new_dataset.py ---

import os

try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    else:
        pass # Pokud selže, OpenSlide vyhodí chybu později
except Exception as e:
    print(f"Chyba při nastavování cesty k OpenSlide DLL: {e}")


from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
from openslide import OpenSlide, OpenSlideError
from torchvision.transforms import functional as TF
import time
import torch


"""
    Tato verze zahrnuje:
    1. Opravu diskretizace při výběru souřadnic dlaždice.
    2. Strategii pro vyvážení tříd (častější výběr dlaždic s karcinomem).
    3. Podporu pro zdravé WSI (bez karcinomu) s návratem nulové masky.
"""

class WSITileDatasetBalanced(Dataset):
    def __init__(self,
                 wsi_paths,             # Cesty k VŠEM WSI souborům (.tif)
                 tissue_mask_paths,     # Cesty k maskám VEŠKERÉ tkáně (.npy, low-res) - pro VŠECHNY WSI
                 mask_paths,            # Cesty k ground truth maskám KARCINOMU (.tif, high-res) - None pro zdravé WSI
                 gt_lowres_mask_paths,  # Cesty k maskám POUZE KARCINOMU (.npy, low-res) - None pro zdravé WSI
                 tile_size=256,
                 wanted_level=2,
                 augmentations=None,
                 healthy_wsi_sampling_prob=0.3, # Pravděpodobnost výběru zdravého WSI (pokud jsou k dispozici)
                 positive_sampling_prob=0.7, # Pravděpodobnost pokusu o výběr dlaždice s karcinomem (POUZE pro WSI s karcinomem)
                 min_cancer_ratio_in_tile=0.05,
                 dataset_len=11200,
                 crop=False
                 ):
        self.wsi_paths = wsi_paths
        self.tissue_mask_paths = tissue_mask_paths
        self.mask_paths = mask_paths
        self.gt_lowres_mask_paths = gt_lowres_mask_paths
        self.tile_size = tile_size
        self.wanted_level = wanted_level
        self.augmentations = augmentations
        self.healthy_wsi_sampling_prob = healthy_wsi_sampling_prob
        self.positive_sampling_prob = positive_sampling_prob
        self.min_cancer_ratio_in_tile = min_cancer_ratio_in_tile
        self.dataset_len = dataset_len
        self.crop = crop

        n = len(wsi_paths)
        assert len(tissue_mask_paths) == n, "Seznam tissue_mask_paths musí mít stejnou délku jako wsi_paths."
        assert len(mask_paths) == n, "Seznam mask_paths (GT TIF) musí mít stejnou délku jako wsi_paths."
        assert len(gt_lowres_mask_paths) == n, "Seznam gt_lowres_mask_paths (karcinom NPY) musí mít stejnou délku jako wsi_paths."

        self.cancer_wsi_indices = []
        self.healthy_wsi_indices = []

        for i in range(n):
            # WSI je považováno za "cancer" pokud má definované obě cesty k maskám karcinomu
            if self.mask_paths[i] is not None and self.gt_lowres_mask_paths[i] is not None:
                # Dále ověřme, že i cancer WSI má tissue_mask_path
                if self.tissue_mask_paths[i] is None:
                    print(f"Varování: WSI s karcinomem na indexu {i} ({self.wsi_paths[i]}) postrádá tissue_mask_path. Bude ignorováno.")
                    continue
                self.cancer_wsi_indices.append(i)
            # Jinak je považováno za "healthy", za předpokladu, že má alespoň tissue_mask
            elif self.tissue_mask_paths[i] is not None:
                # Ověříme, že pro healthy WSI jsou mask_paths a gt_lowres_mask_paths None, jak se očekává
                if self.mask_paths[i] is not None or self.gt_lowres_mask_paths[i] is not None:
                    print(f"Varování: WSI na indexu {i} ({self.wsi_paths[i]}) je považováno za zdravé (má tissue_mask), ale má definované i mask_paths nebo gt_lowres_mask_paths. Zkontrolujte data. Bude bráno jako zdravé.")
                self.healthy_wsi_indices.append(i)
            else:
                print(f"Varování: WSI na indexu {i} ({self.wsi_paths[i]}) nemá ani cesty k maskám karcinomu, ani cestu k tkáňové masce. Bude ignorováno.")

        if not self.cancer_wsi_indices and not self.healthy_wsi_indices:
            raise ValueError("Nebyly poskytnuty žádné validní WSI (ani s karcinomem, ani zdravé). Zkontrolujte cesty k maskám.")

        print(f"Dataset inicializován s {len(self.cancer_wsi_indices)} WSI s karcinomem a {len(self.healthy_wsi_indices)} zdravými WSI.")
        if self.healthy_wsi_indices:
            print(f"Pravděpodobnost výběru zdravého WSI: {self.healthy_wsi_sampling_prob*100:.1f}%")
        if self.cancer_wsi_indices:
            print(f"Pravděpodobnost pozitivního samplingu (v rámci WSI s karcinomem): {self.positive_sampling_prob*100:.1f}%")
            if self.min_cancer_ratio_in_tile > 0:
                print(f"Požadován minimální podíl karcinomu v pozitivních dlaždicích: {self.min_cancer_ratio_in_tile*100:.1f}%")


    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            attempts += 1
            wsi = None
            mask_gt_slide = None 

            try:
                wsi_idx = -1
                is_healthy_wsi_sample = False

                can_sample_healthy = len(self.healthy_wsi_indices) > 0
                can_sample_cancer = len(self.cancer_wsi_indices) > 0

                if not can_sample_healthy and not can_sample_cancer:
                     raise RuntimeError("Nebyly nalezeny žádné WSI k výběru (ani zdravé, ani s karcinomem) po inicializaci. Zkontrolujte data.")


                if can_sample_healthy and (not can_sample_cancer or random.random() < self.healthy_wsi_sampling_prob):
                    wsi_idx = random.choice(self.healthy_wsi_indices)
                    is_healthy_wsi_sample = True
                elif can_sample_cancer:
                    wsi_idx = random.choice(self.cancer_wsi_indices)
                    is_healthy_wsi_sample = False
                else:
                    # Pokud máme jen healthy, první podmínka je true. Pokud jen cancer, druhá.
                    # Pokud žádné, chyba výše. Tento else by neměl být dosažitelný.
                    print("Varování: Neočekávaný stav při výběru typu WSI. Zkouším další pokus.")
                    continue


                wsi_path = self.wsi_paths[wsi_idx]
                tissue_mask_path = self.tissue_mask_paths[wsi_idx] 

                wsi = OpenSlide(wsi_path)
                tissue_mask_lowres = np.load(tissue_mask_path)

                gt_lowres_mask = None
                mask_gt_path = None
                if not is_healthy_wsi_sample:
                    mask_gt_path = self.mask_paths[wsi_idx] 
                    gt_lowres_mask_path = self.gt_lowres_mask_paths[wsi_idx]
                    
                    if mask_gt_path:
                         mask_gt_slide = OpenSlide(mask_gt_path)
                    if gt_lowres_mask_path:
                         gt_lowres_mask = np.load(gt_lowres_mask_path)
                
                attempt_positive_sampling = False
                if not is_healthy_wsi_sample and gt_lowres_mask is not None:
                    attempt_positive_sampling = random.random() < self.positive_sampling_prob

                target_coords_lowres = None
                sampling_type = "healthy_tissue" if is_healthy_wsi_sample else "any_tissue"
                ref_mask_shape = tissue_mask_lowres.shape 

                if attempt_positive_sampling: 
                    cancer_y_low, cancer_x_low = np.where(gt_lowres_mask > 0)
                    if len(cancer_y_low) > 0:
                        rand_idx = random.randint(0, len(cancer_y_low) - 1)
                        target_coords_lowres = (cancer_y_low[rand_idx], cancer_x_low[rand_idx])
                        sampling_type = "positive"
                        ref_mask_shape = gt_lowres_mask.shape
                    else: 
                        attempt_positive_sampling = False

                if target_coords_lowres is None: 
                    tissue_y_low, tissue_x_low = np.where(tissue_mask_lowres > 0)
                    if len(tissue_y_low) > 0:
                        rand_idx = random.randint(0, len(tissue_y_low) - 1)
                        target_coords_lowres = (tissue_y_low[rand_idx], tissue_x_low[rand_idx])
                    else:
                        wsi.close()
                        if mask_gt_slide: mask_gt_slide.close()
                        continue

                if ref_mask_shape[0] == 0 or ref_mask_shape[1] == 0:
                    wsi.close()
                    if mask_gt_slide: mask_gt_slide.close()
                    continue

                if self.wanted_level >= wsi.level_count:
                    wsi.close()
                    if mask_gt_slide: mask_gt_slide.close()
                    continue

                wsi_width, wsi_height = wsi.level_dimensions[self.wanted_level]
                native_width, native_height = wsi.level_dimensions[0]
                lowres_mask_height, lowres_mask_width = ref_mask_shape # Přejmenováno pro srozumitelnost

                if lowres_mask_width <= 0 or lowres_mask_height <= 0 or wsi_width <= 0 or wsi_height <= 0:
                     wsi.close()
                     if mask_gt_slide: mask_gt_slide.close()
                     continue

                scale_x = wsi_width / lowres_mask_width
                scale_y = wsi_height / lowres_mask_height
                native_to_wanted_x = native_width / wsi_width
                native_to_wanted_y = native_height / wsi_height

                tx, ty = target_coords_lowres[1], target_coords_lowres[0]
                x_start_high = tx * scale_x
                y_start_high = ty * scale_y
                x_end_high = (tx + 1) * scale_x
                y_end_high = (ty + 1) * scale_y

                if x_end_high <= x_start_high: x_end_high = x_start_high + 1e-6
                if y_end_high <= y_start_high: y_end_high = y_start_high + 1e-6

                center_x = random.uniform(x_start_high, x_end_high)
                center_y = random.uniform(y_start_high, y_end_high)

                x_float = center_x - self.tile_size / 2.0
                y_float = center_y - self.tile_size / 2.0
                x = max(0, min(int(round(x_float)), wsi_width - self.tile_size))
                y = max(0, min(int(round(y_float)), wsi_height - self.tile_size))

                read_x = int(round(x * native_to_wanted_x))
                read_y = int(round(y * native_to_wanted_y))
                
                read_tile_width_lvl0 = int(round(self.tile_size * native_to_wanted_x))
                read_tile_height_lvl0 = int(round(self.tile_size * native_to_wanted_y))
                if read_x < 0 or read_y < 0 or \
                   read_x + read_tile_width_lvl0 > native_width or \
                   read_y + read_tile_height_lvl0 > native_height:
                    wsi.close()
                    if mask_gt_slide: mask_gt_slide.close()
                    continue

                tile_pil = wsi.read_region((read_x, read_y), self.wanted_level, (self.tile_size, self.tile_size)).convert("RGB")
                
                mask_np = np.zeros((self.tile_size, self.tile_size), dtype=bool) 
                if not is_healthy_wsi_sample and mask_gt_slide is not None:
                    mask_pil = mask_gt_slide.read_region((read_x, read_y), self.wanted_level, (self.tile_size, self.tile_size)).convert("L")
                    mask_np = np.array(mask_pil) > 128 
                
                wsi.close()
                if mask_gt_slide: mask_gt_slide.close()

                if sampling_type == "positive" and self.min_cancer_ratio_in_tile > 0:
                    cancer_ratio = np.sum(mask_np) / mask_np.size
                    if cancer_ratio < self.min_cancer_ratio_in_tile:
                         continue

                mask_pil_final = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")

                if self.augmentations:
                    tile_tensor, mask_tensor = self.augmentations(tile_pil, mask_pil_final)
                else:
                    tile_tensor = TF.to_tensor(tile_pil)
                    mask_tensor = TF.to_tensor(mask_pil_final)

                if self.crop:
                    final_crop_size = 256 
                    tile_tensor = TF.center_crop(tile_tensor, (final_crop_size, final_crop_size))
                    mask_tensor = TF.center_crop(mask_tensor, (final_crop_size, final_crop_size))
                
                return tile_tensor, mask_tensor

            except OpenSlideError as e:
                 print(f"Chyba OpenSlide při zpracování WSI indexu {wsi_idx} ({self.wsi_paths[wsi_idx] if wsi_idx != -1 else 'NEZNÁMÝ'}): {e}")
                 if wsi: wsi.close()
                 if mask_gt_slide: mask_gt_slide.close()
                 continue
            except FileNotFoundError as e:
                 print(f"Chyba: Soubor nenalezen pro WSI index {wsi_idx} ({self.wsi_paths[wsi_idx] if wsi_idx != -1 else 'NEZNÁMÝ'}): {e}")
                 if wsi: wsi.close() # Mohl se otevřít wsi, ale ne maska
                 if mask_gt_slide: mask_gt_slide.close()
                 continue
            except Exception as e:
                print(f"Obecná chyba při zpracování WSI indexu {wsi_idx} ({self.wsi_paths[wsi_idx] if wsi_idx != -1 else 'NEZNÁMÝ'}): {e}")
                import traceback
                # traceback.print_exc()
                if 'wsi' in locals() and wsi: wsi.close()
                if 'mask_gt_slide' in locals() and mask_gt_slide: mask_gt_slide.close()
                continue

        raise RuntimeError(f"Nepodařilo se najít vhodnou dlaždici ani po {max_attempts} pokusech. Zkontrolujte data, cesty nebo parametry.")


# --- __main__ sekce ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from new_load_data import load_and_split_data
    from matplotlib import pyplot as plt
    import time
    from my_functions import basic_transform
    try:
        from my_augmentation import MyAugmentations, AlbumentationsAug
        # from my_functions import basic_transform # Pokud existuje
    except ImportError:
        print("Varování: my_augmentation.py nenalezen nebo neobsahuje potřebné třídy.")
        MyAugmentations = None
        AlbumentationsAug = None
        # basic_transform = None

    # vsuvka nacitani dat z load data
    # Definice cest k adresářům (upravte podle vaší skutečné struktury)
    CANCER_WSI_GT_DIR = r"C:\Users\USER\Desktop\wsi_dir"
    CANCER_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\masky_healthy"
    CANCER_LR_GT_DIR = r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky"
    HEALTHY_WSI_DIR = r"C:\Users\USER\Desktop\normal_wsi"
    HEALTHY_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\normal_lowres"

    # Testovací složka pro WSI, která se nepoužije pro train/val
    # TEST_WSI_DIR = r"C:\cesta\k\test\wsi"
    # TEST_LR_MASKS_DIR = r"C:\cesta\k\test\lr_masks"

    print("Načítání a rozdělování dat pro trénink a validaci...")
    train_data, val_data = load_and_split_data(
        cancer_wsi_gt_main_dir=CANCER_WSI_GT_DIR,
        cancer_lr_tissue_mask_dir=CANCER_LR_TISSUE_DIR,
        cancer_lr_gt_mask_dir=CANCER_LR_GT_DIR,
        healthy_wsi_dir=HEALTHY_WSI_DIR,
        healthy_lr_tissue_mask_dir=HEALTHY_LR_TISSUE_DIR,
        val_size=0.2, # Např. 20% pro validaci
        random_state=42
    )

    train_wsi_paths, train_tissue_masks, train_hr_gt_masks, train_lr_gt_masks = train_data
    val_wsi_paths, val_tissue_masks, val_hr_gt_masks, val_lr_gt_masks = val_data
    #konec vsuvky
    random.seed(int(time.time()))  # Pro reprodukovatelnost
    # --- Nastavení augmentací ---
    albumentations_aug = None
    if AlbumentationsAug:
        albumentations_aug = AlbumentationsAug(
            p_flip=0.0, p_color=0.0, p_elastic=0.0, p_rotate90=0.0,
            p_shiftscalerotate=0.0, p_blur=0.0, p_noise=0.0, p_hestain=0.0,
        )
    else:
        print("AlbumentationsAug nejsou dostupné, augmentace nebudou použity.")

    # --- Vytvoření Datasetu a DataLoaderu ---
    all_wsi_paths = 1
    if not all_wsi_paths:
        print("Žádné WSI nebyly nalezeny nebo správně nakonfigurovány. Ukončuji.")
    else:
        print("\nVytváření datasetu...")
        dataset = WSITileDatasetBalanced(
            wsi_paths=train_wsi_paths,
            tissue_mask_paths=train_tissue_masks,
            mask_paths=train_hr_gt_masks,
            gt_lowres_mask_paths=train_lr_gt_masks,
            tile_size=256, 
            wanted_level=1,
            healthy_wsi_sampling_prob=0.4, 
            positive_sampling_prob=0.8,    
            min_cancer_ratio_in_tile=0.05,
            augmentations=basic_transform,
            dataset_len=100, 
            crop=False 
        )

        print("Vytváření DataLoaderu...")
        trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        print("Načítání první várky...")
        try:
            start_time = time.time()
            images, labels = next(iter(trainloader))
            end_time = time.time()
            print(f"První várka ({images.shape[0]} dlaždic) načtena za {end_time - start_time:.2f} sekund.")
            print(f"Rozměry obrázků: {images.shape}, Rozměry masek: {labels.shape}")
            
            if images.shape[0] == 0:
                print("Žádné obrázky k zobrazení v první várce.")
            else:
                print("Zobrazování obrázků...")
                num_to_show = min(4, images.shape[0])
                fig, axes = plt.subplots(2, num_to_show, figsize=(3 * num_to_show, 6), squeeze=False)
                # squeeze=False zajistí, že axes je vždy 2D pole

                for i in range(num_to_show):
                    img = images[i].permute(1, 2, 0).numpy()
                    mask = labels[i].permute(1, 2, 0).squeeze().numpy()

                    # if albumentations_aug and hasattr(albumentations_aug, 'normalize_transform'): 
                    mean = np.array([0.485, 0.456, 0.406]) 
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    
                    img = np.clip(img, 0, 1)

                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f"Obrázek {i}")
                    axes[0, i].axis("off")

                    axes[1, i].imshow(mask, cmap="gray", vmin=0, vmax=1)
                    cancer_perc = np.sum(mask > 0.5) / mask.size * 100
                    axes[1, i].set_title(f"Maska {i} ({cancer_perc:.1f}% Ca)")
                    axes[1, i].axis("off")

                plt.tight_layout()
                plt.show()
                
                # Uložení jednotlivých výřezků ve SVG formátu (čisté obrázky bez popisků)
                patches_dir = r"C:\Users\USER\Desktop\patches"
                os.makedirs(patches_dir, exist_ok=True)
                
                import datetime
                import matplotlib.pyplot as plt
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for i in range(num_to_show):
                    img = images[i].permute(1, 2, 0).numpy()
                    mask = labels[i].permute(1, 2, 0).squeeze().numpy()
                    
                    # Denormalizace pro zobrazení
                    mean = np.array([0.485, 0.456, 0.406]) 
                    std = np.array([0.229, 0.224, 0.225])
                    img_denorm = img * std + mean
                    img_denorm = np.clip(img_denorm, 0, 1)
                    
                    # Výpočet procenta karcinomu pro název souboru
                    cancer_perc = np.sum(mask > 0.5) / mask.size * 100
                    
                    # Uložení obrázku (bez os, titulků, rámečků)
                    fig = plt.figure(figsize=(4, 4))
                    plt.imshow(img_denorm)
                    plt.axis('off')  # Odstraní osy
                    plt.gca().set_position([0, 0, 1, 1])  # Celý obrázek bez okrajů
                    
                    img_filename = f"patch_{timestamp}_{i:02d}_image_cancer{cancer_perc:.1f}pct.png"
                    img_path = os.path.join(patches_dir, img_filename)
                    plt.savefig(img_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()
                    
                    # Uložení masky s červeným ohraničením (bez os, titulků, rámečků)
                    fig = plt.figure(figsize=(4, 4))
                    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
                    plt.axis('off')  # Odstraní osy
                    plt.gca().set_position([0, 0, 1, 1])  # Celý obrázek bez okrajů
                    
                    # Přidání červeného ohraničení
                    ax = plt.gca()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('red')
                        spine.set_linewidth(2)
                    
                    mask_filename = f"patch_{timestamp}_{i:02d}_mask_cancer{cancer_perc:.1f}pct.png"
                    mask_path = os.path.join(patches_dir, mask_filename)
                    plt.savefig(mask_path, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()
                    
                    print(f"Uložena dlaždice {i}: {img_filename} a {mask_filename}")
                
                print(f"Všechny dlaždice uloženy ve SVG formátu do: {patches_dir}")
                
        except StopIteration:
             print("Chyba: DataLoader je prázdný. Zkontrolujte délku datasetu, __len__ metodu, nebo jestli jsou dostupné WSI po filtraci v __init__.")
        except Exception as e:
            print(f"\nNastala chyba při načítání nebo zobrazování dat: {e}")
            import traceback
            traceback.print_exc()

# --- END OF FILE new_dataset.py ---