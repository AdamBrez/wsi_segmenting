import os

try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    else:
        # Zkusit najít v systémové PATH nebo jiné běžné lokaci, pokud existuje
        # print(f"Varování: Cesta k OpenSlide DLL nebyla nalezena: {openslide_dll_path}")
        pass # Pokud selže, OpenSlide vyhodí chybu později
except Exception as e:
    print(f"Chyba při nastavování cesty k OpenSlide DLL: {e}")


from PIL import Image
from torch.utils.data import Dataset
import random # Používat random modul
import numpy as np
from openslide import OpenSlide, OpenSlideError
from torchvision.transforms import functional as TF
import time # Pro měření času v __main__
import torch
"""
    Tato verze zahrnuje:
    1. Opravu diskretizace při výběru souřadnic dlaždice.
    2. Strategii pro vyvážení tříd (častější výběr dlaždic s karcinomem).
    Byla použita pro trénink, který dosáhl výrazného zlepšení. 
    Průměrná rychlost načítaní batchů: 0.073s.
"""

class WSITileDatasetBalanced(Dataset):
    def __init__(self,
                 wsi_paths,             # Cesty k WSI souborům (.tif)
                 tissue_mask_paths,     # Cesty k maskám VEŠKERÉ tkáně (.npy, low-res)
                 mask_paths,            # Cesty k ground truth maskám (.tif, high-res)
                 gt_lowres_mask_paths,  # Cesty k maskám POUZE KARCINOMU (.npy, low-res) - MUSÍ mít stejné rozlišení jako tissue_mask_paths
                 tile_size=256,         # Velikost dlaždice
                 wanted_level=2,        # Level WSI, ze kterého čteme dlaždice
                 augmentations=None,    # Objekt s augmentacemi
                 positive_sampling_prob=0.7, # Pravděpodobnost pokusu o výběr dlaždice s karcinomem
                 min_cancer_ratio_in_tile=0.05,
                 dataset_len=11200, # Minimální podíl karcinomu v dlaždici při pozitivním samplingu (0 pro vypnutí)
                 context_level=3,
                 context_size=256 # Velikost kontextové dlaždice (v pixelech)
                 ):
        self.wsi_paths = wsi_paths
        self.tissue_mask_paths = tissue_mask_paths
        self.mask_paths = mask_paths
        self.gt_lowres_mask_paths = gt_lowres_mask_paths
        self.tile_size = tile_size
        self.wanted_level = wanted_level
        self.augmentations = augmentations
        self.positive_sampling_prob = positive_sampling_prob
        self.min_cancer_ratio_in_tile = min_cancer_ratio_in_tile
        self.dataset_len = dataset_len # Počet dlaždic na epochu (upravte dle potřeby)
        self.context_level = context_level # Level pro kontextovou dlaždici (nízké rozlišení)
        self.context_size = context_size # Velikost kontextové dlaždice (v pixelech)

        # Ověření délky seznamů
        n = len(wsi_paths)
        assert len(tissue_mask_paths) == n, "Seznam tissue_mask_paths musí mít stejnou délku jako wsi_paths."
        assert len(mask_paths) == n, "Seznam mask_paths (GT TIF) musí mít stejnou délku jako wsi_paths."
        assert len(gt_lowres_mask_paths) == n, "Seznam gt_lowres_mask_paths (karcinom NPY) musí mít stejnou délku jako wsi_paths."

        print(f"Dataset inicializován s {n} WSI.")
        print(f"Pravděpodobnost pozitivního samplingu: {self.positive_sampling_prob*100:.1f}%")
        if self.min_cancer_ratio_in_tile > 0:
            print(f"Požadován minimální podíl karcinomu v pozitivních dlaždicích: {self.min_cancer_ratio_in_tile*100:.1f}%")


    def __len__(self):
        # Můžete nechat fixní, nebo lépe odhadnout
        # Např. počet WSI * průměrný počet dlaždic na WSI
        return self.dataset_len  # Počet dlaždic na epochu (upravte dle potřeby)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 100 # Ochrana proti zacyklení

        while attempts < max_attempts:
            attempts += 1
            # Náhodný výběr WSI
            wsi_idx = random.randint(0, len(self.wsi_paths) - 1)

            wsi = None
            mask_gt = None
            try:
                # --- Otevření souborů ---
                wsi_path = self.wsi_paths[wsi_idx]
                mask_gt_path = self.mask_paths[wsi_idx]
                tissue_mask_path = self.tissue_mask_paths[wsi_idx]
                gt_lowres_mask_path = self.gt_lowres_mask_paths[wsi_idx]

                wsi = OpenSlide(wsi_path)
                mask_gt = OpenSlide(mask_gt_path)

                # --- Načtení nízkorozlišovacích masek ---
                tissue_mask_lowres = np.load(tissue_mask_path)
                gt_lowres_mask = np.load(gt_lowres_mask_path) # Maska POUZE karcinomu

                # --- Rozhodnutí o typu samplingu ---
                attempt_positive_sampling = random.random() < self.positive_sampling_prob

                # Najít souřadnice v nízkorozlišovacích maskách
                # nešlo by přesunout až někde kde vím jakou masku beru?
                # abych zbytečně nenačítal i tu druhou
                cancer_y_low, cancer_x_low = np.where(gt_lowres_mask > 0)
                tissue_y_low, tissue_x_low = np.where(tissue_mask_lowres > 0)

                target_coords_lowres = None # (y, x) pixel v low-res masce, kolem kterého centrujeme

                # --- Výběr cílového pixelu v nízkorozlišovací masce ---
                if attempt_positive_sampling and len(cancer_y_low) > 0:
                    # Povedlo se najít karcinom a chceme ho samplovat
                    rand_idx = random.randint(0, len(cancer_y_low) - 1)
                    target_y_low = cancer_y_low[rand_idx]
                    target_x_low = cancer_x_low[rand_idx]
                    target_coords_lowres = (target_y_low, target_x_low)
                    sampling_type = "positive"
                    # Pro výpočet scale použijeme rozměry masky, ze které jsme vybírali
                    ref_mask_shape = gt_lowres_mask.shape
                elif len(tissue_y_low) > 0:
                    # Buď jsme nechtěli pozitivní, nebo se nenašel karcinom -> samplujeme z jakékoli tkáně
                    rand_idx = random.randint(0, len(tissue_y_low) - 1)
                    target_y_low = tissue_y_low[rand_idx]
                    target_x_low = tissue_x_low[rand_idx]
                    target_coords_lowres = (target_y_low, target_x_low)
                    sampling_type = "any_tissue"
                    ref_mask_shape = tissue_mask_lowres.shape
                else:
                    # print(f"Varování: Žádná tkáň nalezena v {tissue_mask_path} ani {gt_lowres_mask_path}. Zkouším jiný WSI ({wsi_idx}).")
                    wsi.close()
                    mask_gt.close()
                    continue # Přeskočit na další iteraci while cyklu

                # --- Přepočty rozměrů a měřítek ---
                if ref_mask_shape[0] == 0 or ref_mask_shape[1] == 0:
                    # print(f"Varování: Neplatná low-res maska (shape {ref_mask_shape}) pro WSI {wsi_idx}. Zkouším jiný.")
                    wsi.close()
                    mask_gt.close()
                    continue

                # Ověříme existenci levelu
                if self.wanted_level >= wsi.level_count:
                    print(f"Chyba: WSI {wsi_path} nemá level {self.wanted_level}. Maximální level: {wsi.level_count - 1}.")
                    wsi.close()
                    mask_gt.close()
                    continue

                wsi_width, wsi_height = wsi.level_dimensions[self.wanted_level]
                native_width, native_height = wsi.level_dimensions[0]
                tissue_mask_height, tissue_mask_width = ref_mask_shape # Rozměry masky, ze které jsme vybírali

                if tissue_mask_width <= 0 or tissue_mask_height <= 0 or wsi_width <= 0 or wsi_height <= 0:
                     # print(f"Varování: Neplatné rozměry WSI nebo masky pro WSI {wsi_idx}. Zkouším jiný.")
                     wsi.close()
                     mask_gt.close()
                     continue

                scale_x = wsi_width / tissue_mask_width
                scale_y = wsi_height / tissue_mask_height
                # Kontrola dělení nulou (mělo by být pokryto kontrolou wsi_width/height)
                if wsi_width == 0 or wsi_height == 0: continue
                native_to_wanted_x = native_width / wsi_width
                native_to_wanted_y = native_height / wsi_height

                # --- Výpočet pozice dlaždice (centrování kolem target_coords_lowres) ---
                # (Používá metodu z první opravy pro řešení diskretizace)
                tx, ty = target_coords_lowres[1], target_coords_lowres[0] # x, y

                x_start_high = tx * scale_x
                y_start_high = ty * scale_y
                x_end_high = (tx + 1) * scale_x
                y_end_high = (ty + 1) * scale_y

                # Zajistíme, aby end > start pro random.uniform
                if x_end_high <= x_start_high: x_end_high = x_start_high + 1e-6
                if y_end_high <= y_start_high: y_end_high = y_start_high + 1e-6

                center_x = random.uniform(x_start_high, x_end_high)
                center_y = random.uniform(y_start_high, y_end_high)

                x_float = center_x - self.tile_size / 2.0
                y_float = center_y - self.tile_size / 2.0

                x = int(round(x_float))
                y = int(round(y_float))
                x = max(0, min(x, wsi_width - self.tile_size))
                y = max(0, min(y, wsi_height - self.tile_size))

                # --- Načtení dlaždice a masky ---
                read_x = int(round(x * native_to_wanted_x))
                read_y = int(round(y * native_to_wanted_y))

                # Bezpečnostní kontrola souřadnic pro čtení na levelu 0
                read_tile_width_lvl0 = int(round(self.tile_size * native_to_wanted_x))
                read_tile_height_lvl0 = int(round(self.tile_size * native_to_wanted_y))
                if read_x < 0 or read_y < 0 or \
                   read_x + read_tile_width_lvl0 > native_width or \
                   read_y + read_tile_height_lvl0 > native_height:
                    # print(f"Varování: Vypočtené souřadnice pro čtení mimo meze levelu 0. WSI: {wsi_idx}, read: ({read_x},{read_y}), tile_lvl0:({read_tile_width_lvl0},{read_tile_height_lvl0}), native: ({native_width},{native_height})")
                    wsi.close()
                    mask_gt.close()
                    continue # Zkusit znovu
                # print(f"Načítám dlaždici z WSI {wsi_path.split("\\")[-1]} ({x},{y}) na levelu {self.wanted_level} (souřadnice pro čtení: {read_x},{read_y})")
                tile_pil = wsi.read_region((read_x, read_y), self.wanted_level, (self.tile_size, self.tile_size)).convert("RGB")
                mask_pil = mask_gt.read_region((read_x, read_y), self.wanted_level, (self.tile_size, self.tile_size)).convert("L")

                # --- Load context patch at lower resolution ---
                # Compute mapping to native (level 0)
                native_width, native_height = wsi.level_dimensions[0]
                w_high_w, w_high_h = wsi.level_dimensions[self.wanted_level]
                w_ctx_w, w_ctx_h = wsi.level_dimensions[self.context_level]

                native_to_high_x = native_width / w_high_w
                native_to_high_y = native_height / w_high_h
                native_to_ctx_x = native_width / w_ctx_w
                native_to_ctx_y = native_height / w_ctx_h

                # center_x, center_y from above (float in wanted_level px space)
                center_native_x = center_x * native_to_high_x
                center_native_y = center_y * native_to_high_y

                # context patch size in native pixels
                ctx_native_w = int(round(self.context_size * native_to_ctx_x))
                ctx_native_h = int(round(self.context_size * native_to_ctx_y))

                ctx_x0 = int(round(center_native_x - ctx_native_w / 2))
                ctx_y0 = int(round(center_native_y - ctx_native_h / 2))
                # clamp
                ctx_x0 = max(0, min(ctx_x0, native_width - ctx_native_w))
                ctx_y0 = max(0, min(ctx_y0, native_height - ctx_native_h))

                context_pil = wsi.read_region((ctx_x0, ctx_y0), self.context_level,
                                            (self.context_size, self.context_size)).convert("RGB")

                wsi.close()
                mask_gt.close()

                # --- Zpracování a kontrola masky ---
                mask_np = np.array(mask_pil) > 128 # Binarizace GT masky (True/False)

                # KONTROLA: Ověřit minimální podíl karcinomu, POKUD jsme se o to snažili
                if sampling_type == "positive" and self.min_cancer_ratio_in_tile > 0:
                    cancer_ratio = np.sum(mask_np) / mask_np.size
                    if cancer_ratio < self.min_cancer_ratio_in_tile:
                         # print(f"INFO: Dlaždice z WSI {wsi_idx} ({x},{y}) zamítnuta - nízký podíl karcinomu ({cancer_ratio:.3f} < {self.min_cancer_ratio_in_tile}). Hledám dál.")
                         continue # Dlaždice nesplňuje kritérium, zkusit novou

                # Převod masky na formát pro augmentace/ToTensor (0-255)
                mask_pil_final = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")

                # --- Augmentace a převod na Tensor ---
                if self.augmentations:
                    # Předpokládáme, že augmentace vrací tensory
                    tile_tensor, mask_tensor, context_tensor = self.augmentations(tile_pil, mask_pil_final, context_pil)
                      
                else:
                    tile_tensor = TF.to_tensor(tile_pil) # Scales to [0, 1]
                    mask_tensor = TF.to_tensor(mask_pil_final) # Scales [0, 255] to [0, 1]
                    context_tensor = TF.to_tensor(context_pil) # Scales to [0, 1]

                # print(f"Úspěšně vrácena dlaždice z WSI {wsi_idx} (typ: {sampling_type})")
                combined_tensor = torch.cat((tile_tensor, context_tensor), dim=0) # Spojení dlaždice a kontextu
                # print(combined_tensor.shape, tile_tensor.shape, context_tensor.shape, mask_tensor.shape)
                return combined_tensor, mask_tensor

            except OpenSlideError as e:
                 print(f"Chyba OpenSlide při zpracování WSI indexu {wsi_idx} ({self.wsi_paths[wsi_idx]}): {e}")
                 # Zavřít soubory, pokud byly otevřeny
                 if wsi: wsi.close()
                 if mask_gt: mask_gt.close()
                 continue # Zkusit další iteraci
            except FileNotFoundError as e:
                 print(f"Chyba: Soubor nenalezen pro WSI index {wsi_idx}: {e}")
                 # Není co zavírat, pokud se soubor nenalezl
                 continue # Zkusit další iteraci
            except Exception as e:
                print(f"Obecná chyba při zpracování WSI indexu {wsi_idx} ({self.wsi_paths[wsi_idx]}): {e}")
                import traceback
                # traceback.print_exc() # Odkomentovat pro detailní výpis chyby
                if 'wsi' in locals() and wsi: wsi.close()
                if 'mask_gt' in locals() and mask_gt: mask_gt.close()
                continue # Zkusit další iteraci

        # Pokud se ani po max_attempts nepodařilo najít dlaždici
        raise RuntimeError(f"Nepodařilo se najít vhodnou dlaždici ani po {max_attempts} pokusech. Zkontrolujte data, cesty nebo parametry (např. min_cancer_ratio).")


# --- __main__ sekce ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt
    # Předpokládáme existenci my_augmentation.py
    try:
        from my_augmentation import MyAugmentations
    except ImportError:
        print("Varování: my_augmentation.py nenalezen nebo neobsahuje MyAugmentations.")
        MyAugmentations = None # Definujeme jako None, pokud neexistuje


    # --- Definice cest (upravte podle vaší struktury) ---
    # Základní cesty k WSI a GT maskám TIF
    wsi_dir = r"C:\Users\USER\Desktop\wsi_dir"
    # Předpoklad: názvy WSI a masek si odpovídají (tumor_XXX.tif, mask_XXX.tif)
    wsi_ids = ["001", "002", "003", "089", "017"] # Přidejte ID vašich WSI
    wsi_paths_train = [os.path.join(wsi_dir, f"tumor_{id}.tif") for id in wsi_ids]
    mask_paths_train = [os.path.join(wsi_dir, f"mask_{id}.tif") for id in wsi_ids]

    # Cesty k NPY maskám (nízkorozlišovací)
    lowres_mask_base_dir = r"C:\Users\USER\Desktop\colab_unet"
    # 1. Masky veškeré tkáně (např. z Otsu a morfologie na nízkém levelu)
    tissue_mask_dir = os.path.join(lowres_mask_base_dir, "masky_healthy") # Adresář pro masky veškeré tkáně
    tissue_mask_paths_train = [os.path.join(tissue_mask_dir, f"mask_{id}.npy") for id in wsi_ids]

    # 2. Masky POUZE karcinomu (ze zmenšené GT masky)
    gt_lowres_mask_dir = os.path.join(lowres_mask_base_dir, "gt_lowres_masky") # Adresář pro masky karcinomu
    gt_lowres_mask_paths_train = [os.path.join(gt_lowres_mask_dir, f"mask_{id}_cancer.npy") for id in wsi_ids]

    # --- Nastavení augmentací ---
    augmentations = None
    if MyAugmentations:
        IMAGENET_MEAN = (0.485, 0.456, 0.406) # Upravte podle potřeby
        IMAGENET_STD = (0.229, 0.224, 0.225) # Upravte podle potřeby
        color_jitter_params = {"brightness": 0.25, "contrast": 0.25, "saturation": 0.2, "hue": 0.05}
        augmentations = MyAugmentations(
            p_flip=0.5,
            p_color=0.5,
            color_jitter_params=color_jitter_params,
            mean=IMAGENET_MEAN, # Upravte podle potřeby
            std=IMAGENET_STD  # Upravte podle potřeby
        )

    # --- Vytvoření Datasetu a DataLoaderu ---
    print("\nVytváření datasetu...")
    dataset = WSITileDatasetBalanced(
        wsi_paths=wsi_paths_train,
        tissue_mask_paths=tissue_mask_paths_train,
        mask_paths=mask_paths_train,
        gt_lowres_mask_paths=gt_lowres_mask_paths_train, # Přidáno
        tile_size=256,
        wanted_level=1,
        augmentations=augmentations, # Použijeme definované augmentace
        positive_sampling_prob=0.6,  # 70% šance na pokus o karcinom
        min_cancer_ratio_in_tile=0.05 # Vyžadovat alespoň 5% karcinomu v dlaždici při pozitivním samplingu
    )

    print("Vytváření DataLoaderu...")
    # Zvažte použití více workerů pro rychlejší načítání, pokud nemáte problémy s pamětí/DLL
    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 pro snazší debugování

    # --- Testování načítání a zobrazení ---
    print("Načítání první várky...")
    try:
        start_time = time.time()
        images, labels = next(iter(trainloader))
        end_time = time.time()
        print(f"První várka ({images.shape[0]} dlaždic) načtena za {end_time - start_time:.2f} sekund.")

        print("Zobrazování obrázků...")
        fig, axes = plt.subplots(2, min(4, images.shape[0]), figsize=(12, 6))
        if images.shape[0] == 1: # Matplotlib subplot nevrací pole pro 1 sloupec
             axes = np.array([[axes[0]],[axes[1]]])
        elif images.shape[0] < 4: # Pokud je batch menší než 4
             # Upravit rozměry pro zobrazení
             fig, axes = plt.subplots(2, images.shape[0], figsize=(3 * images.shape[0], 6))


        for i in range(images.shape[0]): # Zobrazit všechny obrázky z várky (max 4)
            img = images[i].permute(1, 2, 0).numpy()
            mask = labels[i].permute(1, 2, 0).squeeze().numpy() # squeeze() pro odstranění kanálové dimenze

                # Denormalizace pro ImageNet normalizaci
            if augmentations:
                mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)  # Shape (1,1,3)
                std = np.array(IMAGENET_STD).reshape(1, 1, 3)    # Shape (1,1,3)
                img = img * std + mean 

            img = np.clip(img, 0, 1) # Oříznout hodnoty pro jistotu

            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Obrázek {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(mask, cmap="gray", vmin=0, vmax=1) # Zajistit rozsah 0-1 pro cmap
            cancer_perc = np.sum(mask > 0.5) / mask.size * 100 # Procento karcinomu
            axes[1, i].set_title(f"Maska {i} ({cancer_perc:.1f}% Ca)")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

    except StopIteration:
         print("Chyba: DataLoader je prázdný. Zkontrolujte délku datasetu a __len__ metodu.")
    except Exception as e:
        print(f"\nNastala chyba při načítání nebo zobrazování dat: {e}")
        import traceback
        traceback.print_exc()

# --- END OF FILE dataloader_gemini_balanced.py ---