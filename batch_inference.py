# --- START OF FILE batch_inference_tissue_mask_filter_lvl6.py ---

import os
# !!! Zajistěte správnou cestu k OpenSlide !!!
try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    else:
        print(f"Varování: Cesta k OpenSlide DLL neexistuje: {openslide_dll_path}")
except AttributeError:
    print("os.add_dll_directory není dostupné.")
except Exception as e:
    print(f"Nastala chyba při přidávání OpenSlide DLL: {e}")

import torch
from PIL import ImageOps # ImageOps pro padding
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import numpy as np # Potřebujeme NumPy pro masku tkáně
import openslide
from openslide.deepzoom import DeepZoomGenerator
import time
import segmentation_models_pytorch as smp
import datetime
import gc

"""
    Načítá se celé WSI, je rozřezáváno BEZ PŘEKRYVU.
    Inference je prováděna POUZE na dlaždicích překrývajících tkáň
    podle nízko-rozlišovací masky (generované z WSI level 6).
    Zpracování v DÁVKÁCH. Okrajové dlaždice jsou PADOVÁNY. Používá se normalizace.
    Výstupem je binární maska (HDF5, bool), inicializovaná na False.
    Metadata jsou zachována z původního skriptu.
"""

# --- Konfigurace ---
model_weights_path = r"C:\Users\USER\Desktop\weights\unet_smp_e50_9920len.pth"
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif" # Zkontroluj cestu
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\pred_089_filtered_lvl6.h5" # Nový název!

# <<< Konfigurace pro filtrování podle masky tkáně >>>
tissue_mask_dir = r"C:\Users\USER\Desktop\colab_unet\masky_new" # Adresář s .npy maskami
tissue_mask_level_index = 6 # Úroveň OpenSlide, ze které byla maska generována
# Minimální podíl tkáně v dlaždici (v masce tkáně), aby se spustila inference (0.0 až 1.0)
TISSUE_THRESHOLD = 0.1 # Např. 10%
# <<< KONEC Konfigurace pro filtrování >>>

# Velikost, kterou očekává model (a velikost dlaždice bez overlapu)
TILE_SIZE = 256
OVERLAP = 0 # Překryv NENÍ použit

BATCH_SIZE = 64
THRESHOLD = 0.5
TARGET_LEVEL_OFFSET = 2 # Offset od nejvyšší úrovně DZG (0 = nejvyšší)

# --- Inicializace ---
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
print(f"Model očekává vstup: {TILE_SIZE}x{TILE_SIZE}")

# Načtení modelu
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
try:
    if not os.path.exists(model_weights_path): raise FileNotFoundError(f"Váhy nenalezeny: {model_weights_path}")
    try: model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    except TypeError:
         print("Varování: weights_only není podporováno, načítám standardně.")
         model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
except Exception as e: print(f"Chyba při načítání vah modelu: {e}"); exit()
model.to(device)
model.eval()

# Normalizační transformace
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Načtení WSI
wsi = None
tissue_mask_np = None
hdf5_file = None
try:
    wsi = openslide.OpenSlide(wsi_image_path)
    print(f"WSI načteno: {wsi_image_path}")

    # Ověření existence úrovně pro masku tkáně
    if tissue_mask_level_index >= wsi.level_count:
        raise ValueError(f"Požadovaná úroveň masky tkáně ({tissue_mask_level_index}) neexistuje ve WSI (max index {wsi.level_count - 1}).")
    tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
    print(f"Maska tkáně byla generována z úrovně {tissue_mask_level_index} (downsample {tissue_mask_downsample:.2f}x).")

    # Načtení masky tkáně
    wsi_filename_base = os.path.splitext(os.path.basename(wsi_image_path))[0]
    tissue_mask_filename = "mask_089.npy"#f"{wsi_filename_base}.npy"
    tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
    if not os.path.exists(tissue_mask_full_path):
        raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
    tissue_mask_np = np.load(tissue_mask_full_path)
    print(f"Maska tkáně načtena z: {tissue_mask_full_path}, tvar: {tissue_mask_np.shape}")
    if not np.issubdtype(tissue_mask_np.dtype, np.bool_):
         print(f"Varování: Maska tkáně není typu bool (je {tissue_mask_np.dtype}). Převedu ji.")
         tissue_mask_np = tissue_mask_np.astype(bool)

    # Vytvoření DeepZoomGeneratoru
    deepzoom = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=False)
    print(f"DZG parametry: tile_size={TILE_SIZE}, overlap={OVERLAP}")

    # Výběr úrovně pro zpracování (index DZG)
    dzg_level_index = deepzoom.level_count - 1 - TARGET_LEVEL_OFFSET
    if dzg_level_index < 0:
         print(f"Chyba: Vypočtený DZG level index {dzg_level_index} je záporný. Používám nejvyšší úroveň {deepzoom.level_count - 1}.")
         dzg_level_index = deepzoom.level_count - 1
    elif dzg_level_index >= deepzoom.level_count:
         print(f"Chyba: Vypočtený DZG level index {dzg_level_index} je mimo rozsah. Používám nejvyšší úroveň {deepzoom.level_count - 1}.")
         dzg_level_index = deepzoom.level_count - 1

    # Nalezení odpovídající OpenSlide úrovně pro inferenci
    dzg_level_dims = deepzoom.level_dimensions[dzg_level_index]
    dzg_downsample_est = 2**(deepzoom.level_count - 1 - dzg_level_index)
    closest_os_level_index = -1
    min_diff = float('inf')
    for i in range(wsi.level_count):
        os_ds = wsi.level_downsamples[i]
        diff = abs(os_ds - dzg_downsample_est)
        if diff < min_diff: min_diff = diff; closest_os_level_index = i
        if diff < 0.01: break
    if closest_os_level_index == -1: raise ValueError("Nepodařilo se najít odpovídající OpenSlide úroveň pro inferenci.")

    inference_level_index = closest_os_level_index
    inference_downsample = wsi.level_downsamples[inference_level_index]
    inference_level_dims = wsi.level_dimensions[inference_level_index] # Použijeme OS rozměry pro HDF5

    print(f"Zpracovávám DZG úroveň {dzg_level_index} (odpovídá OpenSlide úrovni {inference_level_index})")
    print(f"   - Rozměry inference úrovně (OpenSlide): {inference_level_dims} (šířka, výška)")
    print(f"   - Downsample inference úrovně: {inference_downsample:.2f}x")

    # Výpočet škálovacího faktoru mezi maskou a inference úrovní
    if inference_downsample <= 0: raise ValueError("Downsample inference úrovně je neplatný.")
    scale_factor = tissue_mask_downsample / inference_downsample
    if scale_factor <= 0: raise ValueError(f"Chyba výpočtu scale_factor ({tissue_mask_downsample}/{inference_downsample}).")
    print(f"Škálovací faktor (Maska / Inference Level): {scale_factor:.3f}")

    # Mřížka DZG
    level_tiles_cols, level_tiles_rows = deepzoom.level_tiles[dzg_level_index]
    total_tiles_grid = level_tiles_rows * level_tiles_cols
    print(f"   - Očekávaný počet dlaždic v mřížce DZG: {level_tiles_cols}x{level_tiles_rows} = {total_tiles_grid}")

    # Výstupní tvar HDF5 (podle OpenSlide rozměrů inference úrovně)
    output_shape = (inference_level_dims[1], inference_level_dims[0]) # (výška, šířka)

    # --- Příprava HDF5 souboru ---
    hdf5_file = h5py.File(output_hdf5_path, "w")
    dset = hdf5_file.create_dataset(
        "mask",
        shape=output_shape,
        dtype=bool,
        fillvalue=False, # <<< DŮLEŽITÉ: Inicializace na False
        chunks=(TILE_SIZE, TILE_SIZE), # Můžeme ponechat chunk TILE_SIZE
        compression="gzip"
    )
    # --- Zápis PŮVODNÍCH metadat (dle požadavku) ---
    hdf5_file.attrs["day_of_creation"] = datetime.datetime.now().isoformat()
    hdf5_file.attrs["tile_size"] = TILE_SIZE # Velikost dlaždice použité DZG
    hdf5_file.attrs["overlap"] = OVERLAP
    hdf5_file.attrs["batch_size"] = BATCH_SIZE
    hdf5_file.attrs["model_weights"] = os.path.basename(model_weights_path) # Jen název souboru
    # Upřesnění informace o úrovni
    hdf5_file.attrs["wsi_level_processed"] = f"DZG Level {dzg_level_index} (OS Level {inference_level_index})"
    hdf5_file.attrs["wsi_level_dims"] = f"{inference_level_dims[0]}x{inference_level_dims[1]}"
    hdf5_file.attrs["augmentation"] = "False"
    # --- Konec zápisu metadat ---

    # --- Zpracování ---
    processed_tiles_for_model = 0
    tiles_skipped_by_mask = 0
    batch_tiles_data = [] # Tenzory pro model
    batch_coords = []     # Souřadnice pro zápis do HDF5 a původní velikosti

    with torch.inference_mode():
        for row in tqdm(range(level_tiles_rows), desc="Processing rows"):
            for col in range(level_tiles_cols):
                # Získání souřadnic a skutečné velikosti dlaždice
                try:
                    tile_coords_level0 = deepzoom.get_tile_coordinates(dzg_level_index, (col, row))[0]
                    x_inf_start = int(tile_coords_level0[0] / inference_downsample)
                    y_inf_start = int(tile_coords_level0[1] / inference_downsample)
                    tile_w_inf, tile_h_inf = deepzoom.get_tile_dimensions(dzg_level_index, (col, row))
                    if tile_w_inf <= 0 or tile_h_inf <= 0: continue
                except Exception as coord_err:
                    print(f"Chyba při získávání souřadnic pro [{col},{row}]: {coord_err}")
                    continue

                # --- Filtrování podle masky tkáně ---
                tm_x_start = int(x_inf_start / scale_factor)
                tm_y_start = int(y_inf_start / scale_factor)
                tm_w = max(1, int(tile_w_inf / scale_factor)) # Zajisti aspoň 1px šířku/výšku
                tm_h = max(1, int(tile_h_inf / scale_factor))

                tm_y_end = min(tm_y_start + tm_h, tissue_mask_np.shape[0])
                tm_x_end = min(tm_x_start + tm_w, tissue_mask_np.shape[1])
                tm_y_start_clipped = max(0, tm_y_start)
                tm_x_start_clipped = max(0, tm_x_start)

                tissue_ratio = 0.0
                if tm_y_end > tm_y_start_clipped and tm_x_end > tm_x_start_clipped:
                    tissue_region = tissue_mask_np[tm_y_start_clipped:tm_y_end, tm_x_start_clipped:tm_x_end]
                    if tissue_region.size > 0:
                        tissue_ratio = np.mean(tissue_region)

                # --- Podmíněná inference ---
                if tissue_ratio >= TISSUE_THRESHOLD:
                    try:
                        tile = deepzoom.get_tile(dzg_level_index, (col, row))
                        if tile is None: continue
                        tile_rgb = tile.convert("RGB")
                        tile_w_orig, tile_h_orig = tile_rgb.size
                        if tile_w_orig == 0 or tile_h_orig == 0: continue

                        delta_w = max(0, TILE_SIZE - tile_w_orig)
                        delta_h = max(0, TILE_SIZE - tile_h_orig)
                        padding = (0, 0, delta_w, delta_h)
                        tile_padded = ImageOps.expand(tile_rgb, padding, fill=0)

                        if tile_padded.size != (TILE_SIZE, TILE_SIZE):
                            print(f"Varování: Velikost po paddingu není {TILE_SIZE}x{TILE_SIZE} ({tile_padded.size}) pro [{col},{row}]. Přeskakuji.")
                            continue

                        tile_tensor = to_tensor(tile_padded)
                        tile_tensor_normalized = normalize(tile_tensor)

                        batch_tiles_data.append(tile_tensor_normalized)
                        batch_coords.append({'x_inf': x_inf_start, 'y_inf': y_inf_start, 'w_orig': tile_w_orig, 'h_orig': tile_h_orig})
                        processed_tiles_for_model += 1

                    except openslide.OpenSlideError as ose: print(f"OpenSlide chyba při čtení dlaždice [{col},{row}]: {ose}"); continue
                    except Exception as prep_err:
                         print(f"Chyba při přípravě dlaždice [{col},{row}]: {prep_err}")
                         if len(batch_coords) > len(batch_tiles_data): batch_coords.pop()
                         continue
                else:
                    tiles_skipped_by_mask += 1
                    continue # Přeskočit inferenci

                # --- Zpracování dávky (pokud je plná) ---
                if len(batch_tiles_data) == BATCH_SIZE:
                    if not batch_tiles_data: continue

                    try:
                        batch_tensor = torch.stack(batch_tiles_data).to(device)
                        prediction_output = model(batch_tensor)
                        predictions = torch.sigmoid(prediction_output).cpu()

                        for i in range(predictions.shape[0]):
                            pred_tensor = predictions[i].squeeze()
                            coords = batch_coords[i]
                            orig_w, orig_h = coords['w_orig'], coords['h_orig']
                            binary_pred_padded = (pred_tensor >= THRESHOLD).numpy().astype(bool)
                            binary_tile_cropped = binary_pred_padded[:orig_h, :orig_w]

                            x_start_hdf5 = coords['x_inf']
                            y_start_hdf5 = coords['y_inf']
                            y_end_hdf5 = min(y_start_hdf5 + orig_h, output_shape[0])
                            x_end_hdf5 = min(x_start_hdf5 + orig_w, output_shape[1])

                            if y_end_hdf5 > y_start_hdf5 and x_end_hdf5 > x_start_hdf5:
                                hdf5_h = y_end_hdf5 - y_start_hdf5
                                hdf5_w = x_end_hdf5 - x_start_hdf5
                                dset[y_start_hdf5:y_end_hdf5, x_start_hdf5:x_end_hdf5] = binary_tile_cropped[:hdf5_h, :hdf5_w]

                    except Exception as batch_proc_err: print(f"Chyba při zpracování dávky: {batch_proc_err}")
                    finally:
                        batch_tiles_data.clear()
                        batch_coords.clear()

        # --- Zpracování poslední (neúplné) dávky ---
        if batch_tiles_data:
            print(f"Zpracovávám poslední dávku ({len(batch_tiles_data)} dlaždic)...")
            try:
                batch_tensor = torch.stack(batch_tiles_data).to(device)
                prediction_output = model(batch_tensor)
                predictions = torch.sigmoid(prediction_output).cpu()

                for i in range(predictions.shape[0]):
                    pred_tensor = predictions[i].squeeze()
                    coords = batch_coords[i]
                    orig_w, orig_h = coords['w_orig'], coords['h_orig']
                    binary_pred_padded = (pred_tensor >= THRESHOLD).numpy().astype(bool)
                    binary_tile_cropped = binary_pred_padded[:orig_h, :orig_w]

                    x_start_hdf5 = coords['x_inf']
                    y_start_hdf5 = coords['y_inf']
                    y_end_hdf5 = min(y_start_hdf5 + orig_h, output_shape[0])
                    x_end_hdf5 = min(x_start_hdf5 + orig_w, output_shape[1])

                    if y_end_hdf5 > y_start_hdf5 and x_end_hdf5 > x_start_hdf5:
                        hdf5_h = y_end_hdf5 - y_start_hdf5
                        hdf5_w = x_end_hdf5 - x_start_hdf5
                        dset[y_start_hdf5:y_end_hdf5, x_start_hdf5:x_end_hdf5] = binary_tile_cropped[:hdf5_h, :hdf5_w]

            except Exception as final_batch_err: print(f"Chyba při zpracování poslední dávky: {final_batch_err}")
            finally:
                batch_tiles_data.clear()
                batch_coords.clear()

    print("\nZpracování dokončeno.")
    print(f"   Celkem dlaždic v mřížce DZG: {total_tiles_grid}")
    print(f"   Dlaždic zpracováno modelem: {processed_tiles_for_model}")
    print(f"   Dlaždic přeskočeno maskou: {tiles_skipped_by_mask}")

except ValueError as e: print(f"Chyba konfigurace nebo úrovně: {e}")
except FileNotFoundError as e: print(f"Chyba souboru: {e}")
except Exception as main_err: print(f"\nNeočekávaná chyba během zpracování: {main_err}")
finally:
    # Bezpečné uzavření souborů
    if hdf5_file:
        try: hdf5_file.close(); print("HDF5 soubor uzavřen.")
        except Exception as close_err: print(f"Chyba při uzavírání HDF5: {close_err}")
    if wsi:
        wsi.close(); print("WSI soubor uzavřen.")

    end_time = time.time()
    print(f"\nPredikovaná maska byla uložena do {output_hdf5_path}.")
    print(f"Skript (Filtrováno maskou tkáně z lvl {tissue_mask_level_index}) běžel {end_time - start_time:.2f} sekund.")

    # Uvolnění paměti
    del batch_tiles_data
    del batch_coords
    if 'batch_tensor' in locals(): del batch_tensor
    if 'predictions' in locals(): del predictions
    if 'tissue_mask_np' in locals(): del tissue_mask_np
    gc.collect()
    print("Paměť uvolněna.")

# --- END OF FILE batch_inference_tissue_mask_filter_lvl6.py ---