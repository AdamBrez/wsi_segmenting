# --- START OF FILE batch_inference_manual_overlap.py ---

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
from PIL import Image, ImageOps # Potřebujeme Image pro read_region
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import numpy as np
import openslide
# DeepZoomGenerator již nepotřebujeme
import time
import segmentation_models_pytorch as smp
import datetime
import math
import gc

"""
    Načítá WSI a provádí inferenci s MANUÁLNĚ ŘÍZENÝM PŘEKRYVEM.
    Nepoužívá DeepZoomGenerator pro tvorbu dlaždic.
    Načítají se regiony TILE_SIZE x TILE_SIZE s krokem STEP.
    Inference je prováděna POUZE na dlaždicích překrývajících tkáň
    podle nízko-rozlišovací masky (generované z WSI level 6).
    Zpracování v DÁVKÁCH. Okrajové dlaždice jsou PADOVÁNY. Používá se normalizace.
    Predikce z překryvů jsou zprůměrovány pomocí akumulačních polí.
    Výstupem je finální binární maska (HDF5, bool).
"""

# --- Konfigurace ---
model_weights_path = r"C:\Users\USER\Desktop\weights\finetuned_e20_11200len.pth"
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_068.tif"
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\pred_068_finetuned_overlap.h5" # Nový název!

# <<< Konfigurace manuálního overlapu >>>
TILE_SIZE = 256     # Velikost, kterou očekává model
OVERLAP_PX = 32     # Počet pixelů překryvu mezi dlaždicemi
STEP = TILE_SIZE - OVERLAP_PX # Krok posunu pro další dlaždici
if STEP <= 0: raise ValueError("Krok (TILE_SIZE - OVERLAP_PX) musí být pozitivní.")
print(f"Manuální overlap: Tile Size = {TILE_SIZE}, Overlap = {OVERLAP_PX}, Step = {STEP}")
# <<< Konec Konfigurace manuálního overlapu >>>

# <<< Konfigurace filtrování podle masky tkáně >>>
tissue_mask_dir = r"C:\Users\USER\Desktop\colab_unet\masky_new"
tissue_mask_level_index = 6 # Úroveň OpenSlide, ze které byla maska generována
TISSUE_THRESHOLD = 0.1 # Minimální podíl tkáně
# <<< Konec Konfigurace pro filtrování >>>

BATCH_SIZE = 32 # Možná bude potřeba snížit kvůli akumulačním polím v RAM
THRESHOLD = 0.5 # Práh pro finální binární masku
TARGET_INFERENCE_LEVEL = 2 # Přímo index OpenSlide úrovně pro inferenci (0 = nejvyšší)

# --- Inicializace ---
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
print(f"Model očekává vstup: {TILE_SIZE}x{TILE_SIZE}")

# Načtení modelu
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
try:
    if not os.path.exists(model_weights_path): raise FileNotFoundError(f"Váhy nenalezeny: {model_weights_path}")
    try: 
        model_and_weights = torch.load(model_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(model_and_weights["model_state_dict"])
    except Exception as e:
         print(f"Varování {e}: weights_only není podporováno, načítám standardně.")
         model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
except Exception as e: print(f"Chyba při načítání vah modelu: {e}"); exit()
model.to(device)
model.eval()

# Normalizační transformace
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Načtení WSI, masky a inicializace
wsi = None
tissue_mask_np = None
hdf5_file = None
prediction_sum = None # Pro finally blok
prediction_count = None # Pro finally blok

try:
    wsi = openslide.OpenSlide(wsi_image_path)
    print(f"WSI načteno: {wsi_image_path}")

    # Ověření existence úrovní
    if tissue_mask_level_index >= wsi.level_count:
        raise ValueError(f"Úroveň masky tkáně ({tissue_mask_level_index}) neexistuje.")
    if TARGET_INFERENCE_LEVEL >= wsi.level_count:
        raise ValueError(f"Cílová úroveň inference ({TARGET_INFERENCE_LEVEL}) neexistuje.")

    # Získání parametrů úrovní
    tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
    inference_level_index = TARGET_INFERENCE_LEVEL
    inference_downsample = wsi.level_downsamples[inference_level_index]
    inference_level_dims = wsi.level_dimensions[inference_level_index]
    output_shape = (inference_level_dims[1], inference_level_dims[0]) # (výška, šířka)

    print(f"Inference na úrovni: {inference_level_index} (Downsample: {inference_downsample:.2f}x, Rozměry: {inference_level_dims})")
    print(f"Maska tkáně z úrovně: {tissue_mask_level_index} (Downsample: {tissue_mask_downsample:.2f}x)")

    # Načtení masky tkáně
    wsi_filename_base = os.path.splitext(os.path.basename(wsi_image_path))[0]
    tissue_mask_filename = "mask_068.npy" # Pevně nastaveno, jak bylo v předchozím kódu
    tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
    if not os.path.exists(tissue_mask_full_path):
        raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
    tissue_mask_np = np.load(tissue_mask_full_path).astype(bool) # Zajistíme bool typ
    print(f"Maska tkáně načtena z: {tissue_mask_full_path}, tvar: {tissue_mask_np.shape}")

    # Výpočet škálovacího faktoru a mřížky
    scale_factor = tissue_mask_downsample / inference_downsample
    if scale_factor <= 0: raise ValueError("Neplatný scale_factor.")
    print(f"Škálovací faktor (Maska / Inference): {scale_factor:.3f}")

    output_w, output_h = inference_level_dims
    cols = math.ceil(output_w / STEP)
    rows = math.ceil(output_h / STEP)
    total_tiles_grid = rows * cols
    print(f"Počet dlaždic v mřížce (krok {STEP}): {cols}x{rows} = {total_tiles_grid}")

    # Alokace akumulačních polí
    print(f"Alokuji akumulační pole {output_shape}...")
    prediction_sum = np.zeros(output_shape, dtype=np.float32)
    prediction_count = np.zeros(output_shape, dtype=np.uint16) # uint16 pro více překryvů
    print("Akumulační pole alokována.")


    # --- Zpracování ---
    processed_tiles_for_model = 0
    tiles_skipped_by_mask = 0
    batch_tiles_data = [] # Tenzory pro model
    batch_coords = []     # Souřadnice pro akumulaci a původní velikosti

    with torch.inference_mode():
        for r in tqdm(range(rows), desc="Processing rows"):
            for c in range(cols):
                # Výpočet souřadnic levého horního rohu pro čtení na inference úrovni
                x_inf_start = c * STEP
                y_inf_start = r * STEP

                # Odhad velikosti dlaždice na inference úrovni (pro masku tkáně)
                # Může být menší než TILE_SIZE na okraji WSI
                current_tile_w = min(TILE_SIZE, output_w - x_inf_start)
                current_tile_h = min(TILE_SIZE, output_h - y_inf_start)
                if current_tile_w <= 0 or current_tile_h <= 0: continue # Mimo obraz

                # --- Filtrování podle masky tkáně ---
                tm_x_start = int(x_inf_start / scale_factor)
                tm_y_start = int(y_inf_start / scale_factor)
                tm_w = max(1, int(current_tile_w / scale_factor))
                tm_h = max(1, int(current_tile_h / scale_factor))

                tm_y_end = min(tm_y_start + tm_h, tissue_mask_np.shape[0])
                tm_x_end = min(tm_x_start + tm_w, tissue_mask_np.shape[1])
                tm_y_start_clipped = max(0, tm_y_start)
                tm_x_start_clipped = max(0, tm_x_start)

                tissue_ratio = 0.0
                if tm_y_end > tm_y_start_clipped and tm_x_end > tm_x_start_clipped:
                    tissue_region = tissue_mask_np[tm_y_start_clipped:tm_y_end, tm_x_start_clipped:tm_x_end]
                    if tissue_region.size > 0: tissue_ratio = np.mean(tissue_region)

                # --- Podmíněná inference ---
                if tissue_ratio >= TISSUE_THRESHOLD:
                    try:
                        # Přepočet souřadnic pro čtení na level 0
                        x_level0 = int(x_inf_start * inference_downsample)
                        y_level0 = int(y_inf_start * inference_downsample)
                        read_size = (TILE_SIZE, TILE_SIZE) # Vždy čteme TILE_SIZE

                        # Načtení regionu TILE_SIZE x TILE_SIZE
                        tile_pil = wsi.read_region((x_level0, y_level0), inference_level_index, read_size)
                        tile_rgb = tile_pil.convert("RGB")
                        tile_w_orig, tile_h_orig = tile_rgb.size # Skutečná načtená velikost

                        if tile_w_orig == 0 or tile_h_orig == 0: continue

                        # Padding, pokud read_region vrátilo menší tile (na okraji)
                        tile_to_process = tile_rgb
                        if tile_w_orig < TILE_SIZE or tile_h_orig < TILE_SIZE:
                            delta_w = max(0, TILE_SIZE - tile_w_orig)
                            delta_h = max(0, TILE_SIZE - tile_h_orig)
                            padding = (0, 0, delta_w, delta_h)
                            tile_to_process = ImageOps.expand(tile_rgb, padding, fill=0)

                        if tile_to_process.size != (TILE_SIZE, TILE_SIZE):
                             print(f"Chyba: Velikost po paddingu není {TILE_SIZE}x{TILE_SIZE} ({tile_to_process.size}) pro [{c},{r}]. Přeskakuji.")
                             continue

                        # Tensorizace a normalizace
                        tile_tensor = to_tensor(tile_to_process)
                        tile_tensor_normalized = normalize(tile_tensor)

                        batch_tiles_data.append(tile_tensor_normalized)
                        # Ukládáme souřadnice na inference úrovni (x_inf, y_inf) a
                        # SKUTEČNOU velikost načtené dlaždice PŘED paddingem (tile_w_orig, tile_h_orig)
                        batch_coords.append({'x_inf': x_inf_start, 'y_inf': y_inf_start,
                                             'w_orig': tile_w_orig, 'h_orig': tile_h_orig,
                                             'col': c, 'row': r}) # Pro debug
                        processed_tiles_for_model += 1

                    except openslide.OpenSlideError as ose: print(f"OpenSlide chyba při čtení regionu [{c},{r}]: {ose}"); continue
                    except Exception as prep_err:
                         print(f"Chyba při přípravě dlaždice [{c},{r}]: {prep_err}")
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
                        prediction_output = model(batch_tensor) # Logits
                        predictions = torch.sigmoid(prediction_output).cpu().numpy() # Pravděpodobnosti [B, 1, TILE_SIZE, TILE_SIZE]

                        # --- Akumulace výsledků dávky ---
                        for i in range(predictions.shape[0]):
                            pred_prob_map = predictions[i, 0, :, :] # [TILE_SIZE, TILE_SIZE]
                            coords = batch_coords[i]
                            x_inf, y_inf = coords['x_inf'], coords['y_inf']
                            w_orig, h_orig = coords['w_orig'], coords['h_orig'] # Skutečná velikost před paddingem

                            # Logika akumulace (z v4): Přičteme jen relevantní část predikce
                            y_start_target = y_inf
                            x_start_target = x_inf
                            # Konec je dán skutečnou velikostí dlaždice, oříznutý hranicemi pole
                            y_end_orig = min(y_start_target + h_orig, output_shape[0])
                            x_end_orig = min(x_start_target + w_orig, output_shape[1])

                            # Rozměry části predikce k přidání (omezeno TILE_SIZE a skutečnou velikostí)
                            h_to_add = min(TILE_SIZE, y_end_orig - y_start_target)
                            w_to_add = min(TILE_SIZE, x_end_orig - x_start_target)

                            if h_to_add > 0 and w_to_add > 0:
                                # Vezmeme výřez z predikce (velikosti h_to_add x w_to_add)
                                pred_slice_to_add = pred_prob_map[:h_to_add, :w_to_add]

                                # Cílové souřadnice v akumulačním poli
                                y_end_target = y_start_target + h_to_add
                                x_end_target = x_start_target + w_to_add

                                # Přičtení k sumě a inkrementace počtu
                                prediction_sum[y_start_target:y_end_target, x_start_target:x_end_target] += pred_slice_to_add
                                prediction_count[y_start_target:y_end_target, x_start_target:x_end_target] += 1

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
                predictions = torch.sigmoid(prediction_output).cpu().numpy()

                for i in range(predictions.shape[0]):
                    pred_prob_map = predictions[i, 0, :, :]
                    coords = batch_coords[i]
                    x_inf, y_inf = coords['x_inf'], coords['y_inf']
                    w_orig, h_orig = coords['w_orig'], coords['h_orig']

                    y_start_target = y_inf
                    x_start_target = x_inf
                    y_end_orig = min(y_start_target + h_orig, output_shape[0])
                    x_end_orig = min(x_start_target + w_orig, output_shape[1])

                    h_to_add = min(TILE_SIZE, y_end_orig - y_start_target)
                    w_to_add = min(TILE_SIZE, x_end_orig - x_start_target)

                    if h_to_add > 0 and w_to_add > 0:
                        pred_slice_to_add = pred_prob_map[:h_to_add, :w_to_add]
                        y_end_target = y_start_target + h_to_add
                        x_end_target = x_start_target + w_to_add
                        prediction_sum[y_start_target:y_end_target, x_start_target:x_end_target] += pred_slice_to_add
                        prediction_count[y_start_target:y_end_target, x_start_target:x_end_target] += 1

            except Exception as final_batch_err: print(f"Chyba při zpracování poslední dávky: {final_batch_err}")
            finally:
                batch_tiles_data.clear()
                batch_coords.clear()

    print("\nZpracování dlaždic dokončeno.")
    print(f"   Celkem dlaždic v mřížce: {total_tiles_grid}")
    print(f"   Dlaždic zpracováno modelem: {processed_tiles_for_model}")
    print(f"   Dlaždic přeskočeno maskou: {tiles_skipped_by_mask}")

    # --- Výpočet finální masky ---
    print("Průměrování pravděpodobností z překryvů...")
    average_probability = np.zeros_like(prediction_sum, dtype=np.float32)
    # Dělení s ošetřením dělení nulou
    np.divide(prediction_sum, prediction_count, out=average_probability, where=prediction_count > 0)

    print(f"Prahování s thresholdem {THRESHOLD}...")
    final_mask = (average_probability >= THRESHOLD)

    # --- Uložení do HDF5 ---
    print(f"Ukládání finální masky do {output_hdf5_path}...")
    hdf5_file = h5py.File(output_hdf5_path, "w")
    dset = hdf5_file.create_dataset(
        "mask",
        shape=output_shape,
        dtype=bool,
        data=final_mask, # Uložíme finální binární masku
        chunks=(TILE_SIZE, TILE_SIZE), # Chunk může být TILE_SIZE nebo STEP
        compression="gzip"
    )
    # Zápis původních metadat
    hdf5_file.attrs["day_of_creation"] = datetime.datetime.now().isoformat()
    # <<< AKTUALIZOVANÁ METADATA OPROTI PŮVODNÍMU SKRIPTU BEZ OVERLAPU >>>
    hdf5_file.attrs["tile_size_model"] = TILE_SIZE # Velikost vstupu modelu
    hdf5_file.attrs["overlap_px"] = OVERLAP_PX  # Manuálně nastavený overlap
    hdf5_file.attrs["step_px"] = STEP          # Krok iterace
    # <<< KONEC AKTUALIZOVANÝCH METADAT >>>
    hdf5_file.attrs["batch_size"] = BATCH_SIZE
    hdf5_file.attrs["model_weights"] = os.path.basename(model_weights_path)
    hdf5_file.attrs["wsi_level_processed"] = f"OS Level {inference_level_index}"
    hdf5_file.attrs["wsi_level_dims"] = f"{inference_level_dims[0]}x{inference_level_dims[1]}"
    hdf5_file.attrs["augmentation"] = "False" # Protože je to inference
    # Metadata o filtrování (volitelné)
    hdf5_file.attrs["tissue_mask_filtering"] = True
    hdf5_file.attrs["tissue_mask_level_index"] = tissue_mask_level_index
    hdf5_file.attrs["tissue_threshold"] = TISSUE_THRESHOLD

    print("\nUkládání dokončeno.")

except ValueError as e: print(f"Chyba konfigurace nebo úrovně: {e}")
except FileNotFoundError as e: print(f"Chyba souboru: {e}")
except MemoryError: print("Chyba: Nedostatek paměti pro akumulační pole.")
except Exception as main_err: print(f"\nNeočekávaná chyba během zpracování: {main_err}")
finally:
    # Bezpečné uzavření souborů a uvolnění paměti
    if hdf5_file:
        try: hdf5_file.close(); print("HDF5 soubor uzavřen.")
        except Exception as close_err: print(f"Chyba při uzavírání HDF5: {close_err}")
    if wsi:
        wsi.close(); print("WSI soubor uzavřen.")

    end_time = time.time()
    print(f"\nPredikovaná maska byla uložena do {output_hdf5_path}.")
    print(f"Skript (Manuální overlap, Filtrováno) běžel {end_time - start_time:.2f} sekund.")

    # Uvolnění paměti
    del batch_tiles_data
    del batch_coords
    if 'batch_tensor' in locals(): del batch_tensor
    if 'predictions' in locals(): del predictions
    if 'tissue_mask_np' in locals(): del tissue_mask_np
    if 'prediction_sum' in locals(): del prediction_sum
    if 'prediction_count' in locals(): del prediction_count
    if 'average_probability' in locals(): del average_probability
    if 'final_mask' in locals(): del final_mask
    gc.collect()
    print("Paměť uvolněna.")

# --- END OF FILE batch_inference_manual_overlap.py ---