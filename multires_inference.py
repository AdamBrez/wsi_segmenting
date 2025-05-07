# --- START OF FILE batch_inference_manual_overlap_multires.py ---

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
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import numpy as np
import openslide
import time
import segmentation_models_pytorch as smp
import datetime
import math
import gc

"""
    Načítá WSI a provádí inferenci s MANUÁLNĚ ŘÍZENÝM PŘEKRYVEM
    a MULTI-RESOLUTION vstupem (high-res + low-res kontext).
    Nepoužívá DeepZoomGenerator. Iteruje s krokem STEP.
    Inference je filtrována maskou tkáně. Zpracování v DÁVKÁCH.
    Okrajové dlaždice jsou PADOVÁNY. Používá se normalizace.
    Predikce z překryvů jsou zprůměrovány.
    Výstup: finální binární maska (HDF5, bool).
"""

# --- Konfigurace ---
model_weights_path = r"C:\Users\USER\Desktop\weights\multi_res_dice_loss_e50_11200len.pth" # Zkontroluj váhy!
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_061.tif"
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\pred_061_manual_overlap_multires.h5" # Nový název!

# <<< Konfigurace manuálního overlapu a velikostí >>>
TILE_SIZE = 256           # Velikost high-res dlaždice (a vstupu modelu)
CONTEXT_TILE_SIZE = 256   # Velikost low-res kontextové dlaždice (a vstupu modelu)
OVERLAP_PX = 32           # Počet pixelů překryvu
STEP = TILE_SIZE - OVERLAP_PX
if STEP <= 0: raise ValueError("Krok (TILE_SIZE - OVERLAP_PX) musí být pozitivní.")
print(f"Manuální overlap: Tile Size = {TILE_SIZE}, Overlap = {OVERLAP_PX}, Step = {STEP}")
# <<< Konec Konfigurace manuálního overlapu >>>

# <<< Konfigurace úrovní a filtrování >>>
INFERENCE_LEVEL_INDEX = 2 # Úroveň OpenSlide pro high-res dlaždici
CONTEXT_LEVEL_INDEX = 3   # Úroveň OpenSlide pro low-res kontextovou dlaždici
tissue_mask_dir = r"C:\Users\USER\Desktop\colab_unet\masky_new"
tissue_mask_level_index = 6 # Úroveň OpenSlide, ze které byla maska tkáně generována
TISSUE_THRESHOLD = 0.1    # Minimální podíl tkáně
# <<< Konec Konfigurace úrovní a filtrování >>>

BATCH_SIZE = 32 # Zvaž snížení BATCH_SIZE kvůli dvojnásobnému vstupu a akumulačním polím
THRESHOLD = 0.5 # Práh pro finální binární masku

# --- Inicializace ---
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
print(f"Model očekává vstup: High-res {TILE_SIZE}x{TILE_SIZE} (L{INFERENCE_LEVEL_INDEX}), Low-res {CONTEXT_TILE_SIZE}x{CONTEXT_TILE_SIZE} (L{CONTEXT_LEVEL_INDEX})")

# Načtení modelu
# Předpokládáme, že model očekává 6 vstupních kanálů (3 z high-res, 3 z low-res)
model = smp.Unet("resnet34", encoder_weights=None, in_channels=6, classes=1) # <<< Pozor na in_channels=6 >>>
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

# Normalizační transformace (předpokládáme stejnou pro oba vstupy)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Načtení WSI, masky a inicializace
wsi = None
tissue_mask_np = None
hdf5_file = None
prediction_sum = None
prediction_count = None

try:
    wsi = openslide.OpenSlide(wsi_image_path)
    print(f"WSI načteno: {wsi_image_path}")

    # Ověření existence úrovní
    required_levels = [tissue_mask_level_index, INFERENCE_LEVEL_INDEX, CONTEXT_LEVEL_INDEX]
    for lvl in required_levels:
        if lvl >= wsi.level_count:
            raise ValueError(f"Požadovaná úroveň WSI ({lvl}) neexistuje (max index {wsi.level_count - 1}).")

    # Získání parametrů úrovní
    tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
    inference_downsample = wsi.level_downsamples[INFERENCE_LEVEL_INDEX]
    context_downsample = wsi.level_downsamples[CONTEXT_LEVEL_INDEX]
    inference_level_dims = wsi.level_dimensions[INFERENCE_LEVEL_INDEX]
    native_dims = wsi.level_dimensions[0]
    output_shape = (inference_level_dims[1], inference_level_dims[0]) # Výstup odpovídá inference levelu

    print(f"Inference Level (High-Res): {INFERENCE_LEVEL_INDEX} (DS: {inference_downsample:.2f}x, Dims: {inference_level_dims})")
    print(f"Context Level (Low-Res): {CONTEXT_LEVEL_INDEX} (DS: {context_downsample:.2f}x)")
    print(f"Tissue Mask Level: {tissue_mask_level_index} (DS: {tissue_mask_downsample:.2f}x)")

    # Načtení masky tkáně
    wsi_filename_base = os.path.splitext(os.path.basename(wsi_image_path))[0]
    tissue_mask_filename = "mask_061.npy" # Pevně nastaveno
    tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
    if not os.path.exists(tissue_mask_full_path):
        raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
    tissue_mask_np = np.load(tissue_mask_full_path).astype(bool)
    print(f"Maska tkáně načtena, tvar: {tissue_mask_np.shape}")

    # Výpočet škálovacího faktoru a mřížky
    scale_factor_tissue = tissue_mask_downsample / inference_downsample
    if scale_factor_tissue <= 0: raise ValueError("Neplatný scale_factor_tissue.")
    print(f"Škálovací faktor (Maska / Inference): {scale_factor_tissue:.3f}")

    output_w, output_h = inference_level_dims
    cols = math.ceil(output_w / STEP)
    rows = math.ceil(output_h / STEP)
    total_tiles_grid = rows * cols
    print(f"Počet dlaždic v mřížce (krok {STEP}): {cols}x{rows} = {total_tiles_grid}")

    # Alokace akumulačních polí
    print(f"Alokuji akumulační pole {output_shape}...")
    prediction_sum = np.zeros(output_shape, dtype=np.float32)
    prediction_count = np.zeros(output_shape, dtype=np.uint16)
    print("Akumulační pole alokována.")


    # --- Zpracování ---
    processed_tiles_for_model = 0
    tiles_skipped_by_mask = 0
    # Seznam pro KOMBINOVANÉ tensory [6, H, W]
    batch_combined_data = []
    # Seznam pro souřadnice a velikosti HIGH-RES dlaždice (pro akumulaci)
    batch_coords_hr = []

    with torch.inference_mode():
        for r in tqdm(range(rows), desc="Processing rows"):
            for c in range(cols):
                # Výpočet souřadnic levého horního rohu HIGH-RES dlaždice na inference úrovni
                x_inf_start = c * STEP
                y_inf_start = r * STEP

                # Odhad velikosti pro kontrolu masky tkáně
                current_tile_w = min(TILE_SIZE, output_w - x_inf_start)
                current_tile_h = min(TILE_SIZE, output_h - y_inf_start)
                if current_tile_w <= 0 or current_tile_h <= 0: continue

                # --- Filtrování podle masky tkáně ---
                # (Logika zůstává stejná jako předtím)
                tm_x_start = int(x_inf_start / scale_factor_tissue); tm_y_start = int(y_inf_start / scale_factor_tissue)
                tm_w = max(1, int(current_tile_w / scale_factor_tissue)); tm_h = max(1, int(current_tile_h / scale_factor_tissue))
                tm_y_end = min(tm_y_start + tm_h, tissue_mask_np.shape[0]); tm_x_end = min(tm_x_start + tm_w, tissue_mask_np.shape[1])
                tm_y_start_clipped = max(0, tm_y_start); tm_x_start_clipped = max(0, tm_x_start)
                tissue_ratio = 0.0
                if tm_y_end > tm_y_start_clipped and tm_x_end > tm_x_start_clipped:
                    tissue_region = tissue_mask_np[tm_y_start_clipped:tm_y_end, tm_x_start_clipped:tm_x_end]
                    if tissue_region.size > 0: tissue_ratio = np.mean(tissue_region)

                # --- Podmíněná inference ---
                if tissue_ratio >= TISSUE_THRESHOLD:
                    try:
                        # --- 1. Načtení HIGH-RES dlaždice ---
                        x_hr_l0 = int(x_inf_start * inference_downsample)
                        y_hr_l0 = int(y_inf_start * inference_downsample)
                        hr_tile_pil = wsi.read_region((x_hr_l0, y_hr_l0), INFERENCE_LEVEL_INDEX, (TILE_SIZE, TILE_SIZE)).convert("RGB")
                        hr_w_orig, hr_h_orig = hr_tile_pil.size # Skutečná velikost high-res
                        if hr_w_orig == 0 or hr_h_orig == 0: continue

                        # --- 2. Načtení LOW-RES kontextové dlaždice ---
                        # (Použijeme logiku centrování z tréninkového datasetu)
                        x_inf_center = x_inf_start + hr_w_orig / 2
                        y_inf_center = y_inf_start + hr_h_orig / 2
                        center_native_x = x_inf_center * inference_downsample
                        center_native_y = y_inf_center * inference_downsample
                        ctx_native_w = CONTEXT_TILE_SIZE * context_downsample
                        ctx_native_h = CONTEXT_TILE_SIZE * context_downsample
                        x_lr_l0 = int(round(center_native_x - ctx_native_w / 2))
                        y_lr_l0 = int(round(center_native_y - ctx_native_h / 2))
                        x_lr_l0 = max(0, min(x_lr_l0, native_dims[0] - int(round(ctx_native_w))))
                        y_lr_l0 = max(0, min(y_lr_l0, native_dims[1] - int(round(ctx_native_h))))

                        lr_tile_pil = wsi.read_region((x_lr_l0, y_lr_l0), CONTEXT_LEVEL_INDEX, (CONTEXT_TILE_SIZE, CONTEXT_TILE_SIZE)).convert("RGB")
                        lr_w_orig, lr_h_orig = lr_tile_pil.size # Skutečná velikost low-res
                        if lr_w_orig == 0 or lr_h_orig == 0: continue # Přeskočit, pokud kontext nelze načíst

                        # --- 3. Padding obou dlaždic ---
                        hr_tile_padded = hr_tile_pil
                        if hr_w_orig < TILE_SIZE or hr_h_orig < TILE_SIZE:
                            delta_w = max(0, TILE_SIZE - hr_w_orig); delta_h = max(0, TILE_SIZE - hr_h_orig)
                            hr_tile_padded = ImageOps.expand(hr_tile_pil, (0, 0, delta_w, delta_h), fill=0)

                        lr_tile_padded = lr_tile_pil
                        if lr_w_orig < CONTEXT_TILE_SIZE or lr_h_orig < CONTEXT_TILE_SIZE:
                            delta_w = max(0, CONTEXT_TILE_SIZE - lr_w_orig); delta_h = max(0, CONTEXT_TILE_SIZE - lr_h_orig)
                            lr_tile_padded = ImageOps.expand(lr_tile_pil, (0, 0, delta_w, delta_h), fill=0)

                        # --- 4. Transformace a spojení ---
                        hr_tensor = normalize(to_tensor(hr_tile_padded)) # [3, TILE_SIZE, TILE_SIZE]
                        lr_tensor = normalize(to_tensor(lr_tile_padded)) # [3, CONTEXT_TILE_SIZE, CONTEXT_TILE_SIZE]

                        # Pokud CONTEXT_TILE_SIZE != TILE_SIZE, bylo by nutné lr_tensor resizovat
                        if hr_tensor.shape[1:] != lr_tensor.shape[1:]:
                            raise NotImplementedError("Resize kontextového tensoru není implementován - velikosti musí být stejné pro cat.")
                            # Sem by přišel torch.nn.functional.interpolate nebo TF.resize

                        combined_tensor = torch.cat((hr_tensor, lr_tensor), dim=0) # [6, TILE_SIZE, TILE_SIZE]

                        # --- 5. Přidání do batche ---
                        batch_combined_data.append(combined_tensor)
                        # Uložíme souřadnice a PŮVODNÍ velikost HIGH-RES dlaždice pro akumulaci
                        batch_coords_hr.append({'x_inf': x_inf_start, 'y_inf': y_inf_start,
                                                'w_orig': hr_w_orig, 'h_orig': hr_h_orig})
                        processed_tiles_for_model += 1

                    except openslide.OpenSlideError as ose: print(f"OpenSlide chyba při čtení dlaždice/kontextu [{c},{r}]: {ose}"); continue
                    except Exception as prep_err:
                         print(f"Chyba při přípravě dlaždice/kontextu [{c},{r}]: {prep_err}")
                         # Pokud se souřadnice přidaly, ale data ne, odstranit souřadnice
                         if len(batch_coords_hr) > len(batch_combined_data): batch_coords_hr.pop()
                         continue
                else:
                    tiles_skipped_by_mask += 1
                    continue # Přeskočit inferenci

                # --- Zpracování dávky (pokud je plná) ---
                if len(batch_combined_data) == BATCH_SIZE:
                    if not batch_combined_data: continue

                    try:
                        # Spojení tensorů do dávky [B, 6, H, W]
                        batch_tensor = torch.stack(batch_combined_data).to(device)
                        # Inference
                        prediction_output = model(batch_tensor) # Očekává [B, 6, H, W]
                        # Získání pravděpodobností pro HIGH-RES výstup
                        predictions = torch.sigmoid(prediction_output).cpu().numpy() # [B, 1, H, W]

                        # --- Akumulace výsledků dávky ---
                        for i in range(predictions.shape[0]):
                            pred_prob_map = predictions[i, 0, :, :] # [TILE_SIZE, TILE_SIZE]
                            coords = batch_coords_hr[i] # Používáme souřadnice high-res
                            x_inf, y_inf = coords['x_inf'], coords['y_inf']
                            w_orig, h_orig = coords['w_orig'], coords['h_orig'] # Pův. velikost high-res

                            # Logika akumulace (stejná jako v předchozí verzi)
                            y_start_target = y_inf; x_start_target = x_inf
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

                    except Exception as batch_proc_err: print(f"Chyba při zpracování dávky: {batch_proc_err}")
                    finally:
                        batch_combined_data.clear()
                        batch_coords_hr.clear()

        # --- Zpracování poslední (neúplné) dávky ---
        if batch_combined_data:
            print(f"Zpracovávám poslední dávku ({len(batch_combined_data)} dlaždic)...")
            try:
                batch_tensor = torch.stack(batch_combined_data).to(device)
                prediction_output = model(batch_tensor)
                predictions = torch.sigmoid(prediction_output).cpu().numpy()

                for i in range(predictions.shape[0]):
                    pred_prob_map = predictions[i, 0, :, :]
                    coords = batch_coords_hr[i]
                    x_inf, y_inf = coords['x_inf'], coords['y_inf']
                    w_orig, h_orig = coords['w_orig'], coords['h_orig']

                    y_start_target = y_inf; x_start_target = x_inf
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
                batch_combined_data.clear()
                batch_coords_hr.clear()

    print("\nZpracování dlaždic dokončeno.")
    print(f"   Celkem dlaždic v mřížce: {total_tiles_grid}")
    print(f"   Dlaždic zpracováno modelem: {processed_tiles_for_model}")
    print(f"   Dlaždic přeskočeno maskou: {tiles_skipped_by_mask}")

    # --- Výpočet finální masky ---
    print("Průměrování pravděpodobností z překryvů...")
    average_probability = np.zeros_like(prediction_sum, dtype=np.float32)
    np.divide(prediction_sum, prediction_count, out=average_probability, where=prediction_count > 0)
    print(f"Prahování s thresholdem {THRESHOLD}...")
    final_mask = (average_probability >= THRESHOLD)

    # --- Uložení do HDF5 ---
    print(f"Ukládání finální masky do {output_hdf5_path}...")
    hdf5_file = h5py.File(output_hdf5_path, "w")
    dset = hdf5_file.create_dataset(
        "mask", shape=output_shape, dtype=bool, data=final_mask,
        chunks=(TILE_SIZE, TILE_SIZE), compression="gzip"
    )
    # Zápis metadat (podobně jako předtím, přidáme info o kontextu)
    hdf5_file.attrs["day_of_creation"] = datetime.datetime.now().isoformat()
    hdf5_file.attrs["tile_size_high_res"] = TILE_SIZE
    hdf5_file.attrs["tile_size_context"] = CONTEXT_TILE_SIZE
    hdf5_file.attrs["overlap_px"] = OVERLAP_PX
    hdf5_file.attrs["step_px"] = STEP
    hdf5_file.attrs["batch_size"] = BATCH_SIZE
    hdf5_file.attrs["model_weights"] = os.path.basename(model_weights_path)
    hdf5_file.attrs["wsi_level_high_res"] = INFERENCE_LEVEL_INDEX
    hdf5_file.attrs["wsi_level_context"] = CONTEXT_LEVEL_INDEX
    hdf5_file.attrs["wsi_level_dims"] = f"{inference_level_dims[0]}x{inference_level_dims[1]}"
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
    print(f"Skript (Manuální overlap, Multi-Res, Filtrováno) běžel {end_time - start_time:.2f} sekund.")

    # Uvolnění paměti
    del batch_combined_data
    del batch_coords_hr
    if 'batch_tensor' in locals(): del batch_tensor
    if 'predictions' in locals(): del predictions
    if 'tissue_mask_np' in locals(): del tissue_mask_np
    if 'prediction_sum' in locals(): del prediction_sum
    if 'prediction_count' in locals(): del prediction_count
    if 'average_probability' in locals(): del average_probability
    if 'final_mask' in locals(): del final_mask
    gc.collect()
    print("Paměť uvolněna.")

# --- END OF FILE batch_inference_manual_overlap_multires.py ---