# --- START OF FINAL MODIFIED FILE ---

import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import glob
import time
import datetime
import math
import gc

# !!! Zajistěte správnou cestu k OpenSlide !!!



import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import numpy as np
import openslide
import segmentation_models_pytorch as smp
from model import UNet

from model import UNet

# --- HLAVNÍ KONFIGURACE ---
# !!! ZDE NASTAVTE POTŘEBNÉ CESTY !!!

# Cesta ke složce, která obsahuje vstupní WSI soubory (test_XXX.tif)
WSI_INPUT_DIR = r"F:\x" 

# Cesta ke složce, která obsahuje MASKY TKÁNĚ (mask_XXX.npy)
TISSUE_MASK_DIR = r"F:\histology_lungs\histology_lungs\converted\valid"

# Cesta k výstupní složce, kam se uloží výsledné predikce (pred_XXX.h5)
OUTPUT_PRED_DIR = r"C:\Users\USER\Desktop\test_preds"  

# Cesta k souboru s naučenými váhami modelu
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-30_02-58-50\best_weights_2025-05-30_02-58-50.pth"

# <<< Konfigurace manuálního overlapu >>>
TILE_SIZE = 256
OVERLAP_PX = 32
STEP = TILE_SIZE - OVERLAP_PX
if STEP <= 0: raise ValueError("Krok (TILE_SIZE - OVERLAP_PX) musí být pozitivní.")

PATCH_CAMELYON_MEAN = [0.702, 0.546, 0.696]
PATCH_CAMELYON_STD = [0.239, 0.282, 0.216]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# <<< Konfigurace filtrování podle masky tkáně >>>
tissue_mask_level_index = 3
TISSUE_THRESHOLD = 0.1

# <<< Ostatní konfigurace >>>
BATCH_SIZE = 64
THRESHOLD = 0.5
TARGET_INFERENCE_LEVEL = 0

# --- FUNKCE PRO ZPRACOVÁNÍ JEDNOHO WSI ---
def process_single_wsi(wsi_image_path, output_hdf5_path, tissue_mask_dir, model, device):
    """
    Zpracuje jeden WSI soubor a uloží výslednou masku.
    """
    script_start_time = time.time()
    print("-" * 80)
    print(f"Zahajuji zpracování: {os.path.basename(wsi_image_path)}")
    print(f"Hledám masku tkáně ve složce: {tissue_mask_dir}")
    print(f"Výstup bude uložen do: {output_hdf5_path}")

    wsi, hdf5_file, prediction_sum, prediction_count = None, None, None, None

    try:
        wsi = openslide.OpenSlide(wsi_image_path)
        print(f"WSI načteno: {os.path.basename(wsi_image_path)}")

        if tissue_mask_level_index >= wsi.level_count:
            raise ValueError(f"Úroveň masky tkáně ({tissue_mask_level_index}) neexistuje.")
        if TARGET_INFERENCE_LEVEL >= wsi.level_count:
            raise ValueError(f"Cílová úroveň inference ({TARGET_INFERENCE_LEVEL}) neexistuje.")

        tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
        inference_downsample = wsi.level_downsamples[TARGET_INFERENCE_LEVEL]
        inference_level_dims = wsi.level_dimensions[TARGET_INFERENCE_LEVEL]
        output_shape = (inference_level_dims[1], inference_level_dims[0])

        print(f"Inference na úrovni: {TARGET_INFERENCE_LEVEL} (Downsample: {inference_downsample:.2f}x, Rozměry: {inference_level_dims})")
        
        wsi_number = os.path.splitext(os.path.basename(wsi_image_path))[0].split('_')[-1]
        tissue_mask_filename = f"all_tissue_tumor_{wsi_number}.npy"
        tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
        
        if not os.path.exists(tissue_mask_full_path):
            raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
        
        tissue_mask_np = np.load(tissue_mask_full_path).astype(bool)
        print(f"Maska tkáně načtena z: {tissue_mask_full_path}, tvar: {tissue_mask_np.shape}")

        scale_factor = tissue_mask_downsample / inference_downsample
        if scale_factor <= 0: raise ValueError("Neplatný scale_factor.")

        output_w, output_h = inference_level_dims
        cols = math.ceil(output_w / STEP)
        rows = math.ceil(output_h / STEP)
        total_tiles_grid = rows * cols
        print(f"Počet dlaždic v mřížce (krok {STEP}): {cols}x{rows} = {total_tiles_grid}")

        print(f"Alokuji akumulační pole {output_shape}...")
        prediction_sum = np.zeros(output_shape, dtype=np.float32)
        prediction_count = np.zeros(output_shape, dtype=np.uint16)
        
        processed_tiles_for_model, tiles_skipped_by_mask = 0, 0
        batch_tiles_data, batch_coords = [], []
        
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.inference_mode():
            for r in tqdm(range(rows), desc=f"Processing {os.path.basename(wsi_image_path)}", unit="row"):
                for c in range(cols):
                    x_inf_start = c * STEP
                    y_inf_start = r * STEP

                    current_tile_w = min(TILE_SIZE, output_w - x_inf_start)
                    current_tile_h = min(TILE_SIZE, output_h - y_inf_start)
                    if current_tile_w <= 0 or current_tile_h <= 0: continue

                    tm_x_start = int(x_inf_start / scale_factor)
                    tm_y_start = int(y_inf_start / scale_factor)
                    tm_w = max(1, int(current_tile_w / scale_factor))
                    tm_h = max(1, int(current_tile_h / scale_factor))
                    tm_y_end = min(tm_y_start + tm_h, tissue_mask_np.shape[0])
                    tm_x_end = min(tm_x_start + tm_w, tissue_mask_np.shape[1])
                    
                    tissue_ratio = 0.0
                    if tm_y_end > tm_y_start and tm_x_end > tm_x_start:
                        tissue_region = tissue_mask_np[tm_y_start:tm_y_end, tm_x_start:tm_x_end]
                        if tissue_region.size > 0: tissue_ratio = np.mean(tissue_region)

                    if tissue_ratio >= TISSUE_THRESHOLD:
                        try:
                            x_level0 = int(x_inf_start * inference_downsample)
                            y_level0 = int(y_inf_start * inference_downsample)
                            
                            tile_pil = wsi.read_region((x_level0, y_level0), TARGET_INFERENCE_LEVEL, (TILE_SIZE, TILE_SIZE))
                            tile_rgb = tile_pil.convert("RGB")
                            tile_w_orig, tile_h_orig = tile_rgb.size
                            if tile_w_orig == 0 or tile_h_orig == 0: continue

                            tile_to_process = tile_rgb
                            if tile_w_orig < TILE_SIZE or tile_h_orig < TILE_SIZE:
                                delta_w = max(0, TILE_SIZE - tile_w_orig)
                                delta_h = max(0, TILE_SIZE - tile_h_orig)
                                tile_to_process = ImageOps.expand(tile_rgb, (0, 0, delta_w, delta_h), fill=0)

                            tile_tensor_normalized = normalize(to_tensor(tile_to_process))
                            batch_tiles_data.append(tile_tensor_normalized)
                            batch_coords.append({'x_inf': x_inf_start, 'y_inf': y_inf_start, 'w_orig': tile_w_orig, 'h_orig': tile_h_orig})
                            processed_tiles_for_model += 1

                        except openslide.OpenSlideError as ose: print(f"OpenSlide chyba u [{c},{r}]: {ose}"); continue
                        except Exception as prep_err: print(f"Chyba přípravy dlaždice [{c},{r}]: {prep_err}"); continue
                    else:
                        tiles_skipped_by_mask += 1
                        continue

                    if len(batch_tiles_data) == BATCH_SIZE:
                        try:
                            batch_tensor = torch.stack(batch_tiles_data).to(device)
                            predictions = torch.sigmoid(model(batch_tensor)).cpu().numpy()
                            for i in range(predictions.shape[0]):
                                pred_map = predictions[i, 0, :, :]
                                coords = batch_coords[i]
                                x, y, w, h = coords['x_inf'], coords['y_inf'], coords['w_orig'], coords['h_orig']
                                h_add = min(TILE_SIZE, min(y + h, output_shape[0]) - y)
                                w_add = min(TILE_SIZE, min(x + w, output_shape[1]) - x)
                                if h_add > 0 and w_add > 0:
                                    prediction_sum[y:y+h_add, x:x+w_add] += pred_map[:h_add, :w_add]
                                    prediction_count[y:y+h_add, x:x+w_add] += 1
                        except Exception as batch_err: print(f"Chyba při zpracování dávky: {batch_err}")
                        finally:
                            batch_tiles_data.clear(); batch_coords.clear()
            
            if batch_tiles_data:
                print(f"Zpracovávám poslední dávku ({len(batch_tiles_data)} dlaždic)...")
                try:
                    batch_tensor = torch.stack(batch_tiles_data).to(device)
                    predictions = torch.sigmoid(model(batch_tensor)).cpu().numpy()
                    for i in range(predictions.shape[0]):
                        pred_map = predictions[i, 0, :, :]
                        coords = batch_coords[i]
                        x, y, w, h = coords['x_inf'], coords['y_inf'], coords['w_orig'], coords['h_orig']
                        h_add = min(TILE_SIZE, min(y + h, output_shape[0]) - y)
                        w_add = min(TILE_SIZE, min(x + w, output_shape[1]) - x)
                        if h_add > 0 and w_add > 0:
                            prediction_sum[y:y+h_add, x:x+w_add] += pred_map[:h_add, :w_add]
                            prediction_count[y:y+h_add, x:x+w_add] += 1
                except Exception as final_batch_err: print(f"Chyba při zpracování poslední dávky: {final_batch_err}")
                finally:
                    batch_tiles_data.clear(); batch_coords.clear()

        print("\nZpracování dlaždic dokončeno.")
        print(f"   Dlaždic zpracováno modelem: {processed_tiles_for_model}")
        print(f"   Dlaždic přeskočeno maskou: {tiles_skipped_by_mask}")

        print("Průměrování pravděpodobností...")
        average_probability = np.zeros_like(prediction_sum, dtype=np.float32)
        np.divide(prediction_sum, prediction_count, out=average_probability, where=prediction_count > 0)
        final_mask = (average_probability >= THRESHOLD)

        print(f"Ukládání finální masky do {output_hdf5_path}...")
        os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
        with h5py.File(output_hdf5_path, "w") as hdf5_file:
            hdf5_file.create_dataset("mask", data=final_mask, dtype=bool, chunks=(TILE_SIZE, TILE_SIZE), compression="gzip")
            hdf5_file.attrs["day_of_creation"] = datetime.datetime.now().isoformat()
            hdf5_file.attrs["tile_size_model"] = TILE_SIZE
            hdf5_file.attrs["model_weights"] = os.path.basename(model_weights_path)
            hdf5_file.attrs["wsi_level_processed"] = f"OS Level {TARGET_INFERENCE_LEVEL}"
        
        print("Ukládání dokončeno.")

    except (ValueError, FileNotFoundError, MemoryError) as e:
        print(f"KRITICKÁ CHYBA při zpracování {os.path.basename(wsi_image_path)}: {e}")
    except Exception as main_err:
        print(f"NEOČEKÁVANÁ CHYBA během zpracování {os.path.basename(wsi_image_path)}: {main_err}")
    finally:
        if wsi: wsi.close(); print("WSI soubor uzavřen.")
        script_end_time = time.time()
        print(f"Doba zpracování souboru: {script_end_time - script_start_time:.2f} sekund.")
        del prediction_sum, prediction_count
        gc.collect()


# --- HLAVNÍ SPUŠTĚCÍ BLOK ---
if __name__ == "__main__":
    
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    # model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    # model = UNet(n_channels=3, n_classes=1)
    try:
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Soubor s váhami modelu nebyl nalezen: {model_weights_path}")
        
        try:
            state_dict = torch.load(model_weights_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict["model_state_dict"])
        except (AttributeError, RuntimeError):
            print("Varování: Načtení s 'weights_only=True' selhalo. Zkouším načíst celý objekt (méně bezpečné).")
            model_and_weights = torch.load(model_weights_path, map_location=device)
            if isinstance(model_and_weights, dict) and "model_state_dict" in model_and_weights:
                 model.load_state_dict(model_and_weights["model_state_dict"])
            else:
                 model.load_state_dict(model_and_weights)

        print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FATÁLNÍ CHYBA: Nepodařilo se načíst váhy modelu: {e}")
        exit()

    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    
    search_pattern = os.path.join(WSI_INPUT_DIR, 'tumor_*.tif')#test_*.tif
    wsi_files_to_process = sorted(glob.glob(search_pattern))

    if not wsi_files_to_process:
        print(f"\nCHYBA: Ve složce '{WSI_INPUT_DIR}' nebyly nalezeny žádné soubory odpovídající vzoru 'test_*.tif'.")
    else:
        print(f"\nNalezeno {len(wsi_files_to_process)} souborů ke zpracování.")
        
        for wsi_full_path in wsi_files_to_process:
            try:
                wsi_basename = os.path.basename(wsi_full_path)
                wsi_number = os.path.splitext(wsi_basename)[0].split('_')[-1]
                output_filename = f"pred_{wsi_number}.h5"
                output_full_path = os.path.join(OUTPUT_PRED_DIR, output_filename)
                
                # ZDE POUŽIJEME NOVOU PROMĚNNOU TISSUE_MASK_DIR
                process_single_wsi(wsi_full_path, output_full_path, TISSUE_MASK_DIR, model, device)

            except Exception as loop_err:
                print(f"\n!!!!!! NEOČEKÁVANÁ CHYBA VE SMYČCE pro soubor {wsi_full_path}: {loop_err} !!!!!!")
                print("Pokračuji dalším souborem...")
                continue

    total_end_time = time.time()
    print("\n" + "="*80)
    print("VŠECHNY SOUBORY ZPRACOVÁNY.")
    print(f"Celkový čas běhu skriptu: {(total_end_time - start_time) / 60:.2f} minut.")
    print("="*80)

# --- END OF FINAL MODIFIED FILE ---