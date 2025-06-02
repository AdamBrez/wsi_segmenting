# -*- coding: utf-8 -*-

import os
# !!! Zajistěte správnou cestu k OpenSlide !!!
try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    else:
        print(f"Varování: Cesta k OpenSlide DLL neexistuje: {openslide_dll_path}")
except AttributeError:
    print("os.add_dll_directory není dostupné ve vaší verzi Pythonu.")
except Exception as e:
    print(f"Nastala chyba při přidávání OpenSlide DLL: {e}")

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import time
import segmentation_models_pytorch as smp
import datetime
import gc
from sklearn.metrics import roc_auc_score, roc_curve # <-- PŘIDÁN IMPORT roc_curve
import matplotlib.pyplot as plt # <-- PŘIDÁN IMPORT pro kreslení

# --- Konfigurace ---
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-23_01-36-24\best_weights_2025-05-23_01-36-24.pth"
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_017.tif"
wsi_mask_path = r"C:\Users\USER\Desktop\wsi_dir\mask_017.tif" # Cesta k ground truth masce
output_dir = r"C:\Users\USER\Desktop\results" # <-- NOVÉ: Adresář pro uložení výsledků (včetně grafu)

# <<< Konfigurace pro filtrování podle masky tkáně >>>
tissue_mask_dir = r"C:\Users\USER\Desktop\colab_unet\masky_new"
tissue_mask_level_index = 6
TISSUE_THRESHOLD = 0.1

# Velikost, kterou očekává model
TILE_SIZE = 256
OVERLAP = 0

BATCH_SIZE = 64
TARGET_LEVEL_OFFSET = 2

# --- Inicializace ---
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
print(f"Model očekává vstup: {TILE_SIZE}x{TILE_SIZE}")

# Zajištění existence výstupního adresáře
os.makedirs(output_dir, exist_ok=True)

# Načtení modelu
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
try:
    if not os.path.exists(model_weights_path): raise FileNotFoundError(f"Váhy nenalezeny: {model_weights_path}")
    model_state = torch.load(model_weights_path, map_location=device)
    if 'model_state_dict' in model_state:
        model.load_state_dict(model_state['model_state_dict'])
    else:
        model.load_state_dict(model_state)
    print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
except Exception as e:
    print(f"Chyba při načítání vah modelu: {e}")
    exit()
model.to(device)
model.eval()

# Normalizační transformace
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

all_predictions = []
all_ground_truth = []

# Načtení WSI a masek
wsi = None
tissue_mask_np = None
gt_mask_slide = None
try:
    wsi = openslide.OpenSlide(wsi_image_path)
    print(f"WSI načteno: {wsi_image_path}")
    gt_mask_slide = openslide.OpenSlide(wsi_mask_path)
    print(f"Ground truth maska načtena: {wsi_mask_path}")

    if tissue_mask_level_index >= wsi.level_count:
        raise ValueError(f"Požadovaná úroveň masky tkáně ({tissue_mask_level_index}) neexistuje (max index {wsi.level_count - 1}).")
    tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
    print(f"Maska tkáně byla generována z úrovně {tissue_mask_level_index} (downsample {tissue_mask_downsample:.2f}x).")

    wsi_filename_base = os.path.splitext(os.path.basename(wsi_image_path))[0]
    tissue_mask_filename = f"{wsi_filename_base.replace('tumor', 'mask')}.npy"
    tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
    if not os.path.exists(tissue_mask_full_path):
        raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
    tissue_mask_np = np.load(tissue_mask_full_path)
    print(f"Maska tkáně načtena z: {tissue_mask_full_path}, tvar: {tissue_mask_np.shape}")
    if not np.issubdtype(tissue_mask_np.dtype, np.bool_):
         print(f"Varování: Maska tkáně není typu bool (je {tissue_mask_np.dtype}). Převedu ji.")
         tissue_mask_np = tissue_mask_np.astype(bool)

    deepzoom = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=False)
    deepzoom_mask = DeepZoomGenerator(gt_mask_slide, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=False)
    print(f"DZG parametry: tile_size={TILE_SIZE}, overlap={OVERLAP}")

    dzg_level_index = deepzoom.level_count - 1 - TARGET_LEVEL_OFFSET
    if not (0 <= dzg_level_index < deepzoom.level_count):
        raise ValueError(f"Neplatný DZG level index: {dzg_level_index}. Zkontrolujte TARGET_LEVEL_OFFSET.")

    inference_downsample = deepzoom.level_count - 1 - dzg_level_index
    inference_level_dims = deepzoom.level_dimensions[dzg_level_index]
    
    print(f"Zpracovávám DZG úroveň {dzg_level_index}")
    print(f"   - Rozměry inference úrovně: {inference_level_dims} (šířka, výška)")
    print(f"   - Downsample inference úrovně: {inference_downsample:.2f}x")

    scale_factor = tissue_mask_downsample / inference_downsample
    print(f"Škálovací faktor (Maska tkáně / Inference Level): {scale_factor:.3f}")

    level_tiles_cols, level_tiles_rows = deepzoom.level_tiles[dzg_level_index]
    total_tiles_grid = level_tiles_rows * level_tiles_cols
    print(f"   - Očekávaný počet dlaždic v mřížce DZG: {level_tiles_cols}x{level_tiles_rows} = {total_tiles_grid}")

    processed_tiles_for_model = 0
    tiles_skipped_by_mask = 0
    batch_tiles_data = []
    batch_masks_data = []
    batch_coords = []

    with torch.inference_mode():
        for row in tqdm(range(level_tiles_rows), desc="Zpracovávám řádky"):
            for col in range(level_tiles_cols):
                try:
                    tile_coords_level0, _, (tile_w_inf, tile_h_inf) = deepzoom.get_tile_coordinates(dzg_level_index, (col, row))
                    x_inf_start = int(tile_coords_level0[0] / inference_downsample)
                    y_inf_start = int(tile_coords_level0[1] / inference_downsample)
                    if tile_w_inf <= 0 or tile_h_inf <= 0: continue
                except Exception as coord_err:
                    print(f"Chyba při získávání souřadnic pro [{col},{row}]: {coord_err}")
                    continue

                tm_x_start = int(x_inf_start / scale_factor)
                tm_y_start = int(y_inf_start / scale_factor)
                tm_w = max(1, int(tile_w_inf / scale_factor))
                tm_h = max(1, int(tile_h_inf / scale_factor))
                tm_y_end = min(tm_y_start + tm_h, tissue_mask_np.shape[0])
                tm_x_end = min(tm_x_start + tm_w, tissue_mask_np.shape[1])
                tissue_region = tissue_mask_np[tm_y_start:tm_y_end, tm_x_start:tm_x_end]
                tissue_ratio = np.mean(tissue_region) if tissue_region.size > 0 else 0.0

                if tissue_ratio >= TISSUE_THRESHOLD:
                    try:
                        tile = deepzoom.get_tile(dzg_level_index, (col, row))
                        gt_mask_tile = deepzoom_mask.get_tile(dzg_level_index, (col, row))
                        tile_rgb = tile.convert("RGB")
                        gt_mask_l = gt_mask_tile.convert("L")
                        tile_w_orig, tile_h_orig = tile_rgb.size
                        tile_padded = ImageOps.pad(tile_rgb, (TILE_SIZE, TILE_SIZE))
                        mask_padded = ImageOps.pad(gt_mask_l, (TILE_SIZE, TILE_SIZE))
                        tile_tensor = to_tensor(tile_padded)
                        tile_tensor_normalized = normalize(tile_tensor)
                        mask_tensor = to_tensor(mask_padded)
                        batch_tiles_data.append(tile_tensor_normalized)
                        batch_masks_data.append(mask_tensor)
                        batch_coords.append({'w_orig': tile_w_orig, 'h_orig': tile_h_orig})
                        processed_tiles_for_model += 1
                    except Exception as prep_err:
                         print(f"Chyba při přípravě dlaždice [{col},{row}]: {prep_err}")
                         continue
                else:
                    tiles_skipped_by_mask += 1
                    continue

                is_last_tile = (row == level_tiles_rows - 1) and (col == level_tiles_cols - 1)
                if len(batch_tiles_data) == BATCH_SIZE or (is_last_tile and batch_tiles_data):
                    try:
                        batch_tensor = torch.stack(batch_tiles_data).to(device)
                        batch_mask_tensor = torch.stack(batch_masks_data)
                        prediction_output = model(batch_tensor)
                        predictions = torch.sigmoid(prediction_output).cpu()

                        for i in range(predictions.shape[0]):
                            pred_tensor = predictions[i].squeeze()
                            gt_tensor = batch_mask_tensor[i].squeeze()
                            coords = batch_coords[i]
                            orig_h, orig_w = coords['h_orig'], coords['w_orig']
                            pred_cropped = pred_tensor[:orig_h, :orig_w]
                            gt_cropped = gt_tensor[:orig_h, :orig_w]
                            gt_binary = (gt_cropped > 0.5).int()
                            all_predictions.append(pred_cropped.flatten())
                            all_ground_truth.append(gt_binary.flatten())

                        batch_tiles_data.clear()
                        batch_masks_data.clear()
                        batch_coords.clear()
                    except Exception as batch_err:
                        print(f"Chyba při zpracování dávky: {batch_err}")
                        batch_tiles_data.clear()
                        batch_masks_data.clear()
                        batch_coords.clear()
                        continue
    
    print("\nZpracování dokončeno.")
    print(f"Celkem dlaždic v mřížce: {total_tiles_grid}")
    print(f"Dlaždice přeskočeny na základě masky tkáně: {tiles_skipped_by_mask}")
    print(f"Dlaždice zpracovány modelem: {processed_tiles_for_model}")

    if not all_predictions:
        print("Nebyly zpracovány žádné dlaždice, nelze vypočítat AUC.")
    else:
        print("Agreguji výsledky pro výpočet AUC...")
        final_preds = torch.cat(all_predictions).numpy()
        final_gts = torch.cat(all_ground_truth).numpy()
        
        if len(np.unique(final_gts)) < 2:
            print("Ground truth data obsahují pouze jednu třídu, nelze vypočítat AUC.")
            print(f"Nalezené třídy: {np.unique(final_gts)}")
        else:
            # Výpočet WSI-level AUC
            wsi_auc = roc_auc_score(final_gts, final_preds)
            print("="*40)
            print(f"WSI-level ROC AUC skóre: {wsi_auc:.6f}")
            print("="*40)

            # --- ZAČÁTEK NOVÉ ČÁSTI: Vykreslení ROC křivky ---
            print("Vykresluji ROC křivku...")
            fpr, tpr, thresholds = roc_curve(final_gts, final_preds)

            plt.figure(figsize=(8, 6), dpi=100)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC křivka (AUC = {wsi_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Referenční čára (náhodný tip)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Míra falešně pozitivních (False Positive Rate)')
            plt.ylabel('Míra správně pozitivních (True Positive Rate)')
            plt.title(f'ROC Křivka pro WSI: {os.path.basename(wsi_image_path)}')
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Uložení grafu
            plot_filename = f"roc_curve_{wsi_filename_base}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close() # Uvolnění paměti
            
            print(f"✔️ ROC křivka byla úspěšně uložena do:\n{plot_path}")
            # --- KONEC NOVÉ ČÁSTI ---


except Exception as e:
    import traceback
    print(f"\nDošlo k závažné chybě: {e}")
    traceback.print_exc() # Vypíše detailní informace o chybě
finally:
    if 'wsi' in locals() and wsi: wsi.close()
    if 'gt_mask_slide' in locals() and gt_mask_slide: gt_mask_slide.close()
    
    del all_predictions, all_ground_truth
    if 'model' in locals(): del model
    if 'batch_tensor' in locals(): del batch_tensor
    if 'prediction_output' in locals(): del prediction_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Celkový čas běhu: {str(datetime.timedelta(seconds=total_time))}")