# -*- coding: utf-8 -*-
import os
# !!! Zajistěte správnou cestu k OpenSlide !!!

openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
os.add_dll_directory(openslide_dll_path)
import os
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
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import glob

# =================================================================================
# --- HLAVNÍ KONFIGURACE ---
# =================================================================================

# --- CESTY K ADRESÁŘŮM A SOUBORŮM ---
# Adresář, kde jsou společně uloženy `tumor_XXX.tif` a `mask_XXX.tif`
DATA_DIR = r"F:\wsi_dir_test"
# Adresář, kde jsou uloženy nízko-rozlišovací .npy masky celé tkáně
TISSUE_MASK_DIR = r"C:\Users\USER\Desktop\colab_unet\test_lowres_masky"
# Adresář, kam se uloží finální grafy a výsledky
OUTPUT_DIR = r"C:\Users\USER\Desktop\norm_patches"
# Cesta k naučeným vahám modelu
MODEL_WEIGHTS_PATH = r"C:\Users\USER\Desktop\results\2025-05-26_02-26-06\best_weights_2025-05-26_02-26-06.pth"

# --- PARAMETRY ZPRACOVÁNÍ ---
TILE_SIZE = 256
BATCH_SIZE = 64
TARGET_LEVEL_OFFSET = 2  # Cílová úroveň DZG (0 = nejvyšší rozlišení, 2 = 4x menší)
TISSUE_THRESHOLD = 0.1   # Min. podíl tkáně v dlaždici pro zpracování (0.0 až 1.0)
TISSUE_MASK_LEVEL_INDEX = 6 # OpenSlide úroveň, ze které byla .npy maska tkáně vytvořena

# --- PARAMETRY PRO VÝPOČET A TESTOVÁNÍ ---
# Kolik pixelů (v procentech) použít z každého WSI pro finální agregovaný graf.
# Šetří RAM při velkých datasetech. 1.0 = všechny, 0.02 = 2% vzorek.
SAMPLE_FRACTION = 0.02
# Omezí počet WSI snímků pro rychlé testování. Nastavte na None pro zpracování všech.
LIMIT_WSI_COUNT = None  # Např. 10 pro test, None pro plný běh

# =================================================================================
# --- FUNKCE PRO ZPRACOVÁNÍ JEDNOHO WSI ---
# =================================================================================

def process_single_wsi(wsi_path, gt_mask_path, tissue_mask_path, model, device, sample_fraction=1.0):
    """
    Kompletně zpracuje jeden WSI snímek a vrátí jeho AUC a (podvzorkované) predikce/ground_truth.
    """
    all_predictions_slide = []
    all_ground_truth_slide = []
    wsi, gt_mask_slide = None, None
    try:
        wsi = openslide.OpenSlide(wsi_path)
        gt_mask_slide = openslide.OpenSlide(gt_mask_path)
        
        if not os.path.exists(tissue_mask_path):
            tqdm.write(f"Varování: Maska tkáně nenalezena pro {os.path.basename(wsi_path)}, přeskakuji.")
            return None, None, None
            
        tissue_mask_np = np.load(tissue_mask_path).astype(bool)

        deepzoom = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=0, limit_bounds=False)
        deepzoom_mask = DeepZoomGenerator(gt_mask_slide, tile_size=TILE_SIZE, overlap=0, limit_bounds=False)

        dzg_level_index = deepzoom.level_count - 1 - TARGET_LEVEL_OFFSET
        if not (0 <= dzg_level_index < deepzoom.level_count):
            tqdm.write(f"Varování: Neplatný DZG index pro {os.path.basename(wsi_path)}, přeskakuji.")
            return None, None, None

        inference_downsample = 2**(deepzoom.level_count - 1 - dzg_level_index)
        tissue_mask_downsample = wsi.level_downsamples[TISSUE_MASK_LEVEL_INDEX]
        scale_factor = tissue_mask_downsample / inference_downsample

        level_tiles_cols, level_tiles_rows = deepzoom.level_tiles[dzg_level_index]
        
        batch_tiles_data, batch_masks_data, batch_coords = [], [], []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.inference_mode():
            for row in range(level_tiles_rows):
                for col in range(level_tiles_cols):
                    try:
                        tile_coords_level0, _, (tile_w_inf, tile_h_inf) = deepzoom.get_tile_coordinates(dzg_level_index, (col, row))
                        if tile_w_inf <= 0 or tile_h_inf <= 0: continue
                        x_inf_start = int(tile_coords_level0[0] / inference_downsample)
                        y_inf_start = int(tile_coords_level0[1] / inference_downsample)
                    except Exception: continue

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
                            tile_tensor_normalized = normalize(to_tensor(tile_padded))
                            mask_tensor = to_tensor(mask_padded)
                            batch_tiles_data.append(tile_tensor_normalized)
                            batch_masks_data.append(mask_tensor)
                            batch_coords.append({'w_orig': tile_w_orig, 'h_orig': tile_h_orig})
                        except Exception: continue
                    
                    is_last_tile = (row == level_tiles_rows - 1) and (col == level_tiles_cols - 1)
                    if len(batch_tiles_data) == BATCH_SIZE or (is_last_tile and batch_tiles_data):
                        if not batch_tiles_data: continue
                        batch_tensor = torch.stack(batch_tiles_data).to(device)
                        batch_mask_tensor = torch.stack(batch_masks_data)
                        prediction_output = model(batch_tensor)
                        predictions = torch.sigmoid(prediction_output).cpu()

                        for i in range(predictions.shape[0]):
                            orig_h, orig_w = batch_coords[i]['h_orig'], batch_coords[i]['w_orig']
                            pred_cropped = predictions[i].squeeze()[:orig_h, :orig_w]
                            gt_cropped = batch_mask_tensor[i].squeeze()[:orig_h, :orig_w]
                            all_predictions_slide.append(pred_cropped.flatten())
                            all_ground_truth_slide.append((gt_cropped > 0.5).flatten())
                        batch_tiles_data.clear(); batch_masks_data.clear(); batch_coords.clear()

        if not all_predictions_slide:
            return None, None, None

        final_preds = torch.cat(all_predictions_slide).numpy()
        final_gts = torch.cat(all_ground_truth_slide).numpy().astype(np.int8)

        slide_auc = None
        if len(np.unique(final_gts)) >= 2:
            slide_auc = roc_auc_score(final_gts, final_preds)

        if sample_fraction < 1.0 and len(final_preds) > 0:
            num_pixels = len(final_preds)
            sample_size = int(num_pixels * sample_fraction)
            random_indices = np.random.choice(num_pixels, size=sample_size, replace=False)
            final_preds = final_preds[random_indices]
            final_gts = final_gts[random_indices]

        return slide_auc, final_preds, final_gts

    except Exception as e:
        tqdm.write(f"Kritická chyba při zpracování {os.path.basename(wsi_path)}: {e}")
        return None, None, None
    finally:
        if wsi: wsi.close()
        if gt_mask_slide: gt_mask_slide.close()


# =================================================================================
# --- HLAVNÍ BĚH SKRIPTU ---
# =================================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    try:
        # model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
        model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        weights_and_data = torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=False)
        model.load_state_dict(weights_and_data['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model úspěšně načten.")
    except Exception as e:
        print(f"Chyba při načítání modelu: {e}")
        exit()

    wsi_files = sorted(glob.glob(os.path.join(DATA_DIR, "test_*.tif")))
    print(f"Nalezeno {len(wsi_files)} 'test_*.tif' souborů v {DATA_DIR}.")
    if not wsi_files:
        print(f"V adresáři {DATA_DIR} nebyly nalezeny žádné 'test_*.tif' soubory.")
        exit()
    
    if LIMIT_WSI_COUNT is not None and LIMIT_WSI_COUNT > 0:
        wsi_files = wsi_files[:LIMIT_WSI_COUNT]
        print(f"--- POZOR: Běh je omezen na prvních {len(wsi_files)} WSI souborů! ---")

    all_slide_aucs = []
    global_predictions_list = []
    global_gts_list = []
    
    for wsi_path in tqdm(wsi_files, desc="Zpracovávám WSI sadu"):
        basename = os.path.basename(wsi_path)
        gt_mask_path = os.path.join(DATA_DIR, basename.replace("test", "mask"))
        npy_basename = basename.replace("test", "mask").replace(".tif", ".npy")
        tissue_mask_path = os.path.join(TISSUE_MASK_DIR, npy_basename)
        
        if not os.path.exists(gt_mask_path):
            tqdm.write(f"Varování: Ground truth maska '{os.path.basename(gt_mask_path)}' nenalezena, přeskakuji {basename}.")
            continue
            
        slide_auc, slide_preds_sample, slide_gts_sample = process_single_wsi(
            wsi_path, gt_mask_path, tissue_mask_path, model, device, sample_fraction=SAMPLE_FRACTION
        )

        if slide_auc is not None:
            all_slide_aucs.append(slide_auc)
            tqdm.write(f"  -> AUC pro {basename}: {slide_auc:.4f}")
        
        if slide_preds_sample is not None and slide_gts_sample is not None:
            global_predictions_list.append(slide_preds_sample)
            global_gts_list.append(slide_gts_sample)
            
    print("\n" + "="*50)
    print("--- ZPRACOVÁNÍ CELÉ SADY DOKONČENO ---")
    print("="*50 + "\n")
    
    if all_slide_aucs:
        mean_auc = np.mean(all_slide_aucs)
        std_auc = np.std(all_slide_aucs)
        print(f"Průměrné AUC napříč {len(all_slide_aucs)} snímky: {mean_auc:.4f} ± {std_auc:.4f}\n")

    if global_predictions_list:
        print("Vykresluji agregovanou ROC křivku pro celý dataset...")
        full_dataset_preds = np.concatenate(global_predictions_list)
        full_dataset_gts = np.concatenate(global_gts_list)
        
        total_auc = roc_auc_score(full_dataset_gts, full_dataset_preds)
        fpr, tpr, thresholds = roc_curve(full_dataset_gts, full_dataset_preds)
        
        # Metoda 1: Youden's J statistic (maximalizace TPR - FPR)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        best_threshold_j = thresholds[best_j_idx]
        print(f"\n--- Analýza optimálního prahu ---")
        print(f"Optimální práh (Youden's J): {best_threshold_j}")
        print(f"  - Při tomto prahu: Citlivost (TPR) = {tpr[best_j_idx]:.4f}, Specificita (1-FPR) = {1 - fpr[best_j_idx]:.4f}")

        # Metoda 2: Nejbližší bod k levému hornímu rohu (0, 1)
        distances = np.sqrt((1 - tpr)**2 + fpr**2)
        best_dist_idx = np.argmin(distances)
        best_threshold_dist = thresholds[best_dist_idx]
        print(f"Optimální práh (nejblíže k [0,1]): {best_threshold_dist}")
        print(f"  - Při tomto prahu: Citlivost (TPR) = {tpr[best_dist_idx]:.4f}, Specificita (1-FPR) = {1 - fpr[best_dist_idx]:.4f}\n")
        
        # =====================================================================
        # --- Kreslení grafu (s vyznačením bodu) ---
        # =====================================================================
        
        plt.figure(figsize=(10, 8), dpi=150)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Agregovaná ROC křivka (AUC = {total_auc:.4f})')
        # Vykreslení optimálního bodu (např. podle Youden's J)
        plt.scatter(fpr[best_j_idx], tpr[best_j_idx], marker='o', color='red', s=100, zorder=10,
                    label=f"Optimální bod (práh ≈ {best_threshold_j:.2f})")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Referenční čára')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Míra falešně pozitivních (False Positive Rate)', fontsize=12)
        plt.ylabel('Míra správně pozitivních (True Positive Rate)', fontsize=12)
        plt.title(f'Agregovaná ROC Křivka pro {len(wsi_files)} WSI', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_path = os.path.join(OUTPUT_DIR, "aggregated_roc_curve_with_threshold.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Agregovaný ROC graf s optimálním prahem uložen do: {plot_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end_time = time.time()
    print(f"\nCelkový čas zpracování: {str(datetime.timedelta(seconds=end_time - start_time))}")