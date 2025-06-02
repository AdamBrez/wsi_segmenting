# --- START OF FINAL COMPLETE FILE ---
# zkopirovano z train_gemini pred tim nez jsem ta upravil ty rozdilne velikosti na okrajich
import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import glob
import time
import datetime
import math
import gc
import csv  # Pro práci s CSV soubory
import traceback # Pro detailní výpis chyb

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import numpy as np
import openslide
from model import UNet # Ujistěte se, že model.py je dostupné

# Importy funkcí pro metriky
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision


# --- HLAVNÍ KONFIGURACE ---
# !!! ZDE NASTAVTE POTŘEBNÉ CESTY !!!

# Cesta ke složce, která obsahuje vstupní WSI soubory (test_XXX.tif) a GT masky (mask_XXX.tif)
WSI_INPUT_DIR = r"F:\wsi_dir_test"  

# Cesta ke složce, která obsahuje MASKY TKÁNĚ (mask_XXX.npy)
TISSUE_MASK_DIR = r"C:\Users\USER\Desktop\colab_unet\test_lowres_masky"

# Cesta k výstupní složce, kam se uloží výsledné predikce (pred_XXX.h5)
OUTPUT_PRED_DIR = r"C:\Users\USER\Desktop\test_preds\pretrained_lvl_1" 

# Cesta k souboru s naučenými váhami modelu
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-26_02-26-06\best_weights_2025-05-26_02-26-06.pth"

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
tissue_mask_level_index = 6
TISSUE_THRESHOLD = 0.05

# <<< Ostatní konfigurace >>>
BATCH_SIZE = 64
THRESHOLD = 0.5
TARGET_INFERENCE_LEVEL = 2


def process_single_wsi(wsi_image_path, output_hdf5_path, tissue_mask_dir, model, device):
    """
    Zpracuje jeden WSI soubor, vypočítá metriky, uloží predikovanou masku a vrátí metriky pro zápis do souboru.
    """
    script_start_time = time.time()
    print("-" * 80)
    print(f"Zahajuji zpracování: {os.path.basename(wsi_image_path)}")
    
    wsi, gt_wsi, prediction_sum, prediction_count = None, None, None, None
    metrics_enabled = False
    
    wsi_basename = os.path.basename(wsi_image_path)
    wsi_number = os.path.splitext(wsi_basename)[0].split('_')[-1]
    gt_mask_filename = f"mask_{wsi_number}.tif"
    gt_mask_path = os.path.join(os.path.dirname(wsi_image_path), gt_mask_filename)

    if os.path.exists(gt_mask_path):
        try:
            gt_wsi = openslide.OpenSlide(gt_mask_path)
            metrics_enabled = True
            print(f"Nalezena a načtena GT maska: {gt_mask_filename}")
            wsi_total_tp, wsi_total_fp, wsi_total_fn, wsi_total_tn = 0, 0, 0, 0
        except openslide.OpenSlideError as e:
            print(f"VAROVÁNÍ: Nepodařilo se otevřít GT masku '{gt_mask_path}': {e}. Metriky nebudou počítány.")
    else:
        print(f"INFO: GT maska '{gt_mask_path}' nenalezena. Metriky nebudou počítány.")

    try:
        wsi = openslide.OpenSlide(wsi_image_path)
        print(f"WSI načteno: {os.path.basename(wsi_image_path)}")

        if TARGET_INFERENCE_LEVEL >= wsi.level_count: raise ValueError(f"Cílová úroveň inference ({TARGET_INFERENCE_LEVEL}) neexistuje.")
        
        inference_downsample = wsi.level_downsamples[TARGET_INFERENCE_LEVEL]
        inference_level_dims = wsi.level_dimensions[TARGET_INFERENCE_LEVEL]
        output_shape = (inference_level_dims[1], inference_level_dims[0])

        print(f"Inference na úrovni: {TARGET_INFERENCE_LEVEL} (Downsample: {inference_downsample:.2f}x, Rozměry: {inference_level_dims})")
        
        tissue_mask_filename = f"mask_{wsi_number}.npy"
        tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)
        if not os.path.exists(tissue_mask_full_path): raise FileNotFoundError(f"Soubor s maskou tkáně nebyl nalezen: {tissue_mask_full_path}")
        
        tissue_mask_np = np.load(tissue_mask_full_path).astype(bool)
        print(f"Maska tkáně načtena z: {tissue_mask_full_path}, tvar: {tissue_mask_np.shape}")
        
        tissue_mask_downsample = wsi.level_downsamples[tissue_mask_level_index]
        scale_factor = tissue_mask_downsample / inference_downsample

        output_w, output_h = inference_level_dims
        cols, rows = math.ceil(output_w / STEP), math.ceil(output_h / STEP)
        
        prediction_sum = np.zeros(output_shape, dtype=np.float32)
        prediction_count = np.zeros(output_shape, dtype=np.uint16)
        
        processed_tiles_for_model, tiles_skipped_by_mask = 0, 0
        batch_tiles_data, batch_coords, batch_gt_labels = [], [], []
        
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        gt_transform = transforms.ToTensor()

        with torch.inference_mode():
            # Cyklus pro iteraci přes dlaždice

            # Zde vkládám plnou logiku smyčky, aby byl kód kompletní
            for r in tqdm(range(rows), desc=f"Processing {os.path.basename(wsi_image_path)}", unit="row"):
                for c in range(cols):
                    x_inf_start, y_inf_start = c * STEP, r * STEP
                    
                    tm_x_start, tm_y_start = int(x_inf_start / scale_factor), int(y_inf_start / scale_factor)
                    tm_x_end, tm_y_end = tm_x_start + math.ceil(TILE_SIZE / scale_factor), tm_y_start + math.ceil(TILE_SIZE / scale_factor)
                    tissue_ratio = np.mean(tissue_mask_np[tm_y_start:tm_y_end, tm_x_start:tm_x_end]) if tm_y_start < tissue_mask_np.shape[0] and tm_x_start < tissue_mask_np.shape[1] else 0

                    if tissue_ratio >= TISSUE_THRESHOLD:
                        try:
                            x_level0, y_level0 = int(x_inf_start * inference_downsample), int(y_inf_start * inference_downsample)
                            
                            tile_pil = wsi.read_region((x_level0, y_level0), TARGET_INFERENCE_LEVEL, (TILE_SIZE, TILE_SIZE)).convert("RGB")
                            tile_w_orig, tile_h_orig = tile_pil.size
                            
                            tile_to_process = tile_pil
                            delta_w, delta_h = TILE_SIZE - tile_w_orig, TILE_SIZE - tile_h_orig
                            if delta_w > 0 or delta_h > 0:
                                tile_to_process = ImageOps.expand(tile_pil, (0, 0, delta_w, delta_h), fill=0)

                            if metrics_enabled:
                                gt_tile_pil = gt_wsi.read_region((x_level0, y_level0), TARGET_INFERENCE_LEVEL, (TILE_SIZE, TILE_SIZE)).convert('L')
                                gt_tile_to_process = gt_tile_pil
                                if delta_w > 0 or delta_h > 0:
                                    gt_tile_to_process = ImageOps.expand(gt_tile_pil, (0, 0, delta_w, delta_h), fill=0)
                                batch_gt_labels.append((gt_transform(gt_tile_to_process) > 0).float())

                            batch_tiles_data.append(normalize(to_tensor(tile_to_process)))
                            batch_coords.append({'x': x_inf_start, 'y': y_inf_start, 'w': tile_w_orig, 'h': tile_h_orig})
                            processed_tiles_for_model += 1
                        except Exception as e: continue
                    else:
                        tiles_skipped_by_mask += 1

                    is_last_tile = (r == rows - 1 and c == cols - 1)
                    if len(batch_tiles_data) == BATCH_SIZE or (is_last_tile and len(batch_tiles_data) > 0):
                        try:
                            if is_last_tile and len(batch_tiles_data) > 0:
                                print(f"Zpracovávám poslední dávku ({len(batch_tiles_data)} dlaždic)...")

                            batch_tensor = torch.stack(batch_tiles_data).to(device)
                            predictions_sigmoid = torch.sigmoid(model(batch_tensor))
                            
                            if metrics_enabled:
                                gt_batch_tensor = torch.stack(batch_gt_labels).to(device)
                                tp, fp, fn, tn = get_stats(predictions_sigmoid, gt_batch_tensor.long(), mode="binary", threshold=THRESHOLD)
                                wsi_total_tp += tp.sum().item()
                                wsi_total_fp += fp.sum().item()
                                wsi_total_fn += fn.sum().item()
                                wsi_total_tn += tn.sum().item()

                            predictions = predictions_sigmoid.cpu().numpy()
                            for i in range(len(batch_coords)):
                                coords = batch_coords[i]
                                x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
                                prediction_sum[y:y+h, x:x+w] += predictions[i, 0, :h, :w]
                                prediction_count[y:y+h, x:x+w] += 1
                        except Exception as e: print(f"Chyba při zpracování dávky: {e}")
                        finally:
                            batch_tiles_data.clear(); batch_coords.clear(); batch_gt_labels.clear()

        print("\nZpracování dlaždic dokončeno.")
        print(f"   Dlaždic zpracováno modelem: {processed_tiles_for_model}")
        print(f"   Dlaždic přeskočeno maskou: {tiles_skipped_by_mask}")
        
        results_to_return = None
        if metrics_enabled and processed_tiles_for_model > 0:
            print("\n--- Metriky pro WSI (vyhodnoceno na úrovni dlaždic) ---")
            
            tp = torch.tensor(wsi_total_tp, dtype=torch.long)
            fp = torch.tensor(wsi_total_fp, dtype=torch.long)
            fn = torch.tensor(wsi_total_fn, dtype=torch.long)
            tn = torch.tensor(wsi_total_tn, dtype=torch.long)

            if (tp + fn) == 0:
                wsi_recall = np.nan
            else:
                wsi_recall = recall(tp=tp, fp=fp, fn=fn, tn=tn).item()

            if (tp + fp) == 0:
                wsi_precision = 1.0 if (tp + fn) == 0 else 0.0
            else:
                wsi_precision = precision(tp=tp, fp=fp, fn=fn, tn=tn).item()

            wsi_dice = f1_score(tp=tp, fp=fp, fn=fn, tn=tn).item()
            wsi_iou = iou_score(tp=tp, fp=fp, fn=fn, tn=tn).item()
            
            print(f"   Dice (F1-Score): {wsi_dice:.5f}")
            print(f"   IoU (Jaccard):   {wsi_iou:.5f}")
            print(f"   Recall:          {'NaN' if np.isnan(wsi_recall) else f'{wsi_recall:.5f}'}")
            print(f"   Precision:       {wsi_precision:.5f}")
            print("---------------------------------------------------------")

            results_to_return = {
                "Dice": wsi_dice, "IoU": wsi_iou, "Recall": wsi_recall, "Precision": wsi_precision
            }

        print("Průměrování pravděpodobností...")
        average_probability = np.zeros_like(prediction_sum, dtype=np.float32)
        np.divide(prediction_sum, prediction_count, out=average_probability, where=prediction_count > 0)
        final_mask = (average_probability >= THRESHOLD)

        print(f"Ukládání finální masky do {output_hdf5_path}...")
        os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
        with h5py.File(output_hdf5_path, "w") as hdf5_file:
            hdf5_file.create_dataset("mask", data=final_mask, dtype=bool, chunks=(TILE_SIZE, TILE_SIZE), compression="gzip")
        print("Ukládání dokončeno.")
        
        return results_to_return

    except Exception as main_err:
        print(f"NEOČEKÁVANÁ CHYBA během zpracování {os.path.basename(wsi_image_path)}: {main_err}")
        traceback.print_exc()
        return None
    finally:
        if wsi: wsi.close()
        if gt_wsi: gt_wsi.close()
        print("WSI a GT soubory uzavřeny.")
        print(f"Doba zpracování souboru: {time.time() - script_start_time:.2f} sekund.")
        del prediction_sum, prediction_count, wsi, gt_wsi
        gc.collect()


# --- HLAVNÍ SPUŠTĚCÍ BLOK ---
if __name__ == "__main__":
    import segmentation_models_pytorch as smp

    main_start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    # model = UNet(n_channels=3, n_classes=1)
    model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
    try:
        state_dict = torch.load(model_weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict.get("model_state_dict", state_dict))
        print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FATÁLNÍ CHYBA: Nepodařilo se načíst váhy modelu: {e}")
        exit()

    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    
    results_csv_path = os.path.join(OUTPUT_PRED_DIR, "evaluation_results.csv")
    csv_header = ["Prediction_File", "Dice", "IoU", "Recall", "Precision"]
    
    if not os.path.exists(results_csv_path):
        try:
            with open(results_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_header)
            print(f"Vytvořen soubor pro výsledky: {results_csv_path}")
        except IOError as e:
            print(f"CHYBA: Nepodařilo se vytvořit soubor pro výsledky: {e}")
            exit()

    search_pattern = os.path.join(WSI_INPUT_DIR, 'test_*.tif')
    wsi_files_to_process = sorted(glob.glob(search_pattern))

    if not wsi_files_to_process:
        print(f"\nCHYBA: Ve složce '{WSI_INPUT_DIR}' nebyly nalezeny žádné soubory 'test_*.tif'.")
    else:
        print(f"\nNalezeno {len(wsi_files_to_process)} souborů ke zpracování.")
        
        for wsi_full_path in wsi_files_to_process:
            try:
                wsi_basename = os.path.basename(wsi_full_path)
                wsi_number = os.path.splitext(wsi_basename)[0].split('_')[-1]
                output_filename = f"pred_{wsi_number}.h5"
                output_full_path = os.path.join(OUTPUT_PRED_DIR, output_filename)
                
                metrics = process_single_wsi(wsi_full_path, output_full_path, TISSUE_MASK_DIR, model, device)

                if metrics:
                    try:
                        with open(results_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile)
                            recall_value = "NaN" if np.isnan(metrics['Recall']) else f"{metrics['Recall']:.5f}"
                            row_data = [
                                output_filename,
                                f"{metrics['Dice']:.5f}",
                                f"{metrics['IoU']:.5f}",
                                recall_value,
                                f"{metrics['Precision']:.5f}"
                            ]
                            writer.writerow(row_data)
                        print(f"Výsledky pro '{output_filename}' byly úspěšně zapsány do CSV souboru.")
                    except IOError as e:
                        print(f"CHYBA při zápisu do CSV souboru: {e}")

            except Exception as loop_err:
                print(f"\n!!!!!! NEOČEKÁVANÁ CHYBA VE SMYČCE pro soubor {wsi_full_path}: {loop_err} !!!!!!")
                traceback.print_exc()
                print("Pokračuji dalším souborem...")
                continue

    total_end_time = time.time()
    print("\n" + "="*80)
    print("VŠECHNY SOUBORY ZPRACOVÁNY.")
    print(f"Celkový čas běhu skriptu: {(total_end_time - main_start_time) / 60:.2f} minut.")
    print(f"Výsledky uloženy v: {results_csv_path}")
    print("="*80)

# --- END OF FINAL COMPLETE FILE ---