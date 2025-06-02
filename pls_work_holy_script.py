# --- START OF FINAL COMPLETE FILE ---
import os
# !!! ZDE NASTAVTE CESTU K OPENSLIDE DLL, pokud není v systémové PATH !!!
# Příklad:
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

from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
import segmentation_models_pytorch as smp


# --- HLAVNÍ KONFIGURACE ---
WSI_INPUT_DIR = r"F:\wsi_dir_test"
TISSUE_MASK_DIR = r"C:\Users\USER\Desktop\colab_unet\test_lowres_masky"
OUTPUT_PRED_DIR = r"C:\Users\USER\Desktop\test_preds\unetpp_2" # Změňte podle potřeby
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-24_00-11-56\best_weights_2025-05-24_00-11-56.pth" # Změňte na správnou cestu

TILE_SIZE = 256
OVERLAP_PX = 32
STEP = TILE_SIZE - OVERLAP_PX
if STEP <= 0: raise ValueError("Krok (TILE_SIZE - OVERLAP_PX) musí být pozitivní.")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CURRENT_MEAN = IMAGENET_MEAN
CURRENT_STD = IMAGENET_STD

tissue_mask_level_index = 6
TISSUE_THRESHOLD = 0.05

BATCH_SIZE = 24
CLASSIFICATION_BATCH_SIZE = BATCH_SIZE
SEGMENTATION_BATCH_SIZE = BATCH_SIZE

THRESHOLD = 0.5
TARGET_INFERENCE_LEVEL = 2

# Nová konstanta: Minimální počet pozitivních dlaždic pro klasifikaci WSI jako nádorového
MIN_POSITIVE_TILES_FOR_TUMOROUS = 10 # Nastavte dle potřeby
TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION = 0.995 # Práh pro jednotlivou dlaždici


# --- FUNKCE PRO RYCHLOU KLASIFIKACI ---
def classify_wsi_quickly(wsi_image_path, tissue_mask_dir, model, device,
                         tile_size_clf=TILE_SIZE,
                         target_level_clf=TARGET_INFERENCE_LEVEL,
                         tissue_threshold_clf=TISSUE_THRESHOLD,
                         batch_size_clf=CLASSIFICATION_BATCH_SIZE,
                         tumor_tile_threshold=TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION,
                         min_positive_tiles_for_tumorous=MIN_POSITIVE_TILES_FOR_TUMOROUS):
    """
    Rychle klasifikuje WSI jako "tumorous" nebo "healthy".
    WSI je "tumorous", pokud počet dlaždic překračujících tumor_tile_threshold
    dosáhne min_positive_tiles_for_tumorous.
    """
    wsi_clf = None
    script_start_time = time.time()
    wsi_basename = os.path.basename(wsi_image_path)
    print(f"--- Rychlá klasifikace pro: {wsi_basename} ---")
    
    classification_result_internal = "unknown_default"
    positive_tiles_count = 0

    try:
        wsi_clf = openslide.OpenSlide(wsi_image_path)
        wsi_number = os.path.splitext(wsi_basename)[0].split('_')[-1]
        
        tissue_mask_filename = f"mask_{wsi_number}.npy"
        tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)

        if not os.path.exists(tissue_mask_full_path):
            print(f"VAROVÁNÍ (Klasifikace): Maska tkáně nenalezena: {tissue_mask_full_path}.")
            classification_result_internal = "unknown_mask_missing"
        elif target_level_clf >= wsi_clf.level_count:
            print(f"VAROVÁNÍ (Klasifikace): Cílová úroveň {target_level_clf} neexistuje pro {wsi_basename}.")
            classification_result_internal = "unknown_invalid_level"
        else:
            inference_downsample_clf = wsi_clf.level_downsamples[target_level_clf]
            step_clf = tile_size_clf # Pro ne-překryvné dlaždice
            
            level_dims_clf = wsi_clf.level_dimensions[target_level_clf]
            output_w_clf, output_h_clf = level_dims_clf
            cols_clf = math.ceil(output_w_clf / step_clf)
            rows_clf = math.ceil(output_h_clf / step_clf)

            tissue_mask_np = np.load(tissue_mask_full_path).astype(bool)
            tissue_mask_downsample_orig = wsi_clf.level_downsamples[tissue_mask_level_index]
            scale_factor_clf = tissue_mask_downsample_orig / inference_downsample_clf
            
            batch_tiles_data_clf = []
            normalize_clf = transforms.Normalize(mean=CURRENT_MEAN, std=CURRENT_STD)
            processed_for_model_clf = 0
            found_enough_positive_tiles = False

            with torch.inference_mode():
                for r_clf in range(rows_clf):
                    if found_enough_positive_tiles: break
                    for c_clf in range(cols_clf):
                        x_inf_start_clf, y_inf_start_clf = c_clf * step_clf, r_clf * step_clf
                        
                        tm_x_start_clf = int(x_inf_start_clf / scale_factor_clf)
                        tm_y_start_clf = int(y_inf_start_clf / scale_factor_clf)
                        tm_x_end_clf = tm_x_start_clf + math.ceil(tile_size_clf / scale_factor_clf)
                        tm_y_end_clf = tm_y_start_clf + math.ceil(tile_size_clf / scale_factor_clf)
                        
                        tm_y_end_clf = min(tm_y_end_clf, tissue_mask_np.shape[0])
                        tm_x_end_clf = min(tm_x_end_clf, tissue_mask_np.shape[1])
                        
                        tissue_region_clf = tissue_mask_np[tm_y_start_clf:tm_y_end_clf, tm_x_start_clf:tm_x_end_clf]
                        tissue_ratio_clf = np.mean(tissue_region_clf) if tissue_region_clf.size > 0 else 0.0

                        if tissue_ratio_clf >= tissue_threshold_clf:
                            try:
                                x_level0_clf = int(x_inf_start_clf * inference_downsample_clf)
                                y_level0_clf = int(y_inf_start_clf * inference_downsample_clf)
                                
                                tile_pil_clf = wsi_clf.read_region((x_level0_clf, y_level0_clf), target_level_clf, (tile_size_clf, tile_size_clf)).convert("RGB")
                                tile_w_orig_clf, tile_h_orig_clf = tile_pil_clf.size

                                tile_to_process_clf = tile_pil_clf
                                delta_w_clf, delta_h_clf = tile_size_clf - tile_w_orig_clf, tile_size_clf - tile_h_orig_clf
                                if delta_w_clf > 0 or delta_h_clf > 0:
                                    tile_to_process_clf = ImageOps.expand(tile_pil_clf, (0, 0, delta_w_clf, delta_h_clf), fill=0)

                                batch_tiles_data_clf.append(normalize_clf(to_tensor(tile_to_process_clf)))
                                processed_for_model_clf +=1
                            except Exception:
                                continue 
                        
                        is_last_tile_clf = (r_clf == rows_clf - 1 and c_clf == cols_clf - 1)
                        if len(batch_tiles_data_clf) == batch_size_clf or (is_last_tile_clf and len(batch_tiles_data_clf) > 0):
                            if not batch_tiles_data_clf: continue
                            
                            batch_tensor_clf = torch.stack(batch_tiles_data_clf).to(device)
                            predictions_output_clf = model(batch_tensor_clf)
                            predictions_sigmoid_clf = torch.sigmoid(predictions_output_clf)
                            
                            # Iteruj přes predikce v batchi a počítej pozitivní dlaždice
                            for pred_single_tile in predictions_sigmoid_clf:
                                if torch.max(pred_single_tile).item() > tumor_tile_threshold:
                                    positive_tiles_count += 1
                                    if positive_tiles_count >= min_positive_tiles_for_tumorous:
                                        found_enough_positive_tiles = True
                                        break # Ukonči iteraci přes batch, pokud bylo dosaženo limitu
                            
                            batch_tiles_data_clf.clear()
                            if found_enough_positive_tiles:
                                break # Ukonči vnitřní smyčku (cols)
            
            if found_enough_positive_tiles:
                classification_result_internal = "tumorous"
            elif classification_result_internal == "unknown_default": # Žádná chyba a nedosažen limit pozitivních
                classification_result_internal = "healthy"
            
            doba_trvani = time.time() - script_start_time
            if classification_result_internal == "tumorous":
                print(f"--- {wsi_basename}: Klasifikováno jako NÁDOROVÉ (prah {tumor_tile_threshold} překročen u {positive_tiles_count}/{min_positive_tiles_for_tumorous} dlaždic). Doba: {doba_trvani:.2f}s ---")
            elif classification_result_internal == "healthy":
                print(f"--- {wsi_basename}: Klasifikováno jako ZDRAVÉ (pozitivních dlaždic: {positive_tiles_count}/{min_positive_tiles_for_tumorous}, zpracováno celkem {processed_for_model_clf} dlaždic). Doba: {doba_trvani:.2f}s ---")

    except Exception as e:
        print(f"CHYBA (Klasifikace) během zpracování {wsi_basename}: {e}")
        traceback.print_exc()
        classification_result_internal = "unknown_error_classification"
    finally:
        if wsi_clf is not None: # Zavři pouze pokud byl objekt úspěšně vytvořen
            try:
                wsi_clf.close()
            except Exception as e_close:
                print(f"Varování: Chyba pri zavirani wsi_clf v classify_wsi_quickly: {e_close}")
    
    return classification_result_internal

# --- FUNKCE PRO SEGMENTACI (tvoje původní process_single_wsi) ---
# ... (Tato funkce zůstává stejná jako v tvém posledním poskytnutém kódu)
def process_single_wsi(wsi_image_path, output_hdf5_path, tissue_mask_dir, model, device):
    """
    Zpracuje jeden WSI soubor, vypočítá metriky, uloží predikovanou masku a vrátí metriky pro zápis do souboru.
    """
    script_start_time = time.time()
    print("-" * 80)
    print(f"Zahajuji PLNOU SEGMENTACI: {os.path.basename(wsi_image_path)}")
    
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
            metrics_enabled = False 
    else:
        print(f"INFO: GT maska '{gt_mask_path}' nenalezena. Metriky nebudou počítány.")
        metrics_enabled = False

    try:
        wsi = openslide.OpenSlide(wsi_image_path)
        print(f"WSI načteno: {os.path.basename(wsi_image_path)}")

        if TARGET_INFERENCE_LEVEL >= wsi.level_count:
            raise ValueError(f"Cílová úroveň inference ({TARGET_INFERENCE_LEVEL}) neexistuje pro {wsi_basename}.")
        
        inference_downsample = wsi.level_downsamples[TARGET_INFERENCE_LEVEL]
        inference_level_dims = wsi.level_dimensions[TARGET_INFERENCE_LEVEL]
        output_shape = (inference_level_dims[1], inference_level_dims[0])

        print(f"Inference na úrovni: {TARGET_INFERENCE_LEVEL} (Downsample: {inference_downsample:.2f}x, Rozměry: {inference_level_dims})")
        
        tissue_mask_filename_seg = f"mask_{wsi_number}.npy" 
        tissue_mask_full_path_seg = os.path.join(tissue_mask_dir, tissue_mask_filename_seg)
        if not os.path.exists(tissue_mask_full_path_seg):
            raise FileNotFoundError(f"Soubor s maskou tkáně (pro segmentaci) nebyl nalezen: {tissue_mask_full_path_seg}")
        
        tissue_mask_np = np.load(tissue_mask_full_path_seg).astype(bool)
        print(f"Maska tkáně (pro segmentaci) načtena z: {tissue_mask_full_path_seg}, tvar: {tissue_mask_np.shape}")
        
        tissue_mask_downsample_orig = wsi.level_downsamples[tissue_mask_level_index] 
        scale_factor = tissue_mask_downsample_orig / inference_downsample

        output_w, output_h = inference_level_dims
        cols, rows = math.ceil(output_w / STEP), math.ceil(output_h / STEP) 
        
        prediction_sum = np.zeros(output_shape, dtype=np.float32)
        prediction_count = np.zeros(output_shape, dtype=np.uint16) 
        
        processed_tiles_for_model, tiles_skipped_by_mask = 0, 0
        batch_tiles_data, batch_coords, batch_gt_labels = [], [], []
        
        normalize = transforms.Normalize(mean=CURRENT_MEAN, std=CURRENT_STD)
        gt_transform = transforms.ToTensor() 

        with torch.inference_mode():
            for r in tqdm(range(rows), desc=f"Segmenting {os.path.basename(wsi_image_path)}", unit="row"):
                for c in range(cols):
                    x_inf_start, y_inf_start = c * STEP, r * STEP 
                    
                    tm_x_start = int(x_inf_start / scale_factor)
                    tm_y_start = int(y_inf_start / scale_factor)
                    tm_x_end = tm_x_start + math.ceil(TILE_SIZE / scale_factor)
                    tm_y_end = tm_y_start + math.ceil(TILE_SIZE / scale_factor)

                    tm_y_end = min(tm_y_end, tissue_mask_np.shape[0])
                    tm_x_end = min(tm_x_end, tissue_mask_np.shape[1])

                    tissue_region = tissue_mask_np[tm_y_start:tm_y_end, tm_x_start:tm_x_end]
                    tissue_ratio = np.mean(tissue_region) if tissue_region.size > 0 else 0.0
                    
                    if tissue_ratio >= TISSUE_THRESHOLD:
                        try:
                            x_level0 = int(x_inf_start * inference_downsample)
                            y_level0 = int(y_inf_start * inference_downsample)
                            
                            tile_pil = wsi.read_region((x_level0, y_level0), TARGET_INFERENCE_LEVEL, (TILE_SIZE, TILE_SIZE)).convert("RGB")
                            tile_w_orig, tile_h_orig = tile_pil.size 
                            
                            tile_to_process = tile_pil
                            delta_w, delta_h = TILE_SIZE - tile_w_orig, TILE_SIZE - tile_h_orig
                            if delta_w > 0 or delta_h > 0:
                                tile_to_process = ImageOps.expand(tile_pil, (0, 0, delta_w, delta_h), fill=(0,0,0)) 

                            batch_tiles_data.append(normalize(to_tensor(tile_to_process)))
                            batch_coords.append({'x': x_inf_start, 'y': y_inf_start, 'w': tile_w_orig, 'h': tile_h_orig})
                            
                            if metrics_enabled: 
                                gt_tile_pil = gt_wsi.read_region((x_level0, y_level0), TARGET_INFERENCE_LEVEL, (TILE_SIZE, TILE_SIZE)).convert('L')
                                gt_tile_to_process = gt_tile_pil
                                if delta_w > 0 or delta_h > 0: 
                                    gt_tile_to_process = ImageOps.expand(gt_tile_pil, (0, 0, delta_w, delta_h), fill=0) 
                                batch_gt_labels.append((gt_transform(gt_tile_to_process) > 0).float())

                            processed_tiles_for_model += 1
                        except Exception: 
                            continue 
                    else:
                        tiles_skipped_by_mask += 1

                    is_last_tile = (r == rows - 1 and c == cols - 1)
                    if len(batch_tiles_data) == SEGMENTATION_BATCH_SIZE or (is_last_tile and len(batch_tiles_data) > 0):
                        try:
                            if is_last_tile and len(batch_tiles_data) > 0:
                                print(f"Zpracovávám poslední segmentační dávku ({len(batch_tiles_data)} dlaždic)...")

                            batch_tensor = torch.stack(batch_tiles_data).to(device)
                            predictions_output = model(batch_tensor) 
                            predictions_sigmoid = torch.sigmoid(predictions_output) 
                            
                            if metrics_enabled and batch_gt_labels: 
                                gt_batch_tensor = torch.stack(batch_gt_labels).to(device)
                                tp, fp, fn, tn = get_stats(predictions_sigmoid, gt_batch_tensor.long(), mode="binary", threshold=THRESHOLD)
                                wsi_total_tp += tp.sum().item()
                                wsi_total_fp += fp.sum().item()
                                wsi_total_fn += fn.sum().item()
                                wsi_total_tn += tn.sum().item()

                            predictions_np = predictions_sigmoid.cpu().numpy() 
                            for i in range(len(batch_coords)):
                                coords = batch_coords[i]
                                x, y, w, h = coords['x'], coords['y'], coords['w'], coords['h']
                                prediction_sum[y:y+h, x:x+w] += predictions_np[i, 0, :h, :w]
                                prediction_count[y:y+h, x:x+w] += 1
                        except Exception as e:
                            print(f"Chyba při zpracování segmentační dávky: {e}")
                            traceback.print_exc()
                        finally:
                            batch_tiles_data.clear(); batch_coords.clear(); batch_gt_labels.clear()

        print("\nZpracování dlaždic pro segmentaci dokončeno.")
        print(f"   Dlaždic zpracováno modelem (segmentace): {processed_tiles_for_model}")
        print(f"   Dlaždic přeskočeno maskou (segmentace): {tiles_skipped_by_mask}")
        
        results_to_return = {"WSI_Basename": wsi_basename, "Status": "Segmented_NoMetrics"} 
        if metrics_enabled and processed_tiles_for_model > 0: 
            print("\n--- Metriky pro WSI (vyhodnoceno na úrovni dlaždic sečtením TP/FP/FN/TN) ---")
            
            tp_t = torch.tensor(wsi_total_tp, dtype=torch.long)
            fp_t = torch.tensor(wsi_total_fp, dtype=torch.long)
            fn_t = torch.tensor(wsi_total_fn, dtype=torch.long)
            tn_t = torch.tensor(wsi_total_tn, dtype=torch.long)

            if (tp_t + fn_t).item() == 0: 
                wsi_recall_val = float('nan') 
            else:
                wsi_recall_val = recall(tp=tp_t, fp=fp_t, fn=fn_t, tn=tn_t, reduction='micro').item()

            if (tp_t + fp_t).item() == 0: 
                wsi_precision_val = float('nan') 
            else:
                wsi_precision_val = precision(tp=tp_t, fp=fp_t, fn=fn_t, tn=tn_t, reduction='micro').item()
            
            if (2 * tp_t + fp_t + fn_t).item() == 0: 
                 wsi_dice_val = float('nan')
            else:
                wsi_dice_val = f1_score(tp=tp_t, fp=fp_t, fn=fn_t, tn=tn_t, reduction='micro').item()

            if (tp_t + fp_t + fn_t).item() == 0: 
                wsi_iou_val = float('nan')
            else:
                wsi_iou_val = iou_score(tp=tp_t, fp=fp_t, fn=fn_t, tn=tn_t, reduction='micro').item()
            
            print(f"   Dice (F1-Score): {wsi_dice_val:.5f}" if not np.isnan(wsi_dice_val) else "   Dice (F1-Score): NaN")
            print(f"   IoU (Jaccard):   {wsi_iou_val:.5f}" if not np.isnan(wsi_iou_val) else "   IoU (Jaccard):   NaN")
            print(f"   Recall:          {wsi_recall_val:.5f}" if not np.isnan(wsi_recall_val) else "   Recall:          NaN")
            print(f"   Precision:       {wsi_precision_val:.5f}" if not np.isnan(wsi_precision_val) else "   Precision:       NaN")
            print("---------------------------------------------------------")

            results_to_return = {
                "WSI_Basename": wsi_basename, "Status": "Segmented_Metrics_Calculated",
                "Dice": wsi_dice_val, "IoU": wsi_iou_val, 
                "Recall": wsi_recall_val, "Precision": wsi_precision_val
            }
        elif processed_tiles_for_model == 0 and metrics_enabled:
             results_to_return = {"WSI_Basename": wsi_basename, "Status": "Segmented_NoTilesForMetrics"}
        elif processed_tiles_for_model == 0 and not metrics_enabled:
             results_to_return = {"WSI_Basename": wsi_basename, "Status": "Segmented_NoTiles_NoGT"}


        if processed_tiles_for_model > 0 : 
            print("Průměrování pravděpodobností pro finální masku...")
            average_probability = np.zeros_like(prediction_sum, dtype=np.float32)
            np.divide(prediction_sum, prediction_count, out=average_probability, where=prediction_count > 0)
            final_mask = (average_probability >= THRESHOLD) 

            print(f"Ukládání finální predikované masky do {output_hdf5_path}...")
            os.makedirs(os.path.dirname(output_hdf5_path), exist_ok=True)
            with h5py.File(output_hdf5_path, "w") as hdf5_file:
                hdf5_file.create_dataset("mask", data=final_mask, dtype=bool, chunks=(TILE_SIZE, TILE_SIZE), compression="gzip")
            print("Ukládání predikované masky dokončeno.")
        else:
            print(f"INFO: Nebyly zpracovány žádné dlaždice pro {wsi_basename}. Predikovaná maska se neukládá.")
            if results_to_return["Status"].startswith("Segmented"):
                 results_to_return["Status"] = "Skipped_Segment_NoTissueTiles"
        return results_to_return

    except Exception as main_err:
        print(f"NEOČEKÁVANÁ CHYBA během plné segmentace {os.path.basename(wsi_image_path)}: {main_err}")
        traceback.print_exc()
        return {"WSI_Basename": wsi_basename, "Status": f"Error_Segmentation_{type(main_err).__name__}"}
    finally:
        if wsi is not None: wsi.close() # Zavři pouze pokud byl objekt úspěšně vytvořen
        if gt_wsi is not None: gt_wsi.close() # Zavři pouze pokud byl objekt úspěšně vytvořen
        print("WSI a GT soubory (segmentace) uzavřeny.")
        print(f"Doba plné segmentace souboru: {time.time() - script_start_time:.2f} sekund.")
        del prediction_sum, prediction_count, wsi, gt_wsi 
        gc.collect()


# --- HLAVNÍ SPUŠTĚCÍ BLOK ---
if __name__ == "__main__":
    pass 

    main_start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1, activation=None) 
    
    try:
        state_data = torch.load(model_weights_path, map_location=device, weights_only=False) 
        if isinstance(state_data, dict) and "model_state_dict" in state_data: # Zkontroluj, zda je to slovník a obsahuje klíč
            model.load_state_dict(state_data["model_state_dict"])
        elif isinstance(state_data, dict): # Pokud je to slovník, ale nemá správný klíč (možná je to přímo state_dict)
             model.load_state_dict(state_data)
        else: # Pokud to není slovník (např. přímo tenzor nebo jiný objekt)
             raise TypeError(f"Očekáván slovník pro state_dict, ale přijato {type(state_data)}")
        print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
        model.to(device)
        model.eval() 
    except FileNotFoundError:
        print(f"FATÁLNÍ CHYBA: Soubor s váhami modelu nebyl nalezen na cestě: {model_weights_path}")
        exit()
    except Exception as e:
        print(f"FATÁLNÍ CHYBA: Nepodařilo se načíst váhy modelu: {e}")
        traceback.print_exc()
        exit()

    os.makedirs(OUTPUT_PRED_DIR, exist_ok=True)
    
    results_csv_path = os.path.join(OUTPUT_PRED_DIR, "evaluation_results_with_classification.csv")
    csv_header = ["WSI_File", "Classification_Status", "Prediction_File", "Dice", "IoU", "Recall", "Precision", "Segmentation_Notes"]
    
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
        print(f"\nNalezeno {len(wsi_files_to_process)} souborů 'test_*.tif' ke zpracování.")
        
        for wsi_full_path in wsi_files_to_process:
            wsi_basename_loop = os.path.basename(wsi_full_path)
            wsi_number_loop = os.path.splitext(wsi_basename_loop)[0].split('_')[-1]
            output_pred_filename = f"pred_{wsi_number_loop}.h5"
            output_pred_full_path = os.path.join(OUTPUT_PRED_DIR, output_pred_filename)

            segmentation_notes = "" 
            classification_status = "ERROR_BEFORE_CLASSIFICATION_CALL" # Default pro případ chyby před voláním

            try: 
                classification_status = classify_wsi_quickly(
                    wsi_full_path, TISSUE_MASK_DIR, model, device,
                    tumor_tile_threshold=TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION,
                    min_positive_tiles_for_tumorous=MIN_POSITIVE_TILES_FOR_TUMOROUS
                )
                
                csv_row_data = {
                    "WSI_File": wsi_basename_loop,
                    "Classification_Status": classification_status,
                    "Prediction_File": "", "Dice": "N/A", "IoU": "N/A",
                    "Recall": "N/A", "Precision": "N/A", "Segmentation_Notes": ""
                }

                if classification_status == "tumorous":
                    print(f"WSI {wsi_basename_loop} klasifikováno jako '{classification_status}', pokračuji plnou segmentací.")
                    
                    segmentation_results = process_single_wsi(
                        wsi_full_path, output_pred_full_path, TISSUE_MASK_DIR, model, device
                    )
                    
                    if segmentation_results:
                        csv_row_data["Prediction_File"] = output_pred_filename if "Segmented" in segmentation_results.get("Status", "") and "NoTiles" not in segmentation_results.get("Status", "") and "Error" not in segmentation_results.get("Status","") else ""
                        
                        if segmentation_results.get("Status") == "Segmented_Metrics_Calculated":
                            csv_row_data["Dice"] = f"{segmentation_results['Dice']:.5f}" if not np.isnan(segmentation_results['Dice']) else "NaN"
                            csv_row_data["IoU"] = f"{segmentation_results['IoU']:.5f}" if not np.isnan(segmentation_results['IoU']) else "NaN"
                            csv_row_data["Recall"] = f"{segmentation_results['Recall']:.5f}" if not np.isnan(segmentation_results['Recall']) else "NaN"
                            csv_row_data["Precision"] = f"{segmentation_results['Precision']:.5f}" if not np.isnan(segmentation_results['Precision']) else "NaN"
                            segmentation_notes = "Metrics Calculated"
                        else: 
                            segmentation_notes = segmentation_results.get("Status", "Unknown Segmentation Status")
                    else: 
                        segmentation_notes = "Error_Segmentation_ReturnedNone"
                
                elif classification_status == "healthy":
                    print(f"WSI {wsi_basename_loop} klasifikováno jako '{classification_status}'. Segmentace se přeskakuje.")
                    segmentation_notes = "Skipped_HealthyClassification"
                else: 
                    print(f"WSI {wsi_basename_loop} má stav klasifikace '{classification_status}'. Segmentace se přeskakuje.")
                    segmentation_notes = f"Skipped_{classification_status}"

                csv_row_data["Segmentation_Notes"] = segmentation_notes
                
                try:
                    with open(results_csv_path, 'a', newline='', encoding='utf-8') as csvfile: 
                        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                        writer.writerow(csv_row_data)
                    print(f"Výsledky/stav pro '{wsi_basename_loop}' byly úspěšně zapsány do CSV.")
                except IOError as e_csv:
                    print(f"CHYBA při zápisu do CSV souboru pro {wsi_basename_loop}: {e_csv}")

            except Exception as loop_err: 
                print(f"\n!!!!!! NEOČEKÁVANÁ FATÁLNÍ CHYBA VE VNĚJŠÍ SMYČCE pro soubor {wsi_full_path}: {loop_err} !!!!!!")
                traceback.print_exc() 
                try:
                    with open(results_csv_path, 'a', newline='', encoding='utf-8') as csvfile: 
                        writer = csv.writer(csvfile)
                        writer.writerow([wsi_basename_loop, classification_status, "", "N/A", "N/A", "N/A", "N/A", f"FATAL_OUTER_LOOP_ERROR: {str(loop_err).replace('\n', ' ').replace('\r', ' ')}"])
                except Exception as e_csv_fatal:
                     print(f"Nelze zapsat ani chybový stav do CSV pro {wsi_basename_loop}: {e_csv_fatal}")
                print("Pokračuji dalším souborem...")
                continue 

    total_end_time = time.time()
    print("\n" + "="*80)
    print("VŠECHNY SOUBORY ZPRACOVÁNY (nebo přeskočeny).")
    print(f"Celkový čas běhu skriptu: {(total_end_time - main_start_time) / 60:.2f} minut.")
    print(f"Výsledky a stavy uloženy v: {results_csv_path}")
    print("="*80)
# --- END OF FINAL COMPLETE FILE ---