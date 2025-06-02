import os
openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
os.add_dll_directory(openslide_dll_path)
import traceback
import csv
import datetime
import pandas as pd

import h5py
import numpy as np
import torch
import segmentation_models_pytorch as smp

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# --- Konfigurace ---
MASK_COUNT = 130
OUTPUT_PRED_DIR = r"C:\Users\USER\Desktop\test_preds\vanilla_unet_2"  # Složka s pred_XXX.h5 soubory
GT_MASK_DIR = r"F:\wsi_dir_test"  # Složka s mask_XXX.tif soubory
CSV_REFERENCE_PATH = r"C:\Users\USER\Desktop\reference_fix.csv"  # CSV s informacemi o nádorových snímcích

# <<< PŘIDANÁ KONFIGURACE >>>
# Adresář s nízkoúrovňovými maskami tkáně (numpy pole, např. lowres_mask_001.npy)
# Pokud je None nebo maska pro daný snímek neexistuje, budou zpracovány všechny dlaždice.
# !!!!! NASTAVTE PROSÍM SPRÁVNOU CESTU !!!!!
LOW_RES_MASK_DIR = r"C:\Users\USER\Desktop\colab_unet\test_lowres_masky" 
# OpenSlide úroveň, ze které byly vytvořeny nízkoúrovňové masky (pro ověření a škálování)
# Např. pokud masky odpovídají úrovni 6 WSI.
LOW_RES_MASK_OPENSLIDE_LEVEL = 6 
# <<< KONEC PŘIDANÉ KONFIGURACE >>>

# Vytvoření CSV souboru s časovým razítkem pro jednotlivé výsledky
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_PRED_DIR, f"individual_results_{timestamp}.csv")

# Hlavička CSV souboru pro jednotlivé snímky
csv_header = ['Mask_ID', 'Processed_Tiles', 'TP', 'FP', 'FN', 'TN', 'Is_Cancer', 'Status']

# Vytvoření CSV souboru a zapsání hlavičky
with open(csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

print(f"CSV soubor pro jednotlivé výsledky vytvořen: {csv_path}")

# Načtení CSV s informacemi o nádorových snímcích
try:
    df_references = pd.read_csv(CSV_REFERENCE_PATH)
    print(f"Načten CSV s referencemi: {CSV_REFERENCE_PATH}")
    print(f"Počet záznamů v CSV: {len(df_references)}")
    
    cancer_map = {}
    for _, row in df_references.iterrows():
        id_str = str(row['id'])
        if id_str.startswith('test_'):
            number = id_str.split('_')[1]
            is_cancer = 1 if str(row['is_cancer']).lower() in ['tumor', '1', 'true'] else 0
            cancer_map[number] = is_cancer
    
    print(f"Nalezeno {sum(cancer_map.values())} nádorových snímků z {len(cancer_map)} celkových")
    
except Exception as e:
    print(f"Chyba při načítání CSV souboru: {e}")
    cancer_map = {}

# Globální akumulátory pro všechny snímky
global_all_tp, global_all_fp, global_all_fn, global_all_tn = 0, 0, 0, 0
# Globální akumulátory pouze pro nádorové snímky
global_tumor_tp, global_tumor_fp, global_tumor_fn, global_tumor_tn = 0, 0, 0, 0

processed_slides_total = 0
processed_slides_tumor = 0

for i in range(1, MASK_COUNT + 1):
    mask_id = f"{i:03d}"
    GT_SLIDE_PATH = os.path.join(GT_MASK_DIR, f"mask_{mask_id}.tif")
    PRED_HDF5_PATH = os.path.join(OUTPUT_PRED_DIR, f"pred_{mask_id}.h5")
    HDF5_DATASET_NAME = "mask"
    TILE_SIZE = 4096
    TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK = 2

    print(f"\n{'='*50}")
    print(f"Zpracovávám snímek {mask_id}/{MASK_COUNT}: {os.path.basename(GT_SLIDE_PATH)}")
    print(f"{'='*50}")

    is_cancer_slide = cancer_map.get(mask_id, 0)
    cancer_status = "Cancer" if is_cancer_slide else "Normal"
    print(f"Stav snímku podle CSV: {cancer_status}")

    slide_tp, slide_fp, slide_fn, slide_tn = 0, 0, 0, 0
    processed_tiles = 0
    wsi = None
    status = "success"

    low_res_tissue_mask = None
    scale_map_coords_from_dzg_target_to_lowres_mask = None

    try:
        if not os.path.exists(GT_SLIDE_PATH):
            print(f"Chyba: GT soubor neexistuje: {GT_SLIDE_PATH}")
            status = "gt_not_found"
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, status])
            continue

        if not os.path.exists(PRED_HDF5_PATH):
            print(f"Chyba: Predikovaný soubor neexistuje: {PRED_HDF5_PATH}")
            status = "pred_not_found"
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, status])
            continue

        wsi = OpenSlide(GT_SLIDE_PATH)
        dz_gen = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=0, limit_bounds=False)

        # <<< PŘIDANÁ LOGIKA PRO NAČTENÍ LOW-RES MASKY A VÝPOČET ŠKÁLOVÁNÍ >>>
        if LOW_RES_MASK_DIR:
            LOW_RES_MASK_PATH = os.path.join(LOW_RES_MASK_DIR, f"mask_{mask_id}.npy")
            try:
                if os.path.exists(LOW_RES_MASK_PATH):
                    low_res_tissue_mask = np.load(LOW_RES_MASK_PATH)
                    print(f"Načtena low-res maska tkáně: {LOW_RES_MASK_PATH}, tvar (HxW): {low_res_tissue_mask.shape}")
                else:
                    print(f"Info: Low-res maska tkáně neexistuje: {LOW_RES_MASK_PATH}. Bude se iterovat přes všechny dlaždice pro tento snímek.")
            except Exception as e_lr_mask:
                print(f"Chyba při načítání low-res masky tkáně {LOW_RES_MASK_PATH}: {e_lr_mask}. Bude se iterovat přes všechny dlaždice pro tento snímek.")
                low_res_tissue_mask = None
        else:
            print("Info: Adresář pro low-res masky (LOW_RES_MASK_DIR) není nastaven. Bude se iterovat přes všechny dlaždice.")

        if low_res_tissue_mask is not None:
            try:
                # Získání rozměrů WSI úrovně, které by měla low-res maska odpovídat
                if LOW_RES_MASK_OPENSLIDE_LEVEL >= wsi.level_count:
                    print(f"Chyba: LOW_RES_MASK_OPENSLIDE_LEVEL ({LOW_RES_MASK_OPENSLIDE_LEVEL}) je mimo rozsah dostupných úrovní WSI ({wsi.level_count}). Low-res maska nebude použita.")
                    low_res_tissue_mask = None
                else:
                    low_res_wsi_level_dims_wh = wsi.level_dimensions[LOW_RES_MASK_OPENSLIDE_LEVEL]
                    loaded_lr_mask_h, loaded_lr_mask_w = low_res_tissue_mask.shape
                    low_res_mask_effective_dims_wh = (loaded_lr_mask_w, loaded_lr_mask_h) # W, H

                    if (loaded_lr_mask_w != low_res_wsi_level_dims_wh[0] or
                        loaded_lr_mask_h != low_res_wsi_level_dims_wh[1]):
                        print(f"Varování: Rozměry načtené low-res masky (HxW: {loaded_lr_mask_h}x{loaded_lr_mask_w}) "
                              f"neodpovídají očekávaným rozměrům WSI úrovně {LOW_RES_MASK_OPENSLIDE_LEVEL} (WxH: {low_res_wsi_level_dims_wh[0]}x{low_res_wsi_level_dims_wh[1]}). "
                              f"Použijí se skutečné rozměry načtené masky (WxH: {loaded_lr_mask_w}x{loaded_lr_mask_h}) pro škálování.")
                    
                    # Najít cílovou DZG úroveň (musí se shodovat s TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK)
                    target_os_dims_wh_check = wsi.level_dimensions[TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK]
                    dzg_target_level_check = -1
                    for lvl_idx_check in range(dz_gen.level_count):
                        if dz_gen.level_dimensions[lvl_idx_check] == target_os_dims_wh_check:
                            dzg_target_level_check = lvl_idx_check
                            break
                    if dzg_target_level_check == -1:
                        raise ValueError("Nepodařilo se najít odpovídající DZG úroveň pro TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK.")

                    processing_level_dims_wh = dz_gen.level_dimensions[dzg_target_level_check] # W, H
                    
                    scale_x_proc_to_lowres_idx = low_res_mask_effective_dims_wh[0] / processing_level_dims_wh[0]
                    scale_y_proc_to_lowres_idx = low_res_mask_effective_dims_wh[1] / processing_level_dims_wh[1]
                    scale_map_coords_from_dzg_target_to_lowres_mask = (scale_x_proc_to_lowres_idx, scale_y_proc_to_lowres_idx)
                    print(f"Škálovací faktory z úrovně DZG ({dzg_target_level_check}, rozměry WxH: {processing_level_dims_wh}) na low-res masku (rozměry WxH: {low_res_mask_effective_dims_wh}): x={scale_x_proc_to_lowres_idx:.4f}, y={scale_y_proc_to_lowres_idx:.4f}")

            except Exception as e_scale:
                print(f"Chyba při výpočtu škálovacích faktorů pro low-res masku: {e_scale}. Low-res maska nebude použita.")
                low_res_tissue_mask = None
                scale_map_coords_from_dzg_target_to_lowres_mask = None
        # <<< KONEC PŘIDANÉ LOGIKY >>>


        with h5py.File(PRED_HDF5_PATH, "r") as f:
            if HDF5_DATASET_NAME not in f:
                print(f"Chyba: Dataset '{HDF5_DATASET_NAME}' nenalezen v {PRED_HDF5_PATH}")
                status = "dataset_not_found"
                with open(csv_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, status])
                if wsi: wsi.close()
                continue
            hdf5_mask_np_raw = f[HDF5_DATASET_NAME][:]

        hdf5_mask_np = (hdf5_mask_np_raw > 0).astype(np.uint8)
        print(f"Načtená HDF5 maska (tvar HxW): {hdf5_mask_np.shape}, typ: {hdf5_mask_np.dtype}")

        target_os_dims_wh = wsi.level_dimensions[TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK]

        if hdf5_mask_np.shape != (target_os_dims_wh[1], target_os_dims_wh[0]):
            print(f"Varování: Tvar HDF5 masky {hdf5_mask_np.shape} (HxW) neodpovídá OpenSlide úrovni {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} {(target_os_dims_wh[1], target_os_dims_wh[0])} (HxW).")

        dzg_target_level = -1
        for lvl_idx in range(dz_gen.level_count):
            if dz_gen.level_dimensions[lvl_idx] == target_os_dims_wh:
                dzg_target_level = lvl_idx
                break
        
        if dzg_target_level == -1:
            print(f"Chyba: Nepodařilo se najít odpovídající DZG úroveň.")
            status = "dzg_level_not_found"
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, status])
            if wsi: wsi.close()
            continue

        print(f"Cílová OpenSlide úroveň: {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (Rozměry WxH: {target_os_dims_wh})")
        print(f"Odpovídající DZG úroveň: {dzg_target_level}")

        level_total_w, level_total_h = dz_gen.level_dimensions[dzg_target_level]
        num_cols, num_rows = dz_gen.level_tiles[dzg_target_level]
        print(f"Očekávaný počet dlaždic: {num_rows} řádků, {num_cols} sloupců.")

        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                # Předběžný výpočet rozměrů dlaždice pro kontrolu s low-res maskou
                _x_on_level_check = c_idx * TILE_SIZE
                _y_on_level_check = r_idx * TILE_SIZE
                _current_tile_w_check = min(TILE_SIZE, level_total_w - _x_on_level_check)
                _current_tile_h_check = min(TILE_SIZE, level_total_h - _y_on_level_check)

                if _current_tile_w_check <= 0 or _current_tile_h_check <= 0:
                    continue
                
                # <<< PŘIDANÁ LOGIKA PRO PŘESKOČENÍ DLAŽDIC NA ZÁKLADĚ LOW-RES MASKY >>>
                if low_res_tissue_mask is not None and scale_map_coords_from_dzg_target_to_lowres_mask is not None:
                    tile_center_x_on_dzg_level = _x_on_level_check + (_current_tile_w_check / 2)
                    tile_center_y_on_dzg_level = _y_on_level_check + (_current_tile_h_check / 2)
                    
                    lr_mask_h_actual, lr_mask_w_actual = low_res_tissue_mask.shape # H, W

                    low_res_x_idx = int(tile_center_x_on_dzg_level * scale_map_coords_from_dzg_target_to_lowres_mask[0])
                    low_res_y_idx = int(tile_center_y_on_dzg_level * scale_map_coords_from_dzg_target_to_lowres_mask[1])

                    low_res_x_idx = min(max(0, low_res_x_idx), lr_mask_w_actual - 1)
                    low_res_y_idx = min(max(0, low_res_y_idx), lr_mask_h_actual - 1)

                    if low_res_tissue_mask[low_res_y_idx, low_res_x_idx] == 0: # Předpoklad: 0 = pozadí
                        continue 
                # <<< KONEC PŘIDANÉ LOGIKY >>>

                try:
                    x_on_level = _x_on_level_check 
                    y_on_level = _y_on_level_check
                    current_tile_w = _current_tile_w_check
                    current_tile_h = _current_tile_h_check
                    
                    # Tato kontrola je již provedena výše, ale pro jistotu může zůstat
                    # if current_tile_w <= 0 or current_tile_h <= 0: 
                    #     continue

                    tile_gt_pil = dz_gen.get_tile(dzg_target_level, (c_idx, r_idx))
                    gt_tile_np_rgba = np.array(tile_gt_pil)
                    gt_tile_binary_hw = (gt_tile_np_rgba[:, :, 0] > 0).astype(np.uint8) if gt_tile_np_rgba.ndim == 3 else (gt_tile_np_rgba > 0).astype(np.uint8)
                    pred_tile_binary_hw = hdf5_mask_np[y_on_level : y_on_level + current_tile_h, x_on_level : x_on_level + current_tile_w]

                    if gt_tile_binary_hw.shape != pred_tile_binary_hw.shape or gt_tile_binary_hw.size == 0: 
                        continue
                    
                    pred_torch = torch.from_numpy(pred_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    gt_torch = torch.from_numpy(gt_tile_binary_hw).long().unsqueeze(0).unsqueeze(0)

                    stats = smp.metrics.get_stats(pred_torch, gt_torch, mode='binary', threshold=0.5)
                    tile_tp = stats[0].item()
                    tile_fp = stats[1].item()
                    tile_fn = stats[2].item()
                    tile_tn = stats[3].item()
                    
                    slide_tp += tile_tp
                    slide_fp += tile_fp
                    slide_fn += tile_fn
                    slide_tn += tile_tn
                    
                    processed_tiles += 1

                except Exception as e_tile:
                    print(f"Chyba při zpracování dlaždice ({r_idx},{c_idx}): {e_tile}")
                    continue

        if processed_tiles > 0:
            print(f"\n--- DOKONČENO: Zpracováno {processed_tiles} dlaždic ---")
            print(f"TP: {slide_tp}, FP: {slide_fp}, FN: {slide_fn}, TN: {slide_tn}")
            
            global_all_tp += slide_tp
            global_all_fp += slide_fp
            global_all_fn += slide_fn
            global_all_tn += slide_tn
            processed_slides_total += 1
            
            if is_cancer_slide:
                global_tumor_tp += slide_tp
                global_tumor_fp += slide_fp
                global_tumor_fn += slide_fn
                global_tumor_tn += slide_tn
                processed_slides_tumor += 1
            
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    mask_id, processed_tiles, int(slide_tp), int(slide_fp), 
                    int(slide_fn), int(slide_tn), cancer_status, status
                ])
        else:
            print("Nebyly zpracovány žádné dlaždice (možná kvůli filtraci low-res maskou nebo jiným chybám).")
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, "no_tiles_processed"])

    except Exception as e:
        print(f"Chyba při zpracování snímku {mask_id}: {e}")
        traceback.print_exc()
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, f"error: {str(e)[:100]}"])
    
    finally:
        if wsi is not None:
            try:
                wsi.close()
            except:
                pass

# ============================================================================
# VÝPOČET GLOBÁLNÍCH METRIK NA KONCI
# ============================================================================

def calculate_metrics(tp, fp, fn, tn, label):
    """Vypočítá metriky z TP, FP, FN, TN hodnot."""
    print(f"\n{'='*60}")
    print(f"GLOBÁLNÍ METRIKY - {label}")
    print(f"{'='*60}")
    
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")
    print(f"Celkový počet pixelů: {tp + fp + fn + tn}")
    
    if (tp + fp + fn + tn) == 0:
        print("VAROVÁNÍ: Žádné pixely k vyhodnocení!")
        return {
            'dice': float('nan'), 'iou': float('nan'), 'recall': float('nan'),
            'precision': float('nan'), 'specificity': float('nan'), 'accuracy': float('nan')
        } # Return NaN for all if no pixels
    
    metrics = {}
    
    if (2 * tp + fp + fn) == 0: # Dice
        metrics['dice'] = 1.0 if tp == 0 else 0.0 # If TP=0, and 2TP+FP+FN=0, means FP+FN=0 -> perfect for no positives
    else:
        metrics['dice'] = (2 * tp) / (2 * tp + fp + fn)
    
    if (tp + fp + fn) == 0: # IoU
        metrics['iou'] = 1.0 if tp == 0 else 0.0 # If TP=0, and TP+FP+FN=0, means FP+FN=0 -> perfect for no positives
    else:
        metrics['iou'] = tp / (tp + fp + fn)
    
    if (tp + fn) == 0: # Recall
        metrics['recall'] = 1.0 if tp == 0 else float('nan') # If TP=0 and TP+FN=0, means FN=0. If no positives GT, recall is 1. If TP>0, it's NaN.
                                                        # More standard: NaN if TP+FN=0
    else:
        metrics['recall'] = tp / (tp + fn)
    
    if (tp + fp) == 0: # Precision
        metrics['precision'] = 1.0 if tp == 0 else float('nan') # If TP=0 and TP+FP=0, means FP=0. If no positive preds, precision is 1.
                                                           # More standard: NaN if TP+FP=0
    else:
        metrics['precision'] = tp / (tp + fp)
    
    if (tn + fp) == 0: # Specificity
        metrics['specificity'] = 1.0 if tn == 0 else float('nan') # If TN=0 and TN+FP=0, means FP=0. If no negatives GT, specificity is 1.
                                                            # More standard: NaN if TN+FP=0
    else:
        metrics['specificity'] = tn / (tn + fp)
    
    metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    
    print(f"\nVýsledné metriky:")
    for key, value in metrics.items():
        print(f"{key.capitalize():<15}: {value:.5f}" if not np.isnan(value) else f"{key.capitalize():<15}: NaN")
        
    return metrics

if processed_slides_total > 0:
    all_metrics = calculate_metrics(
        global_all_tp, global_all_fp, global_all_fn, global_all_tn,
        f"VŠECHNY SNÍMKY ({processed_slides_total} snímků)"
    )
else:
    print("\nŽádné snímky nebyly úspěšně zpracovány pro celkové metriky!")
    all_metrics = {} # Initialize empty if no slides processed

if processed_slides_tumor > 0:
    tumor_metrics = calculate_metrics(
        global_tumor_tp, global_tumor_fp, global_tumor_fn, global_tumor_tn,
        f"POUZE NÁDOROVÉ SNÍMKY ({processed_slides_tumor} snímků)"
    )
else:
    print(f"\n{'='*60}")
    print("POUZE NÁDOROVÉ SNÍMKY")
    print(f"{'='*60}")
    print("Žádné nádorové snímky nebyly úspěšně zpracovány!")
    tumor_metrics = {} # Initialize empty if no tumor slides processed

global_metrics_csv_path = os.path.join(OUTPUT_PRED_DIR, f"global_metrics_{timestamp}.csv")
with open(global_metrics_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Category', 'TP', 'FP', 'FN', 'TN', 'Dice', 'IoU', 'Recall', 'Precision', 'Specificity', 'Accuracy', 'Slides_Count'])
    
    if processed_slides_total > 0 and all_metrics:
        writer.writerow([
            'All_Slides',
            int(global_all_tp), int(global_all_fp), int(global_all_fn), int(global_all_tn),
            f"{all_metrics.get('dice', float('nan')):.5f}" if not np.isnan(all_metrics.get('dice', float('nan'))) else "NaN",
            f"{all_metrics.get('iou', float('nan')):.5f}" if not np.isnan(all_metrics.get('iou', float('nan'))) else "NaN",
            f"{all_metrics.get('recall', float('nan')):.5f}" if not np.isnan(all_metrics.get('recall', float('nan'))) else "NaN",
            f"{all_metrics.get('precision', float('nan')):.5f}" if not np.isnan(all_metrics.get('precision', float('nan'))) else "NaN",
            f"{all_metrics.get('specificity', float('nan')):.5f}" if not np.isnan(all_metrics.get('specificity', float('nan'))) else "NaN",
            f"{all_metrics.get('accuracy', float('nan')):.5f}" if not np.isnan(all_metrics.get('accuracy', float('nan'))) else "NaN",
            processed_slides_total
        ])
    
    if processed_slides_tumor > 0 and tumor_metrics:
        writer.writerow([
            'Tumor_Slides_Only',
            int(global_tumor_tp), int(global_tumor_fp), int(global_tumor_fn), int(global_tumor_tn),
            f"{tumor_metrics.get('dice', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('dice', float('nan'))) else "NaN",
            f"{tumor_metrics.get('iou', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('iou', float('nan'))) else "NaN",
            f"{tumor_metrics.get('recall', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('recall', float('nan'))) else "NaN",
            f"{tumor_metrics.get('precision', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('precision', float('nan'))) else "NaN",
            f"{tumor_metrics.get('specificity', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('specificity', float('nan'))) else "NaN",
            f"{tumor_metrics.get('accuracy', float('nan')):.5f}" if not np.isnan(tumor_metrics.get('accuracy', float('nan'))) else "NaN",
            processed_slides_tumor
        ])

print(f"\n{'='*60}")
print("SOUBORY VÝSTUPŮ")
print(f"{'='*60}")
print(f"Jednotlivé výsledky: {csv_path}")
print(f"Globální metriky:    {global_metrics_csv_path}")
print("\nDokončeno.")