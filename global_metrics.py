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
MASK_COUNT = 5
OUTPUT_PRED_DIR = r"C:\Users\USER\Desktop\test_preds\try"  # Složka s pred_XXX.h5 soubory
GT_MASK_DIR = r"F:\wsi_dir_test"  # Složka s mask_XXX.tif soubory
CSV_REFERENCE_PATH = r"C:\Users\USER\Desktop\reference_fix.csv"  # CSV s informacemi o nádorových snímcích

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
    
    # Vytvoření mapy id -> is_cancer
    cancer_map = {}
    for _, row in df_references.iterrows():
        # Extrakce čísla z id (např. "test_001" -> "001")
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

    # Kontrola, zda je snímek nádorový podle CSV
    is_cancer_slide = cancer_map.get(mask_id, 0)  # Výchozí hodnota 0 (zdravý)
    cancer_status = "Cancer" if is_cancer_slide else "Normal"
    print(f"Stav snímku podle CSV: {cancer_status}")

    # Inicializace akumulátorů pro aktuální snímek
    slide_tp, slide_fp, slide_fn, slide_tn = 0, 0, 0, 0
    processed_tiles = 0
    wsi = None
    status = "success"

    try:
        # Kontrola existence souborů
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

        with h5py.File(PRED_HDF5_PATH, "r") as f:
            if HDF5_DATASET_NAME not in f:
                print(f"Chyba: Dataset '{HDF5_DATASET_NAME}' nenalezen v {PRED_HDF5_PATH}")
                status = "dataset_not_found"
                
                with open(csv_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([mask_id, 0, 0, 0, 0, 0, cancer_status, status])
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
            continue

        print(f"Cílová OpenSlide úroveň: {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (Rozměry WxH: {target_os_dims_wh})")
        print(f"Odpovídající DZG úroveň: {dzg_target_level}")

        level_total_w, level_total_h = dz_gen.level_dimensions[dzg_target_level]
        num_cols, num_rows = dz_gen.level_tiles[dzg_target_level]
        print(f"Očekávaný počet dlaždic: {num_rows} řádků, {num_cols} sloupců.")

        # Smyčka přes dlaždice - AKUMULACE TP, FP, FN, TN
        for r_idx in range(num_rows):
            for c_idx in range(num_cols):
                try:
                    x_on_level = c_idx * TILE_SIZE
                    y_on_level = r_idx * TILE_SIZE
                    current_tile_w = min(TILE_SIZE, level_total_w - x_on_level)
                    current_tile_h = min(TILE_SIZE, level_total_h - y_on_level)

                    if current_tile_w <= 0 or current_tile_h <= 0: 
                        continue

                    tile_gt_pil = dz_gen.get_tile(dzg_target_level, (c_idx, r_idx))
                    gt_tile_np_rgba = np.array(tile_gt_pil)
                    gt_tile_binary_hw = (gt_tile_np_rgba[:, :, 0] > 0).astype(np.uint8) if gt_tile_np_rgba.ndim == 3 else (gt_tile_np_rgba > 0).astype(np.uint8)
                    pred_tile_binary_hw = hdf5_mask_np[y_on_level : y_on_level + current_tile_h, x_on_level : x_on_level + current_tile_w]

                    if gt_tile_binary_hw.shape != pred_tile_binary_hw.shape or gt_tile_binary_hw.size == 0: 
                        continue
                    
                    pred_torch = torch.from_numpy(pred_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    gt_torch = torch.from_numpy(gt_tile_binary_hw).long().unsqueeze(0).unsqueeze(0)

                    # Výpočet TP, FP, FN, TN pro aktuální dlaždici
                    stats = smp.metrics.get_stats(pred_torch, gt_torch, mode='binary', threshold=0.5)
                    tile_tp = stats[0].item()
                    tile_fp = stats[1].item()
                    tile_fn = stats[2].item()
                    tile_tn = stats[3].item()
                    
                    # Akumulace pro aktuální snímek
                    slide_tp += tile_tp
                    slide_fp += tile_fp
                    slide_fn += tile_fn
                    slide_tn += tile_tn
                    
                    processed_tiles += 1

                except Exception as e_tile:
                    print(f"Chyba při zpracování dlaždice ({r_idx},{c_idx}): {e_tile}")
                    continue

        # Výpis výsledků pro aktuální snímek
        if processed_tiles > 0:
            print(f"\n--- DOKONČENO: Zpracováno {processed_tiles} dlaždic ---")
            print(f"TP: {slide_tp}, FP: {slide_fp}, FN: {slide_fn}, TN: {slide_tn}")
            
            # Akumulace do globálních čítačů
            global_all_tp += slide_tp
            global_all_fp += slide_fp
            global_all_fn += slide_fn
            global_all_tn += slide_tn
            processed_slides_total += 1
            
            # Akumulace pro nádorové snímky
            if is_cancer_slide:
                global_tumor_tp += slide_tp
                global_tumor_fp += slide_fp
                global_tumor_fn += slide_fn
                global_tumor_tn += slide_tn
                processed_slides_tumor += 1
            
            # Zápis do CSV
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    mask_id, 
                    processed_tiles, 
                    int(slide_tp), 
                    int(slide_fp), 
                    int(slide_fn), 
                    int(slide_tn), 
                    cancer_status, 
                    status
                ])
        else:
            print("Nebyly zpracovány žádné dlaždice.")
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
        return {}
    
    # Výpočet metrik
    metrics = {}
    
    # Dice (F1-Score)
    if (2 * tp + fp + fn) == 0:
        metrics['dice'] = 1.0  # Perfektní shoda při absenci pozitivních případů
    else:
        metrics['dice'] = (2 * tp) / (2 * tp + fp + fn)
    
    # IoU (Jaccard)
    if (tp + fp + fn) == 0:
        metrics['iou'] = 1.0  # Perfektní shoda při absenci pozitivních případů
    else:
        metrics['iou'] = tp / (tp + fp + fn)
    
    # Recall (Sensitivity)
    if (tp + fn) == 0:
        metrics['recall'] = float('nan')  # Nedefinováno
    else:
        metrics['recall'] = tp / (tp + fn)
    
    # Precision
    if (tp + fp) == 0:
        if (tp + fn) == 0:
            metrics['precision'] = 1.0  # Perfektní při absenci pozitivních případů
        else:
            metrics['precision'] = 0.0  # Žádné pozitivní predikce, ale existují pozitivní GT
    else:
        metrics['precision'] = tp / (tp + fp)
    
    # Specificity
    if (tn + fp) == 0:
        metrics['specificity'] = float('nan')  # Nedefinováno
    else:
        metrics['specificity'] = tn / (tn + fp)
    
    # Accuracy
    metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn)
    
    # Výpis metrik
    print(f"\nVýsledné metriky:")
    print(f"Dice (F1-Score): {metrics['dice']:.5f}")
    print(f"IoU (Jaccard):   {metrics['iou']:.5f}")
    print(f"Recall:          {metrics['recall']:.5f}" if not np.isnan(metrics['recall']) else "Recall:          NaN")
    print(f"Precision:       {metrics['precision']:.5f}")
    print(f"Specificity:     {metrics['specificity']:.5f}" if not np.isnan(metrics['specificity']) else "Specificity:     NaN")
    print(f"Accuracy:        {metrics['accuracy']:.5f}")
    
    return metrics

# Výpočet globálních metrik pro všechny snímky
if processed_slides_total > 0:
    all_metrics = calculate_metrics(
        global_all_tp, global_all_fp, global_all_fn, global_all_tn,
        f"VŠECHNY SNÍMKY ({processed_slides_total} snímků)"
    )
else:
    print("\nŽádné snímky nebyly úspěšně zpracovány!")

# Výpočet globálních metrik pouze pro nádorové snímky
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

# Uložení globálních metrik do samostatného CSV
global_metrics_csv_path = os.path.join(OUTPUT_PRED_DIR, f"global_metrics_{timestamp}.csv")
with open(global_metrics_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Category', 'TP', 'FP', 'FN', 'TN', 'Dice', 'IoU', 'Recall', 'Precision', 'Specificity', 'Accuracy', 'Slides_Count'])
    
    if processed_slides_total > 0:
        writer.writerow([
            'All_Slides',
            int(global_all_tp), int(global_all_fp), int(global_all_fn), int(global_all_tn),
            f"{all_metrics['dice']:.5f}",
            f"{all_metrics['iou']:.5f}",
            f"{all_metrics['recall']:.5f}" if not np.isnan(all_metrics['recall']) else "NaN",
            f"{all_metrics['precision']:.5f}",
            f"{all_metrics['specificity']:.5f}" if not np.isnan(all_metrics['specificity']) else "NaN",
            f"{all_metrics['accuracy']:.5f}",
            processed_slides_total
        ])
    
    if processed_slides_tumor > 0:
        writer.writerow([
            'Tumor_Slides_Only',
            int(global_tumor_tp), int(global_tumor_fp), int(global_tumor_fn), int(global_tumor_tn),
            f"{tumor_metrics['dice']:.5f}",
            f"{tumor_metrics['iou']:.5f}",
            f"{tumor_metrics['recall']:.5f}" if not np.isnan(tumor_metrics['recall']) else "NaN",
            f"{tumor_metrics['precision']:.5f}",
            f"{tumor_metrics['specificity']:.5f}" if not np.isnan(tumor_metrics['specificity']) else "NaN",
            f"{tumor_metrics['accuracy']:.5f}",
            processed_slides_tumor
        ])

print(f"\n{'='*60}")
print("SOUBORY VÝSTUPŮ")
print(f"{'='*60}")
print(f"Jednotlivé výsledky: {csv_path}")
print(f"Globální metriky:    {global_metrics_csv_path}")
print("\nDokončeno.")