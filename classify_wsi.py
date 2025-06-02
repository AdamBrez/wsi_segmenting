# --- SCRIPT FOR CLASSIFICATION THRESHOLD OPTIMIZATION ON VALIDATION SET ---
import os
# !!! ZDE NASTAVTE CESTU K OPENSLIDE DLL, pokud není v systémové PATH !!!
# Příklad:
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import glob
import time
import datetime
import math
import gc
import csv
import traceback
import itertools
from collections import defaultdict

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import numpy as np
import openslide
import segmentation_models_pytorch as smp

# Import funkce pro načtení dat
from new_load_data import load_and_split_data

# --- HLAVNÍ KONFIGURACE ---
# Cesty k datům (stejné jako v new_load_data.py)
CANCER_WSI_GT_DIR = r"C:\Users\USER\Desktop\wsi_dir"
CANCER_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\masky_healthy"
CANCER_LR_GT_DIR = r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky"
HEALTHY_WSI_DIR = r"C:\Users\USER\Desktop\normal_wsi"
HEALTHY_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\normal_lowres"

# Cesta k výstupní složce pro CSV soubory s výsledky optimalizace
OUTPUT_CLASSIFICATION_DIR = r"C:\Users\USER\Desktop\threshold_optimization_results"
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-24_00-11-56\best_weights_2025-05-24_00-11-56.pth"

TILE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CURRENT_MEAN = IMAGENET_MEAN
CURRENT_STD = IMAGENET_STD

tissue_mask_level_index = 6
TISSUE_THRESHOLD = 0.05
CLASSIFICATION_BATCH_SIZE = 24
TARGET_INFERENCE_LEVEL = 2

# --- PARAMETRY PRO OPTIMALIZACI ---
# Různé prahy pro klasifikaci jednotlivých dlaždic jako nádorové
TUMOR_TILE_THRESHOLDS_TO_TEST = [0.995, 0.9995, 0.99995]
# Různé minimální počty pozitivních dlaždic pro klasifikaci WSI jako nádorového
MIN_POSITIVE_TILES_TO_TEST = [10, 15, 20]

def get_ground_truth_label(wsi_path):
    """
    Určí ground truth label na základě cesty k WSI.
    Vrací 'tumorous' pro tumor_*.tif a 'healthy' pro normal_*.tif
    """
    basename = os.path.basename(wsi_path)
    if basename.startswith('tumor_'):
        return 'tumorous'
    elif basename.startswith('normal_'):
        return 'healthy'
    else:
        return 'unknown'

def classify_wsi_with_params(wsi_image_path, tissue_mask_dir, model, device,
                            tumor_tile_threshold, min_positive_tiles_for_tumorous,
                            tile_size_clf=TILE_SIZE,
                            target_level_clf=TARGET_INFERENCE_LEVEL,
                            tissue_threshold_clf=TISSUE_THRESHOLD,
                            batch_size_clf=CLASSIFICATION_BATCH_SIZE):
    """
    Klasifikuje WSI s konkrétními parametry pro práh a minimální počet dlaždic.
    """
    wsi_clf = None
    wsi_basename = os.path.basename(wsi_image_path)
    positive_tiles_count = 0

    try:
        wsi_clf = openslide.OpenSlide(wsi_image_path)
        
        # Určení čísla WSI pro nalezení tissue masky
        if wsi_basename.startswith('tumor_'):
            wsi_number = wsi_basename.split('_')[1].split('.')[0]
            tissue_mask_filename = f"mask_{wsi_number}.npy"
        elif wsi_basename.startswith('normal_'):
            wsi_number = wsi_basename.split('_')[1].split('.')[0]
            tissue_mask_filename = f"tissue_mask_{wsi_number}.npy"
        else:
            return "unknown_filename_format"
        
        tissue_mask_full_path = os.path.join(tissue_mask_dir, tissue_mask_filename)

        if not os.path.exists(tissue_mask_full_path):
            return "unknown_mask_missing"
            
        if target_level_clf >= wsi_clf.level_count:
            return "unknown_invalid_level"

        inference_downsample_clf = wsi_clf.level_downsamples[target_level_clf]
        step_clf = tile_size_clf 
        
        level_dims_clf = wsi_clf.level_dimensions[target_level_clf]
        output_w_clf, output_h_clf = level_dims_clf
        cols_clf = math.ceil(output_w_clf / step_clf)
        rows_clf = math.ceil(output_h_clf / step_clf)

        tissue_mask_np = np.load(tissue_mask_full_path).astype(bool)
        tissue_mask_downsample_orig = wsi_clf.level_downsamples[tissue_mask_level_index]
        scale_factor_clf = tissue_mask_downsample_orig / inference_downsample_clf
        
        batch_tiles_data_clf = []
        normalize_clf = transforms.Normalize(mean=CURRENT_MEAN, std=CURRENT_STD)

        with torch.inference_mode():
            for r_clf in range(rows_clf):
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
                        except Exception:
                            continue 
                    
                    is_last_tile_clf = (r_clf == rows_clf - 1 and c_clf == cols_clf - 1)
                    if len(batch_tiles_data_clf) == batch_size_clf or (is_last_tile_clf and len(batch_tiles_data_clf) > 0):
                        if not batch_tiles_data_clf: 
                            continue
                        
                        batch_tensor_clf = torch.stack(batch_tiles_data_clf).to(device)
                        predictions_output_clf = model(batch_tensor_clf)
                        predictions_sigmoid_clf = torch.sigmoid(predictions_output_clf)
                        
                        # Počítáme pozitivní dlaždice v aktuální dávce
                        for pred_single_tile in predictions_sigmoid_clf:
                            max_prediction = torch.max(pred_single_tile).item()
                            if max_prediction > tumor_tile_threshold:
                                positive_tiles_count += 1
                                # EARLY STOPPING: Jakmile dosáhneme požadovaného počtu, končíme
                                if positive_tiles_count >= min_positive_tiles_for_tumorous:
                                    return "tumorous"
                        
                        batch_tiles_data_clf.clear()
        
        # Pokud jsme prošli všechny dlaždice a nedosáhli požadovaného počtu pozitivních
        return "healthy"

    except Exception as e:
        return "unknown_error_classification"
    finally:
        if wsi_clf is not None: 
            try:
                wsi_clf.close()
            except Exception:
                pass

def evaluate_classification_performance(results):
    """
    Vyhodnotí výkonnost klasifikace a vypočítá metriky.
    """
    tp = sum(1 for r in results if r['gt_label'] == 'tumorous' and r['predicted_label'] == 'tumorous')
    tn = sum(1 for r in results if r['gt_label'] == 'healthy' and r['predicted_label'] == 'healthy')
    fp = sum(1 for r in results if r['gt_label'] == 'healthy' and r['predicted_label'] == 'tumorous')
    fn = sum(1 for r in results if r['gt_label'] == 'tumorous' and r['predicted_label'] == 'healthy')
    
    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total
    }

def main():
    print("=== OPTIMALIZACE PRAHŮ KLASIFIKACE NA VALIDAČNÍ SADĚ ===")
    
    # Načtení a rozdělení dat
    print("Načítání a rozdělování dat...")
    train_data, val_data = load_and_split_data(
        cancer_wsi_gt_main_dir=CANCER_WSI_GT_DIR,
        cancer_lr_tissue_mask_dir=CANCER_LR_TISSUE_DIR,
        cancer_lr_gt_mask_dir=CANCER_LR_GT_DIR,
        healthy_wsi_dir=HEALTHY_WSI_DIR,
        healthy_lr_tissue_mask_dir=HEALTHY_LR_TISSUE_DIR,
        val_size=0.2,
        random_state=42
    )
    
    val_wsi_paths, val_tissue_masks, val_hr_gt_masks, val_lr_gt_masks = val_data
    
    print(f"Validační sada obsahuje {len(val_wsi_paths)} WSI snímků")
    cancer_count = sum(1 for m in val_hr_gt_masks if m is not None)
    healthy_count = len(val_wsi_paths) - cancer_count
    print(f"  - Nádorové: {cancer_count}")
    print(f"  - Zdravé: {healthy_count}")
    
    # Načtení modelu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")
    
    model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
    # model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None)

    try:
        state_data = torch.load(model_weights_path, map_location=device, weights_only=False) 
        if isinstance(state_data, dict) and "model_state_dict" in state_data:
            model.load_state_dict(state_data["model_state_dict"])
        elif isinstance(state_data, dict): 
             model.load_state_dict(state_data)
        else: 
             raise TypeError(f"Očekáván slovník pro state_dict, ale přijato {type(state_data)}")
        print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FATÁLNÍ CHYBA: Nepodařilo se načíst váhy modelu: {e}")
        exit()
    
    # Vytvoření výstupní složky
    os.makedirs(OUTPUT_CLASSIFICATION_DIR, exist_ok=True)
    
    # Generování všech kombinací parametrů
    param_combinations = list(itertools.product(TUMOR_TILE_THRESHOLDS_TO_TEST, MIN_POSITIVE_TILES_TO_TEST))
    print(f"\nTestuji {len(param_combinations)} kombinací parametrů...")
    
    all_results = []
    
    for tile_threshold, min_tiles in param_combinations:
        print(f"\nTestuji: Práh dlaždice = {tile_threshold}, Min. pozitivních dlaždic = {min_tiles}")
        
        combination_results = []
        
        # Testování na všech WSI ve validační sadě
        for i, wsi_path in enumerate(tqdm(val_wsi_paths, desc=f"Threshold {tile_threshold}, Min tiles {min_tiles}")):
            gt_label = get_ground_truth_label(wsi_path)
            
            # Určení správné tissue mask directory
            if gt_label == 'tumorous':
                tissue_mask_dir = CANCER_LR_TISSUE_DIR
            elif gt_label == 'healthy':
                tissue_mask_dir = HEALTHY_LR_TISSUE_DIR
            else:
                continue
            
            predicted_label = classify_wsi_with_params(
                wsi_path, tissue_mask_dir, model, device,
                tumor_tile_threshold=tile_threshold,
                min_positive_tiles_for_tumorous=min_tiles
            )
            
            combination_results.append({
                'wsi_file': os.path.basename(wsi_path),
                'gt_label': gt_label,
                'predicted_label': predicted_label,
                'tile_threshold': tile_threshold,
                'min_tiles': min_tiles
            })
        
        # Vyhodnocení výkonnosti pro tuto kombinaci
        performance = evaluate_classification_performance(combination_results)
        
        result_summary = {
            'tile_threshold': tile_threshold,
            'min_tiles': min_tiles,
            **performance
        }
        
        all_results.append(result_summary)
        
        print(f"  Accuracy: {performance['accuracy']:.4f}, F1: {performance['f1_score']:.4f}, "
              f"Sensitivity: {performance['sensitivity']:.4f}, Specificity: {performance['specificity']:.4f}")
    
    # Uložení výsledků do CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Souhrnné výsledky pro všechny kombinace
    summary_csv_path = os.path.join(OUTPUT_CLASSIFICATION_DIR, f"threshold_optimization_summary_{timestamp}.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['tile_threshold', 'min_tiles', 'accuracy', 'sensitivity', 'specificity', 
                     'precision', 'f1_score', 'tp', 'tn', 'fp', 'fn', 'total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nSouhrnné výsledky uloženy do: {summary_csv_path}")
    
    # Nalezení nejlepší kombinace podle F1 skóre
    best_result = max(all_results, key=lambda x: x['f1_score'])
    print(f"\n=== NEJLEPŠÍ KOMBINACE (podle F1 skóre) ===")
    print(f"Práh dlaždice: {best_result['tile_threshold']}")
    print(f"Min. pozitivních dlaždic: {best_result['min_tiles']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"Sensitivity (Recall): {best_result['sensitivity']:.4f}")
    print(f"Specificity: {best_result['specificity']:.4f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"TP: {best_result['tp']}, TN: {best_result['tn']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    
    # Uložení nejlepších parametrů do samostatného souboru
    best_params_path = os.path.join(OUTPUT_CLASSIFICATION_DIR, f"best_classification_params_{timestamp}.txt")
    with open(best_params_path, 'w', encoding='utf-8') as f:
        f.write(f"Nejlepší parametry klasifikace (podle F1 skóre):\n")
        f.write(f"TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION = {best_result['tile_threshold']}\n")
        f.write(f"MIN_POSITIVE_TILES_FOR_TUMOROUS = {best_result['min_tiles']}\n")
        f.write(f"\nVýkonnost:\n")
        f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        f.write(f"F1 Score: {best_result['f1_score']:.4f}\n")
        f.write(f"Sensitivity: {best_result['sensitivity']:.4f}\n")
        f.write(f"Specificity: {best_result['specificity']:.4f}\n")
        f.write(f"Precision: {best_result['precision']:.4f}\n")
    
    print(f"Nejlepší parametry uloženy do: {best_params_path}")
    print(f"\nOptimalizace dokončena!")

if __name__ == "__main__":
    main()