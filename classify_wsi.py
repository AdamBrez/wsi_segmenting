# --- START OF MODIFIED SCRIPT FOR CLASSIFICATION ONLY ---
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
# import h5py # Již není potřeba pro ukládání predikcí
import numpy as np
import openslide

# Importy pro model, ale ne pro detailní metriky segmentace
import segmentation_models_pytorch as smp

# --- HLAVNÍ KONFIGURACE ---
WSI_INPUT_DIR = r"F:\wsi_dir_test" # Cesta ke vstupním WSI souborům
TISSUE_MASK_DIR = r"C:\Users\USER\Desktop\colab_unet\test_lowres_masky" # Cesta k .npy maskám tkáně
# Cesta k výstupní složce pro CSV soubor s výsledky klasifikace
OUTPUT_CLASSIFICATION_DIR = r"C:\Users\USER\Desktop\classification_results_unetpp_2" # Upravte dle potřeby
model_weights_path = r"C:\Users\USER\Desktop\results\2025-05-24_00-11-56\best_weights_2025-05-24_00-11-56.pth" # Správná cesta k vahám

TILE_SIZE = 256 # Použito ve funkci classify_wsi_quickly

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CURRENT_MEAN = IMAGENET_MEAN
CURRENT_STD = IMAGENET_STD

tissue_mask_level_index = 6
TISSUE_THRESHOLD = 0.05

CLASSIFICATION_BATCH_SIZE = 24 # Batch size pro klasifikaci
TARGET_INFERENCE_LEVEL = 2 # Cílová úroveň OpenSlide pro inferenci v klasifikaci

# Minimální počet pozitivních dlaždic pro klasifikaci WSI jako nádorového
MIN_POSITIVE_TILES_FOR_TUMOROUS = 20
# Práh pro klasifikaci jednotlivé dlaždice jako nádorové
TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION = 0.9995


# --- FUNKCE PRO RYCHLOU KLASIFIKACI (zůstává z předchozí verze) ---
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
                            
                            for pred_single_tile in predictions_sigmoid_clf:
                                if torch.max(pred_single_tile).item() > tumor_tile_threshold:
                                    positive_tiles_count += 1
                                    if positive_tiles_count >= min_positive_tiles_for_tumorous:
                                        found_enough_positive_tiles = True
                                        break 
                            
                            batch_tiles_data_clf.clear()
                            if found_enough_positive_tiles:
                                break 
            
            if found_enough_positive_tiles:
                classification_result_internal = "tumorous"
            elif classification_result_internal == "unknown_default": 
                classification_result_internal = "healthy"
            
            doba_trvani = time.time() - script_start_time
            if classification_result_internal == "tumorous":
                print(f"--- {wsi_basename}: Klasifikováno jako NÁDOROVÉ (prah {tumor_tile_threshold} překročen u {positive_tiles_count}/{min_positive_tiles_for_tumorous} požadovaných dlaždic). Doba: {doba_trvani:.2f}s ---")
            elif classification_result_internal == "healthy":
                print(f"--- {wsi_basename}: Klasifikováno jako ZDRAVÉ (pozitivních dlaždic: {positive_tiles_count}/{min_positive_tiles_for_tumorous} požadovaných, zpracováno celkem {processed_for_model_clf} dlaždic). Doba: {doba_trvani:.2f}s ---")

    except Exception as e:
        print(f"CHYBA (Klasifikace) během zpracování {wsi_basename}: {e}")
        traceback.print_exc()
        classification_result_internal = "unknown_error_classification"
    finally:
        if wsi_clf is not None: 
            try:
                wsi_clf.close()
            except Exception as e_close:
                print(f"Varování: Chyba pri zavirani wsi_clf v classify_wsi_quickly: {e_close}")
    
    return classification_result_internal

# --- HLAVNÍ SPUŠTĚCÍ BLOK ---
if __name__ == "__main__":
    main_start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    # Definice modelu (musí odpovídat trénovanému modelu)
    # Zvolte si model, který jste použili pro trénink vah
    # model = smp.DeepLabV3Plus(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
    model = smp.UnetPlusPlus(encoder_name="resnet34", in_channels=3, classes=1, activation=None)
    
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
    except FileNotFoundError:
        print(f"FATÁLNÍ CHYBA: Soubor s váhami modelu nebyl nalezen na cestě: {model_weights_path}")
        exit()
    except Exception as e:
        print(f"FATÁLNÍ CHYBA: Nepodařilo se načíst váhy modelu: {e}")
        traceback.print_exc()
        exit()

    os.makedirs(OUTPUT_CLASSIFICATION_DIR, exist_ok=True)
    
    # Název a cesta k CSV souboru pro výsledky klasifikace
    classification_csv_path = os.path.join(OUTPUT_CLASSIFICATION_DIR, "classification_summary.csv")
    csv_header = ["WSI_File", "Classification_Status"]
    
    # Vytvoření CSV souboru s hlavičkou, pokud neexistuje
    if not os.path.exists(classification_csv_path):
        try:
            with open(classification_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_header)
            print(f"Vytvořen soubor pro výsledky klasifikace: {classification_csv_path}")
        except IOError as e:
            print(f"CHYBA: Nepodařilo se vytvořit soubor pro výsledky klasifikace: {e}")
            exit()

    search_pattern = os.path.join(WSI_INPUT_DIR, 'test_*.tif')
    wsi_files_to_process = sorted(glob.glob(search_pattern))

    all_classification_results = [] # Seznam pro uložení všech výsledků pro pozdější zápis do CSV

    if not wsi_files_to_process:
        print(f"\nCHYBA: Ve složce '{WSI_INPUT_DIR}' nebyly nalezeny žádné soubory 'test_*.tif'.")
    else:
        print(f"\nNalezeno {len(wsi_files_to_process)} souborů 'test_*.tif' ke zpracování.")
        
        for wsi_full_path in wsi_files_to_process:
            wsi_basename_loop = os.path.basename(wsi_full_path)
            current_result = {"WSI_File": wsi_basename_loop, "Classification_Status": "ERROR_BEFORE_CALL"}

            try:
                classification_status = classify_wsi_quickly(
                    wsi_full_path, TISSUE_MASK_DIR, model, device,
                    tumor_tile_threshold=TUMOR_TILE_THRESHOLD_FOR_CLASSIFICATION,
                    min_positive_tiles_for_tumorous=MIN_POSITIVE_TILES_FOR_TUMOROUS
                )
                current_result["Classification_Status"] = classification_status
                print(f"WSI {wsi_basename_loop} - Stav klasifikace: {classification_status}")

            except Exception as loop_err:
                print(f"\n!!!!!! NEOČEKÁVANÁ FATÁLNÍ CHYBA PŘI ZPRACOVÁNÍ {wsi_full_path}: {loop_err} !!!!!!")
                traceback.print_exc()
                current_result["Classification_Status"] = f"FATAL_PROCESSING_ERROR: {str(loop_err).replaceCSV()}"
            
            all_classification_results.append(current_result)

        # Zápis všech výsledků do CSV po dokončení smyčky
        try:
            with open(classification_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                for row_data in all_classification_results:
                    writer.writerow(row_data)
            print(f"\nVýsledky klasifikace byly úspěšně zapsány do: {classification_csv_path}")
        except IOError as e_csv:
            print(f"CHYBA při finálním zápisu do CSV souboru: {e_csv}")
        except Exception as e_final_csv:
            print(f"Neočekávaná CHYBA při finálním zápisu do CSV souboru: {e_final_csv}")


        # Výpis statistik
        healthy_slides_names = [res["WSI_File"] for res in all_classification_results if res["Classification_Status"] == "healthy"]
        tumorous_slides_names = [res["WSI_File"] for res in all_classification_results if res["Classification_Status"] == "tumorous"]
        other_slides_count = len(all_classification_results) - len(healthy_slides_names) - len(tumorous_slides_names)

        print("\n--- Souhrn Klasifikace ---")
        print(f"Celkem zpracováno WSI: {len(all_classification_results)}")
        print(f"Počet zdravých WSI: {len(healthy_slides_names)}")
        print(f"Počet nádorových WSI: {len(tumorous_slides_names)}")
        if other_slides_count > 0:
            print(f"Počet WSI s jiným stavem (chyba, chybějící maska atd.): {other_slides_count}")

        # Volitelně vypsat názvy (může být dlouhé)
        # print("\nNázvy zdravých WSI:")
        # for name in healthy_slides_names: print(name)
        # print("\nNázvy nádorových WSI:")
        # for name in tumorous_slides_names: print(name)

    total_end_time = time.time()
    print("\n" + "="*80)
    print("KLASIFIKACE VŠECH SOUBORŮ DOKONČENA.")
    print(f"Celkový čas běhu skriptu: {(total_end_time - main_start_time) / 60:.2f} minut.")
    print(f"Výsledky uloženy v: {classification_csv_path}")
    print("="*80)

# --- END OF MODIFIED SCRIPT FOR CLASSIFICATION ONLY ---