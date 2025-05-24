import os
openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
os.add_dll_directory(openslide_dll_path)
import traceback  # Pro detailní výpis chyb

import h5py
import numpy as np
import torch
import segmentation_models_pytorch as smp # NOVÉ: Import knihovny
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# --- Konfigurace ---
# Nastavte cestu k OpenSlide DLL, pokud je potřeba
# openslide_dll_path = r"C:\path\to\openslide-bin\bin"
# os.add_dll_directory(openslide_dll_path)

GT_SLIDE_PATH = r"C:\Users\USER\Desktop\wsi_dir\mask_068.tif"
PRED_HDF5_PATH = r"C:\Users\USER\Desktop\test_output\best_pretrainded68.h5"
HDF5_DATASET_NAME = "mask"
TILE_SIZE = 4096
TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK = 2  # Předpoklad

# --- Inicializace ---
try:
    wsi = OpenSlide(GT_SLIDE_PATH)
except Exception as e:
    print(f"Chyba při otevírání WSI souboru: {e}")
    exit()

dz_gen = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=0, limit_bounds=False)

try:
    with h5py.File(PRED_HDF5_PATH, "r") as f:
        if HDF5_DATASET_NAME not in f:
            raise ValueError(f"Dataset '{HDF5_DATASET_NAME}' nenalezen v {PRED_HDF5_PATH}")
        hdf5_mask_np_raw = f[HDF5_DATASET_NAME][:]
except Exception as e:
    print(f"Chyba při načítání HDF5 souboru: {e}")
    exit()

if hdf5_mask_np_raw.dtype == bool:
    hdf5_mask_np = hdf5_mask_np_raw.astype(np.uint8)
else:
    hdf5_mask_np = (hdf5_mask_np_raw > 0).astype(np.uint8)
print(f"Načtená HDF5 maska (tvar HxW): {hdf5_mask_np.shape}, typ: {hdf5_mask_np.dtype}")

try:
    target_os_dims_wh = wsi.level_dimensions[TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK]
except IndexError:
    print(f"Chyba: OpenSlide úroveň {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} neexistuje.")
    exit()

if hdf5_mask_np.shape != (target_os_dims_wh[1], target_os_dims_wh[0]):
    print(
        f"Varování: Tvar HDF5 masky {hdf5_mask_np.shape} (HxW) neodpovídá OpenSlide úrovni {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} {(target_os_dims_wh[1], target_os_dims_wh[0])} (HxW).")

dzg_target_level = -1
for lvl_idx in range(dz_gen.level_count):
    if dz_gen.level_dimensions[lvl_idx] == target_os_dims_wh:
        dzg_target_level = lvl_idx
        break
if dzg_target_level == -1:
    print(
        f"Chyba: Nepodařilo se najít DZG úroveň odpovídající OS úrovni {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (rozměry {target_os_dims_wh}).")
    print("Dostupné DZG úrovně a jejich rozměry (WxH):")
    for lvl_idx_print in range(dz_gen.level_count):
        print(f"  DZG úroveň {lvl_idx_print}: {dz_gen.level_dimensions[lvl_idx_print]}")
    exit()

print(f"Cílová OpenSlide úroveň pro porovnání: {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (Rozměry WxH: {target_os_dims_wh})")
print(f"Odpovídající DZG úroveň pro iteraci: {dzg_target_level} (Rozměry WxH: {dz_gen.level_dimensions[dzg_target_level]})")

level_total_w, level_total_h = dz_gen.level_dimensions[dzg_target_level]

# NOVÉ: Inicializace čítačů pro základní statistiky
total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0

processed_tiles = 0
num_cols, num_rows = dz_gen.level_tiles[dzg_target_level]

print(f"Očekávaný počet dlaždic: {num_rows} řádků, {num_cols} sloupců.")

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
            pil_w, pil_h = tile_gt_pil.size
            
            final_w = min(pil_w, current_tile_w)
            final_h = min(pil_h, current_tile_h)

            if final_w <= 0 or final_h <= 0: continue

            if pil_w > final_w or pil_h > final_h:
                tile_gt_pil = tile_gt_pil.crop((0, 0, final_w, final_h))

            current_tile_w, current_tile_h = final_w, final_h
            gt_tile_np_rgba = np.array(tile_gt_pil)

            if gt_tile_np_rgba.ndim == 3 and gt_tile_np_rgba.shape[2] >= 3:
                gt_tile_binary_hw = (gt_tile_np_rgba[:, :, 0] > 0).astype(np.uint8)
            elif gt_tile_np_rgba.ndim == 2:
                gt_tile_binary_hw = (gt_tile_np_rgba > 0).astype(np.uint8)
            else:
                continue

            pred_tile_binary_hw = hdf5_mask_np[y_on_level: y_on_level + current_tile_h,
                                  x_on_level: x_on_level + current_tile_w]

            if gt_tile_binary_hw.shape != pred_tile_binary_hw.shape:
                print(f"KRITICKÁ CHYBA ({r_idx},{c_idx}): Neshoda tvarů! GT={gt_tile_binary_hw.shape}, PRED={pred_tile_binary_hw.shape}. Přeskakuji.")
                continue

            if gt_tile_binary_hw.size == 0:
                continue
            
            # UPRAVENO: Konverze na tenzory pro 'smp'
            # Predikce by měla být float, ground truth by měl být integer (long)
            pred_torch = torch.from_numpy(pred_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            gt_torch = torch.from_numpy(gt_tile_binary_hw.astype(np.int64)).unsqueeze(0).unsqueeze(0)

            # UPRAVENO: Výpočet a agregace statistik pomocí 'smp'
            stats = smp.metrics.get_stats(
                pred_torch, 
                gt_torch, 
                mode='binary', 
                threshold=0.5 # Prahová hodnota pro převod predikce na binární masku
            )
            
            total_tp += stats[0]
            total_fp += stats[1]
            total_fn += stats[2]
            total_tn += stats[3]
            
            processed_tiles += 1

        except Exception as e_tile:
            print(f"Chyba při zpracování dlaždice ({r_idx},{c_idx}): {e_tile}")
            traceback.print_exc()
            continue

if processed_tiles > 0:
    try:
        # UPRAVENO: Finální výpočet metrik z agregovaných statistik
        # Přidáváme malou hodnotu (epsilon) do jmenovatele, abychom předešli dělení nulou
        epsilon = 1e-6
        
        final_iou_score = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
        final_dice_score = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction='micro') # F1-score je to samé co Dice
        final_recall_score = smp.metrics.recall(total_tp, total_fp, total_fn, total_tn, reduction='micro')
        final_precision_score = smp.metrics.precision(total_tp, total_fp, total_fn, total_tn, reduction='micro')

        print("\n--- Finální výsledky (počítáno s segmentation-models-pytorch) ---")
        print(f"Zpracováno dlaždic: {processed_tiles}")
        print(f"IoU (Jaccard): {final_iou_score:.4f}")
        print(f"Dice Score (F1): {final_dice_score:.4f}")
        print(f"Recall: {final_recall_score:.4f}")
        print(f"Precision: {final_precision_score:.4f}")
        print("-----------------------------------------------------------------")

    except Exception as e_agg:
        print(f"Chyba při agregaci skóre: {e_agg}")
else:
    print("Nebyly zpracovány žádné dlaždice.")

wsi.close()
print("Dokončeno.")