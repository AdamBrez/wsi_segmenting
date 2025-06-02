import os
openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
os.add_dll_directory(openslide_dll_path)
import traceback  # Pro detailní výpis chyb

import h5py
import numpy as np
import torch
import segmentation_models_pytorch as smp
import csv
# PŘIDÁNO: Import pro metriku z MONAI
from monai.metrics import DiceMetric

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# --- Konfigurace ---¨


GT_SLIDE_PATH = rf"F:\wsi_dir_test\mask_002.tif"
PRED_HDF5_PATH = rf"C:\Users\USER\Desktop\test_preds\unetpp\pred_002.h5"
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
    print(f"Chyba: Nepodařilo se najít odpovídající DZG úroveň.")
    exit()

print(f"Cílová OpenSlide úroveň pro porovnání: {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (Rozměry WxH: {target_os_dims_wh})")
print(f"Odpovídající DZG úroveň pro iteraci: {dzg_target_level} (Rozměry WxH: {dz_gen.level_dimensions[dzg_target_level]})")

level_total_w, level_total_h = dz_gen.level_dimensions[dzg_target_level]

# --- Inicializace čítačů a agregátorů ---
# 1. Pro segmentation-models-pytorch (mikro průměr)
total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0

# 2. PŘIDÁNO: Pro MONAI (makro průměr)
monai_dice_macro_agg = DiceMetric(include_background=False, reduction="mean")

processed_tiles = 0
num_cols, num_rows = dz_gen.level_tiles[dzg_target_level]
print(f"Očekávaný počet dlaždic: {num_rows} řádků, {num_cols} sloupců.")

# --- Smyčka přes dlaždice ---
for r_idx in range(num_rows):
    for c_idx in range(num_cols):
        try:
            x_on_level = c_idx * TILE_SIZE
            y_on_level = r_idx * TILE_SIZE
            current_tile_w = min(TILE_SIZE, level_total_w - x_on_level)
            current_tile_h = min(TILE_SIZE, level_total_h - y_on_level)

            if current_tile_w <= 0 or current_tile_h <= 0: continue

            tile_gt_pil = dz_gen.get_tile(dzg_target_level, (c_idx, r_idx))
            gt_tile_np_rgba = np.array(tile_gt_pil)
            gt_tile_binary_hw = (gt_tile_np_rgba[:, :, 0] > 0).astype(np.uint8) if gt_tile_np_rgba.ndim == 3 else (gt_tile_np_rgba > 0).astype(np.uint8)
            pred_tile_binary_hw = hdf5_mask_np[y_on_level : y_on_level + current_tile_h, x_on_level : x_on_level + current_tile_w]

            if gt_tile_binary_hw.shape != pred_tile_binary_hw.shape or gt_tile_binary_hw.size == 0: continue
            
            pred_torch = torch.from_numpy(pred_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            gt_torch = torch.from_numpy(gt_tile_binary_hw).long().unsqueeze(0).unsqueeze(0)

            # --- VÝPOČTY ---
            # 1. Výpočet pro smp
            stats = smp.metrics.get_stats(pred_torch, gt_torch, mode='binary', threshold=0.5)
            total_tp += stats[0]
            total_fp += stats[1]
            total_fn += stats[2]
            total_tn += stats[3]
            
            # 2. PŘIDÁNO: Výpočet pro MONAI
            monai_dice_macro_agg(y_pred=pred_torch, y=gt_torch)
            
            processed_tiles += 1

        except Exception as e_tile:
            print(f"Chyba při zpracování dlaždice ({r_idx},{c_idx}): {e_tile}")
            traceback.print_exc()

# --- Finální vyhodnocení a výpis ---
if processed_tiles > 0:
    print(f"\n--- DOKONČENO: Zpracováno {processed_tiles} dlaždic ---")
    
    # Výsledky z `smp` (mikro průměr)
    print("\n--- Metriky (MIKRO průměr, počítáno s 'smp') ---")
    iou_smp = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn).item()
    dice_smp = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn).item()
    recall_smp = smp.metrics.recall(total_tp, total_fp, total_fn, total_tn).item()
    precision_smp = smp.metrics.precision(total_tp, total_fp, total_fn, total_tn).item()
    

    print(f"Dice (F1) Score: {dice_smp:.5f}")
    print(f"IoU (Jaccard):   {iou_smp:.5f}")
    print(f"Recall:          {recall_smp:.5f}")
    print(f"Precision:       {precision_smp:.5f}")
    
    if dice_smp == 0.0 and precision_smp == 0.0 and recall_smp == 1.0:
        recall_smp = "NaN"
    # PŘIDÁNO: Výsledky z MONAI (makro průměr)
    print("\n--- Dice skóre (MAKRO průměr, počítáno s 'monai') ---")
    dice_monai_macro = monai_dice_macro_agg.aggregate().item()
    print(f"Dice Score:      {dice_monai_macro:.5f}  <-- Očekává se jiná hodnota než u mikro průměru")

    # PŘIDÁNO: Resetování MONAI agregátoru
    monai_dice_macro_agg.reset()

else:
    print("Nebyly zpracovány žádné dlaždice.")

wsi.close()
print("\nDokončeno.")