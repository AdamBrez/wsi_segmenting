import os
openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
os.add_dll_directory(openslide_dll_path)
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import h5py
import numpy as np
import torch
from monai.metrics import DiceMetric
import traceback # Pro detailní výpis chyb

# --- Konfigurace ---
GT_SLIDE_PATH = r"C:\Users\USER\Desktop\wsi_dir\mask_068.tif"
PRED_HDF5_PATH = r"C:\Users\USER\Desktop\test_output\best_pretrainded68.h5"
HDF5_DATASET_NAME = "mask"
TILE_SIZE = 4096
TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK = 2 # Předpoklad

# --- Inicializace ---
try:
    wsi = OpenSlide(GT_SLIDE_PATH)
except Exception as e:
    print(f"Chyba při otevírání WSI souboru: {e}")
    exit()

# DŮLEŽITÉ: Použijte limit_bounds=False pro robustnější pokrytí
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

if hdf5_mask_np.shape != (target_os_dims_wh[1], target_os_dims_wh[0]): # HDF5 je HxW, OS_dims je WxH
    print(f"Varování: Tvar HDF5 masky {hdf5_mask_np.shape} (HxW) neodpovídá OpenSlide úrovni {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} {(target_os_dims_wh[1], target_os_dims_wh[0])} (HxW).")

dzg_target_level = -1
for lvl_idx in range(dz_gen.level_count):
    if dz_gen.level_dimensions[lvl_idx] == target_os_dims_wh:
        dzg_target_level = lvl_idx
        break
if dzg_target_level == -1:
    print(f"Chyba: Nepodařilo se najít DZG úroveň odpovídající OS úrovni {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (rozměry {target_os_dims_wh}).")
    # Fallback na základě předchozího úspěšného nalezení (pokud bylo)
    # dzg_target_level = 16 # JAKO PŘÍKLAD, pokud víte, že to má být 16
    # if dzg_target_level == -1: # Stále nenalezeno
    print("Dostupné DZG úrovně a jejich rozměry (WxH):")
    for lvl_idx_print in range(dz_gen.level_count):
        print(f"  DZG úroveň {lvl_idx_print}: {dz_gen.level_dimensions[lvl_idx_print]}")
    exit()

print(f"Cílová OpenSlide úroveň pro porovnání: {TARGET_OPENSLIDE_LEVEL_FOR_HDF5_MASK} (Rozměry WxH: {target_os_dims_wh})")
print(f"Odpovídající DZG úroveň pro iteraci: {dzg_target_level} (Rozměry WxH: {dz_gen.level_dimensions[dzg_target_level]})")

# Celkové rozměry cílové úrovně (WxH) z DZG, které by měly odpovídat HDF5 po prohození
level_total_w, level_total_h = dz_gen.level_dimensions[dzg_target_level]

dice_metric_aggregator = DiceMetric(include_background=False, num_classes=1)

processed_tiles = 0
num_cols, num_rows = dz_gen.level_tiles[dzg_target_level]

print(f"Očekávaný počet dlaždic: {num_rows} řádků, {num_cols} sloupců.")

for r_idx in range(num_rows):
    for c_idx in range(num_cols):
        try:
            # Souřadnice levého horního rohu dlaždice (c_idx, r_idx)
            # v rámci mřížky dlaždic na úrovni dzg_target_level (v pixelech dané úrovně).
            x_on_level = c_idx * TILE_SIZE # souřadnice X (sloupec)
            y_on_level = r_idx * TILE_SIZE # souřadnice Y (řádek)

            # Skutečná šířka a výška této dlaždice na dané úrovni
            # (může být menší než TILE_SIZE na pravém/dolním okraji `level_total_w/h`)
            current_tile_w = min(TILE_SIZE, level_total_w - x_on_level)
            current_tile_h = min(TILE_SIZE, level_total_h - y_on_level)

            if current_tile_w <= 0 or current_tile_h <= 0:
                # print(f"DEBUG ({r_idx},{c_idx}): Dlaždice je mimo oblast (w={current_tile_w},h={current_tile_h}). x_on_level={x_on_level}, y_on_level={y_on_level}")
                continue

            # 1. Načtení Ground Truth dlaždice
            tile_gt_pil = dz_gen.get_tile(dzg_target_level, (c_idx, r_idx)) # Vrací PIL (WxH)
            
            # Velikost vráceného PIL obrázku (měla by odpovídat current_tile_w, current_tile_h)
            pil_w, pil_h = tile_gt_pil.size
            
            # Pokud get_tile vrátí dlaždici větší než vypočtená (méně pravděpodobné s limit_bounds=False),
            # nebo pokud je vypočtená menší (např. na okraji), ořízneme PIL na menší z nich.
            # Cílem je, aby GT i PRED měly naprosto stejné rozměry current_tile_w, current_tile_h.
            if pil_w != current_tile_w or pil_h != current_tile_h:
                # Toto by se ideálně nemělo stávat, pokud level_total_w/h jsou správně
                # a current_tile_w/h jsou správně odvozeny.
                # Pokud se stane, znamená to, že get_tile() vrací něco jiného než jsme čekali
                # pro danou adresu (c_idx, r_idx) na okraji.
                # print(f"DEBUG ({r_idx},{c_idx}): Neshoda velikosti GT PIL. Očekáváno(WxH):({current_tile_w},{current_tile_h}), get_tile vrátil:({pil_w},{pil_h}). Upravuji cílové rozměry.")
                # Ořízneme PIL na menší z rozměrů, pokud je to nutné
                # A POUŽIJEME TYTO MENŠÍ ROZMĚRY i pro PRED dlaždici.
                final_w = min(pil_w, current_tile_w)
                final_h = min(pil_h, current_tile_h)

                if final_w <= 0 or final_h <= 0: continue

                if pil_w > final_w or pil_h > final_h:
                    tile_gt_pil = tile_gt_pil.crop((0,0, final_w, final_h))
                
                # Aktualizujeme current_tile_w/h na skutečně použitou velikost
                current_tile_w = final_w
                current_tile_h = final_h
            
            gt_tile_np_rgba = np.array(tile_gt_pil) # Výstup je HxWx(kanály), tvar (current_tile_h, current_tile_w, kanaly)

            # Převod GT na binární masku (0 nebo 1) - PŘIZPŮSOBTE PODLE VAŠICH GT DAT!
            if gt_tile_np_rgba.ndim == 3 and gt_tile_np_rgba.shape[2] >= 3: # RGB nebo RGBA
                 gt_tile_binary_hw = (gt_tile_np_rgba[:,:,0] > 0).astype(np.uint8)
            elif gt_tile_np_rgba.ndim == 2: # Grayscale
                 gt_tile_binary_hw = (gt_tile_np_rgba > 0).astype(np.uint8)
            else:
                 print(f"DEBUG ({r_idx},{c_idx}): Neočekávaný formát GT dlaždice: shape={gt_tile_np_rgba.shape}. Přeskakuji.")
                 continue

            # 2. Zpracování Predikované dlaždice (z HDF5, která je HxW)
            # Výřez: hdf5_mask_np[řádky_od:řádky_do, sloupce_od:sloupce_do]
            # řádky: y_on_level až y_on_level + current_tile_h
            # sloupce: x_on_level až x_on_level + current_tile_w
            pred_tile_binary_hw = hdf5_mask_np[y_on_level : y_on_level + current_tile_h, \
                                               x_on_level : x_on_level + current_tile_w]

            # --- DEBUG PRINTY ---
            if r_idx == 0 and c_idx <=15 : # Např. pro prvních pár dlaždic nebo problematické
                print(f"--- Debug dlaždice ({r_idx},{c_idx}) ---")
                print(f"  x_on_level={x_on_level}, y_on_level={y_on_level}")
                print(f"  level_total_w={level_total_w}, level_total_h={level_total_h}")
                print(f"  Vypočteno: current_tile_w={current_tile_w}, current_tile_h={current_tile_h}")
                print(f"  GT PIL .size (WxH): {tile_gt_pil.size if 'tile_gt_pil' in locals() else 'N/A'}")
                print(f"  Tvar gt_tile_binary_hw (HxW): {gt_tile_binary_hw.shape}")
                print(f"  Tvar pred_tile_binary_hw (HxW): {pred_tile_binary_hw.shape}")
                print(f"  Očekávaný společný tvar pro oba (HxW): ({current_tile_h}, {current_tile_w})")
            # --- KONEC DEBUG PRINTŮ ---

            if gt_tile_binary_hw.shape != pred_tile_binary_hw.shape:
                print(f"KRITICKÁ CHYBA ({r_idx},{c_idx}): Neshoda tvarů! GT={gt_tile_binary_hw.shape}, PRED={pred_tile_binary_hw.shape}. Cílové (HxW): ({current_tile_h},{current_tile_w}). Přeskakuji.")
                continue
            
            if gt_tile_binary_hw.size == 0: # Přeskočit, pokud je dlaždice nulové velikosti
                # print(f"DEBUG ({r_idx},{c_idx}): Dlaždice nulové velikosti po zpracování. Přeskakuji.")
                continue

            gt_torch = torch.from_numpy(gt_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            pred_torch = torch.from_numpy(pred_tile_binary_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            dice_metric_aggregator(y_pred=pred_torch, y=gt_torch)
            processed_tiles += 1
        
        except Exception as e_tile:
            print(f"Chyba při zpracování dlaždice ({r_idx},{c_idx}): {e_tile}")
            traceback.print_exc()
            continue

if processed_tiles > 0:
    try:
        final_dice_score = dice_metric_aggregator.aggregate().item()
        print(f"Celkové Dice skóre (agregované přes {processed_tiles} dlaždic): {final_dice_score:.4f}")
    except Exception as e_agg:
        print(f"Chyba při agregaci Dice skóre: {e_agg}")
else:
    print("Nebyly zpracovány žádné dlaždice.")

dice_metric_aggregator.reset()
wsi.close()
print("Dokončeno.")