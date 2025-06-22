import os
# Ujistěte se, že tato cesta je správná pro vaše prostředí
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from openslide import OpenSlide
import matplotlib.ticker as ticker
from monai.metrics import DiceMetric
import torch
"""
Vizualizace vytvořených masek, a jejich porovnání s ground truth maskou.
"""

gt_mask_path = r"F:\wsi_dir_test\mask_016.tif"
hdf5_path = r"C:\Users\USER\Desktop\test_preds\unetpp\pred_016.h5"
hdf5_path2 = r"C:\Users\USER\Desktop\test_preds\unetpp\pred_016.h5"

with h5py.File(hdf5_path, "r") as f:
    mask = f["mask"][:] # Načte jako uint8 (0, 1)

with h5py.File(hdf5_path2, "r") as f:
    mask2 = f["mask"][:] # Načte jako uint8 (0, 1)

print(f"Načtená maska 1 - shape: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")
print(f"Načtená maska 2 - shape: {mask2.shape}, dtype: {mask2.dtype}, min: {np.min(mask2)}, max: {np.max(mask2)}")

scale_factor = 0.1 # Menší faktor pro přehlednost

# Použití 'with' pro automatické uzavření souboru OpenSlide
with OpenSlide(gt_mask_path) as gt_slide:
    print(f"Rozměry úrovní v gt_mask (úroveň 2): {gt_slide.level_dimensions[0] if len(gt_slide.level_dimensions) > 2 else 'N/A'}")

    # 1. Určete *orientační* cílové rozměry pro miniaturu gt_mask
    #    Použijeme rozměry jedné z HDF5 masek jako referenci pro plné rozlišení.
    #    Předpokládáme, že 'mask' má rozměry odpovídající plnému rozlišení gt_slide.
    if mask.shape[0] > 0 and mask.shape[1] > 0:
        guiding_thumb_w = int(mask.shape[1] * scale_factor)
        guiding_thumb_h = int(mask.shape[0] * scale_factor)
    else:
        # Fallback, pokud je 'mask' prázdná, použijeme rozměry z 'mask2' nebo nějaké defaultní
        if mask2.shape[0] > 0 and mask2.shape[1] > 0:
            guiding_thumb_w = int(mask2.shape[1] * scale_factor)
            guiding_thumb_h = int(mask2.shape[0] * scale_factor)
        else:
            # Pokud jsou obě HDF5 masky prázdné, je těžké určit správnou velikost.
            # Můžete použít rozměry nejvyšší úrovně WSI nebo nastavit pevné hodnoty.
            # Pro jednoduchost zde použijeme rozměry nejvyšší úrovně WSI.
            base_w, base_h = gt_slide.dimensions
            guiding_thumb_w = int(base_w * scale_factor)
            guiding_thumb_h = int(base_h * scale_factor)
            print(f"Varování: Obě HDF5 masky jsou prázdné nebo nevalidní. Používám rozměry WSI pro miniaturu: ({guiding_thumb_w}, {guiding_thumb_h})")


    # 2. Vytvořte miniaturu gt_mask jako PIL Image objekt
    gt_mask_thumb_pil = gt_slide.get_thumbnail((guiding_thumb_w, guiding_thumb_h))

    # 3. Získejte *skutečné* rozměry vytvořené miniatury (šířka, výška)
    actual_thumb_w, actual_thumb_h = gt_mask_thumb_pil.size

    # 4. Převeďte miniaturu gt_mask na NumPy pole pro zobrazení a případný Dice.
    #    Převedeme na grayscale ("L") a pak na binární (0 nebo 1),
    #    protože masky jsou typicky binární.
    gt_mask_thumb_L_np = np.array(gt_mask_thumb_pil.convert("L"))
    # Předpokládáme, že v .tif masce jsou nenulové hodnoty popředí
    gt_mask_thumb_binary = (gt_mask_thumb_L_np > 0).astype(np.uint8)


# 5. Změňte velikost masek z HDF5 na *skutečné* rozměry miniatury gt_mask
if mask.size > 0 and mask.shape[0] > 0 and mask.shape[1] > 0:
    small_mask = np.array(Image.fromarray(mask).resize(
        (actual_thumb_w, actual_thumb_h), # Použít skutečné rozměry
        resample=Image.NEAREST)) # NEAREST je vhodný pro binární masky
else:
    print(f"Původní 'mask' je prázdná nebo má neplatné rozměry ({mask.shape}). Vytvářím prázdnou zmenšenou masku.")
    small_mask = np.zeros((actual_thumb_h, actual_thumb_w), dtype=np.uint8)

if mask2.size > 0 and mask2.shape[0] > 0 and mask2.shape[1] > 0:
    small_mask2 = np.array(Image.fromarray(mask2).resize(
        (actual_thumb_w, actual_thumb_h), # Použít skutečné rozměry
        resample=Image.NEAREST))
else:
    print(f"Původní 'mask2' je prázdná nebo má neplatné rozměry ({mask2.shape}). Vytvářím prázdnou zmenšenou masku.")
    small_mask2 = np.zeros((actual_thumb_h, actual_thumb_w), dtype=np.uint8)


# np.save(r"C:\Users\USER\Desktop\unetpp_.npy", small_mask)
# np.save(r"C:\Users\USER\Desktop\finetuned_.npy", small_mask2)
# np.save(r"C:\Users\USER\Desktop\gt_111.npy", gt_mask_thumb_binary)

def dice_coefficient(mask1, mask2):
    """
    Vypočítá Dice koeficient mezi dvěma binárními maskami (NumPy pole).

    Argumenty:
    mask1 (np.ndarray): První binární maska (hodnoty 0 nebo 1).
    mask2 (np.ndarray): Druhá binární maska (hodnoty 0 nebo 1), stejných rozměrů jako mask1.

    Návratová hodnota:
    float: Dice koeficient (hodnota mezi 0 a 1).
           1.0 pokud jsou obě masky prázdné (perfektní shoda na "nic").
    """
    # Ujistíme se, že pracujeme s booleovskými hodnotami pro logické operace,
    # nebo můžeme předpokládat, že vstupem jsou již 0 a 1.
    # Pro robustnost je lepší převod na bool, pokud by vstupem byly jiné číselné hodnoty.
    # Pokud víte jistě, že máte 0 a 1, můžete tento krok přeskočit.
    # mask1_bool = mask1.astype(np.bool_)
    # mask2_bool = mask2.astype(np.bool_)
    # print(f"mask1 - shape: {mask1.shape}, dtype: {mask1.dtype}, min: {np.min(mask1)}, max: {np.max(mask1)}")
    # print(f"mask2 - shape: {mask2.shape}, dtype: {mask2.dtype}, min: {np.min(mask2)}, max: {np.max(mask2)}")
    # Pokud jsou vstupem již 0 a 1 (např. np.uint8), můžeme počítat přímo:
    intersection = np.sum(mask1 * mask2)  # Logický AND a součet, nebo prostě násobení pro 0/1
    sum_masks = np.sum(mask1) + np.sum(mask2)

    if sum_masks == 0:
        # Obě masky jsou prázdné, což můžeme považovat za perfektní shodu.
        return 1.0

    dice = (2. * intersection) / sum_masks
    return dice
print(f"Zmenšená maska 1 - max hodnota: {np.max(small_mask) if small_mask.size > 0 else 'N/A'}")
print(f"Zmenšená maska 2 - max hodnota: {np.max(small_mask2) if small_mask2.size > 0 else 'N/A'}")
print(f"Ground truth (zmenšená, binární) velikost: {gt_mask_thumb_binary.shape}")
print(f"Predikovaná maska 1 (zmenšená) velikost: {small_mask.shape}")
print(f"Predikovaná maska 2 (zmenšená) velikost: {small_mask2.shape}")

# Kontrola, zda se rozměry shodují
if gt_mask_thumb_binary.shape == small_mask.shape == small_mask2.shape:
    print("Všechny zmenšené masky mají shodné rozměry.")
else:
    print("CHYBA: Zmenšené masky NEMAJÍ shodné rozměry!")

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
gt_tensor = torch.from_numpy(gt_mask_thumb_binary).unsqueeze(0).unsqueeze(0) # Přidání dimenze pro batch a kanál
mask1_tensor = torch.from_numpy(small_mask).unsqueeze(0).unsqueeze(0)
mask2_tensor = torch.from_numpy(small_mask2).unsqueeze(0).unsqueeze(0)
dice_metric(y_pred=mask1_tensor, y=gt_tensor)
dice1 = dice_metric.aggregate().item()
dice_metric.reset()
dice_metric(y_pred=mask2_tensor, y=gt_tensor)
dice2 = dice_metric.aggregate().item()
print(f"Dice koeficient mezi maskou 1 a gt: {dice1}")
print(f"Dice koeficient mezi maskou 2 a gt: {dice2}")
print(f"Dice koeficient mezi maskou 1 a maskou 2: {dice_coefficient(small_mask, small_mask2)}")
# print(f"Dice koeficient mezi maskou 1 a maskou 2: {dice_coefficient(gt_mask_thumb_binary, small_mask2)}")

grid_spacing = 200 # Velikost mřížky (může být potřeba upravit podle velikosti miniatur)
plt.figure(figsize=(20, 5)) # Upravena velikost pro čtyři obrázky vedle sebe

# Ground truth maska
plt.subplot(1, 4, 1)
plt.imshow(gt_mask_thumb_binary, cmap="gray", vmin=0, vmax=1) # Používáme binární gt masku
plt.title("Ground truth maska (zmenšená)")
# plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)

# Predikovaná maska 1
plt.subplot(1, 4, 2)
plt.imshow(small_mask, cmap="gray", vmin=0, vmax=1)
plt.title("Predikovaná maska 1 (z HDF5)") # Upravený název
# plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)

# Predikovaná maska 2
plt.subplot(1, 4, 3)
plt.imshow(small_mask2, cmap="gray", vmin=0, vmax=1)
plt.title("Predikovaná maska 2 (z HDF5)") # Upravený název
# plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)

# Překrytí GT masky a Predikovanej masky 1
plt.subplot(1, 4, 4)
# Vytvoříme prázdný RGB obrázek pro překrytí
overlay_image = np.zeros((gt_mask_thumb_binary.shape[0], gt_mask_thumb_binary.shape[1], 3), dtype=np.uint8)

# GT maska bude červená (kde je GT a není predikcia)
overlay_image[(gt_mask_thumb_binary == 1) & (small_mask == 0)] = [255, 0, 0]  # Červená

# Predikovaná maska 1 bude modrá (kde je predikcia a není GT)
overlay_image[(gt_mask_thumb_binary == 0) & (small_mask == 1)] = [0, 0, 255]  # Modrá

# Priesečník bude fialový (kde je GT aj predikcia)
overlay_image[(gt_mask_thumb_binary == 1) & (small_mask == 1)] = [255, 0, 255] # Fialová (Magenta)

plt.imshow(overlay_image)
plt.title("Překrytí: GT (Č) vs Maska1 (M)")
plt.xticks(rotation=45)
# Legenda pre prekrytie
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Pouze GT')
blue_patch = mpatches.Patch(color='blue', label='Pouze Maska 1')
magenta_patch = mpatches.Patch(color='magenta', label='GT & Maska 1 (Překrytí)')
plt.legend(handles=[red_patch, blue_patch, magenta_patch], loc='upper right', fontsize='small')


plt.tight_layout() # Pro lepší uspořádání subplotů
plt.show()

# Vytvoření nového obrázku se dvěma subploty (GT + první HDF5 maska)
plt.figure(figsize=(16, 6))

# Ground truth maska
plt.subplot(1, 2, 1)
plt.imshow(gt_mask_thumb_binary, cmap="gray", vmin=0, vmax=1)
# plt.title("Ground truth maska (zmenšená)", fontsize=14)
plt.axis('off')  # Odstraní osy pro čistší vzhled

# Predikovaná maska 1 (první HDF5)
plt.subplot(1, 2, 2)
plt.imshow(small_mask, cmap="gray", vmin=0, vmax=1)
# plt.title(f"Predikovaná maska 1 (Dice: {dice1:.3f})", fontsize=14)
plt.axis('off')  # Odstraní osy pro čistší vzhled

plt.tight_layout()

# Uložení jako SVG
svg_output_path = r"C:\Users\USER\Desktop\gt_vs_pred_comparison.svg"
plt.savefig(svg_output_path, format='svg', bbox_inches='tight', dpi=300)
print(f"Porovnání GT vs Predikce uloženo jako SVG: {svg_output_path}")

plt.show()

a = np.load(r"C:\Users\USER\Desktop\gt_2.npy")
print(f"Načtená maska - shape: {a.shape}, dtype: {a.dtype}, min: {np.min(a)}, max: {np.max(a)}")

