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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

"""
Vizualizace vytvořených masek, a jejich porovnání s ground truth maskou.
"""

gt_mask_path = r"F:\wsi_dir_test\mask_108.tif"
hdf5_path = r"C:\Users\USER\Desktop\test_preds\unetpp\pred_108.h5"

with h5py.File(hdf5_path, "r") as f:
    mask = f["mask"][:] # Načte jako uint8 (0, 1)

print(f"Načtená maska - shape: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")

scale_factor = 0.1 # Menší faktor pro přehlednost

# Použití 'with' pro automatické uzavření souboru OpenSlide
with OpenSlide(gt_mask_path) as gt_slide:
    # Určení rozměrů pro miniaturu na základě HDF5 masky
    if mask.shape[0] > 0 and mask.shape[1] > 0:
        guiding_thumb_w = int(mask.shape[1] * scale_factor)
        guiding_thumb_h = int(mask.shape[0] * scale_factor)
    else:
        base_w, base_h = gt_slide.dimensions
        guiding_thumb_w = int(base_w * scale_factor)
        guiding_thumb_h = int(base_h * scale_factor)
        print(f"Varování: HDF5 maska je prázdná. Používám rozměry WSI pro miniaturu: ({guiding_thumb_w}, {guiding_thumb_h})")

    # Vytvoření miniatury gt_mask
    gt_mask_thumb_pil = gt_slide.get_thumbnail((guiding_thumb_w, guiding_thumb_h))
    actual_thumb_w, actual_thumb_h = gt_mask_thumb_pil.size

    # Převod na binární masku
    gt_mask_thumb_L_np = np.array(gt_mask_thumb_pil.convert("L"))
    gt_mask_thumb_binary = (gt_mask_thumb_L_np > 0).astype(np.uint8)

# Změna velikosti HDF5 masky na rozměry GT miniatury
if mask.size > 0 and mask.shape[0] > 0 and mask.shape[1] > 0:
    small_mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
        (actual_thumb_w, actual_thumb_h),
        resample=Image.NEAREST))
else:
    print(f"Původní 'mask' je prázdná. Vytvářím prázdnou zmenšenou masku.")
    small_mask = np.zeros((actual_thumb_h, actual_thumb_w), dtype=np.uint8)

print(f"Ground truth (zmenšená, binární) velikost: {gt_mask_thumb_binary.shape}")
print(f"Predikovaná maska (zmenšená) velikost: {small_mask.shape}")

# Kontrola rozměrů
if gt_mask_thumb_binary.shape == small_mask.shape:
    print("Masky mají shodné rozměry.")
else:
    print("CHYBA: Masky NEMAJÍ shodné rozměry!")

# VYTVOŘENÍ OVERLAY_IMAGE PŘED VYKRESLENÍM
overlay_image = np.zeros((gt_mask_thumb_binary.shape[0], gt_mask_thumb_binary.shape[1], 3), dtype=np.uint8)

# GT maska bude zelená (kde je GT a není predikce)
overlay_image[(gt_mask_thumb_binary == 1) & (small_mask == 0)] = [0, 255, 0]      # Zelená

# Predikovaná maska bude oranžová (kde je predikce a není GT)
overlay_image[(gt_mask_thumb_binary == 0) & (small_mask == 1)] = [255, 165, 0]    # Oranžová

# Průsečík bude fialový (kde je GT i predikce)
overlay_image[(gt_mask_thumb_binary == 1) & (small_mask == 1)] = [255, 0, 255]    # Magenta

# Pozadí zůstává černé (kde není GT ani predikce) - [0, 0, 0]

# Vytvoření obrázku se třemi subploty (GT, HDF5, Překryv)
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3, hspace=0.1)

# První subplot
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(gt_mask_thumb_binary, cmap="gray", vmin=0, vmax=1)
ax1.set_title("Maska")
ax1.axis('off')
ax1.set_aspect('equal')

# Druhý subplot
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(small_mask, cmap="gray", vmin=0, vmax=1)
ax2.set_title("Predikce")
ax2.axis('off')
ax2.set_aspect('equal')

# Třetí subplot
ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(overlay_image)
ax3.set_title("Překrytí")
ax3.axis('off')
ax3.set_aspect('equal')

# Legenda
green_patch = mpatches.Patch(color='green', label='Pouze maska')
orange_patch = mpatches.Patch(color='orange', label='Pouze predikce')
magenta_patch = mpatches.Patch(color='magenta', label='Maska + predikce')

# Umístění legendy mimo graf
plt.figlegend(handles=[green_patch, orange_patch, magenta_patch], 
             loc='lower center', ncol=3, fontsize=12,
             frameon=True, fancybox=True, shadow=True, 
             facecolor='white', edgecolor='black')

plt.subplots_adjust(bottom=0.15)
plt.show()

print("Vizualizace dokončena.")

