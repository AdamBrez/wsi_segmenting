import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from openslide import OpenSlide
import matplotlib.ticker as ticker

"""
Vizualizace vytvořených masek, a jejich porovnání s ground truth maskou.
"""

gt_mask_path = r"C:\Users\USER\Desktop\wsi_dir\mask_068.tif" 
hdf5_path = r"C:\Users\USER\Desktop\test_output\nooverlap.h5" # Cesta k opravenému souboru
hdf5_path2 = r"C:\Users\USER\Desktop\test_output\pred_068_finetuned_overlap.h5" # Cesta k opravenému souboru

with h5py.File(hdf5_path, "r") as f:
    mask = f["mask"][:] # Načte jako uint8 (0, 1)

with h5py.File(hdf5_path2, "r") as f:
    mask2 = f["mask"][:] # Načte jako uint8 (0, 1)

print(f"Načtená maska - shape: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")

scale_factor = 0.1 # Menší faktor pro přehlednost
gt_mask = OpenSlide(gt_mask_path)
print(gt_mask.level_dimensions[2])
gt_mask_thumb = gt_mask.get_thumbnail((mask.shape[1]*scale_factor, mask.shape[0]*scale_factor)).convert("RGB") # Zmenšení na stejnou velikost jako maska
gt_mask_thumb = np.array(gt_mask_thumb) # Převod na numpy array
if mask.shape[0] > 0 and mask.shape[1] > 0: # Kontrola neprázdné masky
    small_mask = np.array(Image.fromarray(mask).resize( # Převod uint8 na PIL funguje
        (int(mask.shape[1] * scale_factor), int(mask.shape[0] * scale_factor)),
        resample=Image.NEAREST)) # NEAREST je vhodný pro binární masky
else:
    small_mask = mask # Pokud je maska prázdná, ponechat ji tak

small_mask2 = np.array(Image.fromarray(mask2).resize( # Převod uint8 na PIL funguje
    (int(mask2.shape[1] * scale_factor), int(mask2.shape[0] * scale_factor)),
    resample=Image.NEAREST)) # NEAREST je vhodný pro binární masky

# np.save(r"C:\Users\USER\Desktop\maska1_.npy", small_mask) # Uložení masky zdravých buněk
# np.save(r"C:\Users\USER\Desktop\maska2_.npy", small_mask2) # Uložení masky zdravých buněk
# np.save(r"C:\Users\USER\Desktop\gt_.npy", gt_mask_thumb) # Uložení masky zdravých buněk


print(f"Zmenšená maska - max hodnota: {np.max(small_mask) if small_mask.size > 0 else 'N/A'}")
print(f"Ground truth velikost: {gt_mask_thumb.shape}, Predikovaná velikost: {small_mask.shape}")
grid_spacing = 200 # Velikost mřížky
plt.figure(figsize=(10, 10))
# Ground truth maska
plt.subplot(1, 3, 1)
plt.imshow(gt_mask_thumb, cmap="gray", vmin=0, vmax=1)
plt.title("Ground truth maska")
plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)  # Přidání mřížky
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)  # Rotace popisků osy x pro lepší čitelnost
# Predikovaná maska 1
plt.subplot(1, 3, 2)
plt.imshow(small_mask, cmap="gray", vmin=0, vmax=1)
plt.title(f"Predikovaná maska vahy_1 (overlap)")
plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)  # Přidání mřížky
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)  # Rotace popisků osy x pro lepší čitelnost

# Predikovaná maska 2
plt.subplot(1, 3, 3)
plt.imshow(small_mask2, cmap="gray", vmin=0, vmax=1)
plt.title(f"Predikovaná maska vahy_2")
plt.grid(visible=True, color="red", linestyle="--", linewidth=0.5)  # Přidání mřížky
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(grid_spacing))
plt.xticks(rotation=45)  # Rotace popisků osy x pro lepší čitelnost
plt.show()


mask_healthy = r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_091.npy" # Cesta k maskám zdravých buněk
mask_healthy = np.load(mask_healthy) # Načtení masky zdravých buněk
plt.imshow(mask_healthy, cmap="gray", vmin=0, vmax=1) # Zobrazit masku zdravých buněk
plt.title("Maska zdravé tkáně")
# plt.show()
# --- END OF FILE test_opt3.py (Opraveno) ---