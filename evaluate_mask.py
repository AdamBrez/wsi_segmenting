import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")


import h5py
import numpy as np
from openslide import OpenSlide
from tqdm import tqdm
# from scipy.spatial.distance import dice
# from sklearn.metrics import jaccard_score

# Cesty k souborům
predicted_mask_path = r"C:\Users\USER\Desktop\test_output\predicted_mask018_bez_ol.h5"
ground_truth_mask_path = r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_018.tif"

# Velikost bloku (dlaždice) pro zpracování
tile_size = 256  # Můžete upravit podle dostupné paměti

intersection_sum = 0
union_sum = 0
predicted_sum = 0
ground_truth_sum = 0

# Načtení ground truth masky pomocí OpenSlide
ground_truth_slide = OpenSlide(ground_truth_mask_path)
ground_truth_shape = (ground_truth_slide.dimensions[1], ground_truth_slide.dimensions[0])  # (výška, šířka)

# Načtení predikované masky z HDF5
with h5py.File(predicted_mask_path, "r") as f:
    predicted_shape = f["mask"].shape  # Rozměry masky
    predicted_mask_dataset = f["mask"]

# Ověření, že masky mají stejné rozměry
    assert predicted_shape == ground_truth_shape, "Masky nemají stejné rozměry!"

    # Příprava akumulátorů pro průnik, sjednocení a celkové počty pixelů


    # Iterace přes bloky
    height, width = predicted_shape
    for y_start in tqdm(range(0, height, tile_size), desc="Processing blocks (rows)"):
        for x_start in range(0, width, tile_size):
            # Konec bloku
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)

            # Načtení bloku z predikované masky
            predicted_block = predicted_mask_dataset[y_start:y_end, x_start:x_end]

            # Načtení bloku z ground truth masky
            ground_truth_block = np.array(
                ground_truth_slide.read_region((x_start, y_start), 0, (x_end - x_start, y_end - y_start))
            )[:, :, 0] > 0  # Předpokládáme binární masku s hodnotami >0 jako True

            # Převod bloků na boolean
            predicted_block = predicted_block.astype(bool)
            ground_truth_block = ground_truth_block.astype(bool)

            # Výpočet průniku a sjednocení
            intersection = np.sum(predicted_block & ground_truth_block)
            union = np.sum(predicted_block | ground_truth_block)

            # Akumulace
            intersection_sum += intersection
            union_sum += union
            predicted_sum += np.sum(predicted_block)
            ground_truth_sum += np.sum(ground_truth_block)

# Výpočet Dice a Jaccard indexů
dice_index = (2 * intersection_sum) / (predicted_sum + ground_truth_sum)
jaccard_index = intersection_sum / union_sum

print(f"Dice Index: {dice_index:.4f}")
print(f"Jaccard Index: {jaccard_index:.4f}")
