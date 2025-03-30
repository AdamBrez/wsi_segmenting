import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm
import h5py
import openslide
from openslide.deepzoom import DeepZoomGenerator

import time
import segmentation_models_pytorch as smp
"""
    Načítá se celé WSI, to je dále rozřezáváno a posláno do sítě.
    Výstupem je binární maska, která je ukládána do HDF5 souboru.
    Datový typ v HDF5 souboru je boolean, což umožňuje úsporu místa.
    Není zde překryv.
    Skript běžel 739.58 sekudn u WSI - tumor_091.tif
"""

# # Cesta k modelu a obrázkům
model_weights_path = r"C:\Users\USER\Desktop\weights\unet_16_3_100e.pth"
wsi_image_path = r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_018.tif"  # Cesta k WSI obrazu
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\test_opt3_t_018.h5"
tile_size = 256  # Velikost jednotlivých výřezů (tiles)
overlap = 0  # Překryv mezi dlaždicemi
threshold = 0.5  # Prahování pro binární masku

start_time = time.time()

# Načtení modelu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Načtení WSI pomocí OpenSlide
wsi = openslide.OpenSlide(wsi_image_path)

# Vytvoření DeepZoomGeneratoru
deepzoom = DeepZoomGenerator(wsi, tile_size=tile_size, overlap=overlap, limit_bounds=True)

# Použití nejvyšší úrovně pyramidy (nativní rozlišení)
level = deepzoom.level_count - 1  # Nejvyšší rozlišení
level_dimensions = deepzoom.level_dimensions[level]
print(f"Rozměry WSI na nejvyšší úrovni: {level_dimensions}")

# Vytvoření HDF5 souboru pro ukládání masek
with h5py.File(output_hdf5_path, "w") as hdf5_file:
    # Vytvoření datasetu pro masku
    dset = hdf5_file.create_dataset(
        "mask",
        shape=level_dimensions[::-1],  # (výška, šířka)
        # shape=(level_dimensions[1], level_dimensions[0], 3),
        dtype=bool,  # Boolean pro úsporu paměti
        chunks=(tile_size, tile_size),  # Chunky odpovídají velikosti tiles
        compression="gzip"  # Komprese pro úsporu místa
    )

    # Iterace přes dlaždice na nejvyšší úrovni
    cols, rows = deepzoom.level_tiles[level]
    for row in tqdm(range(rows), desc="Processing rows"):
        for col in range(cols):
            # Načtení dlaždice z DeepZoomGenerator
            tile = deepzoom.get_tile(level, (col, row))
            tile = tile.convert("RGB")  # Převod na RGB
            tile_tensor = ToTensor()(tile).unsqueeze(0).to(device)

            # # Inferování s modelem
            with torch.no_grad():
                prediction = model(tile_tensor)
                prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()

            # # Prahování na boolean masku
            binary_tile = prediction >= threshold

            # # Získání souřadnic dlaždice
            # gray_tile = np.array(tile.convert("RGB"))
            x, y = col * tile_size, row * tile_size
            
            # Uložení predikované dlaždice do HDF5 datasetu
            dset[y:y + tile_size, x:x + tile_size] = binary_tile

print(f"Predikovaná maska byla uložena do {output_hdf5_path}.")

end_time = time.time()
print(f"Skript běžel {end_time - start_time:.2f} sekund.")
