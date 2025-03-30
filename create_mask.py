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
    Paralelizovaná verze skriptu pro segmentaci WSI.
    Používá dávkové zpracování (batch processing) pro zrychlení inference.
"""

# Cesta k modelu a obrázkům
model_weights_path = r"C:\Users\USER\Desktop\weights\unet_16_3_100e.pth"
wsi_image_path = r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_044.tif"  # Cesta k WSI obrazu
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\mask044_bez_ol_16_3_100e_parallel.h5"
tile_size = 256  # Velikost jednotlivých výřezů (tiles)
overlap = 0  # Překryv mezi dlaždicemi
threshold = 0.5  # Prahování pro binární masku
batch_size = 16  # Počet dlaždic zpracovávaných najednou

def process_batch(model, batch_tensors, device):
    """Zpracování dávky dlaždic modelem."""
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        predictions = model(batch)
        predictions = torch.sigmoid(predictions).squeeze().cpu().numpy()
    
    # Prahování na boolean masku
    if len(predictions.shape) == 2:  # Pokud je jen jedna dlaždice v dávce
        predictions = np.expand_dims(predictions, 0)
    binary_predictions = predictions >= threshold
    return binary_predictions

def main():
    start_time = time.time()

    # Načtení modelu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Model bude spuštěn na zařízení: {device}")
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
    
    # Iterace přes dlaždice na nejvyšší úrovni
    cols, rows = deepzoom.level_tiles[level]
    
    # Vytvoření seznamu souřadnic dlaždic
    tile_coords = [(col, row) for row in range(rows) for col in range(cols)]
    
    # Vytvoření HDF5 souboru pro ukládání masek
    with h5py.File(output_hdf5_path, "w") as hdf5_file:
        # Vytvoření datasetu pro masku
        dset = hdf5_file.create_dataset(
            "mask",
            shape=level_dimensions[::-1],  # (výška, šířka)
            dtype=bool,  # Boolean pro úsporu paměti
            chunks=(tile_size, tile_size),  # Chunky odpovídají velikosti tiles
            compression="gzip"  # Komprese pro úsporu místa
        )
        
        # Zpracování dlaždic po dávkách
        for batch_idx in tqdm(range(0, len(tile_coords), batch_size), desc="Processing batches"):
            batch_coords = tile_coords[batch_idx:batch_idx + batch_size]
            
            # Načtení dlaždic sekvenčně (nelze použít multiprocessing s OpenSlide)
            batch_tensors = []
            for col, row in batch_coords:
                tile = deepzoom.get_tile(level, (col, row))
                tile = tile.convert("RGB")  # Převod na RGB
                batch_tensors.append(ToTensor()(tile))
            
            # Zpracování dávky dlaždic
            binary_predictions = process_batch(model, batch_tensors, device)
            
            # Uložení predikovaných dlaždic do HDF5 datasetu
            for i, (col, row) in enumerate(batch_coords):
                x, y = col * tile_size, row * tile_size
                if i < len(binary_predictions):  # Kontrola pro případ poslední neúplné dávky
                    dset[y:y + tile_size, x:x + tile_size] = binary_predictions[i]
    
    print(f"Predikovaná maska byla uložena do {output_hdf5_path}.")
    
    end_time = time.time()
    print(f"Skript běžel {end_time - start_time:.2f} sekund.")

if __name__ == "__main__":
    main()