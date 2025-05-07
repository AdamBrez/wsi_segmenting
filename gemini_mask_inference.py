# --- START OF FILE test_opt3.py (Opraveno) ---

import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import torch
# import torchvision.transforms.functional as TF  # <-- POTŘEBNÝ IMPORT

from tqdm import tqdm
import h5py
import openslide
from openslide.deepzoom import DeepZoomGenerator
import time
import segmentation_models_pytorch as smp
import torchvision.transforms as T
"""
    Načítá se celé WSI, to je dále rozřezáváno a posláno do sítě.
    Výstupem je binární maska, která je ukládána do HDF5 souboru.
    Datový typ v HDF5 souboru je boolean.
"""

# # Cesta k modelu a obrázkům
model_weights_path = r"C:\Users\USER\Desktop\weights\constant_lr_e50_11200len.pth" # Upraven název souboru podle trénovacího skriptu
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_068.tif"
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\mask068gemini_constant_lr.h5" # Změna názvu výstupního souboru
tile_size = 256
overlap = 0
threshold = 0.5

# --- Definice ImageNet statistik (stejně jako v basic_transform) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# --- Vytvoření transformace ---
# Použijeme transforms.Compose pro přehlednost
infer_transform = T.Compose([
    T.ToTensor(), # Převede PIL [0-255] na Tensor [0.0-1.0]
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalizuje
])
# --------------------------------------------------------------------

start_time = time.time()

# Načtení modelu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1) # encoder_weights=None, protože váhy načítáme
try:
    model.load_state_dict(torch.load(model_weights_path, map_location=device)) # Odstraněno weights_only=True pro jistotu, pokud není soubor uložen jen jako state_dict
    print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
except Exception as e:
    print(f"Chyba při načítání vah modelu: {e}")
    print("Ujistěte se, že cesta k vahám a formát souboru jsou správné.")
    exit()

model.to(device)
model.eval()

# Načtení WSI pomocí OpenSlide
try:
    wsi = openslide.OpenSlide(wsi_image_path)
    print(f"WSI úspěšně načteno: {wsi_image_path}")
except openslide.OpenSlideError as e:
    print(f"Chyba při načítání WSI: {e}")
    exit()

# Vytvoření DeepZoomGeneratoru
deepzoom = DeepZoomGenerator(wsi, tile_size=tile_size, overlap=overlap, limit_bounds=False)

# --- Kontrola a výběr správného levelu ---
# V tréninku používáte wanted_level=2
train_level_dims = wsi.level_dimensions[2]
print(f"Rozměry na trénovacím levelu 2 (WSI): {train_level_dims}")

# Najdeme odpovídající level v DeepZoom
# DeepZoom levely jsou číslované od nejnižšího rozlišení
# OpenSlide levely jsou číslované od nejvyššího rozlišení (0)
dz_level = -1
for i in range(deepzoom.level_count):
    if deepzoom.level_dimensions[i] == train_level_dims:
        dz_level = i
        break

if dz_level == -1:
    print(f"Chyba: Nepodařilo se najít DeepZoom level odpovídající WSI levelu 2 ({train_level_dims}).")
    # Pokud se nenajde přesná shoda, vybereme nejbližší nebo použijeme level s nejvyšším rozlišením
    # Prozatím použijeme původní logiku, ale s varováním.
    # dz_level = deepzoom.level_count - 1 # Nejvyšší rozlišení deepzoomu
    print(f"DeepZoom level dimensions:")
    for i in range(deepzoom.level_count): print(f" Level {i}: {deepzoom.level_dimensions[i]}")
    print("Používám nejvyšší rozlišení DeepZoom (level_count - 1), což nemusí odpovídat tréninku!")
    dz_level = deepzoom.level_count - 1 # Default k nejvyššímu rozlišení DZ, pokud shoda nenalezena

level_dimensions = deepzoom.level_dimensions[dz_level]
print(f"Použitý DeepZoom level: {dz_level}")
print(f"Použité rozměry pro predikci (Šířka, Výška): {level_dimensions}")
print(f"Očekávaný tvar výstupní masky (Výška, Šířka): {level_dimensions[::-1]}")
# -----------------------------------------

# Vytvoření HDF5 souboru pro ukládání masek
try:
    with h5py.File(output_hdf5_path, "w") as hdf5_file:
        print(f"Vytvářím HDF5 soubor: {output_hdf5_path}")
        # Vytvoření datasetu pro masku
        dset = hdf5_file.create_dataset(
            "mask",
            shape=level_dimensions[::-1], # (výška, šířka)
            dtype=np.uint8, # <-- ZMĚNA: Ukládáme 0/1 jako uint8 pro snazší vizualizaci
            chunks=(tile_size, tile_size),
            compression="gzip"
        )
        print(f"HDF5 dataset 'mask' vytvořen s tvarem {dset.shape} a typem {dset.dtype}")

        # Iterace přes dlaždice na zvoleném levelu
        cols, rows = deepzoom.level_tiles[dz_level]
        print(f"Počet dlaždic ke zpracování: {cols} x {rows} = {cols * rows}")

        for row in tqdm(range(rows), desc="Zpracování řádků"):
            for col in range(cols):
                try:
                    # Načtení dlaždice
                    tile = deepzoom.get_tile(dz_level, (col, row))
                    tile = tile.convert("RGB")

                    # --- Aplikace transformace (ToTensor + Normalizace) ---
                    tile_tensor = infer_transform(tile).unsqueeze(0).to(device)
                    # -------------------------------------------------------

                    # Inferování s modelem
                    with torch.inference_mode(): # Použití inference_mode
                        prediction = model(tile_tensor)
                        prediction = torch.sigmoid(prediction).squeeze().cpu().numpy() # Výstup [0, 1]

                    # Prahování na binární masku (0 nebo 1)
                    binary_tile = (prediction >= threshold).astype(np.uint8) # <-- Převod na uint8 (0, 1)

                    # Získání skutečných rozměrů dlaždice (může být menší na okrajích)
                    tile_h, tile_w = binary_tile.shape
                    # Výpočet souřadnic pro uložení
                    x_start = col * tile_size
                    y_start = row * tile_size
                    x_end = x_start + tile_w
                    y_end = y_start + tile_h

                    # Oříznutí koncových souřadnic na rozměry datasetu (pro jistotu)
                    x_end = min(x_end, level_dimensions[0])
                    y_end = min(y_end, level_dimensions[1])

                    # Výpočet rozměrů pro zápis (pokud byla dlaždice oříznuta)
                    write_w = x_end - x_start
                    write_h = y_end - y_start

                    # Uložení predikované dlaždice (pouze relevantní část) do HDF5
                    if write_w > 0 and write_h > 0:
                         dset[y_start:y_end, x_start:x_end] = binary_tile[:write_h, :write_w]
                    # else:
                    #      print(f"Varování: Nulový rozměr pro zápis pro dlaždici ({row},{col}). Přeskakuji.")

                except Exception as tile_error:
                    print(f"Chyba při zpracování dlaždice ({row}, {col}): {tile_error}")
                    # Pokračovat na další dlaždici

except Exception as h5_error:
    print(f"Chyba při práci s HDF5 souborem: {h5_error}")
finally:
    wsi.close() # Zajistit uzavření WSI souboru
    print("WSI soubor uzavřen.")


print(f"Predikovaná maska byla uložena do {output_hdf5_path}.")

end_time = time.time()
print(f"Skript běžel {end_time - start_time:.2f} sekund.")

# --- Doporučení pro vizualizaci (show_h5_img.py) ---
# V show_h5_img.py se ujistěte, že čtete masku jako uint8:
#
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

hdf5_path = r"C:\Users\USER\Desktop\test_output\mask068gemini_constant_lr.h5" # Cesta k opravenému souboru

with h5py.File(hdf5_path, "r") as f:
    mask = f["mask"][:] # Načte jako uint8 (0, 1)

print(f"Načtená maska - shape: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")

scale_factor = 0.1 # Menší faktor pro přehlednost
if mask.shape[0] > 0 and mask.shape[1] > 0: # Kontrola neprázdné masky
    small_mask = np.array(Image.fromarray(mask).resize( # Převod uint8 na PIL funguje
        (int(mask.shape[1] * scale_factor), int(mask.shape[0] * scale_factor)),
        resample=Image.NEAREST)) # NEAREST je vhodný pro binární masky
else:
    small_mask = mask # Pokud je maska prázdná, ponechat ji tak

print(f"Zmenšená maska - max hodnota: {np.max(small_mask) if small_mask.size > 0 else 'N/A'}")

plt.figure(figsize=(10, 10))
plt.imshow(small_mask, cmap="gray", vmin=0, vmax=1) # Explicitně nastavit vmin/vmax
plt.title(f"Predikovaná maska (zmenšeno na {scale_factor*100:.0f} %)")
plt.axis("off")
plt.show()

# --- END OF FILE test_opt3.py (Opraveno) ---