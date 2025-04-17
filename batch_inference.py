# --- START OF FILE batch_inference_no_overlap.py ---

import os
# !!! Zajistěte správnou cestu k OpenSlide !!!
try:
    openslide_dll_path = r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"
    if os.path.exists(openslide_dll_path):
        os.add_dll_directory(openslide_dll_path)
    else:
        print(f"Varování: Cesta k OpenSlide DLL neexistuje: {openslide_dll_path}")
except AttributeError:
    print("os.add_dll_directory není dostupné.")
except Exception as e:
    print(f"Nastala chyba při přidávání OpenSlide DLL: {e}")

import torch
# Potřebujeme PIL pro padding obrázků před tensorizací
from PIL import ImageOps
# Přidáme transformace pro normalizaci a ToTensor
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import h5py
import openslide
from openslide.deepzoom import DeepZoomGenerator
import time
import segmentation_models_pytorch as smp
import datetime

"""
    Načítá se celé WSI, je rozřezáváno BEZ PŘEKRYVU.
    Zpracování probíhá v DÁVKÁCH (batch processing).
    Okrajové (menší) dlaždice jsou PADOVÁNY na TILE_SIZE pro batching.
    Používá se normalizace.
    Výsledná predikce je oříznuta na původní velikost dlaždice před uložením.
    Výstupem je binární maska (HDF5, bool).
"""

# --- Konfigurace ---
model_weights_path = r"C:\Users\USER\Desktop\weights\unet_smp_e50_9920len.pth"
wsi_image_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif" # Zkontroluj cestu
output_hdf5_path = r"C:\Users\USER\Desktop\test_output\pred_089.h5" # Nový název!

# Velikost, kterou očekává model (a velikost dlaždice bez overlapu)
TILE_SIZE = 256
# Překryv NENÍ použit
OVERLAP = 0

BATCH_SIZE = 64    # Upravte podle VRAM (můžete zkusit 32 nebo 64)
THRESHOLD = 0.5
TARGET_LEVEL_OFFSET = 2 # Použijeme nejvyšší rozlišení (level 0)

# --- Inicializace ---
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Používám zařízení: {device}")
print(f"Model očekává vstup: {TILE_SIZE}x{TILE_SIZE}")
print(f"DZG parametry: tile_size={TILE_SIZE}, overlap={OVERLAP}")

# Načtení modelu
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
try:
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Soubor s vahami nebyl nalezen: {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
    print(f"Váhy modelu úspěšně načteny z: {model_weights_path}")
except TypeError:
     model.load_state_dict(torch.load(model_weights_path, map_location=device))
except Exception as e:
    print(f"Chyba při načítání vah modelu: {e}")
    exit()
model.to(device)
model.eval()

# Normalizační transformace
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Načtení WSI
try:
    wsi = openslide.OpenSlide(wsi_image_path)
except openslide.OpenSlideError as e:
    print(f"Chyba při otevírání WSI: {e}")
    exit()
except FileNotFoundError:
    print(f"WSI soubor nebyl nalezen: {wsi_image_path}")
    exit()


# Vytvoření DeepZoomGeneratoru BEZ OVERLAPU
deepzoom = DeepZoomGenerator(wsi, tile_size=TILE_SIZE, overlap=OVERLAP, limit_bounds=False)

# Výběr úrovně pro zpracování
level = deepzoom.level_count - 1 - TARGET_LEVEL_OFFSET
if level < 0 or level >= deepzoom.level_count:
     print(f"Chyba: Neplatný level index {level} (offset={TARGET_LEVEL_OFFSET}, count={deepzoom.level_count}).")
     level = deepzoom.level_count - 1
     print(f"Používám nejvyšší dostupnou DZG úroveň: {level}")

try:
    level_dimensions = deepzoom.level_dimensions[level]
    level_tiles = deepzoom.level_tiles[level]
    print(f"Cílová OpenSlide úroveň: {wsi.level_count - 1 - level}")
except IndexError:
    print(f"Chyba: Neplatný level index {level}.")
    wsi.close()
    exit()

print(f"Zpracovávám DZG úroveň {level} s rozměry {level_dimensions} (šířka, výška).")
print(f"Očekávaný počet dlaždic (sloupce, řádky): {level_tiles}")
output_shape = (level_dimensions[1], level_dimensions[0]) # (výška, šířka)

# --- Zpracování ---
with h5py.File(output_hdf5_path, "w") as hdf5_file:
    dset = hdf5_file.create_dataset(
        "mask",
        shape=output_shape,
        dtype=bool,
        chunks=(TILE_SIZE, TILE_SIZE),
        compression="gzip"
    )

    hdf5_file.attrs["day_of_creation"] = datetime.datetime.now().isoformat()
    hdf5_file.attrs["tile_size"] = TILE_SIZE
    hdf5_file.attrs["overlap"] = OVERLAP
    hdf5_file.attrs["batch_size"] = BATCH_SIZE
    hdf5_file.attrs["model_weights"] = model_weights_path.split("\\")[-1]
    hdf5_file.attrs["wsi_level"] = f"Velikost wsi na úrovni {level}: {level_dimensions}"
    hdf5_file.attrs["augmentation"] = "False"

    cols, rows = level_tiles
    total_tiles = rows * cols
    processed_tiles = 0
    batch_tiles_data = [] # Seznam pro tenzory dlaždic (padované na TILE_SIZE)
    batch_coords = []     # Seznam pro souřadnice a PŮVODNÍ velikosti (col, row, w, h)

    with torch.inference_mode():
        for row in tqdm(range(rows), desc="Processing rows"):
            for col in range(cols):
                tile = deepzoom.get_tile(level, (col, row))
                if tile is None:
                    continue

                tile_rgb = tile.convert("RGB")
                tile_w, tile_h = tile_rgb.size # PŮVODNÍ velikost

                # Dlaždice s nulovou velikostí přeskočíme
                if tile_w == 0 or tile_h == 0:
                    continue

                # --- Padding na TILE_SIZE ---
                # Pokud je dlaždice menší, dopadujeme ji na TILE_SIZE x TILE_SIZE
                delta_w = TILE_SIZE - tile_w
                delta_h = TILE_SIZE - tile_h
                padding = (0, 0, delta_w, delta_h) # (left, top, right, bottom)
                tile_padded = ImageOps.expand(tile_rgb, padding, fill=0) # fill=0 pro černou

                # Převod padované dlaždice na tensor A NORMALIZACE
                tile_tensor = to_tensor(tile_padded) # Vytvoří tensor [C, TILE_SIZE, TILE_SIZE]
                tile_tensor_normalized = normalize(tile_tensor)

                batch_tiles_data.append(tile_tensor_normalized) # Přidáme normalizovaný tensor
                # Uložíme si souřadnice a PŮVODNÍ velikost
                batch_coords.append({'col': col, 'row': row, 'w': tile_w, 'h': tile_h})

                processed_tiles += 1

                # Zpracování dávky
                if len(batch_tiles_data) == BATCH_SIZE or processed_tiles == total_tiles:
                    if not batch_tiles_data:
                        continue

                    # Spojení tenzorů do dávky (všechny mají TILE_SIZE x TILE_SIZE)
                    batch_tensor = torch.stack(batch_tiles_data).to(device)

                    # Inferování s modelem
                    prediction_output = model(batch_tensor) # Výstup [B, 1, TILE_SIZE, TILE_SIZE]
                    predictions = torch.sigmoid(prediction_output).cpu() # Výstup [B, 1, TILE_SIZE, TILE_SIZE] na CPU

                    # Zpracování výsledků dávky
                    for i in range(predictions.shape[0]):
                        pred_tensor = predictions[i].squeeze() # Výstup [TILE_SIZE, TILE_SIZE]
                        coords = batch_coords[i]
                        orig_w, orig_h = coords['w'], coords['h'] # Původní velikost dlaždice

                        # Prahování na boolean masku (celé TILE_SIZE x TILE_SIZE)
                        binary_pred_padded = (pred_tensor >= THRESHOLD).numpy().astype(bool)

                        # --- Oříznutí na původní velikost ---
                        # Z predikce (která má TILE_SIZE x TILE_SIZE) vezmeme jen
                        # levou horní část odpovídající původní velikosti dlaždice.
                        binary_tile_cropped = binary_pred_padded[:orig_h, :orig_w]

                        # --- Uložení do HDF5 ---
                        # Vypočítáme cílové souřadnice x, y v HDF5 souboru
                        x_start = coords['col'] * TILE_SIZE
                        y_start = coords['row'] * TILE_SIZE

                        # Ošetření okrajů HDF5 datasetu (pro jistotu)
                        y_end = min(y_start + orig_h, output_shape[0])
                        x_end = min(x_start + orig_w, output_shape[1])

                        # Zapíšeme oříznutou binární masku na správné místo
                        # Rozměry by měly sedět: (y_end - y_start = orig_h, x_end - x_start = orig_w)
                        if y_end > y_start and x_end > x_start: # Zkontrolujeme, jestli máme co zapsat
                           dset[y_start:y_end, x_start:x_end] = binary_tile_cropped[:(y_end - y_start), :(x_end - x_start)]

                    # Vyčištění dávky
                    batch_tiles_data.clear()
                    batch_coords.clear()

# --- Dokončení ---
wsi.close()
end_time = time.time()
print(f"\nPredikovaná maska byla uložena do {output_hdf5_path}.")
print(f"Skript (Batch, No Overlap, Padding, Norm, Level 0) běžel {end_time - start_time:.2f} sekund.")

# --- END OF FILE batch_inference_no_overlap.py ---