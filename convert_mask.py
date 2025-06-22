import os
import numpy as np
import tifffile
from PIL import Image
import pyvips
import openslide

# --- KONFIGURACE ---
input_folder = r"F:\histology_lungs\histology_lungs\preprocessed\valid\mask"
output_folder = r"F:\histology_lungs\histology_lungs\converted\valid\mask"

def convert_mask_single_lvl(input_path, output_path):
    # Nacteme cely TIFF (vcetne vsech kanalu)
    img = tifffile.imread(input_path)
    # img ma tvar (H, W, bands)
    # Sjednotime do jedne masky: kdekoli je jakykoli kanal >0, nastavime 1
    if img.ndim == 3:
        # pripad RGB ci RGBA
        bin_mask = np.any(img > 0, axis=2).astype(np.uint8)
    else:
        # uz je to jednoanalove
        bin_mask = (img > 0).astype(np.uint8)

    # Pro vetsi viditelnost muzeme masku skalovat na 255
    bin_mask *= 255

    # Ulozime ztratove bezeztrátovym LZW, photometric='minisblack'
    tifffile.imwrite(
        output_path,
        bin_mask,
        dtype='uint8',
        photometric='minisblack',
        compression='lzw',
        tile=(256,256),
        bigtiff=True
    )
    print(f"Hotovo: {output_path}")

def convert_mask_with_pyramid(input_path, output_path, levels=(2,4,8,16)):
    """
    Prevede masku na pyramidalni TIFF kompatibilni s OpenSlide.
    """
    try:
        print(f"Nacitani masky z: {input_path}")
        # 1) Nacteme puvodni viceanalovy TIFF
        img = tifffile.imread(input_path)
        # 2) Sjednotime do jedne 8-bitove bitmasky (0/255)
        if img.ndim == 3:
            bin_mask = (np.any(img > 0, axis=2)).astype(np.uint8) * 255
        else:
            bin_mask = (img > 0).astype(np.uint8) * 255

        height, width = bin_mask.shape
        print(f"Maska nactena. Rozmery: {width}x{height}")

        # 3) Otevreme TiffWriter pro vice IFD (bigtiff, LZW, dlazdice)
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            # --- zakladni uroven (IFD0) ---
            tif.write(
                bin_mask,
                photometric='minisblack',
                compression='lzw',
                tile=(256,256),
                subfiletype=0  # 0 = full-resolution image
            )

            # --- jednotlive overview urovne (IFD1, IFD2, ...) ---
            base = Image.fromarray(bin_mask)
            for down in levels:
                # velikost v overview
                w2, h2 = width//down, height//down
                # nearest-neighbor resize, aby se nerozmazaly okraje masky
                ov = base.resize((w2, h2), resample=Image.NEAREST)
                arr = np.asarray(ov, dtype=np.uint8)
                tif.write(
                    arr,
                    photometric='minisblack',
                    compression='lzw',
                    tile=(256,256),
                    subfiletype=1  # 1 = reduced-resolution (overview)
                )

        print(f"Pyramidalni maska ulozena do: {output_path}")
        return True
    except Exception as e:
        print(f"Chyba pri prevodu masky {input_path}: {e}")
        return False

def verify_with_openslide(tiff_path):
    """
    Pokusi se otevrit TIFF soubor pomoci OpenSlide a vrati True/False.
    """
    try:
        wsi = openslide.OpenSlide(tiff_path)
        wsi.close()
        return True
    except openslide.OpenSlideError:
        return False
    except Exception:
        return False

def process_masks_in_folder():
    """
    Projde vsechny masky v input slozce, overi je pres OpenSlide
    a pokud nejdou otevrit, prevede je pomoci pyramid funkce a ulozi do output slozky.
    Preskoci soubory obsahujici "kontrola" v nazvu.
    """
    # Zkontroluj, zda slozky existuji
    if not os.path.exists(input_folder):
        print(f"CHYBA: Input slozka neexistuje: {input_folder}")
        return
    
    # Vytvor output slozku, pokud neexistuje
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistiky
    total_files = 0
    skipped_files = 0
    openslide_ok = 0
    converted = 0
    failed = 0
    
    # Projdi vsechny soubory v input slozce
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            # Preskoc soubory s "kontrola" v nazvu
            if "kontrola" in filename.lower():
                print(f"⏭️  Preskakuji kontrolni soubor: {filename}")
                skipped_files += 1
                continue
                
            total_files += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"\n--- Zpracovavam masku: {filename} ---")
            
            # Over, zda se da otevrit pres OpenSlide
            if verify_with_openslide(input_path):
                print(f"✓ Maska {filename} se da otevrit pres OpenSlide - kopiruji do vystupni slozky")
                # Pokud uz funguje s OpenSlide, jen ji zkopiruj
                try:
                    import shutil
                    shutil.copy2(input_path, output_path)
                    openslide_ok += 1
                except Exception as e:
                    print(f"Chyba pri kopirovani {filename}: {e}")
                    failed += 1
            else:
                print(f"✗ Maska {filename} nejde otevrit pres OpenSlide - prevadim pomoci pyramid funkce...")
                # Pokud nejde otevrit, preved ji pomoci pyramid funkce
                if convert_mask_with_pyramid(input_path, output_path):
                    # Over, zda prevedena maska funguje
                    if verify_with_openslide(output_path):
                        print(f"✓ Maska {filename} uspesne prevedena a funguje s OpenSlide")
                        converted += 1
                    else:
                        print(f"✗ Prevedena maska {filename} stale nejde otevrit pres OpenSlide")
                        failed += 1
                else:
                    print(f"✗ Prevod masky {filename} se nezdaril")
                    failed += 1
    
    # Vysledne statistiky
    print(f"\n=== SHRNUTI ===")
    print(f"Celkem zpracovanych masek: {total_files}")
    print(f"Preskocenych kontrolnich souboru: {skipped_files}")
    print(f"Masky kompatibilni s OpenSlide: {openslide_ok}")
    print(f"Uspesne prevedene masky: {converted}")
    print(f"Neuspesne masky: {failed}")

if __name__ == "__main__":
    process_masks_in_folder()
