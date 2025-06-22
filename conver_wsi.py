import pyvips
import openslide
import os
from pathlib import Path

# --- Konfigurace ---
input_folder = r"F:\histology_lungs\histology_lungs\preprocessed\train\data"
output_folder = r"F:\histology_lungs\histology_lungs\converted\train\data"

def convert_tif_for_openslide(input_path, output_path):
    """
    Nacte TIFF soubor pomoci pyvips a ulozi ho jako pyramidalni TIFF
    kompatibilni s OpenSlide.
    """
    try:
        print(f"Nacitani snimku z: {input_path}")
        image = pyvips.Image.new_from_file(input_path, access='sequential')
        print(f"Snimek nacten. Rozmery (sirka x vyska): {image.width}x{image.height}, Pocet pasiem: {image.bands}")

        print(f"Ukladani snimku do: {output_path} (pyramidalni TIFF, lzw komprese)")
        image.tiffsave(output_path, tile=True, pyramid=True, compression='lzw', bigtiff=True)
        print("Snimek uspesne preulozen.")
        return True
    except pyvips.Error as e:
        print(f"Chyba pyvips pri zpracovani {input_path}: {e}")
        return False
    except Exception as e:
        print(f"Neznama chyba pri zpracovani {input_path}: {e}")
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

def process_files_in_folder():
    """
    Projde vsechny soubory v input slozce, overi je pres OpenSlide
    a pokud nejdou otevrit, prevede je a ulozi do output slozky.
    """
    # Zkontroluj, zda slozky existuji
    if not os.path.exists(input_folder):
        print(f"CHYBA: Input slozka neexistuje: {input_folder}")
        return
    
    # Vytvor output slozku, pokud neexistuje
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistiky
    total_files = 0
    openslide_ok = 0
    converted = 0
    failed = 0
    
    # Projdi vsechny soubory v input slozce
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            total_files += 1
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"\n--- Zpracovavam soubor: {filename} ---")
            
            # Over, zda se da otevrit pres OpenSlide
            if verify_with_openslide(input_path):
                print(f"✓ Soubor {filename} se da otevrit pres OpenSlide - kopiruji do vystupni slozky")
                # Pokud uz funguje s OpenSlide, jen ho zkopiruj
                try:
                    import shutil
                    shutil.copy2(input_path, output_path)
                    openslide_ok += 1
                except Exception as e:
                    print(f"Chyba pri kopirovani {filename}: {e}")
                    failed += 1
            else:
                print(f"✗ Soubor {filename} nejde otevrit pres OpenSlide - prevadim...")
                # Pokud nejde otevrit, preved ho
                if convert_tif_for_openslide(input_path, output_path):
                    # Over, zda prevedeny soubor funguje
                    if verify_with_openslide(output_path):
                        print(f"✓ Soubor {filename} uspesne preveden a funguje s OpenSlide")
                        converted += 1
                    else:
                        print(f"✗ Prevedeny soubor {filename} stale nejde otevrit pres OpenSlide")
                        failed += 1
                else:
                    print(f"✗ Prevod souboru {filename} se nezdaril")
                    failed += 1
    
    # Vysledne statistiky
    print(f"\n=== SHRNUTI ===")
    print(f"Celkem zpracovanych souboru: {total_files}")
    print(f"Soubory kompatibilni s OpenSlide: {openslide_ok}")
    print(f"Uspesne prevedene soubory: {converted}")
    print(f"Neuspesne soubory: {failed}")

if __name__ == "__main__":
    process_files_in_folder()

