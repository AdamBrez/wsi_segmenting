import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide
import matplotlib.patches as mpatches

"""
    Skript vytvoří masku tkáně za pomocí WSI obrazu.
    Určeno pro zdravou tkáň, kde ground truth maska neexistuje.
    Masky se uloží ve formátu .npy
    Uživatel by měl upravit smyčku pro iteraci souborů a výstupní cesty.
"""
# TODO: Uživatel by měl upravit tuto smyčku a cesty k souborům
# Příklad pro jeden soubor, nahraďte 'tumor_009.tif' názvem vašeho souboru se zdravou tkání
# a upravte výstupní název souboru.
for x in range(95,100):
    healthy_wsi_filename = f"normal_0{x}.tif" # PŘÍKLAD: Nahraďte názvem vašeho souboru
    output_mask_filename_base = f"tissue_mask_0{x}" # PŘÍKLAD: Základ pro výstupní název

    slide_path = r"C:\Users\USER\Desktop\normal_wsi\\" + healthy_wsi_filename
    # mask_path a slide_mask se již nebudou používat

    # Otevřeme slide
    try:
        slide = openslide.OpenSlide(slide_path)
    except openslide.OpenSlideError as e:
        print(f"Chyba při otevírání WSI souboru {slide_path}: {e}")
        # continue # Pokud je ve smyčce, přeskočí na další iteraci
        exit() # Nebo ukončí skript, pokud zpracovává jen jeden soubor

    selected_level = 6 # Úroveň pro generování masky, můžete upravit
    if selected_level >= slide.level_count:
        print(f"Varování: Zvolená úroveň {selected_level} je mimo rozsah dostupných úrovní ({slide.level_count}). Používám nejnižší dostupnou úroveň.")
        selected_level = slide.level_count - 1

    level_dims = slide.level_dimensions[selected_level]

    # Načtení náhledu WSI
    # Pozor: read_region vrací RGBA, takže poslední kanál je alfa
    thumbnail_rgba = slide.read_region(location=(0, 0),
                                    level=selected_level,
                                    size=level_dims)

    thumbnail_rgb = np.array(thumbnail_rgba.convert("RGB"))  # převedeme do RGB

    # Dvojprahová filtrace pro detekci tkáně
    image_for_thresholding = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR) # OpenCV očekává BGR
    hsv = cv2.cvtColor(image_for_thresholding, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_black = cv2.inRange(v, 0, 50) # Detekce velmi tmavých oblastí (potenciální pozadí)

    mask_not_black = cv2.bitwise_not(mask_black)
    # Kombinace masky saturace a masky "ne černé" pro získání tkáně
    tissue_mask_from_wsi = cv2.bitwise_and(mask_s, mask_not_black)

    # Morfologické operace pro vyčištění masky tkáně
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(tissue_mask_from_wsi, cv2.MORPH_OPEN, kernel)
    mask_final = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # Finální maska tkáně

    # Uložení výsledné masky tkáně
    output_save_path = r"C:\Users\USER\Desktop\colab_unet\normal_lowres\\" + output_mask_filename_base + ".npy" # TODO: Upravte cestu pro ukládání
    np.save(output_save_path, mask_final)
    print(f"Maska tkáně byla uložena do: {output_save_path}")

    print(mask_final.shape) # Zobrazíme rozměry masky

    ####################################################################################################
    ############################## Vizualizace vygenerované masky ######################################
    ####################################################################################################

    plt.figure(figsize=(15, 5))

    # Původní obrázek
    plt.subplot(1, 3, 1)
    plt.imshow(thumbnail_rgb) # Zobrazujeme původní RGB náhled
    plt.axis("off")
    plt.title("Původní obrázek (náhled)")

    # Vygenerovaná maska tkáně
    plt.subplot(1, 3, 2)
    plt.imshow(mask_final, cmap="gray") # Zobrazujeme finální binární masku
    plt.axis("off")
    plt.title("Vygenerovaná maska tkáně")

    # Maska tkáně překrytá na původním obraze
    plt.subplot(1, 3, 3)
    plt.imshow(thumbnail_rgb)
    plt.imshow(mask_final, cmap='Blues', alpha=0.5) # Modré překrytí masky
    plt.axis("off")
    plt.title("Maska na původním obraze")

    # Přidání legendy k třetímu subplotu
    tissue_patch = mpatches.Patch(color='blue', alpha=0.5, label='Vygenerovaná maska tkáně')
    plt.legend(handles=[tissue_patch],
            loc='upper right', bbox_to_anchor=(1.25, 1), # Upraveno pro lepší umístění
            frameon=True, facecolor='white', framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Upraveno pro zobrazení legendy
    plt.show()

    slide.close()