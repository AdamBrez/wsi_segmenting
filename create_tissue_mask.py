import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide

"""
    Skript vytvoří masku CELÉ (zdravá + nadorová) tkáně za pomocí WSI obazu a jeho ground truth.
    Masky se uloží ve formátu .npy
    x: číslo WSI, pro které chceme vytvořit masku
"""
for x in range(130, 131):
    slide_id = f"{x:03d}"  
    # Cesta k WSI
    try:
        slide_path = rf"F:\wsi_dir_test\test_{slide_id}.tif"
        mask_path = rf"F:\wsi_dir_test\mask_{slide_id}.tif"
        a = slide_path.split("\\")[-1]

        # Otevřeme slide
        slide = openslide.OpenSlide(slide_path)
        slide_mask = openslide.OpenSlide(mask_path)

        selected_level = 6
        level_dims = slide.level_dimensions[selected_level]

        # Pozor: read_region vrací RGBA, takže poslední kanál je alfa
        thumbnail_rgba = slide.read_region(location=(0, 0),
                                        level=selected_level,
                                        size=level_dims)

        mask_thumbnail = slide_mask.read_region(location=(0, 0),
                                                level=selected_level,
                                                size=level_dims)

        mask_thumb_np = np.array(mask_thumbnail.convert("L"))
        mask_thumb_np = (mask_thumb_np > 128).astype(np.uint8) * 255 # binarizace masky
        thumbnail_rgb = np.array(thumbnail_rgba.convert("RGB"))  # převedeme do RGB

        # dvojprahová filtrace (16.3.2025)
        image = cv2.cvtColor(np.array(thumbnail_rgb), cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_black = cv2.inRange(v, 0, 50)

        mask_not_black = cv2.bitwise_not(mask_black)
        mask_combined = cv2.bitwise_and(mask_s, mask_not_black)

        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Spojení vytvořené masky prahováním s ground truth maskou
        mask_final = cv2.bitwise_or(mask_combined, mask_thumb_np)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)

        np.save(rf"C:\Users\USER\Desktop\colab_unet\test_lowres_masky\mask_{slide_id}.npy", mask_final)
        print("Maska byla uložena.")
    except Exception as e:
        print(f"Chyba při zpracování souboru {slide_id}: {e}")

    ####################################################################################################
    ############################## vizualice překrytí masek ############################################
    ####################################################################################################

    import matplotlib.patches as mpatches

    plt.figure(figsize=(15, 5))

    # Původní obrázek
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Původní obrázek")

    # Finální maska
    plt.subplot(1, 3, 2)
    plt.imshow(mask_final, cmap="gray") # Zobrazení finální masky
    plt.axis("off")
    plt.title("Finální maska tkáně")


    # Kompozitní vizualizace s barevným kódováním
    plt.subplot(1, 3, 3)
    overlay = np.zeros((mask_final.shape[0], mask_final.shape[1], 3), dtype=np.uint8)
    overlay[mask_final > 0] = [0, 0, 255]  # Modrá pro mask_clean
    mask_thumb_array = np.array(mask_thumbnail.convert("L"))
    overlay[mask_thumb_array > 0] = [255, 0, 0]  # Červená pro mask_thumbnail
    # Překrytí - fialová barva
    overlay[(mask_final > 0) & (mask_thumb_array > 0)] = [255, 0, 255]
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Barevná vizualizace překrytí")

    # Přidání legendy k třetímu subplotu
    blue_patch = mpatches.Patch(color='blue', label='Mask Clean')
    red_patch = mpatches.Patch(color='red', label='Mask Thumbnail')
    purple_patch = mpatches.Patch(color='magenta', label='Překrytí')
    plt.legend(handles=[blue_patch, red_patch, purple_patch], 
            loc='upper right', bbox_to_anchor=(1.05, 1),
            frameon=True, facecolor='white', framealpha=0.8)

    plt.tight_layout()
    plt.show()
    plt.close(4)  # Zavření figury po zobrazení