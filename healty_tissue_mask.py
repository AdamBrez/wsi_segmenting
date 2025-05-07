import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide

"""
    Skript vytvoří masku ZDRAVÉ tkáně za pomocí WSI obrazu a jeho ground truth.
    Masky se uloží ve formátu .npy
    x: číslo WSI, pro které chceme vytvořit masku
"""
# x = 15
for x in range(10,100):
    # Cesta k WSI
    slide_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_0"+f"{x}"+".tif"
    mask_path = r"C:\Users\USER\Desktop\wsi_dir\mask_0"+f"{x}"+".tif"
    a = slide_path.split("\\")[-1]
    print(f"Zpracovává se wsi č.: {a}")
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

    # Převedení ground truth masky na binární formát (0 a 255)
    mask_thumb_np = np.array(mask_thumbnail.convert("L"))
    mask_thumb_np = (mask_thumb_np > 128).astype(np.uint8) * 255  # binarizace masky
    thumbnail_rgb = np.array(thumbnail_rgba.convert("RGB"))  # převedeme do RGB

    # Dvojprahová filtrace pro detekci tkáně
    image = cv2.cvtColor(np.array(thumbnail_rgb), cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    _, mask_s = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_black = cv2.inRange(v, 0, 50)

    mask_not_black = cv2.bitwise_not(mask_black)
    mask_combined = cv2.bitwise_and(mask_s, mask_not_black)

    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # ZMĚNA: Odečtení nádorové tkáně od celkové tkáně místo spojení
    # Vytvoření "inverzní" masky nádoru (255 kde není nádor, 0 kde je)
    tumor_mask_inv = cv2.bitwise_not(mask_thumb_np)

    # Ponechání pouze tkáně, která NENÍ nádorová
    healthy_tissue_mask = cv2.bitwise_and(tissue_mask, tumor_mask_inv)

    # Další čištění masky pro odstranění malých izolovaných oblastí
    healthy_tissue_mask = cv2.morphologyEx(healthy_tissue_mask, cv2.MORPH_OPEN, kernel)
    healthy_tissue_mask = cv2.morphologyEx(healthy_tissue_mask, cv2.MORPH_CLOSE, kernel)

    # Uložení masky zdravé tkáně
    output_path = r"C:\Users\USER\Desktop\colab_unet\masky_healthy\mask_0"+f"{x}"+".npy"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, healthy_tissue_mask)
    print(f"Maska zdravé tkáně byla uložena do {output_path}")

    ####################################################################################################
    ############################## vizualizace překrytí masek ##########################################
    ####################################################################################################

    # import matplotlib.patches as mpatches

    # plt.figure(figsize=(12, 8))

    # # Původní obrázek
    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    # plt.axis("off")
    # plt.title("Původní obrázek")

    # # Maska celkové tkáně
    # plt.subplot(2, 2, 2)
    # plt.imshow(tissue_mask, cmap="Blues")
    # plt.axis("off")
    # plt.title("Celková maska tkáně")

    # # Maska nádorové tkáně
    # plt.subplot(2, 2, 3)
    # plt.imshow(mask_thumb_np, cmap="Reds")
    # plt.axis("off")
    # plt.title("Maska nádorové tkáně")

    # # Maska zdravé tkáně
    # plt.subplot(2, 2, 4)
    # plt.imshow(healthy_tissue_mask, cmap="Greens")
    # plt.axis("off")
    # plt.title("Maska zdravé tkáně")

    # plt.tight_layout()

    # # Druhý graf s překrytím
    # plt.figure(figsize=(12, 8))

    # # Kompozitní vizualizace s barevným kódováním
    # overlay = np.zeros((tissue_mask.shape[0], tissue_mask.shape[1], 3), dtype=np.uint8)
    # overlay[tissue_mask > 0] = [0, 0, 255]  # Modrá pro celkovou tkáň
    # overlay[mask_thumb_np > 0] = [255, 0, 0]  # Červená pro nádorovou tkáň
    # overlay[healthy_tissue_mask > 0] = [0, 255, 0]  # Zelená pro zdravou tkáň

    # plt.imshow(image)
    # plt.imshow(overlay, alpha=0.5)
    # plt.axis("off")
    # plt.title("Barevná vizualizace jednotlivých typů tkání")

    # # Přidání legendy
    # # blue_patch = mpatches.Patch(color='blue', label='Celková tkáň')
    # red_patch = mpatches.Patch(color='red', label='Nádorová tkáň')
    # green_patch = mpatches.Patch(color='green', label='Zdravá tkáň')
    # plt.legend(handles=[red_patch, green_patch], 
    #         loc='upper right', bbox_to_anchor=(1.05, 1),
    #         frameon=True, facecolor='white', framealpha=0.8)

    # plt.tight_layout()
    # plt.show()