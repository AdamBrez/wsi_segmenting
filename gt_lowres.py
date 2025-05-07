import os
import numpy as np

os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide

for x in range(111,112):
    if x == 91 or x == 84 or x == 89:
        continue
    # slide_path = r"C:\Users\USER\Desktop\wsi_dir\tumor_0"+f"{x}"+".tif"
    mask_path = r"C:\Users\USER\Desktop\wsi_dir\mask_"+f"{x}"+".tif"
    a = mask_path.split("\\")[-1]
    print(f"Zpracovává se wsi č.: {a}")
    # Otevřeme slide

    slide_mask = openslide.OpenSlide(mask_path)
    selected_level = 6
    level_dims = slide_mask.level_dimensions[selected_level]
    lowres_mask = slide_mask.read_region(location=(0, 0),
                                        level=selected_level,
                                        size=level_dims)
    lowres_mask_np = np.array(lowres_mask.convert("L"))
    lowres_mask_np = (lowres_mask_np > 128).astype(np.uint8) * 255  # binarizace masky
    np.save(r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky\mask_"+f"{x}" +"_cancer"+".npy", lowres_mask_np)
    