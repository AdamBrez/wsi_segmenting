import os
import numpy as np
from matplotlib import pyplot as plt
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")
import openslide

for x in range(1,67):
    slide_id = f"{x:03d}"  
    mask_path = fr"F:\histology_lungs\histology_lungs\converted\train\mask\mask_{slide_id}.tif"
    a = mask_path.split("\\")[-1]
    print(f"Zpracovává se wsi č.: {a}")
    # Otevřeme slide

    slide_mask = openslide.OpenSlide(mask_path)
    selected_level = 3
    level_dims = slide_mask.level_dimensions[selected_level]
    print(level_dims)
    lowres_mask = slide_mask.read_region(location=(0, 0),
                                        level=selected_level,
                                        size=level_dims)
    lowres_mask_np = np.array(lowres_mask.convert("L"))
    lowres_mask_np = (lowres_mask_np > 128).astype(np.uint8) * 255  # binarizace masky
    np.save(rf"F:\histology_lungs\histology_lungs\converted\train\gt_lowres\mask_{slide_id}_cancer.npy", lowres_mask_np)
    if x < 4:
        plt.imshow(lowres_mask_np, cmap='gray')
        plt.show()
    