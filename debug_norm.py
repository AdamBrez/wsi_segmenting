import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import staintools
from PIL import Image
import torch
from torchvision import transforms
import torchstain
import shutil # Pro mazání složky, pokud existuje

# Řešení pro OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

####################################################
####### KONFIGURACE CEST A REFERENČNÍHO OBRÁZKU #######
####################################################

# Cesta k referenčnímu obrázku (použije se pro obě metody)
ref_image_path = r"C:\Users\USER\Desktop\patches\patch_20250526_154106_01_cancer0.0pct.png"

# Vstupní složka s obrázky k normalizaci
input_patches_dir = r"C:\Users\USER\Desktop\patches"

# Výstupní složka pro normalizované obrázky
output_norm_patches_dir = r"C:\Users\USER\Desktop\norm_patches"

# Vytvoření výstupní složky (pokud neexistuje)
# Volitelně: Smazat a znovu vytvořit výstupní složku, pokud již existuje, aby se zabránilo starým souborům
if os.path.exists(output_norm_patches_dir):
    shutil.rmtree(output_norm_patches_dir) # Smaže složku a její obsah
os.makedirs(output_norm_patches_dir, exist_ok=True)
os.makedirs(os.path.join(output_norm_patches_dir, "staintools"), exist_ok=True)
os.makedirs(os.path.join(output_norm_patches_dir, "torchstain"), exist_ok=True)
os.makedirs(os.path.join(output_norm_patches_dir, "original"), exist_ok=True) # Pro kopii originálů

print(f"Referenční obrázek: {ref_image_path}")
print(f"Vstupní složka: {input_patches_dir}")
print(f"Výstupní složka: {output_norm_patches_dir}")

####################################################
####### PŘÍPRAVA NORMALIZÁTORŮ #######
####################################################

# --- Staintools Normalizer ---
print("\nInicializace Staintools normalizátoru...")
# Načtení referenčního obrázku pro Staintools
target_st_ref = staintools.read_image(ref_image_path)
# Standardizace jasu referenčního obrázku (důležité pro Staintools s LuminosityStandardizer)
target_st_ref_standardized = staintools.LuminosityStandardizer.standardize(target_st_ref)

# Vytvoření a fitování Staintools normalizátoru
# Můžete si vybrat metodu, např. 'macenko' nebo 'vahadane'
# normalizer_st = staintools.StainNormalizer(method='macenko')
normalizer_st = staintools.StainNormalizer(method='vahadane') # Používám Vahadane jako ve vašem finálním grafu
normalizer_st.fit(target_st_ref_standardized) # Fit na JASOVĚ STANDARDIZOVANOU referenci
print("Staintools normalizátor připraven.")

# --- TorchStain Normalizer ---
print("\nInicializace TorchStain normalizátoru...")
# Načtení referenčního obrázku pro TorchStain
target_ts_ref_cv = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)

# Transformace pro TorchStain
T_ts = transforms.Compose([
    transforms.ToTensor(),      # Převede obrázek (H,W,C) na tensor (C,H,W) a škáluje na [0,1]
    transforms.Lambda(lambda x: x * 255)  # Převede tensor z [0,1] na [0,255]
])

# Vytvoření a fitování TorchStain normalizátoru
normalizer_ts = torchstain.normalizers.MacenkoNormalizer(backend='torch') # Nebo jiný, např. Vahadane
normalizer_ts.fit(T_ts(target_ts_ref_cv)) # Fituje na tensor 0-255
print("TorchStain normalizátor připraven.")


####################################################
####### ZPRACOVÁNÍ A NORMALIZACE OBRÁZKŮ #######
####################################################
print(f"\nZpracování obrázků ze složky: {input_patches_dir}")

# Získání seznamu všech .png souborů ve vstupní složce
image_files = [f for f in os.listdir(input_patches_dir) if f.lower().endswith('.png')]

if not image_files:
    print("Ve vstupní složce nebyly nalezeny žádné .png obrázky.")
else:
    print(f"Nalezeno {len(image_files)} obrázků ke zpracování.")

for i, filename in enumerate(image_files):
    if filename == os.path.basename(ref_image_path):
        print(f"Přeskakuji referenční obrázek: {filename}")
        # Volitelně zkopírovat referenční obrázek do výstupních složek
        shutil.copy2(os.path.join(input_patches_dir, filename), os.path.join(output_norm_patches_dir, "staintools", f"REF_{filename}"))
        shutil.copy2(os.path.join(input_patches_dir, filename), os.path.join(output_norm_patches_dir, "torchstain", f"REF_{filename}"))
        shutil.copy2(os.path.join(input_patches_dir, filename), os.path.join(output_norm_patches_dir, "original", f"REF_{filename}"))
        continue

    current_image_path = os.path.join(input_patches_dir, filename)
    print(f"\n--- Zpracovávám ({i+1}/{len(image_files)}): {filename} ---")

    # --- Zkopírování originálu ---
    original_output_path = os.path.join(output_norm_patches_dir, "original", filename)
    shutil.copy2(current_image_path, original_output_path)
    print(f"Originál zkopírován do: {original_output_path}")

    # --- Normalizace pomocí Staintools ---
    try:
        print("Normalizace pomocí Staintools...")
        source_img_st = staintools.read_image(current_image_path)
        # Standardizace jasu zdrojového obrázku
        source_img_st_standardized = staintools.LuminosityStandardizer.standardize(source_img_st)
        # Transformace JASOVĚ STANDARDIZOVANÉHO zdrojového obrázku
        transformed_st = normalizer_st.transform(source_img_st_standardized)

        # Uložení výsledku Staintools
        staintools_output_path = os.path.join(output_norm_patches_dir, "staintools", f"st_{filename}")
        Image.fromarray(transformed_st).save(staintools_output_path)
        print(f"Výsledek Staintools uložen do: {staintools_output_path}")
    except Exception as e:
        print(f"Chyba při normalizaci pomocí Staintools pro {filename}: {e}")

    # --- Normalizace pomocí TorchStain ---
    try:
        print("Normalizace pomocí TorchStain...")
        source_img_ts_cv = cv2.cvtColor(cv2.imread(current_image_path), cv2.COLOR_BGR2RGB)
        
        # Normalizace
        ts_normalized_tensor, _, _ = normalizer_ts.normalize(I=T_ts(source_img_ts_cv), stains=True)

        # Převod výsledku TorchStain do zobrazitelné formy (numpy array H,W,C uint8)
        if ts_normalized_tensor.shape[0] == 3:  # [C, H, W] formát
            normalized_by_torchstain_display = ts_normalized_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
        else:  # [H, W, C] formát
            normalized_by_torchstain_display = ts_normalized_tensor.cpu().numpy().astype('uint8')
        
        # Uložení výsledku TorchStain
        torchstain_output_path = os.path.join(output_norm_patches_dir, "torchstain", f"ts_{filename}")
        Image.fromarray(normalized_by_torchstain_display).save(torchstain_output_path)
        print(f"Výsledek TorchStain uložen do: {torchstain_output_path}")
    except Exception as e:
        print(f"Chyba při normalizaci pomocí TorchStain pro {filename}: {e}")


####################################################
####### VOLITELNÉ ZOBRAZENÍ NĚKOLIKA VÝSLEDKŮ #######
####################################################
# Můžete přidat kód pro zobrazení několika náhodných originálů a jejich normalizovaných verzí
# pro rychlou vizuální kontrolu, pokud chcete.
# Například:
num_examples_to_show = 3
if len(image_files) > 1 : # Abychom se vyhnuli referenčnímu obrázku, pokud je jediný
    processed_files = [f for f in image_files if f != os.path.basename(ref_image_path)]
    if len(processed_files) >= num_examples_to_show:
        sample_files = random.sample(processed_files, num_examples_to_show)
    elif processed_files:
        sample_files = processed_files
    else:
        sample_files = []

    if sample_files:
        print("\nZobrazuji několik příkladů normalizace...")
        fig_samples, axes_samples = plt.subplots(num_examples_to_show, 3, figsize=(15, 5 * num_examples_to_show))
        if num_examples_to_show == 1: # plt.subplots vrací jiný tvar pro 1 řádek
            axes_samples = np.array([axes_samples])

        fig_samples.suptitle("Příklady normalizace", fontsize=16)

        for idx, sample_filename in enumerate(sample_files):
            original_sample_path = os.path.join(output_norm_patches_dir, "original", sample_filename)
            staintools_sample_path = os.path.join(output_norm_patches_dir, "staintools", f"st_{sample_filename}")
            torchstain_sample_path = os.path.join(output_norm_patches_dir, "torchstain", f"ts_{sample_filename}")

            if os.path.exists(original_sample_path):
                axes_samples[idx, 0].imshow(Image.open(original_sample_path))
                axes_samples[idx, 0].set_title(f"Originál: {sample_filename[:20]}...")
                axes_samples[idx, 0].axis('off')
            else:
                axes_samples[idx, 0].set_title(f"Originál nenalezen")
                axes_samples[idx, 0].axis('off')


            if os.path.exists(staintools_sample_path):
                axes_samples[idx, 1].imshow(Image.open(staintools_sample_path))
                axes_samples[idx, 1].set_title("Staintools")
                axes_samples[idx, 1].axis('off')
            else:
                axes_samples[idx, 1].set_title(f"Staintools výsledek nenalezen")
                axes_samples[idx, 1].axis('off')

            if os.path.exists(torchstain_sample_path):
                axes_samples[idx, 2].imshow(Image.open(torchstain_sample_path))
                axes_samples[idx, 2].set_title("TorchStain")
                axes_samples[idx, 2].axis('off')
            else:
                axes_samples[idx, 2].set_title(f"TorchStain výsledek nenalezen")
                axes_samples[idx, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

print("\n\nSkript dokončen. Normalizované obrázky jsou uloženy v:", output_norm_patches_dir)