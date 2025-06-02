import os
import random
from sklearn.model_selection import train_test_split

def load_and_split_data(
    cancer_wsi_gt_main_dir,
    cancer_lr_tissue_mask_dir,
    cancer_lr_gt_mask_dir,
    healthy_wsi_dir,
    healthy_lr_tissue_mask_dir,
    val_size=0.2, # Procentuální velikost validační sady
    random_state=42 # Pro reprodukovatelnost rozdělení
):
    """
    Načte cesty k WSI souborům a jejich maskám, rozdělí je na trénovací a validační sady
    se zohledněním center původu.

    Args:
        cancer_wsi_gt_main_dir (str): Adresář s WSI nádorů (tumor_xxx.tif) a HR masek (mask_xxx.tif).
        cancer_lr_tissue_mask_dir (str): Adresář s LR maskami tkáně pro nádorové WSI (mask_{id}.npy).
        cancer_lr_gt_mask_dir (str): Adresář s LR GT maskami pro nádorové WSI (mask_{id}_cancer.npy).
        healthy_wsi_dir (str): Adresář se zdravými WSI (normal_xxx.tif).
        healthy_lr_tissue_mask_dir (str): Adresář s LR maskami tkáně pro zdravé WSI (tissue_mask_xxx.npy).
        val_size (float): Podíl dat, která se použijí pro validaci (např. 0.2 pro 20%).
        random_state (int): Seed pro náhodné generátory pro zajištění reprodukovatelnosti.

    Returns:
        tuple: Dva tuply (train_data, val_data). Každý tuple obsahuje:
               (wsi_paths, tissue_mask_paths, hr_gt_mask_paths, lr_gt_mask_paths)
    """
    random.seed(random_state)

    cancer_wsi_paths_c1, tissue_masks_cancer_c1, hr_gt_masks_cancer_c1, lr_gt_masks_cancer_c1 = [], [], [], []
    cancer_wsi_paths_c2, tissue_masks_cancer_c2, hr_gt_masks_cancer_c2, lr_gt_masks_cancer_c2 = [], [], [], []
    healthy_wsi_paths_c1, tissue_masks_healthy_c1, hr_gt_masks_healthy_c1, lr_gt_masks_healthy_c1 = [], [], [], []
    healthy_wsi_paths_c2, tissue_masks_healthy_c2, hr_gt_masks_healthy_c2, lr_gt_masks_healthy_c2 = [], [], [], []

    # Načtení dat nádorů (Camelyon16)
    for i in range(1, 112): # tumor_001 až tumor_111
        slide_id = f"{i:03d}"
        wsi_path = os.path.join(cancer_wsi_gt_main_dir, f"tumor_{slide_id}.tif")
        hr_mask_path = os.path.join(cancer_wsi_gt_main_dir, f"mask_{slide_id}.tif")
        lr_tissue_mask_path = os.path.join(cancer_lr_tissue_mask_dir, f"mask_{slide_id}.npy")
        lr_gt_mask_path = os.path.join(cancer_lr_gt_mask_dir, f"mask_{slide_id}_cancer.npy")

        if not (os.path.exists(wsi_path) and \
                os.path.exists(hr_mask_path) and \
                os.path.exists(lr_tissue_mask_path) and \
                os.path.exists(lr_gt_mask_path)):
            print(f"Varování: Chybí některé soubory pro nádor ID {slide_id}. Přeskakuji.")
            continue

        if 1 <= i <= 70: # Centrum 1
            cancer_wsi_paths_c1.append(wsi_path)
            tissue_masks_cancer_c1.append(lr_tissue_mask_path)
            hr_gt_masks_cancer_c1.append(hr_mask_path)
            lr_gt_masks_cancer_c1.append(lr_gt_mask_path)
        else: # Centrum 2 (71-111)
            cancer_wsi_paths_c2.append(wsi_path)
            tissue_masks_cancer_c2.append(lr_tissue_mask_path)
            hr_gt_masks_cancer_c2.append(hr_mask_path)
            lr_gt_masks_cancer_c2.append(lr_gt_mask_path)

    # Načtení zdravých dat (Camelyon16)
    for i in range(1, 161): # normal_001 až normal_160
        if i == 86: # normal_086 neexistuje
            continue
        slide_id = f"{i:03d}"
        # Název souboru pro zdravé WSI je "normal_xxx.tif"
        wsi_path = os.path.join(healthy_wsi_dir, f"normal_{slide_id}.tif")
        # Název souboru pro masky tkáně zdravých WSI je "tissue_mask_xxx.npy" dle vašeho __main__
        lr_tissue_mask_path = os.path.join(healthy_lr_tissue_mask_dir, f"tissue_mask_{slide_id}.npy")


        if not (os.path.exists(wsi_path) and os.path.exists(lr_tissue_mask_path)):
            print(f"Varování: Chybí některé soubory pro zdravý WSI ID {slide_id}. Přeskakuji.")
            print(f"  WSI: {wsi_path} (existuje: {os.path.exists(wsi_path)})")
            print(f"  Maska: {lr_tissue_mask_path} (existuje: {os.path.exists(lr_tissue_mask_path)})")
            continue

        if 1 <= i <= 100: # Centrum 1
            healthy_wsi_paths_c1.append(wsi_path)
            tissue_masks_healthy_c1.append(lr_tissue_mask_path)
            hr_gt_masks_healthy_c1.append(None) # Zdravé nemají HR GT masku nádoru
            lr_gt_masks_healthy_c1.append(None) # Zdravé nemají LR GT masku nádoru
        else: # Centrum 2 (101-160)
            healthy_wsi_paths_c2.append(wsi_path)
            tissue_masks_healthy_c2.append(lr_tissue_mask_path)
            hr_gt_masks_healthy_c2.append(None)
            lr_gt_masks_healthy_c2.append(None)

    # Rozdělení dat z Centra 1
    c1_cancer_indices = list(range(len(cancer_wsi_paths_c1)))
    c1_healthy_indices = list(range(len(healthy_wsi_paths_c1)))

    if c1_cancer_indices:
        c1_cancer_train_idx, c1_cancer_val_idx = train_test_split(c1_cancer_indices, test_size=val_size, random_state=random_state)
    else:
        c1_cancer_train_idx, c1_cancer_val_idx = [], []

    if c1_healthy_indices:
        c1_healthy_train_idx, c1_healthy_val_idx = train_test_split(c1_healthy_indices, test_size=val_size, random_state=random_state)
    else:
        c1_healthy_train_idx, c1_healthy_val_idx = [], []

    # Rozdělení dat z Centra 2
    c2_cancer_indices = list(range(len(cancer_wsi_paths_c2)))
    c2_healthy_indices = list(range(len(healthy_wsi_paths_c2)))

    if c2_cancer_indices:
        c2_cancer_train_idx, c2_cancer_val_idx = train_test_split(c2_cancer_indices, test_size=val_size, random_state=random_state)
    else:
        c2_cancer_train_idx, c2_cancer_val_idx = [], []

    if c2_healthy_indices:
        c2_healthy_train_idx, c2_healthy_val_idx = train_test_split(c2_healthy_indices, test_size=val_size, random_state=random_state)
    else:
        c2_healthy_train_idx, c2_healthy_val_idx = [], []


    # Sestavení trénovací sady
    train_wsi_paths = \
        [cancer_wsi_paths_c1[i] for i in c1_cancer_train_idx] + \
        [healthy_wsi_paths_c1[i] for i in c1_healthy_train_idx] + \
        [cancer_wsi_paths_c2[i] for i in c2_cancer_train_idx] + \
        [healthy_wsi_paths_c2[i] for i in c2_healthy_train_idx]

    train_tissue_masks = \
        [tissue_masks_cancer_c1[i] for i in c1_cancer_train_idx] + \
        [tissue_masks_healthy_c1[i] for i in c1_healthy_train_idx] + \
        [tissue_masks_cancer_c2[i] for i in c2_cancer_train_idx] + \
        [tissue_masks_healthy_c2[i] for i in c2_healthy_train_idx]

    train_hr_gt_masks = \
        [hr_gt_masks_cancer_c1[i] for i in c1_cancer_train_idx] + \
        [hr_gt_masks_healthy_c1[i] for i in c1_healthy_train_idx] + \
        [hr_gt_masks_cancer_c2[i] for i in c2_cancer_train_idx] + \
        [hr_gt_masks_healthy_c2[i] for i in c2_healthy_train_idx]

    train_lr_gt_masks = \
        [lr_gt_masks_cancer_c1[i] for i in c1_cancer_train_idx] + \
        [lr_gt_masks_healthy_c1[i] for i in c1_healthy_train_idx] + \
        [lr_gt_masks_cancer_c2[i] for i in c2_cancer_train_idx] + \
        [lr_gt_masks_healthy_c2[i] for i in c2_healthy_train_idx]

    # Sestavení validační sady
    val_wsi_paths = \
        [cancer_wsi_paths_c1[i] for i in c1_cancer_val_idx] + \
        [healthy_wsi_paths_c1[i] for i in c1_healthy_val_idx] + \
        [cancer_wsi_paths_c2[i] for i in c2_cancer_val_idx] + \
        [healthy_wsi_paths_c2[i] for i in c2_healthy_val_idx]

    val_tissue_masks = \
        [tissue_masks_cancer_c1[i] for i in c1_cancer_val_idx] + \
        [tissue_masks_healthy_c1[i] for i in c1_healthy_val_idx] + \
        [tissue_masks_cancer_c2[i] for i in c2_cancer_val_idx] + \
        [tissue_masks_healthy_c2[i] for i in c2_healthy_val_idx]

    val_hr_gt_masks = \
        [hr_gt_masks_cancer_c1[i] for i in c1_cancer_val_idx] + \
        [hr_gt_masks_healthy_c1[i] for i in c1_healthy_val_idx] + \
        [hr_gt_masks_cancer_c2[i] for i in c2_cancer_val_idx] + \
        [hr_gt_masks_healthy_c2[i] for i in c2_healthy_val_idx]

    val_lr_gt_masks = \
        [lr_gt_masks_cancer_c1[i] for i in c1_cancer_val_idx] + \
        [lr_gt_masks_healthy_c1[i] for i in c1_healthy_val_idx] + \
        [lr_gt_masks_cancer_c2[i] for i in c2_cancer_val_idx] + \
        [lr_gt_masks_healthy_c2[i] for i in c2_healthy_val_idx]

    # Promíchání finálních sad (volitelné, ale doporučené)
    train_combined = list(zip(train_wsi_paths, train_tissue_masks, train_hr_gt_masks, train_lr_gt_masks))
    random.shuffle(train_combined)
    train_wsi_paths, train_tissue_masks, train_hr_gt_masks, train_lr_gt_masks = zip(*train_combined) if train_combined else ([], [], [], [])

    val_combined = list(zip(val_wsi_paths, val_tissue_masks, val_hr_gt_masks, val_lr_gt_masks))
    random.shuffle(val_combined)
    val_wsi_paths, val_tissue_masks, val_hr_gt_masks, val_lr_gt_masks = zip(*val_combined) if val_combined else ([], [], [], [])

    train_data = (list(train_wsi_paths), list(train_tissue_masks), list(train_hr_gt_masks), list(train_lr_gt_masks))
    val_data = (list(val_wsi_paths), list(val_tissue_masks), list(val_hr_gt_masks), list(val_lr_gt_masks))
    
    # print(f"Data načtena a rozdělena:")
    # print(f"  Počet trénovacích WSI: {len(train_data[0])}")
    # print(f"    Nádorové: {sum(1 for m in train_data[2] if m is not None)}")
    # print(f"    Zdravé: {sum(1 for m in train_data[2] if m is None)}")
    # print(f"  Počet validačních WSI: {len(val_data[0])}")
    # print(f"    Nádorové: {sum(1 for m in val_data[2] if m is not None)}")
    # print(f"    Zdravé: {sum(1 for m in val_data[2] if m is None)}")

    return train_data, val_data

# --- Příklad použití ---
if __name__ == "__main__":
    # Definice cest k adresářům (upravte podle vaší skutečné struktury)
    CANCER_WSI_GT_DIR = r"C:\Users\USER\Desktop\wsi_dir"
    CANCER_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\masky_healthy"
    CANCER_LR_GT_DIR = r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky"
    HEALTHY_WSI_DIR = r"C:\Users\USER\Desktop\normal_wsi"
    HEALTHY_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\normal_lowres"

    # Testovací složka pro WSI, která se nepoužije pro train/val
    # TEST_WSI_DIR = r"C:\cesta\k\test\wsi"
    # TEST_LR_MASKS_DIR = r"C:\cesta\k\test\lr_masks"

    print("Načítání a rozdělování dat pro trénink a validaci...")
    train_data, val_data = load_and_split_data(
        cancer_wsi_gt_main_dir=CANCER_WSI_GT_DIR,
        cancer_lr_tissue_mask_dir=CANCER_LR_TISSUE_DIR,
        cancer_lr_gt_mask_dir=CANCER_LR_GT_DIR,
        healthy_wsi_dir=HEALTHY_WSI_DIR,
        healthy_lr_tissue_mask_dir=HEALTHY_LR_TISSUE_DIR,
        val_size=0.2, # Např. 20% pro validaci
        random_state=42
    )

    train_wsi_paths, train_tissue_masks, train_hr_gt_masks, train_lr_gt_masks = train_data
    val_wsi_paths, val_tissue_masks, val_hr_gt_masks, val_lr_gt_masks = val_data

    # Zde můžete data předat do vašeho WSITileDatasetBalanced
    # Například:
    # train_dataset = WSITileDatasetBalanced(
    #     wsi_paths=train_wsi_paths,
    #     tissue_mask_paths=train_tissue_masks,
    #     mask_paths=train_hr_gt_masks, # High-res GT pro nádory, None pro zdravé
    #     gt_lowres_mask_paths=train_lr_gt_masks, # Low-res GT pro nádory, None pro zdravé
    #     tile_size=256, # Nebo vaše preferovaná velikost
    #     wanted_level=2,
    #     # ... další parametry
    # )
    # val_dataset = WSITileDatasetBalanced(...)

    print("\nPříklad prvních 5 cest z trénovací sady:")
    for i in range(min(5, len(train_wsi_paths))):
        print(f"  WSI: {train_wsi_paths[i]}")
        print(f"    Tissue Mask: {train_tissue_masks[i]}")
        print(f"    HR GT Mask: {train_hr_gt_masks[i]}")
        print(f"    LR GT Mask: {train_lr_gt_masks[i]}")
        print("-" * 20)

    print("\nPříklad prvních 5 cest z validační sady:")
    for i in range(min(5, len(val_wsi_paths))):
        print(f"  WSI: {val_wsi_paths[i]}")
        print(f"    Tissue Mask: {val_tissue_masks[i]}")
        print(f"    HR GT Mask: {val_hr_gt_masks[i]}")
        print(f"    LR GT Mask: {val_lr_gt_masks[i]}")
        print("-" * 20)
    # Můžete přidat i logiku pro načtení testovacích dat, pokud je potřeba.
    # To by byla samostatná sada, která se nepoužívá během tréninku ani validace.