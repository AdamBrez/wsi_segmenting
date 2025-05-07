import os

wsi_dir_path = r"C:\Users\USER\Desktop\wsi_dir"

def load_wsi_and_ground_truth(files_path):
    try_tumor_train = []
    try_mask_train = []


    # Projdeme všechny soubory v adresáři
    for filename in os.listdir(files_path):
        full_path = os.path.join(files_path, filename)

        # Přeskočíme podadresáře
        if not os.path.isfile(full_path):
            continue

        # Zkontrolujeme, zda se jedná o .tif soubor
        if filename.endswith(".tif"):
            # Zkontrolujeme, zda se jedná o tumor nebo mask soubor
            if filename.startswith("tumor_"):
                try_tumor_train.append(full_path)
                # print(f"Nalezen tumor: {full_path}")
            elif filename.startswith("mask_"):
                try_mask_train.append(full_path)
                # print(f"Nalezena maska: {full_path}")

    # Výpis počtu nalezených souborů
    # print(f"Počet tumor souborů: {len(try_tumor_train)}, počet mask souborů: {len(try_mask_train)}")
    train_wsi = []
    train_mask = []
    val_wsi = []
    val_mask = []
    test_wsi = []
    test_mask = []

    for i in range(len(try_tumor_train)):
        if i <= 80:
            train_wsi.append(try_tumor_train[i])
            train_mask.append(try_mask_train[i])
        elif i <= 95:
            val_wsi.append(try_tumor_train[i])
            val_mask.append(try_mask_train[i])
        else:
            test_wsi.append(try_tumor_train[i])
            test_mask.append(try_mask_train[i])
    return train_wsi, train_mask, val_wsi, val_mask, test_wsi, test_mask

train_wsi, train_mask, val_wsi, val_mask, test_wsi, test_mask = load_wsi_and_ground_truth(wsi_dir_path)
# print(type(train_wsi), len(train_wsi), type(val_wsi), len(val_wsi), type(test_wsi), len(test_wsi))
# print(type(train_mask), len(train_mask), type(val_mask), len(val_mask), type(test_mask), len(test_mask))

def load_lowres_masks(path_to_masks):
    lowres_masks_train = []
    lowres_masks_val = []
    lowres_masks_test = []
    i = 1
    for filename in os.listdir(path_to_masks):
        if i <= 81:
            full_path = os.path.join(path_to_masks, filename)
            lowres_masks_train.append(full_path)
        elif i <= 96:
            full_path = os.path.join(path_to_masks, filename)
            lowres_masks_val.append(full_path)
        else:
            full_path = os.path.join(path_to_masks, filename)
            lowres_masks_test.append(full_path)
        i += 1
    
    return lowres_masks_train, lowres_masks_val, lowres_masks_test
train, val, test = load_lowres_masks(r"C:\Users\USER\Desktop\colab_unet\masky_healthy")
# print(type(train), len(train), type(val), len(val), type(test), len(test))
# print(train[0])
# print(val[0])
# print(test[0])