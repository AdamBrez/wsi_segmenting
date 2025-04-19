# --- START OF FILE train_smp_with_metrics.py ---

import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# Odebrány importy vlastních metrik, nahrazeny smp
# from my_functions import dice_coefficient, calculate_iou, basic_transform, dice_bce_loss
from my_functions import basic_transform, dice_bce_loss # Ponecháme loss a transformaci
import segmentation_models_pytorch as smp
# <<< Import smp metrik >>>
from segmentation_models_pytorch.metrics import precision, recall, f1_score, iou_score, get_stats
from new_dataset import WSITileDatasetBalanced
from my_augmentation import MyAugmentations # Předpokládáme verzi MyAugmentationsProbColor
from segmentation_models_pytorch.losses import TverskyLoss


# trénovací data
wsi_paths_train = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_001.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_002.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_003.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_004.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_005.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_006.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_007.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_008.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_009.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_010.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_011.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_012.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_013.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_014.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_015.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_016.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_017.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_018.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_019.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_020.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_021.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_022.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_023.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_024.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_025.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_026.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_027.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_028.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_029.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_030.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_031.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_032.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_033.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_034.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_036.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_037.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_038.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_039.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_040.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_041.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_042.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_043.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_044.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_045.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_046.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_047.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_048.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_049.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_050.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_089.tif",
]
tissue_mask_paths_train = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\masky_healthy\\mask_{p.split('_')[-1].split('.')[0]}.npy" for p in wsi_paths_train ]
mask_paths_train = [ f"C:\\Users\\USER\\Desktop\\wsi_dir\\mask_{p.split('_')[-1].split('.')[0]}.tif" for p in wsi_paths_train ]
gt_lowres_mask_paths_train = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\gt_lowres_masky\\mask_{p.split('_')[-1].split('.')[0]}_cancer.npy" for p in wsi_paths_train ]

# validační data
wsi_paths_val = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_051.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_052.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_053.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_054.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_055.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_056.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_057.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_058.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_059.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_060.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_061.tif",
]
tissue_mask_paths_val = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\masky_healthy\\mask_{p.split('_')[-1].split('.')[0]}.npy" for p in wsi_paths_val ]
mask_paths_val = [ f"C:\\Users\\USER\\Desktop\\wsi_dir\\mask_{p.split('_')[-1].split('.')[0]}.tif" for p in wsi_paths_val ]
gt_lowres_mask_paths_val = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\gt_lowres_masky\\mask_{p.split('_')[-1].split('.')[0]}_cancer.npy" for p in wsi_paths_val ]

# testovací data
wsi_paths_test = [
    r"C:\Users\USER\Desktop\wsi_dir\tumor_062.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_063.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_064.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_065.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_066.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_067.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_068.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_069.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_070.tif", r"C:\Users\USER\Desktop\wsi_dir\tumor_084.tif",
    r"C:\Users\USER\Desktop\wsi_dir\tumor_091.tif",
]
tissue_mask_paths_test = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\masky_healthy\\mask_{p.split('_')[-1].split('.')[0]}.npy" for p in wsi_paths_test ]
mask_paths_test = [ f"C:\\Users\\USER\\Desktop\\wsi_dir\\mask_{p.split('_')[-1].split('.')[0]}.tif" for p in wsi_paths_test ]
gt_lowres_mask_paths_test = [ f"C:\\Users\\USER\\Desktop\\colab_unet\\gt_lowres_masky\\mask_{p.split('_')[-1].split('.')[0]}_cancer.npy" for p in wsi_paths_test ]


if __name__ == "__main__":

    # --- Konfigurace augmentací ---
    # Použijeme MyAugmentationsProbColor z předchozího příkladu
    color_jitter_params = {
        'brightness': 0.25, 'contrast': 0.25,
        'saturation': 0.20, 'hue': 0.05
    }
    # Zkontroluj název třídy, pokud jsi ji uložil jinak
    augmentations = MyAugmentations( # Nebo MyAugmentationsProbColor
        p_flip=0.5,
        p_color=0.8, # Pokud používáš MyAugmentationsProbColor
        color_jitter_params=color_jitter_params,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    # --- Nastavení tréninku ---
    start = time.time()
    epochs = 81
    batch = 32
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Používám zařízení: {device}")

    # --- Datasety a DataLoadery ---
    train_dataset = WSITileDatasetBalanced(
        wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
        mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
        tile_size=256, wanted_level=2, positive_sampling_prob=0.6,
        min_cancer_ratio_in_tile=0.05, augmentations=augmentations # Použití definovaných augmentací
    )
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = WSITileDatasetBalanced(
        wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
        mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
        tile_size=256, wanted_level=2, positive_sampling_prob=0.6, # Můžeš zvážit positive_sampling_prob=0.5 pro val
        min_cancer_ratio_in_tile=0.05, augmentations=basic_transform # Bez augmentací pro validaci
    )
    validloader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=False, pin_memory=True)

    test_dataset = WSITileDatasetBalanced(
        wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
        mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
        tile_size=256, wanted_level=2, positive_sampling_prob=0.6, # Můžeš zvážit positive_sampling_prob=0.5 pro test
        min_cancer_ratio_in_tile=0.05, augmentations=basic_transform # Bez augmentací pro test
    )
    testloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False) # batch_size=1 pro test je běžný

    # --- Model, Optimizer, Scheduler ---
    net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 70], gamma=0.1)

    tversky_loss = TverskyLoss(mode='binary', alpha=0.7, beta=0.3, from_logits=True) # Pro příklad

    # --- Listy pro ukládání metrik ---
    train_loss_hist = []
    valid_loss_hist = []
    train_iou_hist = []
    valid_iou_hist = []
    train_dice_hist = []
    valid_dice_hist = []
    # <<< Nové listy pro precision a recall >>>
    train_precision_hist = []
    valid_precision_hist = []
    train_recall_hist = []
    valid_recall_hist = []

    print(f"Start tréninku na {epochs} epoch...")
    # --- Trénovací smyčka ---
    for epoch in range(epochs):
        print("-" * 20)
        print(f"Epoch {epoch}/{epochs-1}")

        # --- Trénovací fáze ---
        net.train()
        epoch_train_loss = 0.0
        # Akumulátory statistik pro trénink
        epoch_train_tp, epoch_train_fp, epoch_train_fn, epoch_train_tn = 0, 0, 0, 0

        batch_start_time = time.time() # Časovač pro načítání batchů - resetovat zde
        batch_load_times_epoch = []

        for i, (data, lbl) in enumerate(trainloader):
            load_time = time.time() - batch_start_time
            batch_load_times_epoch.append(load_time)

            data = data.to(device)
            lbl = lbl.to(device) # Očekáváme float tensor [0, 1] z datasetu

            # Forward pass
            output_logits = net(data) # Získat logits [B, 1, H, W]
            loss = tversky_loss(output_logits, lbl) # Loss pro logits
            output_probs = torch.sigmoid(output_logits) # Pravděpodobnosti [B, 1, H, W]

            # Výpočet loss (předpokládá logits jako vstup)
            # loss = dice_bce_loss(output_probs, lbl) # Upravit dice_bce_loss, pokud očekává sigmoid výstup

            # Backward pass a optimalizace
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # Výpočet statistik pro metriky
            with torch.inference_mode(): # Nepotřebujeme gradienty pro metriky
                # Převedeme lbl na long pro smp metriky
                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode='binary', threshold=0.5)
                epoch_train_tp += tp.sum()
                epoch_train_fp += fp.sum()
                epoch_train_fn += fn.sum()
                epoch_train_tn += tn.sum()

            batch_start_time = time.time() # Resetovat časovač pro další batch

        # Výpočet průměrné loss a metrik za trénovací epochu
        avg_train_loss = epoch_train_loss / len(trainloader)
        epoch_train_precision = precision(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_recall = recall(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_dice = f1_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_iou = iou_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()

        train_loss_hist.append(avg_train_loss)
        train_iou_hist.append(epoch_train_iou)
        train_dice_hist.append(epoch_train_dice)
        train_precision_hist.append(epoch_train_precision)
        train_recall_hist.append(epoch_train_recall)

        print(f"Train - Loss: {avg_train_loss:.4f}, Dice: {epoch_train_dice:.4f}, IoU: {epoch_train_iou:.4f}, Precision: {epoch_train_precision:.4f}, Recall: {epoch_train_recall:.4f}")
        if batch_load_times_epoch: print(f"Avg Batch Load Time: {np.mean(batch_load_times_epoch):.4f} s")

        # --- Validační fáze ---
        net.eval()
        epoch_val_loss = 0.0
        # Akumulátory statistik pro validaci
        epoch_val_tp, epoch_val_fp, epoch_val_fn, epoch_val_tn = 0, 0, 0, 0

        with torch.inference_mode():
            for data, lbl in validloader:
                data = data.to(device)
                lbl = lbl.to(device) # Očekáváme float tensor [0, 1]

                output_logits = net(data) # Logits [B, 1, H, W]
                loss = tversky_loss(output_logits, lbl) # Loss pro logits
                output_probs = torch.sigmoid(output_logits) # Pravděpodobnosti [B, 1, H, W]

                # Výpočet loss
                # loss = dice_bce_loss(output_probs, lbl) # Upravit, pokud očekává sigmoid
                epoch_val_loss += loss.item()

                # Výpočet statistik pro metriky
                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode='binary', threshold=0.5)
                epoch_val_tp += tp.sum()
                epoch_val_fp += fp.sum()
                epoch_val_fn += fn.sum()
                epoch_val_tn += tn.sum()

        # Výpočet průměrné loss a metrik za validační epochu
        avg_valid_loss = epoch_val_loss / len(validloader)
        # Přidáme malou hodnotu (epsilon) do jmenovatele, aby se předešlo dělení nulou, pokud nejsou žádné predikce
        epsilon = 1e-6
        epoch_val_precision = precision(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_recall = recall(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_dice = f1_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_iou = iou_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()

        valid_loss_hist.append(avg_valid_loss)
        valid_iou_hist.append(epoch_val_iou)
        valid_dice_hist.append(epoch_val_dice)
        valid_precision_hist.append(epoch_val_precision)
        valid_recall_hist.append(epoch_val_recall)

        print(f"Valid - Loss: {avg_valid_loss:.4f}, Dice: {epoch_val_dice:.4f}, IoU: {epoch_val_iou:.4f}, Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}")

        # Krok scheduleru
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()[0]}")

        # Výpis průměrných hodnot za celou dobu (každých 10 epoch) - Nyní používáme *_hist listy
        if epoch >= 9 and (epoch + 1) % 10 == 0: # Upraveno pro správné epochy (10, 20, ...)
            print("-" * 10 + f" Averages after Epoch {epoch} " + "-" * 10)
            print(f"Avg Train - Loss: {np.mean(train_loss_hist):.4f}, Dice: {np.mean(train_dice_hist):.4f}, IoU: {np.mean(train_iou_hist):.4f}, Precision: {np.mean(train_precision_hist):.4f}, Recall: {np.mean(train_recall_hist):.4f}")
            print(f"Avg Valid - Loss: {np.mean(valid_loss_hist):.4f}, Dice: {np.mean(valid_dice_hist):.4f}, IoU: {np.mean(valid_iou_hist):.4f}, Precision: {np.mean(valid_precision_hist):.4f}, Recall: {np.mean(valid_recall_hist):.4f}")
            print("-" * 40)


    # --- Konec tréninku ---
    print("\nTrénink dokončen.")

    # --- Vykreslení loss křivky ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss_hist, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    # Uložení grafu
    loss_plot_path = r'C:\Users\USER\Desktop\loss_curve_unet_smp_e{}_len{}.png'.format(epochs, len(train_dataset))
    plt.savefig(loss_plot_path)
    print(f"Graf loss uložena do: {loss_plot_path}")
    #plt.show(block=False) # Zobrazí neblokující okno
    #plt.pause(5)          # Pauza pro zobrazení
    plt.close()            # Zavře okno

    # --- Testovací fáze ---
    print("\nZačátek testování...")
    net.eval()
    # Akumulátory statistik pro test
    test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
    THRESHOLD = 0.5 # Prah pro binární klasifikaci
    with torch.no_grad():
        for kk, (data, lbl) in enumerate(tqdm(testloader, desc="Testing")):
            data = data.to(device)
            lbl = lbl.to(device) # Float tensor [0, 1]

            # Predikce
            output_logits = net(data)
            output_probs = torch.sigmoid(output_logits) # Pravděpodobnosti pro metriky

            # Akumulace statistik
            lbl_long = lbl.long()
            tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode='binary', threshold=0.5)
            test_tp += tp.sum()
            test_fp += fp.sum()
            test_fn += fn.sum()
            test_tn += tn.sum()

            # Vizualizace prvních několika výsledků
            if kk < 3:
                plt.figure(figsize=(15, 5)) # Upravena velikost pro lepší zobrazení
                plt.subplot(1, 3, 1)
                plt.title("Input (Sample)")
                # Pro zobrazení normalizovaného obrázku je třeba ho denormalizovat nebo zobrazit jeden kanál
                # Zobrazení jednoho kanálu (např. červeného)
                # plt.imshow(data[0, 0, :, :].cpu().numpy(), cmap='gray')
                # Nebo pokus o zobrazení jako RGB (může být zkreslené kvůli normalizaci)
                img_display = data[0].cpu().permute(1, 2, 0).numpy()
                # Velmi jednoduchá denormalizace pro zobrazení (nemusí být perfektní)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_display = std * img_display + mean
                img_display = np.clip(img_display, 0, 1)
                plt.imshow(img_display)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title(f"Prediction (Threshold={THRESHOLD})")
                plt.imshow(output_probs[0, 0, :, :].cpu().numpy() >= THRESHOLD, cmap="gray") # Prahovaná predikce
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title("Ground Truth")
                plt.imshow(lbl[0, 0, :, :].cpu().numpy(), cmap="gray") # Ground truth
                plt.axis('off')

                plot_save_path = r'C:\Users\USER\Desktop\test_sample_{}_e{}_len{}.png'.format(kk, epochs, len(train_dataset))
                plt.savefig(plot_save_path)
                print(f"Testovací obrázek {kk} uložen do: {plot_save_path}")
                #plt.show(block=False)
                #plt.pause(1) # Krátká pauza
                plt.close()

    # Výpočet finálních metrik na testovací sadě
    epsilon = 1e-6
    test_precision = precision(tp=test_tp, fp=test_fp, fn=test_fn, tn=test_tn).item()
    test_recall = recall(tp=test_tp, fp=test_fp, fn=test_fn, tn=test_tn).item()
    test_dice = f1_score(tp=test_tp, fp=test_fp, fn=test_fn, tn=test_tn).item()
    test_iou = iou_score(tp=test_tp, fp=test_fp, fn=test_fn, tn=test_tn).item()

    print("-" * 20 + " Test Results " + "-" * 20)
    print(f"Average Test Dice: {test_dice:.4f}")
    print(f"Average Test IoU: {test_iou:.4f}")
    print(f"Average Test Precision: {test_precision:.4f}")
    print(f"Average Test Recall: {test_recall:.4f}")
    print("-" * 54)

    # Uložení modelu
    model_save_path = r"C:\Users\USER\Desktop\weights\unet_smp_e{}_len{}.pth".format(epochs, len(train_dataset))
    try:
        torch.save(net.state_dict(), model_save_path)
        print(f"Model byl uložen do {model_save_path}")
    except Exception as save_err:
        print(f"Chyba při ukládání modelu: {save_err}")

    end = time.time()
    total_time = end - start
    print(f"\nCelkové trvání skriptu: {total_time:.2f} s ({total_time/60:.2f} min)")
    print("Skript dokončen.")

# --- END OF FILE train_smp_with_metrics.py ---