import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from my_functions import basic_transform
import segmentation_models_pytorch as smp
from new_dataset2 import WSITileDatasetBalanced
from my_augmentation import MyAugmentations, AlbumentationsAug
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, TverskyLoss
import datetime
from new_load_data import load_and_split_data
from model import UNet
from monai.losses import DiceCELoss
import random

# nacitani dat
CANCER_WSI_GT_DIR = r"C:\Users\USER\Desktop\wsi_dir"
CANCER_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\masky_healthy"
CANCER_LR_GT_DIR = r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky"
HEALTHY_WSI_DIR = r"C:\Users\USER\Desktop\normal_wsi"
HEALTHY_LR_TISSUE_DIR = r"C:\Users\USER\Desktop\colab_unet\normal_lowres"

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
if __name__ == "__main__":

# Začátek trénovacího skriptu
    color_jitter_params = {
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.15,
        "hue": 0.05
    }

    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        p_color=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    albumentations_aug = AlbumentationsAug(
    p_flip=0.4,
    p_color=0.0,
    p_elastic=0.2,
    p_rotate90=0.3,
    p_shiftscalerotate=0.5,
    p_blur=0.05,
    p_noise=0.1,
    p_hestain=0.5
    )
    # albumentations_aug = AlbumentationsAug(
    # p_flip=0.4,
    # p_color=0.4,
    # p_elastic=0.0,
    # p_rotate90=0.4,
    # p_shiftscalerotate=0.0,
    # p_blur=0.0,
    # p_noise=0.0,
    # p_hestain=0.0
    # )
    
    start = time.time()
    epochs = 100
    batch = 12
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDatasetBalanced(
        wsi_paths=train_wsi_paths,
        tissue_mask_paths=train_tissue_masks,
        mask_paths=train_hr_gt_masks,
        gt_lowres_mask_paths=train_lr_gt_masks,
        tile_size=400, 
        wanted_level=2,
        healthy_wsi_sampling_prob=0.5, 
        positive_sampling_prob=0.8,    
        min_cancer_ratio_in_tile=0.05,
        augmentations=albumentations_aug,
        dataset_len=7200, 
        crop=True 
    )
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = WSITileDatasetBalanced(
            wsi_paths=val_wsi_paths,
            tissue_mask_paths=val_tissue_masks,
            mask_paths=val_hr_gt_masks,
            gt_lowres_mask_paths=val_lr_gt_masks,
            tile_size=256, 
            wanted_level=2,
            healthy_wsi_sampling_prob=0.5, 
            positive_sampling_prob=0.8,    
            min_cancer_ratio_in_tile=0.05,
            augmentations=basic_transform,
            dataset_len=5040, 
            crop=False 
        )
    
    validloader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=False, pin_memory=True)

    # test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
    #                                       mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
    #                                       tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
    #                                       min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
    #                                       dataset_len=72000)
    
    # testloader = DataLoader(test_dataset,batch_size=batch, num_workers=2, shuffle=False)
    
    # net = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    # net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    # net = UNet(n_channels=3, n_classes=1)
    net = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    # net = UNet(spatial_dims=2, in_channels=3, out_channels=1, channels=(64, 128, 256, 512, 1024),
    #            strides=(2, 2, 2, 2), num_res_units=0, act="relu", norm="batch", dropout=0.0)
    net = net.to(device)
    lr_start = 0.001
    optimizer = optim.AdamW(net.parameters(), lr=lr_start, weight_decay=1e-4) #weight_decay=1e-5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,40], gamma=0.3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=7, threshold=0.001, factor=0.2, min_lr=1e-6, cooldown=1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6) # pro CosineAnnealingLR
    # early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    loss_func = DiceLoss(mode="binary", from_logits=True, smooth=1e-6)

    # prvni faze
    for param in net.encoder.parameters():
        param.requires_grad = False
        
        optimizer = optim.AdamW([
            {'params': net.decoder.parameters(), 'lr': 1e-3},
            {'params': net.segmentation_head.parameters(), 'lr': 1e-3},
        ], weight_decay=1e-4)
    scheduler = None
    n_epochs_avg = 5 
    best_val_dice = float("-inf")
    weights_patience = 0
    min_impovement = 0.001
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(rf"C:\Users\USER\Desktop\results\{current_date}", exist_ok=True)

    smp_f1_train = []
    smp_f1_val = []

    smp_iou_train = []
    smp_iou_val = []

    smp_recall_train = []
    smp_recall_val = []
    
    smp_precision_train = []
    smp_precision_val = []

    train_loss_hist = []
    valid_loss_hist = []

    batch_load_time = []
    lr_change_epochs = []

    for epoch in range(epochs):
        start_epoch = time.time()
        print(f"Epoch {epoch}/{epochs-1}\n")
        if epoch >= 10 and epoch % 10 == 0:
            print(f"F1: train: {np.mean(smp_f1_train)}, val: {np.mean(smp_f1_val)}")
            print(f"Recall: train: {np.mean(smp_recall_train)}, val: {np.mean(smp_recall_val)}")
            print(f"Precision: train: {np.mean(smp_precision_train)}, val: {np.mean(smp_precision_val)}")
            print(f"IoU (smp): train: {np.mean(smp_iou_train)}, val: {np.mean(smp_iou_val)}")
            print(f"Loss (gemini): train: {np.mean(train_loss_hist)}, val: {np.mean(valid_loss_hist)}")
            print("-"*30)
        if epoch == 10:
            print("-"*30)
            print(f"FÁZE 2: Rozmrazování encoderu v epoše {epoch}. Přepínám na diferenciální LR.")
            
            for param in net.encoder.parameters():
                param.requires_grad = True
                
            optimizer = optim.AdamW([
                {'params': net.encoder.parameters(), 'lr': 1e-5},
                {'params': net.decoder.parameters(), 'lr': 2e-4},
                {'params': net.segmentation_head.parameters(), 'lr': 2e-4},
            ], weight_decay=1e-4)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=1e-6)
            print("Optimalizátor a scheduler byly aktualizovány pro trénink celé sítě.")
            print("-" * 30)

        epoch_train_loss = 0.0
        epoch_train_tp, epoch_train_fp, epoch_train_fn, epoch_train_tn = 0, 0, 0, 0

        start_time = time.time()
        for k,(data,lbl) in enumerate(trainloader):
            # mask_range_values = lbl.max(), lbl.min()
            # print(f"Mask range values: {mask_range_values}")
            # print(k)
            batch_time = time.time() - start_time
            batch_load_time.append(batch_time)

            data = data.to(device)
            lbl = lbl.to(device)

            net.train()
            output = net(data)
            loss = loss_func(output, lbl)
            output = torch.sigmoid(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            with torch.inference_mode():
                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5) 
                epoch_train_fp += fp.sum()
                epoch_train_fn += fn.sum()
                epoch_train_tp += tp.sum()
                epoch_train_tn += tn.sum()

            start_time = time.time()

        avg_train_loss = epoch_train_loss / len(trainloader)
        epoch_train_precision = precision(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_recall = recall(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_dice = f1_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()
        epoch_train_iou = iou_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item()

        train_loss_hist.append(avg_train_loss)
        smp_precision_train.append(epoch_train_precision)
        smp_recall_train.append(epoch_train_recall)
        smp_f1_train.append(epoch_train_dice)
        smp_iou_train.append(epoch_train_iou)

        print(f"Epoch {epoch}- Train stats- Precision: {epoch_train_precision:.4f}, Recall: {epoch_train_recall:.4f}, F1: {epoch_train_dice:.5f}, IoU (smp): {epoch_train_iou:.4f}, Loss (gemini): {avg_train_loss:.4f}")
        print("-"*30)

        # val loop
        net.eval()
        epoch_val_tp, epoch_val_fp, epoch_val_fn, epoch_val_tn = 0, 0, 0, 0
        epoch_val_loss = 0.0
        with torch.inference_mode():
            for kk,(data, lbl) in enumerate(validloader):

                data = data.to(device)
                lbl = lbl.to(device)

                output = net(data)
                loss = loss_func(output, lbl)
                output = torch.sigmoid(output)

                epoch_val_loss += loss.item()

                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5)
                epoch_val_fp += fp.sum()
                epoch_val_fn += fn.sum()
                epoch_val_tp += tp.sum()
                epoch_val_tn += tn.sum()

        avg_valid_loss = epoch_val_loss / len(validloader)
        epoch_val_precision = precision(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_recall = recall(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_dice = f1_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()
        epoch_val_iou = iou_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item()

        valid_loss_hist.append(avg_valid_loss)
        smp_precision_val.append(epoch_val_precision)
        smp_recall_val.append(epoch_val_recall)
        smp_f1_val.append(epoch_val_dice)
        smp_iou_val.append(epoch_val_iou)
        
        print(f"Epoch {epoch}- Val stats- Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}, F1: {epoch_val_dice:.5f}, IoU (smp): {epoch_val_iou:.4f}, Loss (gemini): {avg_valid_loss:.4f}")
        print("-"*30)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        print("-"*30)

        # pridani vypoctu poslednich peti val dice pro stabilnejsi early stopping
        if epoch >= 5:
            if len(smp_f1_val) > n_epochs_avg:
                avg_val_dice = np.mean(smp_f1_val[-n_epochs_avg:])
            else:
                avg_val_dice = np.mean(smp_f1_val)

            if avg_val_dice > (best_val_dice + min_impovement):
                best_val_dice = avg_val_dice
                print(f"Uložení váh modelu z epochy {epoch} s průměrem za posledních {n_epochs_avg} F1: {avg_val_dice:.5f}")
                print("-"*30)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_hist,
                    'valid_loss': valid_loss_hist,
                    "train_dice": smp_f1_train,
                    "train_iou": smp_iou_train,
                    "train_precision": smp_precision_train,
                    "train_recall": smp_recall_train,
                    "val_dice": smp_f1_val,
                    "val_iou": smp_iou_val,
                    "val_precision": smp_precision_val,
                    "val_recall": smp_recall_val,
                    "lr": current_lr,
                    "scheduler": "cosine_annealing",  # type(scheduler).__name__
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else "None"
                }
                try:
                    torch.save(checkpoint, rf"C:\Users\USER\Desktop\results\{current_date}\best_weights_{current_date}.pth")
                except Exception as e:
                    print(f"Chyba při ukládání váh v checkpointu: {e}")
                weights_patience = 0
            else:
                weights_patience += 1
        if epoch >= 5:
            # scheduler.step(avg_val_dice)  # pro ReduceLROnPlateau
            pass
        if scheduler:
            scheduler.step()  #<-- pro multi step LR a CosineAnnealingLR
        if current_lr != optimizer.param_groups[0]['lr']:
            current_lr_encoder = optimizer.param_groups[0]['lr']
            print(f"Current LR (Encoder): {current_lr_encoder:.7f}", end="")
            if len(optimizer.param_groups) > 1:
                current_lr_decoder = optimizer.param_groups[1]['lr']
                print(f" | Current LR (Decoder): {current_lr_decoder:.7f}")
            else:
                print("") # Jen pro zarovnání
        print("-" * 30)

        print(f"Epocha {epoch} trvala {time.time() - start_epoch:.2f} s")

        if weights_patience >= 20:
            print(f"Trénink byl zastaven po {epoch} epochách.")
            break


    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss_hist, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(rf'C:\Users\USER\Desktop\results\{current_date}\loss_plot{current_date}.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s")
    print("Proběhl celý skript.")

    end = time.time()
    model_save_path = rf"C:\Users\USER\Desktop\results\{current_date}\final_weights_{current_date}.pth"
    
    model_and_metadata = {
        "model": type(net).__name__,
        "encoder": "resnet34",
        "epochs": epochs,
        "lr_start": lr_start, 
        "lr_end": optimizer.param_groups[0]['lr'],
        "lr_change_epochs": lr_change_epochs,
        "scheduler": type(scheduler).__name__,
        "optimizer": type(optimizer).__name__,
        "batch_size": batch,
        "train_dataset_len": vars(train_dataset)["dataset_len"],
        "train_dice": smp_f1_train,
        "train_iou": smp_iou_train,
        "train_precision": smp_precision_train,
        "train_recall": smp_recall_train,
        "val_dice": smp_f1_val,
        "val_iou": smp_iou_val,
        "val_precision": smp_precision_val,
        "val_recall": smp_recall_val,
        # "test_dice": epoch_test_dice,
        # "test_iou": epoch_test_iou,
        # "test_precision": epoch_test_precision,
        # "test_recall": epoch_test_recall,
        "loss_function": type(loss_func).__name__,
        "augmentation": train_dataset.augmentations,
        # "wanted_level": vars(test_dataset)["wanted_level"],
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_hist": train_loss_hist,
        "valid_loss_hist": valid_loss_hist,
        "runtime": end - start,
        "info": (f"Trénování unet++ s pretrained vahami,"
                 "scheduler byl použit (cosineAnnealingLr). "
                 "Augmentace pokorcile. WD 1e-4 a trenénink na celem camelyon16 bez test sady"),
        
    }
    # Uložení váh modelu
    torch.save(model_and_metadata, model_save_path)
    print(f"Model byl uložen do {model_save_path}")

    print(f"Trénování trvalo {end - start:.2f} s")
