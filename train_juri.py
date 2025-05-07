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
from new_dataset import WSITileDatasetBalanced
from my_augmentation import MyAugmentations, AlbumentationsAug
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
from segmentation_models_pytorch.losses import DiceLoss
import datetime
from load_data import load_wsi_and_ground_truth, load_lowres_masks


# tkan a masky
wsi_paths_train, mask_paths_train, wsi_paths_val, mask_paths_val, wsi_paths_test, mask_paths_test = load_wsi_and_ground_truth(r"C:\Users\USER\Desktop\wsi_dir")

tissue_mask_paths_train, tissue_mask_paths_val, tissue_mask_paths_test = load_lowres_masks(r"C:\Users\USER\Desktop\colab_unet\masky_healthy")

gt_lowres_mask_paths_train, gt_lowres_mask_paths_val, gt_lowres_mask_paths_test = load_lowres_masks(r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky")
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

    alubmentations_aug = AlbumentationsAug()
    
    start = time.time()
    epochs = 80
    batch = 32
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
                                           mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
                                           tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                           min_cancer_ratio_in_tile=0.05, augmentations=augmentations,
                                           dataset_len=11200)
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
                                         mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
                                         tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                         min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                         dataset_len=5600)
    
    validloader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=False, pin_memory=True)

    test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
                                          mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
                                          tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                          min_cancer_ratio_in_tile=0.05, augmentations=basic_transform,
                                          dataset_len=22400)
    
    testloader = DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=False)

    # net = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = net.to(device)
    lr_start = 0.01
    optimizer = optim.Adam(net.parameters(), lr=lr_start) #weight_decay=1e-5
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, threshold=0.001)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainloader), epochs=epochs, max_lr=0.001, pct_start=0.3, div_factor=25, final_div_factor=1000)
    # early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.001)
    loss_func = DiceLoss(mode="binary", from_logits=True, smooth=1e-6)

    best_val_dice = float("-inf")
    weights_patience = 7
    min_impovement = 0.001
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
        print(f"Epoch {epoch}/{epochs-1}\n")
        if epoch >= 10 and epoch % 10 == 0:
            print(f"F1: train: {np.mean(smp_f1_train)}, val: {np.mean(smp_f1_val)}")
            print(f"Recall: train: {np.mean(smp_recall_train)}, val: {np.mean(smp_recall_val)}")
            print(f"Precision: train: {np.mean(smp_precision_train)}, val: {np.mean(smp_precision_val)}")
            print(f"IoU (smp): train: {np.mean(smp_iou_train)}, val: {np.mean(smp_iou_val)}")
            print(f"Loss (gemini): train: {np.mean(train_loss_hist)}, val: {np.mean(valid_loss_hist)}")
            print("-"*30)

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

        print(f"Epoch {epoch}- Train stats- Precision: {epoch_train_precision:.4f}, Recall: {epoch_train_recall:.4f}, F1: {epoch_train_dice:.4f}, IoU (smp): {epoch_train_iou:.4f}, Loss (gemini): {avg_train_loss:.4f}")
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
        
        print(f"Epoch {epoch}- Val stats- Precision: {epoch_val_precision:.4f}, Recall: {epoch_val_recall:.4f}, F1: {epoch_val_dice:.4f}, IoU (smp): {epoch_val_iou:.4f}, Loss (gemini): {avg_valid_loss:.4f}")
        print("-"*30)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        print("-"*30)
        if epoch > 10 and epoch < epochs - 10:
            if epoch_val_dice > (best_val_dice + min_impovement):
                best_val_dice = epoch_val_dice
                print(f"Uložení váh modelu z epochy {epoch} s F1: {epoch_val_dice:.4f}")
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
                    "scheduler": type(scheduler).__name__
                }
                try:
                    torch.save(checkpoint, rf"C:\Users\USER\Desktop\weights\best_weights{current_date}.pth")
                except Exception as e:
                    print(f"Chyba při ukládání váh v checkpointu: {e}")
                weights_patience = 0
            else:
                weights_patience += 1


        scheduler.step(epoch_val_dice)  # pro ReduceLROnPlateau
        # scheduler.step()  #<-- pro multi step LR
        if current_lr > optimizer.param_groups[0]['lr']:
            print(f"Změna LR: {optimizer.param_groups[0]['lr']} -> {current_lr:.6f} | Epoch: {epoch}")
            lr_change_epochs.append(epoch)



    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss_hist, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(rf'C:\Users\USER\Desktop\loss_plot{current_date}.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    test_dice_scores = []
    test_iou_scores = []
    test_recall_scores = []
    test_precision_scores = []
    epoch_test_tp, epoch_test_fp, epoch_test_fn, epoch_test_tn = 0, 0, 0, 0
    # Test loop
    with torch.inference_mode():
        for kk, (data, lbl) in enumerate(testloader):
            data = data.to(device)
            lbl = lbl.to(device)

            # Předpověď sítě
            net.eval()
            output = net(data)
            output = torch.sigmoid(output)

            lbl_long = lbl.long()
            tp, fp, fn, tn = get_stats(output, lbl_long, mode="binary", threshold=0.5)
            epoch_test_fp += fp.sum()
            epoch_test_fn += fn.sum()
            epoch_test_tp += tp.sum()
            epoch_test_tn += tn.sum()

    epoch_test_precision = precision(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_recall = recall(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_dice = f1_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_iou = iou_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()

    test_dice_scores.append(epoch_test_dice)
    test_iou_scores.append(epoch_test_iou)
    test_recall_scores.append(epoch_test_recall)
    test_precision_scores.append(epoch_test_precision)

    print(f"Average Dice coefficient on test data: {np.mean(test_dice_scores):.4f}")
    print(f"Average IoU on test data: {np.mean(test_iou_scores):.4f}")
    print(f"Average recall on test data: {np.mean(test_recall_scores):.4f}")
    print(f"Average precision on test data: {np.mean(test_precision_scores):.4f}")
    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s")
    print("Proběhl celý skript.")

    print(len(test_dice_scores), len(test_iou_scores), len(test_recall_scores), len(test_precision_scores))

    end = time.time()
    model_save_path = rf"C:\Users\USER\Desktop\weights\{current_date}.pth"
    
    model_and_metadata = {
        "model": type(net).__name__,
        "encoder": "resnet34",
        "epochs": epochs,
        "lr_start": lr_start, 
        "lr_end": optimizer.param_groups[0]['lr'],
        # scheduler: type(scheduler).__name__,
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
        "test_dice": epoch_test_dice,
        "test_iou": epoch_test_iou,
        "test_precision": epoch_test_precision,
        "test_recall": epoch_test_recall,
        "loss_function": type(loss_func).__name__,
        "augmentation": type(augmentations).__name__,
        "wanted_level": vars(test_dataset)["wanted_level"],
        "model_state_dict": net.state_dict(),
        "train_loss_hist": train_loss_hist,
        "valid_loss_hist": valid_loss_hist,
        "runtime": end - start,
        "info": f"Prvni spusteni, bez scheduleru, augmentace, lr je {lr_start}"
    }
    # Uložení váh modelu
    torch.save(model_and_metadata, model_save_path)
    print(f"Model byl uložen do {model_save_path}")

    print(f"Trénování trvalo {end - start:.2f} s")
