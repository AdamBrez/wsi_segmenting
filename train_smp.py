import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from my_dataloader import WSITileDataset
from my_model import Unet2D
from my_functions import dice_loss, dice_coefficient, calculate_iou, basic_transform, dice_bce_loss
from my_augmentation import MyAugmentations
import segmentation_models_pytorch as smp

wsi_paths_train = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_001.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_002.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_003.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_004.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_005.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_006.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_007.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_008.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_009.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_010.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_011.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_012.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_013.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_089.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_091.tif",
]
tissue_mask_paths_train = [
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_001.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_002.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_003.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_004.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_005.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_006.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_007.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_008.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_009.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_010.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_011.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_012.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_013.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_089.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_091.npy",
]
mask_paths_train = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_001.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_002.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_003.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_004.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_005.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_006.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_007.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_008.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_009.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_010.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_011.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_012.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_013.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_089.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_091.tif",
]

wsi_paths_val = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_014.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_015.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_084.tif",
]
tissue_mask_paths_val = [
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_014.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_015.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_084.npy",
]
mask_paths_val = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_014.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_015.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_084.tif",
]

wsi_paths_test = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_017.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\tumor_018.tif",
]
tissue_mask_paths_test = [
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_017.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky_new\mask_018.npy",
]
mask_paths_test = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_017.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_018.tif",
]

if __name__ == "__main__":

# Začátek trénovacího skriptu
    color_jitter_params = {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    }

    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    start = time.time()
    epochs = 101
    batch = 32
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDataset(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train, mask_paths=mask_paths_train, tile_size=256, augmentations=augmentations)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)


    val_dataset = WSITileDataset(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val, mask_paths=mask_paths_val, tile_size=256, augmentations=basic_transform)
    validloader= DataLoader(val_dataset,batch_size=batch, num_workers=4, shuffle=True)

    test_dataset = WSITileDataset(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test, mask_paths=mask_paths_test, tile_size=256, augmentations=basic_transform)
    testloader = DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=True)

    net = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.0003)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,11], gamma=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainloader), epochs=epochs, max_lr=0.001, pct_start=0.3, div_factor=25, final_div_factor=1000)

    train_loss = []
    valid_loss = []
    train_iou = []
    valid_iou = []
    train_dice = []
    valid_dice = []

    batch_load_time = [] 
    it =-1  #'https://krunker.io/?game=FRA:cniz2'
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs-1}")
        if epoch >=10  and epoch % 10 == 0:
            print(f"IoU: train: {np.mean(train_iou)}, val: {np.mean(valid_iou)}")
            print(f"Loss: train: {np.mean(train_loss)}, val: {np.mean(valid_loss)}")
            print(f"Dice: train: {np.mean(train_dice)}, val: {np.mean(valid_dice)}")
        
        iou_tmp = []
        loss_tmp = []
        dice_tmp = []

        start_time = time.time()
        for k,(data,lbl) in enumerate(trainloader):
            # mask_range_values = lbl.max(), lbl.min()
            # print(f"Mask range values: {mask_range_values}")
            # print(k)
            batch_time = time.time() - start_time
            batch_load_time.append(batch_time)

            it+=1

            data = data.to(device)
            lbl = lbl.to(device)

            net.train()
            output = net(data)

            output = torch.sigmoid(output)
            loss = dice_bce_loss(output, lbl)
            dice = dice_coefficient(output, lbl)
            iou = calculate_iou(output=output, lbl=lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            iou_tmp.append(iou)
            loss_tmp.append(loss.cpu().detach().numpy())
            dice_tmp.append(dice.cpu().detach().numpy())

            start_time = time.time()

        train_loss.append(np.mean(loss_tmp))
        train_iou.append(np.mean(iou_tmp))
        train_dice.append(np.mean(dice_tmp))


        iou_tmp = []
        loss_tmp = []
        dice_tmp = []

        # val loop
        for kk,(data, lbl) in enumerate(validloader):
            with torch.no_grad():

                data = data.to(device)
                lbl = lbl.to(device)

                net.eval()
                output = net(data)

                output = torch.sigmoid(output)
                loss = dice_bce_loss(output, lbl)
                dice = dice_coefficient(output, lbl)

                # Výpočet metrik
                iou = calculate_iou(output=output, lbl=lbl)

                iou_tmp.append(iou)
                loss_tmp.append(loss.cpu().detach().numpy())
                dice_tmp.append(dice.cpu().detach().numpy())

        valid_loss.append(np.mean(loss_tmp))
        valid_iou.append(np.mean(iou_tmp))
        valid_dice.append(np.mean(dice_tmp))


    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\Users\USER\Desktop\unet_16_3_100e.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    test_dice_scores = []
    test_iou_scores = []

    # Test loop
    for kk, (data, lbl) in enumerate(testloader):
        if kk > 3:
            break
        with torch.no_grad():
            data = data.to(device)
            lbl = lbl.to(device)

            # Předpověď sítě
            net.eval()
            output = net(data)
            output = torch.sigmoid(output)

            # Výpočet metrik
            dice_score = dice_coefficient(output, lbl)  # Dice koeficient
            iou = calculate_iou(lbl=lbl, output=output)

            test_dice_scores.append(dice_score.cpu().item())
            test_iou_scores.append(iou)

            # Vizualizace
            if kk < 3:
                plt.figure(figsize=(15, 10))
                plt.subplot(131)
                plt.title("Input")
                # plt.imshow(data[0, 1, :, :].detach().cpu().numpy())
                plt.imshow(data[0].permute(1, 2, 0).detach().cpu().numpy())
                plt.subplot(132)
                plt.title("Prediction")
                plt.imshow(output[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
                plt.subplot(133)
                plt.title("Ground Truth")
                plt.imshow(lbl[0, 0, :, :].detach().cpu().numpy(), cmap="gray")
                plt.show(block=False)
                plt.pause(5)
                plt.close()
    # Výpočet průměrných metrik na testovacích datech
    average_test_dice = np.mean(test_dice_scores)
    average_test_iou = np.mean(test_iou_scores)

    print(f"Average Dice coefficient on test data: {average_test_dice:.4f}")
    print(f"Average IoU on test data: {average_test_iou:.4f}")
    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s")
    print("Proběhl celý skript.")

    model_save_path = r"C:\Users\USER\Desktop\weights\unet_16_3_100e.pth"

    # Uložení váh modelu
    torch.save(net.state_dict(), model_save_path)
    print(f"Model byl uložen do {model_save_path}")
    end = time.time()
    print(f"Trénování trvalo {end - start:.2f} s")
