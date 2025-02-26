import os
os.add_dll_directory(r"C:\Users\USER\miniforge3\envs\mamba_env\lib\site-packages\openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin")

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from my_dataloader import WSITileDataset
from my_model import Unet2D
from my_functions import dice_loss, calculate_accuracy, calculate_iou



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
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_001.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_002.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_003.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_004.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_005.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_006.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_007.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_008.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_009.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_010.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_011.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_012.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_013.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_089.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_091.npy",
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
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_014.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_015.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_084.npy",
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
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_017.npy",
    r"C:\Users\USER\Desktop\colab_unet\masky\mask_018.npy",
]
mask_paths_test = [
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_017.tif",
    r"E:\skola\U-Net\Pytorch-UNet\wsi_dir\mask_018.tif",
]

if __name__ == "__main__":
# Začátek trénovacího skriptu
    transform = ToTensor()
    start = time.time()
    epochs = 51
    batch = 16
    # device = torch.device('cpu')
    device = torch.device('cuda:0')
    train_dataset = WSITileDataset(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train, mask_paths=mask_paths_train, tile_size=256, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)


    val_dataset = WSITileDataset(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val, mask_paths=mask_paths_val, tile_size=256, transform=transform)
    validloader= torch.utils.data.DataLoader(val_dataset,batch_size=batch, num_workers=4, shuffle=True)

    test_dataset = WSITileDataset(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test, mask_paths=mask_paths_test, tile_size=256, transform=transform)
    testloader= torch.utils.data.DataLoader(test_dataset,batch_size=1, num_workers=0, shuffle=True)

    net = Unet2D(in_size=3)
    net = net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9,11], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    train_iou = []
    valid_iou = []

    test_dice_scores = []
    test_accuracy_scores = []
    test_iou_scores = []

    it =-1  #'https://krunker.io/?game=FRA:cniz2'
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs-1}")
        if epoch > 9:
            print(f"Accuracy: train: {np.mean(train_acc)}, val: {np.mean(valid_acc)}")
            print(f"IoU: train: {np.mean(train_iou)}, val: {np.mean(valid_iou)}")
            print(f"Loss: train: {np.mean(train_loss)}, val: {np.mean(valid_loss)}")
        acc_tmp = []
        iou_tmp = []
        loss_tmp = []
        for k,(data,lbl) in enumerate(trainloader):
            it+=1

            data = data.to(device)
            lbl = lbl.to(device)

            net.train()
            output = net(data)

            output = torch.sigmoid(output)
            loss = dice_loss(lbl, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Výpočet metrik
            acc = calculate_accuracy(lbl, output)
            iou = calculate_iou(lbl, output)

            acc_tmp.append(acc)
            iou_tmp.append(iou)
            loss_tmp.append(loss.cpu().detach().numpy())

        train_loss.append(np.mean(loss_tmp))
        train_acc.append(np.mean(acc_tmp))
        train_iou.append(np.mean(iou_tmp))

        acc_tmp = []
        iou_tmp = []
        loss_tmp = []
        for kk,(data, lbl) in enumerate(validloader):
            with torch.no_grad():

                data = data.to(device)
                lbl = lbl.to(device)

                net.eval()
                output = net(data)

                output = torch.sigmoid(output)
                loss = dice_loss(lbl, output)

                # Výpočet metrik
                acc = calculate_accuracy(lbl, output)
                iou = calculate_iou(lbl, output)

                acc_tmp.append(acc)
                iou_tmp.append(iou)
                loss_tmp.append(loss.cpu().detach().numpy())

        valid_loss.append(np.mean(loss_tmp))
        valid_acc.append(np.mean(acc_tmp))
        valid_iou.append(np.mean(iou_tmp))

        scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(valid_loss, label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'C:\Users\USER\Desktop\loss_curve_50_vol2.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    # Testování
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
            dice_score = 1 - dice_loss(lbl, output)  # Dice koeficient
            acc = calculate_accuracy(lbl, output)
            iou = calculate_iou(lbl, output)

            test_dice_scores.append(dice_score.cpu().item())
            test_accuracy_scores.append(acc)
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
    average_test_acc = np.mean(test_accuracy_scores)
    average_test_iou = np.mean(test_iou_scores)

    print(f"Average Dice coefficient on test data: {average_test_dice:.4f}")
    print(f"Average Accuracy on test data: {average_test_acc:.4f}")
    print(f"Average IoU on test data: {average_test_iou:.4f}")
    print(f"IoU trénování: {np.mean(train_iou)}")
    print(f"IoU validace: {np.mean(valid_iou)}")

    print("Proběhl celý skript.")

    model_save_path = r"C:\Users\USER\Desktop\weights\unet_model_50_vol2.pth"

    # Uložení váh modelu
    torch.save(net.state_dict(), model_save_path)
    print(f"Model byl uložen do {model_save_path}")
    end = time.time()
    print(f"Trénování trvalo {end - start:.2f} s")