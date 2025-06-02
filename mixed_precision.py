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
from new_dataset import WSITileDatasetBalanced # Ujistěte se, že tento soubor existuje a je správný
from my_augmentation import MyAugmentations # Ujistěte se, že tento soubor existuje a je správný
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score, recall, precision
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss # FocalLoss je definována, ale níže není použita
import datetime
from load_data import load_wsi_and_ground_truth, load_lowres_masks

# <<< --- NOVÉ IMPORTY PRO MIXED PRECISION --- >>>
from torch.amp import autocast, GradScaler

# nacitani dat
wsi_paths_train, mask_paths_train, wsi_paths_val, mask_paths_val, wsi_paths_test, mask_paths_test = load_wsi_and_ground_truth(r"C:\Users\USER\Desktop\wsi_dir")
tissue_mask_paths_train, tissue_mask_paths_val, tissue_mask_paths_test = load_lowres_masks(r"C:\Users\USER\Desktop\colab_unet\masky_healthy")
gt_lowres_mask_paths_train, gt_lowres_mask_paths_val, gt_lowres_mask_paths_test = load_lowres_masks(r"C:\Users\USER\Desktop\colab_unet\gt_lowres_masky")

if __name__ == "__main__":

    # <<< --- PŘEPÍNAČ PRO MIXED PRECISION --- >>>
    use_amp = True # Nastavte na True pro použití mixed precision, False pro standardní FP32

    # Začátek trénovacího skriptu
    color_jitter_params = {
        "brightness": 0.20,
        "contrast": 0.20,
        "saturation": 0.15,
        "hue": 0.05
    }

    # Předpokládám, že MyAugmentations je vaše vlastní třída, basic_transform může být jiná sada transformací
    # V kódu datasetu používáte augmentations=basic_transform, takže MyAugmentations zde není přímo použito pro dataset.
    # Pokud chcete použít MyAugmentations, musíte ji předat do WSITileDatasetBalanced.
    augmentations = MyAugmentations(
        p_flip=0.5,
        color_jitter_params=color_jitter_params,
        p_color=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    # alubmentations_aug = AlbumentationsAug() # Tato proměnná není dále použita

    start = time.time()
    epochs = 100
    batch = 48
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if use_amp and device.type == 'cpu':
        print("Warning: Mixed precision (use_amp=True) is intended for CUDA devices. Running on CPU.")
        # use_amp = False # Můžete se rozhodnout AMP vypnout, pokud běží na CPU

    train_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_train, tissue_mask_paths=tissue_mask_paths_train,
                                           mask_paths=mask_paths_train, gt_lowres_mask_paths=gt_lowres_mask_paths_train,
                                           tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                           min_cancer_ratio_in_tile=0.01, augmentations=augmentations, # Zde používáte basic_transform
                                           dataset_len=10560)
    
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=6, pin_memory=True if device.type == 'cuda' else False)

    val_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_val, tissue_mask_paths=tissue_mask_paths_val,
                                         mask_paths=mask_paths_val, gt_lowres_mask_paths=gt_lowres_mask_paths_val,
                                         tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                         min_cancer_ratio_in_tile=0.01, augmentations=basic_transform,
                                         dataset_len=7680)
    
    validloader = DataLoader(val_dataset, batch_size=batch, num_workers=4, shuffle=False, pin_memory=True if device.type == 'cuda' else False)

    test_dataset = WSITileDatasetBalanced(wsi_paths=wsi_paths_test, tissue_mask_paths=tissue_mask_paths_test,
                                          mask_paths=mask_paths_test, gt_lowres_mask_paths=gt_lowres_mask_paths_test,
                                          tile_size=256, wanted_level=2, positive_sampling_prob=0.5,
                                          min_cancer_ratio_in_tile=0.01, augmentations=basic_transform,
                                          dataset_len=21120)
    
    testloader = DataLoader(test_dataset,batch_size=batch, num_workers=0, shuffle=False) # Pro testování je num_workers=0 často bezpečnější

    net = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    net = net.to(device)
    lr_start = 0.01
    optimizer = optim.Adam(net.parameters(), lr=lr_start) #weight_decay=1e-5
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 40, 65, 90], gamma=0.1)
    
    loss_func = DiceLoss(mode="binary", from_logits=True, smooth=1e-6)
    # focal_loss = FocalLoss(mode="binary", alpha=0.6, gamma=2.0) # Není použito níže, pokud chcete kombinovat, upravte výpočet loss

    # <<< --- INICIALIZACE GRADSCALER PRO MIXED PRECISION --- >>>
    scaler = GradScaler(enabled=use_amp, device = 'cuda')

    best_val_dice = float("-inf")
    # weights_patience = 7 # Tato proměnná se zdá být používána, ale není jasné, jaký je její účel bez early stopping logiky
    patience_counter = 0 # Přejmenoval jsem pro srozumitelnost, pokud chcete implementovat vlastní patience
    min_impovement_for_saving = 0.001 # Přejmenováno pro srozumitelnost
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Historie metrik
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_iou': [], 'val_iou': [],
        'train_recall': [], 'val_recall': [],
        'train_precision': [], 'val_precision': []
    }
    batch_load_time = []
    lr_change_epochs = []

    for epoch in range(epochs):
        start_epoch = time.time()
        print(f"Epoch {epoch+1}/{epochs}\n") # Upraveno pro přehlednější výpis epoch
        
        # Výpis průměrných metrik každých 10 epoch
        if epoch >= 10 and epoch % 10 == 0:
            print("--- Average Metrics (last 10 epochs) ---") # Upraveno pro přehlednost
            if metrics_history['train_f1']: # Zkontrolujte, zda seznamy nejsou prázdné
                print(f"F1: train: {np.mean(metrics_history['train_f1'][-10:]):.4f}, val: {np.mean(metrics_history['val_f1'][-10:]):.4f}")
                print(f"Recall: train: {np.mean(metrics_history['train_recall'][-10:]):.4f}, val: {np.mean(metrics_history['val_recall'][-10:]):.4f}")
                print(f"Precision: train: {np.mean(metrics_history['train_precision'][-10:]):.4f}, val: {np.mean(metrics_history['val_precision'][-10:]):.4f}")
                print(f"IoU: train: {np.mean(metrics_history['train_iou'][-10:]):.4f}, val: {np.mean(metrics_history['val_iou'][-10:]):.4f}")
                print(f"Loss: train: {np.mean(metrics_history['train_loss'][-10:]):.4f}, val: {np.mean(metrics_history['val_loss'][-10:]):.4f}")
            print("-" * 30)

        # Tréninková fáze
        net.train()
        epoch_train_loss_sum = 0.0
        epoch_train_tp, epoch_train_fp, epoch_train_fn, epoch_train_tn = 0, 0, 0, 0
        
        loop_start_time = time.time() # Pro měření času načítání batchů
        for k, (data, lbl) in enumerate(trainloader):
            batch_time = time.time() - loop_start_time
            batch_load_time.append(batch_time)

            data = data.to(device, non_blocking=True) # non_blocking=True může mírně zrychlit přenosy
            lbl = lbl.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # set_to_none=True může být mírně efektivnější

            # <<< --- FORWARD PASS S AUTOCAST --- >>>
            with autocast(enabled=use_amp, device_type='cuda'):
                output_logits = net(data) # Přejmenováno pro srozumitelnost, že jde o logity
                loss = loss_func(output_logits, lbl)
                # Pokud byste kombinovali loss funkce:
                # loss_d = dice_loss(output_logits, lbl)
                # loss_f = focal_loss(output_logits, lbl)
                # loss = loss_d + loss_f

            # <<< --- SCALER PRO BACKWARD PASS --- >>>
            if use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_train_loss_sum += loss.item()

            # Výpočet metrik (po sigmoid aktivaci)
            with torch.inference_mode(): # Používáme inference_mode pro výpočty bez gradientů
                output_probs = torch.sigmoid(output_logits) # Sigmoid aplikujeme zde pro metriky
                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode="binary", threshold=0.5) 
                epoch_train_fp += fp.sum()
                epoch_train_fn += fn.sum()
                epoch_train_tp += tp.sum()
                epoch_train_tn += tn.sum()
            
            loop_start_time = time.time() # Reset času pro další batch

        # Výpočet průměrných tréninkových metrik za epochu
        avg_train_loss = epoch_train_loss_sum / len(trainloader)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['train_precision'].append(precision(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item())
        metrics_history['train_recall'].append(recall(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item())
        metrics_history['train_f1'].append(f1_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item())
        metrics_history['train_iou'].append(iou_score(tp=epoch_train_tp, fp=epoch_train_fp, fn=epoch_train_fn, tn=epoch_train_tn).item())

        print(f"Epoch {epoch+1} - Train: F1: {metrics_history['train_f1'][-1]:.4f}, IoU: {metrics_history['train_iou'][-1]:.4f}, Loss: {metrics_history['train_loss'][-1]:.4f}")
        print("-" * 30)

        # Validační fáze
        net.eval()
        epoch_val_loss_sum = 0.0
        epoch_val_tp, epoch_val_fp, epoch_val_fn, epoch_val_tn = 0, 0, 0, 0
        with torch.inference_mode(): # inference_mode je zde vhodnější než no_grad
            for kk, (data, lbl) in enumerate(validloader):
                data = data.to(device, non_blocking=True)
                lbl = lbl.to(device, non_blocking=True)

                # <<< --- FORWARD PASS S AUTOCAST PRO VALIDACI --- >>>
                with autocast(enabled=use_amp, device_type='cuda'):
                    output_logits = net(data)
                    loss = loss_func(output_logits, lbl)
                    # Pokud byste kombinovali:
                    # loss_d = dice_loss(output_logits, lbl)
                    # loss_f = focal_loss(output_logits, lbl)
                    # loss = loss_d + loss_f
                
                output_probs = torch.sigmoid(output_logits)
                epoch_val_loss_sum += loss.item()

                lbl_long = lbl.long()
                tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode="binary", threshold=0.5)
                epoch_val_fp += fp.sum()
                epoch_val_fn += fn.sum()
                epoch_val_tp += tp.sum()
                epoch_val_tn += tn.sum()

        # Výpočet průměrných validačních metrik za epochu
        avg_val_loss = epoch_val_loss_sum / len(validloader)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_precision'].append(precision(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item())
        metrics_history['val_recall'].append(recall(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item())
        metrics_history['val_f1'].append(f1_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item())
        metrics_history['val_iou'].append(iou_score(tp=epoch_val_tp, fp=epoch_val_fp, fn=epoch_val_fn, tn=epoch_val_tn).item())
        
        print(f"Epoch {epoch+1} - Val:   F1: {metrics_history['val_f1'][-1]:.4f}, IoU: {metrics_history['val_iou'][-1]:.4f}, Loss: {metrics_history['val_loss'][-1]:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.7f}") # Přidáno :.7f pro lepší čitelnost LR
        print("-" * 30)

        # Ukládání nejlepšího modelu
        # Logika ukládání byla mírně upravena - ukládá se, pokud je aktuální val_dice lepší než dosud nejlepší
        if metrics_history['val_f1'][-1] > best_val_dice + min_impovement_for_saving:
            best_val_dice = metrics_history['val_f1'][-1]
            print(f"New best model found! Val F1: {best_val_dice:.4f}. Saving model from epoch {epoch+1}...")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Uložení stavu scheduleru
                'scaler_state_dict': scaler.state_dict() if use_amp and device.type == 'cuda' else None, # Uložení stavu scaleru
                'metrics_history': metrics_history, # Uložení celé historie metrik
                'best_val_dice': best_val_dice,
                'lr': current_lr,
            }
            model_checkpoint_path = rf"C:\Users\USER\Desktop\weights\best_weights_{current_date}.pth"
            try:
                torch.save(checkpoint, model_checkpoint_path)
                print(f"Model checkpoint saved to {model_checkpoint_path}")
            except Exception as e:
                print(f"Error saving model checkpoint: {e}")
            patience_counter = 0 # Reset počítadla patience
        else:
            patience_counter += 1
        print(f"Patience counter: {patience_counter}") # Pro sledování
        print("-" * 30)

        # Krok scheduleru
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step() 
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"Learning rate changed: {prev_lr:.7f} -> {new_lr:.7f} at epoch {epoch+1}")
            lr_change_epochs.append(epoch + 1)
        
        print(f"Epoch {epoch+1} finished in {time.time() - start_epoch:.2f} s")
        print("=" * 40 + "\n")


    # Vykreslení grafu ztrát
    plt.figure(figsize=(12, 6)) # Mírně zvětšeno pro lepší čitelnost
    plt.plot(metrics_history['train_loss'], label='Tréninková ztráta', color='blue', linestyle='--')
    plt.plot(metrics_history['val_loss'], label='Validační ztráta', color='orange', linestyle='-')
    plt.xlabel('Epocha')
    plt.ylabel('Ztráta')
    plt.title('Ztrátová křivka')
    plt.legend()
    plt.grid(True)
    plot_save_path = rf'C:\Users\USER\Desktop\loss_plot_{current_date}.png'
    try:
        plt.savefig(plot_save_path)
        print(f"Loss plot saved to {plot_save_path}")
    except Exception as e:
        print(f"Error saving loss plot: {e}")
    # plt.show(block=False) # Odkomentujte, pokud chcete zobrazit graf okamžitě a neblokovat skript
    # plt.pause(5)
    # plt.close()

    # Testovací fáze (načtení nejlepšího modelu)
    print("\n" + "="*10 + " Starting Testing Phase " + "="*10)
    best_model_path = rf"C:\Users\USER\Desktop\weights\best_weights_{current_date}.pth" # Cesta k nejlepšímu uloženému modelu
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        # scaler.load_state_dict(checkpoint['scaler_state_dict']) # Pro případné obnovení tréninku
    else:
        print(f"Warning: Best model checkpoint not found at {best_model_path}. Testing with the last state of the model.")

    net.eval()
    epoch_test_tp, epoch_test_fp, epoch_test_fn, epoch_test_tn = 0, 0, 0, 0
    with torch.inference_mode():
        for kk, (data, lbl) in enumerate(testloader):
            data = data.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            # <<< --- FORWARD PASS S AUTOCAST PRO TESTOVÁNÍ --- >>>
            with autocast(enabled=use_amp, device_type='cuda'):
                output_logits = net(data)
            
            output_probs = torch.sigmoid(output_logits)

            lbl_long = lbl.long()
            tp, fp, fn, tn = get_stats(output_probs, lbl_long, mode="binary", threshold=0.5)
            epoch_test_fp += fp.sum()
            epoch_test_fn += fn.sum()
            epoch_test_tp += tp.sum()
            epoch_test_tn += tn.sum()

    # Výpočet testovacích metrik
    # Seznamy pro testovací metriky nejsou potřeba, pokud testujete jen jednou na konci
    epoch_test_precision = precision(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_recall = recall(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_dice = f1_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()
    epoch_test_iou = iou_score(tp=epoch_test_tp, fp=epoch_test_fp, fn=epoch_test_fn, tn=epoch_test_tn).item()

    print("\n--- Test Results ---")
    print(f"Test Dice (F1): {epoch_test_dice:.4f}")
    print(f"Test IoU: {epoch_test_iou:.4f}")
    print(f"Test Recall: {epoch_test_recall:.4f}")
    print(f"Test Precision: {epoch_test_precision:.4f}")
    print(f"Average batch load time: {np.mean(batch_load_time):.4f} s (approx.)") # Přesnější by bylo měřit jen načítání
    print("="*40)

    end = time.time()
    final_model_save_path = rf"C:\Users\USER\Desktop\weights\final_model_{current_date}.pth" # Přejmenováno pro odlišení od checkpointu
    
    # Metadata k uložení (s použitím struktury metrics_history)
    model_and_metadata = {
        "model_name": type(net).__name__, # Přejmenováno
        "encoder": net.encoder.__class__.__name__ if hasattr(net, 'encoder') else "N/A", # Získání jména enkodéru
        "epochs_completed": epoch + 1,
        "lr_start": lr_start, 
        "lr_final": optimizer.param_groups[0]['lr'], # Přejmenováno
        "scheduler_type": type(scheduler).__name__, # Přejmenováno
        "optimizer_type": type(optimizer).__name__, # Přejmenováno
        "batch_size": batch,
        "train_dataset_len": len(train_dataset), # Správný způsob získání délky
        "metrics_history": metrics_history, # Uložení celé historie
        "test_dice": epoch_test_dice,
        "test_iou": epoch_test_iou,
        "test_precision": epoch_test_precision,
        "test_recall": epoch_test_recall,
        "loss_function_type": type(loss_func).__name__, # Přejmenováno
        "augmentations_used": type(train_dataset.augmentations).__name__ if hasattr(train_dataset, 'augmentations') and train_dataset.augmentations else "N/A",
        "wanted_level": train_dataset.wanted_level if hasattr(train_dataset, 'wanted_level') else "N/A",
        "model_state_dict": net.state_dict(), # Stav modelu po všech epochách
        "runtime_seconds": end - start,
        "info": f"Run with use_amp={use_amp and device.type=='cuda'}. LR_start: {lr_start}, Scheduler: {type(scheduler).__name__}, Milestones: {getattr(scheduler, 'milestones', 'N/A')}",
        "best_val_dice_at_checkpoint": best_val_dice,
        "current_date_timestamp": current_date
    }
    try:
        torch.save(model_and_metadata, final_model_save_path)
        print(f"Final model and metadata saved to {final_model_save_path}")
    except Exception as e:
        print(f"Error saving final model and metadata: {e}")

    print(f"Total training and testing time: {end - start:.2f} s")
    print("Script finished.")