import torch
import numpy as np
import torchvision.transforms.functional as TF

# def dice_loss(X, Y):

#     eps = 1.

#     dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )

#     return 1 - dice
def dice_loss(X, Y, reduction='mean'):
    """
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    """
    eps = 0.000001 
    
    # Převod na vhodný tvar pro výpočet po dávkách
    batch_size = X.size(0)
    X_flat = X.view(batch_size, -1)
    Y_flat = Y.view(batch_size, -1)
    
    intersection = (X_flat * Y_flat).sum(dim=1)
    dice_coef = (2. * intersection + eps) / (X_flat.sum(dim=1) + Y_flat.sum(dim=1) + eps)
    dice_loss = 1 - dice_coef
    
    if reduction == 'mean':
        return dice_loss.mean()
    elif reduction == 'sum':
        return dice_loss.sum()
    else:
        return dice_loss
    
def dice_coefficient(X, Y, reduction='mean'):
    """
    Výpočet Dice koeficientu mezi predikcí a ground truth.
    
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    reduction: způsob agregace výsledků ('mean', 'sum', 'none')
    
    Vrací hodnotu v rozmezí [0, 1], kde 1 znamená perfektní shodu.
    """
    eps = 0.000001
    
    # Převod na vhodný tvar pro výpočet po dávkách
    batch_size = X.size(0)
    X_flat = X.view(batch_size, -1)
    Y_flat = Y.view(batch_size, -1)
    
    intersection = (X_flat * Y_flat).sum(dim=1)
    dice_coef = (2. * intersection + eps) / (X_flat.sum(dim=1) + Y_flat.sum(dim=1) + eps)
    
    if reduction == 'mean':
        return dice_coef.mean()
    elif reduction == 'sum':
        return dice_coef.sum()
    else:
        return dice_coef

def calculate_iou(lbl, output):
    """
    Výpočet Intersection over Union (IoU).
    """
    lbl_binary = lbl.detach().cpu().numpy() > 0.5
    output_binary = output.detach().cpu().numpy() > 0.5
    intersection = np.logical_and(lbl_binary, output_binary).sum()
    union = np.logical_or(lbl_binary, output_binary).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

def basic_transform(tile, mask):
    image = TF.to_tensor(tile)
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    mask = TF.to_tensor(mask)

    return image, mask