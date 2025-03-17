import torch
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# def dice_loss(X, Y):

#     eps = 1.

#     dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )

#     return 1 - dice
def dice_loss(X, Y, reduction='mean', smooth=1.0):
    """
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    """
    # eps = 0.000001 //claude mi odstranil to eps a přidal smooth do argumentů tak hádám že to vyjde stejně
    
    # Převod na vhodný tvar pro výpočet po dávkách
    batch_size = X.size(0)
    X_flat = X.view(batch_size, -1)
    Y_flat = Y.view(batch_size, -1)
    
    intersection = (X_flat * Y_flat).sum(dim=1)
    union = X_flat.sum(dim=1) + Y_flat.sum(dim=1)

    dice_coef = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coef
    
    if reduction == 'mean':
        return dice_loss.mean()
    elif reduction == 'sum':
        return dice_loss.sum()
    else:
        return dice_loss

def dice_bce_loss(pred, target, alpha=0.5, smooth=1.0):
    # Dice component
    dice_component = dice_loss(pred, target, smooth=smooth)
    # BCE component
    bce_component = F.binary_cross_entropy(pred, target)
    # Combined loss
    return alpha * dice_component + (1-alpha) * bce_component
    
def dice_coefficient(X, Y, reduction='mean', smooth=1.0):
    """
    Výpočet Dice koeficientu mezi predikcí a ground truth.
    
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    reduction: způsob agregace výsledků ('mean', 'sum', 'none')
    
    Vrací hodnotu v rozmezí [0, 1], kde 1 znamená perfektní shodu.
    """
    # eps = 0.000001
    
    # Převod na vhodný tvar pro výpočet po dávkách
    batch_size = X.size(0)
    X_flat = X.view(batch_size, -1)
    Y_flat = Y.view(batch_size, -1)
    
    intersection = (X_flat * Y_flat).sum(dim=1)
    union = X_flat.sum(dim=1) + Y_flat.sum(dim=1)
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    
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
    image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    mask = TF.to_tensor(mask)

    return image, mask

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred * target + (1 - pred) * (1 - target)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss.mean()