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
def dice_coef(X:torch.Tensor, Y:torch.Tensor, reduction=None, smooth=1.0):
    """
    Výpočet Dice koeficientu mezi predikcí a ground truth.
    
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    reduction: způsob agregace výsledků ('mean', 'sum', 'none')
    
    Vrací hodnotu v rozmezí [0, 1], kde 1 znamená perfektní shodu.
    """
    # Make sure inputs are flattened properly
    if X.dim() == 2 and X.size(0) == 1:  # Single image, not in batch form
        X_flat = X.view(-1)
        Y_flat = Y.view(-1)
    else:  # Batch of images or already proper shape
        batch_size = X.size(0)
        X_flat = X.view(batch_size, -1)
        Y_flat = Y.view(batch_size, -1)
    
    intersection = (X_flat * Y_flat).sum(dim=-1)
    union = X_flat.sum(dim=-1) + Y_flat.sum(dim=-1)
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    
    if reduction == 'mean':
        return dice_coef.mean()
    elif reduction == 'sum':
        return dice_coef.sum()
    else:
        return dice_coef
    
def dice_coefficient(X:torch.Tensor, Y:torch.Tensor, reduction=None, smooth=0.000001):
    """
    Výpočet Dice koeficientu mezi predikcí a ground truth.
    
    X: predikce (B, ...)
    Y: ground truth (B, ...)
    reduction: způsob agregace výsledků ('mean', 'sum', 'none')
    
    Vrací hodnotu v rozmezí [0, 1], kde 1 znamená perfektní shodu.
    """
    # eps = 0.000001
    
    # Převod na vhodný tvar pro výpočet po dávkách
    X_binary = (X > 0.5).float()
    Y_binary = (Y > 0.5).float()
    batch_size = X_binary.size(0)
    X_flat = X_binary.view(batch_size, -1)
    Y_flat = Y_binary.view(batch_size, -1)
    

    intersection = (X_flat * Y_flat).sum(dim=1)
    union = X_flat.sum(dim=1) + Y_flat.sum(dim=1)
    dice_coef = (2. * intersection + smooth) / (union + smooth)
    
    if reduction == 'mean':
        return dice_coef.mean()
    elif reduction == 'sum':
        return dice_coef.sum()
    else:
        return dice_coef

def calculate_iou(lbl:torch.Tensor, output:torch.Tensor):
    """
    Výpočet Intersection over Union (IoU) na úrovni jednotlivých snímků.
    """
    # Convert to numpy and apply threshold
    lbl_np = lbl.detach().cpu().numpy() > 0.5
    output_np = output.detach().cpu().numpy() > 0.5

    batch_size = lbl_np.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        # Extract single image and mask
        lbl_single = lbl_np[i].flatten()
        output_single = output_np[i].flatten()
        
        # Calculate intersection and union
        intersection = np.logical_and(lbl_single, output_single).sum()
        union = np.logical_or(lbl_single, output_single).sum()
        
        # Compute IoU with safeguard for empty masks
        if union > 0:
            iou_scores.append(intersection / union)
        else:
            # If the union is empty, need to handle this case
            # If both prediction and ground truth are empty, IoU is 1.0 (perfect match)
            # If one is empty and other isn't, IoU is 0.0 (complete mismatch)
            if np.sum(lbl_single) == 0 and np.sum(output_single) == 0:
                iou_scores.append(1.0)  # Both empty is a perfect match
            else:
                iou_scores.append(0.0)  # One empty, one not is a complete mismatch
    
    # Return average IoU across the batch
    return np.mean(iou_scores) if iou_scores else 0.0

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

if __name__ == "__main__":
    # Test Dice loss
    # Create identical tensors
    identical_tensor = torch.rand(1, 5, 4)
    print(identical_tensor)
    result = dice_coefficient(identical_tensor, identical_tensor, "mean")
    print(f"Dice coefficient for identical tensors: {result}")
    print(f"IoU for identical tensors: {calculate_iou(identical_tensor, identical_tensor)}")
    
    
    # Your original image test
    # from PIL import Image
    # img = Image.open(r'C:\Users\USER\Desktop\img_sem\test_dice1.png')
    # img_copy = Image.open(r"C:\Users\USER\Desktop\img_sem\test_dice2.png")
    # img = np.array(img)/255
    # img_copy = np.array(img_copy)/255
    # result = dice_coef(torch.tensor(img), torch.tensor(img_copy), "mean")
    # print(f"Dice coefficient for loaded images: {result}")