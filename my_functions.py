import torch
import numpy as np

def dice_loss(X, Y):

    eps = 1.

    dice = ((2. * torch.sum(X*Y) + eps) / (torch.sum(X) + torch.sum(Y) + eps) )

    return 1 - dice

def calculate_accuracy(lbl, output):
    """
    Výpočet přesnosti (accuracy).
    """
    lbl_binary = lbl.detach().cpu().numpy() > 0.5
    output_binary = output.detach().cpu().numpy() > 0.5
    correct = (lbl_binary == output_binary).sum()
    total = lbl_binary.size
    accuracy = correct / total
    return accuracy

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