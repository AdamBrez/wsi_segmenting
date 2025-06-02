# SOUBOR: aug_norm.py (verze opravená pro multiprocessing)

import cv2
import numpy as np
import albumentations as A
import torch
import torchstain
from torchvision import transforms
from PIL import Image

# ================================================================= #
# 1. KROK: Nahrazení lambda funkcí normálními funkcemi              #
# Tyto funkce definujeme na nejvyšší úrovni, aby byly snadno "picklovatelné". #
# ================================================================= #
def multiply_by_255(tensor):
    """Násobí tensor hodnotou 255."""
    return tensor * 255

def divide_by_255(tensor):
    """Dělí tensor hodnotou 255.0."""
    return tensor / 255.0

class AlbumentationsAugForStain:
    # ... (tato třída zůstává beze změny) ...
    def __init__(self,
                 p_flip=0.5, p_rotate90=0.5, p_shiftscalerotate=0.5, p_elastic=0.0,
                 p_color=0.5, p_hestain=0.0, p_noise=0.0, p_blur=0.0):
        self.transform = A.Compose([
            A.HorizontalFlip(p=p_flip), A.VerticalFlip(p=p_flip), A.RandomRotate90(p=p_rotate90),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0625, 0.0625), rotate=(-15, 15), p=p_shiftscalerotate),
            A.ElasticTransform(alpha=300, sigma=10, p=p_elastic, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=p_color, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
            A.HueSaturationValue(p=p_color, hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10)),
            A.RGBShift(p=p_color, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            A.HEStain(p=p_hestain, method="random_preset", intensity_scale_range=(0.9, 1.1), intensity_shift_range=(-0.1, 0.1)),
            A.GaussNoise(std_range=[0.03, 0.07], p=p_noise), A.GaussianBlur(blur_limit=5, sigma_limit=[0.1, 0.9], p=p_blur),
        ])

    def __call__(self, image, mask=None):
        image_np = np.array(image)
        mask_np = np.array(mask) if mask is not None else None
        augmented = self.transform(image=image_np, mask=mask_np)
        return augmented['image'], augmented['mask']


class StainNormalizingAugmentationsTorchStain:
    def __init__(self, 
                 stain_target_image_path,
                 albumentations_aug,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        
        self.albumentations_aug = albumentations_aug
        
        print("Inicializace TorchStain normalizátoru...")
        target_cv = cv2.cvtColor(cv2.imread(stain_target_image_path), cv2.COLOR_BGR2RGB)
        
        # ================================================================= #
        # 2. KROK: Použití nových pojmenovaných funkcí v transformacích     #
        # ================================================================= #
        self.to_tensor_255 = transforms.Compose([
            transforms.ToTensor(),
            # Původně: transforms.Lambda(lambda x: x * 255)
            multiply_by_255 
        ])
        
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(self.to_tensor_255(target_cv))
        print("TorchStain normalizátor připraven.")

        self.final_transform = transforms.Compose([
            # Původně: transforms.Lambda(lambda x: x / 255.0)
            divide_by_255,
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, mask=None):
        # ... (zbytek třídy __call__ zůstává beze změny) ...
        image_aug_np, mask_aug_np = self.albumentations_aug(image, mask)
        image_to_normalize_tensor = self.to_tensor_255(image_aug_np)
        image_normalized_tensor_HWC, _, _ = self.normalizer.normalize(I=image_to_normalize_tensor, stains=False)
        image_normalized_tensor_CHW = image_normalized_tensor_HWC.permute(2, 0, 1)
        final_image_tensor = self.final_transform(image_normalized_tensor_CHW)
        
        if mask is None:
            return final_image_tensor
        else:
            if mask_aug_np.max() > 1:
                 mask_aug_np = (mask_aug_np > 128).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_aug_np).unsqueeze(0).float()
            return final_image_tensor, mask_tensor