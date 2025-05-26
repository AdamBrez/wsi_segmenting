# SOUBOR: aug_norm.py (finální opravená verze)

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchstain
from torchvision import transforms

class AlbumentationsAugForStain:
    def __init__(self,
                 p_flip=0.5,
                 p_rotate90=0.5,
                 p_shiftscalerotate=0.5,
                 p_elastic=0.0,
                 p_color=0.5,
                 p_hestain=0.0,
                 p_noise=0.0,
                 p_blur=0.0):
        self.transform = A.Compose([
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_rotate90),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0625, 0.0625),
                     rotate=(-15, 15), p=p_shiftscalerotate),
            A.ElasticTransform(alpha=300, sigma=10, p=p_elastic,
                               border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=p_color, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
            A.HueSaturationValue(p=p_color, hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10)),
            A.RGBShift(p=p_color, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            A.HEStain(p=p_hestain, method="random_preset", intensity_scale_range=(0.9, 1.1), intensity_shift_range=(-0.1, 0.1)),
            A.GaussNoise(std_range=[0.03, 0.07], p=p_noise),
            A.GaussianBlur(blur_limit=5, sigma_limit=[0.1, 0.9], p=p_blur),
        ])

    def __call__(self, image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)
        if mask_np.max() > 1:
            mask_np = (mask_np > 128).astype(np.uint8)
        augmented = self.transform(image=image_np, mask=mask_np)
        return augmented['image'], augmented['mask']


class StainNormalizingAugmentationsTorchStain:
    def __init__(self, 
                 stain_target_image_path,
                 albumentations_aug,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        
        self.albumentations_aug = albumentations_aug
        
        # print("Inicializace TorchStain normalizátoru...")
        target_cv = cv2.cvtColor(cv2.imread(stain_target_image_path), cv2.COLOR_BGR2RGB)
        
        self.to_tensor_255 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.fit(self.to_tensor_255(target_cv))
        # print("TorchStain normalizátor připraven.")

        self.final_transform = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, image, mask):
        # 1. Aplikace geometrických a barevných augmentací z Albumentations
        image_aug_np, mask_aug_np = self.albumentations_aug(image, mask)

        # 2. Příprava obrázku pro TorchStain a normalizace barvení
        image_to_normalize_tensor = self.to_tensor_255(image_aug_np)
        
        # Zde torchstain vrací tenzor ve formátu (H, W, C)
        image_normalized_tensor_HWC, _, _ = self.normalizer.normalize(I=image_to_normalize_tensor, stains=False)

        # Přehození dimenzí z (H, W, C) na (C, H, W) pro torchvision
        image_normalized_tensor_CHW = image_normalized_tensor_HWC.permute(2, 0, 1)

        # 3. Aplikace finální normalizace (mean/std) na tenzor se správným tvarem
        final_image_tensor = self.final_transform(image_normalized_tensor_CHW)
        
        # 4. Příprava masky
        mask_tensor = torch.from_numpy(mask_aug_np).unsqueeze(0).float()

        return final_image_tensor, mask_tensor