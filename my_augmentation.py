import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, Normalize
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import numpy as np



class MyAugmentations:
    def __init__(self,
                 p_flip=0.5,
                 color_jitter_params=None,
                 p_color = 0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        """
        p_flip: pravděpodobnost pro horizontální/vertikální flip
        p_color: pravděpodobnost pro přidání barevných úprav
        color_jitter_params: dict s parametry pro ColorJitter (brightness, contrast, saturation, hue)
        mean, std: hodnoty pro normalizaci
        """
        self.p_flip = p_flip
        self.color_jitter = ColorJitter(**color_jitter_params) if color_jitter_params else None
        self.mean = mean
        self.std = std
        self.p_color = p_color

    def __call__(self, image, mask):  # ,context_image
        # 1) Náhodný horizontální flip
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            # context_image = TF.hflip(context_image)

        # 2) Náhodný vertikální flip
        if random.random() < self.p_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            # context_image = TF.vflip(context_image)

        # 3) Barevné úpravy pouze na obraz
        if self.color_jitter and random.random() < self.p_color:
            image = self.color_jitter(image)
            # context_image = self.color_jitter(context_image)

        # 4) Převod na tensor
        image = TF.to_tensor(image)       # shape [C, H, W]
        mask = TF.to_tensor(mask)         # shape [1, H, W] (u binární masky)
        # context_image = TF.to_tensor(context_image)

        # 5) Normalizace obrazu
        image = TF.normalize(image, mean=self.mean, std=self.std)
        # context_image = TF.normalize(context_image, mean=self.mean, std=self.std)

        return image, mask #context_image
# mean=(0.702, 0.546, 0.696),
# std=(0.239, 0.282, 0.216),

# mean=(0.485, 0.456, 0.406),
# std=(0.229, 0.224, 0.225),
class AlbumentationsAug:
    def __init__(self,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                 p_flip=0,
                 p_rotate90=0.0,
                 p_shiftscalerotate=0.0,
                 p_elastic=0.0,
                 p_color=0.0,
                 p_hestain=0.0,
                 p_noise=0.0,
                 p_blur=0.0):
        """
        Inicializuje Albumentations pipeline pro trénink.
        Args:
            mean (tuple): Průměry pro normalizaci.
            std (tuple): Směrodatné odchylky pro normalizaci.
            p_*: Pravděpodobnosti pro jednotlivé augmentace.
        """
        self.mean = mean
        self.std = std

        # Definice transformací
        self.transform = A.Compose([
            # Geometrické
            A.HorizontalFlip(p=p_flip),
            A.VerticalFlip(p=p_flip),
            A.RandomRotate90(p=p_rotate90),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0625, 0.0625),
                                    rotate=(-15, 15), p=p_shiftscalerotate),
            A.ElasticTransform(alpha=300, sigma=10, p=p_elastic,
                               border_mode=cv2.BORDER_CONSTANT),

            # Barevné/Intenzitní (pouze na obrázek)
            A.RandomBrightnessContrast(p=p_color, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1)),
            A.HueSaturationValue(p=p_color, hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10)),
            A.RGBShift(p=p_color, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),

            # H&E zbarveni
            A.HEStain(p=p_hestain, method="random_preset", intensity_scale_range=(0.9, 1.1), intensity_shift_range=(-0.1, 0.1)),

            # Šum/Blur (pouze na obrázek)
            A.GaussNoise(std_range=[0.03, 0.07], p=p_noise),
            A.GaussianBlur(blur_limit=5, sigma_limit=[0.1, 0.9], p=p_blur),

            # Normalizace a převod na Tensor (aplikuje se na obrázek, ToTensorV2 i na masku)
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(), # Důležité pro správný výstup pro PyTorch

        ])

    def __call__(self, image, mask):
        """
        Aplikuje augmentace.
        Args:
            image (PIL.Image): Vstupní obrázek.
            mask (PIL.Image): Vstupní maska (očekává se 'L' mód, 0-255).
        Returns:
            tuple: (augmented_image_tensor, augmented_mask_tensor)
        """
        # Převod PIL na NumPy (Albumentations pracuje s NumPy HWC)
        image_np = np.array(image)
        mask_np = np.array(mask) # Očekáváme HxW

        # Binarizace masky na 0/1, pokud ještě není
        if mask_np.max() > 1:
            mask_np = (mask_np > 128).astype(np.uint8) # Převedeme na 0/1 uint8

        # Aplikace transformací
        augmented = self.transform(image=image_np, mask=mask_np)

        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float() # Přidat dimenzi kanálu (C=1) a převést na float

        return image_tensor, mask_tensor

