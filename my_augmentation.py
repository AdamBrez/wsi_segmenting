import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

class MyAugmentations:
    def __init__(self,
                 p_flip=0.5,
                 color_jitter_params=None,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        """
        p_flip: pravděpodobnost pro horizontální/vertikální flip
        color_jitter_params: dict s parametry pro ColorJitter (brightness, contrast, saturation, hue)
        mean, std: hodnoty pro normalizaci
        """
        self.p_flip = p_flip
        self.color_jitter = ColorJitter(**color_jitter_params) if color_jitter_params else None
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        # 1) Náhodný horizontální flip
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 2) Náhodný vertikální flip
        if random.random() < self.p_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 3) Barevné úpravy pouze na obraz
        if self.color_jitter:
            image = self.color_jitter(image)

        # 4) Převod na tensor
        image = TF.to_tensor(image)       # shape [C, H, W]
        mask = TF.to_tensor(mask)         # shape [1, H, W] (u binární masky)

        # 5) Normalizace obrazu
        image = TF.normalize(image, mean=self.mean, std=self.std)

        return image, mask
