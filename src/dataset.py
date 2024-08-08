import os
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ImagesAndMasksDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        binary_mask: bool = False,
        desired_h: int = 256,
        desired_w: int = 256,
        normalize_images: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.binary_mask = binary_mask
        self.desired_h = desired_h
        self.desired_w = desired_w
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.normalize_images = normalize_images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = image.shape
        if (h != self.desired_h) or (w != self.desired_w):
            image = cv2.resize(
                image,
                dsize=(self.desired_w, self.desired_h),
                interpolation=cv2.INTER_CUBIC,
            )
            mask = cv2.resize(
                mask,
                dsize=(self.desired_w, self.desired_h),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.binary_mask:
            mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

        image = np.expand_dims(image, axis=0)  # C x H x W (C = 1)
        if self.normalize_images:
            image = image / 255

        image = image.astype(np.float32)
        mask = mask.astype(np.int64)
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor
