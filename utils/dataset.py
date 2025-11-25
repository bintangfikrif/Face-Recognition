import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): List or array of images (H, W, C) or paths.
            labels (np.ndarray): Array of labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure image is uint8 before passing to albumentations
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if self.transform:
            # Albumentations expects 'image' key
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # If no transform, convert to tensor manually (HWC -> CHW)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)
