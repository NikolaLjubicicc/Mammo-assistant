import torch
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from config import IMAGE_SIZE

class MammoDataset(Dataset):
    
    def __init__(self, df, augment=True):
        self.df = df.reset_index(drop=True)
        self.augment_flag = augment
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=10, p=0.5),
        ]) if augment else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {row.img_path}")
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.repeat(img[..., None], 3, axis=2)
        
        if self.augment is not None:
            img = self.augment(image=img)["image"]
        
        img = np.moveaxis(img, -1, 0) / 255.0
        
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(row.label, dtype=torch.float32)
        )