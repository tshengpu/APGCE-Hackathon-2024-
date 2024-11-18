from typing import List
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from config import DatasetConfig, DataProcessingConfig


# Custom thresholding transform
class ThresholdTransform:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, tensor):
        return torch.where(tensor < self.threshold, 0, tensor)

        
class DataLoader(Dataset): 
    def __init__(self, seismic_files: List, fault_files: List, horizon_files: List, img_size = (448, 448)):
        self.RAW_SEISMIC_FOLDER = DatasetConfig.RAW_SEISMIC_FOLDER
        self.RAW_FAULT_FOLDER = DatasetConfig.RAW_FAULT_FOLDER
        self.RAW_HORIZON_FOLDER = DatasetConfig.RAW_HORIZON_FOLDER
        self.seismic_files = seismic_files
        self.fault_files = fault_files
        self.horizon_files = horizon_files
        self.IMAGE_SIZE = img_size
        self.transfermers = self._transformation()
        # NOT BEING USED
        with open(os.path.join(DatasetConfig.DATA_FOLDER, 'class_names.txt'), 'r') as f:
            labels = f.readlines()
        self.label = {}
        for i, label in enumerate(labels):
            self.label[i] = label.strip()
        print(self.label)

    def _transformation(self):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),               # Resize image to 1024x1024
            transforms.ToTensor(),                         # Convert to tensor
            ThresholdTransform(                            # Apply thresholding
                threshold=DataProcessingConfig.PIXEL_CUTOFF_THRESHOLD
            ),
            # transforms.ToPILImage()                       # Convert back to PIL image
        ])
        return transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.RAW_SEISMIC_FOLDER, self.seismic_files[idx]))
        img = self.transforms(img)
        # fault_name = '.'.join(self.img_name[idx].split('.', maxsplit = 1)[:-1]) + '.npy'
        fault = np.load(os.path.join(self.RAW_FAULT_FOLDER, self.fault_files[idx])).astype(np.uint8)
        fault = cv2.resize(mask, self.img_size, cv2.INTER_AREA)
        horizon = np.load(os.path.join(self.RAW_HORIZON_FOLDER, self.horizon_files[idx])).astype(np.uint8)
        horizon = cv2.resize(mask, self.img_size, cv2.INTER_AREA)
        return img, torch.from_numpy(fault.astype(np.int64)), torch.from_numpy(horizon.astype(np.int64))

    def __len__(self):
        return len(self.seismic_files)